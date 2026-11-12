"""Fine-tune the pretrained encoder to classify transit signals (planet / candidate / FP / no planet).

    caffeinate -i python3 -u finetune.py --run runs/ft_main                      # pretrained encoder
    python3 finetune.py --run runs/ft_random --encoder random
    python3 finetune.py --run runs/ft_none   --encoder none
    python3 finetune.py --run runs/ft_frozen --encoder frozen
    python3 finetune.py --run runs/ft_overfit --overfit 64 --epochs1 0 --epochs2 30

Stage 1: encoder frozen, head/CNN/scalar layers trained (lr --head-lr1).
Stage 2: encoder unfrozen with layer-wise lr decay (top layer --enc-lr), head at --head-lr2.
Model selection: best TESS val macro-F1 (Kepler val logged too). Test sets are never read here.

Run directory: config.json, last.pt (resume), best.pt, log.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import preprocess as P
from build_examples import OUT as EX_DIR, W, NS
from finetune_model import SignalClassifier, param_groups, N_CLASSES
from model import ModelConfig
from pretrain import pick_device, autocast, CSVLog

CLASS_NAMES = ["planet", "candidate", "false_positive", "no_planet"]
CANDIDATE_WEIGHT = 0.5


class SignalSet(Dataset):
    """Memory-mapped examples of one split. Scaling (and Kepler noise augmentation) per item."""

    def __init__(self, split, train=False, tess_sigmas=None, missions=("kepler", "tess"),
                 noise_frac=0.5, centroid_drop=0.1, limit=0, seed=0):
        self.dir = os.path.join(EX_DIR, split)
        self.meta = pd.read_parquet(os.path.join(self.dir, "meta.parquet"))
        keep = np.flatnonzero(self.meta.mission.isin(missions).to_numpy())
        if limit:
            keep = np.random.default_rng(seed).choice(keep, min(limit, len(keep)), replace=False)
        self.rows = keep
        self.train, self.tess_sigmas = train, tess_sigmas
        self.noise_frac, self.centroid_drop = noise_frac, centroid_drop
        self.a = None
        self.seed = seed

    def _open(self):
        if self.a is None:
            names = ["flux_rel", "err_rel", "mask", "cx", "cy", "has_c", "sigma", "tpatch", "n_win",
                     "star", "glob", "loc", "scal", "cls"]
            self.a = {k: np.load(os.path.join(self.dir, f"{k}.npy"), mmap_mode="r") for k in names}
            wi = torch.utils.data.get_worker_info()
            self.rng = np.random.default_rng([self.seed, wi.id if wi else 0, os.getpid()])

    def __len__(self):
        return len(self.rows)

    def labels(self):
        return self.meta.cls.to_numpy()[self.rows], self.meta.mission.to_numpy()[self.rows]

    def __getitem__(self, k):
        self._open()
        a, i, rng = self.a, int(self.rows[k]), self.rng
        mission = self.meta.mission.iat[i]
        n = int(a["n_win"][i])
        order = np.arange(n)
        if self.train and n > 1:
            order = rng.permutation(n)[:int(rng.integers(max(1, n // 2), n + 1))]
        mag, teff, rad = (None if not np.isfinite(v) else float(v) for v in a["star"][i])
        values = np.zeros((W, P.WINDOW_BINS, 4), np.float32)
        mask = np.zeros((W, P.WINDOW_BINS), bool)
        side = np.zeros((W, P.N_SIDE), np.float32)
        tpatch = np.zeros((W, a["tpatch"].shape[-1]), bool)
        wmask = np.zeros(W, bool)
        noisy = (self.train and mission == "kepler" and self.tess_sigmas is not None
                 and rng.random() < self.noise_frac)
        target = float(rng.choice(self.tess_sigmas)) if noisy else 0.0
        for j, wj in enumerate(order):
            has_c = bool(a["has_c"][i, wj])
            b = P.Binned(np.asarray(a["flux_rel"][i, wj]), np.asarray(a["err_rel"][i, wj], np.float32),
                         np.asarray(a["mask"][i, wj]),
                         np.asarray(a["cx"][i, wj]) if has_c else None,
                         np.asarray(a["cy"][i, wj]) if has_c else None, mag, teff, rad, {})
            extra = float(np.sqrt(max(target ** 2 - float(a["sigma"][i, wj]) ** 2, 0.0))) if noisy else 0.0
            w = P.scale_window(b, extra, rng)
            values[j], mask[j], side[j] = w.values, w.mask, w.side
            tpatch[j] = a["tpatch"][i, wj]; wmask[j] = True
        glob = np.array(a["glob"][i]); loc = np.array(a["loc"][i]); scal = np.array(a["scal"][i])
        if self.train and rng.random() < self.centroid_drop:
            values[..., 3] = 0; side[:, 1:3] = 0; glob[1] = 0; loc[4] = 0; scal[14] = 0
        cls = int(a["cls"][i])
        return dict(values=values, mask=mask, side=side, tpatch=tpatch, wmask=wmask, glob=glob, loc=loc,
                    scal=scal, label=cls, weight=np.float32(CANDIDATE_WEIGHT if cls == 1 else 1.0),
                    mission=int(mission == "tess"), idx=i)


def collate(items):
    return {k: torch.as_tensor(np.stack([it[k] for it in items])) for k in items[0]}


def balanced_sampler(ds, kepler_frac=0.5, seed=0):
    """50/50 missions, class-balanced inside each mission."""
    y, m = ds.labels()
    w = np.zeros(len(y))
    for mis, share in (("kepler", kepler_frac), ("tess", 1 - kepler_frac)):
        sel = m == mis
        present = np.unique(y[sel])
        for c in present:
            cs = sel & (y == c)
            w[cs] = share / len(present) / cs.sum()
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w), num_samples=len(y), replacement=True, generator=g)


def tess_sigma_pool(n_shards=20, seed=0):
    import glob as G
    paths = sorted(G.glob("data/shards/tess/train/*.npz"))
    pick = np.random.default_rng(seed).choice(len(paths), min(n_shards, len(paths)), replace=False)
    return np.concatenate([np.load(paths[i])["sigma"] for i in pick])


def to_dev(b, device):
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


def metrics(y, prob):
    from sklearn.metrics import f1_score, balanced_accuracy_score
    pred = prob.argmax(1)
    labels = [c for c in range(N_CLASSES) if (y == c).any()]
    return dict(macro_f1=float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
                bal_acc=float(balanced_accuracy_score(y, pred)))


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    P_, Y, M, I = [], [], [], []
    for b in loader:
        b = to_dev(b, device)
        with autocast(device, amp):
            logits = model(b)
        P_.append(torch.softmax(logits.float(), -1).cpu().numpy())
        Y.append(b["label"].cpu().numpy()); M.append(b["mission"].cpu().numpy()); I.append(b["idx"].cpu().numpy())
    model.train()
    return np.concatenate(P_), np.concatenate(Y), np.concatenate(M), np.concatenate(I)


def build_model(args, device):
    ck = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_cfg"])
    model = SignalClassifier(cfg, NS, args.encoder, args.dropout)
    if args.encoder in ("pretrained", "frozen"):
        enc_state = {k[len("encoder."):]: v for k, v in ck["model"].items() if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc_state)
    return model.to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--pretrained", default="runs/base/ckpt-040000.pt")
    ap.add_argument("--encoder", default="pretrained", choices=["pretrained", "random", "frozen", "none"])
    ap.add_argument("--epochs1", type=int, default=5)
    ap.add_argument("--epochs2", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--head-lr1", type=float, default=1e-3)
    ap.add_argument("--head-lr2", type=float, default=3e-4)
    ap.add_argument("--enc-lr", type=float, default=1e-4)
    ap.add_argument("--layer-decay", type=float, default=0.75)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--smoothing", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=4, help="stage-2 epochs without val improvement")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overfit", type=int, default=0, help="train and validate on N fixed train examples")
    ap.add_argument("--max-steps", type=int, default=0, help="stop each epoch after N steps (speed tests)")
    args = ap.parse_args()

    os.makedirs(args.run, exist_ok=True)
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    model = build_model(args, device)
    sig = tess_sigma_pool()
    if args.overfit:
        tr = SignalSet("train", train=False, limit=args.overfit, seed=args.seed)
        va_t = va_k = tr
    else:
        tr = SignalSet("train", train=True, tess_sigmas=sig, seed=args.seed)
        va_t = SignalSet("val", missions=("tess",))
        va_k = SignalSet("val", missions=("kepler",))
    kw = dict(batch_size=args.batch, num_workers=args.workers, collate_fn=collate,
              persistent_workers=args.workers > 0)
    tl = DataLoader(tr, sampler=None if args.overfit else balanced_sampler(tr, seed=args.seed),
                    shuffle=bool(args.overfit), drop_last=not args.overfit, **kw)
    vl_t = DataLoader(va_t, shuffle=False, **kw)
    vl_k = DataLoader(va_k, shuffle=False, **kw)
    steps_per_epoch = args.max_steps or len(tl)

    log = CSVLog(os.path.join(args.run, "log.csv"),
                 ["epoch", "stage", "train_loss", "tess_val_macro_f1", "tess_val_bal_acc",
                  "kepler_val_macro_f1", "kepler_val_bal_acc", "minutes"])
    start_epoch, best = 0, -1.0
    total = args.epochs1 + args.epochs2
    last = os.path.join(args.run, "last.pt")
    state = None
    if os.path.exists(last):
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); start_epoch = state["epoch"] + 1; best = state["best"]
        print(f"resumed after epoch {state['epoch']}", flush=True)
    else:
        with open(os.path.join(args.run, "config.json"), "w") as fh:
            json.dump(vars(args), fh, indent=2)
    n_tr = sum(p.numel() for p in model.parameters())
    print(f"device {device} ({args.amp}); encoder {args.encoder}; {n_tr / 1e6:.2f}M params; "
          f"train {len(tr)}, val tess {len(va_t)} kepler {len(va_k)}; {steps_per_epoch} steps/epoch", flush=True)

    def make_opt(stage):
        model.encoder_trainable(stage == 2)
        if stage == 1:
            groups = param_groups(model, args.head_lr1, 0.0, 1.0, args.wd)
        else:
            groups = param_groups(model, args.head_lr2, args.enc_lr, args.layer_decay, args.wd)
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))

    opt, cur_stage, bad = None, None, 0
    t0 = time.time()
    for epoch in range(start_epoch, total):
        stage = 1 if epoch < args.epochs1 else 2
        if stage != cur_stage:
            opt = make_opt(stage); cur_stage = stage
            if state is not None and state.get("stage") == stage:
                opt.load_state_dict(state["opt"])
            state = None
        e_in = epoch if stage == 1 else epoch - args.epochs1
        n_e = args.epochs1 if stage == 1 else args.epochs2
        model.train()
        losses = []
        for step, b in enumerate(tl):
            if step >= steps_per_epoch:
                break
            frac = (e_in + step / steps_per_epoch) / max(n_e, 1)
            warm = min(1.0, (e_in * steps_per_epoch + step + 1) / 200)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * warm * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))
            b = to_dev(b, device)
            with autocast(device, args.amp):
                logits = model(b)
            loss = (F.cross_entropy(logits.float(), b["label"], label_smoothing=args.smoothing, reduction="none")
                    * b["weight"]).sum() / b["weight"].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {np.mean(losses[-50:]):.4f} "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        pt, yt, _, _ = predict(model, vl_t, device, args.amp)
        mt = metrics(yt, pt)
        if args.overfit:
            mk = mt
        else:
            pk, yk, _, _ = predict(model, vl_k, device, args.amp)
            mk = metrics(yk, pk)
        row = dict(epoch=epoch, stage=stage, train_loss=f"{np.mean(losses):.4f}",
                   tess_val_macro_f1=f"{mt['macro_f1']:.4f}", tess_val_bal_acc=f"{mt['bal_acc']:.4f}",
                   kepler_val_macro_f1=f"{mk['macro_f1']:.4f}", kepler_val_bal_acc=f"{mk['bal_acc']:.4f}",
                   minutes=f"{(time.time() - t0) / 60:.1f}")
        log.write(row)
        print(f"epoch {epoch} (stage {stage}) loss {np.mean(losses):.4f} | TESS val macro-F1 {mt['macro_f1']:.3f} "
              f"bal-acc {mt['bal_acc']:.3f} | Kepler val macro-F1 {mk['macro_f1']:.3f}", flush=True)
        improved = mt["macro_f1"] > best
        if improved:
            best = mt["macro_f1"]; bad = 0
            torch.save(dict(model=model.state_dict(), epoch=epoch, args=vars(args), tess_val=mt, kepler_val=mk),
                       os.path.join(args.run, "best.pt"))
        elif stage == 2:
            bad += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=epoch, stage=stage, best=best),
                   last + ".tmp")
        os.replace(last + ".tmp", last)
        if stage == 2 and bad >= args.patience:
            print(f"early stop: no TESS val improvement for {bad} epochs", flush=True)
            break
    print(f"done; best TESS val macro-F1 {best:.3f}; {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
