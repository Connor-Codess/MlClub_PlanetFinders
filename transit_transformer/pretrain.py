"""Masked-patch pretraining on joint Kepler + TESS windows.

    caffeinate -i python3 -u pretrain.py --run runs/base            # start or resume
    python3 pretrain.py --run runs/smoke --steps 500 --val-every 250 --probe-every 0
    python3 pretrain.py --run runs/overfit --overfit 64 --steps 400 --probe-every 0

Run directory:
    config.json      args + model config
    last.pt          latest checkpoint (resume target); ckpt-XXXXXX.pt every --keep-every
    train.csv        step, loss, per-channel loss, lr, windows/s
    val.csv          step, group, loss, per-channel loss (+ baselines at step -1)
    probe.csv        linear-probe metrics on DR24 labels
    plots/           reconstructions of fixed val windows
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import time

import numpy as np
import torch

import preprocess as P
from dataset import MixedPretrainDataset, collate, _iter_shard, shard_paths
from model import ModelConfig, MaskedPatchModel, make_patch_mask, count_params, TARGET_CHANNELS

CH_NAMES = ["flux", "detrended", "centroid"]


# ---------------------------------------------------------------- utilities

def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device, amp):
    import contextlib
    if amp == "none" or device.type == "cpu":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dict(bf16=torch.bfloat16, fp16=torch.float16)[amp])


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))


class CSVLog:
    def __init__(self, path, fields):
        new = not os.path.exists(path)
        self.fh = open(path, "a", newline="")
        self.w = csv.DictWriter(self.fh, fields, extrasaction="ignore")
        if new:
            self.w.writeheader()

    def write(self, row):
        self.w.writerow(row)
        self.fh.flush()


def to_dev(b, device):
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


# ---------------------------------------------------------------- fixed validation set

def build_val_set(path, root, n_per_group, patch, mask_ratio, seed=1234):
    """Fixed windows + fixed hidden masks for three groups: tess, kepler, kepler_aug."""
    if os.path.exists(path):
        return dict(np.load(path))
    rng = np.random.default_rng(seed)
    tess_sig = np.concatenate([np.load(p)["sigma"] for p in shard_paths(root, "tess", "val")[:20]])
    out = {k: [] for k in ("values", "mask", "side", "group", "hidden")}
    for g, (mission, aug) in enumerate([("tess", False), ("kepler", False), ("kepler", True)]):
        paths = shard_paths(root, mission, "val")
        per_shard = -(-n_per_group // len(paths)) + 1     # spread the sample over all shards
        got = 0
        for p in rng.permutation(paths):
            for k, (b, sigma) in enumerate(_iter_shard(p, rng)):
                extra = 0.0
                if aug:
                    target = float(rng.choice(tess_sig))
                    extra = float(np.sqrt(max(target ** 2 - sigma ** 2, 0.0)))
                w = P.scale_window(b, extra, rng)
                out["values"].append(w.values); out["mask"].append(w.mask)
                out["side"].append(w.side); out["group"].append(g)
                got += 1
                if k + 1 >= per_shard or got >= n_per_group:
                    break
            if got >= n_per_group:
                break
    d = {k: np.stack(v) for k, v in out.items() if v}
    d["group"] = np.array(out["group"])
    d["hidden"] = make_patch_mask(d["mask"], patch, mask_ratio, rng=rng)
    np.savez(path, **d)
    return d


GROUPS = ["tess", "kepler", "kepler_aug"]


def baseline_losses(val, patch, cw):
    """MSE of two trivial predictors on the val targets: zero and linear interpolation."""
    L = val["mask"].shape[1]
    res = {}
    for name in ("zero", "interp"):
        per = np.zeros((len(GROUPS), 3)); cnt = np.zeros((len(GROUPS), 3))
        for i in range(len(val["mask"])):
            m = val["mask"][i]
            hb = np.repeat(val["hidden"][i], patch)
            tgt_bins = m & hb
            vis = m & ~hb
            g = val["group"][i]
            for j, c in enumerate(TARGET_CHANNELS):
                if c == 3 and val["side"][i, 2] == 0:
                    continue
                y = val["values"][i, :, c]
                if name == "zero":
                    pred = np.zeros(tgt_bins.sum())
                else:
                    xs = np.flatnonzero(vis)
                    pred = np.interp(np.flatnonzero(tgt_bins), xs, y[xs]) if len(xs) else 0.0
                per[g, j] += ((pred - y[tgt_bins]) ** 2).sum()
                cnt[g, j] += tgt_bins.sum()
        ch = per / np.maximum(cnt, 1)
        for g, gn in enumerate(GROUPS):
            res[(name, gn)] = (float((ch[g] * cw).sum() / cw.sum()), ch[g])
    return res


@torch.no_grad()
def evaluate(model, val, device, batch=256, amp="none"):
    model.eval()
    per = torch.zeros(len(GROUPS), 3); n = torch.zeros(len(GROUPS))
    for s in range(0, len(val["mask"]), batch):
        sl = slice(s, s + batch)
        v = torch.from_numpy(val["values"][sl]).to(device)
        m = torch.from_numpy(val["mask"][sl]).to(device)
        sd = torch.from_numpy(val["side"][sl]).to(device)
        h = torch.from_numpy(val["hidden"][sl]).to(device)
        g = val["group"][sl]
        for gi in range(len(GROUPS)):
            sel = torch.from_numpy(np.flatnonzero(g == gi)).to(device)
            if len(sel) == 0:
                continue
            with autocast(device, amp):
                _, parts = model(v[sel], m[sel], sd[sel], h[sel])
            per[gi] += parts["per_channel"].cpu() * len(sel)
            n[gi] += len(sel)
    model.train()
    ch = per / n.clamp(min=1)[:, None]
    cw = model.cw.cpu()
    return {gn: (float((ch[g] * cw).sum() / cw.sum()), ch[g].numpy()) for g, gn in enumerate(GROUPS)}


@torch.no_grad()
def plot_recon(model, val, device, path, patch, n_per=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    idx = np.concatenate([np.flatnonzero(val["group"] == g)[:n_per] for g in (0, 1)])
    v = torch.from_numpy(val["values"][idx]).to(device)
    m = torch.from_numpy(val["mask"][idx]).to(device)
    sd = torch.from_numpy(val["side"][idx]).to(device)
    h = torch.from_numpy(val["hidden"][idx]).to(device)
    _, parts = model(v, m, sd, h)
    pred = parts["pred"].reshape(len(idx), -1, len(TARGET_CHANNELS)).cpu().numpy()
    model.train()
    fig, axes = plt.subplots(len(idx), 1, figsize=(14, 2.0 * len(idx)), sharex=True)
    t = np.arange(v.shape[1]) * P.BIN_DAYS
    for a, i, k in zip(axes, idx, range(len(idx))):
        mm = val["mask"][i]; hb = np.repeat(val["hidden"][i], patch)
        y = val["values"][i, :, 1]
        a.plot(t[mm & ~hb], y[mm & ~hb], ".", ms=1.5, color="0.6", label="visible")
        a.plot(t[mm & hb], y[mm & hb], ".", ms=1.5, color="C0", label="hidden (true)")
        a.plot(t[mm & hb], pred[k, :, 1][mm & hb], ".", ms=1.5, color="C3", label="predicted")
        a.set_ylabel(["TESS", "Kepler"][val["group"][i]], fontsize=8)
    axes[0].legend(loc="upper right", fontsize=7, markerscale=5)
    axes[-1].set_xlabel("days (detrended flux channel)")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


# ---------------------------------------------------------------- data

def make_loader(args, seed):
    ds = MixedPretrainDataset(args.data, "train", kepler_frac=args.kepler_frac,
                              augment_frac=args.augment_frac, zero_centroids=args.zero_centroids,
                              seed=seed)
    return torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                                       collate_fn=collate, persistent_workers=args.workers > 0,
                                       prefetch_factor=4 if args.workers > 0 else None)


def overfit_batches(args, seed):
    """Endless repeats of one fixed small batch (sanity check)."""
    ds = MixedPretrainDataset(args.data, "train", augment_frac=0.0, seed=seed)
    it = iter(ds)
    b = collate([next(it) for _ in range(args.overfit)])
    while True:
        yield b


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="data/shards")
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--span-frac", type=float, default=0.5)
    ap.add_argument("--kepler-frac", type=float, default=0.5)
    ap.add_argument("--augment-frac", type=float, default=0.5)
    ap.add_argument("--zero-centroids", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"], help="autocast dtype on GPU/MPS")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-n", type=int, default=1024, help="windows per val group")
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--keep-every", type=int, default=10_000)
    ap.add_argument("--probe-every", type=int, default=10_000, help="0 disables the DR24 probe")
    ap.add_argument("--overfit", type=int, default=0, help="train on one fixed batch of N windows")
    for k, v in ModelConfig().to_dict().items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.run, "plots"), exist_ok=True)
    device = pick_device(args.device)
    cfg = ModelConfig(**{k: getattr(args, k) for k in ModelConfig().to_dict()})
    torch.manual_seed(args.seed)
    model = MaskedPatchModel(cfg).to(device)

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "pos" not in n and "token" not in n and "cls" not in n]
    no_decay = [p for n, p in model.named_parameters() if not any(p is q for q in decay)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))

    step = 0
    last = os.path.join(args.run, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        print(f"resumed from step {step}", flush=True)
    else:
        with open(os.path.join(args.run, "config.json"), "w") as fh:
            json.dump(dict(args=vars(args), model=cfg.to_dict()), fh, indent=2)
    print(f"device {device} ({args.amp}), {count_params(model) / 1e6:.2f}M params, run {args.run}", flush=True)

    tlog = CSVLog(os.path.join(args.run, "train.csv"),
                  ["step", "loss", *CH_NAMES, "lr", "windows_per_s", "elapsed_s"])
    vlog = CSVLog(os.path.join(args.run, "val.csv"), ["step", "group", "loss", *CH_NAMES])

    val = build_val_set(os.path.join(args.run, "val_set.npz"), args.data, args.val_n,
                        cfg.patch, args.mask_ratio)
    if step == 0:
        for (name, gn), (loss, ch) in baseline_losses(val, cfg.patch, model.cw.cpu().numpy()).items():
            vlog.write(dict(step=f"baseline_{name}", group=gn, loss=f"{loss:.5f}",
                            **{c: f"{x:.5f}" for c, x in zip(CH_NAMES, ch)}))
            print(f"baseline {name:6s} {gn:10s} {loss:.4f}", flush=True)

    probe = None
    if args.probe_every and not args.overfit:
        import probe as probe_mod
        probe = probe_mod.Probe(args.data, os.path.join(args.data, "..", "probe"), device)
        plog = CSVLog(os.path.join(args.run, "probe.csv"), probe_mod.FIELDS)

    def save(path):
        tmp = path + ".tmp"
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step,
                        model_cfg=cfg.to_dict(), args=vars(args)), tmp)
        os.replace(tmp, path)

    def validate():
        res = evaluate(model, val, device, amp=args.amp)
        for gn, (loss, ch) in res.items():
            vlog.write(dict(step=step, group=gn, loss=f"{loss:.5f}", **{c: f"{x:.5f}" for c, x in zip(CH_NAMES, ch)}))
        print(f"[val {step}] " + "  ".join(f"{g} {l:.4f}" for g, (l, _) in res.items()), flush=True)
        plot_recon(model, val, device, os.path.join(args.run, "plots", f"recon-{step:06d}.png"), cfg.patch)

    def run_probe():
        r = probe.run(model.encoder)
        plog.write(dict(step=step, **r))
        print(f"[probe {step}] " + "  ".join(f"{k} {v:.3f}" for k, v in r.items() if isinstance(v, float)), flush=True)

    if step == 0 and not args.overfit:
        validate()
        if probe:
            run_probe()

    batches = overfit_batches(args, args.seed) if args.overfit else iter(make_loader(args, args.seed + step))
    rng = np.random.default_rng([args.seed, step])
    model.train()
    t0 = time.time(); tw = time.time(); seen = 0; start = time.time()
    acc = torch.zeros(4)
    while step < args.steps:
        b = next(batches)
        hidden = torch.from_numpy(make_patch_mask(b["mask"].numpy(), cfg.patch, args.mask_ratio,
                                                  args.span_frac, rng=rng))
        b = to_dev(b, device); hidden = hidden.to(device)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        with autocast(device, args.amp):
            loss, parts = model(b["values"], b["mask"], b["side"], hidden)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        step += 1; seen += b["mask"].shape[0]
        acc += torch.cat([loss.detach().cpu()[None], parts["per_channel"].cpu()])
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"non-finite loss at step {step}")

        if step % args.log_every == 0:
            a = acc / args.log_every; acc.zero_()
            wps = seen / (time.time() - tw); seen = 0; tw = time.time()
            tlog.write(dict(step=step, loss=f"{a[0]:.5f}", **{c: f"{x:.5f}" for c, x in zip(CH_NAMES, a[1:])},
                            lr=f"{lr_at(step, args):.2e}", windows_per_s=f"{wps:.0f}",
                            elapsed_s=f"{time.time() - start:.0f}"))
            print(f"step {step} loss {a[0]:.4f} ({' '.join(f'{x:.3f}' for x in a[1:])}) "
                  f"lr {lr_at(step, args):.1e} {wps:.0f} win/s", flush=True)
        if not args.overfit:
            if step % args.val_every == 0 or step == args.steps:
                validate()
            if step % args.ckpt_every == 0 or step == args.steps:
                save(last)
                if step % args.keep_every == 0 or step == args.steps:
                    save(os.path.join(args.run, f"ckpt-{step:06d}.pt"))
            if probe and (step % args.probe_every == 0 or step == args.steps):
                run_probe()
    print(f"done: {step} steps in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
