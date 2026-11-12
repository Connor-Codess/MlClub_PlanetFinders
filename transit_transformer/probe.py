"""Linear probe on DR24 TCE labels (PC / AFP / NTP) for Kepler windows.

A window is used if it covers at least one transit of a TCE on that star
(a filled bin within the transit duration). If several TCEs qualify, the one
with the highest model SNR is used. At most 3 windows per TCE.

    fit:   Kepler *train* stars (balanced, up to N_PER_CLASS per class)
    score: Kepler *val* stars (Kepler test stars are never touched)

Features: frozen encoder -> [CLS, mean over patches, token of the transit patch].
Classifier: standardise + multinomial logistic regression.
A hand-made baseline (side features + in/out-of-transit means) is scored once,
which gives the number the encoder has to beat.

    python3 probe.py --ckpt runs/base/last.pt        # probe a checkpoint
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

import preprocess as P

CLASSES = ["PC", "AFP", "NTP"]
N_PER_CLASS = 6000
MAX_PER_TCE = 3
TCE_URL = "hf://datasets/nyu-planet-finders/kepler-raw-lightcurves/tces/dr24_tce.parquet"
FIELDS = ["step", "n_fit", "n_score", "bal_acc", "macro_auc", "auc_PC", "auc_AFP", "auc_NTP",
          "base_bal_acc", "base_macro_auc"]


def load_tces(cache_dir):
    path = os.path.join(cache_dir, "dr24_tce.parquet")
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        pd.read_parquet(TCE_URL).to_parquet(path)
    df = pd.read_parquet(path)
    return df[df.av_training_set.isin(CLASSES)].copy()


def find_transits(z, tces_by_star):
    """For each window in shard arrays z, return (label, centre bin) or None."""
    out = []
    L = P.WINDOW_BINS
    for i in range(len(z["star_id"])):
        rows = tces_by_star.get(int(z["star_id"][i]))
        if rows is None:
            out.append(None); continue
        start = z["t0_bjd"][i] - P.BJD_OFFSET["kepler"]          # BKJD of bin 0 centre
        m = z["mask"][i]
        best = None
        for per, ep, dur, snr, lab, tid in rows:
            n0 = np.ceil((start - ep) / per); n1 = np.floor((start + L * P.BIN_DAYS - ep) / per)
            if n1 < n0:
                continue
            half = max(1, int(round(dur / 24 / P.BIN_DAYS / 2)))
            for n in np.arange(n0, n1 + 1):
                k = int(round((ep + n * per - start) / P.BIN_DAYS))
                if 0 <= k < L and m[max(0, k - half):k + half + 1].any():
                    if best is None or snr > best[2]:
                        best = (lab, k, snr, half, tid)
                    break
        out.append(best)
    return out


def _binned(z, i):
    def opt(v):
        return None if np.isnan(v) else float(v)
    has_c = bool(z["has_c"][i])
    return P.Binned(z["flux_rel"][i], z["err_rel"][i].astype(np.float32), z["mask"][i],
                    z["cx"][i] if has_c else None, z["cy"][i] if has_c else None,
                    opt(z["magnitude"][i]), opt(z["teff"][i]), opt(z["radius"][i]),
                    dict(star_id=int(z["star_id"][i])))


def build_split(root, split, tces, rng, per_class=None):
    by_star = {}
    for r in tces.itertuples():
        by_star.setdefault(int(r.kepid), []).append(
            (r.tce_period, r.tce_time0bk, r.tce_duration, r.tce_model_snr,
             CLASSES.index(r.av_training_set), int(r.kepid) * 100 + int(r.tce_plnt_num)))
    cand = []                                                  # (path, i, label, k, half, tce_id)
    for p in sorted(glob.glob(os.path.join(root, "kepler", split, "*.npz"))):
        with np.load(p) as zz:
            z = {k: zz[k] for k in ("star_id", "t0_bjd", "mask")}
        for i, r in enumerate(find_transits(z, by_star)):
            if r is not None:
                cand.append((p, i, r[0], r[1], r[3], r[4]))
    rng.shuffle(cand)
    keep, per_tce, per_lab = [], {}, np.zeros(3, int)
    for c in cand:
        key = c[5]
        if per_tce.get(key, 0) >= MAX_PER_TCE:
            continue
        if per_class and per_lab[c[2]] >= per_class:
            continue
        per_tce[key] = per_tce.get(key, 0) + 1; per_lab[c[2]] += 1
        keep.append(c)
    vals, masks, sides, labs, ks, halves = [], [], [], [], [], []
    for p in sorted({c[0] for c in keep}):
        with np.load(p) as zz:
            z = {k: zz[k] for k in zz.files}
        for c in (c for c in keep if c[0] == p):
            w = P.scale_window(_binned(z, c[1]))
            vals.append(w.values.astype(np.float16)); masks.append(w.mask); sides.append(w.side)
            labs.append(c[2]); ks.append(c[3]); halves.append(c[4])
    return dict(values=np.stack(vals), mask=np.stack(masks), side=np.stack(sides),
                label=np.array(labs), k=np.array(ks), half=np.array(halves))


def hand_features(d):
    """Baseline: side features + in-transit / out-of-transit means of each channel."""
    f = []
    L = d["mask"].shape[1]
    for i in range(len(d["label"])):
        k, h, m = d["k"][i], d["half"][i], d["mask"][i]
        inn = np.zeros(L, bool); inn[max(0, k - h):k + h + 1] = True
        near = np.zeros(L, bool); near[max(0, k - 6 * h):k + 6 * h + 1] = True
        a, b = m & inn, m & near & ~inn
        v = d["values"][i].astype(np.float32)
        fi = [v[a, c].mean() if a.any() else 0 for c in range(4)]
        fo = [v[b, c].mean() if b.any() else 0 for c in range(4)]
        f.append(np.concatenate([d["side"][i], fi, fo, [fi[1] - fo[1], fi[3] - fo[3]]]))
    return np.array(f, np.float32)


def fit_score(Xf, yf, Xs, ys):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
    clf.fit(Xf, yf)
    pr = clf.predict_proba(Xs)
    aucs = [roc_auc_score(ys == c, pr[:, c]) for c in range(3)]
    return dict(bal_acc=float(balanced_accuracy_score(ys, pr.argmax(1))),
                macro_auc=float(np.mean(aucs)), **{f"auc_{n}": float(a) for n, a in zip(CLASSES, aucs)})


class Probe:
    def __init__(self, data_root, cache_dir, device, seed=0):
        self.device = device
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, "probe_set.npz")
        if not os.path.exists(path):
            print("building probe set (one time) ...", flush=True)
            tces = load_tces(cache_dir)
            rng = np.random.default_rng(seed)
            fit = build_split(data_root, "train", tces, rng, N_PER_CLASS)
            sc = build_split(data_root, "val", tces, rng)
            np.savez(path, **{f"fit_{k}": v for k, v in fit.items()}, **{f"score_{k}": v for k, v in sc.items()})
        z = np.load(path)
        self.fit = {k[4:]: z[k] for k in z.files if k.startswith("fit_")}
        self.score = {k[6:]: z[k] for k in z.files if k.startswith("score_")}
        b = fit_score(hand_features(self.fit), self.fit["label"], hand_features(self.score), self.score["label"])
        self.base = dict(base_bal_acc=b["bal_acc"], base_macro_auc=b["macro_auc"])
        print(f"probe set: fit {np.bincount(self.fit['label'], minlength=3)}, "
              f"score {np.bincount(self.score['label'], minlength=3)} (PC/AFP/NTP); "
              f"hand-feature baseline bal_acc {b['bal_acc']:.3f} macro_auc {b['macro_auc']:.3f}", flush=True)

    @torch.no_grad()
    def features(self, encoder, d, batch=256):
        was = encoder.training
        encoder.eval()
        out = []
        p = encoder.cfg.patch
        for s in range(0, len(d["label"]), batch):
            v = torch.from_numpy(d["values"][s:s + batch].astype(np.float32)).to(self.device)
            m = torch.from_numpy(d["mask"][s:s + batch]).to(self.device)
            sd = torch.from_numpy(d["side"][s:s + batch]).to(self.device)
            t = encoder(v, m, sd)
            has = m.reshape(m.shape[0], -1, p).any(-1).float()[..., None]
            mean = (t[:, 2:] * has).sum(1) / has.sum(1).clamp(min=1)
            kp = torch.from_numpy(d["k"][s:s + batch] // p + 2).to(self.device)
            tr = t[torch.arange(len(kp), device=self.device), kp]
            out.append(torch.cat([t[:, 0], mean, tr], -1).cpu().numpy())
        encoder.train(was)
        return np.concatenate(out)

    def run(self, encoder):
        r = fit_score(self.features(encoder, self.fit), self.fit["label"],
                      self.features(encoder, self.score), self.score["label"])
        return dict(n_fit=len(self.fit["label"]), n_score=len(self.score["label"]), **r, **self.base)


def main():
    from model import ModelConfig, MaskedPatchModel
    from pretrain import pick_device
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="checkpoint; omit to probe a random-init encoder")
    ap.add_argument("--data", default="data/shards")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        model = MaskedPatchModel(ModelConfig(**ck["model_cfg"])).to(device)
        model.load_state_dict(ck["model"])
    else:
        model = MaskedPatchModel(ModelConfig()).to(device)
    probe = Probe(args.data, os.path.join(args.data, "..", "probe"), device)
    print(probe.run(model.encoder))


if __name__ == "__main__":
    main()
