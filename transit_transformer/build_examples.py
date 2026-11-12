"""Per-signal fine-tuning examples from the window shards.

    python3 build_examples.py               # all splits -> data/examples/<split>/
    python3 build_examples.py --plot TOI-101.01 KIC10797460.01   # check folds

For each signal (labels.py) and all of its star's windows in the same split:

  encoder windows (up to W=8 that contain a transit; most transits first), unscaled:
      flux_rel f32 [N,W,L], err_rel f16, mask, cx/cy f32, has_c [N,W], sigma [N,W],
      tpatch [N,W,P] (patches holding a transit centre), n_win [N]
  folded views (detrended with in-transit bins masked out):
      glob [N,3,201]  flux, centroid, filled-fraction over the full orbit
      loc  [N,6,81]   flux, odd, even, secondary (phase 0.5), centroid, filled over +-3 durations
      Flux views: (x - oot median) / depth, so a transit bottoms near -1.
      Centroid views: (c - oot median) / oot robust sigma.
  scalars [N, NS]  see SCALARS
  meta.parquet     one row per example: signal_id, mission, star_id, cls, raw_label, split
Arrays are .npy so the data loader can memory-map them.
"""

from __future__ import annotations

import argparse
import collections
import os

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

import preprocess as P
from labels import INDEX, SIGNALS, CLASSES

OUT = "data/examples"
W = 8
L = P.WINDOW_BINS
PATCH = 16
NP = L // PATCH
NG, NL, LOC_DUR = 201, 81, 3.0
SCALARS = ["log_period", "log_dur_h", "log_depth_meas", "log_depth_cat", "has_depth_cat",
           "log_snr", "log_n_transits", "mag", "has_mag", "teff", "has_teff", "log_rad",
           "has_rad", "log_sigma", "has_c", "frac_win_with_transit"]
NS = len(SCALARS)
SPLITS = ["train", "val", "test", "test_future"]


# ---------------------------------------------------------------- folding

def _detrend(y, mask, keep):
    """y minus a 1-day running median fitted on mask & keep; trend interpolated elsewhere."""
    fit = mask & keep & np.isfinite(y)
    if fit.sum() < 10:
        return np.where(mask, y - np.nanmedian(y[mask]) if mask.any() else y, np.nan)
    tr = P._running_median(np.where(fit, y, 0), fit)
    x = np.arange(len(y))
    trend = np.interp(x, x[fit], tr[fit])
    return np.where(mask, y - trend, np.nan)


def _bin(x, v, lo, hi, n):
    ok = np.isfinite(x) & np.isfinite(v) & (x >= lo) & (x < hi)
    idx = ((x[ok] - lo) / (hi - lo) * n).astype(int).clip(0, n - 1)
    c = np.bincount(idx, minlength=n).astype(float)
    m = np.full(n, np.nan)                                # median per bin: robust to single bad points
    if len(idx):
        med = pd.Series(v[ok]).groupby(idx).median()
        m[med.index.to_numpy()] = med.to_numpy()
    if (c > 0).any() and (c == 0).any():
        g = np.arange(n)
        m[c == 0] = np.interp(g[c == 0], g[c > 0], m[c > 0])
    m[~np.isfinite(m)] = 0.0
    return m, c


def fold_signal(wins, sig):
    """wins: list of window dicts for one star. Returns (glob, loc, scal_part, per-window transit info)."""
    per, ep, dur_d = sig.period, sig.epoch_bjd, sig.duration_h / 24.0
    T, F, C, info = [], [], [], []
    for w in wins:
        t = w["t0_bjd"] + np.arange(L) * P.BIN_DAYS
        n = np.round((t - ep) / per)
        dt = t - (ep + n * per)
        intr = np.abs(dt) < dur_d                       # mask +-1 duration around each transit
        f = _detrend(w["flux_rel"].astype(float), w["mask"], ~intr)
        if w["has_c"]:
            dx = _detrend(w["cx"].astype(float), w["mask"], ~intr)
            dy = _detrend(w["cy"].astype(float), w["mask"], ~intr)
            c = np.hypot(dx, dy)
        else:
            c = np.full(L, np.nan)
        T.append(t); F.append(f); C.append(c)
        # transit centres inside this window with a filled bin within +-half duration
        k0 = np.ceil((w["t0_bjd"] - P.BIN_DAYS / 2 - ep) / per); k1 = np.floor((t[-1] - ep) / per)
        half = max(1, int(round(dur_d / P.BIN_DAYS / 2)))
        tp = np.zeros(NP, bool); cnt = 0
        for kk in np.arange(k0, k1 + 1):
            b = int(round((ep + kk * per - w["t0_bjd"]) / P.BIN_DAYS))
            if 0 <= b < L and w["mask"][max(0, b - half):b + half + 1].any():
                cnt += 1; tp[b // PATCH] = True
        info.append((cnt, tp))
    t = np.concatenate(T); f = np.concatenate(F); c = np.concatenate(C)
    ok = np.isfinite(f)
    t, f, c = t[ok], f[ok], c[ok]
    cyc = (t - ep) / per
    ph = (cyc + 0.5) % 1 - 0.5                           # transit at 0
    x = ph * per / dur_d                                 # in durations
    ph2 = cyc % 1 - 0.5                                  # secondary at 0
    x2 = ph2 * per / dur_d
    odd = np.round(cyc) % 2 == 1

    oot = np.abs(x) > 1.5
    base = np.median(f[oot]) if oot.sum() > 10 else 0.0
    sig_pt = P._robust_sigma(f[oot] - base) if oot.sum() > 10 else P._robust_sigma(f)
    core = np.abs(x) < 0.25
    n_core = max(int(core.sum()), 1)
    depth = base - (np.mean(f[core]) if core.any() else base)
    floor = 3 * sig_pt / np.sqrt(n_core)
    norm = max(depth, floor, 1e-6)

    fz = (f - base) / norm
    g_f, g_n = _bin(ph, fz, -0.5, 0.5, NG)
    lo, hi = -LOC_DUR, LOC_DUR
    l_f, l_n = _bin(x, fz, lo, hi, NL)
    l_o, _ = _bin(x[odd], fz[odd], lo, hi, NL)
    l_e, _ = _bin(x[~odd], fz[~odd], lo, hi, NL)
    l_s, _ = _bin(x2, fz, lo, hi, NL)
    cok = np.isfinite(c)
    if cok.sum() > 20:
        coot = cok & oot
        cb = np.median(c[coot]) if coot.sum() > 10 else np.median(c[cok])
        cs = P._robust_sigma(c[coot] - cb) if coot.sum() > 10 else 1.0
        cz = np.where(cok, (c - cb) / max(cs, 1e-9), np.nan)
        g_c, _ = _bin(ph, cz, -0.5, 0.5, NG)
        l_c, _ = _bin(x, cz, lo, hi, NL)
    else:
        g_c = np.zeros(NG); l_c = np.zeros(NL)
    glob = np.stack([g_f, g_c, g_n / max(g_n.max(), 1)]).astype(np.float32)
    loc = np.stack([l_f, l_o, l_e, l_s, l_c, l_n / max(l_n.max(), 1)]).astype(np.float32)
    np.clip(glob[:2], -20, 20, out=glob[:2]); np.clip(loc[:5], -20, 20, out=loc[:5])
    n_tr = len(np.unique(np.round(cyc[np.abs(x) < 0.5])))
    snr = depth / (sig_pt / np.sqrt(n_core)) if sig_pt > 0 else 0.0
    return glob, loc, dict(depth_ppm=depth * 1e6, snr=snr, n_transits=n_tr), info


def scalars(sig, wins, fold, info):
    w0 = wins[0]
    def opt(v, fn):
        return (fn(v), 1.0) if v is not None and np.isfinite(v) and v > 0 else (0.0, 0.0)
    dc, has_dc = opt(sig.depth_ppm, np.log10)
    mag, hm = opt(w0["magnitude"], lambda v: v / 10)
    teff, ht = opt(w0["teff"], lambda v: v / 5000)
    rad, hr = opt(w0["radius"], np.log10)
    return np.array([
        np.log10(sig.period), np.log10(sig.duration_h), np.log10(max(fold["depth_ppm"], 1.0)),
        dc, has_dc, np.sign(fold["snr"]) * np.log1p(abs(fold["snr"])), np.log1p(fold["n_transits"]),
        mag, hm, teff, ht, rad, hr,
        np.log10(np.median([w["sigma"] for w in wins])), float(any(w["has_c"] for w in wins)),
        np.mean([c > 0 for c, _ in info]),
    ], np.float32)


# ---------------------------------------------------------------- writer

class SplitWriter:
    def __init__(self, split, cap):
        d = os.path.join(OUT, split)
        os.makedirs(d, exist_ok=True)
        self.d, self.n, self.meta = d, 0, []
        spec = dict(flux_rel=((cap, W, L), np.float32), err_rel=((cap, W, L), np.float16),
                    mask=((cap, W, L), bool), cx=((cap, W, L), np.float32), cy=((cap, W, L), np.float32),
                    has_c=((cap, W), bool), sigma=((cap, W), np.float32), tpatch=((cap, W, NP), bool),
                    n_win=((cap,), np.int8), star=((cap, 3), np.float32),
                    glob=((cap, 3, NG), np.float32), loc=((cap, 6, NL), np.float32),
                    scal=((cap, NS), np.float32), cls=((cap,), np.int8))
        self.a = {k: open_memmap(os.path.join(d, f"{k}.npy"), "w+", dt, shp) for k, (shp, dt) in spec.items()}

    def add(self, sig, wins, sel, glob, loc, scal):
        i = self.n
        for j, wi in enumerate(sel):
            w = wins[wi]
            self.a["flux_rel"][i, j] = w["flux_rel"]; self.a["err_rel"][i, j] = w["err_rel"]
            self.a["mask"][i, j] = w["mask"]; self.a["has_c"][i, j] = w["has_c"]
            self.a["cx"][i, j] = w["cx"]; self.a["cy"][i, j] = w["cy"]
            self.a["sigma"][i, j] = w["sigma"]
        self.a["n_win"][i] = len(sel)
        w0 = wins[0]
        self.a["star"][i] = [np.nan if w0[k] is None else w0[k] for k in ("magnitude", "teff", "radius")]
        self.a["glob"][i] = glob; self.a["loc"][i] = loc; self.a["scal"][i] = scal
        self.a["cls"][i] = sig.cls
        self.meta.append(dict(signal_id=sig.signal_id, mission=sig.mission, star_id=int(sig.star_id),
                              cls=int(sig.cls), raw_label=sig.raw_label, period=sig.period,
                              duration_h=sig.duration_h, n_win=len(sel), n_star_windows=len(wins)))
        self.n += 1
        return i

    def close(self):
        for v in self.a.values():
            v.flush()
        m = pd.DataFrame(self.meta)
        m.to_parquet(os.path.join(self.d, "meta.parquet"))
        with open(os.path.join(self.d, "n.txt"), "w") as fh:
            fh.write(str(self.n))
        return m


def load_window(z, i):
    has_c = bool(z["has_c"][i])
    def opt(v):
        return None if np.isnan(v) else float(v)
    return dict(flux_rel=z["flux_rel"][i], err_rel=z["err_rel"][i], mask=z["mask"][i],
                cx=z["cx"][i] if has_c else np.full(L, np.nan, np.float32),
                cy=z["cy"][i] if has_c else np.full(L, np.nan, np.float32),
                has_c=has_c, sigma=float(z["sigma"][i]), t0_bjd=float(z["t0_bjd"][i]),
                magnitude=opt(z["magnitude"][i]), teff=opt(z["teff"][i]), radius=opt(z["radius"][i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", nargs="*", help="signal ids: plot their views instead of building")
    args = ap.parse_args()
    sig = pd.read_parquet(SIGNALS)
    sig = sig[sig.split.isin(SPLITS)].reset_index(drop=True)
    if args.plot:
        sig = sig[sig.signal_id.isin(args.plot)].reset_index(drop=True)
    idx = pd.read_parquet(INDEX)
    idx["folder"] = idx.path.str.split("/").str[-2]
    star_split = {(r.mission, r.star_id): r.split for r in sig[["mission", "star_id", "split"]].drop_duplicates().itertuples()}
    idx = idx[[star_split.get((m, s)) == f for m, s, f in zip(idx.mission, idx.star_id, idx.folder)]]
    need = collections.Counter(zip(idx.mission, idx.star_id))
    by_star = collections.defaultdict(list)
    for r in sig.itertuples():
        by_star[(r.mission, r.star_id)].append(r)
    print(f"{len(sig)} signals on {len(by_star)} stars, {len(idx):,} windows to read", flush=True)

    writers = {} if args.plot else {s: SplitWriter(s, int((sig.split == s).sum())) for s in SPLITS}
    pending = collections.defaultdict(list)
    no_win = 0; done = 0; plots = []
    idx = idx.assign(base=idx.path.str.split("/").str[-1]).sort_values(["mission", "base", "path", "idx"])
    for path, grp in idx.groupby("path", sort=False):
        with np.load(path) as zz:
            z = {k: zz[k] for k in zz.files}
        for r in grp.itertuples():
            key = (r.mission, r.star_id)
            pending[key].append(load_window(z, r.idx))
            if len(pending[key]) < need[key]:
                continue
            wins = sorted(pending.pop(key), key=lambda w: w["t0_bjd"])
            for s in by_star[key]:
                glob, loc, fold, info = fold_signal(wins, s)
                cnt = np.array([c for c, _ in info])
                order = np.argsort(-cnt, kind="stable")
                sel = [int(i) for i in order[:W] if cnt[i] > 0] or [int(order[0])]
                sc = scalars(s, wins, fold, info)
                if args.plot:
                    plots.append((s, glob, loc, fold)); continue
                wr = writers[s.split]
                i = wr.add(s, wins, sel, glob, loc, sc)
                for j, wi in enumerate(sel):
                    wr.a["tpatch"][i, j] = info[wi][1]
                no_win += cnt.sum() == 0
                done += 1
                if done % 2000 == 0:
                    print(f"{done}/{len(sig)} signals", flush=True)
    if args.plot:
        plot_views(plots)
        return
    for s, w in writers.items():
        m = w.close()
        print(s, len(m), "examples;", m.groupby(["mission", "cls"]).size().to_dict())
    print(f"signals with no covered transit (window chosen anyway): {no_win}; stars never completed: {len(pending)}")


def plot_views(items):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(items), 3, figsize=(15, 2.6 * len(items)), squeeze=False)
    for (s, glob, loc, fold), ax in zip(items, axes):
        ax[0].plot(np.linspace(-0.5, 0.5, NG), glob[0], ".-", ms=2)
        ax[0].set_title(f"{s.signal_id} [{CLASSES[s.cls]}] P={s.period:.3f}d global", fontsize=8)
        xl = np.linspace(-LOC_DUR, LOC_DUR, NL)
        for k, lab in enumerate(["all", "odd", "even", "secondary"]):
            ax[1].plot(xl, loc[k], ".-", ms=2, label=lab)
        ax[1].legend(fontsize=6); ax[1].set_title(f"local; depth meas {fold['depth_ppm']:.0f} ppm "
                                                  f"(cat {s.depth_ppm:.0f}), snr {fold['snr']:.0f}", fontsize=8)
        ax[2].plot(xl, loc[4], ".-", ms=2); ax[2].set_title("local centroid (sigma)", fontsize=8)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "fold_check.png"), dpi=90)
    print("wrote", os.path.join(OUT, "fold_check.png"))


if __name__ == "__main__":
    main()
