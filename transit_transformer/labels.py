"""Labeled transit signals for fine-tuning: Kepler DR24 TCEs + TESS TOIs.

    python3 labels.py          # writes data/examples/window_index.parquet, data/labels/signals.parquet

Classes:  0 planet  1 candidate  2 false positive  3 no planet

    Kepler   planet     DR24 PC whose KOI (same kepid, period within 0.1%) is CONFIRMED
             candidate  other DR24 PC
             FP         DR24 AFP
             no planet  DR24 NTP                      (UNK dropped)
    TESS     planet     TFOPWG CP, KP
             candidate  PC, APC
             FP         FP
             no planet  FA, plus random-ephemeris negatives on stars with no TOI
                        (raw_label "RAND"; period/duration pairs copied from real TOIs)

Splits are the pretraining star splits (splits.assign). TESS stars whose only
windows are from sectors >= 100 are "test_future". A TESS signal within 2"
of a Kepler star in the Kepler train split is "excluded" (the Kepler model saw it).

Inputs (data/labels/): toi_ephem.csv and koi_cumulative.csv (NASA Exoplanet Archive
TAP), kepler_positions.parquet; DR24 table from data/probe/dr24_tce.parquet.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

import preprocess as P
import splits as S

CLASSES = ["planet", "candidate", "false_positive", "no_planet"]
LAB = "data/labels"
INDEX = "data/examples/window_index.parquet"
SIGNALS = os.path.join(LAB, "signals.parquet")
N_RANDOM_NEG = 2000
MATCH_ARCSEC = 2.0


def build_window_index(root="data/shards"):
    """One row per window: mission, star_id, segment, t0_bjd, path, idx."""
    if os.path.exists(INDEX):
        return pd.read_parquet(INDEX)
    rows = []
    for mission in ("kepler", "tess"):
        for p in sorted(glob.glob(os.path.join(root, mission, "*", "*.npz"))):
            with np.load(p) as z:
                sid, seg, t0 = z["star_id"], z["segment"], z["t0_bjd"]
            rows.append(pd.DataFrame(dict(mission=mission, star_id=sid, segment=seg, t0_bjd=t0,
                                          path=p, idx=np.arange(len(sid), dtype=np.int32))))
    df = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    df.to_parquet(INDEX)
    return df


def tess_star_split(index):
    """Star-level split: hash split, but stars only seen at sectors >= 100 -> test_future."""
    t = index[index.mission == "tess"].groupby("star_id").segment.min()
    out = {}
    for sid, smin in t.items():
        s = S.assign("tess", sid)
        if s == "train" and smin >= S.FUTURE_SECTOR:
            s = "test_future"
        if s == "val" and smin >= S.FUTURE_SECTOR:
            s = "test_future"
        out[int(sid)] = s
    return out


def kepler_signals():
    tce = pd.read_parquet("data/probe/dr24_tce.parquet")
    tce = tce[tce.av_training_set.isin(["PC", "AFP", "NTP"])].copy()
    koi = pd.read_csv(os.path.join(LAB, "koi_cumulative.csv"))
    m = tce.merge(koi[["kepid", "koi_period", "koi_disposition"]], on="kepid", how="left")
    m["match"] = (m.tce_period - m.koi_period).abs() / m.koi_period < 1e-3
    confirmed = set(m.loc[m.match & (m.koi_disposition == "CONFIRMED"), "rowid"])
    cls = tce.av_training_set.map({"PC": 1, "AFP": 2, "NTP": 3}).to_numpy().copy()
    cls[(tce.av_training_set == "PC").to_numpy() & tce.rowid.isin(confirmed).to_numpy()] = 0
    return pd.DataFrame(dict(
        mission="kepler", star_id=tce.kepid.astype(np.int64),
        signal_id=[f"KIC{k}.{n:02d}" for k, n in zip(tce.kepid, tce.tce_plnt_num)],
        period=tce.tce_period, epoch_bjd=tce.tce_time0bk + P.BJD_OFFSET["kepler"],
        duration_h=tce.tce_duration, depth_ppm=tce.tce_depth, cls=cls,
        raw_label=np.where(cls == 0, "PC-CONFIRMED", tce.av_training_set),
        ra=np.nan, dec=np.nan))


def tess_signals():
    t = pd.read_csv(os.path.join(LAB, "toi_ephem.csv"))
    t = t[t.pl_orbper.gt(0) & t.pl_tranmid.notna() & t.pl_trandurh.gt(0) & t.tfopwg_disp.notna()]
    cmap = {"CP": 0, "KP": 0, "PC": 1, "APC": 1, "FP": 2, "FA": 3}
    return pd.DataFrame(dict(
        mission="tess", star_id=t.tid.astype(np.int64), signal_id="TOI-" + t.toi.astype(str),
        period=t.pl_orbper, epoch_bjd=t.pl_tranmid, duration_h=t.pl_trandurh,
        depth_ppm=t.pl_trandep, cls=t.tfopwg_disp.map(cmap).astype(int),
        raw_label=t.tfopwg_disp, ra=t.ra, dec=t.dec))


def random_negatives(index, toi, star_split, n, seed=0):
    rng = np.random.default_rng(seed)
    tess = index[index.mission == "tess"]
    first = tess.groupby("star_id").t0_bjd.min()
    pool = np.setdiff1d(first.index.to_numpy(), toi.star_id.to_numpy())
    stars = rng.choice(pool, n, replace=False)
    pairs = toi[["period", "duration_h"]].to_numpy()
    pick = pairs[rng.integers(len(pairs), size=n)]
    epoch = first.loc[stars].to_numpy() + rng.random(n) * pick[:, 0]
    return pd.DataFrame(dict(
        mission="tess", star_id=stars.astype(np.int64), signal_id=[f"RAND-{s}" for s in stars],
        period=pick[:, 0], epoch_bjd=epoch, duration_h=pick[:, 1], depth_ppm=np.nan,
        cls=3, raw_label="RAND", ra=np.nan, dec=np.nan))


def kepler_overlap(tess_sig):
    """TESS signals within MATCH_ARCSEC of a Kepler star in the Kepler train split."""
    kp = pd.read_parquet(os.path.join(LAB, "kepler_positions.parquet"))
    kp = kp[[S.assign("kepler", k) == "train" for k in kp.kepid]]
    has = tess_sig.ra.notna()
    if not has.any():
        return np.zeros(len(tess_sig), bool)
    ra1, de1 = np.radians(tess_sig.ra[has].to_numpy())[:, None], np.radians(tess_sig.dec[has].to_numpy())[:, None]
    hit = np.zeros(has.sum(), bool)
    kra, kde = np.radians(kp.ra.to_numpy())[None], np.radians(kp.dec.to_numpy())[None]
    for s in range(0, len(ra1), 500):
        sl = slice(s, s + 500)
        cosd = np.sin(de1[sl]) * np.sin(kde) + np.cos(de1[sl]) * np.cos(kde) * np.cos(ra1[sl] - kra)
        sep = np.degrees(np.arccos(np.clip(cosd, -1, 1))) * 3600
        hit[sl] = (sep < MATCH_ARCSEC).any(1)
    out = np.zeros(len(tess_sig), bool)
    out[np.flatnonzero(has.to_numpy())] = hit
    return out


def main():
    index = build_window_index()
    print(f"window index: {len(index):,} windows")
    star_split = tess_star_split(index)
    kep = kepler_signals()
    tess = tess_signals()
    have = set(star_split)
    tess = tess[tess.star_id.isin(have)].copy()
    kep["split"] = [S.assign("kepler", s) for s in kep.star_id]
    tess["split"] = tess.star_id.map(star_split)
    ov = kepler_overlap(tess)
    tess.loc[ov, "split"] = "excluded"
    print(f"TESS signals near a Kepler training star -> excluded: {ov.sum()}")
    neg = random_negatives(index, tess, star_split, N_RANDOM_NEG)
    neg["split"] = neg.star_id.map(star_split)
    sig = pd.concat([kep, tess, neg], ignore_index=True)
    sig.to_parquet(SIGNALS)
    tab = pd.crosstab([sig.mission, sig.split], sig.cls.map(dict(enumerate(CLASSES))))
    print(tab.to_string())


if __name__ == "__main__":
    main()
