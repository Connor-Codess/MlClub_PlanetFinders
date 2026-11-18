"""How realistic are the batman injections? Three checks against real TESS and Kepler transits.

What it does
    1. Distributions. For period bins, compares the duration and depth of lc-finetune batman
       injections (v3 physics) with TOIs on the TESS planet candidate list (reference-catalogs
       toi_ephemerides: PC, CP and KP). Reports quantiles and the share of injections whose
       duration falls outside the TOIs' 1st-99th percentile range for the same period bin.
    2. Classifier. Cuts single transits out of lc-finetune train windows: real planets, batman
       injections and transplants, each with per-transit SNR 7-40. Samples are matched in SNR
       and duration bins, so neither can give a class away. A gradient-boosted classifier then
       tries to tell real from batman (and, as a control, real from transplant) with 5-fold
       cross-validation:
           shape      the transit resampled to 40 bins over -2 to +2 durations, divided by its
                      depth, so only the shape is left
           raw        the flux_z cutout of +-16 hours in noise units, as the detector sees it
       ROC AUC near 0.5 means the classifier cannot tell them apart.
    3. Matched recall is computed by analyze.py from the training runs (real long-period
       planets against batman injections matched in SNR, period and transit count).

Why
    A detector trained on injections that differ from real transits in some systematic way can
    learn that difference instead of what a transit is.

Outputs (in --out)
    realism.json   every number above

How to run
    python -m experiments.finetune_mix.realism --data build/mix/source --out build/mix/results/realism \
        [--shards 8]
    (reads lc-finetune train shards from --data, downloading them if absent, plus reference-catalogs)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config

LOG = logging.getLogger("mix.realism")
PERIOD_BINS = (1, 3, 10, 30, 100, 730)
SNR_BINS = (7, 10, 15, 25, 40)
DURATION_BINS_H = (0, 2, 4, 8, 16, 48)
SHAPE_BINS = 40
RAW_HALF_SLOTS = 32
REFERENCE_REPO = "nyu-planet-finders/reference-catalogs"


def distributions(meta: pd.DataFrame, tois: pd.DataFrame) -> dict:
    bat = meta[(meta["inj_model"] == "batman")]
    out = []
    for lo, hi in zip(PERIOD_BINS[:-1], PERIOD_BINS[1:]):
        b = bat[(bat["inj_period_d"] >= lo) & (bat["inj_period_d"] < hi)]
        t = tois[(tois["period_d"] >= lo) & (tois["period_d"] < hi)]
        entry = {"period_lo": lo, "period_hi": hi, "batman": int(len(b)), "toi": int(len(t))}
        for name, col_b, col_t in (("duration_h", "inj_duration_h", "duration_h"), ("depth_ppm", "inj_depth_ppm", "depth_ppm")):
            entry[f"batman_{name}_q10_50_90"] = b[col_b].quantile([0.1, 0.5, 0.9]).round(2).tolist() if len(b) else None
            entry[f"toi_{name}_q10_50_90"] = t[col_t].quantile([0.1, 0.5, 0.9]).round(2).tolist() if len(t) else None
        if len(t) >= 20 and len(b):
            lo_d, hi_d = t["duration_h"].quantile([0.01, 0.99])
            entry["batman_outside_toi_duration_range"] = round(float(((b["inj_duration_h"] < lo_d)
                                                                       | (b["inj_duration_h"] > hi_d)).mean()), 4)
        out.append(entry)
    return {"period_bins": out, "batman_rows": int(len(bat)), "toi_rows": int(len(tois)),
            "eccentric_share": round(float((bat["inj_ecc"].fillna(0) > 0).mean()), 3) if "inj_ecc" in bat else None}


def cutouts(table: pd.DataFrame) -> pd.DataFrame:
    """Single-transit cutouts (shape and raw) from window rows with events."""
    out = []
    slots = np.arange(config.WINDOW_SLOTS) + 0.5
    for _, row in table.iterrows():
        flux = np.frombuffer(row["flux_z"], "<f2").astype(np.float64)
        observed = (np.frombuffer(row["flags"], np.uint8) & 1) > 0
        for e in (row["events"] if row["events"] is not None else []):
            if not (e["is_planet"] and e["visible"]) or not (SNR_BINS[0] <= e["snr"] < SNR_BINS[-1]):
                continue
            c, d = float(e["center_slot"]), max(float(e["duration_slots"]), 1.0)
            lo, hi = int(c) - RAW_HALF_SLOTS, int(c) + RAW_HALF_SLOTS
            if lo < 0 or hi > config.WINDOW_SLOTS or observed[lo:hi].mean() < 0.9:
                continue
            raw = np.where(observed[lo:hi], np.clip(flux[lo:hi], -config.INPUT_CLIP, config.INPUT_CLIP), 0.0)
            phase = (slots - c) / d
            near = (np.abs(phase) <= 2) & observed
            if near.sum() < 8:
                continue
            edges = np.linspace(-2, 2, SHAPE_BINS + 1)
            which = np.digitize(phase[near], edges) - 1
            sums = np.bincount(which, weights=flux[near], minlength=SHAPE_BINS)[:SHAPE_BINS]
            counts = np.bincount(which, minlength=SHAPE_BINS)[:SHAPE_BINS]
            shape = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
            centre = shape[SHAPE_BINS // 2 - 3:SHAPE_BINS // 2 + 3]
            depth = -np.nanmean(centre)
            if not np.isfinite(depth) or depth <= 0:
                continue
            shape = np.nan_to_num(shape / depth, nan=0.0)
            out.append({"category": row["category"], "snr": float(e["snr"]),
                        "duration_h": d * 0.5, "shape": shape, "raw": raw})
    return pd.DataFrame(out)


def matched(cut: pd.DataFrame, a: str, b: str, rng: np.random.Generator) -> pd.DataFrame:
    """Equal numbers of a and b in every (SNR bin, duration bin) cell."""
    cut = cut.assign(snr_bin=np.digitize(cut["snr"], SNR_BINS), dur_bin=np.digitize(cut["duration_h"], DURATION_BINS_H))
    picks = []
    for _, cell in cut[cut["category"].isin([a, b])].groupby(["snr_bin", "dur_bin"]):
        ga, gb = cell[cell["category"] == a], cell[cell["category"] == b]
        k = min(len(ga), len(gb))
        if k:
            picks += [ga.sample(k, random_state=int(rng.integers(1 << 30))),
                      gb.sample(k, random_state=int(rng.integers(1 << 30)))]
    return pd.concat(picks) if picks else cut.iloc[:0]


def classify(cut: pd.DataFrame, a: str, b: str, rng: np.random.Generator) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    data = matched(cut, a, b, rng)
    result = {"pairs": int(len(data) // 2)}
    if len(data) < 200:
        result["note"] = "too few matched cutouts"
        return result
    y = (data["category"] == a).to_numpy().astype(int)
    for name in ("shape", "raw"):
        x = np.stack(data[name].to_numpy())
        folds = StratifiedKFold(5, shuffle=True, random_state=0)
        prob = cross_val_predict(HistGradientBoostingClassifier(max_iter=300, random_state=0), x, y, cv=folds,
                                 method="predict_proba")[:, 1]
        result[f"{name}_auc"] = round(float(roc_auc_score(y, prob)), 4)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="folder with data/train/*.parquet of lc-finetune")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=8, help="lc-finetune train shards to cut transits from")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from huggingface_hub import hf_hub_download

    from .data import download

    files = sorted(args.data.glob("data/train/*.parquet")) or download(config.FINETUNE_REPO, "train", args.data)
    meta = pd.concat([pq.read_table(f, columns=["category", "inj_model", "inj_period_d", "inj_duration_h", "inj_depth_ppm",
                                                "inj_ecc", "inj_deep"]).to_pandas() for f in files], ignore_index=True)
    tois = pd.read_parquet(hf_hub_download(REFERENCE_REPO, "toi_ephemerides.parquet", repo_type="dataset",
                                           token=os.environ.get("HF_TOKEN")))
    tois = tois[tois["disposition"].isin(["PC", "CP", "KP"]) & (tois["period_d"] > 0)]
    result = {"distributions": distributions(meta, tois)}

    rng = np.random.default_rng(0)
    columns = ["category", "flux_z", "flags", "events"]
    frames = [pq.read_table(f, columns=columns).to_pandas() for f in files[: args.shards]]
    table = pd.concat(frames, ignore_index=True)
    table = table[table["category"].isin(["real_planet_short", "real_planet_long", "injected_batman", "injected_transplant"])]
    cut = cutouts(table)
    cut["category"] = cut["category"].replace({"real_planet_short": "real", "real_planet_long": "real",
                                               "injected_batman": "batman", "injected_transplant": "transplant"})
    result["cutouts"] = cut["category"].value_counts().to_dict()
    result["real_vs_batman"] = classify(cut, "real", "batman", rng)
    result["real_vs_transplant"] = classify(cut, "real", "transplant", rng)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "realism.json").write_text(json.dumps(result, indent=2))
    LOG.info("realism: %s", json.dumps({k: v for k, v in result.items() if k != "distributions"}))


if __name__ == "__main__":
    main()
