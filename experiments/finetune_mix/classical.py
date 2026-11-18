"""A fair classical baseline: wotan detrending, a single-event box search and BLS.

What it does
    For every lc-eval val window (packed by pack.py --eval):
        1. Detrends flux_z with wotan's biweight filter (window 0.75 d; Hippke et al. 2019).
        2. Rescales the residuals by their robust scatter (1.4826 x MAD).
        3. Single-event search: box SNR -sum/sqrt(n) for durations of 1-13 h at every slot.
        4. Periodic search: astropy's BoxLeastSquares over 0.6-13.5 d (two or more transits in
           27 days; 0.6 d is the shortest period longer than every duration) with the same
           durations, objective "snr".
        5. Patch score: the best single-event SNR in the patch, raised to the BLS SNR at the
           patches holding the best period's transits.
    The patch scores are then put through the same thresholds and recall numbers as the
    trained models (evaluate.score).

Why
    The box filter of wave 1 used a crude running-median detrend and no period search, so it
    was a weak bar. This version follows the standard practice of transit searches.

Outputs
    --chunk i/k   patch scores of the i-th of k slices of the eval rows (chunk_<i>.npz)
    --merge       metrics.json and scores.npz for all rows (arm "classical")

How to run
    python -m experiments.finetune_mix.classical --eval build/mix/eval --chunk 0/10 --out build/mix/classical
    python -m experiments.finetune_mix.classical --eval build/mix/eval --merge --out build/mix/classical \
        --result build/mix/results/classical
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from . import config
from .data import load_pack
from .evaluate import average_precision, score, window_scores

LOG = logging.getLogger("mix.classical")
SLOT_DAYS = 30.0 / 1440.0
WOTAN_WINDOW_D = 0.75
DURATIONS_SLOTS = (2, 4, 6, 10, 16, 26)
BLS_PERIODS = np.exp(np.linspace(np.log(0.6), np.log(13.5), 2000))


def single_event(resid: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Best box SNR per slot over the durations (resid in units of its scatter, 0 where unobserved)."""
    cs = np.concatenate([[0.0], np.cumsum(resid)])
    cn = np.concatenate([[0.0], np.cumsum(observed)])
    slots = np.arange(config.WINDOW_SLOTS)
    best = np.full(config.WINDOW_SLOTS, -np.inf)
    for d in DURATIONS_SLOTS:
        lo = np.clip(slots - d // 2, 0, config.WINDOW_SLOTS)
        hi = np.clip(lo + d, 0, config.WINDOW_SLOTS)
        n = cn[hi] - cn[lo]
        snr = np.where(n >= max(d / 2, 1), -(cs[hi] - cs[lo]) / np.sqrt(np.maximum(n, 1)), -np.inf)
        best = np.maximum(best, snr)
    return best


def window_patch_scores(flux: np.ndarray, observed: np.ndarray) -> np.ndarray:
    from astropy.timeseries import BoxLeastSquares
    from wotan import flatten

    out = np.full(config.N_PATCHES, -np.inf, dtype=np.float32)
    obs = observed.astype(bool)
    if obs.sum() < 100:
        return out
    t = (np.arange(config.WINDOW_SLOTS) + 0.5)[obs] * SLOT_DAYS
    f = 1.0 + 1e-3 * np.clip(np.nan_to_num(flux[obs]), -config.INPUT_CLIP, config.INPUT_CLIP)
    flat = flatten(t, f, method="biweight", window_length=WOTAN_WINDOW_D, edge_cutoff=0.0, break_tolerance=0.5)
    good = np.isfinite(flat)
    resid_obs = np.where(good, (flat - 1.0) / 1e-3, 0.0)
    scatter = 1.4826 * np.median(np.abs(resid_obs[good] - np.median(resid_obs[good]))) if good.any() else 1.0
    resid_obs = resid_obs / (scatter if scatter > 0 else 1.0)
    resid = np.zeros(config.WINDOW_SLOTS)
    used = np.zeros(config.WINDOW_SLOTS)
    resid[np.flatnonzero(obs)[good]] = resid_obs[good]
    used[np.flatnonzero(obs)[good]] = 1.0
    per_slot = single_event(resid, used)
    out = per_slot.reshape(config.N_PATCHES, config.PATCH_SLOTS).max(axis=1).astype(np.float32)

    if good.sum() >= 200:
        bls = BoxLeastSquares(t[good], resid_obs[good], dy=np.ones(int(good.sum())))
        durations = np.array(DURATIONS_SLOTS) * SLOT_DAYS
        power = bls.power(BLS_PERIODS, durations, objective="snr")
        best = int(np.nanargmax(power.power))
        period, t0, snr = float(power.period[best]), float(power.transit_time[best]), float(power.power[best])
        if np.isfinite(snr):
            epochs = t0 + period * np.arange(-int(30 / period) - 1, int(30 / period) + 2)
            epochs = epochs[(epochs >= 0) & (epochs < config.WINDOW_SLOTS * SLOT_DAYS)]
            if len(epochs) >= 2:
                patches = np.unique((epochs / SLOT_DAYS // config.PATCH_SLOTS).astype(int))
                out[patches] = np.maximum(out[patches], snr)
    return out


def run_chunk(pack: dict, index: int, count: int) -> np.ndarray:
    rows = np.array_split(np.arange(len(pack["x"])), count)[index]
    scores = np.empty((len(rows), config.N_PATCHES), dtype=np.float32)
    started = time.time()
    for k, i in enumerate(rows):
        scores[k] = window_patch_scores(pack["x"][i, 0].astype(np.float64), pack["observed"][i])
        if (k + 1) % 1000 == 0:
            LOG.info("%d/%d windows (%.1f min)", k + 1, len(rows), (time.time() - started) / 60)
    return rows, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk", help="i/k: score the i-th of k slices")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--result", type=Path, help="with --merge: where metrics.json and scores.npz go")
    parser.add_argument("--limit", type=int, help="score only the first N rows (smoke tests)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pack = load_pack(args.eval, mmap=True)
    if args.limit:
        pack = {k: (v[: args.limit] if k != "events" else v) for k, v in pack.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    if args.chunk:
        index, count = (int(v) for v in args.chunk.split("/"))
        rows, scores = run_chunk(pack, index, count)
        np.savez_compressed(args.out / f"chunk_{index}.npz", rows=rows, scores=scores)
        return
    if not args.merge:
        parser.error("pass --chunk i/k or --merge")
    patch = np.full((len(pack["x"]), config.N_PATCHES), -np.inf, dtype=np.float32)
    for path in sorted(args.out.glob("chunk_*.npz")):
        data = np.load(path)
        patch[data["rows"]] = data["scores"]
    label = np.asarray(pack["label"])
    any_score, on_target = window_scores(patch, np.asarray(pack["target"]), label)
    finite = np.where(np.isfinite(patch), patch, -1e9)
    result = score(pack["meta"], any_score, on_target, average_precision(finite, label))
    result["run"] = {"arm": "classical", "seed": 0, "rows": 0,
                     "method": "wotan biweight 0.75 d + single-event box + astropy BLS 0.6-13.5 d"}
    target = args.result or args.out
    target.mkdir(parents=True, exist_ok=True)
    (target / "metrics.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(target / "scores.npz", any_score=any_score.astype(np.float32),
                        on_target=on_target.astype(np.float32))
    LOG.info("classical: %s", json.dumps({k: v for k, v in result["operating_points"]["0.01"].items()
                                          if k != "bins"}))


if __name__ == "__main__":
    main()
