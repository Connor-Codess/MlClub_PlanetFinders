"""Scoring a detector on lc-eval val, and the box-filter baseline.

What it does
    window_scores()  per eval window: the highest patch score anywhere (used for false
                     alarms) and the highest score on the window's transit patches
    score()          thresholds at fixed false-alarm rates and every recall number
    box_filter()     a classical single-event box search (BLS-style without folding)
                     run on the same windows, as the bar the trained models must beat

Operating points
    For each rate in config.FALSE_ALARM_RATES, the threshold is the score that that share
    of negative_random windows exceed somewhere in the window (1% = one random 27-day
    window in 100 raises an alarm). A transit window counts as found when a patch within
    one patch of a transit mid-time is above threshold.

Numbers reported per operating point
    injected_recall   batman and transplant windows together
    faint_recall      injected windows with combined SNR in config.FAINT_RANGE
    strong_recall     injected windows with combined SNR >= 10
    batman_recall, transplant_recall
    real_short_recall, real_long_recall   real planets on held-out stars
    eb_false_alarm    share of negative_eb windows raising an alarm
    bins              recall by combined SNR, per-transit SNR and period (injections) and
                      by stacked SNR (real planets)
    Plus patch_average_precision (threshold-free, patches labeled 0 or 1).

How to run
    python -m experiments.finetune_mix.evaluate --eval build/mix/eval --out build/mix/results/box_filter
    (runs the box-filter baseline; trained models are scored by train.py)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import load_pack

LOG = logging.getLogger("mix.evaluate")


def window_scores(patch_scores: np.ndarray, target: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = label != config.LABEL_IGNORE
    any_score = np.where(valid, patch_scores, -np.inf).max(axis=1)
    on_target = np.where(target.astype(bool), patch_scores, -np.inf).max(axis=1)
    return any_score, on_target


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    keep = labels != config.LABEL_IGNORE
    s, y = scores[keep].astype(np.float64), (labels[keep] == config.LABEL_TRANSIT)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def binned(values: pd.Series, found: pd.Series, edges) -> list[dict]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (values >= lo) & (values < hi)
        n = int(mask.sum())
        out.append({"lo": float(lo), "hi": float(hi), "n": n,
                    "recall": float(found[mask].mean()) if n else None})
    return out


def score(meta: pd.DataFrame, any_score: np.ndarray, on_target: np.ndarray, patch_ap: float) -> dict:
    meta = meta.reset_index(drop=True)
    random = any_score[(meta["category"] == "negative_random").to_numpy()]
    result = {"patch_average_precision": patch_ap, "operating_points": {}}
    has_target = np.isfinite(on_target)
    injected = meta["category"].isin(config.INJECTED).to_numpy() & has_target
    combined = meta["inj_snr_total"]
    for rate in config.FALSE_ALARM_RATES:
        threshold = float(np.quantile(random, 1.0 - rate))
        found = pd.Series(on_target > threshold)
        alarm = pd.Series(any_score > threshold)

        def recall(mask) -> float | None:
            mask = np.asarray(mask, dtype=bool)
            return float(found[mask].mean()) if mask.sum() else None

        cat = meta["category"].to_numpy()
        lo, hi = config.FAINT_RANGE
        point = {
            "threshold": threshold,
            "random_false_alarm": float(alarm[cat == "negative_random"].mean()),
            "injected_recall": recall(injected),
            "faint_recall": recall(injected & (combined >= lo).to_numpy() & (combined < hi).to_numpy()),
            "strong_recall": recall(injected & (combined >= 10).to_numpy()),
            "batman_recall": recall(injected & (cat == "injected_batman")),
            "transplant_recall": recall(injected & (cat == "injected_transplant")),
            "real_short_recall": recall((cat == "real_planet_short") & has_target),
            "real_long_recall": recall((cat == "real_planet_long") & has_target),
            "eb_false_alarm": float(alarm[cat == "negative_eb"].mean()) if (cat == "negative_eb").any() else None,
            "bins": {},
        }
        inj = meta[injected]
        point["bins"]["combined_snr"] = binned(inj["inj_snr_total"], found[injected], config.COMBINED_SNR_BINS)
        point["bins"]["per_transit_snr"] = binned(inj["inj_snr"], found[injected], config.PER_TRANSIT_SNR_BINS)
        point["bins"]["period_d"] = binned(inj["inj_period_d"], found[injected], config.PERIOD_BINS_D)
        short_p = injected & (meta["inj_period_d"] < 10).to_numpy()
        point["bins"]["short_period_per_transit_snr"] = binned(meta.loc[short_p, "inj_snr"], found[short_p],
                                                               config.PER_TRANSIT_SNR_BINS)
        for name in config.REAL:
            mask = (cat == name) & has_target
            point["bins"][f"{name}_stacked_snr"] = binned(meta.loc[mask, "stacked_snr"].fillna(0), found[mask],
                                                          config.COMBINED_SNR_BINS)
        result["operating_points"][str(rate)] = point
    return result


def box_filter(x: np.ndarray, observed: np.ndarray, chunk: int = 4000) -> np.ndarray:
    """Patch scores of a single-event box search on detrended flux_z (higher = deeper dip).

    flux_z is clipped like the model input and detrended with a running median of about a day;
    slots where that median window is under 80% observed (edges of the window and of gaps) are
    not scored.
    For each duration, the box statistic -sum/sqrt(n) is standardized by its median and MAD over
    the window, and each patch takes the best value over its slots and the durations.
    """
    n = len(x)
    out = np.empty((n, config.N_PATCHES), dtype=np.float32)
    half = config.BOX_DETREND_SLOTS // 2
    for start in range(0, n, chunk):
        flux = np.clip(np.nan_to_num(x[start:start + chunk, 0].astype(np.float32)), -config.INPUT_CLIP,
                       config.INPUT_CLIP)
        obs = observed[start:start + chunk].astype(bool)
        frame = pd.DataFrame(np.where(obs, flux, np.nan).T)
        trend = frame.rolling(config.BOX_DETREND_SLOTS, center=True, min_periods=half // 2).median().to_numpy().T
        resid = np.where(obs, flux - np.nan_to_num(trend), 0.0)
        cs = np.concatenate([np.zeros((len(flux), 1)), np.cumsum(resid, axis=1)], axis=1)
        cn = np.concatenate([np.zeros((len(flux), 1)), np.cumsum(obs, axis=1)], axis=1)
        best = np.full(flux.shape, -np.inf, dtype=np.float32)
        slots = np.arange(config.WINDOW_SLOTS)
        # slots whose detrending window is under 80% observed (window edges, gap edges) are not
        # scored: a one-sided running median cannot follow a ramp there
        w_lo = np.clip(slots - half, 0, config.WINDOW_SLOTS)
        w_hi = np.clip(slots + half + 1, 0, config.WINDOW_SLOTS)
        supported = (cn[:, w_hi] - cn[:, w_lo]) >= 0.8 * config.BOX_DETREND_SLOTS
        for d in config.BOX_DURATIONS_SLOTS:
            lo = np.clip(slots - d // 2, 0, config.WINDOW_SLOTS)
            hi = np.clip(lo + d, 0, config.WINDOW_SLOTS)
            total = cs[:, hi] - cs[:, lo]
            count = cn[:, hi] - cn[:, lo]
            valid = (count >= max(d / 2, 1)) & supported
            ses = np.where(valid, -total / np.sqrt(np.maximum(count, 1)), np.nan)
            # divide by the window's own scatter of the statistic at this duration, so red noise
            # on long timescales does not dominate (as BLS normalizes by noise at each duration)
            center = np.nanmedian(ses, axis=1, keepdims=True)
            spread = 1.4826 * np.nanmedian(np.abs(ses - center), axis=1, keepdims=True)
            ses = (ses - center) / np.where(spread > 0, spread, 1.0)
            best = np.fmax(best, np.where(valid, ses, -np.inf).astype(np.float32))
        out[start:start + chunk] = best.reshape(len(flux), config.N_PATCHES, config.PATCH_SLOTS).max(axis=2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", type=Path, required=True, help="packed lc-eval val (pack.py --eval)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pack = load_pack(args.eval)
    patch = box_filter(pack["x"], pack["observed"])
    any_score, on_target = window_scores(patch, pack["target"], pack["label"])
    result = score(pack["meta"], any_score, on_target, average_precision(patch, pack["label"]))
    result["run"] = {"arm": "box_filter", "seed": 0, "rows": 0}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(args.out / "scores.npz", any_score=any_score.astype(np.float32),
                        on_target=on_target.astype(np.float32))
    LOG.info("box filter: %s", json.dumps({k: v for k, v in result["operating_points"]["0.01"].items()
                                           if k != "bins"}))


if __name__ == "__main__":
    main()


def hiding_rows(meta: pd.DataFrame, events: pd.DataFrame, target: np.ndarray) -> pd.DataFrame:
    """Injected eval windows with two or more visible transits and combined SNR >= 4.

    One row per window: the transit kept (the first visible one), the patches that count as
    finding it (its mid-time patch and one either side) and the slots of every other transit,
    which the test hides.
    """
    planets = events[events["is_planet"] & events["visible"]].sort_values(["row", "center_slot"])
    counts = planets.groupby("row").size()
    rows = []
    for row, group in planets[planets["row"].isin(counts[counts >= 2].index)].groupby("row"):
        m = meta.iloc[row]
        if m["category"] not in config.INJECTED or not (m["inj_snr_total"] >= 4):
            continue
        keep = group.iloc[0]
        patch = int(keep["center_slot"] // config.PATCH_SLOTS)
        patches = [p for p in range(patch - config.TARGET_DILATION, patch + config.TARGET_DILATION + 1)
                   if 0 <= p < config.N_PATCHES]
        hide = np.zeros(config.WINDOW_SLOTS, dtype=bool)
        slots = np.arange(config.WINDOW_SLOTS) + 0.5
        for _, other in group.iloc[1:].iterrows():
            hide |= np.abs(slots - other["center_slot"]) < other["duration_slots"] / 2.0 + 2.0
        rows.append({"row": int(row), "keep_patches": patches, "hide": hide,
                     "n_transits": len(group), "inj_snr_total": float(m["inj_snr_total"]),
                     "inj_snr": float(m["inj_snr"]), "inj_period_d": float(m["inj_period_d"])})
    return pd.DataFrame(rows)


def hiding_test(predict_fn, pack: dict, threshold: float) -> dict:
    """Recall of one kept transit with its siblings present, and with them hidden.

    predict_fn(x, observed) returns patch probabilities. A large drop when siblings are hidden
    means the model combines repeated transits; no drop means it finds each one alone.
    """
    table = hiding_rows(pack["meta"], pack["events"], pack["target"])
    if table.empty:
        return {"windows": 0}
    idx = table["row"].to_numpy()
    x = pack["x"][idx].astype(np.float32)
    observed = pack["observed"][idx].copy()
    full = predict_fn(x, observed)
    hide = np.stack(table["hide"].to_numpy())
    x_hidden = np.where(hide[:, None, :], 0.0, x)
    hidden = predict_fn(x_hidden, np.where(hide, 0, observed))
    keep = [np.asarray(p) for p in table["keep_patches"]]
    found_full = np.array([full[i, k].max() > threshold for i, k in enumerate(keep)])
    found_hidden = np.array([hidden[i, k].max() > threshold for i, k in enumerate(keep)])
    out = {"windows": int(len(table)), "kept_found_with_siblings": float(found_full.mean()),
           "kept_found_alone": float(found_hidden.mean()), "bins": []}
    stack = (table["inj_period_d"] < 10) & (table["inj_snr"] < 4) & (table["inj_snr_total"] >= 7)
    out["stack_regime"] = {"windows": int(stack.sum()),
                           "kept_found_with_siblings": float(found_full[stack].mean()) if stack.any() else None,
                           "kept_found_alone": float(found_hidden[stack].mean()) if stack.any() else None}
    for lo, hi in zip(config.PER_TRANSIT_SNR_BINS[:-1], config.PER_TRANSIT_SNR_BINS[1:]):
        m = ((table["inj_snr"] >= lo) & (table["inj_snr"] < hi)).to_numpy()
        out["bins"].append({"per_transit_lo": float(lo), "per_transit_hi": float(hi), "n": int(m.sum()),
                            "with_siblings": float(found_full[m].mean()) if m.any() else None,
                            "alone": float(found_hidden[m].mean()) if m.any() else None})
    return out
