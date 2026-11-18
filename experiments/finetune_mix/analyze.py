"""Extra analysis of finished runs from their per-window scores.

What it does
    aggregate.py reports the numbers each run wrote. This script recomputes recall
    for subsets that answer specific questions, straight from every run's scores.npz
    and the packed lc-eval val metadata:

    stack_regime_inj          short-period injections (P < 10 d) with per-transit SNR < 4
                              and combined SNR >= 7: visible only when transits are combined
    single_transit_inj        injected windows with one visible transit
    real_short_faint          real short-period windows with no transit at SNR 4 but stacked SNR >= 4
    real_long_single          real long-period windows with one visible transit
    real_long_deep            real long-period windows with stacked SNR >= 25
    neg1_*                    recall and EB false alarms with the threshold set so that 1% of
                              random and EB windows together raise an alarm (a matched comparison
                              for arms that fire more on EBs)
    batman_matched_to_real_long  recall of single-transit batman injections with P >= 20 d,
                              weighted to the SNR distribution of the real long-period planets;
                              next to real_long_single_for_matching it tests whether injections
                              are as easy as real planets of the same SNR
    hiding_test               mean over seeds of train.py's transit-hiding test, per arm

    Each value is the mean and standard deviation over an arm's seeds.

Outputs (in --out)
    extra.json   per arm and subset: [mean, sd]; subset sizes; runs per arm

How to run
    python -m experiments.finetune_mix.analyze --results build/mix/results \
        --eval-meta build/mix/eval/meta.parquet --out build/mix/summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def subsets(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    cat = meta["category"].to_numpy()
    inj = np.isin(cat, config.INJECTED)
    period = meta["inj_period_d"].to_numpy()
    per_transit = meta["inj_snr"].to_numpy()
    combined = meta["inj_snr_total"].to_numpy()
    visible = meta["n_visible_transits"].to_numpy()
    best = meta["best_transit_snr"].to_numpy()
    stacked = meta["stacked_snr"].fillna(0).to_numpy()
    short, long_ = cat == "real_planet_short", cat == "real_planet_long"
    return {
        "stack_regime_inj": inj & (period < 10) & (per_transit < 4) & (combined >= 7),
        "single_transit_inj": inj & (visible == 1),
        "real_short_faint": short & (best < 4) & (stacked >= 4),
        "real_short_bright": short & (best >= 4),
        "real_long_single": long_ & (visible == 1),
        "real_long_deep": long_ & (stacked >= 25),
    }


def run_values(scores: dict, meta: pd.DataFrame, masks: dict[str, np.ndarray]) -> dict[str, float]:
    cat = meta["category"].to_numpy()
    any_score, on_target = scores["any_score"], scores["on_target"]
    has = np.isfinite(on_target)
    random, eb = cat == "negative_random", cat == "negative_eb"
    threshold = np.quantile(any_score[random], 1.0 - config.FALSE_ALARM_RATES[0])
    matched = np.quantile(any_score[random | eb], 1.0 - config.FALSE_ALARM_RATES[0])

    def recall(mask: np.ndarray, t: float) -> float:
        mask = mask & has
        return float((on_target[mask] > t).mean()) if mask.any() else float("nan")

    out = {name: recall(mask, threshold) for name, mask in masks.items()}
    # batman injections matched to real long-period planets: P >= 20 d, one visible transit, and
    # recall per combined-SNR bin weighted by the real planets' stacked-SNR distribution
    real_long = (cat == "real_planet_long") & has & (meta["n_visible_transits"].to_numpy() == 1)
    bat = (cat == "injected_batman") & has & (meta["inj_period_d"].to_numpy() >= 20) \
        & (meta["n_visible_transits"].to_numpy() == 1)
    edges = np.array(config.COMBINED_SNR_BINS)
    real_bins = np.digitize(meta["stacked_snr"].fillna(0).to_numpy()[real_long], edges) - 1
    bat_bins = np.digitize(meta["inj_snr_total"].to_numpy()[bat], edges) - 1
    found_bat = on_target[bat] > threshold
    weights, recalls = [], []
    for b in range(len(edges) - 1):
        if (bat_bins == b).sum() >= 20:
            weights.append((real_bins == b).sum())
            recalls.append(found_bat[bat_bins == b].mean())
    out["batman_matched_to_real_long"] = (float(np.average(recalls, weights=weights))
                                          if weights and sum(weights) > 0 else float("nan"))
    out["real_long_single_for_matching"] = recall(real_long, threshold)
    out.update({
        "neg1_injected": recall(np.isin(cat, config.INJECTED), matched),
        "neg1_real_short": recall(cat == "real_planet_short", matched),
        "neg1_real_long": recall(cat == "real_planet_long", matched),
        "neg1_eb_false_alarm": float((any_score[eb] > matched).mean()),
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--eval-meta", type=Path, required=True, help="meta.parquet of the packed lc-eval val")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = pd.read_parquet(args.eval_meta)
    masks = subsets(meta)
    rows, hiding = [], []
    for path in sorted(args.results.glob("*/scores.npz")):
        metrics = json.loads((path.parent / "metrics.json").read_text())
        run = metrics["run"]
        rows.append({"arm": run["arm"], "seed": run["seed"], **run_values(np.load(path), meta, masks)})
        test = metrics.get("hiding_test") or {}
        if test.get("windows"):
            hiding.append({"arm": run["arm"], "seed": run["seed"],
                           "with_siblings": test["kept_found_with_siblings"], "alone": test["kept_found_alone"],
                           "stack_with_siblings": test["stack_regime"]["kept_found_with_siblings"],
                           "stack_alone": test["stack_regime"]["kept_found_alone"], "windows": test["windows"]})
    frame = pd.DataFrame(rows)
    columns = [c for c in frame.columns if c not in ("arm", "seed")]
    stats = frame.groupby("arm")[columns].agg(["mean", "std"])
    arms = {arm: {c: [round(float(stats.loc[arm, (c, "mean")]), 4),
                      None if np.isnan(stats.loc[arm, (c, "std")]) else round(float(stats.loc[arm, (c, "std")]), 4)]
                  for c in columns} for arm in stats.index}
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"arms": arms, "sizes": {k: int(v.sum()) for k, v in masks.items()},
              "runs": frame.groupby("arm").size().to_dict()}
    if hiding:
        h = pd.DataFrame(hiding)
        result["hiding_test"] = {arm: {c: round(float(g[c].mean()), 4) for c in
                                       ("with_siblings", "alone", "stack_with_siblings", "stack_alone")}
                                 | {"windows": int(g["windows"].iloc[0])} for arm, g in h.groupby("arm")}
    (args.out / "extra.json").write_text(json.dumps(result, indent=1))
    for arm, values in arms.items():
        print(arm.ljust(18), " ".join(f"{c}={v[0]:.3f}" for c, v in values.items()))


if __name__ == "__main__":
    main()
