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
    rows = []
    for path in sorted(args.results.glob("*/scores.npz")):
        run = json.loads((path.parent / "metrics.json").read_text())["run"]
        rows.append({"arm": run["arm"], "seed": run["seed"], **run_values(np.load(path), meta, masks)})
    frame = pd.DataFrame(rows)
    columns = [c for c in frame.columns if c not in ("arm", "seed")]
    stats = frame.groupby("arm")[columns].agg(["mean", "std"])
    arms = {arm: {c: [round(float(stats.loc[arm, (c, "mean")]), 4),
                      None if np.isnan(stats.loc[arm, (c, "std")]) else round(float(stats.loc[arm, (c, "std")]), 4)]
                  for c in columns} for arm in stats.index}
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"arms": arms, "sizes": {k: int(v.sum()) for k, v in masks.items()},
              "runs": frame.groupby("arm").size().to_dict()}
    (args.out / "extra.json").write_text(json.dumps(result, indent=1))
    for arm, values in arms.items():
        print(arm.ljust(18), " ".join(f"{c}={v[0]:.3f}" for c, v in values.items()))


if __name__ == "__main__":
    main()
