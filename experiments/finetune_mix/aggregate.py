"""Combine every run's metrics into one summary of the data mix experiments.

What it does
    Reads results/<arm>-s<seed>/metrics.json for every run (and results/box_filter), and
    for each arm takes the mean and standard deviation over seeds of every headline number.
    Each arm is compared with the reference arm (config.REFERENCE_ARM): difference of the means, its standard error
    (sqrt(sd_arm^2/n_arm + sd_base^2/n_base)) and whether the difference exceeds two
    standard errors.

Outputs (in --out)
    summary.json   per arm: settings, runs, mean/sd per metric, differences with the reference arm,
                   and the mean recall per bin (combined SNR, per-transit SNR, period, real)
    summary.md     the same as tables, for reading
    runs.csv       one row per run with the headline numbers

How to run
    python -m experiments.finetune_mix.aggregate --results build/mix/results --out build/mix/summary
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

HEADLINE = ["injected_recall", "faint_recall", "strong_recall", "batman_recall", "transplant_recall",
            "real_short_recall", "real_long_recall", "eb_false_alarm"]


def load_runs(results: Path) -> tuple[pd.DataFrame, dict]:
    rows, bins = [], {}
    for path in sorted(results.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        run = m["run"]
        for rate, point in m["operating_points"].items():
            row = {"arm": run["arm"], "seed": run["seed"], "rate": rate,
                   "patch_average_precision": m["patch_average_precision"],
                   "train_minutes": run.get("train_minutes"), "steps": run.get("steps"),
                   "stopped_early": run.get("stopped_early", False), "rows": run.get("rows")}
            row.update({k: point.get(k) for k in HEADLINE})
            rows.append(row)
            for name, table in point["bins"].items():
                for b in table:
                    bins.setdefault((run["arm"], rate, name, b["lo"], b["hi"]), []).append(
                        (b["recall"], b["n"]))
    return pd.DataFrame(rows), bins


def compare(frame: pd.DataFrame) -> dict:
    out = {}
    metrics = HEADLINE + ["patch_average_precision"]
    for rate, group in frame.groupby("rate"):
        stats = group.groupby("arm")[metrics].agg(["mean", "std", "count"])
        base = stats.loc[config.REFERENCE_ARM] if config.REFERENCE_ARM in stats.index else None
        per_arm = {}
        for arm, row in stats.iterrows():
            entry = {}
            for metric in metrics:
                mean, sd, n = row[(metric, "mean")], row[(metric, "std")], row[(metric, "count")]
                item = {"mean": _num(mean), "sd": _num(sd), "n": int(n)}
                if base is not None and arm not in (config.REFERENCE_ARM, "box_filter", "classical"):
                    bm, bsd, bn = base[(metric, "mean")], base[(metric, "std")], base[(metric, "count")]
                    diff = mean - bm
                    se = math.sqrt((sd if np.isfinite(sd) else 0) ** 2 / max(n, 1)
                                   + (bsd if np.isfinite(bsd) else 0) ** 2 / max(bn, 1))
                    item.update({"diff": _num(diff), "se": _num(se),
                                 "beyond_noise": bool(se > 0 and abs(diff) > 2 * se)})
                entry[metric] = item
            per_arm[arm] = entry
        out[rate] = per_arm
    return out


def bin_table(bins: dict) -> dict:
    out = {}
    for (arm, rate, name, lo, hi), values in bins.items():
        recalls = [r for r, _ in values if r is not None]
        out.setdefault(rate, {}).setdefault(name, {}).setdefault(arm, []).append({
            "lo": lo, "hi": hi, "n": int(values[0][1]),
            "recall_mean": _num(np.mean(recalls)) if recalls else None,
            "recall_sd": _num(np.std(recalls, ddof=1)) if len(recalls) > 1 else None,
        })
    for rate in out.values():
        for table in rate.values():
            for arm in table:
                table[arm].sort(key=lambda b: b["lo"])
    return out


def _num(x) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def markdown(comparison: dict, bins: dict) -> str:
    lines = ["# Fine-tuning data mix experiments: summary", ""]
    for rate in [str(r) for r in config.FALSE_ALARM_RATES if str(r) in comparison]:
        arms = comparison[rate]
        lines += [f"## Operating point: {float(rate):.1%} of random windows raise a false alarm", ""]
        header = "| Arm | " + " | ".join(HEADLINE + ["patch AP"]) + " |"
        lines += [header, "|---|" + "---:|" * (len(HEADLINE) + 1)]
        order = ["box_filter", "classical", config.REFERENCE_ARM] + [a for a in config.ARMS if a != config.REFERENCE_ARM]
        for arm in [a for a in order if a in arms]:
            cells = []
            for metric in HEADLINE + ["patch_average_precision"]:
                item = arms[arm][metric]
                if item["mean"] is None:
                    cells.append("-")
                    continue
                text = f"{item['mean']:.3f}"
                if item.get("sd") is not None:
                    text += f" ± {item['sd']:.3f}"
                if item.get("beyond_noise"):
                    text += f" ({item['diff']:+.3f})"
                cells.append(text)
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        lines += ["", "Values are mean ± sd over seeds; a signed number in parentheses is a difference from",
                  f"the reference arm ({config.REFERENCE_ARM}) larger than two standard errors.", ""]
        table = bins.get(rate, {}).get("combined_snr", {})
        if table:
            edges = [f"{b['lo']:g}-{b['hi']:g}" if b["hi"] < 1e8 else f">={b['lo']:g}"
                     for b in next(iter(table.values()))]
            lines += ["Injected recall by combined SNR:", "", "| Arm | " + " | ".join(edges) + " |",
                      "|---|" + "---:|" * len(edges)]
            for arm in [a for a in order if a in table]:
                cells = ["-" if b["recall_mean"] is None else f"{b['recall_mean']:.3f}" for b in table[arm]]
                lines.append(f"| {arm} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame, raw_bins = load_runs(args.results)
    if frame.empty:
        raise SystemExit(f"no metrics.json under {args.results}")
    comparison = compare(frame)
    bins = bin_table(raw_bins)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "runs.csv", index=False)
    summary = {"arms": {a: config.ARMS.get(a, {}).get("question", "classical baseline") for a in frame["arm"].unique()},
               "runs": int(frame[["arm", "seed"]].drop_duplicates().shape[0]),
               "comparison": comparison, "bins": bins}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "summary.md").write_text(markdown(comparison, bins))
    print((args.out / "summary.md").read_text())


if __name__ == "__main__":
    main()
