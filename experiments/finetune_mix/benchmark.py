"""Score trained detectors on the published papers' test sets in lc-benchmarks.

What it does
    Runs one or more trained Detector checkpoints (the seeds of one arm) over every window
    of nyu-planet-finders/lc-benchmarks and turns patch probabilities into one score per
    published signal, following the protocol in the lc-benchmarks card:

        signal score = mean, over the signal's visible predicted transits in its windows,
                       of the model's transit probability at the patch holding the mid-time

    Then it computes, per benchmark and per seed, the metrics each paper reports, on the
    signals that have windows:

        astronet_dr24       Shallue & Vanderburg 2018: PC against AFP and NTP. ROC AUC, average
                            precision, and AUC of PC against AFP and against NTP separately
        astronet_triage     Yu et al. 2019: triage (PC and EB against J) and vetting (PC against
                            EB and J). ROC AUC and average precision
        astronet_triage_v2  Tey et al. 2023: E against B, J, N and S (signals without a consensus
                            label are left out). Average precision (their AUC-PR) and ROC AUC
        salinas2025         Salinas et al. 2025: every signal is a published candidate, so the share
                            the model detects, and ROC AUCs against TESS non-transit signals and
                            lc-eval random and EB windows (see detection_aucs)

    "Detected" uses the operating point of the experiments: the threshold at which 1% of
    lc-eval val random windows raise an alarm, taken from each run's own scores.npz, and a
    signal counts as detected when any of its visible transits has a patch within one of its
    mid-time patch above that threshold. Detection rates at that threshold are reported for
    every label, which is the number closest to how the detector will be used.

Why
    These papers classify a signal given its period and epoch, mostly with phase-folded views
    of the whole light curve; our model scores raw 27-day windows with no ephemeris, and it was
    not trained to reject astrophysical false positives other than eclipsing binaries. The
    comparison is close, not identical, so every number comes with the covered fraction of the
    paper's test set.

Outputs (in --out)
    benchmark.json   per benchmark: coverage, per-seed and mean metrics, detection rates by label
    benchmark.md     the comparison table with the papers' published numbers
    signals.parquet  one row per covered signal and seed: score, peak, detected, label
    windows.parquet, patches-<run>.npz  every benchmark window's split, label and signal id, and its
                     patch probabilities from each seed, for analyses beyond this script

How to run
    python -m experiments.finetune_mix.benchmark --runs build/runs --data build/bench --out build/benchmark
    --runs holds one folder per seed with model.pt and scores.npz (the result-<arm>-s<seed>
    artifacts of finetune-mix-experiments.yml); --eval-meta is the packed lc-eval val meta.parquet
    used to set the threshold. Needs HF_TOKEN with read access to the private lc-benchmarks repo.
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
import torch

from . import config
from .data import _decode
from .model import Detector
from .train import predict_arrays

LOG = logging.getLogger("benchmark")

REPO = "nyu-planet-finders/lc-benchmarks"
BENCHMARKS = ("astronet_dr24", "astronet_triage", "astronet_triage_v2", "salinas2025")
FALSE_ALARM_RATE = 0.01

# What each paper reports on its own test set (copied from docs/finetune_dataset_build_log.md, Stage 0).
PUBLISHED = {
    "astronet_dr24": ("Shallue & Vanderburg 2018 (AstroNet)", "Kepler DR24, PC vs AFP+NTP",
                      "AUC 0.988; accuracy 0.960 at threshold 0.5 (1,523 test TCEs)"),
    "astronet_triage": ("Yu et al. 2019 (Astronet-Triage)", "TESS QLP sectors 1-5",
                        "Triage (PC+EB vs J): AUC 0.992, AP 0.970. Vetting (PC vs rest): AUC 0.984, AP 0.693"),
    "astronet_triage_v2": ("Tey et al. 2023 (Astronet-Triage-v2)", "TESS QLP cycles 1-3, E vs rest",
                           "AUC-PR 0.965; recall 97.2% at precision 75.7%"),
    "salinas2025": ("Salinas et al. 2025", "TESS-SPOC FFI sectors 1-26",
                    "AUC 0.88, F1 0.82 on an unreleased 10% split; 213 published candidates"),
}


def roc_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return None
    ranks = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y: np.ndarray, s: np.ndarray) -> float | None:
    y = np.asarray(y, bool)
    if not y.any():
        return None
    order = np.argsort(-s, kind="stable")
    hits = y[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def load_windows(files: list[Path]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Arrays of every benchmark window and a table of its benchmark transits (row, mid slot)."""
    xs, obs, metas, transits = [], [], [], []
    start = 0
    for path in files:
        table = pq.read_table(path, columns=list(config.INPUT_CHANNELS) + ["flags", "split", "category",
                                                                           "benchmark_tce_id", "events"])
        n = table.num_rows
        x = np.stack([_decode(table.column(c).to_pylist(), "<f2", config.WINDOW_SLOTS)
                      for c in config.INPUT_CHANNELS], axis=1)
        flags = _decode(table.column("flags").to_pylist(), np.uint8, config.WINDOW_SLOTS)
        xs.append(x)
        obs.append(((flags & 1) > 0).astype(np.uint8))
        meta = table.select(["split", "category", "benchmark_tce_id"]).to_pandas()
        meta.index = np.arange(start, start + n)
        metas.append(meta)
        for i, (ref, events) in enumerate(zip(meta["benchmark_tce_id"], table.column("events").to_pylist())):
            for e in events or []:
                if e.get("kind") != "benchmark" or e.get("ref") not in (None, ref):
                    continue
                transits.append({"row": start + i, "center_slot": float(e["center_slot"]),
                                 "visible": bool(e.get("visible", (e.get("observed_fraction") or 0) >= 0.5))})
        start += n
    return (np.concatenate(xs), np.concatenate(obs), pd.concat(metas),
            pd.DataFrame(transits, columns=["row", "center_slot", "visible"]))


def signal_scores(patch: np.ndarray, observed: np.ndarray, meta: pd.DataFrame, transits: pd.DataFrame,
                  threshold: float) -> pd.DataFrame:
    """One row per signal with at least one visible transit: protocol score and detection."""
    rows = transits["row"].to_numpy(dtype=int)
    slot = np.clip(transits["center_slot"].to_numpy().astype(int), 0, config.WINDOW_SLOTS - 1)
    visible = transits["visible"].to_numpy(dtype=bool) & (observed[rows, slot] > 0)
    rows, mid = rows[visible], slot[visible] // config.PATCH_SLOTS
    near = np.stack([patch[rows, np.clip(mid + k, 0, config.N_PATCHES - 1)] for k in (-1, 0, 1)], axis=1)
    frame = pd.DataFrame({"row": rows, "prob": patch[rows, mid], "near": near.max(axis=1),
                          "hit": near.max(axis=1) > threshold})
    frame = frame.join(meta, on="row")
    return (frame.groupby(["split", "benchmark_tce_id"])
                 .agg(label=("category", "first"), score=("prob", "mean"), peak=("near", "max"),
                      detected=("hit", "any"),
                      transits=("prob", "size"))
                 .reset_index())


def detection_aucs(scores: pd.DataFrame, eval_any: np.ndarray, eval_meta: pd.DataFrame) -> dict:
    """ROC AUC of Salinas et al.'s candidates against negatives, on the detection score.

    A candidate's score is its peak: the highest transit probability within one patch of any of its
    visible transits. Two kinds of negative:
        TESS non-transit signals in lc-benchmarks (Tey et al. J, N and S; Yu et al. J), scored the
        same way at their own catalogued times
        lc-eval val random and EB windows, scored by their highest patch anywhere in the window, the
        quantity the experiments' false-alarm threshold is set on; this is the ROC curve of the
        detection rates reported at 1% false alarms
    """
    sal = scores[scores["split"] == "salinas2025"]
    junk = scores[((scores["split"] == "astronet_triage_v2") & scores["label"].isin(["J", "N", "S"]))
                  | ((scores["split"] == "astronet_triage") & (scores["label"] == "J"))]["peak"].to_numpy()
    cat = eval_meta["category"].to_numpy()
    negatives = {"tess_nontransit_signals": junk, "eval_random_windows": eval_any[cat == "negative_random"],
                 "eval_eb_windows": eval_any[cat == "negative_eb"]}
    out = {}
    for group, pos in (("all", sal), ("single", sal[sal["label"] == "single_candidate"]),
                       ("periodic", sal[sal["label"] == "periodic_candidate"])):
        p = pos["peak"].to_numpy()
        for name, neg in negatives.items():
            y = np.r_[np.ones(len(p), bool), np.zeros(len(neg), bool)]
            out[f"auc_{group}_vs_{name}"] = roc_auc(y, np.r_[p, neg])
    return out


def metrics(name: str, s: pd.DataFrame) -> dict:
    lab, score = s["label"].to_numpy(), s["score"].to_numpy()
    out: dict = {"signals": int(len(s))}
    if name == "astronet_dr24":
        y = lab == "PC"
        out.update(auc=roc_auc(y, score), ap=average_precision(y, score))
        for other in ("AFP", "NTP"):
            keep = np.isin(lab, ["PC", other])
            out[f"auc_pc_vs_{other.lower()}"] = roc_auc(lab[keep] == "PC", score[keep])
    elif name == "astronet_triage":
        triage = np.isin(lab, ["PC", "EB"])
        out.update(triage_auc=roc_auc(triage, score), triage_ap=average_precision(triage, score),
                   vetting_auc=roc_auc(lab == "PC", score), vetting_ap=average_precision(lab == "PC", score))
    elif name == "astronet_triage_v2":
        keep = np.isin(lab, ["E", "B", "J", "N", "S"])
        y = lab[keep] == "E"
        out.update(ap=average_precision(y, score[keep]), auc=roc_auc(y, score[keep]), labelled=int(keep.sum()))
    out["detected_by_label"] = s.groupby("label")["detected"].mean().round(4).to_dict()
    out["count_by_label"] = s.groupby("label").size().to_dict()
    return out


def summarize(per_seed: dict[str, list[dict]]) -> dict:
    out = {}
    for name, runs in per_seed.items():
        keys = [k for k, v in runs[0].items() if isinstance(v, float) or v is None]
        mean = {}
        for k in keys:
            vals = [r[k] for r in runs if r.get(k) is not None]
            mean[k] = None if not vals else [round(float(np.mean(vals)), 4),
                                             round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else None]
        labels = sorted({lab for r in runs for lab in r["detected_by_label"]})
        mean["detected_by_label"] = {lab: round(float(np.mean([r["detected_by_label"].get(lab, np.nan) for r in runs])), 4)
                                     for lab in labels}
        mean["count_by_label"] = runs[0]["count_by_label"]
        mean["signals"] = runs[0]["signals"]
        out[name] = mean
    return out


def _fmt(v, digits=3) -> str:
    if v is None:
        return "-"
    m, sd = v
    return f"{m:.{digits}f}" if sd is None else f"{m:.{digits}f} ± {sd:.{digits}f}"


def table(summary: dict, coverage: dict, arm: str) -> str:
    lines = [f"# lc-benchmarks: {arm} against the published papers", "",
             "| Benchmark | Paper | Their data | Their result | Ours (mean ± sd over seeds) | Test set covered |",
             "|---|---|---|---|---|---|"]
    for name in BENCHMARKS:
        if name not in summary:
            continue
        s, (paper, data, result) = summary[name], PUBLISHED[name]
        if name == "astronet_dr24":
            ours = (f"AUC {_fmt(s['auc'])}; AP {_fmt(s['ap'])}; PC vs NTP AUC {_fmt(s['auc_pc_vs_ntp'])}; "
                    f"PC vs AFP AUC {_fmt(s['auc_pc_vs_afp'])}")
        elif name == "astronet_triage":
            ours = (f"Triage: AUC {_fmt(s['triage_auc'])}, AP {_fmt(s['triage_ap'])}. "
                    f"Vetting: AUC {_fmt(s['vetting_auc'])}, AP {_fmt(s['vetting_ap'])}")
        elif name == "astronet_triage_v2":
            ours = f"AUC-PR {_fmt(s['ap'])}; AUC {_fmt(s['auc'])}"
        else:
            ours = ("Candidates detected: " + ", ".join(f"{k} {v:.1%}" for k, v in s["detected_by_label"].items())
                    + f". AUC against TESS non-transit signals {_fmt(s.get('auc_all_vs_tess_nontransit_signals'))}"
                    + f" (single {_fmt(s.get('auc_single_vs_tess_nontransit_signals'))},"
                    + f" periodic {_fmt(s.get('auc_periodic_vs_tess_nontransit_signals'))});"
                    + f" against lc-eval random windows {_fmt(s.get('auc_all_vs_eval_random_windows'))}")
        c = coverage[name]
        lines.append(f"| `{name}` | {paper} | {data} | {result} | {ours} | {c['covered']:,} of {c['signals']:,} "
                     f"({c['covered'] / c['signals']:.0%}) |")
    lines += ["", f"Detection rate by label at 1% random-window false alarms on lc-eval val:", "",
              "| Benchmark | Label | Signals | Detected |", "|---|---|---:|---:|"]
    for name in BENCHMARKS:
        if name in summary:
            for lab, rate in summary[name]["detected_by_label"].items():
                lines.append(f"| `{name}` | {lab} | {summary[name]['count_by_label'][lab]:,} | {rate:.1%} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--eval-meta", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", default="")
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)

    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=REPO, repo_type="dataset", local_dir=args.data, token=os.environ.get("HF_TOKEN"),
                      allow_patterns=["data/*/*.parquet", "benchmark_targets.parquet"], max_workers=4)
    targets = pd.read_parquet(args.data / "benchmark_targets.parquet")
    files = sorted((args.data / "data").glob("*/*.parquet"))
    x, observed, meta, transits = load_windows(files)
    LOG.info("%d benchmark windows, %d benchmark transits", len(x), len(transits))

    eval_meta = pd.read_parquet(args.eval_meta)
    random = (eval_meta["category"] == "negative_random").to_numpy()
    args.out.mkdir(parents=True, exist_ok=True)
    per_seed: dict[str, list[dict]] = {}
    frames = []
    for run in sorted(p for p in args.runs.iterdir() if (p / "model.pt").exists()):
        model = Detector()
        model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
        threshold = float(np.quantile(np.load(run / "scores.npz")["any_score"][random], 1.0 - FALSE_ALARM_RATE))
        patch = predict_arrays(model, x, observed)
        scores = signal_scores(patch, observed, meta, transits, threshold).assign(run=run.name)
        frames.append(scores)
        np.savez_compressed(args.out / f"patches-{run.name}.npz", patch=patch.astype(np.float16))
        for name, s in scores.groupby("split"):
            m = metrics(name, s)
            if name == "salinas2025":
                m.update(detection_aucs(scores, np.load(run / "scores.npz")["any_score"], eval_meta))
            per_seed.setdefault(name, []).append(m)
        LOG.info("%s: threshold %.4f, %d signals scored", run.name, threshold, len(scores))

    all_scores = pd.concat(frames, ignore_index=True)
    meta.to_parquet(args.out / "windows.parquet")
    coverage = {name: {"signals": int((targets["benchmark"] == name).sum()),
                       "covered": int(all_scores.loc[all_scores["split"] == name, "benchmark_tce_id"].nunique())}
                for name in BENCHMARKS}
    summary = summarize(per_seed)
    args.out.mkdir(parents=True, exist_ok=True)
    all_scores.to_parquet(args.out / "signals.parquet", index=False)
    (args.out / "benchmark.json").write_text(json.dumps({"arm": args.arm, "coverage": coverage, "summary": summary,
                                                         "per_seed": per_seed, "published": PUBLISHED}, indent=1))
    (args.out / "benchmark.md").write_text(table(summary, coverage, args.arm))
    print((args.out / "benchmark.md").read_text())


if __name__ == "__main__":
    main()
