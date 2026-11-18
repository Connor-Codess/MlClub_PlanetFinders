"""Choose the lc-finetune train rows of every arm.

What it does
    Reads the per-row metadata of every lc-finetune train shard, works out each arm's
    quota per category from config.ARMS (real_share, moves, filters, sources and host_sources),
    and takes that many rows. Within a category,
    rows are taken in one fixed random order, skipping rows that fail the arm's filters,
    so arms share as many rows as they can and the scaling arms are nested
    (rows_20k is part of rows_40k, which is part of baseline, which is part of rows_160k).

Outputs (in --out)
    selection.parquet  arm, category, path, row, sample_id, best_transit_snr
    quotas.json        rows per arm and category, and the base category shares
    arms.json          the arm names, for the workflow's job matrix

How to run
    python -m experiments.finetune_mix.select --out build/mix/select [--data build/mix/finetune] [--scale 0.01]
        [--arms baseline no_real] [--request experiments/finetune_mix/run_request.json]
    Without --data the train shards are downloaded from Hugging Face first.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import download, read_meta

LOG = logging.getLogger("mix.select")

ROW_FILTERS = {
    "best_transit_snr_ge_4": lambda m: m["best_transit_snr"] >= 4.0,
    "inj_snr_ge_3": lambda m: m["inj_snr"] >= 3.0,
    "inj_snr_total_ge_7": lambda m: m["inj_snr_total"] >= 7.0,
    "not_cropped": lambda m: ~m["single_transit_crop"].astype(bool),
    "not_deep": lambda m: ~m["inj_deep"].fillna(False).astype(bool),
}
HOST_CATEGORIES = (*config.INJECTED, "negative_random")


def base_shares(meta: pd.DataFrame) -> dict[str, float]:
    counts = meta["category"].value_counts()
    total = float(counts[list(config.CATEGORIES)].sum())
    return {c: float(counts.get(c, 0)) / total for c in config.CATEGORIES}


def arm_quotas(spec: dict, shares: dict[str, float], scale: float = 1.0) -> dict[str, int]:
    rows = int(round(spec.get("rows", config.ROWS_PER_ARM) * scale))
    quota = {c: rows * shares[c] for c in config.CATEGORIES}
    if "real_share" in spec:
        base = sum(shares[c] for c in config.REAL)
        keep = spec["real_share"] / base if base > 0 else 0.0
        freed = sum(quota[c] for c in config.REAL) * (1.0 - keep)
        for c in config.REAL:
            quota[c] *= keep
        weight = sum(shares[t] for t in config.INJECTED)
        for t in config.INJECTED:
            quota[t] += freed * shares[t] / weight
    for source, targets in spec.get("move", {}).items():
        moved = quota[source]
        quota[source] = 0.0
        weight = sum(shares[t] for t in targets)
        for t in targets:
            quota[t] += moved * shares[t] / weight
    out = {c: int(round(q)) for c, q in quota.items()}
    largest = max(out, key=out.get)
    out[largest] += rows - sum(out.values())
    return out


def select_arm(name: str, spec: dict, meta: pd.DataFrame, order: dict[str, np.ndarray],
               shares: dict[str, float], scale: float = 1.0) -> pd.DataFrame:
    quotas = arm_quotas(spec, shares, scale)
    picked = []
    for category, quota in quotas.items():
        if quota == 0:
            continue
        rows = meta.loc[order[category]]
        keep = np.ones(len(rows), dtype=bool)
        if "sources" in spec:
            keep &= rows["source"].isin(spec["sources"]).to_numpy()
        if "host_sources" in spec and category in HOST_CATEGORIES:
            keep &= rows["source"].isin(spec["host_sources"]).to_numpy()
        test = spec.get("filters", {}).get(category)
        if test is not None:
            keep &= ROW_FILTERS[test](rows).to_numpy()
        rows = rows[keep]
        if len(rows) < quota:
            raise RuntimeError(f"arm {name}: {category} needs {quota} rows, only {len(rows)} pass")
        picked.append(rows.iloc[:quota])
    out = pd.concat(picked)
    return out.assign(arm=name)[["arm", "category", "path", "row", "sample_id", "best_transit_snr"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("build/mix/finetune"))
    parser.add_argument("--no-download", action="store_true", help="use shards already in --data")
    parser.add_argument("--scale", type=float, default=1.0, help="multiply every arm's row count (smoke tests)")
    parser.add_argument("--arms", nargs="+", help="only these arms (default: every arm in config.ARMS)")
    parser.add_argument("--request", type=Path, help="run_request.json; its 'arms' list, if set, picks the arms")
    args = parser.parse_args()
    if args.request and not args.arms:
        args.arms = json.loads(args.request.read_text()).get("arms") or None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    files = (sorted(args.data.glob("data/train/*.parquet")) if args.no_download
             else download(config.FINETUNE_REPO, "train", args.data))
    meta = read_meta(files, args.data)
    LOG.info("lc-finetune train: %d rows in %d shards", len(meta), len(files))
    shares = base_shares(meta)
    rng = np.random.default_rng(config.SELECTION_SEED)
    order = {c: rng.permutation(np.flatnonzero(meta["category"].to_numpy() == c)) for c in config.CATEGORIES}

    arms = args.arms or list(config.ARMS)
    selections, quotas = [], {}
    for name in arms:
        spec = config.ARMS[name]
        chosen = select_arm(name, spec, meta, order, shares, args.scale)
        selections.append(chosen)
        quotas[name] = chosen["category"].value_counts().reindex(config.CATEGORIES, fill_value=0).astype(int).to_dict()
        LOG.info("%-18s %s", name, quotas[name])

    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat(selections, ignore_index=True).to_parquet(args.out / "selection.parquet", index=False)
    (args.out / "quotas.json").write_text(json.dumps({"base_shares": shares, "arms": quotas}, indent=2))
    (args.out / "arms.json").write_text(json.dumps(arms))
    if "GITHUB_OUTPUT" in os.environ:
        specs = {**config.WAVE1_ARMS, **config.WAVE2_ARMS}
        scratch = [{"arm": a, "seed": s} for a in arms if not specs[a].get("init") for s in config.SEEDS]
        pretrained = [{"arm": a, "seed": s} for a in arms if specs[a].get("init") for s in config.SEEDS]
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"arms={json.dumps(arms)}\n")
            fh.write(f"jobs_scratch={json.dumps(scratch)}\n")
            fh.write(f"jobs_pretrained={json.dumps(pretrained)}\n")
            fh.write(f"has_scratch={'true' if scratch else 'false'}\n")
            fh.write(f"has_pretrained={'true' if pretrained else 'false'}\n")


if __name__ == "__main__":
    main()
