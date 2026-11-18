"""Stage 4 (one job): shuffle, check, document and upload lc-finetune, lc-eval and lc-benchmarks.

What it does
    1. Reads every rows_<dataset>_<split>_*.parquet written by the materialize jobs,
       measures each row for the checks below, and deals the rows at random into
       shards of up to 10,000 (pass A), then shuffles each shard (pass B), so
       streaming a split in file order is already a random order.
    2. Checks the build (any failure stops the upload):
         row accounting   every recipe became exactly one row
         leakage          no star in both lc-finetune and lc-eval; no benchmark star in
                          lc-finetune
         labels           in every category with transits, the median flux of
                          transit-labeled patches is at least 1 sigma below that of
                          negative patches (0.25 sigma for lc-eval real windows,
                          which have no SNR cut)
         injections       measured / intended per-transit SNR has a median within 0.8-1.3
    3. Writes summary.json and validation_report.md, the dataset cards, and uploads
       each dataset (lc-finetune public, lc-eval and lc-benchmarks private).

How to run
    python -m finetune_dataset.assemble --rows build/finetune/rows --plan build/finetune/plan \
        --benchmarks build/finetune/benchmarks --out build/finetune/final [--upload]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .cards import write_cards
from .schema import SAMPLE_SCHEMA

LOG = logging.getLogger("finetune.assemble")

ROWS_PER_SHARD = 10_000
REPOS = {"finetune": (config.FINETUNE_REPO, False), "eval": (config.EVAL_REPO, True),
         "benchmark": (config.BENCHMARK_REPO, True)}


def row_measurements(table: pa.Table) -> pd.DataFrame:
    """Per row: mean flux of transit-labeled and negative patches, and the visible-transit SNR."""
    flux = table.column("flux_z").to_pylist()
    flags = table.column("flags").to_pylist()
    labels = table.column("patch_label").to_pylist()
    events = table.column("events").to_pylist()
    out = []
    for f, g, lab, ev in zip(flux, flags, labels, events):
        z = np.frombuffer(f, "<f2").astype(np.float32).reshape(config.N_PATCHES, config.PATCH_SLOTS)
        o = ((np.frombuffer(g, np.uint8) & 1) > 0).reshape(config.N_PATCHES, config.PATCH_SLOTS)
        lab = np.frombuffer(lab, np.uint8)
        counts = o.sum(axis=1)
        means = np.where(counts > 0, (z * o).sum(axis=1) / np.maximum(counts, 1), np.nan)
        pos = means[(lab == config.LABEL_TRANSIT) & (counts > 0)]
        neg = means[(lab == config.LABEL_NEGATIVE) & (counts > 0)]
        visible = [e["snr"] for e in ev if e["is_planet"] and e["visible"]]
        out.append((float(np.nanmean(pos)) if pos.size else np.nan, float(np.nanmedian(neg)) if neg.size else np.nan,
                    max(visible) if visible else np.nan))
    frame = pd.DataFrame(out, columns=["transit_patch_flux", "negative_patch_flux", "best_visible_snr"])
    for name in ("sample_id", "dataset", "split", "category", "star_key", "source", "inj_model", "inj_snr",
                 "inj_snr_target", "inj_snr_total", "stacked_snr", "n_visible_transits", "n_transit_patches",
                 "coverage", "benchmark_tce_id", "inj_deep", "inj_ecc", "inj_ld_source", "inj_period_d"):
        frame[name] = table.column(name).to_pandas()
    return frame


def shuffle_split(files: list[Path], out_dir: Path, tmp_dir: Path, rng: np.random.Generator) -> tuple[pd.DataFrame, int]:
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    n_shards = max(1, -(-total // ROWS_PER_SHARD))
    measured = []
    for f in files:
        table = pq.read_table(f, schema=SAMPLE_SCHEMA)
        measured.append(row_measurements(table))
        shard_of = rng.integers(0, n_shards, table.num_rows)
        for shard in np.unique(shard_of):
            part = tmp_dir / f"{shard:05d}"
            part.mkdir(parents=True, exist_ok=True)
            pq.write_table(table.filter(pa.array(shard_of == shard)), part / f.name, compression="zstd")
        f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    for shard in range(n_shards):
        part = tmp_dir / f"{shard:05d}"
        if not part.exists():
            continue
        table = pa.concat_tables([pq.read_table(p, schema=SAMPLE_SCHEMA) for p in sorted(part.glob("*.parquet"))])
        table = table.take(pa.array(rng.permutation(table.num_rows)))
        pq.write_table(table, out_dir / f"part-{shard:05d}-of-{n_shards:05d}.parquet",
                       compression="zstd", row_group_size=500)
        shutil.rmtree(part)
    return pd.concat(measured, ignore_index=True), total


def checks(measured: pd.DataFrame, plan: dict) -> tuple[list[dict], dict]:
    results = []
    expected = sum(plan["counts"].values())
    results.append({"gate": "row accounting", "ok": len(measured) == expected,
                    "detail": f"{len(measured):,} rows for {expected:,} recipes"})

    stars = {d: set(measured.loc[measured["dataset"] == d, "star_key"]) for d in ("finetune", "eval", "benchmark")}
    shared = stars["finetune"] & stars["eval"]
    bench = stars["finetune"] & plan.get("benchmark_stars", set())
    results.append({"gate": "leakage", "ok": not shared and not bench,
                    "detail": f"{len(shared)} stars in both lc-finetune and lc-eval; "
                              f"{len(bench)} benchmark stars in lc-finetune"})

    label_rows = []
    ok = True
    for (dataset, category), group in measured[measured["dataset"] != "benchmark"].groupby(["dataset", "category"]):
        pos = group["transit_patch_flux"].dropna()
        if pos.empty:
            continue
        gap = float(group["negative_patch_flux"].median() - pos.median())
        # lc-eval real windows have no SNR cut, and v2 lc-finetune windows and injections include faint
        # transits that only add up over the window, so those groups only need to dip on average
        faint = category.startswith("real_planet") or category.startswith("injected")
        needed = 0.25 if dataset == "eval" and category.startswith("real_planet") else 0.5 if faint else 1.0
        ok &= gap >= needed
        label_rows.append(f"{dataset}/{category}: {gap:.2f} (needs {needed:g})")
    results.append({"gate": "labels", "ok": ok,
                    "detail": "negative minus transit patch flux, median: " + "; ".join(label_rows)})

    inj = measured[measured["inj_model"].notna() & (measured["inj_snr_total"] > 0)]
    ratio = (inj["stacked_snr"] / inj["inj_snr_total"]).replace([np.inf, -np.inf], np.nan).dropna()
    median_ratio = float(ratio.median()) if len(ratio) else float("nan")
    target = (inj["inj_snr_total"] / inj["inj_snr_target"]).replace([np.inf, -np.inf], np.nan).dropna()
    median_target = float(target.median()) if len(target) else float("nan")
    results.append({"gate": "injections", "ok": bool(0.8 <= median_ratio <= 1.3 and 0.8 <= median_target <= 1.25),
                    "detail": f"measured stacked / expected combined SNR, median {median_ratio:.3f}; expected / "
                              f"target combined SNR, median {median_target:.3f}; over {len(ratio):,} rows"})

    stats = {
        "rows": measured.groupby(["dataset", "split", "category"]).size().rename("rows").reset_index()
        .to_dict("records"),
        "stars": {d: len(s) for d, s in stars.items()},
        "sources": measured.groupby(["dataset", "split", "source"]).size().rename("rows").reset_index()
        .to_dict("records"),
        "injected_snr_quantiles": {m: g["inj_snr"].quantile([0.1, 0.5, 0.9]).round(2).tolist()
                                   for m, g in inj.groupby("inj_model")},
        "injected_combined_snr_quantiles": {m: g["inj_snr_total"].quantile([0.1, 0.5, 0.9]).round(2).tolist()
                                            for m, g in inj.groupby("inj_model")},
        # v3 physics: achieved / target combined SNR, deep share, eccentric share and limb-darkening source
        "injection_physics": {
            f"{d}/{m}/{'deep' if bool(deep) else 'main'}": {
                "rows": int(len(g)),
                "achieved_over_target_median": round(float((g["inj_snr_total"] / g["inj_snr_target"]).median()), 3),
                "eccentric_share": round(float((g["inj_ecc"].fillna(0) > 0).mean()), 3),
                "ld_sources": g["inj_ld_source"].fillna("v2").value_counts().to_dict(),
            }
            for (d, m, deep), g in inj.assign(inj_deep=inj["inj_deep"].fillna(False))
            .groupby(["dataset", "inj_model", "inj_deep"])},
        # fingerprint of every lc-eval row's identity and injection, to show v3 left lc-eval unchanged
        "eval_fingerprint": hashlib.sha256(
            measured[measured["dataset"] == "eval"].sort_values("sample_id")[
                ["sample_id", "inj_snr_total", "inj_period_d", "stacked_snr"]].round(5).to_csv(index=False)
            .encode()).hexdigest(),
        "real_stacked_snr_quantiles": {
            f"{d}/{s}/{c}": g["stacked_snr"].quantile([0.1, 0.5, 0.9]).round(2).tolist()
            for (d, s, c), g in measured[measured["category"].str.startswith("real_planet")]
            .groupby(["dataset", "split", "category"])},
        "real_transit_snr_median": measured[measured["category"].str.startswith("real_planet")]
        .groupby(["dataset", "split", "category"])["best_visible_snr"].median().round(2)
        .rename("median").reset_index().to_dict("records"),
    }
    return results, stats


def benchmark_coverage(measured: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    bench = measured[measured["dataset"] == "benchmark"]
    covered = set(zip(bench["split"], bench["benchmark_tce_id"]))
    targets = targets.assign(has_windows=[(b, t) in covered for b, t in zip(targets["benchmark"], targets["tce_id"])])
    return (targets.groupby(["benchmark", "label"])
            .agg(signals=("tce_id", "size"), with_windows=("has_windows", "sum")).reset_index())


def upload(dataset: str, folder: Path) -> None:
    from huggingface_hub import HfApi

    repo, private = REPOS[dataset]
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    if any(f.startswith("data/") for f in api.list_repo_files(repo, repo_type="dataset")):
        # shard names depend on the row count, so old shards would otherwise linger
        api.delete_folder("data", repo_id=repo, repo_type="dataset", commit_message="Remove previous build")
    for attempt in range(6):
        try:
            api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(folder),
                              commit_message=f"{dataset} build {config.BUILD_VERSION}")
            break
        except Exception as error:  # rate limits and dropped connections: retry with backoff
            status = getattr(getattr(error, "response", None), "status_code", None)
            if attempt == 5 or (status is not None and 400 <= status < 500 and status != 429):
                raise
            LOG.warning("Upload of %s failed (%s); retrying", repo, error)
            time.sleep(120 * (attempt + 1))
    LOG.info("Uploaded %s to %s", folder, repo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(config.SEED + 1)

    plan = json.loads((args.plan / "plan.json").read_text())
    targets = pd.read_parquet(args.benchmarks / "benchmark_targets.parquet")
    provenance = json.loads((args.benchmarks / "benchmark_provenance.json").read_text())
    plan["benchmark_stars"] = set(targets["star_key"])

    files: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for f in sorted(args.rows.rglob("rows_*.parquet")):
        dataset, split = f.name.split("_")[1], f.name.split("_")[2]
        if dataset == "benchmark":
            split = "_".join(f.name.split("_")[2:-2])
        files[(dataset, split)].append(f)

    measured_parts = []
    for (dataset, split), group in sorted(files.items()):
        folder = args.out / dataset / "data" / split
        part, n = shuffle_split(group, folder, args.out / "tmp" / dataset / split, rng)
        measured_parts.append(part)
        LOG.info("%s/%s: %d rows in %d files", dataset, split, n, len(list(folder.glob("*.parquet"))))
    measured = pd.concat(measured_parts, ignore_index=True)
    shutil.rmtree(args.out / "tmp", ignore_errors=True)
    results, stats = checks(measured, plan)
    coverage = benchmark_coverage(measured, targets)

    templates = pd.read_parquet(args.plan / "templates.parquet")
    summary = {"build_version": config.BUILD_VERSION, "gates": results, "plan_counts": plan["counts"],
               "templates": plan["templates"], "benchmark_coverage": coverage.to_dict("records"), **stats}
    for dataset in REPOS:
        root = args.out / dataset
        if not root.exists():
            continue
        (root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        if dataset != "benchmark":
            pool = config.TEMPLATE_POOLS["finetune" if dataset == "finetune" else "eval"]
            templates[templates["pool"] == pool].to_parquet(root / "templates.parquet", index=False)
        else:
            targets.to_parquet(root / "benchmark_targets.parquet", index=False)
            (root / "benchmark_provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    write_cards(args.out, summary, measured, coverage, provenance)

    for r in results:
        level = "notice" if r["ok"] else "error"
        print(f"::{level} title=check {r['gate']}::{'PASS' if r['ok'] else 'FAIL'} {r['detail']}")
    if not all(r["ok"] for r in results):
        raise SystemExit("a build check failed; nothing was uploaded")
    if args.upload:
        for dataset in REPOS:
            if (args.out / dataset).exists():
                upload(dataset, args.out / dataset)


if __name__ == "__main__":
    main()
