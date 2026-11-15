"""Pass 2 of the lc-pretrain build: window index, star table and default sampling weights.

What it does (build guide Steps 8-9)
    Reads only the metadata columns of every data file written by pass 1 (the
    byte channels are never downloaded; 8 files are read at a time), keeps
    them in Arrow and categorical form so a full build fits in a 16 GB
    runner, then writes:

    index/windows_train.parquet, index/windows_validation.parquet   (lc-pretrain)
    index/windows_heldout.parquet                                    (lc-heldout)
        One row per default window: file, row (position of the segment in that
        file), segment_id, star_key, source, split, start_offset, coverage,
        quality_tier, var_ratio, and the weights w_source, w_star, w_var,
        w_tier and w_default.
    index/stars.parquet
        One row per star_key: split, segments and windows per source.
    index/summary.json
        Counts per source and split, tiers, duplicate FFI/2-min pairs, and the
        variability threshold used for w_var.

Weights (computed separately for each split)
    w_source = target share of the source / windows from that source
    w_star   = min(1, 20 / windows this star has in the split)
    w_var    = 2 for the top 20% of var_ratio within the source, else 1
    w_tier   = 0.5 for quality tier 3, else 1
    w_default = product of the four, rescaled to average 1

Run
    python -m pretrain_dataset.build_index --out build/index                  # reads the Hub
    python -m pretrain_dataset.build_index --local build/pass1 --out build/index
    HF_TOKEN=... python -m pretrain_dataset.build_index --out build/index --upload
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import config

LOG = logging.getLogger("pretrain_dataset")

META_COLUMNS = [
    "segment_id", "star_key", "source", "split", "n_slots", "n_obs", "coverage",
    "sigma_p2p_ppm", "var_ratio", "quality_tier", "window_offsets", "window_coverage",
    "tmag", "kepmag", "dup_group",
]


def list_files(local: Path | None) -> dict[str, list[tuple[str, str]]]:
    """Return {'pretrain': [(path_for_reading, path_in_repo), ...], 'heldout': [...]}."""
    if local is not None:
        out = {}
        for part in ("pretrain", "heldout"):
            base = local / part
            out[part] = [(str(p), str(p.relative_to(base))) for p in sorted(base.glob("data/**/*.parquet"))]
        return out
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    out = {}
    for part, repo in (("pretrain", config.PRETRAIN_REPO), ("heldout", config.HELDOUT_REPO)):
        root = f"datasets/{repo}/"
        paths = sorted(fs.glob(root + "data/**/*.parquet"))
        out[part] = [(p, p[len(root):]) for p in paths]
    return out


SEGMENT_COLUMNS = [c for c in META_COLUMNS if c not in ("window_offsets", "window_coverage")]
WINDOW_COLUMNS = ["file", "row", "segment_id", "star_key", "source", "split", "quality_tier", "var_ratio"]
CATEGORICAL = ("file", "star_key", "source", "split", "dup_group", "part")
READ_BLOCK_BYTES = 256 * 2**10
READ_WORKERS = 8


def _split_file(table: pa.Table, repo_path: str) -> tuple[pa.Table, pa.Table]:
    """One pass-1 file's metadata as (segments without list columns, one row per window)."""
    n = table.num_rows
    lengths = pc.list_value_length(table["window_offsets"]).to_numpy(zero_copy_only=False).astype(np.int64)
    segments = (table.select(SEGMENT_COLUMNS)
                .append_column("file", pa.array([repo_path] * n, pa.string()))
                .append_column("row", pa.array(np.arange(n, dtype=np.int64)))
                .append_column("n_windows", pa.array(lengths.astype(np.int16))))
    parents = np.repeat(np.arange(n), lengths)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]) if n else np.zeros(0, np.int64)
    windows = segments.select(WINDOW_COLUMNS).take(pa.array(parents))
    windows = (windows
               .append_column("window_index", pa.array((np.arange(parents.size) - starts[parents]).astype(np.int16)))
               .append_column("start_offset", pc.cast(pc.list_flatten(table["window_offsets"]), pa.int32()))
               .append_column("coverage", pc.cast(pc.list_flatten(table["window_coverage"]), pa.float32())))
    return segments, windows


def _to_pandas(table: pa.Table) -> pd.DataFrame:
    """Convert with repeated strings as categoricals and unique strings as Arrow strings.

    A full build has about 17 million segments and 23 million windows; with Python
    string objects that would not fit in a 16 GB GitHub runner.
    """
    for name in CATEGORICAL:
        if name in table.column_names:
            index = table.column_names.index(name)
            table = table.set_column(index, name, pc.dictionary_encode(table[name]).combine_chunks())
    frame = table.to_pandas(types_mapper={pa.string(): pd.ArrowDtype(pa.string()),
                                          pa.large_string(): pd.ArrowDtype(pa.string())}.get)
    for name in CATEGORICAL:
        if name in frame:
            frame[name] = frame[name].cat.remove_unused_categories()
    return frame


def _read_file(read_path: str, repo_path: str, fs) -> tuple[pa.Table, pa.Table]:
    if fs is None:
        table = pq.read_table(read_path, columns=META_COLUMNS)
    else:
        # Small blocks plus pre_buffer fetch only the metadata column chunks; with large
        # blocks the byte channels in between are downloaded too (about 7x slower).
        with fs.open(read_path, "rb", block_size=READ_BLOCK_BYTES) as handle:
            table = pq.ParquetFile(handle, pre_buffer=True).read(columns=META_COLUMNS)
    return _split_file(table, repo_path)


def read_metadata(files: list[tuple[str, str]], local: bool, part: str) -> tuple[pa.Table, pa.Table]:
    """Read the metadata columns of every file in one repo; returns (segments, windows) as Arrow."""
    from concurrent.futures import ThreadPoolExecutor

    fs = None
    if not local:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    segment_parts, window_parts = [], []
    with ThreadPoolExecutor(max_workers=1 if local else READ_WORKERS) as pool:
        futures = [pool.submit(_read_file, read_path, repo_path, fs) for read_path, repo_path in files]
        for (_, repo_path), future in zip(files, futures):
            segments, windows = future.result()
            segment_parts.append(segments.append_column("part", pa.array([part] * segments.num_rows, pa.string())))
            window_parts.append(windows)
            LOG.info("Read metadata of %s (%d segments, %d windows)", repo_path, segments.num_rows, windows.num_rows)
    if not segment_parts:
        from .build_segments import SCHEMA

        empty = pa.schema([SCHEMA.field(c) for c in META_COLUMNS]).empty_table()
        empty_seg, empty_win = _split_file(empty, "")
        return empty_seg.append_column("part", pa.array([], pa.string())), empty_win
    return (pa.concat_tables(segment_parts, promote_options="permissive"),
            pa.concat_tables(window_parts, promote_options="permissive"))


def explode_windows(segments: pd.DataFrame) -> pd.DataFrame:
    counts = segments["window_offsets"].map(len).to_numpy()
    idx = np.repeat(np.arange(len(segments)), counts)
    windows = segments.iloc[idx][
        ["file", "row", "segment_id", "star_key", "source", "split", "quality_tier", "var_ratio"]
    ].reset_index(drop=True)
    windows["window_index"] = np.concatenate([np.arange(c) for c in counts]).astype(np.int16) if len(counts) else []
    windows["start_offset"] = np.concatenate(segments["window_offsets"].map(np.asarray).to_list()).astype(np.int32) if len(counts) else []
    windows["coverage"] = np.concatenate(segments["window_coverage"].map(np.asarray).to_list()).astype(np.float32) if len(counts) else []
    return windows


def add_weights(windows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = []
    thresholds = {}
    for split, group in windows.groupby("split", sort=False, observed=True):
        group = group.copy()
        per_source = group["source"].value_counts()
        per_source = per_source[per_source > 0]
        present = {s: config.SOURCE_SHARES[s] for s in per_source.index}
        total_share = sum(present.values())
        source_weight = {s: present[s] / total_share / per_source[s] * len(group) for s in present}
        group["w_source"] = group["source"].astype(str).map(source_weight).astype(np.float64)
        star_windows = group.groupby("star_key", observed=True)["segment_id"].transform("size")
        group["w_star"] = np.minimum(1.0, config.STAR_WINDOW_CAP / star_windows)
        group["w_var"] = 1.0
        for source, part in group.groupby("source", observed=True):
            cut = float(part["var_ratio"].quantile(1 - config.VARIABILITY_TOP_FRACTION))
            thresholds[f"{split}/{source}"] = round(cut, 4)
            group.loc[part.index[part["var_ratio"] >= cut], "w_var"] = config.VARIABILITY_WEIGHT
        group["w_tier"] = np.where(group["quality_tier"] == 3, config.TIER3_WEIGHT, 1.0)
        product = group["w_source"] * group["w_star"] * group["w_var"] * group["w_tier"]
        group["w_default"] = product / product.mean()
        out.append(group)
    weighted = pd.concat(out, ignore_index=True) if out else windows
    for column in ("w_source", "w_star", "w_var", "w_tier", "w_default"):
        weighted[column] = weighted[column].astype(np.float32)
    return weighted, thresholds


def build(local: Path | None, out_dir: Path) -> dict:
    files = list_files(local)
    out_dir.mkdir(parents=True, exist_ok=True)
    pre_segments, pre_windows = read_metadata(files["pretrain"], local is not None, "pretrain")
    held_segments, held_windows = read_metadata(files["heldout"], local is not None, "heldout")

    windows, thresholds = add_weights(_to_pandas(pre_windows))
    del pre_windows
    for split in config.PRETRAIN_SPLITS:
        windows[windows["split"] == split].to_parquet(out_dir / f"windows_{split}.parquet", index=False)
    windows_by_split = windows.groupby(["split", "source"], observed=True).size().unstack(fill_value=0).to_dict()
    del windows
    pq.write_table(held_windows, out_dir / "windows_heldout.parquet")
    del held_windows

    everything = _to_pandas(pa.concat_tables([pre_segments, held_segments], promote_options="permissive"))
    leaked = sorted(set(pc.unique(pre_segments["star_key"]).to_pylist())
                    & set(pc.unique(held_segments["star_key"]).to_pylist()))
    del pre_segments, held_segments

    stars = (everything.groupby(["star_key", "split", "source"], observed=True)["n_windows"]
             .agg(segments="size", windows="sum").unstack("source", fill_value=0))
    stars.columns = [f"{kind}_{source}" for kind, source in stars.columns]
    stars = stars.reset_index()
    stars["star_key"] = stars["star_key"].astype(str)
    stars["split"] = stars["split"].astype(str)
    stars.to_parquet(out_dir / "stars.parquet", index=False)

    tess = everything[everything["dup_group"].notna()]
    dup_pairs = int((tess.groupby("dup_group", observed=True)["source"].nunique() > 1).sum())

    def counts(*keys):
        return everything.groupby(list(keys), observed=True).size().unstack(fill_value=0).to_dict()

    summary = {
        "build_version": config.BUILD_VERSION,
        "files": {"pretrain": len(files["pretrain"]), "heldout": len(files["heldout"])},
        "segments": counts("part", "source"),
        "segments_by_split": counts("split", "source"),
        "windows_by_split": windows_by_split,
        "quality_tiers": counts("source", "quality_tier"),
        "stars": int(stars["star_key"].nunique()),
        "ffi_2min_duplicate_pairs": dup_pairs,
        "variability_thresholds": thresholds,
        "star_keys_in_both_pretrain_and_heldout": len(leaked),
        "leaked_examples": leaked[:20],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    LOG.info("Index written to %s: %s", out_dir, json.dumps({k: summary[k] for k in ("files", "stars", "star_keys_in_both_pretrain_and_heldout")}))
    return summary


def upload(out_dir: Path) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    pretrain_files = ["windows_train.parquet", "windows_validation.parquet", "stars.parquet", "summary.json"]
    api.create_commit(config.PRETRAIN_REPO, [
        CommitOperationAdd(path_in_repo=f"index/{name}", path_or_fileobj=str(out_dir / name)) for name in pretrain_files
    ], repo_type="dataset", commit_message=f"{config.BUILD_VERSION}: window index and weights")
    heldout_files = ["windows_heldout.parquet", "summary.json"]
    if (out_dir / "exclusions.parquet").exists():
        heldout_files.append("exclusions.parquet")
    api.create_commit(config.HELDOUT_REPO, [
        CommitOperationAdd(path_in_repo=f"index/{name}", path_or_fileobj=str(out_dir / name)) for name in heldout_files
    ], repo_type="dataset", commit_message=f"{config.BUILD_VERSION}: held-out window index and exclusion list")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pass 2: window index, star table and weights")
    parser.add_argument("--local", type=Path, default=None, help="read pass-1 output from this folder instead of the Hub")
    parser.add_argument("--out", type=Path, default=Path("build/index"))
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
    for noisy in ("httpx", "httpx2", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    build(args.local, args.out)
    if args.upload:
        upload(args.out)


if __name__ == "__main__":
    main()
