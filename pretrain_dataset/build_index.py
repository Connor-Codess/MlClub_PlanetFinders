"""Pass 2 of the lc-pretrain build: window index, star table and default sampling weights.

What it does (build guide Steps 8-9)
    Reads only the metadata columns of every data file written by pass 1 (the
    byte channels are never downloaded), then writes:

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


def read_metadata(files: list[tuple[str, str]], local: bool) -> pd.DataFrame:
    frames = []
    fs = None
    if not local:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    for read_path, repo_path in files:
        if local:
            table = pq.read_table(read_path, columns=META_COLUMNS)
        else:
            with fs.open(read_path, "rb", block_size=8 * 2**20) as handle:
                table = pq.ParquetFile(handle).read(columns=META_COLUMNS)
        frame = table.to_pandas()
        frame["file"] = repo_path
        frame["row"] = np.arange(len(frame), dtype=np.int64)
        frames.append(frame)
        LOG.info("Read metadata of %s (%d rows)", repo_path, len(frame))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=META_COLUMNS + ["file", "row"])


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
    for split, group in windows.groupby("split", sort=False):
        group = group.copy()
        per_source = group["source"].value_counts()
        present = {s: config.SOURCE_SHARES[s] for s in per_source.index}
        total_share = sum(present.values())
        group["w_source"] = group["source"].map(lambda s: present[s] / total_share / per_source[s] * len(group))
        star_windows = group.groupby("star_key")["segment_id"].transform("size")
        group["w_star"] = np.minimum(1.0, config.STAR_WINDOW_CAP / star_windows)
        group["w_var"] = 1.0
        for source, part in group.groupby("source"):
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
    pretrain = read_metadata(files["pretrain"], local is not None)
    heldout = read_metadata(files["heldout"], local is not None)

    windows, thresholds = add_weights(explode_windows(pretrain))
    for split in config.PRETRAIN_SPLITS:
        windows[windows["split"] == split].to_parquet(out_dir / f"windows_{split}.parquet", index=False)
    heldout_windows = explode_windows(heldout)
    heldout_windows.to_parquet(out_dir / "windows_heldout.parquet", index=False)

    everything = pd.concat([pretrain.assign(part="pretrain"), heldout.assign(part="heldout")], ignore_index=True)
    star_windows = everything.assign(n_windows=everything["window_offsets"].map(len))
    stars = star_windows.pivot_table(index=["star_key", "split"], columns="source", values="n_windows",
                                     aggfunc=["size", "sum"], fill_value=0)
    stars.columns = [f"{'segments' if a == 'size' else 'windows'}_{s}" for a, s in stars.columns]
    stars = stars.reset_index()
    stars.to_parquet(out_dir / "stars.parquet", index=False)

    tess = everything[everything["dup_group"].notna()]
    dup_pairs = int((tess.groupby("dup_group")["source"].nunique() > 1).sum())
    leaked = sorted(set(pretrain["star_key"]) & set(heldout["star_key"]))
    summary = {
        "build_version": config.BUILD_VERSION,
        "files": {"pretrain": len(files["pretrain"]), "heldout": len(files["heldout"])},
        "segments": everything.groupby(["part", "source"]).size().unstack(fill_value=0).to_dict(),
        "segments_by_split": everything.groupby(["split", "source"]).size().unstack(fill_value=0).to_dict(),
        "windows_by_split": windows.groupby(["split", "source"]).size().unstack(fill_value=0).to_dict(),
        "quality_tiers": everything.groupby(["source", "quality_tier"]).size().unstack(fill_value=0).to_dict(),
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
