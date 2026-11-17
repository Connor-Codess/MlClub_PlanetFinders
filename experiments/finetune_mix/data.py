"""Reading lc-finetune and lc-eval rows into compact arrays for the experiments.

What it does
    download()     fetches the Parquet shards of one split from Hugging Face
    read_meta()    the per-row columns used to pick arms and bin scores
    pack_rows()    turns chosen rows into arrays on disk:
                       x.npy         float16 (n, 4, 1296): flux_z, cen_x, cen_y, bkg_z
                       observed.npy  uint8   (n, 1296): 1 where the slot was observed
                       label.npy     uint8   (n, 162): training target per patch
                       target.npy    uint8   (n, 162): 1 on patches that count as finding a transit
                       meta.parquet  one row per sample
    load_pack()    reads such a directory back

Why
    A training job on a 4-CPU runner cannot afford to decode Parquet rows every epoch.
    The packed arrays are read once into memory (about 12 KB per row).

Training target
    patch_label as published (0, 1, 255), except that patches a planet transit overlaps
    without containing its mid-time are set to ignore (255): their dip is real, so they
    should be neither rewarded nor punished.

How to run
    Used by select.py, pack.py and train.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config

META_COLUMNS = ["sample_id", "category", "source", "star_key", "single_transit_crop", "inj_model", "inj_snr",
                "inj_snr_total", "inj_period_d", "stacked_snr", "n_visible_transits", "coverage"]
ARRAY_COLUMNS = list(config.INPUT_CHANNELS) + ["flags", "patch_label", "patch_overlap"]


def download(repo: str, split: str, out: Path, only: list[str] | None = None) -> list[Path]:
    """Download data/<split>/*.parquet (or the listed repo paths) of a dataset repo."""
    from huggingface_hub import snapshot_download

    patterns = only if only is not None else [f"data/{split}/*.parquet"]
    root = snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns, local_dir=out,
                             max_workers=4, token=os.environ.get("HF_TOKEN"))
    return sorted(Path(out).glob(f"data/{split}/*.parquet")) if root else []


def best_transit_snr(events) -> float:
    snrs = [e["snr"] for e in (events if events is not None else []) if e["is_planet"] and e["visible"]]
    return float(max(snrs)) if snrs else 0.0


def read_meta(files: list[Path], root: Path) -> pd.DataFrame:
    """Per-row metadata of every shard, with the shard path and row number."""
    frames = []
    for f in files:
        table = pq.read_table(f, columns=META_COLUMNS + ["events"])
        frame = table.drop(["events"]).to_pandas()
        frame["best_transit_snr"] = [best_transit_snr(ev) for ev in table.column("events").to_pylist()]
        frame["path"] = str(Path(f).resolve().relative_to(Path(root).resolve()))
        frame["row"] = np.arange(len(frame), dtype=np.int64)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def train_label(patch_label: np.ndarray, patch_overlap: np.ndarray) -> np.ndarray:
    label = patch_label.copy()
    label[(patch_label == config.LABEL_NEGATIVE) & (patch_overlap > 0)] = config.LABEL_IGNORE
    return label


def target_mask(patch_label: np.ndarray) -> np.ndarray:
    """Patches within TARGET_DILATION of a labeled transit patch, per row."""
    hit = patch_label == config.LABEL_TRANSIT
    out = hit.copy()
    for k in range(1, config.TARGET_DILATION + 1):
        out[:, k:] |= hit[:, :-k]
        out[:, :-k] |= hit[:, k:]
    return out.astype(np.uint8)


def _decode(column, dtype, width: int) -> np.ndarray:
    return np.frombuffer(b"".join(column), dtype=dtype).reshape(len(column), width)


def pack_rows(chosen: pd.DataFrame, root: Path, out: Path) -> None:
    """Write the arrays of the chosen rows (columns path, row) in the order given."""
    out.mkdir(parents=True, exist_ok=True)
    n = len(chosen)
    x = np.lib.format.open_memmap(out / "x.npy", mode="w+", dtype=np.float16,
                                  shape=(n, len(config.INPUT_CHANNELS), config.WINDOW_SLOTS))
    observed = np.lib.format.open_memmap(out / "observed.npy", mode="w+", dtype=np.uint8, shape=(n, config.WINDOW_SLOTS))
    label = np.lib.format.open_memmap(out / "label.npy", mode="w+", dtype=np.uint8, shape=(n, config.N_PATCHES))
    target = np.lib.format.open_memmap(out / "target.npy", mode="w+", dtype=np.uint8, shape=(n, config.N_PATCHES))
    chosen = chosen.reset_index(drop=True)
    metas = []
    for path, group in chosen.groupby("path", sort=True):
        rows = group["row"].to_numpy()
        table = pq.read_table(root / path, columns=ARRAY_COLUMNS + META_COLUMNS).take(rows)
        at = group.index.to_numpy()
        for i, name in enumerate(config.INPUT_CHANNELS):
            x[at, i] = _decode(table.column(name).to_pylist(), "<f2", config.WINDOW_SLOTS)
        flags = _decode(table.column("flags").to_pylist(), np.uint8, config.WINDOW_SLOTS)
        observed[at] = (flags & 1) > 0
        patch_label = _decode(table.column("patch_label").to_pylist(), np.uint8, config.N_PATCHES)
        overlap = _decode(table.column("patch_overlap").to_pylist(), np.uint8, config.N_PATCHES)
        label[at] = train_label(patch_label, overlap)
        target[at] = target_mask(patch_label)
        meta = table.select(META_COLUMNS).to_pandas()
        meta.index = at
        metas.append(meta)
    for arr in (x, observed, label, target):
        arr.flush()
    meta = pd.concat(metas).sort_index()
    extra = [c for c in ("best_transit_snr",) if c in chosen.columns]
    for c in extra:
        meta[c] = chosen[c].to_numpy()
    meta.to_parquet(out / "meta.parquet", index=False)


def load_pack(path: Path, mmap: bool = False) -> dict:
    mode = "r" if mmap else None
    out = {name: np.load(path / f"{name}.npy", mmap_mode=mode) for name in ("x", "observed", "label", "target")}
    out["meta"] = pd.read_parquet(path / "meta.parquet")
    return out


def model_input(x: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """float32 (n, 5, 1296): clipped and scaled channels, zero where unobserved, plus the mask."""
    obs = observed.astype(np.float32)[:, None, :]
    z = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    z = np.clip(z, -config.INPUT_CLIP, config.INPUT_CLIP) / config.INPUT_SCALE * obs
    return np.concatenate([z, obs], axis=1)
