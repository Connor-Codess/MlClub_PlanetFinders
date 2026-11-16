"""Reference loader for lc-pretrain: sample windows and return model-ready arrays.

This is the "default loader" from the build guide, kept free of any deep
learning framework so any model code can wrap it.

What one sample contains (window_slots = 1296 by default)
    channels   float32 array (window_slots, 9):
               0 flux        asinh-compressed flux_z: 10 * asinh(flux_z / 10)
               1 flux_err    flux_err_z
               2 cen_x       arcsec
               3 cen_y       arcsec
               4 bkg         bkg_z
               5 fill        0-1
               6 observed    1 where the slot has data
               7 cen_valid   1 where the centroid is valid
               8 bkg_valid   1 where the background is valid
    meta       float32 array (8,): log10 sigma_p2p_ppm, magnitude (Tmag or Kp),
               Teff / 10000, log10 radius, source one-hot (ffi, 2min, kepler),
               missing-star-parameter flag
    info       dict: segment_id, star_key, source, start_offset, file, row

Catalog tags and known events are deliberately not returned: they are never
model inputs (build guide, Step 7).

Usage
    from pretrain_dataset.loader import WindowSampler
    sampler = WindowSampler(split="train")                 # reads the index from the Hub
    batch = sampler.sample(32)                             # list of 32 samples
    sampler = WindowSampler(split="train", local_dir="build/pass1/pretrain",
                            index_dir="build/index", sources=["ffi"])

Keeping benchmark test stars out of pretraining (their light curves are in
lc-pretrain without labels; lc-finetune already excludes them):
    targets = pd.read_parquet(hf_hub_download("nyu-planet-finders/lc-benchmarks",
                                              "benchmark_targets.parquet", repo_type="dataset"))
    sampler = WindowSampler(split="train", exclude_star_keys=set(targets["star_key"]))
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config
from .segments import decode

DATA_COLUMNS = ["segment_id", "star_key", "source", "n_slots", "sigma_p2p_ppm", "tmag", "kepmag",
                "teff", "radius", *config.CHANNELS_FLOAT16, *config.CHANNELS_UINT8]
SOURCES = ("ffi", "2min", "kepler")


def compress_flux(flux_z: np.ndarray) -> np.ndarray:
    """Linear near zero, logarithmic beyond about +-10 sigma (build guide, Step 5)."""
    return 10.0 * np.arcsinh(flux_z / 10.0)


class WindowSampler:
    def __init__(self, split: str = "train", sources: list[str] | None = None,
                 local_dir: str | Path | None = None, index_dir: str | Path | None = None,
                 window_slots: int = config.WINDOW_SLOTS, weight_column: str = "w_default",
                 seed: int = 0, cache_row_groups: int = 64, exclude_star_keys: set[str] | None = None):
        self.local_dir = Path(local_dir) if local_dir else None
        self.window_slots = window_slots
        self.rng = np.random.default_rng(seed)
        self._fs = None
        self._parquet: dict[str, pq.ParquetFile] = {}
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_row_groups

        index_path = (Path(index_dir) / f"windows_{split}.parquet") if index_dir else None
        if index_path is None:
            from huggingface_hub import hf_hub_download

            index_path = hf_hub_download(config.PRETRAIN_REPO, f"index/windows_{split}.parquet", repo_type="dataset")
        windows = pd.read_parquet(index_path)
        if sources:
            windows = windows[windows["source"].isin(sources)]
        if exclude_star_keys:
            windows = windows[~windows["star_key"].astype(str).isin(exclude_star_keys)]
        weights = windows[weight_column].to_numpy(dtype=np.float64)
        self.windows = windows.reset_index(drop=True)
        self.probabilities = weights / weights.sum()

    def _open(self, file: str) -> pq.ParquetFile:
        if file not in self._parquet:
            if self.local_dir is not None:
                self._parquet[file] = pq.ParquetFile(self.local_dir / file)
            else:
                from huggingface_hub import HfFileSystem

                self._fs = self._fs or HfFileSystem(token=os.environ.get("HF_TOKEN"))
                handle = self._fs.open(f"datasets/{config.PRETRAIN_REPO}/{file}", "rb", block_size=16 * 2**20)
                self._parquet[file] = pq.ParquetFile(handle)
        return self._parquet[file]

    def _row(self, file: str, row: int) -> dict:
        parquet = self._open(file)
        group, start = 0, 0
        while start + parquet.metadata.row_group(group).num_rows <= row:
            start += parquet.metadata.row_group(group).num_rows
            group += 1
        key = (file, group)
        if key not in self._cache:
            self._cache[key] = parquet.read_row_group(group, columns=DATA_COLUMNS)
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        table = self._cache[key]
        return {name: table.column(name)[row - start].as_py() for name in DATA_COLUMNS}

    def window(self, index: int) -> dict:
        info = self.windows.iloc[index]
        row = self._row(info["file"], int(info["row"]))
        arrays = decode(row)
        start, length, n = int(info["start_offset"]), self.window_slots, int(row["n_slots"])
        stop = min(start + length, n)

        def crop(values: np.ndarray) -> np.ndarray:
            out = np.zeros(length, dtype=np.float32)
            out[: stop - start] = values[start:stop]
            return out

        flags = np.zeros(length, dtype=np.uint8)
        flags[: stop - start] = arrays["flags"][start:stop]
        channels = np.stack([
            compress_flux(crop(arrays["flux_z"])),
            crop(arrays["flux_err_z"]),
            crop(arrays["cen_x"]),
            crop(arrays["cen_y"]),
            crop(arrays["bkg_z"]),
            crop(arrays["fill"]),
            ((flags & config.FLAG_OBSERVED) > 0).astype(np.float32),
            ((flags & config.FLAG_CENTROID_VALID) > 0).astype(np.float32),
            ((flags & config.FLAG_BACKGROUND_VALID) > 0).astype(np.float32),
        ], axis=1)

        magnitude = row["tmag"] if row["tmag"] is not None else row["kepmag"]
        missing = int(magnitude is None or row["teff"] is None or row["radius"] is None)
        one_hot = [1.0 if row["source"] == s else 0.0 for s in SOURCES]
        meta = np.array([
            math.log10(max(row["sigma_p2p_ppm"], 1.0)),
            magnitude if magnitude is not None else 0.0,
            (row["teff"] or 0.0) / 10000.0,
            math.log10(row["radius"]) if row["radius"] and row["radius"] > 0 else 0.0,
            *one_hot,
            float(missing),
        ], dtype=np.float32)
        return {
            "channels": channels,
            "meta": meta,
            "info": {"segment_id": row["segment_id"], "star_key": row["star_key"], "source": row["source"],
                     "start_offset": start, "file": info["file"], "row": int(info["row"])},
        }

    def sample(self, n: int) -> list[dict]:
        picks = self.rng.choice(len(self.windows), size=n, p=self.probabilities)
        return [self.window(int(i)) for i in picks]
