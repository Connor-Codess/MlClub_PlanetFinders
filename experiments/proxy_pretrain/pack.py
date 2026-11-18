"""Pack lc-pretrain windows for one proxy pretraining variant.

What it does
    Reads lc-pretrain's window index, keeps the windows of the files in config.FILES, drops
    every star that appears in lc-benchmarks, and draws config.WINDOWS windows without
    replacement, with each source's share taken from config.VARIANTS and probabilities in
    proportion to the index's stored weights (w_default). Files are downloaded one at a time
    and deleted after use, so the job never holds more than one source file on disk.

Outputs (in --out)
    x.npy         float16 (n, 4, 1296): flux_z, cen_x, cen_y, bkg_z
    observed.npy  uint8 (n, 1296)
    meta.parquet  source, star_key, sigma_p2p_ppm, file, row, start_offset, validation flag

How to run (needs HF_TOKEN for the private lc-benchmarks star list)
    python -m experiments.proxy_pretrain.pack --variant pt_mix --out build/pretrain/pt_mix [--windows 400000]
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pretrain_dataset import config as pretrain

from . import config

LOG = logging.getLogger("pretrain.pack")
COLUMNS = ["star_key", "n_slots", "sigma_p2p_ppm", "flux_z", "cen_x", "cen_y", "bkg_z", "flags"]


def choose(index: pd.DataFrame, variant: str, n: int, excluded: set[str], rng: np.random.Generator) -> pd.DataFrame:
    picks = []
    for source, share in config.VARIANTS[variant].items():
        pool = index[(index["source"] == source) & index["file"].isin(config.FILES[source])
                     & ~index["star_key"].astype(str).isin(excluded)]
        want = int(round(n * share))
        if len(pool) < want:
            raise RuntimeError(f"{variant}: {source} has {len(pool)} windows, {want} wanted")
        weights = pool["w_default"].to_numpy(dtype=np.float64)
        chosen = rng.choice(len(pool), size=want, replace=False, p=weights / weights.sum())
        picks.append(pool.iloc[np.sort(chosen)])
        LOG.info("%s: %d of %d windows from %d files", source, want, len(pool), pool["file"].nunique())
    out = pd.concat(picks, ignore_index=True)
    out["validation"] = rng.uniform(size=len(out)) < config.VALIDATION_SHARE
    return out


def pack(chosen: pd.DataFrame, out: Path, cache: Path) -> None:
    from huggingface_hub import hf_hub_download

    n = len(chosen)
    out.mkdir(parents=True, exist_ok=True)
    x = np.lib.format.open_memmap(out / "x.npy", mode="w+", dtype=np.float16, shape=(n, 4, pretrain.WINDOW_SLOTS))
    observed = np.lib.format.open_memmap(out / "observed.npy", mode="w+", dtype=np.uint8, shape=(n, pretrain.WINDOW_SLOTS))
    sigma = np.zeros(n, dtype=np.float32)
    star = np.empty(n, dtype=object)
    for file, group in chosen.groupby("file", sort=True):
        local = Path(hf_hub_download(config.PRETRAIN_REPO, file, repo_type="dataset", local_dir=cache,
                                     token=os.environ.get("HF_TOKEN")))
        parquet = pq.ParquetFile(local)
        bounds = np.cumsum([0] + [parquet.metadata.row_group(g).num_rows for g in range(parquet.num_row_groups)])
        rows = group["row"].to_numpy()
        groups = np.searchsorted(bounds, rows, side="right") - 1
        for g in np.unique(groups):
            table = parquet.read_row_group(int(g), columns=COLUMNS)
            for at, row, start in zip(group.index[groups == g], rows[groups == g],
                                      group["start_offset"].to_numpy()[groups == g]):
                record = {c: table.column(c)[int(row - bounds[g])].as_py() for c in COLUMNS}
                arrays = {name: np.frombuffer(record[name], dtype="<f2").astype(np.float32)
                          for name in ("flux_z", "cen_x", "cen_y", "bkg_z")}
                arrays["flags"] = np.frombuffer(record["flags"], dtype=np.uint8)
                stop = min(int(start) + pretrain.WINDOW_SLOTS, record["n_slots"])
                length = stop - int(start)
                for i, name in enumerate(("flux_z", "cen_x", "cen_y", "bkg_z")):
                    x[at, i, :length] = arrays[name][int(start):stop]
                    x[at, i, length:] = 0
                obs = (arrays["flags"][int(start):stop] & pretrain.FLAG_OBSERVED) > 0
                observed[at, :length] = obs
                observed[at, length:] = 0
                sigma[at] = record["sigma_p2p_ppm"]
                star[at] = record["star_key"]
        local.unlink()
        LOG.info("packed %d windows from %s", len(group), file)
    x.flush()
    observed.flush()
    meta = chosen[["source", "file", "row", "start_offset", "validation"]].copy()
    meta["star_key"] = star
    meta["sigma_p2p_ppm"] = sigma
    meta.to_parquet(out / "meta.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, choices=sorted(config.VARIANTS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("build/pretrain/source"))
    parser.add_argument("--windows", type=int, default=config.WINDOWS)
    parser.add_argument("--index", type=Path, help="a local windows_train.parquet (default: download it)")
    parser.add_argument("--exclude", type=Path, help="a local benchmark_targets.parquet (default: download it)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    index_path = args.index or hf_hub_download(config.PRETRAIN_REPO, "index/windows_train.parquet",
                                               repo_type="dataset", token=token)
    columns = ["file", "row", "star_key", "source", "start_offset", "w_default"]
    files = [f for fs in config.FILES.values() for f in fs]
    index = pq.read_table(index_path, columns=columns, filters=[("file", "in", files)]).to_pandas()
    exclude_path = args.exclude or hf_hub_download(config.BENCHMARK_REPO, "benchmark_targets.parquet",
                                                   repo_type="dataset", token=token)
    excluded = set(pd.read_parquet(exclude_path)["star_key"].astype(str))
    LOG.info("index: %d windows in the chosen files; %d benchmark stars excluded", len(index), len(excluded))
    chosen = choose(index, args.variant, args.windows, excluded, np.random.default_rng(config.SAMPLE_SEED))
    pack(chosen, args.out, args.cache)
    LOG.info("wrote %d windows (%d validation) to %s", len(chosen), int(chosen["validation"].sum()), args.out)


if __name__ == "__main__":
    main()
