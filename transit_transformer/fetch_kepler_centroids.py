"""Fetch Kepler centroid columns from MAST for every file in the
kepler-raw-lightcurves manifest.

The HF set `nyu-planet-finders/kepler-raw-lightcurves` dropped MOM_CENTR1/2 and
POS_CORR1/2. This re-downloads each original *_llc.fits from MAST, checks its
sha256 against the manifest, keeps only the centroid columns, and deletes the
FITS file.

Output mirrors the light-curve shards one-to-one: centroids/part-XXXXX.parquet
holds one row per light-curve row, in the same `row_in_shard` order, so the two
can be zipped and checked by `cadenceno`.

Usage:
    # 1. fetch (resumable; rerun to retry failures)
    python fetch_kepler_centroids.py fetch --manifest conversion_manifest.csv --out kepler_centroids
    # 2. pack per-file results into parquet shards
    python fetch_kepler_centroids.py pack --manifest conversion_manifest.csv --out kepler_centroids

--manifest may be a local path or
hf://datasets/nyu-planet-finders/kepler-raw-lightcurves/conversion_manifest.csv
(needs `hf auth login`, since that dataset is private).
Requires: numpy pandas astropy pyarrow (and huggingface_hub for hf:// paths).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

COLUMNS = ["CADENCENO", "MOM_CENTR1", "MOM_CENTR1_ERR", "MOM_CENTR2",
           "MOM_CENTR2_ERR", "POS_CORR1", "POS_CORR2"]
ARCHIVE = "https://archive.stsci.edu/"
USER_AGENT = "kepler-centroid-fetch/1.0"


def read_manifest(path: str) -> pd.DataFrame:
    if path.startswith("hf://"):
        from huggingface_hub import HfFileSystem
        with HfFileSystem().open(path[len("hf://"):], "rb") as fh:
            return pd.read_csv(fh)
    return pd.read_csv(path)


def mast_url(source_uri: str) -> str:
    # mast:KEPLER/url/missions/kepler/lightcurves/0007/000757450/kplr...fits
    return ARCHIVE + source_uri.split("/url/", 1)[1]


def part_path(out: str, source_filename: str) -> str:
    return os.path.join(out, "parts", source_filename.replace(".fits", ".npz"))


def fetch_one(row, out: str, retries: int = 4) -> str:
    dest = part_path(out, row.source_filename)
    if os.path.exists(dest):
        return "skip"
    from astropy.io import fits
    url = mast_url(row.source_uri)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                blob = r.read()
            break
        except Exception as e:          # network hiccup / MAST throttling
            last = e
            time.sleep(2 ** attempt * 5)
    else:
        raise RuntimeError(f"download failed: {url}: {last}")

    digest = hashlib.sha256(blob).hexdigest()
    if digest != row.sha256:
        raise RuntimeError(f"sha256 mismatch for {row.source_filename}")

    with fits.open(io.BytesIO(blob), memmap=False) as hdul:
        data = hdul["LIGHTCURVE"].data
        arrays = {c.lower(): np.asarray(data[c]) for c in COLUMNS}
    n = len(arrays["cadenceno"])
    if n != row.row_count:
        raise RuntimeError(f"{row.source_filename}: {n} cadences, manifest says {row.row_count}")

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp.npz"
    np.savez_compressed(tmp, cadenceno=arrays["cadenceno"].astype(np.int32),
                        **{k: v.astype(np.float32) for k, v in arrays.items() if k != "cadenceno"})
    os.replace(tmp, dest)               # atomic: a part file exists only if complete
    return "ok"


def cmd_fetch(args):
    man = read_manifest(args.manifest)
    if args.limit:
        man = man.head(args.limit)
    todo = [r for r in man.itertuples() if not os.path.exists(part_path(args.out, r.source_filename))]
    print(f"{len(man)} files in manifest, {len(man) - len(todo)} already done, {len(todo)} to fetch")
    failures = []
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(fetch_one, r, args.out): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            try:
                fut.result()
            except Exception as e:
                failures.append((r.source_filename, str(e)))
            if i % 500 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {len(failures)} failed", flush=True)
    if failures:
        path = os.path.join(args.out, "failures.csv")
        pd.DataFrame(failures, columns=["source_filename", "error"]).to_csv(path, index=False)
        print(f"{len(failures)} failures written to {path}; rerun to retry")
        sys.exit(1)


def cmd_pack(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    man = read_manifest(args.manifest)
    if args.limit:
        man = man.head(args.limit)
    os.makedirs(os.path.join(args.out, "centroids"), exist_ok=True)
    for shard, grp in man.groupby("shard", sort=True):
        grp = grp.sort_values("row_in_shard")
        missing = [f for f in grp.source_filename if not os.path.exists(part_path(args.out, f))]
        if missing:
            print(f"{shard}: {len(missing)} files not fetched yet, skipping")
            continue
        cols = {"kepid": [], "quarter": [], "source_filename": [], "row_in_shard": []}
        lists = {c.lower(): [] for c in COLUMNS}
        for r in grp.itertuples():
            with np.load(part_path(args.out, r.source_filename)) as z:
                for c in lists:
                    lists[c].append(z[c])
            cols["kepid"].append(r.kepid)
            cols["quarter"].append(r.quarter)
            cols["source_filename"].append(r.source_filename)
            cols["row_in_shard"].append(r.row_in_shard)
        table = pa.table({**cols, **{c: pa.array(v) for c, v in lists.items()}})
        dest = os.path.join(args.out, "centroids", os.path.basename(shard))
        pq.write_table(table, dest, compression="zstd")
        print(f"{shard}: {len(grp)} rows -> {dest}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["fetch", "pack"])
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="only the first N manifest rows (testing)")
    args = p.parse_args()
    {"fetch": cmd_fetch, "pack": cmd_pack}[args.command](args)


if __name__ == "__main__":
    main()
