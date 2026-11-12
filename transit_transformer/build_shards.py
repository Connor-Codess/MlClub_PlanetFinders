"""Convert the Hugging Face light-curve sets into binned window shards.

Each source Parquet file becomes one or more .npz shards per split:
    <out>/<mission>/<split>/<source>-<k>.npz
plus a marker <out>/<mission>/_done/<source> so reruns skip finished files.

Shards hold *binned, unscaled* windows (preprocess.Binned); scaling and noise
augmentation happen in dataset.py at load time.

    flux_rel  float32 [N, L]   flux / median - 1     (float16 loses ~60 ppm on variables)
    err_rel   float16 [N, L]
    mask      bool    [N, L]
    cx, cy    float32 [N, L]   arcsec, NaN where unavailable
    has_c     bool    [N]
    magnitude, teff, radius   float32 [N]  (NaN if missing)
    sigma     float32 [N]      noise per 30-min bin (for augmentation targets)
    star_id int64, segment int16, window int8, t0_bjd float64, fill float32  [N]

Usage:
    python build_shards.py tess   --out shards                    # all sectors
    python build_shards.py tess   --out shards --sources s0013     # one sector
    python build_shards.py kepler --out shards --centroids kepler_centroids/centroids
Kepler needs `hf auth login` (private dataset).
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from collections import defaultdict

import numpy as np

import preprocess as P
import splits as S

TESS_REPO = "datasets/nyu-planet-finders/tess-spoc-2min-binned"
KEPLER_REPO = "datasets/nyu-planet-finders/kepler-raw-lightcurves"
WINDOWS_PER_SHARD = 10_000


class ShardWriter:
    def __init__(self, out, mission, source):
        self.out, self.mission, self.source = out, mission, source
        self.buf = defaultdict(list)
        self.count = defaultdict(int)

    def add(self, split, b: P.Binned, sigma: float):
        self.buf[split].append((b, sigma))
        if len(self.buf[split]) >= WINDOWS_PER_SHARD:
            self.flush(split)

    def flush(self, split):
        items = self.buf.pop(split, [])
        if not items:
            return
        L = P.WINDOW_BINS
        nan = np.full(L, np.nan, np.float32)

        def opt(v):
            return np.float32(np.nan if v is None else v)

        arrays = dict(
            flux_rel=np.stack([b.flux_rel for b, _ in items]).astype(np.float32),
            err_rel=np.stack([b.err_rel for b, _ in items]).astype(np.float16),
            mask=np.stack([b.mask for b, _ in items]),
            cx=np.stack([b.cx if b.cx is not None else nan for b, _ in items]).astype(np.float32),
            cy=np.stack([b.cy if b.cy is not None else nan for b, _ in items]).astype(np.float32),
            has_c=np.array([b.cx is not None for b, _ in items]),
            magnitude=np.array([opt(b.magnitude) for b, _ in items]),
            teff=np.array([opt(b.teff) for b, _ in items]),
            radius=np.array([opt(b.radius) for b, _ in items]),
            sigma=np.array([s for _, s in items], np.float32),
            star_id=np.array([b.meta["star_id"] for b, _ in items], np.int64),
            segment=np.array([b.meta["segment"] for b, _ in items], np.int16),
            window=np.array([b.meta["window"] for b, _ in items], np.int8),
            t0_bjd=np.array([b.meta["t0_bjd"] for b, _ in items], np.float64),
            fill=np.array([b.meta["fill_fraction"] for b, _ in items], np.float32),
        )
        d = os.path.join(self.out, self.mission, split)
        os.makedirs(d, exist_ok=True)
        k = self.count[split]
        path = os.path.join(d, f"{self.source}-{k:03d}.npz")
        np.savez_compressed(path + ".tmp.npz", **arrays)
        os.replace(path + ".tmp.npz", path)
        self.count[split] += 1

    def close(self):
        for split in list(self.buf):
            self.flush(split)


def process(lcs, out, mission, source, overlap=frozenset()):
    """lcs: iterable of preprocess.LightCurve from one source file."""
    w = ShardWriter(out, mission, source)
    stats = defaultdict(int)
    for lc in lcs:
        stats["light_curves"] += 1
        bins = P.bin_windows(lc)
        if not bins:
            stats["light_curves_no_window"] += 1
        for b in bins:
            split = S.assign(mission, lc.star_id, lc.segment, overlap)
            sigma = float(10 ** P.scale_window(b).side[0])
            w.add(split, b, sigma)
            stats[f"windows_{split}"] += 1
    w.close()
    done = os.path.join(out, mission, "_done")
    os.makedirs(done, exist_ok=True)
    with open(os.path.join(done, source), "w") as fh:
        fh.write(repr(dict(stats)))
    return dict(stats)


def _iter_parquet(fs, path, batch_rows=256):
    import pyarrow.parquet as pq
    with fs.open(path, "rb") as fh:
        pf = pq.ParquetFile(fh)
        for batch in pf.iter_batches(batch_size=batch_rows):
            yield from batch.to_pylist()


def tess_sources(fs, wanted):
    files = sorted(p.rsplit("/", 1)[1][:-8] for p in fs.glob(f"{TESS_REPO}/s*.parquet"))
    return [s for s in files if not wanted or s in wanted]


def clear_partial(out, mission, src):
    """Remove shards left by an interrupted run of this source."""
    for p in glob.glob(os.path.join(out, mission, "*", f"{src}-*.npz")):
        os.remove(p)


def with_retries(fn, src, attempts=3, wait=60):
    """Rerun a whole source after network errors (HF read timeouts etc.)."""
    for i in range(1, attempts + 1):
        try:
            return fn()
        except (OSError, TimeoutError, ConnectionError) as e:   # httpx timeouts subclass these? not always
            err = e
        except Exception as e:
            if "timed out" not in str(e).lower() and "connection" not in str(e).lower():
                raise
            err = e
        print(f"{src}: attempt {i}/{attempts} failed ({type(err).__name__}: {err}); "
              f"retrying in {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"{src}: failed after {attempts} attempts") from err


def run_tess(args, fs, overlap):
    for src in tess_sources(fs, args.sources):
        if os.path.exists(os.path.join(args.out, "tess", "_done", src)):
            print(f"{src}: done, skipping")
            continue
        t = time.time()

        def one():
            clear_partial(args.out, "tess", src)
            rows = _iter_parquet(fs, f"{TESS_REPO}/{src}.parquet")
            return process((P.tess_from_hf_row(r) for r in rows), args.out, "tess", src, overlap)

        stats = with_retries(one, src)
        print(f"{src}: {stats} in {time.time() - t:.0f}s", flush=True)


def run_kepler(args, fs):
    import pyarrow.parquet as pq
    parts = sorted(p.rsplit("/", 1)[1][:-8] for p in fs.glob(f"{KEPLER_REPO}/lightcurves/*.parquet"))
    for src in parts:
        if args.sources and src not in args.sources:
            continue
        if os.path.exists(os.path.join(args.out, "kepler", "_done", src)):
            print(f"{src}: done, skipping")
            continue
        cent_path = os.path.join(args.centroids, f"{src}.parquet") if args.centroids else None
        if cent_path and not os.path.exists(cent_path):
            print(f"{src}: centroid shard missing ({cent_path}), skipping")
            continue
        cent_rows = pq.read_table(cent_path).to_pylist() if cent_path else None

        def lcs():
            cents = iter(cent_rows) if cent_rows is not None else None
            for row in _iter_parquet(fs, f"{KEPLER_REPO}/lightcurves/{src}.parquet"):
                c = next(cents) if cents else None
                if c is not None and (c["kepid"], c["quarter"]) != (row["kepid"], row["quarter"]):
                    raise ValueError(f"{src}: centroid row order does not match light curves")
                yield P.kepler_from_hf_row(row, c)

        def one():
            clear_partial(args.out, "kepler", src)
            return process(lcs(), args.out, "kepler", src)

        t = time.time()
        stats = with_retries(one, src)
        print(f"{src}: {stats} in {time.time() - t:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mission", choices=["tess", "kepler"])
    p.add_argument("--out", required=True)
    p.add_argument("--sources", nargs="*", default=None, help="e.g. s0013 or part-00000")
    p.add_argument("--centroids", default=None, help="dir of centroid parquet shards (Kepler)")
    p.add_argument("--overlap", default=None, help="CSV of tic_id also observed by Kepler")
    args = p.parse_args()
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    if args.mission == "tess":
        run_tess(args, fs, S.load_overlap(args.overlap))
    else:
        run_kepler(args, fs)


if __name__ == "__main__":
    main()
