"""Stage 1 (one job per source file): find every candidate window and transit cutout.

What it does
    Reads the up to three pass-1 files of one source file (lc-pretrain train and
    validation, lc-heldout) and writes three small tables:

    <source>_<stem>_events.parquet
        One row per default 27-day window that contains a known event (TOI, EB,
        Kepler TCE, Hawthorn duotransit), with every event measured in place:
        class, period, SNR, observed fraction. Candidates for the real-transit and
        eclipsing-binary categories.
    <source>_<stem>_hosts.parquet
        A deterministic hash sample of default windows of stars with no catalog
        entry at all. Candidates for injections and random negatives.
    <source>_<stem>_cutouts.parquet
        Detrended cutouts around transits of known planets on train and
        validation stars, for transplant templates.
    <source>_<stem>_bench.parquet
        Default windows containing a predicted transit of a published benchmark
        signal (benchmarks.py), with that signal as the window's event.

Why
    Choosing samples needs the whole build in view (per-star caps, quotas,
    templates stacked across sectors), but reading 230 GB in one job is slow. Each
    scan job reads one file; the choose stage combines their small outputs.

How to run
    python -m finetune_dataset.scan --source ffi --stem s0001 --out build/finetune/scan \
        --catalog-revision <sha> --benchmarks build/finetune/benchmarks/benchmark_targets.parquet \
        [--local-root build/trial]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pretrain_dataset import config as pretrain
from pretrain_dataset.catalogs import Ephemeris, ReferenceCatalogs, predict_events

from . import config
from .injection import measure_event
from .pools import EventCatalog, pool_for
from .storage import fetch, iter_row_groups
from .templates import cutout

LOG = logging.getLogger("finetune.scan")

READ_COLUMNS = ["segment_id", "star_key", "source", "split", "sector", "slot0", "n_slots", "sigma_p2p_ppm",
                "quality_tier", "window_offsets", "window_coverage", "known_events", "flux_z", "flags",
                "radius", "teff", "tmag", "kepmag"]

EVENT_STRUCT = pa.struct([
    ("center_slot", pa.float32()), ("duration_slots", pa.float32()), ("kind", pa.string()),
    ("ref", pa.string()), ("cls", pa.string()), ("period_d", pa.float64()), ("snr", pa.float32()),
    ("observed_fraction", pa.float32()), ("depth_sigma", pa.float32()),
])
IDENTITY = [
    ("repo", pa.string()), ("path", pa.string()), ("row", pa.int64()), ("segment_id", pa.string()),
    ("star_key", pa.string()), ("split", pa.string()), ("pool", pa.string()), ("source", pa.string()),
    ("window_index", pa.int16()), ("start_offset", pa.int32()), ("coverage", pa.float32()),
    ("sigma_p2p_ppm", pa.float32()), ("quality_tier", pa.int8()), ("radius", pa.float32()),
    ("teff", pa.float32()),
]
EVENTS_SCHEMA = pa.schema(IDENTITY + [("planet_star", pa.bool_()), ("events", pa.list_(EVENT_STRUCT))])
HOSTS_SCHEMA = pa.schema(IDENTITY)
BENCH_SCHEMA = pa.schema(IDENTITY + [("benchmark", pa.string()), ("tce_id", pa.string()), ("label", pa.string()),
                                     ("events", pa.list_(EVENT_STRUCT))])
BENCH_SECTOR_LIMIT = {"salinas2025": 26}
CUTOUTS_SCHEMA = pa.schema([
    ("ref", pa.string()), ("star_key", pa.string()), ("pool", pa.string()), ("source", pa.string()),
    ("segment_id", pa.string()), ("period_d", pa.float64()), ("duration_h", pa.float32()),
    ("snr", pa.float32()), ("dt_h", pa.list_(pa.float32())), ("rel", pa.list_(pa.float32())),
])


def unit_hash(*parts) -> float:
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def duration_or_default(value) -> float:
    if value is None or not math.isfinite(value) or value <= 0:
        return config.DEFAULT_EVENT_DURATION_H * 2.0
    return float(value)


class BenchmarkIndex:
    """Benchmark signals by star, with the source and time range each may be matched in."""

    def __init__(self, targets: pd.DataFrame | None):
        self.by_star: dict[str, list[dict]] = {}
        if targets is None:
            return
        for r in targets.to_dict("records"):
            self.by_star.setdefault(r["star_key"], []).append(r)

    def windows(self, row: dict, base: dict, windows: list) -> list[dict]:
        targets = self.by_star.get(row["star_key"])
        if not targets:
            return []
        source = row["source"]
        t_start = pretrain.GRID_EPOCH_BTJD + row["slot0"] * pretrain.SLOT_DAYS
        t_end = t_start + row["n_slots"] * pretrain.SLOT_DAYS
        flux_z = np.frombuffer(row["flux_z"], dtype="<f2").astype(np.float32)
        observed = (np.frombuffer(row["flags"], dtype=np.uint8) & pretrain.FLAG_OBSERVED) > 0
        out = []
        for target in targets:
            if (target["mission"] == "kepler") != (source == "kepler") or source == "2min":
                continue
            sector = row.get("sector")
            if pd.notna(target.get("sector")) and sector != target["sector"]:
                continue
            limit = BENCH_SECTOR_LIMIT.get(target["benchmark"])
            if limit is not None and (sector is None or sector > limit):
                continue
            lo, hi = target.get("t_min_btjd"), target.get("t_max_btjd")
            if pd.notna(lo) and (t_end < lo or t_start > hi):
                continue
            period = target["period_d"] if pd.notna(target["period_d"]) else None
            duration_h = target["duration_h"] if pd.notna(target["duration_h"]) else config.DEFAULT_EVENT_DURATION_H
            eph = Ephemeris("benchmark", target["tce_id"], float(target["epoch_btjd"]), period, float(duration_h))
            centers = [e["center_slot"] for e in predict_events([eph], row["slot0"], row["n_slots"])]
            if pd.notna(lo):
                centers = [c for c in centers if lo <= t_start + c * pretrain.SLOT_DAYS <= hi]
            for index, (offset, coverage) in enumerate(windows):
                inside = [c - offset for c in centers if 0 <= c - offset < config.WINDOW_SLOTS]
                if not inside:
                    continue
                stop = offset + config.WINDOW_SLOTS
                flux_w = np.zeros(config.WINDOW_SLOTS, np.float32)
                obs_w = np.zeros(config.WINDOW_SLOTS, bool)
                n = min(stop, flux_z.size) - offset
                flux_w[:n] = flux_z[offset:offset + n]
                obs_w[:n] = observed[offset:offset + n]
                events = []
                for c in inside:
                    snr, frac, depth = measure_event(flux_w, obs_w, c, duration_h * 2.0)
                    events.append({"center_slot": c, "duration_slots": duration_h * 2.0, "kind": "benchmark",
                                   "ref": target["tce_id"], "cls": target["cls"], "period_d": period, "snr": snr,
                                   "observed_fraction": frac, "depth_sigma": depth})
                out.append({**base, "window_index": index, "start_offset": offset, "coverage": coverage,
                            "benchmark": target["benchmark"], "tce_id": target["tce_id"],
                            "label": target["label"], "events": events})
        return out


def scan_file(files, catalog: EventCatalog, bench: BenchmarkIndex) -> dict[str, list[dict]]:
    out = {"events": [], "hosts": [], "cutouts": [], "bench": []}
    for f in files:
        for first, table in iter_row_groups(f.local, READ_COLUMNS):
            for i, row in enumerate(table.to_pylist()):
                row_index = first + i
                star = row["star_key"]
                pool = pool_for(star, row["split"], catalog.list_sets)
                base = {
                    "repo": f.repo, "path": f.path, "row": row_index, "segment_id": row["segment_id"],
                    "star_key": star, "split": row["split"], "pool": pool, "source": row["source"],
                    "sigma_p2p_ppm": row["sigma_p2p_ppm"], "quality_tier": row["quality_tier"],
                    "radius": row["radius"], "teff": row["teff"],
                }
                windows = list(zip(row["window_offsets"] or [], row["window_coverage"] or []))
                out["bench"] += bench.windows(row, base, windows)
                if pool == config.POOL_UNUSED:
                    continue
                if star in catalog.tagged_stars:
                    if row["known_events"]:
                        scan_events(row, base, windows, catalog, out)
                else:
                    sample_hosts(row, base, windows, pool, out)
    return out


def scan_events(row: dict, base: dict, windows: list, catalog: EventCatalog, out: dict) -> None:
    flux_z = np.frombuffer(row["flux_z"], dtype="<f2").astype(np.float32)
    observed = (np.frombuffer(row["flags"], dtype=np.uint8) & pretrain.FLAG_OBSERVED) > 0
    s = row["sigma_p2p_ppm"] * 1e-6
    events = []
    for e in row["known_events"]:
        info = catalog.lookup(e["kind"], e["ref"])
        events.append({**e, "cls": info.cls, "period_d": info.period_d,
                       "duration_slots": duration_or_default(e["duration_slots"])})

    for index, (offset, coverage) in enumerate(windows):
        stop = offset + config.WINDOW_SLOTS
        flux_w = np.zeros(config.WINDOW_SLOTS, np.float32)
        obs_w = np.zeros(config.WINDOW_SLOTS, bool)
        n = min(stop, flux_z.size) - offset
        flux_w[:n] = flux_z[offset:offset + n]
        obs_w[:n] = observed[offset:offset + n]
        inside = []
        for e in events:
            center = e["center_slot"] - offset
            if not 0 <= center < config.WINDOW_SLOTS:
                continue
            snr, frac, depth = measure_event(flux_w, obs_w, center, e["duration_slots"])
            inside.append({"center_slot": center, "duration_slots": e["duration_slots"], "kind": e["kind"],
                           "ref": e["ref"], "cls": e["cls"], "period_d": e["period_d"], "snr": snr,
                           "observed_fraction": frac, "depth_sigma": depth})
        if inside:
            out["events"].append({**base, "window_index": index, "start_offset": offset, "coverage": coverage,
                                  "planet_star": base["star_key"] in catalog.planet_stars, "events": inside})

    if base["pool"] not in config.TEMPLATE_POOLS.values():
        return
    per_ref: dict[str, int] = {}
    for e in events:
        if e["cls"] != "planet" or e["period_d"] is None or not math.isfinite(e["period_d"]):
            continue
        if per_ref.get(e["ref"], 0) >= config.TEMPLATE_MAX_CUTOUTS_PER_SEGMENT:
            continue
        half = e["duration_slots"] / 2.0
        reach = half + max(8.0, half)
        crowded = any(o is not e and abs(o["center_slot"] - e["center_slot"]) < reach + o["duration_slots"] / 2.0
                      for o in events)
        if crowded:
            continue
        snr, frac, _ = measure_event(flux_z, observed, e["center_slot"], e["duration_slots"])
        if frac < 0.8 or snr < config.TEMPLATE_MIN_CUTOUT_SNR:
            continue
        piece = cutout(flux_z, observed, s, e["center_slot"], e["duration_slots"])
        if piece is None:
            continue
        per_ref[e["ref"]] = per_ref.get(e["ref"], 0) + 1
        out["cutouts"].append({
            "ref": e["ref"], "star_key": base["star_key"], "pool": base["pool"], "source": base["source"],
            "segment_id": base["segment_id"], "period_d": e["period_d"],
            "duration_h": e["duration_slots"] / 2.0, "snr": snr, "dt_h": piece[0], "rel": piece[1],
        })


def sample_hosts(row: dict, base: dict, windows: list, pool: str, out: dict) -> None:
    rate = config.HOST_SAMPLE_RATE.get(pool, {}).get(base["source"], 0.0)
    if rate <= 0:
        return
    for index, (offset, coverage) in enumerate(windows):
        if coverage < config.HOST_MIN_COVERAGE:
            continue
        if unit_hash(config.SEED, row["segment_id"], index) < rate:
            out["hosts"].append({**base, "window_index": index, "start_offset": offset, "coverage": coverage})


def write(records: list[dict], schema: pa.Schema, path: Path) -> int:
    columns = {name: [r.get(name) for r in records] for name in schema.names}
    table = pa.table({name: pa.array(columns[name], type=schema.field(name).type) for name in schema.names})
    pq.write_table(table, path, compression="zstd")
    return table.num_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, choices=["ffi", "2min", "kepler"])
    parser.add_argument("--stem", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--catalog-revision", required=True)
    parser.add_argument("--benchmarks", type=Path, help="benchmark_targets.parquet from benchmarks.py")
    parser.add_argument("--local-root", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    started = time.time()
    catalog = EventCatalog(ReferenceCatalogs.load(args.catalog_revision))
    bench = BenchmarkIndex(pd.read_parquet(args.benchmarks) if args.benchmarks else None)
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        files = fetch(args.source, args.stem, args.local_root, Path(tmp))
        LOG.info("Scanning %s", [f.key for f in files])
        found = scan_file(files, catalog, bench)
    prefix = args.out / f"{args.source}_{args.stem}"
    counts = {
        "events": write(found["events"], EVENTS_SCHEMA, Path(f"{prefix}_events.parquet")),
        "hosts": write(found["hosts"], HOSTS_SCHEMA, Path(f"{prefix}_hosts.parquet")),
        "cutouts": write(found["cutouts"], CUTOUTS_SCHEMA, Path(f"{prefix}_cutouts.parquet")),
        "bench": write(found["bench"], BENCH_SCHEMA, Path(f"{prefix}_bench.parquet")),
    }
    LOG.info("%s %s: %s in %.0f s", args.source, args.stem, counts, time.time() - started)
    print(f"::notice title=scan {args.source} {args.stem}::{counts}")


if __name__ == "__main__":
    main()
