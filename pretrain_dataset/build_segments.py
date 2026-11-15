"""Pass 1 of the lc-pretrain build: turn one source file into finished segment rows.

What it does (build guide Steps 1-7 for every light curve in one file)
    1. Streams one source Parquet file (an FFI sector, a 2-min sector or a
       Kepler quarter) from Hugging Face, one row group at a time, visiting the
       row groups in a seeded random order.
    2. For each light curve: resolves star_key, hash bucket and split
       (star_identity.py), builds the gridded segment (segments.py), attaches
       catalog tags and predicted known events (catalogs.py).
    3. Routes rows by split:
         train, validation          -> lc-pretrain   data/<source>/<split>/<file>.parquet
         search_holdout, scoreboard_hosts, list_excluded
                                    -> lc-heldout    data/<source>/<file>.parquet
    4. Writes a stats file (rows in, rows written per split, drop reasons,
       round-trip check) to index/pass1/<source>_<file>.json in both repos.
    5. With --upload, commits the files to Hugging Face (needs HF_TOKEN).

Output row schema: see SCHEMA below and pretrain_dataset/README.md.

Run
    python -m pretrain_dataset.build_segments --source ffi --file s0001.parquet --out build/pass1
    python -m pretrain_dataset.build_segments --source kepler --file q07.parquet --max-row-groups 2
    HF_TOKEN=... python -m pretrain_dataset.build_segments --source 2min --file s0013.parquet --upload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .catalogs import ReferenceCatalogs
from .segments import Dropped, build_segment, decode, quality_tier
from .star_identity import IdentityResolver

LOG = logging.getLogger("pretrain_dataset")
ROWS_PER_GROUP = 1000
ROUNDTRIP_EVERY = 500

EVENT_TYPE = pa.struct([
    ("center_slot", pa.float32()),
    ("duration_slots", pa.float32()),
    ("kind", pa.string()),
    ("ref", pa.string()),
])

SCHEMA = pa.schema([
    ("segment_id", pa.string()),
    ("star_key", pa.string()),
    ("tic_id", pa.int64()),
    ("kic_id", pa.int64()),
    ("source", pa.string()),
    ("sector", pa.int16()),
    ("quarter", pa.int16()),
    ("native_cadence_s", pa.int16()),
    ("quality_policy", pa.string()),
    ("slot0", pa.int32()),
    ("n_slots", pa.int32()),
    ("n_obs", pa.int32()),
    ("coverage", pa.float32()),
    ("flux_median", pa.float32()),
    ("sigma_p2p_ppm", pa.float32()),
    ("sigma_mad_ppm", pa.float32()),
    ("var_ratio", pa.float32()),
    ("sigma_6h_ppm", pa.float32()),
    ("bkg_sigma", pa.float32()),
    ("cen_x_median_px", pa.float32()),
    ("cen_y_median_px", pa.float32()),
    ("tmag", pa.float32()),
    ("kepmag", pa.float32()),
    ("teff", pa.float32()),
    ("radius", pa.float32()),
    ("flux_z", pa.binary()),
    ("flux_err_z", pa.binary()),
    ("cen_x", pa.binary()),
    ("cen_y", pa.binary()),
    ("bkg_z", pa.binary()),
    ("fill", pa.binary()),
    ("flags", pa.binary()),
    ("window_offsets", pa.list_(pa.int32())),
    ("window_coverage", pa.list_(pa.float32())),
    ("quality_tier", pa.int8()),
    ("hash_bucket", pa.int16()),
    ("split", pa.string()),
    ("toi_ids", pa.string()),
    ("toi_disposition", pa.string()),
    ("eb_catalog", pa.string()),
    ("kepler_tce_labels", pa.string()),
    ("hawthorn_candidate", pa.int16()),
    ("known_events", pa.list_(EVENT_TYPE)),
    ("dup_group", pa.string()),
    ("source_repo", pa.string()),
    ("source_file", pa.string()),
    ("source_row_group", pa.int32()),
    ("source_row", pa.int32()),
    ("build_version", pa.string()),
])

LIST_COLUMNS = ("time", "flux", "flux_err", "centr_x", "centr_y", "bkg", "n_points")


def file_number(filename: str) -> int:
    """'s0013.parquet' -> 13, 'q07.parquet' -> 7."""
    return int(Path(filename).stem.lstrip("sq"))


def _float(value) -> float | None:
    return None if value is None or (isinstance(value, float) and np.isnan(value)) else float(value)


class SplitWriters:
    """One Parquet writer per output file, opened lazily, flushed every ROWS_PER_GROUP rows."""

    def __init__(self, out_dir: Path, source: str, stem: str):
        self.paths = {
            "train": out_dir / "pretrain" / "data" / source / "train" / f"{stem}.parquet",
            "validation": out_dir / "pretrain" / "data" / source / "validation" / f"{stem}.parquet",
            "heldout": out_dir / "heldout" / "data" / source / f"{stem}.parquet",
        }
        self.writers: dict[str, pq.ParquetWriter] = {}
        self.buffers: dict[str, list[dict]] = {k: [] for k in self.paths}
        self.counts: Counter = Counter()

    def add(self, row: dict) -> None:
        target = row["split"] if row["split"] in config.PRETRAIN_SPLITS else "heldout"
        self.buffers[target].append(row)
        self.counts[row["split"]] += 1
        if len(self.buffers[target]) >= ROWS_PER_GROUP:
            self._flush(target)

    def _flush(self, target: str) -> None:
        rows = self.buffers[target]
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        if target not in self.writers:
            self.paths[target].parent.mkdir(parents=True, exist_ok=True)
            self.writers[target] = pq.ParquetWriter(self.paths[target], SCHEMA, compression="zstd")
        self.writers[target].write_table(table)
        self.buffers[target] = []

    def close(self) -> dict[str, Path]:
        for target in self.paths:
            self._flush(target)
        for writer in self.writers.values():
            writer.close()
        return {k: p for k, p in self.paths.items() if k in self.writers}


def iter_rows(table: pa.Table):
    """Yield one dict per row, with list columns as numpy arrays (no Python float lists)."""
    lists = {}
    for name in LIST_COLUMNS:
        if name in table.column_names:
            col = table.column(name).combine_chunks()
            lists[name] = (col.offsets.to_numpy(), col.values.to_numpy(zero_copy_only=False))
    scalars = {name: table.column(name).to_pylist() for name in table.column_names if name not in LIST_COLUMNS}
    for i in range(table.num_rows):
        row = {name: values[i] for name, values in scalars.items()}
        for name, (offsets, values) in lists.items():
            row[name] = values[offsets[i]:offsets[i + 1]]
        yield row


def roundtrip_error(row: dict, segment, source: str) -> float:
    """Largest |decoded - original| in sigma units over slots holding exactly one source bin.

    Only slots with |flux_z| <= 100 are compared: float16 keeps about three
    significant digits, so deep eclipses at thousands of sigma round by more
    than 0.1 sigma by design.
    """
    time = np.asarray(row["time"], float) + (config.BKJD_TO_BTJD if source == "kepler" else 0.0)
    flux = np.asarray(row["flux"], float)
    ok = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
    slots = np.floor((time[ok] - config.GRID_EPOCH_BTJD) / config.SLOT_DAYS).astype(np.int64) - segment.slot0
    inside = (slots >= 0) & (slots < segment.n_slots)
    slots, flux = slots[inside], flux[ok][inside]
    unique, counts = np.unique(slots, return_counts=True)
    single = set(unique[counts == 1].tolist())
    mask = np.array([s in single for s in slots])
    if not mask.any():
        return 0.0
    decoded = decode({k: segment.channels[k] for k in segment.channels})["flux_z"][slots[mask]]
    sigma = segment.stats["sigma_p2p_ppm"] * 1e-6 * segment.stats["flux_median"]
    moderate = np.abs(decoded) <= 100
    if not moderate.any():
        return 0.0
    rebuilt = decoded[moderate] * sigma + segment.stats["flux_median"]
    return float(np.max(np.abs(rebuilt - flux[mask][moderate])) / sigma)


def process_file(source: str, filename: str, out_dir: Path, catalogs: ReferenceCatalogs,
                 max_row_groups: int | None = None) -> dict:
    from huggingface_hub import HfFileSystem

    repo = config.SOURCE_REPOS[source]
    number = file_number(filename)
    stem = Path(filename).stem
    resolver = IdentityResolver(catalogs.kic_to_tic, catalogs.exclusion_map())
    writers = SplitWriters(out_dir, source, stem)
    dropped: Counter = Counter()
    seen_ids: Counter = Counter()
    roundtrip_max = 0.0
    tiers: Counter = Counter()
    fill_sum, fill_n = 0.0, 0
    events_total = 0
    rows_in = 0
    started = time.monotonic()

    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    with fs.open(f"datasets/{repo}/{filename}", "rb", block_size=16 * 2**20) as handle:
        parquet = pq.ParquetFile(handle)
        groups = list(range(parquet.num_row_groups))
        random.Random(int(hashlib.sha256(f"{source}/{filename}".encode()).hexdigest()[:8], 16)).shuffle(groups)
        if max_row_groups:
            groups = groups[:max_row_groups]
        for done, rg in enumerate(groups, start=1):
            table = parquet.read_row_group(rg)
            for row_in_group, row in enumerate(iter_rows(table)):
                rows_in += 1
                if source == "kepler":
                    identity = resolver.resolve(kic_id=row["kic_id"])
                    tic_id, kic_id, base_id = None, int(row["kic_id"]), int(row["kic_id"])
                    if identity.star_key.startswith("tic:"):
                        tic_id = int(identity.star_key[4:])
                else:
                    identity = resolver.resolve(tic_id=row["tic_id"])
                    tic_id, kic_id, base_id = int(row["tic_id"]), None, int(row["tic_id"])
                try:
                    segment = build_segment(row, source, number)
                except Dropped as reason:
                    dropped[str(reason)] += 1
                    continue
                except Exception as exc:  # a malformed row must not stop the file
                    dropped[f"error:{type(exc).__name__}"] += 1
                    LOG.warning("Row %d of row group %d failed: %s", row_in_group, rg, exc)
                    continue

                segment_id = f"{source}:{stem}:{base_id}"
                seen_ids[segment_id] += 1
                if seen_ids[segment_id] > 1:
                    segment_id = f"{segment_id}-{seen_ids[segment_id]}"

                coverage = segment.n_obs / segment.n_slots
                tier = quality_tier(coverage, segment.stats["sigma_p2p_ppm"])
                tiers[tier] += 1
                events = catalogs.events_for(identity.star_key, kic_id, source, segment.slot0, segment.n_slots)
                events_total += len(events)
                if rows_in % ROUNDTRIP_EVERY == 0:
                    roundtrip_max = max(roundtrip_max, roundtrip_error(row, segment, source))
                    decoded = decode(segment.channels)
                    observed = (decoded["flags"] & config.FLAG_OBSERVED) > 0
                    fill_sum += float(decoded["fill"][observed].sum())
                    fill_n += int(observed.sum())

                writers.add({
                    "segment_id": segment_id,
                    "star_key": identity.star_key,
                    "tic_id": tic_id,
                    "kic_id": kic_id,
                    "source": source,
                    "sector": number if source != "kepler" else None,
                    "quarter": number if source == "kepler" else None,
                    "native_cadence_s": config.native_cadence_s(source, number),
                    "quality_policy": config.quality_policy(source),
                    "slot0": segment.slot0,
                    "n_slots": segment.n_slots,
                    "n_obs": segment.n_obs,
                    "coverage": coverage,
                    **segment.stats,
                    "tmag": _float(row.get("tmag")),
                    "kepmag": _float(row.get("kepmag")),
                    "teff": _float(row.get("teff")),
                    "radius": _float(row.get("radius")),
                    **segment.channels,
                    "window_offsets": segment.window_offsets,
                    "window_coverage": segment.window_coverage,
                    "quality_tier": tier,
                    "hash_bucket": identity.hash_bucket,
                    "split": identity.split,
                    **catalogs.tags_for(identity.star_key),
                    "known_events": events,
                    "dup_group": f"{identity.star_key}:s{number}" if source != "kepler" else None,
                    "source_repo": repo,
                    "source_file": filename,
                    "source_row_group": rg,
                    "source_row": row_in_group,
                    "build_version": config.BUILD_VERSION,
                })
            LOG.info("%s %s: row group %d/%d done, %d rows so far", source, filename, done, len(groups), rows_in)

    paths = writers.close()
    written = sum(writers.counts.values())
    stats = {
        "source": source,
        "file": filename,
        "build_version": config.BUILD_VERSION,
        "reference_catalogs_revision": catalogs.revision,
        "row_groups_processed": len(groups),
        "rows_in": rows_in,
        "rows_written": written,
        "rows_dropped": sum(dropped.values()),
        "written_by_split": dict(writers.counts),
        "dropped_by_reason": dict(dropped),
        "quality_tiers": {str(k): v for k, v in sorted(tiers.items())},
        "known_events": events_total,
        "roundtrip_max_error_sigma": round(roundtrip_max, 4),
        "mean_fill_sampled": round(fill_sum / fill_n, 4) if fill_n else None,
        "seconds": round(time.monotonic() - started, 1),
        "outputs": {k: str(p.relative_to(out_dir)) for k, p in paths.items()},
    }
    if rows_in != written + stats["rows_dropped"]:
        raise RuntimeError(f"Row accounting mismatch: {stats}")
    stats_path = out_dir / "index" / "pass1" / f"{source}_{stem}.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2))
    return stats


def upload(out_dir: Path, stats: dict, attempts: int = 12) -> None:
    """Commit the outputs of one file to lc-pretrain and lc-heldout, retrying on rate limits."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    stats_rel = f"index/pass1/{stats['source']}_{Path(stats['file']).stem}.json"
    plans = {
        config.PRETRAIN_REPO: ([stats["outputs"][k] for k in ("train", "validation") if k in stats["outputs"]], False),
        config.HELDOUT_REPO: ([stats["outputs"]["heldout"]] if "heldout" in stats["outputs"] else [], True),
    }
    for repo, (files, private) in plans.items():
        api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
        prefix = "pretrain/" if repo == config.PRETRAIN_REPO else "heldout/"
        operations = [CommitOperationAdd(path_in_repo=f[len(prefix):], path_or_fileobj=str(out_dir / f)) for f in files]
        operations.append(CommitOperationAdd(path_in_repo=stats_rel, path_or_fileobj=str(out_dir / stats_rel)))
        for attempt in range(1, attempts + 1):
            try:
                api.create_commit(repo, operations, repo_type="dataset",
                                  commit_message=f"{config.BUILD_VERSION}: {stats['source']} {stats['file']}")
                LOG.info("Committed %d files to %s", len(operations), repo)
                break
            except Exception as exc:
                if attempt == attempts:
                    raise
                wait = min(60 * attempt, 600)
                LOG.warning("Commit to %s failed (%s); retrying in %d s", repo, exc, wait)
                time.sleep(wait)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pass 1: build segments for one source file")
    parser.add_argument("--source", choices=sorted(config.SOURCE_REPOS), required=True)
    parser.add_argument("--file", required=True, help="e.g. s0001.parquet or q07.parquet")
    parser.add_argument("--out", type=Path, default=Path("build/pass1"))
    parser.add_argument("--catalog-revision", default=None)
    parser.add_argument("--max-row-groups", type=int, default=None, help="trial runs: stop early")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
    for noisy in ("httpx", "httpx2", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    catalogs = ReferenceCatalogs.load(args.catalog_revision)
    stats = process_file(args.source, args.file, args.out, catalogs, args.max_row_groups)
    print(f"::notice title={args.source} {args.file}::" + json.dumps({
        k: stats[k] for k in ("rows_in", "rows_written", "written_by_split", "dropped_by_reason",
                              "quality_tiers", "roundtrip_max_error_sigma", "seconds")}), flush=True)
    if args.upload:
        upload(args.out, stats)


if __name__ == "__main__":
    main()
