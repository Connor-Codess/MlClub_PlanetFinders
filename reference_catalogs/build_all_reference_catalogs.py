"""Build every reference catalog and write build_manifest.json.

What it does
    Runs the five builders in this package one after another:

        toi_ephemerides              fetch_toi_ephemerides.py
        tess_eb_catalog_prsa2022     fetch_tess_eb_catalog.py
        kepler_eb_catalog_kirk2016   fetch_kepler_eb_catalog.py
        duotransits_hawthorn2024     fetch_hawthorn2024_duotransits.py
        kic_tic_crossmatch           build_kic_tic_crossmatch.py

    A failure in one builder does not stop the others. Every outcome (rows,
    notes, or the error message) is recorded in build_manifest.json in the
    output directory, which upload_reference_catalogs_to_hub.py reads to write
    the dataset card.

Exit status
    0 when every builder succeeded. With --allow-partial, 0 when at least one
    succeeded, so a CI job can still upload the finished files before failing
    in a later step. 1 otherwise.

Run
    python -m reference_catalogs.build_all_reference_catalogs --out build/reference_catalogs
    python -m reference_catalogs.build_all_reference_catalogs --only toi_ephemerides duotransits_hawthorn2024
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

from . import (
    build_kic_tic_crossmatch,
    fetch_hawthorn2024_duotransits,
    fetch_kepler_eb_catalog,
    fetch_tess_eb_catalog,
    fetch_toi_ephemerides,
)
from .common import LOG, CatalogResult, configure_logging, github_annotation, utc_now

MANIFEST_NAME = "build_manifest.json"

# Wall-clock limit per builder, in seconds. A stalled remote service fails that
# one catalog instead of holding up the whole run.
TIME_LIMITS_S = {
    fetch_toi_ephemerides.NAME: 15 * 60,
    fetch_tess_eb_catalog.NAME: 15 * 60,
    fetch_kepler_eb_catalog.NAME: 15 * 60,
    fetch_hawthorn2024_duotransits.NAME: 10 * 60,
    build_kic_tic_crossmatch.NAME: 60 * 60,
}


class BuilderTimeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise BuilderTimeout("time limit reached")

BUILDERS: dict[str, Callable[[Path], CatalogResult]] = {
    fetch_toi_ephemerides.NAME: fetch_toi_ephemerides.build,
    fetch_tess_eb_catalog.NAME: fetch_tess_eb_catalog.build,
    fetch_kepler_eb_catalog.NAME: fetch_kepler_eb_catalog.build,
    fetch_hawthorn2024_duotransits.NAME: fetch_hawthorn2024_duotransits.build,
    build_kic_tic_crossmatch.NAME: build_kic_tic_crossmatch.build,
}


def run(out_dir: Path, only: list[str] | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = only or list(BUILDERS)
    unknown = sorted(set(selected) - set(BUILDERS))
    if unknown:
        raise SystemExit(f"Unknown catalog names: {unknown}. Choose from {list(BUILDERS)}")

    manifest = {"started_utc": utc_now(), "catalogs": {}, "failed": {}}
    for name in selected:
        LOG.info("=== Building %s ===", name)
        limit = TIME_LIMITS_S.get(name, 30 * 60)
        started = time.monotonic()
        use_alarm = hasattr(signal, "SIGALRM")
        if use_alarm:
            signal.signal(signal.SIGALRM, _raise_timeout)
            signal.alarm(limit)
        try:
            result = BUILDERS[name](out_dir)
            elapsed = time.monotonic() - started
            result.notes["build_seconds"] = round(elapsed, 1)
            manifest["catalogs"][name] = result.to_dict()
            github_annotation("notice", name, f"{result.rows:,} rows -> {result.filename}; {json.dumps(result.notes)}")
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            elapsed = time.monotonic() - started
            message = f"{type(exc).__name__}: {exc} (after {elapsed:.0f} s, limit {limit} s)"
            LOG.error("%s failed: %s", name, message)
            traceback.print_exc()
            manifest["failed"][name] = message
            github_annotation("error", name, message)
        finally:
            if use_alarm:
                signal.alarm(0)
    manifest["finished_utc"] = utc_now()

    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, default=str))
    LOG.info("Wrote %s: %d built, %d failed", path, len(manifest["catalogs"]), len(manifest["failed"]))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("build/reference_catalogs"))
    parser.add_argument("--only", nargs="+", default=None, help="build only these catalogs")
    parser.add_argument("--allow-partial", action="store_true",
                        help="exit 0 if at least one catalog was built")
    args = parser.parse_args()
    configure_logging()
    manifest = run(args.out, args.only)
    if not manifest["failed"]:
        sys.exit(0)
    sys.exit(0 if args.allow_partial and manifest["catalogs"] else 1)


if __name__ == "__main__":
    main()
