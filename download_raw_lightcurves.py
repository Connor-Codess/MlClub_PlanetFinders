from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd
from astropy.table import Table, unique, vstack
from astroquery.mast import Observations


PROJECT = Path(__file__).resolve().parents[1]
CATALOG = PROJECT / "data/raw/dr24_tce.csv"
MANIFEST_DIR = PROJECT / "data/manifests"
DOWNLOAD_DIR = PROJECT / "data/raw/kepler_lightcurves"

ALLOWED_LABELS = {"PC", "AFP", "NTP"}
QUERY_BATCH_SIZE = 100
DOWNLOAD_BATCH_SIZE = 500


def labeled_star_ids(limit=None):
    catalog = pd.read_csv(CATALOG)
    labeled = catalog[catalog["av_training_set"].isin(ALLOWED_LABELS)]
    kepids = sorted(labeled["kepid"].astype(int).unique())

    if limit is not None:
        kepids = kepids[:limit]

    return kepids


def query_batch(kepids):
    target_names = [f"kplr{kepid:09d}" for kepid in kepids]

    # Retry transient MAST failures.
    for attempt in range(3):
        try:
            observations = Observations.query_criteria(
                obs_collection="Kepler",
                target_name=target_names,
                dataproduct_type="timeseries",
                dataRights="PUBLIC",
            )
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))

    # Exclude short-cadence observations.
    obs_ids = np.asarray(observations["obs_id"], dtype=str)
    observations = observations[np.char.find(obs_ids, "_lc_") >= 0]

    products = Observations.get_unique_product_list(observations)
    products = Observations.filter_products(
        products,
        productSubGroupDescription="LLC",
        extension="fits",
    )

    filenames = np.asarray(products["productFilename"], dtype=str)
    return products[np.char.endswith(filenames, "_llc.fits")]


def build_manifest(kepids, manifest_path):
    tables = []

    for start in range(0, len(kepids), QUERY_BATCH_SIZE):
        batch = kepids[start : start + QUERY_BATCH_SIZE]
        products = query_batch(batch)

        if len(products):
            tables.append(products)

        print(
            f"Queried {min(start + len(batch), len(kepids)):,}"
            f"/{len(kepids):,} stars; found {len(products):,} files"
        )

    if not tables:
        raise RuntimeError("MAST returned no long-cadence FITS products")

    manifest = vstack(tables, metadata_conflicts="silent")
    manifest = unique(manifest, keys="dataURI")

    manifest.write(manifest_path, format="ascii.ecsv", overwrite=True)
    manifest.write(
        manifest_path.with_suffix(".csv"),
        format="ascii.csv",
        overwrite=True,
    )

    return manifest


def download(manifest, tag):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Observations.enable_cloud_dataset(provider="AWS")

    for start in range(0, len(manifest), DOWNLOAD_BATCH_SIZE):
        batch = manifest[start : start + DOWNLOAD_BATCH_SIZE]

        status = Observations.download_products(
            batch,
            download_dir=str(DOWNLOAD_DIR),
            cache=True,
            mrp_only=False,
        )

        status.write(
            MANIFEST_DIR / f"download_status_{tag}_{start:06d}.csv",
            format="ascii.csv",
            overwrite=True,
        )

        print(f"Processed files {start:,}–{start + len(batch) - 1:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        help="Restrict the run to the first N labeled stars",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download after building/loading the manifest",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild an existing manifest",
    )
    args = parser.parse_args()

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    tag = f"pilot_{args.limit}" if args.limit else "full"
    manifest_path = MANIFEST_DIR / f"dr24_llc_{tag}.ecsv"
    kepids = labeled_star_ids(args.limit)

    if manifest_path.exists() and not args.refresh:
        manifest = Table.read(manifest_path)
        print(f"Loaded existing manifest: {manifest_path}")
    else:
        manifest = build_manifest(kepids, manifest_path)

    total_bytes = sum(int(size) for size in manifest["size"])

    print(f"Stars: {len(kepids):,}")
    print(f"FITS files: {len(manifest):,}")
    print(f"Download: {total_bytes / 1_000_000_000:.2f} GB")
    print(f"Manifest: {manifest_path}")

    if args.download:
        download(manifest, tag)
    else:
        print("Manifest only; no FITS files were downloaded.")


if __name__ == "__main__":
    main()