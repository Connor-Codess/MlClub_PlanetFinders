from pathlib import Path
import argparse
import csv

import pandas as pd
from astropy.io import fits


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT / "data/inspection/raw_lightcurves"

LIGHT_CURVE_COLUMNS = [
    "TIME",
    "CADENCENO",
    "SAP_FLUX",
    "SAP_FLUX_ERR",
    "PDCSAP_FLUX",
    "PDCSAP_FLUX_ERR",
    "SAP_QUALITY",
]


def export_fits(fits_path, output_path):
    with fits.open(fits_path, memmap=False) as hdul:
        primary_header = hdul[0].header
        light_curve = hdul["LIGHTCURVE"].data

        missing = [name for name in LIGHT_CURVE_COLUMNS if name not in light_curve.names]
        if missing:
            raise ValueError(f"Missing expected FITS columns: {missing}")

        frame = pd.DataFrame(
            {name: light_curve[name] for name in LIGHT_CURVE_COLUMNS}
        )
        frame.insert(0, "QUARTER", primary_header.get("QUARTER"))
        frame.insert(0, "KEPLERID", primary_header.get("KEPLERID"))
        frame["QUALITY_OK"] = frame["SAP_QUALITY"] == 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)
    return len(frame)


def export_directory(input_dir, output_dir, limit=None, overwrite=False):
    fits_paths = sorted(input_dir.rglob("*_llc.fits"))
    if limit is not None:
        fits_paths = fits_paths[:limit]

    if not fits_paths:
        raise FileNotFoundError(f"No *_llc.fits files found under {input_dir}")

    converted = 0
    skipped = 0
    failures = []

    for index, fits_path in enumerate(fits_paths, start=1):
        relative_path = fits_path.relative_to(input_dir).with_suffix(".csv")
        output_path = output_dir / relative_path

        if output_path.exists() and not overwrite:
            skipped += 1
        else:
            try:
                export_fits(fits_path, output_path)
                converted += 1
            except Exception as error:
                failures.append((fits_path, str(error)))

        if index % 100 == 0 or index == len(fits_paths):
            print(
                f"Processed {index:,}/{len(fits_paths):,} files "
                f"(converted {converted:,}, skipped {skipped:,}, "
                f"failed {len(failures):,})"
            )

    if failures:
        failure_path = output_dir / "conversion_failures.csv"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_path.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["fits_path", "error"])
            writer.writerows(failures)
        print(f"Failure log: {failure_path}")

    return converted, skipped, len(failures)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export useful light-curve columns from one Kepler FITS file or "
            "recursively from a directory."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to an *_llc.fits file or a directory containing FITS files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV output path when converting one FITS file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output root for directory conversion",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Convert only the first N files (useful for testing)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace CSV files that already exist",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")

    input_path = args.input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_dir():
        if args.output is not None:
            parser.error("Use --output-dir, not --output, with a directory")

        output_dir = args.output_dir.expanduser().resolve()
        converted, skipped, failed = export_directory(
            input_path,
            output_dir,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        print(
            f"Finished: converted {converted:,}, skipped {skipped:,}, "
            f"failed {failed:,}"
        )
        if failed:
            raise SystemExit(1)
        return

    if args.limit is not None:
        parser.error("--limit is only valid with a directory")

    output_path = args.output
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / f"{input_path.stem}.csv"
    else:
        output_path = output_path.expanduser().resolve()

    if output_path.exists() and not args.overwrite:
        print(f"Skipped existing {output_path}")
        return

    row_count = export_fits(input_path, output_path)
    print(f"Wrote {output_path}")
    print(f"Rows: {row_count:,}")


if __name__ == "__main__":
    main()
