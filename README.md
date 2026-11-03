# ML Club Planet Finders

Utilities for downloading and inspecting the Kepler Q1–Q17 DR24 TCE light
curves used by the planet-vetting project.

The repository includes the DR24 TCE catalog and one small CSV example. Raw
FITS light curves, generated manifests, and bulk CSV exports are intentionally
excluded from Git because the complete dataset is much too large for GitHub.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Build the full download manifest

This queries MAST and reports the exact number and size of the available files.
It does not download the FITS files.

```bash
python scripts/download_raw_lightcurves.py
```

## Download 20,000 FITS files at a time

First batch:

```bash
caffeinate -i python scripts/download_raw_lightcurves.py --download --download-start 0 --download-count 20000
```

For the next batch, change `--download-start` to `20000`, then `40000`, and so
on. Completed files are cached, so an interrupted batch can be resumed by
running the same command again. Closing a MacBook lid still puts it to sleep.

## Convert downloaded FITS files to compact CSVs

```bash
caffeinate -i python scripts/export_fits_csv.py data/raw/kepler_lightcurves
```

The converter is resumable and skips CSVs that already exist. FITS files remain
the authoritative copies because the compact CSV export omits FITS metadata,
aperture images, and less commonly used columns.
