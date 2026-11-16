---
license: other
pretty_name: Kepler long-cadence light curves, all stars, Q0-Q17
tags:
- astronomy
- exoplanets
- kepler
- light-curves
- time-series
size_categories:
- 1M<n<10M
configs:
- config_name: light_curves
  data_files: "q*.parquet"
  default: true
- config_name: manifest
  data_files: "manifest.parquet"
---

# Kepler long-cadence light curves, all stars, Q0-Q17

Every Kepler long-cadence (29.4-minute) light curve of the prime mission, one row per star per quarter, in the
same layout as the nyu-planet-finders TESS datasets.

- One Parquet file per quarter (`q00.parquet` ... `q17.parquet`), 195 GB in total
- 2,869,929 light curves of 207,617 stars
- `manifest.parquet`: one row per original FITS file (2,871,321 files): `key` (path in the MAST bucket),
  `kic_id`, `timestamp`, `quarter`

## Columns

| Column | Type | Description |
|---|---|---|
| `kic_id` | int64 | Kepler Input Catalog ID |
| `quarter` | int16 | Kepler observing quarter (0-17) |
| `kepmag` | float64 | Kepler magnitude (from the light-curve header) |
| `teff` | float64 | Stellar effective temperature [K]; may be missing |
| `radius` | float64 | Stellar radius [R_sun]; may be missing |
| `time` | list[float64] | Cadence mid-times, BKJD (BJD - 2454833). BTJD = BKJD - 2167 |
| `flux` | list[float32] | PDCSAP flux divided by its median |
| `flux_err` | list[float32] | Uncertainty, same units |
| `centr_x`, `centr_y` | list[float32] | Flux-weighted centroid minus its median [pixels, 3.98 arcsec] |

A full quarter has about 4,000 cadences (Q0 is 10 days, Q1 one month, Q17 about 32 days).

## Processing

1. Read every `*_llc.fits` file from the MAST public S3 bucket (`s3://stpubdata/kepler/public/lightcurves/`).
2. Kept cadences with `SAP_QUALITY == 0` and finite time, PDCSAP flux and centroids.
3. Divided flux and flux error by the median PDCSAP flux; subtracted the median centroid.
4. No binning: the native 29.4-minute cadence is kept, so the data line up with the 30-minute TESS products.

Kepler rotated by 90 degrees every quarter, so the centroid axes change meaning between quarters; work per
quarter. 1,392 manifest entries have no row in the quarter files.

## Usage

```python
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

fs = HfFileSystem()
with fs.open("datasets/nyu-planet-finders/kepler-llc/q05.parquet", "rb") as handle:
    table = pq.ParquetFile(handle).read_row_group(0)    # read one row group at a time: files are up to 13 GB
```

## Related datasets

[kepler-raw-lightcurves](https://huggingface.co/datasets/nyu-planet-finders/kepler-raw-lightcurves) holds the
unbinned, unfiltered DR25 light curves and the DR24 TCE table for the 9,865 TCE training stars;
[lc-pretrain](https://huggingface.co/datasets/nyu-planet-finders/lc-pretrain) puts these light curves on the
common 30-minute grid with the TESS data.

## Source and acknowledgment

Derived from Kepler mission data obtained from the Mikulski Archive for Space Telescopes (MAST) at the Space
Telescope Science Institute. Funding for the Kepler mission is provided by NASA's Science Mission Directorate.
See the [MAST data use policy](https://archive.stsci.edu/publishing/data-use).
