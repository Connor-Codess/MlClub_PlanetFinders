---
license: cc-by-4.0
pretty_name: TESS-SPOC full-frame-image light curves, binned to 30 minutes
tags:
- astronomy
- exoplanets
- tess
- light-curves
- time-series
size_categories:
- 10M<n<100M
configs:
- config_name: default
  data_files:
  - split: train
    path: "s*.parquet"
---

# TESS-SPOC full-frame-image light curves, binned to 30 minutes

Every TESS-SPOC full-frame-image (FFI) light curve of Sectors 1-81, reduced to 30-minute bins for machine-learning
transit searches. This is the data the nyu-planet-finders detector searches.

- One Parquet file per sector (`s0001.parquet` ... `s0081.parquet`)
- One row per star per sector: 12,797,720 rows, about 160,000 stars per sector, 252.8 GB
- `split: train` only means "all rows"; this repository is not divided into training and test sets

## Columns

| Column | Type | Description |
|---|---|---|
| `tic_id` | int64 | TESS Input Catalog ID |
| `sector` | int16 | TESS sector |
| `tmag` | float64 | TESS magnitude (from the light-curve header) |
| `teff` | float64 | Stellar effective temperature [K]; may be missing |
| `radius` | float64 | Stellar radius [R_sun]; may be missing |
| `time` | list[float64] | Bin centers, BTJD (BJD - 2457000) |
| `flux` | list[float32] | PDCSAP flux divided by its median, averaged into 30-minute bins |
| `flux_err` | list[float32] | Uncertainty of each binned flux value, same units |
| `centr_x`, `centr_y` | list[float32] | Flux-weighted centroid minus its median [pixels, about 21 arcsec]; NaN in a bin with no valid centroid |
| `bkg` | list[float32] | SAP background divided by the star's median flux, binned like the flux |
| `n_points` | list[int16] | Number of original cadences averaged into each bin |

The list columns of a row are aligned element by element; their length is the number of non-empty bins (about
950-1,300 per sector).

## Processing

1. Read every `*_tess_v1_lc.fits` file of the TESS-SPOC high-level science product from the MAST public S3
   bucket (`s3://stpubdata/mast/hlsp/tess-spoc/`).
2. Removed cadences flagged by lightkurve's default TESS quality bitmask (17087), and cadences without a
   finite time or PDCSAP flux. A cadence with good flux but no centroid is kept.
3. Divided flux, flux error and background by the median PDCSAP flux; subtracted the median centroid.
4. Averaged into 30-minute bins. Empty bins are omitted, so gaps appear as missing time.

The FFI cadence changed during the mission, so `n_points` is about 1 for Sectors 1-26 (30-minute images),
about 3 for Sectors 27-55 (10-minute) and about 9 for Sectors 56-81 (200-second).

Differences from [tess-spoc-2min-binned](https://huggingface.co/datasets/nyu-planet-finders/tess-spoc-2min-binned):
the 2-minute data dropped every flagged cadence (QUALITY == 0) and every cadence without a centroid, and has no
`bkg` or `n_points` columns. The original quality flags are not stored.

## Usage

```python
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

fs = HfFileSystem()
with fs.open("datasets/nyu-planet-finders/tess-spoc-ffi-binned/s0040.parquet", "rb") as handle:
    table = pq.ParquetFile(handle).read_row_group(0)    # about 2,000 stars; read one row group at a time
row = table.slice(0, 1).to_pylist()[0]
```

Each file is 2-4 GB. Read row groups rather than whole files on machines with little memory. The Hugging Face
viewer shows only the first few hundred elements of each list.

## Derived datasets

[lc-pretrain](https://huggingface.co/datasets/nyu-planet-finders/lc-pretrain) puts these light curves, with the
2-minute and Kepler data, on one global 30-minute grid with noise normalization, quality flags and splits by
star; [lc-finetune](https://huggingface.co/datasets/nyu-planet-finders/lc-finetune) adds transit labels.

## Source and acknowledgment

TESS-SPOC FFI light curves (Caldwell et al. 2020, RNAAS 4, 201; doi:10.17909/t9-wpz1-8s54), produced by the
TESS Science Processing Operations Center at NASA Ames and obtained from the Mikulski Archive for Space
Telescopes (MAST) at the Space Telescope Science Institute. The TESS-SPOC products are released under CC BY 4.0.
This dataset includes data collected by the TESS mission; funding for the TESS mission is provided by NASA's
Science Mission Directorate.
