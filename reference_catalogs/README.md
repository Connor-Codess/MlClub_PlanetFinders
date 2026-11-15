# reference_catalogs

Step 0 of the pretraining dataset build: download and standardize the external catalogs that the
build needs for splits, exclusions and tags, and publish them as one Hugging Face dataset,
[`nyu-planet-finders/reference-catalogs`](https://huggingface.co/datasets/nyu-planet-finders/reference-catalogs).

These catalogs are **never model inputs or pretraining targets**. They decide which stars are held out
for testing, which segments get an `eb_catalog` or `toi` tag, and where known transits and eclipses fall.

## Files in this folder

| File | What it does | Output |
|---|---|---|
| `common.py` | Shared helpers: HTTP retries, BJD/BKJD/BTJD conversion, snake_case column names, Parquet writing with provenance in the file footer | none |
| `fetch_toi_ephemerides.py` | Queries the NASA Exoplanet Archive TOI table for ephemerides and stellar parameters; flags TOIs with P > 20 d or no period | `toi_ephemerides.parquet` |
| `fetch_tess_eb_catalog.py` | Downloads the TESS EB catalog (Prsa et al. 2022, VizieR J/ApJS/258/16) | `tess_eb_catalog_prsa2022.parquet` |
| `fetch_kepler_eb_catalog.py` | Downloads the Kepler EB catalog (Kirk et al. 2016, VizieR J/AJ/151/68), converting epochs to BTJD | `kepler_eb_catalog_kirk2016.parquet` |
| `fetch_hawthorn2024_duotransits.py` | Parses the 85 duotransit candidates from appendix Tables A1-A3 of arXiv:2310.17268 | `duotransits_hawthorn2024.parquet` |
| `build_kic_tic_crossmatch.py` | Reads every kic_id from the kepler-llc manifest and finds its TIC ID in TIC v8.2 at MAST | `kic_tic_crossmatch.parquet` |
| `build_all_reference_catalogs.py` | Runs all five builders, continues past failures, writes `build_manifest.json` | `build_manifest.json` |
| `upload_reference_catalogs_to_hub.py` | Writes the dataset card from the manifest and uploads everything to Hugging Face | `README.md` on the Hub |
| `requirements.txt` | Python dependencies | |

Every module has a docstring at the top that explains its inputs, outputs and columns in more detail.

## Conventions

- All times are **BTJD = BJD_TDB - 2457000**. Kepler epochs are converted from whatever convention the
  source uses; the detected convention is stored in the file.
- Every output keeps the standardized columns first (`tic_id`, `kic_id`, `period_d`, `t0_btjd`,
  `duration_h`, ...) and then the source's own columns with an `archive_` or `vizier_` prefix.
- VizieR tables are read from the asu-tsv export of the Harvard mirror first, because the CDS archive
  and TAP servers have been unresponsive at times; the CDS archive files and TAPVizieR are fallbacks.
- Every Parquet file stores its source, retrieval time and build notes as JSON in the file footer
  under the key `reference_catalog`:
  `json.loads(pyarrow.parquet.read_schema(path).metadata[b"reference_catalog"])`.

## How to run

### In GitHub Actions (recommended)

The workflow `.github/workflows/build-reference-catalogs.yml` runs on every push that touches this
folder, monthly, and on demand (Actions tab > "Build reference catalogs" > Run workflow).

1. Add a repository secret named `HF_TOKEN` (Settings > Secrets and variables > Actions) holding a
   Hugging Face token with write access to the `nyu-planet-finders` organization.
2. Run the workflow. The built files are attached to the run as an artifact even if the upload is
   skipped, and the run summary lists row counts for every catalog.

### Locally or in TIKE

Needs internet access to exoplanetarchive.ipac.caltech.edu, a VizieR mirror (vizier.cfa.harvard.edu is
tried first, then the CDS servers), arxiv.org, mast.stsci.edu and huggingface.co. A full build takes
about 2 minutes, most of it the MAST crossmatch.

```bash
python -m pip install -r reference_catalogs/requirements.txt
python -m reference_catalogs.build_all_reference_catalogs --out build/reference_catalogs
HF_TOKEN=hf_... python -m reference_catalogs.upload_reference_catalogs_to_hub --src build/reference_catalogs
```

Build a subset with `--only toi_ephemerides duotransits_hawthorn2024`. Each builder can also be run on
its own, for example `python -m reference_catalogs.fetch_toi_ephemerides --out build/reference_catalogs`.

### Tests

```bash
python -m pytest tests/test_reference_catalogs.py -q
```

The tests use small hand-made inputs and need no network access.

## Checks built into the scripts

| Catalog | Check | On failure |
|---|---|---|
| TOI | Required columns present in the archive table | error |
| TESS EBs | 4,584 rows | warning |
| Kepler EBs | at least 2,500 rows; epoch convention detected | error |
| Hawthorn duotransits | exactly 85 candidates, unique TIC IDs, Tc2 later than Tc1, TIC IDs agree across tables | error |
| KIC-TIC crossmatch | at least 50% of KIC IDs matched | error |
