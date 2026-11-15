# MlClub PlanetFinders

NYU ML Club project: a transformer that searches TESS full-frame-image light curves for new transiting
planets, focusing on long-period planets that transit only once per TESS cycle (single transits and
duotransits).

## Data

All light curves are on Hugging Face under [nyu-planet-finders](https://huggingface.co/nyu-planet-finders):

| Dataset | Contents |
|---|---|
| [tess-spoc-ffi-binned](https://huggingface.co/datasets/nyu-planet-finders/tess-spoc-ffi-binned) | TESS-SPOC FFI light curves, sectors 1-81, 30-minute bins |
| [tess-spoc-2min-binned](https://huggingface.co/datasets/nyu-planet-finders/tess-spoc-2min-binned) | SPOC 2-minute light curves, sectors 1-107, 30-minute bins |
| [kepler-llc](https://huggingface.co/datasets/nyu-planet-finders/kepler-llc) | Kepler long-cadence light curves, one file per quarter |
| [reference-catalogs](https://huggingface.co/datasets/nyu-planet-finders/reference-catalogs) | TOI ephemerides, TESS and Kepler EB catalogs, Hawthorn et al. 2024 duotransits, KIC-TIC crossmatch |

## Repository

| Path | What it is |
|---|---|
| [`docs/project_handoff.md`](docs/project_handoff.md) | Project status, decisions and open items |
| [`docs/pretraining_dataset_build_guide.md`](docs/pretraining_dataset_build_guide.md) | Step-by-step design of the pretraining dataset, with options and recommendations |
| [`reference_catalogs/`](reference_catalogs/README.md) | Step 0 of the build: downloads and publishes the reference catalogs |
| [`.github/workflows/build-reference-catalogs.yml`](.github/workflows/build-reference-catalogs.yml) | Runs Step 0 in GitHub Actions |
| [`tests/`](tests) | Offline tests: `python -m pytest tests -q` |
| `tessLightCurveData/`, `kplr001430207-2012179063303_llc.csv` | Early sample light curves |

## Running Step 0

1. Add a repository secret `HF_TOKEN` with write access to the `nyu-planet-finders` organization.
2. Actions tab > "Build reference catalogs" > Run workflow (it also runs on pushes to `reference_catalogs/`).
3. Check the run's annotations for row counts, then the dataset card on Hugging Face.

To run it on your own machine or in TIKE instead, see [`reference_catalogs/README.md`](reference_catalogs/README.md).
