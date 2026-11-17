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
| [lc-pretrain](https://huggingface.co/datasets/nyu-planet-finders/lc-pretrain) | Pretraining segments (train and validation stars) and the window index |
| lc-heldout (private) | The same format for held-out stars: search holdout, scoreboard hosts, list-excluded |
| [lc-finetune](https://huggingface.co/datasets/nyu-planet-finders/lc-finetune) | Labeled 27-day windows for fine-tuning: real, injected and transplanted transits, negatives |
| lc-eval (private) | Labeled validation and test windows from held-out stars only |
| lc-benchmarks (private) | AstroNet, Astronet-Triage, Astronet-Triage-v2 test sets and Salinas et al. 2025 candidates on our light curves |

## Repository

| Path | What it is |
|---|---|
| [`docs/project_handoff.md`](docs/project_handoff.md) | Project status, decisions and open items |
| [`docs/pretraining_dataset_build_guide.md`](docs/pretraining_dataset_build_guide.md) | Step-by-step design of the pretraining dataset, with options and recommendations |
| [`reference_catalogs/`](reference_catalogs/README.md) | Step 0 of the build: downloads and publishes the reference catalogs |
| [`.github/workflows/build-reference-catalogs.yml`](.github/workflows/build-reference-catalogs.yml) | Runs Step 0 in GitHub Actions |
| [`pretrain_dataset/`](pretrain_dataset/README.md) | Steps 1-11: builds lc-pretrain and lc-heldout, plus a reference window loader |
| [`.github/workflows/build-pretrain-dataset.yml`](.github/workflows/build-pretrain-dataset.yml) | Runs Steps 1-11 in GitHub Actions, one job per source file |
| [`finetune_dataset/`](finetune_dataset/README.md) | Builds lc-finetune, lc-eval and lc-benchmarks from lc-pretrain and lc-heldout |
| [`.github/workflows/build-finetune-dataset.yml`](.github/workflows/build-finetune-dataset.yml) | Runs that build: plan, scan, choose, materialize, assemble |
| [`dataset_cards/`](dataset_cards/) | Hand-written cards of the source datasets, published by `publish-dataset-cards.yml` |
| [`docs/finetune_dataset_build_log.md`](docs/finetune_dataset_build_log.md) | Every decision, number and change of the fine-tuning build |
| [`experiments/finetune_mix/`](experiments/finetune_mix/README.md) | Data mix experiments: which lc-finetune rows train the best detector, scored on lc-eval val |
| [`.github/workflows/finetune-mix-experiments.yml`](.github/workflows/finetune-mix-experiments.yml) | Runs those experiments on GitHub's free CPU runners |
| [`tests/`](tests) | Offline tests: `python -m pytest tests -q` |
| `tessLightCurveData/`, `kplr001430207-2012179063303_llc.csv` | Early sample light curves |

## Running Step 0

1. Add a repository secret `HF_TOKEN` with write access to the `nyu-planet-finders` organization.
2. Actions tab > "Build reference catalogs" > Run workflow (it also runs on pushes to `reference_catalogs/`).
3. Check the run's annotations for row counts, then the dataset card on Hugging Face.

To run it on your own machine or in TIKE instead, see [`reference_catalogs/README.md`](reference_catalogs/README.md).

## Running Steps 1-11

Edit [`pretrain_dataset/build_request.json`](pretrain_dataset/build_request.json) and push, or run
"Build pretraining dataset" from the Actions tab. Already-processed source files are skipped. The
checks are written to `index/validation_report.md` in lc-pretrain. See
[`pretrain_dataset/README.md`](pretrain_dataset/README.md).

## Building the fine-tuning datasets

Edit [`finetune_dataset/build_request.json`](finetune_dataset/build_request.json) and push. `files` and `scale`
make a trial; `reuse_run_id` reruns only the final assembly on an earlier run's rows. See
[`finetune_dataset/README.md`](finetune_dataset/README.md) and the build log in `docs/`.

## Running the data mix experiments

Edit [`experiments/finetune_mix/run_request.json`](experiments/finetune_mix/run_request.json) and push, or run
"Fine-tuning data mix experiments" from the Actions tab. Each arm (a combination of lc-finetune rows) trains
the same small detector three times and is scored on lc-eval val; the run's `summary` artifact has the
comparison. See [`experiments/finetune_mix/README.md`](experiments/finetune_mix/README.md).
