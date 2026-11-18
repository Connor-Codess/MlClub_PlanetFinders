# Project context for Claude Code

Read `docs/project_handoff.md` first, then `docs/pretraining_dataset_build_guide.md` and
`docs/finetune_dataset_build_log.md`. Together they hold
every decision made so far on this project: goal, data on Hugging Face, the pretraining dataset design,
splits and exclusions, and open items.

## One-paragraph summary

NYU undergraduate team (Hugging Face org `nyu-planet-finders`) building a transformer that finds
long-period transiting planets in TESS full-frame-image light curves, especially single transits and
duotransits. Light curves for TESS-SPOC FFI (sectors 1-81), SPOC 2-min (sectors 1-107) and Kepler are
already on Hugging Face, binned to 30 minutes. The model is pretrained with masked reconstruction on real
light curves, then fine-tuned to output a transit probability per 4-hour patch.

## Repository layout

| Path | Purpose |
|---|---|
| `reference_catalogs/` | Step 0: builds TOI ephemerides, EB catalogs, Hawthorn duotransits and the KIC-TIC crossmatch; publishes `nyu-planet-finders/reference-catalogs` |
| `.github/workflows/build-reference-catalogs.yml` | Runs Step 0 on GitHub's runners (needs the `HF_TOKEN` secret) |
| `pretrain_dataset/` | Steps 1-11: builds `lc-pretrain` (public) and `lc-heldout` (private); `config.py` holds every constant |
| `.github/workflows/build-pretrain-dataset.yml` | Runs Steps 1-11; triggered by pushing `pretrain_dataset/build_request.json` |
| `finetune_dataset/` | Builds `lc-finetune` (public), `lc-eval` and `lc-benchmarks` (private); `config.py` holds every number |
| `.github/workflows/build-finetune-dataset.yml` | Runs that build; triggered by pushing `finetune_dataset/build_request.json` |
| `experiments/finetune_mix/` | Data mix experiments on lc-finetune, scored on lc-eval val; `config.py` holds every number |
| `.github/workflows/finetune-mix-experiments.yml` | Runs them; triggered by pushing `experiments/finetune_mix/run_request.json` |
| `.github/workflows/benchmark-eval.yml` | Scores one finished arm on lc-benchmarks; triggered by pushing `experiments/finetune_mix/benchmark_request.json` |
| `.github/workflows/realism-check.yml`, `fetch-limb-darkening.yml` | Realism checks on the published lc-finetune; refetch of the Claret limb-darkening table |
| `experiments/proxy_pretrain/` | Masked-reconstruction pretraining of the small experiment model on lc-pretrain |
| `.github/workflows/proxy-pretrain.yml` | Runs it; triggered by pushing `experiments/proxy_pretrain/run_request.json` |
| `dataset_cards/` | Cards of the source datasets; `publish-dataset-cards.yml` uploads them |
| `tests/` | Offline tests (`python -m pytest tests -q`) |
| `docs/` | Handoff notes and the build guide |
| `tessLightCurveData/`, `kplr*.csv` | Early sample CSVs; the full data lives on Hugging Face |

## Conventions that must not drift

- Times are BTJD (BJD - 2457000) everywhere. Kepler BKJD converts with BTJD = BKJD - 2167.
- Star identity is `star_key`: `tic:<TIC ID>`, or `kic:<KIC ID>` for Kepler stars without a TIC match.
  Splits come from SHA-256(star_key) mod 1000 (see the build guide, Step 1).
- Catalog tags (TOI, EB, DR24 labels) are never model inputs or pretraining targets. They are used for
  exclusions, sampling, ablations, fine-tuning and evaluation only.
- Held-out stars must be physically absent from the pretraining repo.
- Rank model and data variants on the injection-recovery validation set, never on pretraining loss.
- lc-eval holds held-out stars only; every benchmark star is excluded from lc-finetune. Keep it so.
- lc-eval keeps the v2 injection physics (`finetune_dataset/config.py` INJ_PHYSICS) so experiment waves stay
  comparable; changes to injections go into lc-finetune only unless a wave deliberately resets the baseline.

## Working notes

- Code style: professional and plain. No emoji, no decorative comments. Each script starts with a
  docstring saying what it does, why, its outputs and how to run it.
- Network access differs by environment. Claude's cloud sandbox may only reach GitHub, so jobs that need
  NASA, MAST, VizieR, arXiv or Hugging Face run in GitHub Actions or in TIKE.
- Never load a whole multi-GB Parquet file in TIKE (7.6 GB RAM); read one row group at a time.
