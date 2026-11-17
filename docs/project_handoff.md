# Exoplanet ML pipeline: status and handoff (updated 2026-11-17)

Read this first when picking up the planet-finding project. The same notes are kept in the team's
Claude project "Ml Club Space".

## Goal

Undergraduate team (NYU, "nyu-planet-finders") building a transformer that finds new transiting-planet
candidates in TESS data. Main target: long-period planets seen as single transits and duotransits (one
transit in each of two different TESS cycles), which are the least-searched part of TESS. An earlier
model was trained on Kepler (AstroNet-style global/local views).

## Strategy decisions so far

- Detector: transformer on raw (unfolded) light-curve windows with a per-patch transit-probability
  output (plus center-offset and duration heads). Cross-cycle matching and period-alias elimination
  are done with ordinary code, not ML.
- Pretraining: self-supervised masked reconstruction on real light curves only (TESS FFI, TESS 2-min,
  Kepler). No synthetic transits and no labels in pretraining.
- Fine-tuning: injected transits (batman) into real light curves, real known transits, broad negatives
  (EBs, systematics near gaps, flares). Real long-period TOIs and the Hawthorn duotransits are held out
  for testing.
- Search target: every TESS-SPOC FFI light curve, prioritizing bright stars (Tmag < ~12-13), stars
  observed in two or more cycles, later sectors first.
- Model and data choices are made with small proxy models scored on a fixed injection-recovery
  validation set, never on pretraining loss.
- Key references: Hawthorn et al. 2024 (MNRAS 528, 1841, arXiv:2310.17268); Salinas et al. 2025
  (arXiv:2502.07542, transformer on FFI light curves, conv embedding); Malik et al. 2025 AJ;
  Shallue & Vanderburg 2018 (AstroNet).
- Compute: TIKE has no GPU (2 cores, 7.6 GB RAM). Plan: NYU HPC "Torch" (needs a faculty sponsor).
  Fallbacks: Kaggle (about 30 GPU-hours per week per account), Colab, rented GPUs.

## Pretraining dataset build guide

Full plan with every decision and recommendation: [pretraining_dataset_build_guide.md](pretraining_dataset_build_guide.md)
(live version with figures: https://claude.ai/artifact/3q2VWUSb4UxhiQoQZ6Hyqr).

- Build once as lc-pretrain v1: one row per segment (star x sector or star x quarter) on one global
  30-minute grid (slot = floor((BTJD + 2200) / 30 min)), 7 per-slot channels at 12 bytes per slot
  (flux_z, flux_err_z, cen_x, cen_y in arcsec, bkg_z, fill, flags), point-to-point sigma
  normalization, catalog tags and known events, stored sampling weights (FFI / 2-min / Kepler
  50 / 25 / 25, cap 20 windows per star).
- Splits by SHA-256 of star_key mod 1000: 0-849 pretrain-train, 850-899 pretrain-val, 900-949
  search-holdout (searched, never pretrained on), 950-999 scoreboard hosts. Held-out stars plus the
  Hawthorn hosts and long-period TOIs (P > 20 d or no period) go to a private lc-heldout repo.
- Variants by config: P-ALL (recommended, ~17.4M segments, ~23M windows, ~370 GB before compression),
  P-TESS (~233 GB), P-FFI (~205 GB), proxy (200k segments, ~4 GB).

## Datasets on Hugging Face (org: nyu-planet-finders), checked 2026-11-14

| Repo | Contents | Status |
|---|---|---|
| tess-spoc-ffi-binned | TESS-SPOC FFI light curves binned to 30 min; tic_id, sector, tmag, teff, radius, time, flux, flux_err, centr_x, centr_y, bkg, n_points | Complete: s0001-s0081, 12.8M rows, 252.8 GB |
| tess-spoc-2min-binned | SPOC 2-min light curves binned to 30 min, sectors 1-107 | Complete: 1,734,057 rows, 27.7 GB |
| kepler-llc | Kepler long cadence (29.4 min), one file per quarter + manifest.parquet | Complete: q00-q17 (q15 13.5 GB, q16 11.2 GB, q17 4.2 GB) |
| kepler-raw-lightcurves | Unbinned DR25 light curves for the 9,865 DR24 TCE stars plus the DR24 TCE table | Complete; for supervised Kepler work |
| kepler-centroids | Centroids and pointing corrections, row-aligned with kepler-raw-lightcurves | Complete |
| TESS_TCES | TOI list with 4 columns only | Superseded by reference-catalogs/toi_ephemerides |
| lc-pretrain | Pretraining segments for train and validation stars, window index with weights, validation report | Built 2026-11-15 (v1) |
| lc-heldout (private) | Same format for search_holdout, scoreboard_hosts and list_excluded stars, plus exclusions.parquet | Built 2026-11-15 (v1) |
| lc-finetune | 27-day labeled windows, train 400,000 / validation 24,000: real short and long transits, batman and transplanted injections, random and EB negatives; benchmark stars excluded | Built 2026-11-16 (v2) |
| lc-eval (private) | Labeled val (49,990) and test (49,885) windows from held-out stars only | Built 2026-11-16 (v2) |
| lc-benchmarks (private) | AstroNet DR24, Astronet-Triage, Astronet-Triage-v2 test sets and Salinas et al. 2025 candidates on our light curves (16,499 windows) | Built 2026-11-16 (v2) |
| reference-catalogs | TOI ephemerides (8,175), TESS EBs (4,584), Kepler EBs (2,876), Hawthorn duotransits (85), KIC-TIC crossmatch (207,617 KIC, 99.0% matched) | Built 2026-11-15 by `reference_catalogs/` via GitHub Actions; refreshed monthly |

Time systems: TESS times are BTJD = BJD - 2457000; Kepler times are BKJD = BJD - 2454833;
BTJD = BKJD - 2167. Centroid plate scales: TESS ~21"/pixel, Kepler 3.98"/pixel.

Processing differences: 2-min TESS and Kepler used QUALITY == 0 and required finite centroids. FFI used
lightkurve's default quality bitmask (17087) and only requires finite time and flux.

## Open items

1. Done 2026-11-15: Kepler q15-q17 on the Hub.
2. Done 2026-11-15: Step 0 reference catalogs on Hugging Face (nyu-planet-finders/reference-catalogs).
3. Done 2026-11-15: lc-pretrain v1 built by `pretrain_dataset/` in GitHub Actions
   (`.github/workflows/build-pretrain-dataset.yml`), all four Step 11 gates passing: 206 of 206 source
   files accounted for, 0 star_keys in both repos, worst round-trip error 0.032 sigma, one catalog
   revision (116cd81b). 17,401,706 light curves read, 17,151,144 segments written (250,562 dropped, almost
   all too noisy), 4,173,056 stars. lc-pretrain holds 15.43M segments and 20.35M default windows
   (FFI 11.43M, 2-min 1.44M, Kepler 7.49M windows); lc-heldout (private) holds 1.72M segments. Report:
   `index/validation_report.md` in lc-pretrain. One change from the guide: no separate re-shuffle pass
   (Step 10); sampling order comes from the weighted window index. A resumed build reuses the
   reference-catalogs revision already in use.
4. Done 2026-11-16: a dataset card for every repository (`dataset_cards/`, `publish-dataset-cards.yml`).
5. Done 2026-11-16: lc-finetune, lc-eval and lc-benchmarks v1 by `finetune_dataset/`; all four build checks
   passed (540,374 rows, no shared or benchmark stars, transit patches 1.2-2.7 sigma below negatives, measured /
   intended injection SNR 1.02). Real transits filled 10% of lc-finetune train instead of the planned 22.5%
   (per-star cap of 12, few Kepler long-period PCs); batman took the rest.
   v2 (2026-11-16, run 43913578674) keeps faint short-period planets: planets are verified by the stacked SNR of
   all their transits (>= 7), real windows need a stacked in-window SNR >= 4 instead of one transit at SNR 4,
   30 windows per star, and injections target a combined SNR, so short periods reach per-transit SNR near 1.
   Real transits are now 16% of lc-finetune train (53,268 short, 10,900 long). Batman injections land at a
   median 0.83 of their target combined SNR; the stored inj_snr_total is the achieved value. Every decision and change:
   `docs/finetune_dataset_build_log.md` (live copy: https://claude.ai/artifact/7h6fmA3BG5vne1v8yvHAek).
   Open: benchmark stars are still in lc-pretrain (skip them with the loader's `exclude_star_keys`, or rebuild),
   TESS benchmark coverage is partial (QLP versus TESS-SPOC), masking function for pretraining not written yet.
6. Done 2026-11-17: data mix experiments (`experiments/finetune_mix/`, run 44000179513). Thirteen
   lc-finetune mixes were each used to train a 0.35M-parameter detector 3 times on GitHub's CPU runners,
   scored on lc-eval val. The v2 mix is the best-balanced; its stacked-SNR rule raises recall on faint
   short-period transits from 14.3% to 20.3%. Real planets deeper than SNR 25 are mostly missed (28.7% of
   long-period ones), because training injections stop near SNR 50 while EB negatives are deep.
   Next: add deep injections (combined SNR 60-300) and TESS-only injection hosts, then test them in a second
   wave. Findings: `experiments/finetune_mix/results/wave1/README.md`.
7. Faculty sponsor and Torch access.
8. Detector training on lc-finetune; injection-recovery curves on lc-eval; scores on lc-benchmarks.
9. Duotransit matching, vetting, CTOI submission.
