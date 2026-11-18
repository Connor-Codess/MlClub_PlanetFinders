# Exoplanet ML pipeline: status and handoff (updated 2026-11-18)

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
| lc-finetune | 27-day labeled windows, train 400,000 / validation 24,000: real short and long transits, batman and transplanted injections, random and EB negatives; benchmark stars excluded | Built 2026-11-18 (v3: deep tail, SNR correction, Claret limb darkening, eccentric orbits) |
| lc-eval (private) | Labeled val (49,990) and test (49,885) windows from held-out stars only | v2 rows, unchanged by v3 (2026-11-18) |
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
   Findings: `experiments/finetune_mix/results/wave1/README.md`.
7. Done 2026-11-18: lc-finetune v3 (run 44279662349). Batman injections in lc-finetune only: 10% deep tail
   (combined SNR 60-300), a radius correction so injections reach their target SNR (achieved / target 1.000),
   Claret (2017) TESS limb darkening by host Teff and log g (table committed in `finetune_dataset/data/`,
   since VizieR's CDS hosts time out from the runners) and 30% eccentric orbits. lc-eval and lc-benchmarks
   keep v2 physics; `summary.json` now carries an `eval_fingerprint`, unchanged between builds.
8. Done 2026-11-18: proxy pretraining (`experiments/proxy_pretrain/`, run 44271393676) of the 0.35M model on
   three lc-pretrain mixes (`pt_mix` 50/25/25 FFI/2-min/Kepler, `pt_ffi`, `pt_tess`), 400k windows, 12,000
   steps, about 3 CPU hours each; hidden flux predicted 37-39% better than the median.
9. Done 2026-11-18: data mix wave 2 (run 44297108133), 26 arms x 3 seeds. Pre-registered winner
   `pt_mix-r00-deep-all`: pretrained on `pt_mix`, no real planets in fine-tuning, deep tail, all hosts. Against
   the v2 mix, pretrained: faint recall 21.7% against 17.9%, faint repeated short-period transits 32.9% against
   21.1%, real long-period recall unchanged (38.1%), real short-period about 6 points lower. Pretraining adds
   3.5 points of real long-period recall and 9 on deep ones; the deep tail lifts deep real long-period planets
   from 48.4% to 56.9% (wave 1: 28.7%). Kepler hosts help; Kepler noise degradation does not. Realism checks:
   batman shapes and durations match real planets once real transits are measured at fitted times; the
   remaining gap (injections 52% against real 39% at matched SNR) most likely comes from catalog-ephemeris
   labels on real planets. Classical wotan + BLS finds 0.4% of injections at 1% false alarms.
   On lc-benchmarks (run 44392254586) the winner detects 85% of Salinas et al. 2025's single-transit candidates
   and reaches AUC 0.93 for DR24 planet candidates against non-transiting signals (0.86 against all false
   positives; AstroNet: 0.988); it is a detector, not a vetter. Findings:
   `experiments/finetune_mix/results/wave2/README.md`; paper: https://claude.ai/artifact/56sFibKwzFLu1Zezqpd3Sy.
   Next: lc-finetune v4 (impact parameters up to 1, flux median recomputed after injection, real-planet labels
   from fitted transit times), full-size pretraining on `pt_mix`, a vetting stage, one look at lc-eval test.
10. Faculty sponsor and Torch access.
11. Detector training on lc-finetune at full size; injection-recovery curves on lc-eval; scores on lc-benchmarks.
12. Duotransit matching, vetting, CTOI submission.
