# Fine-tuning, evaluation and benchmark dataset build log

Nov 16, 2026, updated Nov 18 for v3. Repository copy of the live document at https://claude.ai/artifact/7h6fmA3BG5vne1v8yvHAek
(the live copy also has a diagram of the build pipeline).

## Summary

Three labeled datasets are built from lc-pretrain and lc-heldout by `finetune_dataset/` in GitHub Actions: **lc-finetune** (public, fine-tuning), **lc-eval** (private, validation and test) and **lc-benchmarks** (private, published papers' test sets on our light curves). This log records every decision, number and change, in the style of the pretraining build guide.

- **One row = one 27-day window** (1,296 slots of 30 minutes) cut from an lc-pretrain or lc-heldout segment, with a label for each of its 162 four-hour patches. Channels and normalization are unchanged from pretraining.
- **Transit sources:** real known planets (short and long period), batman model transits injected into stars with no catalog entry, and measured transit profiles of real planets transplanted into such stars. Negatives are random windows and eclipsing-binary or false-positive events.
- **Leakage rules:** lc-eval uses held-out stars only. Every star of every benchmark test set is removed from lc-finetune.
- **Status:** the current build is v3 (2026-11-18), which changes only lc-finetune's batman injections; lc-eval and lc-benchmarks are the same rows as v2. v1 was built at commit `e9a6eab` and v2 at `9e1693c` on branch `reference-catalogs`. Every build has passed all four checks and holds 540,374 rows.

Code: [finetune\_dataset](https://github.com/Connor-Codess/MlClub_PlanetFinders/tree/reference-catalogs/finetune_dataset). Workflow: `.github/workflows/build-finetune-dataset.yml`.

## Datasets produced

Each dataset draws on a fixed set of star pools, so no star can feed both training and testing.

| Repository | Visibility | Splits | Stars it may use | Purpose |
| --- | --- | --- | --- | --- |
| `nyu-planet-finders/lc-finetune` | Public | train, validation | lc-pretrain `train` (hash 0-849) and `validation` (850-899), minus every benchmark star | Fine-tuning the per-patch detector |
| `nyu-planet-finders/lc-eval` | Private | val, test | Held-out stars only: `scoreboard_hosts` (950-999) and `list_excluded` | Choosing model variants (val), final score (test) |
| `nyu-planet-finders/lc-benchmarks` | Private | one per paper | Stars of the published test sets, wherever they sit | Comparing with AstroNet, Astronet-Triage, Astronet-Triage-v2, Salinas et al. 2025 |

How held-out stars split between lc-eval val and test:

| Pool | Rule | lc-eval split |
| --- | --- | --- |
| `scoreboard_val` | `scoreboard_hosts` with SHA-256("scoreboard:" + star\_key) first 8 bytes even | val |
| `scoreboard_test` | the same hash odd | test |
| `list_val` | `list_excluded` marked `real_validation` in `exclusions.parquet` | val |
| `list_test` | `list_excluded` marked `real_test` | test |
| `unused` | `search_holdout` (900-949) | never used here; kept for the seen-versus-unseen search check |

**Decision:** lc-eval is built only from lc-heldout stars. An earlier plan in chat used validation-split stars for real short-period evaluation; that was dropped so lc-eval never shares a star with lc-finetune's validation split. Validation-split short-period planets (291 TOIs) stay in lc-finetune's validation split.

## Inputs and pinned versions

Every input is pinned, so a rerun gives identical rows.

| Input | Version used | Why it is pinned |
| --- | --- | --- |
| lc-pretrain, lc-heldout | Build v1, 206 pass-1 files (81 FFI, 107 2-min, 18 Kepler) | Windows are cut from these rows |
| reference-catalogs | Revision `116cd81b683f`, read from lc-pretrain's pass-1 stats | Pools, known events and exclusions must match what pass 1 stored |
| DR24 TCE table | `tces/dr24_tce.parquet` in kepler-raw-lightcurves | Kepler planet and false-positive labels; AstroNet test set |
| Astronet-Triage test records | `yuliang419/Astronet-Triage` commit `5675a57d` | Yu et al. 2019 test set |
| Astronet-Triage-v2 TCE table | Zenodo record 7411579, `tces_with_labels_v3.csv` | Tey et al. 2023 test set |
| Salinas et al. 2025 candidates | arXiv e-print 2502.07542v2 (`candidates/*.csv`) | Their published candidates |
| Random seeds | `SEED = 20261116`; one 62-bit seed per sample in `recipes.parquet` | Any single sample can be rebuilt alone |

`benchmark_provenance.json` in lc-benchmarks stores the SHA-256 of every downloaded benchmark file.

## Stage 0: published benchmarks

Three published test sets are rebuilt exactly, and one paper is covered by its candidate list because its test set was never released. `benchmarks.py` writes one row per test signal to `benchmark_targets.parquet`.

| Benchmark | Paper | Data | How the test set is rebuilt | Signals | Stars | Labels |
| --- | --- | --- | --- | --: | --: | --- |
| `astronet_dr24` | Shallue & Vanderburg 2018, AJ 155, 94 | Kepler | DR24 TCEs labeled PC/AFP/NTP, in archive order (kepid, tce\_plnt\_num), `np.random.seed(123)` permutation, last 10% | 1,574 | 1,501 | PC 343, AFP 967, NTP 264 |
| `astronet_triage` | Yu et al. 2019, AJ 158, 25 | TESS QLP FFI, sectors 1-5 | Read from their released `test-00000-of-00001` TFRecord | 1,658 | 1,635 | PC 49, EB 230, J 1,379 |
| `astronet_triage_v2` | Tey et al. 2023, AJ 165, 95 | TESS QLP FFI, cycles 1-3 | Rows with `Split == test` in their Zenodo table | 2,516 | 2,511 | E 250, S 34, B 72, J 1,610, N 15, no consensus 535 |
| `salinas2025` | Salinas et al. 2025, MNRAS 538, 2031 | TESS-SPOC FFI, sectors 1-26 | Test set not released; their 213 published candidates instead | 213 | 208 | 122 periodic, 82 single transit, 9 multi-planet |

Results each paper reports on its own test set (the numbers to compare against):

| Benchmark | Published result |
| --- | --- |
| `astronet_dr24` | AUC 0.988, accuracy 0.960 at threshold 0.5, on 1,523 test TCEs |
| `astronet_triage` | Triage (PC+EB vs rest): AUC 0.992, average precision 0.970, accuracy 0.974, precision 0.749 and recall 0.975 at threshold 0.1. Vetting (PC only): AUC 0.984, average precision 0.693 |
| `astronet_triage_v2` | Positive class E: AUC-PR 0.965; recall 97.2% at precision 75.7% (threshold 0.215) |
| `salinas2025` | AUC 0.88, F1 0.82 on an unreleased random 10% split |

Notes on fidelity:

- **AstroNet count gap.** The rebuilt split has 1,574 TCEs; the paper evaluates 1,523, which matches their pipeline skipping TCEs it could not process. The split assumes their CSV was in the archive's default order (kepid, tce\_plnt\_num); our table's `rowid` is an Oracle string, so it cannot confirm the order. Their precomputed TFRecords (Google Cloud Storage) now return 404 and are not in the Internet Archive.
- **Yu et al. count gap.** The repository's test file holds 1,658 TCEs; the paper quotes 1,650.
- **Splits are by TCE in both AstroNet papers**, so their test stars also appear in their training sets. We remove the whole star from lc-finetune, which is stricter.
- **Different light curves.** The TESS papers used QLP light curves; ours are TESS-SPOC FFI. Signals on stars TESS-SPOC did not process get no windows; coverage is reported per label.
- **Different task.** These papers classify a signal given its period and epoch. The lc-benchmarks card defines the comparison: a signal's score is the mean, over its visible predicted transits, of our model's transit probability at the patch holding the mid-time. Report the covered fraction next to every number.

Label classes used for patch labels: PC and Salinas candidates are planets; EB and B are eclipsing binaries; AFP, NTP and J are false positives; Tey's E (planets and detached EBs together), S, N and blank are "unknown" (patch label 255), because our model treats eclipsing binaries as negatives. Each signal also carries `positive_paper` (the paper's own positive class) and `positive_planet`.

## Stage 1: scan

One job per source file reads its up to three pass-1 files (lc-pretrain train and validation, lc-heldout) and writes four small tables; choosing samples needs the whole build in view, but reading 230 GB in one job would be slow.

| Output | One row per | Rule |
| --- | --- | --- |
| `_events.parquet` | Default 27-day window holding a known event (TOI, EB, DR24 TCE, Hawthorn duotransit) | Every event measured in place: class, period, SNR, observed fraction, depth |
| `_hosts.parquet` | Default window of a star with no catalog entry at all | Kept by a deterministic hash of (seed, segment\_id, window index); coverage at least 0.3 |
| `_cutouts.parquet` | Transit of a known planet on a train or validation star | Period known, at least 80% observed, per-transit SNR at least 2, no other event nearby, at most 20 per planet per segment |
| `_bench.parquet` | Default window holding a predicted transit of a benchmark signal | Kepler signals on Kepler segments; TESS signals on FFI segments only, matching sector or time range; Salinas candidates only in sectors 1-26 |

Host sampling rates (share of eligible windows kept):

| Pool | FFI | 2-min | Kepler |
| --- | --: | --: | --: |
| train | 5% | 12% | 3% |
| validation | 6% | 15% | 3% |
| scoreboard\_val, scoreboard\_test | 40% | 0 | 0 |

The rates are about four times what the quotas need, so per-star caps and source shares can still be met.

**Event measurement.** For an event at fractional slot c with duration d slots: depth = (median of observed out-of-transit slots within d/2 + max(d, 12) of c) minus (mean of observed in-transit slots); SNR = depth × sqrt(observed in-transit slots), since flux\_z is in units of slot-to-slot noise. Observed fraction counts in-transit slots against the transit's nominal length, including slots past a window edge. Events with no catalog duration (Kepler EBs) use 4 hours.

**Event classes.** TOI PC/APC/CP/KP, DR24 PC and Hawthorn duotransits are planets; TESS and Kepler EB catalog eclipses are eclipsing binaries; TOI FP/FA and DR24 AFP/NTP are false positives; TOIs with no disposition and DR24 UNK are unknown.

## Stage 2: choosing samples

lc-finetune holds 400,000 train and 24,000 validation rows in a fixed mix; any category that cannot fill its share is topped up with batman injections.

| Category | Share | Train rows | Validation rows | Source | Per-star cap | Selection rule |
| --- | --: | --: | --: | --- | --: | --- |
| `real_planet_short` | 15% | 60,000 | 3,600 | 60% TESS, 40% Kepler | 12 | A verified planet whose visible transits in the window stack to SNR at least 4, none of them long-period (v2; v1 required one transit at per-transit SNR 4) |
| `real_planet_long` | 7.5% | 30,000 | 1,800 | Kepler (every TESS P >= 20 d candidate is held out) | 12 | A visible long-period (P >= 20 d or unknown) planet transit with SNR at least 4 |
| `injected_batman` | 35% | 140,000 | 8,400 | 60% FFI, 20% 2-min, 20% Kepler | 3 | Host star with no catalog entry, radius at most 5 R\_sun or unknown |
| `injected_transplant` | 15% | 60,000 | 3,600 | same | 3 | same host rule; template from a train-split planet |
| `negative_random` | 17.5% | 70,000 | 4,200 | same | 3 | Unchanged window of a star with no catalog entry; any radius |
| `negative_eb` | 10% | 40,000 | 2,400 | 60% TESS, 40% Kepler | 10 | Star with no planet; an EB or false-positive event of SNR at least 3, at least half observed |

- **Single-transit crops.** 25% of `real_planet_short` rows with two or more visible transits keep one transit at random. Slots nearer to any other transit are hidden like a data gap, so the model does not learn to expect repeats.
- **Distinct hosts.** Each host window serves one category. Hosts are handed out in the order random negatives, transplants, batman, so a shortage shrinks batman (the top-up category) first.
- **Benchmark stars** (5,843) are removed from every lc-finetune category and from the finetune template library.

lc-eval takes every qualifying real window and fixed counts for the rest, per split (val and test):

| Category | Rows per split | Source | Rule |
| --- | --: | --- | --- |
| `real_planet_long` | all | all | Planet transit at least half observed, no SNR cut, at most 10 windows per star; long if any visible planet transit has P >= 20 d or unknown period |
| `real_planet_short` | all | all | same, otherwise |
| `injected_batman` | 20,000 | FFI | Scoreboard host with no catalog entry |
| `injected_transplant` | 5,000 | FFI | same; template from a validation-split planet |
| `negative_random` | 20,000 | FFI | Unchanged scoreboard window, no catalog entry |
| `negative_eb` | up to 5,000 | all | As in lc-finetune, at most 5 per star |

lc-benchmarks keeps up to 8 windows per signal, chosen at random. `choose_samples.py` refuses to write recipes if any lc-finetune row comes from outside the train/validation pools, any lc-eval row from outside the held-out pools, any star sits in both, or any benchmark star is in lc-finetune.

## Batman injections

The depth of every injected transit is set from a target SNR (v2: the combined SNR of the transits observed in the window, each getting target / sqrt(n); v1: per transit), not from a planet-radius distribution, so the set covers the detection threshold evenly instead of filling up with easy transits.

| Parameter | Distribution or rule |
| --- | --- |
| Period P | Log-uniform 1-730 d (lower bound raised so a/R\* >= 3). About 52% of draws exceed 27 d, so those windows hold one transit |
| Star | TIC radius of the host (1 R\_sun if missing); mass = R^1.25 for R < 1.5 R\_sun (clipped 0.1-1.6), else 1.2 M\_sun |
| a/R\* | 215.032 × (M × (P / 365.25 d)^2)^(1/3) / R |
| Impact parameter b | Uniform 0-0.9; circular orbit |
| Limb darkening | Quadratic, q1 and q2 uniform 0-1 (Kipping 2013), covering the physical range |
| Target SNR | Log-uniform: lc-finetune 3-60, lc-eval 2-40 |
| Rp/R\* | Solved from the target depth = SNR × s / sqrt(in-transit slots), with the limb-darkening factor at the transit chord; three iterations for the duration; clipped 0.003-0.3 |
| Placement | One transit centered on a random observed slot (plus a uniform sub-slot offset); the others at multiples of P |
| Integration | batman with 30-minute exposure, 15× supersampling |

The model multiplies relative flux on observed slots only, keeping the host's median and noise s = sigma\_p2p\_ppm × 1e-6:

```latex
z' = \frac{(1 + s z)\, m(t) - 1}{s}
```

Every row stores the drawn parameters (`inj_period_d`, `inj_t0_btjd`, `inj_rp_rs`, `inj_a_rs`, `inj_b`, `inj_u1`, `inj_u2`, `inj_duration_h`, `inj_depth_ppm`, `inj_snr_target`), the matched-filter SNR actually achieved on observed slots (`inj_snr` for the best transit, `inj_snr_total` combined) and the star values used. Injections get no centroid or background shift, as for a transit on the target star itself.

**Known limits:** circular orbits only; the mass-radius relation is rough for evolved stars, which is why hosts above 5 R\_sun are excluded; transits falling in a gap are labeled 255 (ignore).

## Transplanted real transits

A transplant multiplies the measured, phase-folded transit of a real known planet into a star with no catalog entry, at that planet's own period. This gives the model real transit shapes (limb darkening, ingress, smearing) that a model curve can only approximate.

1. **Cutouts (scan stage).** Around each usable transit, take the slots within d/2 + max(4 h, d/2) of mid-transit, fit a straight line to the slots more than d/2 + 0.75 h away, and divide it out. Store (hours from mid-transit, relative flux).
2. **Stack (choose stage).** Pool all cutouts of one planet across sectors, sources and quarters into 10-minute bins (median per bin), fill empty bins by interpolation, smooth with a 3-bin running median, normalize to the out-of-transit median.
3. **Accept** a planet when it has at least 3 transits, at least 60% of its in-transit bins hold data, the folded depth is at least 10 times the out-of-transit bin scatter, and the depth is at most 50%.
4. **Clean edges.** Beyond d/2 + 0.75 h the template is exactly 1, and inside it is capped at 1, so no noise from the source star is copied into the host.
5. **Inject.** Place one transit on a random observed slot, the others at the planet's period. Keep the real depth, or scale it down (`inj_depth_scale` < 1) to reach the target SNR; it is never scaled up.

| Template pool | Source stars | Used by |
| --- | --- | --- |
| finetune | Planets on train-split stars, benchmark stars removed | lc-finetune (train and validation) |
| eval | Planets on validation-split stars | lc-eval only, so test transits come from planets lc-finetune never transplants |

The source light curves are already binned to 30 minutes, so the stacked profile is the 30-minute-smeared shape that a host slot sees; it is not smeared again. Kepler templates keep Kepler-band limb darkening. The template's duration belongs to its source star, not the host, so a transplant is physically a planet of that duration on a different star.

**Changes during development.** The first version needed every in-transit bin to have 2 or more points, which no planet with fewer than about 6 transits could meet: zero templates in the local trial. It now needs 1 point in 60% of bins and interpolates the rest. The out-of-transit baseline was widened from 2 h to 4 h past the transit edge, because short transits left too few points to fit the line.

## Labels and row format

Every row carries four per-patch targets (162 patches of 4 hours), all derived from the events measured in the final window.

| `patch_label` | Meaning |
| --- | --- |
| 0 | No planet transit has its mid-time in this patch |
| 1 | A planet transit (real or injected) has its mid-time here and at least 50% of its in-transit slots are observed |
| 255 | Ignore in the loss: a planet transit mostly in a gap, or a known event of unknown nature |

| `patch_kind` code | Meaning (highest priority first) |
| --- | --- |
| 1 | Real planet |
| 2 | Batman injection |
| 3 | Transplanted real transit |
| 4 | Eclipsing binary (labeled 0) |
| 5 | False positive: TOI FP/FA, DR24 AFP/NTP (labeled 0) |
| 6 | Known event of unknown disposition (labeled 255) |
| 0 | Nothing |

`patch_overlap` (uint8 / 255) is the share of the patch inside a planet transit; `patch_center_offset` (float16) is where the mid-time sits inside a labeled patch, NaN elsewhere. Center labels make each transit count once even when it spans several patches; the overlap supports other labeling choices without a rebuild.

Row columns: identity (`sample_id`, `dataset`, `split`, `category`), star and segment (`star_key`, `source`, `segment_id`, `sector`, `quarter`), window position (`window_slot0`, `t_start_btjd`, `start_offset`), provenance (`source_repo`, `source_path`, `source_row`), star and noise (`sigma_p2p_ppm`, `tmag`, `kepmag`, `teff`, `radius`, `quality_tier`, `coverage`), the seven lc-pretrain channels as bytes (`flux_z`, `flux_err_z`, `cen_x`, `cen_y`, `bkg_z` float16; `fill`, `flags` uint8), the four targets, `n_transit_patches`, `has_transit`, `events` (center, duration, kind, catalog reference, class, period, measured SNR, observed fraction, visible, is\_planet), `single_transit_crop`, the `inj_*` parameters, `benchmark_tce_id` and `build_version`. `finetune_dataset/schema.py` holds the schema and `decode_sample()`.

The flux is not detrended, as in pretraining: stellar variability is part of the task.

## Stages 3-4: materialize, shuffle, checks, upload

The build runs as five GitHub Actions jobs; the two per-file stages run as matrices of up to 206 jobs, 20 at a time.

&#91;embedded content: build pipeline · 5 jobs\]

Scan and materialize both read lc-pretrain and lc-heldout directly; only small tables pass between the jobs until the rows themselves.

**Materialize (one job per source file).** For each recipe: cut the window from its pass-1 row (padding past the segment end as unobserved), apply the single-transit crop if set, inject (batman or transplant) with the recipe's own seed, re-measure every event in the final window, compute the four targets and write the row. The job fails if it writes a different number of rows than it has recipes, or if a row's `segment_id` does not match its recipe.

**Assemble (one job).** Deals every row at random into shards of up to 10,000 (pass A), shuffles inside each shard (pass B), so reading a split in file order is already random. Then four gates; any failure stops the upload:

| Gate | Passes when |
| --- | --- |
| Row accounting | Every recipe became exactly one row |
| Leakage | No star in both lc-finetune and lc-eval; no benchmark star in lc-finetune |
| Labels | Per category, the median flux of transit-labeled patches is at least 1 sigma below that of negative patches (0.25 sigma for lc-eval real windows, which have no SNR cut) |
| Injections | Measured stacked / expected combined SNR has a median between 0.8 and 1.3, and expected / target combined SNR between 0.8 and 1.25 (v2; v1 compared per-transit SNR) |

Then it writes `summary.json`, the transplant templates of each dataset, `benchmark_targets.parquet` and `benchmark_provenance.json`, the three dataset cards, and uploads. Before uploading, it deletes the repository's old `data/` folder, because shard names depend on row counts and old shards would otherwise linger.

Artifacts between jobs are kept 5 days: `benchmarks`, `scan-<source>-<stem>`, `plan` (`recipes.parquet`, `templates.parquet`, `plan.json`) and `rows-<source>-<stem>`.

## Dataset cards for every repository

Every repository in the nyu-planet-finders organization now has a card; generated datasets write theirs from their own build numbers, hand-written ones live in `dataset_cards/`.

| Repository | Card before | Card now | Written by |
| --- | --- | --- | --- |
| tess-spoc-ffi-binned | None | Columns, processing (bitmask 17087, PDCSAP / median, 30-minute bins, `n_points` by cadence era), usage, credits (Caldwell et al. 2020, CC BY 4.0) | `dataset_cards/tess-spoc-ffi-binned.md` |
| kepler-llc | Config block only | Columns, processing (SAP\_QUALITY == 0, native 29.4-minute cadence), 2,869,929 light curves of 207,617 stars, manifest, usage | `dataset_cards/kepler-llc.md` |
| TESS\_TCES | None | Columns, disposition counts (8,151 TOIs), CSV comment handling, marked superseded by reference-catalogs | `dataset_cards/TESS_TCES.md` |
| lc-heldout | Short, with a misleading train/validation count block | Each held-out split, its rule and use; counts for held-out splits only; layout, columns, usage | `pretrain_dataset/publish_cards.py` |
| lc-pretrain | Full | Adds usage (decoding the byte columns) and related datasets | `pretrain_dataset/publish_cards.py` |
| lc-finetune, lc-eval, lc-benchmarks | New | Contents by category, split and source; labels; injection details; build gates; usage; for lc-benchmarks the comparison protocol, coverage and published results | `finetune_dataset/cards.py` |
| tess-spoc-2min-binned, kepler-raw-lightcurves, kepler-centroids, reference-catalogs | Full | Unchanged | Earlier work |

The workflow `publish-dataset-cards.yml` publishes `dataset_cards/*.md` and refreshes the lc-pretrain and lc-heldout cards from their published summary whenever those files change. It first ran 2026-11-16 at commit `8808a98` and succeeded.

## v2: faint short-period planets

v2 keeps faint short-period planets in lc-finetune: a real window now counts when its transits add up to a detection inside the window, not when one transit is visible alone. The change came from review: v1's per-transit SNR >= 4 cut removed planets that were found by stacking many shallow transits, which would have made the model worse at the faint short-period case it should still catch. The long-period setup is unchanged.

| Rule | v1 | v2 | Why |
| --- | --- | --- | --- |
| Planet verification | none | All visible transits of the planet on the star, stacked, reach SNR 7; failing planets' transits are labeled ignore (255) | Checks the ephemeris still lands on real dips, without asking any single transit to be visible |
| Real window selection (lc-finetune) | Best single transit SNR >= 4 | One verified planet's visible transits in the window stack to SNR >= 4 | A 2-day planet with 12 transits at SNR 1.5 each stacks to 5.2 and is learnable |
| Windows per star (real) | 12 | 30 | The per-star cap, not the SNR cut, was what limited real short-period rows |
| Injection depth target | Per-transit SNR, log-uniform 3-60 (eval 2-40) | Combined SNR of the observed transits, same ranges; each transit gets target / sqrt(n) | Short-period injections now reach per-transit SNR near 1 |
| Single-transit crop | Any real short window with 2+ transits | Only when the kept transit has per-transit SNR >= 4 | A crop must not leave only an invisible transit |
| New row columns | - | `stacked_snr`, `n_visible_transits`; `events[].expected_snr` for injections | Recall can be binned by per-transit and combined SNR |
| Injection gate | Measured best transit / expected per-transit SNR, median in 0.8-1.3 | Measured stacked / expected combined SNR in 0.8-1.3, and expected / target combined in 0.8-1.25 | Matches the new target |
| Label gate | 1 sigma (0.25 for lc-eval real) | 0.5 sigma for real and injected groups in lc-finetune; 0.25 for lc-eval real; 1 otherwise | Faint rows dip less on average by design |

The stacked SNR is the matched-filter sum for a planet whose transits share one depth:

```latex
\mathrm{SNR}_{\mathrm{stack}} = \frac{1}{\sqrt{n}} \sum_{i=1}^{n} \mathrm{SNR}_i
```

For equal depths it equals the square root of the summed squared SNRs of the noiseless signal, the figure the proposal used. It is used on measured values instead because it averages to zero on pure noise: with the summed-squares form, 16 noise-only transit slots would score about 4 and pass. Overlapping windows count each transit once, and TESS FFI and 2-min data are stacked separately (the best source is kept) because they observe the same sectors.

**Effect, from the v1 scan tables (train pool, before benchmark removal).** Planets with visible transits: 7,572; verified: 7,264. Real windows passing the rule: 84,321 in v2 against 76,379 in v1; after removing benchmark stars, 76,481. At 12 windows per star that gives 35,688 short and 6,199 long rows, barely above v1's 33,777 and 6,713, so the cap rises to 30: 52,779 short (8,264 of them with no transit visible alone) and 10,406 long. The earlier estimate of about double the short-period rows counted windows before the per-star cap.

**Local trial (3 files, 1% scale).** All four gates passed: measured stacked / expected SNR median 0.941, expected / target 0.909. Batman injections with P < 5 d: per-transit SNR 10th/50th/90th percentiles 1.07 / 2.77 / 10.67 at combined 3.25 / 9.58 / 36.1; 57% of them have per-transit SNR below 3.

**v2 build.** Run 43913578674 (commit 9e1693c, about 31 minutes) passed every gate and replaced v1 on Hugging Face at 17:30 UTC; each dataset's `summary.json` now reads `build_version: v2`. The Build log section below describes the v1 runs. Planets on train and validation stars with visible transits: 7,993, of which 7,675 are verified. Template counts are unchanged (3,195 and 193).

| lc-finetune train | v1 | v2 | Change |
| --- | --: | --: | --: |
| `real_planet_short` | 33,777 | 53,268 | +58% |
| `real_planet_long` | 6,713 | 10,900 | +62% |
| `injected_batman` (top-up) | 189,510 | 165,832 | -12% |
| `injected_transplant` | 60,000 | 60,000 | 0 |
| `negative_random` | 70,000 | 70,000 | 0 |
| `negative_eb` | 40,000 | 40,000 | 0 |

Real transits are now 16.0% of training rows (v1: 10.1%; the plan is 22.5%). lc-finetune validation has 3,184 short and 486 long real rows (v1: 1,993 and 308). lc-eval and lc-benchmarks select the same windows as v1 (540,374 rows in all); only their injections changed, to the combined-SNR target.

| Gate | Result |
| --- | --- |
| Row accounting | 540,374 rows for 540,374 recipes |
| Leakage | 0 stars shared by lc-finetune and lc-eval; 0 benchmark stars in lc-finetune |
| Labels (median negative minus transit patch flux, sigma) | lc-finetune: real short 1.72, real long 1.96, batman 1.97, transplant 1.05; lc-eval: real short 1.51, real long 2.60, batman 1.36, transplant 0.70 |
| Injections | Measured stacked / expected combined SNR 0.943; expected / target 0.840; 289,230 rows |

**Check of a published shard.** `data/train/part-00011-of-00040.parquet` (9,979 rows, all `v2`) was downloaded and re-measured with a plain box filter at each labeled transit, independent of the build code. Stored and re-measured `stacked_snr` correlate at 0.93 for real short-period rows (medians 9.2 and 10.5), 0.76 for real long, 0.84 for transplants and 0.67 for batman, whose re-measure has no local baseline. In this shard 16% of real short-period rows have no transit at per-transit SNR 4, the regime v1 removed. Batman injections with P < 5 d have per-transit SNR 1.06 / 3.7 / 12.5 (10th / 50th / 90th percentiles) at combined 3.3 / 11.3 / 36.0. Of the real rows, 53% carry an ignore patch, almost all from a transit less than half observed at a gap or window edge; unverified planets cause 1.3%.

**Known offset.** Batman injections reach a median 0.83 of their target combined SNR (transplants 1.00): the depth solve ignores ingress, egress and the slot integration. The stored `inj_snr_total` is the achieved value, so recall curves binned on it are correct; only the drawn range sits about 17% lower than nominal (roughly 2.5-50 in lc-finetune). A one-step correction in `batman_injection` would remove it at the next rebuild.

## v3: deep tail, SNR correction, star-matched limb darkening, eccentric orbits

v3 changes only how batman injections are made, and only in lc-finetune (`config.INJ_PHYSICS = {"finetune": "v3", "eval": "v2"}`). lc-eval and lc-benchmarks keep the v2 physics, so fine-tuning experiments from before and after v3 are scored on identical windows. The changes come from wave 1 of the data-mix experiments (`experiments/finetune_mix/results/wave1/`): real planets above combined SNR 25 were found only 29% of the time, and no training injection was that deep, while eclipsing binaries are.

| Rule | v2 | v3 | Why |
| --- | --- | --- | --- |
| Deep tail | Combined SNR 3-60 for every injection | 10% of batman rows draw from 60-300 (`inj_deep`); the draw has its own random stream, so a row's other draws are unchanged | Teach the model that deep, planet-shaped dips are planets, not only EBs |
| SNR correction | Depth from a box-shaped estimate; achieved a median 0.83 of target | After the first batman model, the planet radius is scaled by sqrt(target / achieved), two rounds | Injections land on their target SNR |
| Limb darkening | Kipping (2013) uniform draw over the physical range | Claret (2017) TESS-band quadratic coefficients at the host's Teff and log g (`inj_ld_source`) | Transit shape matches the host star |
| Orbits | Circular | 30% eccentric: e from Beta(0.867, 3.03) (Kipping 2013), uniform argument of periastron, periastron beyond 2 stellar radii (`inj_ecc`, `inj_w_deg`) | Realistic spread of durations at a given period |

The Claret table is VizieR J/A+A/600/A30, table25 (least-squares quadratic coefficients), cut to solar metallicity and 2 km/s microturbulence: 476 grid points, Teff 3,500-50,000 K, log g 0-5. A host gets the nearest grid point, with Teff scaled by 250 K and log g by 0.5 dex; log g comes from the radius and the mass-radius relation the injections already use. For a Sun-like star (5,800 K, log g 4.5) u1 = 0.31, u2 = 0.27. The table is committed at `finetune_dataset/data/ld_claret2017_tess.parquet` because the CDS hosts time out from GitHub's runners; `limb_darkening.py --fetch` rebuilds it from the Harvard VizieR mirror (`fetch-limb-darkening.yml`).

`assemble.py` now writes two more entries to `summary.json`: `injection_physics` (achieved / target SNR, eccentric share and limb-darkening sources per dataset, model and deep flag) and `eval_fingerprint` (SHA-256 over lc-eval's sample ids, injected SNRs, periods and stacked SNRs), so a later build can show that lc-eval did not change.

**Local trial (3 files, 1% scale).** 459 batman rows. Achieved / target combined SNR: median 1.000 (10th and 90th percentiles also 1.000). Deep tail: 43 rows (9.4%) at combined SNR 75 / 166 / 266; 9 of them hit the planet-size cap of 0.3 stellar radii on noisy stars and stop short of target. Eccentric: 32%. lc-eval rows built by the v3 code are byte-identical to v2's apart from the new, empty columns.

**First full build, run 44270171416 (commit aeb9db8).** The fetch of the Claret table failed (CDS timeouts), so every v3 batman row fell back to the Kipping draw (`inj_ld_source = kipping2013`). Everything else came out as planned and all four gates passed:

| Check | Result |
| --- | --- |
| Rows | 540,374, the same per category as v2 |
| lc-finetune batman, main | 158,338 rows; achieved / target 1.000; 30.0% eccentric |
| lc-finetune batman, deep tail | 17,624 rows (10.0% of batman); achieved / target 1.000; 29.9% eccentric |
| lc-eval batman | 40,000 rows; achieved / target 0.834, circular, v2 limb darkening: unchanged |
| Labels (median negative minus transit patch flux, sigma) | lc-finetune batman 2.79 (v2: 1.97, the deep tail and the correction); every lc-eval group identical to v2 (batman 1.36, transplant 0.70, real long 2.60, real short 1.51) |
| lc-eval fingerprint | `80d955edc1cd257eae8a23a2d3a811aead9dded7dbaaa90b756673cf97b81ce5` |

Kepler materialize jobs took 6.4 minutes on average (v2: 1.6) and two took 27 and 45 minutes. The injection code does not explain it (a v3 injection costs about 7 ms locally against 4 ms for v2, with no slow outliers in 3,000 draws); the likeliest cause is slow Hugging Face downloads while the pretraining jobs were reading from the same repos. The log does not time the download separately.

**Second full build, run 44279662349 (commit 96d7eef), the published v3.** Same code, with the table committed; the fetch step in the choose job also reached the Harvard mirror this time. Published 2026-11-18 11:41 UTC; all four gates passed.

| Check | Result |
| --- | --- |
| Limb darkening | `claret2017` for all 175,962 lc-finetune batman rows |
| lc-finetune batman, main / deep tail | 158,338 / 17,624 rows (10.0%); achieved / target 1.000 for both; 30.1% / 29.5% eccentric |
| lc-finetune batman combined SNR | 3.70 / 13.28 / 54.81 (10th / 50th / 90th percentiles) |
| Labels | lc-finetune batman 2.76 sigma; lc-eval groups identical to v2 |
| Injection gate | measured stacked / expected combined SNR 0.945; expected / target 1.000; 289,233 rows |
| lc-eval fingerprint | `80d955ed...b81ce5`, the same as the first v3 build, and lc-eval's per-category counts, label medians and achieved / target (0.834) equal v2's |

## Build log

The full build produced 540,374 rows and passed all four checks. It reached Hugging Face on the third assembly attempt, after two fixes to the upload path; the rows never changed.

| Run | Scope | Result |
| --- | --- | --- |
| Local trial (sandbox) | 3 small pass-1 files, scale 1% | 2,494 rows; labels line up with dips; measured / intended SNR 1.07; led to the template, host-order and eval-gate fixes in the change log |
| Actions trial, run 43808974918 | FFI s0010, s0040; 2-min s0010, s0040; Kepler q05; scale 2% | 15,411 rows, all four checks passed (SNR ratio 1.029); upload failed: `upload_large_folder` missing in huggingface\_hub 2.x |
| Full build, run 43810358608 | All 206 files, scale 1.0 | 206 scans and 206 materialize jobs succeeded; 540,374 rows; checks passed; upload refused: unquoted colon in the card titles' YAML |
| Assembly rerun | Rows of run 43810358608 | Run 43817596888 (commit c585d5a) stopped at GitHub's secondary rate limit while fetching 206 artifacts. Run 43818162469 (commit c98f621) passed all four checks again and uploaded all three datasets. Run at commit 0d3792a republishes with a corrected lc-benchmarks usage example |

Full-build checks:

| Gate | Result |
| --- | --- |
| Row accounting | PASS: 540,374 rows for 540,374 recipes |
| Leakage | PASS: 0 stars in both lc-finetune and lc-eval; 0 benchmark stars in lc-finetune |
| Labels (negative minus transit patch flux, median, sigma) | PASS: finetune batman 2.73, transplant 1.80, real long 1.90, real short 2.06; eval batman 1.85, transplant 1.20, real long 2.63, real short 1.51 |
| Injections | PASS: measured / intended per-transit SNR median 1.023 over 314,213 rows |

Scan totals: 636,822 event windows, 1,194,213 host windows, 372,967 transit cutouts, 72,810 benchmark windows. Templates accepted: 3,195 for lc-finetune (2,344 TESS, 851 Kepler) and 193 for lc-eval (135 TESS, 58 Kepler).

Rows by category:

| Category | lc-finetune train | lc-finetune validation | lc-eval val | lc-eval test |
| --- | --: | --: | --: | --: |
| `real_planet_short` | 33,777 | 1,993 | 1,641 | 1,505 |
| `real_planet_long` | 6,713 | 308 | 2,129 | 2,112 |
| `injected_batman` | 189,510 | 11,499 | 20,000 | 20,000 |
| `injected_transplant` | 60,000 | 3,600 | 5,000 | 5,000 |
| `negative_random` | 70,000 | 4,200 | 20,000 | 20,000 |
| `negative_eb` | 40,000 | 2,400 | 1,220 | 1,268 |
| **Total** | **400,000** | **24,000** | **49,990** | **49,885** |

**Shortfall in real transits.** Real windows filled 40,490 of the 90,000 planned train rows (10% of train instead of 22.5%); batman took the rest. Real short-period windows are limited by the cap of 12 windows per star with a visible (SNR >= 4) transit; real long-period windows come only from Kepler DR24 planet candidates with P >= 20 d, after removing the AstroNet test stars. Raising the per-star cap would add repeats of the same planets rather than new ones; see open items.

lc-benchmarks rows (up to 8 windows per signal): astronet\_dr24 12,150 (PC 2,711, AFP 7,688, NTP 1,751); astronet\_triage 695 (PC 40, EB 139, J 516); astronet\_triage\_v2 3,395 (E 460, S 50, B 66, J 1,861, N 32, no consensus 926); salinas2025 259 (171 periodic, 81 single, 7 multi-planet). Coverage per label is in the lc-benchmarks card.

Benchmark coverage (signals with at least one of our windows):

| Benchmark | Covered | By label |
| --- | --: | --- |
| `astronet_dr24` | 1,574 of 1,574 (100%) | PC 343/343, AFP 967/967, NTP 264/264 |
| `astronet_triage` | 695 of 1,658 (42%) | PC 40/49, EB 139/230, J 516/1,379 |
| `astronet_triage_v2` | 1,050 of 2,516 (42%) | E 168/250, S 21/34, B 20/72, J 629/1,610, N 5/15, no consensus 207/535 |
| `salinas2025` | 210 of 213 (99%) | periodic 122/122, single 81/82, multi-planet 7/9 |

The TESS QLP benchmarks are covered at 42%: QLP processes many more stars per sector than TESS-SPOC, which our FFI data comes from. Planet-like signals are covered better than junk (82% of Yu's PCs, 67% of Tey's E).

**Check of the published data.** One train shard (`part-00007-of-00040`, 10,012 rows) was downloaded from Hugging Face and decoded independently: the category shares match the build (batman 47%, transplant 14%, real short 8%, real long 2%, random 18%, EB 11%); rows are mixed across categories in file order; median flux of transit-labeled patches is -2.43 sigma for batman, -1.72 transplant, -1.55 real long, -1.49 real short, against about 0 for negative patches; `summary.json` lists all four gates as passed.

## Change log

Every change to the build, newest first. Times are UTC; commits are on branch `reference-catalogs` of Connor-Codess/MlClub\_PlanetFinders.

| When (UTC) | Commit | Change | Why |
| --- | --- | --- | --- |
| 2026-11-18 09:50 | 96d7eef | Claret (2017) table committed in `finetune_dataset/data`; second v3 build started (run 44279662349) | The first v3 build could not reach VizieR and used the Kipping fallback for every batman row |
| 2026-11-18 09:31-09:45 | 5de4196, a43b70c, 6d118c2, 656f69e | Limb-darkening fetch tries the CDS archive and four VizieR mirrors and queries table25 directly, with retries; keeps Claret's least-squares pair (aLSM, bLSM); `fetch-limb-darkening.yml` | CDS hosts time out from the runners and the unistra votable endpoint returns 404; the Harvard mirror works |
| 2026-11-18 08:48 | aeb9db8 | v3: deep tail (10% of batman rows at combined SNR 60-300), two rounds of SNR correction, Claret limb darkening, 30% eccentric orbits, all for lc-finetune only; `injection_physics` and `eval_fingerprint` in summary.json; 3 new tests. Full build started (run 44270171416) | Wave 1: deep real planets were missed and batman injections reached only 0.83 of their target SNR |
| 2026-11-16 16:59 | 9e1693c | v2: planets verified by stacked SNR >= 7 over the star; real windows selected by stacked in-window SNR >= 4; 30 real windows per star; injections targeted on combined SNR; crops keep a transit of SNR >= 4; stacked\_snr and n\_visible\_transits columns; gates rewritten; 4 new tests (69 pass). Full build started | Review: the per-transit SNR >= 4 cut dropped faint short-period planets found by stacking, and the model should still detect those |
| 2026-11-16 07:45 | 0d3792a | lc-benchmarks usage example opens a data shard and shows the join to benchmark\_targets.parquet; assembly rerun on the same rows | The published example decoded a row of the signal table, which has no light-curve channels (found while checking the published card) |
| 2026-11-16 07:20 | c98f621 | A reused run's row artifacts are fetched one at a time with retries (gh api), not by download-artifact in parallel. README, CLAUDE.md and handoff notes updated | The first assembly-only rerun (run 43817596888) hit GitHub's secondary API rate limit fetching 206 artifacts of another run at once |
| 2026-11-16 06:40 | c585d5a | Card titles quoted in the YAML header; a test parses every generated header; 4xx upload errors are not retried; reuse\_run\_id in build\_request.json reruns only the assembly on an earlier run's artifacts | Run 43810358608 built and checked every row, then Hugging Face refused the card YAML (a pretty\_name with an unquoted colon) |
| 2026-11-16 05:27 | `e9a6eab` | Full build started (`files: null`, `scale: 1.0`). Upload switched from `upload_large_folder` to `upload_folder` with 6 retries and growing back-off | The trial's checks all passed, but its upload failed: huggingface\_hub 2.x has no `upload_large_folder` |
| 2026-11-16 05:17 | `8808a98` | Cards for tess-spoc-ffi-binned, kepler-llc and TESS\_TCES; rewritten lc-heldout card; usage and related-dataset sections on lc-pretrain; `publish-dataset-cards.yml` | Request: a good card for every dataset |
| 2026-11-16 05:14 | `188e640` | `finetune_dataset/` package, `build-finetune-dataset.yml`, 25 offline tests, benchmark rebuilds, `exclude_star_keys` option in the pretraining loader. Trial: 5 files at 2% | Request: fine-tuning and test datasets, plus the papers' test sets for comparison |
| 2026-11-16, before `188e640` | local | Upload deletes the old `data/` folder first | Shard names depend on row counts; a trial's shards would have stayed next to the full build's |
| same | local | Label gate for lc-eval real windows lowered from 1 to 0.25 sigma | lc-eval real windows have no SNR cut on purpose; small planets dip only 0.8 sigma on average |
| same | local | Template: 1 point in 60% of bins (was 2 in every bin); baseline 4 h past the edge (was 2 h) | Zero templates passed in the local trial |
| same | local | Benchmark stars removed from lc-finetune, including its template library | Fair comparison with the published papers |
| same | local | Hosts handed out in the order random negatives, transplants, batman | A host shortage took every host for batman and left transplants empty |
| same | local | Expected in-transit slots counted past window edges | Partial transits at an edge looked fully observed |
| same | local | lc-eval uses held-out stars only | Keeps lc-eval and lc-finetune's validation split star-disjoint |
| same | local | AstroNet split uses archive order (kepid, tce\_plnt\_num), not `rowid` | Our `rowid` is an Oracle string, not their integer index |

## Open items and decisions for the team

- [ ] **Benchmark stars in pretraining.** Their light curves are in lc-pretrain without labels. Current choice: skip them at load time (`WindowSampler(exclude_star_keys=...)`). Alternative: remove them physically from lc-pretrain, a rebuild of about 3 hours.
- [ ] **Merge `reference-catalogs` into `main`.** Card links point to `main`, and the monthly catalog refresh only runs from there.
- [ ] **TESS benchmark coverage.** Yu et al. and Tey et al. used QLP light curves; signals on stars TESS-SPOC did not process have no windows. Decide whether a partial-coverage comparison is enough or QLP light curves should be added.
- [ ] **AstroNet split order.** The rebuild assumes their CSV was in archive order (kepid, tce\_plnt\_num). Confirm against their released TFRecords if a copy turns up (the original link returns 404).
- [ ] **ExoMiner (Valizadegan et al. 2022, Kepler DR25).** Not included: it uses DR25 labels and a by-star split whose lists were not found yet.
- [ ] **Masking for pretraining.** Add the random patch-masking function to `pretrain_dataset/loader.py` (planned 30-40% of patches, some contiguous blocks; loss only on observed hidden slots).
- [ ] **Mix ablations.** Rank the share of short-period real transits (for example 0%, 20%, 40% of real positives) and the single-transit crop share on the lc-eval injection-recovery curve, not on fine-tuning loss.
- [x] Batman depth solve: corrected in v3 (achieved / target 1.000 in lc-finetune); lc-eval keeps the v2 offset on purpose, see the v3 section
