# Wave 2 results (2026-11-18)

Workflow run 44297108133 on `reference-catalogs` (launched at commit `93dcef0`): 26 arms x 3 seeds, 10,000 steps each on
80,000 rows of lc-finetune v3, scored on lc-eval val. Pretrained arms start from proxy pretraining run 44271393676.
All 78 runs finished. The full write-up, with design and discussion, is the wave 2 paper (Claude doc "Wave 2:
pretraining, deep injections and realism checks").

| File | What |
|---|---|
| `summary.md`, `summary.json`, `runs.csv` | `aggregate.py`: every metric at 1%, 0.1% and 5% random-window false alarms, per run and per arm |
| `extra.json` | `analyze.py`: subsets, matched false alarms, false-alarm curves (`fa<rate>_*`), transit-hiding test and the pre-registered `decision` |
| `pretraining.json` | Proxy pretraining runs: settings and validation curves |
| `realism.json` | `realism.py` on the published lc-finetune v3: distributions, per-star durations, real-versus-injection classifiers |
| `benchmark.md`, `benchmark.json` | `benchmark.py`: the winning arm on lc-benchmarks against the published papers (run 44442425451; the first run, 44392254586, lacked the Salinas AUCs) |

Each run's weights and per-window scores are in its `result-<arm>-s<seed>` artifact (30 days).

## Decision (pre-registered)

The winner has the highest mean of faint recall (combined SNR 4-10), the stacking-regime recall and single-transit faint
recall, all at 1% random-window false alarms, among arms that against the reference `pt_mix-r16-nodeep-all` (the v2
mix, pretrained) lose no more real long-period recall at matched false alarms than twice the standard error of the
difference, and raise EB false alarms by at most 1 point.

**Winner: `pt_mix-r00-deep-all`**: pretrained on `pt_mix`, no real planets among the fine-tuning rows (their share goes
to injections), the deep injection tail kept, every host source. Score 0.272 ± 0.007 against 0.212 for the reference;
real long-period recall at matched false alarms -2.2 points (allowed -2.9); EB false alarms +0.4 points. Runner-up:
`pt_mix-r00-deep-kdeg` (0.263).

## Findings

1. **Fine-tuning without real planets wins the faint regime.** With injections taking the 16% real share, the
   stacking regime (faint repeated short-period transits) rises from 21.1% to 30.7-32.9%, faint recall from 17.9% to
   20.4-21.7% and single-transit faint recall from 24.6% to 26.5-27.2%. Real long-period recall is unchanged
   (38.1-40.1% against 38.7%). The cost is real short-period planets: 31.7-32.8% against 38.1% at 1% false alarms,
   and 26.0-28.1% against 33.1% at matched false alarms, concentrated in bright short-period planets (42.8-44.3% against
   51.5%). A 5% real share sits in between on every measure.
2. **The deep tail fixes deep real planets, as intended, but makes EB-like dips more alarming.** Real long-period
   windows above SNR 25 are found 56.9% of the time with it against 48.4% without (16% real share; wave 1: 28.7%).
   EB false alarms rise 0.9 points, and with no real planets in fine-tuning the matched real long-period recall
   falls 4.0 points, because the matched threshold rises with the EB scores.
3. **Pretraining helps, mostly on real planets.** Against the same mix from scratch: real long-period recall 38.7%
   against 35.2% (16% real share), deep real long-period planets 48.4% against 39.5%, and the decision score
   0.272 against 0.236 for the winning mix. Injection recall barely moves (34.5% against 34.3%). The from-scratch
   arms also vary more between seeds (real short-period sd 3.0 points against 1.5 for the 16% mix).
4. **Pretraining data matters little at this scale.** With the winning mix, `pt_mix` scores 0.272, `pt_ffi` 0.254 and
   `pt_tess` 0.253 (seed sd about 0.01); with a 16% real share 0.216, 0.211 and 0.208. `pt_ffi` is best on deep real
   long-period planets (56.4%). The Kepler share in `pt_mix` is not a handicap; it may help a little.
5. **Kepler hosts are worth keeping.** Taking injected and random-negative rows from TESS only (`tess`) costs little
   with a 16% real share (1-2 points of real short- and long-period recall) but, with no real planets in fine-tuning,
   8 points of real short-period recall and 3-5 points of real long-period recall, while cutting EB false alarms by
   2-3 points. Real planet rows and Kepler-hosted injections seem to be partly interchangeable sources of real-data
   signal. Wave 1's small gain from TESS-only rows does not hold on v3. Degrading half the Kepler rows to FFI noise
   (`kdeg`) is neutral to slightly worse than leaving them alone.
6. **The new input compression and v3 data lift real long-period recall on their own.** The from-scratch v2 mix
   (`scratch-r16-nodeep-all`, asinh flux) against wave 1's baseline (clipped flux, v2 data): real long 35.2% against
   31.3%, real short 38.2% against 36.7%, injections 34.3% against 34.4%.
7. **Injections are still easier than real planets of the same SNR**, but less so. Single-transit batman injections
   with P >= 20 d, weighted to the real long-period planets' SNR, are found 52.3% of the time by the winner against
   38.8% for the real planets themselves (wave 1: 50.7% against 31.4%). The realism checks put the transit shapes
   right; catalog ephemeris offsets in the real labels are the likeliest remaining cause (paper, section 7).
8. **The model stacks transits.** In the transit-hiding test, a kept transit is found 39.9% of the time with its
   sibling transits present and 29.2% when they are hidden; in the faint stacking regime 9.2% against 1.8%.
9. **Classical baselines are not competitive at strict false-alarm rates.** wotan + box search + BLS finds 0.4% of
   injections at 1% false alarms, 10% at 5% and 42% at 30%, where the winner finds 63%. At SNR 7.1, its usual cut,
   27% of random windows alarm.

## Headline numbers at 1% random false alarms (%, mean of 3 seeds)

Bold values differ from the reference arm by more than two standard errors; the difference is in parentheses. Arms are
in decision-score order.

| Arm | Injected | Faint (SNR 4-10) | Strong (SNR >= 10) | Real short | Real long | EB false alarms | Patch AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pt_mix-r16-nodeep-all` | 34.5 | 17.9 | 74.9 | 38.1 | 38.7 | 6.1 | 0.476 |
| `pt_mix-r00-deep-all` | **36.7** (+2.2) | **21.7** (+3.7) | **77.4** (+2.5) | **31.7** (-6.3) | 38.1 | **6.5** (+0.4) | 0.476 |
| `pt_mix-r00-deep-kdeg` | **36.4** (+1.9) | **21.1** (+3.2) | **77.2** (+2.4) | **29.3** (-8.8) | 38.3 | 5.9 | 0.475 |
| `pt_mix-r00-nodeep-all` | **36.5** (+2.0) | **20.4** (+2.5) | **78.2** (+3.3) | **32.8** (-5.3) | 40.1 | **5.3** (-0.8) | **0.471** (-0.005) |
| `pt_mix-r00-nodeep-kdeg` | **36.7** (+2.2) | **20.9** (+3.0) | **78.1** (+3.2) | **31.3** (-6.7) | 38.7 | **4.9** (-1.3) | **0.471** (-0.005) |
| `pt_mix-r00-deep-tess` | **36.4** (+1.9) | **20.7** (+2.8) | **77.6** (+2.7) | **23.5** (-14.6) | **34.8** (-3.9) | **3.4** (-2.7) | **0.466** (-0.010) |
| `pt_ffi-r00-deep-all` | **36.1** (+1.6) | **20.7** (+2.8) | **76.8** (+1.9) | **32.9** (-5.2) | 40.3 | 6.5 | 0.476 |
| `pt_mix-r00-nodeep-tess` | **36.5** (+2.1) | **20.2** (+2.2) | **78.4** (+3.5) | **25.2** (-12.9) | **35.3** (-3.4) | **3.2** (-2.9) | **0.463** (-0.013) |
| `pt_tess-r00-deep-all` | **35.8** (+1.4) | **20.0** (+2.1) | **76.7** (+1.8) | **32.4** (-5.7) | 37.6 | 6.1 | 0.476 |
| `pt_mix-r05-nodeep-kdeg` | **36.3** (+1.8) | **20.5** (+2.6) | **77.4** (+2.5) | **32.6** (-5.5) | 37.0 | **4.6** (-1.5) | **0.480** (+0.004) |
| `pt_mix-r05-deep-tess` | **36.3** (+1.8) | **20.4** (+2.5) | **77.5** (+2.6) | **29.4** (-8.7) | 36.6 | **4.5** (-1.6) | **0.478** (+0.002) |
| `scratch-r00-nodeep-all` | **36.3** (+1.8) | **20.1** (+2.1) | **77.9** (+3.0) | **34.8** (-3.3) | 38.2 | **5.5** (-0.7) | **0.471** (-0.005) |
| `pt_mix-r05-deep-all` | **36.1** (+1.7) | **20.6** (+2.7) | **76.9** (+2.0) | **35.1** (-3.0) | **41.2** (+2.6) | 6.3 | **0.480** (+0.004) |
| `pt_mix-r05-deep-kdeg` | **36.1** (+1.6) | **20.5** (+2.5) | **76.8** (+2.0) | **34.1** (-4.0) | 39.0 | 5.8 | **0.479** (+0.004) |
| `pt_mix-r05-nodeep-all` | **36.3** (+1.8) | **20.2** (+2.3) | **77.7** (+2.8) | **34.0** (-4.1) | 39.2 | 5.1 | **0.480** (+0.004) |
| `pt_mix-r05-nodeep-tess` | **36.3** (+1.9) | **20.0** (+2.1) | **77.9** (+3.0) | **28.9** (-9.2) | 36.9 | **3.5** (-2.6) | **0.479** (+0.003) |
| `scratch-r00-deep-all` | **35.3** (+0.9) | **19.4** (+1.5) | **75.9** (+1.0) | **33.5** (-4.6) | 35.9 | 6.0 | 0.476 |
| `pt_mix-r16-deep-kdeg` | **35.3** (+0.8) | **19.0** (+1.1) | **76.1** (+1.2) | 39.1 | 38.5 | 6.5 | **0.477** (+0.001) |
| `pt_mix-r16-nodeep-tess` | **35.6** (+1.1) | **18.8** (+0.9) | **77.0** (+2.1) | 37.4 | 36.6 | 5.8 | **0.480** (+0.004) |
| `pt_mix-r16-deep-tess` | **35.3** (+0.9) | **18.6** (+0.7) | **76.6** (+1.7) | 37.3 | 38.8 | 6.4 | **0.480** (+0.004) |
| `pt_mix-r16-nodeep-kdeg` | **35.0** (+0.5) | **18.7** (+0.8) | **75.5** (+0.7) | 37.6 | 37.4 | **5.7** (-0.4) | **0.478** (+0.002) |
| `pt_mix-r16-deep-all` | 34.7 | 18.0 | **75.5** (+0.6) | 39.8 | **40.9** (+2.2) | 7.0 | **0.475** (-0.001) |
| `pt_ffi-r16-deep-all` | 34.6 | 17.9 | 75.1 | 40.2 | 39.1 | 6.6 | **0.474** (-0.002) |
| `scratch-r16-nodeep-all` | 34.3 | 18.0 | 74.5 | 38.2 | **35.2** (-3.5) | 6.0 | **0.472** (-0.004) |
| `pt_tess-r16-deep-all` | 34.3 | 17.1 | 75.1 | 39.7 | 38.8 | **7.0** (+0.8) | 0.475 |
| `scratch-r16-deep-all` | 34.3 | 17.6 | 74.6 | 39.6 | 37.8 | **6.6** (+0.5) | **0.471** (-0.005) |
| `classical` | 0.4 | 0.4 | 0.3 | 14.3 | 8.0 | 25.9 | 0.129 |
| `box_filter` | 0.6 | 0.3 | 1.1 | 3.9 | 14.2 | 15.7 | 0.147 |

## Subsets and matched false alarms (%, mean of 3 seeds)

"Matched" columns set the threshold so that 1% of random and EB windows together raise an alarm. Subset sizes in lc-eval
val: stacking regime 1,269; single-transit faint 3,807; real long-period at SNR >= 25, 561.

| Arm | Decision score | Stacking regime | Single-transit faint | Real long, SNR >= 25 | Real long (matched) | Real short (matched) | EB false alarms (matched) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pt_mix-r16-nodeep-all` | 0.212 | 21.1 | 24.6 | 48.4 | 34.1 | 33.1 | 4.6 |
| `pt_mix-r00-deep-all` | **0.272** (+0.060) | **32.9** (+11.7) | **27.2** (+2.6) | 50.9 | 31.9 | **26.0** (-7.1) | **5.3** (+0.7) |
| `pt_mix-r00-deep-kdeg` | **0.263** (+0.051) | **30.6** (+9.5) | **27.3** (+2.7) | 49.1 | 32.5 | **23.4** (-9.7) | 4.9 |
| `pt_mix-r00-nodeep-all` | **0.259** (+0.047) | **30.7** (+9.6) | **26.5** (+2.0) | 48.2 | 35.9 | **28.1** (-5.0) | 4.5 |
| `pt_mix-r00-nodeep-kdeg` | **0.258** (+0.046) | **29.2** (+8.0) | **27.3** (+2.7) | 46.1 | 34.2 | **26.9** (-6.2) | **4.2** (-0.4) |
| `pt_mix-r00-deep-tess` | **0.254** (+0.042) | **29.2** (+8.0) | **26.4** (+1.8) | 46.3 | 32.1 | **21.8** (-11.2) | **2.9** (-1.6) |
| `pt_ffi-r00-deep-all` | **0.254** (+0.042) | **28.0** (+6.9) | **27.5** (+3.0) | **56.4** (+8.0) | 34.2 | **27.6** (-5.4) | **5.4** (+0.8) |
| `pt_mix-r00-nodeep-tess` | **0.253** (+0.041) | **29.8** (+8.7) | **26.0** (+1.5) | 45.8 | 33.2 | **23.7** (-9.3) | **2.8** (-1.8) |
| `pt_tess-r00-deep-all` | **0.253** (+0.041) | **29.9** (+8.8) | 25.9 | 52.6 | 32.2 | **26.8** (-6.3) | 4.7 |
| `pt_mix-r05-nodeep-kdeg` | **0.252** (+0.040) | **28.2** (+7.1) | **26.8** (+2.2) | **42.2** (-6.2) | 33.6 | **28.9** (-4.2) | **3.9** (-0.7) |
| `pt_mix-r05-deep-tess` | **0.251** (+0.039) | **29.1** (+8.0) | **25.9** (+1.3) | 51.9 | 33.2 | **26.5** (-6.5) | **3.6** (-1.0) |
| `scratch-r00-nodeep-all` | **0.249** (+0.037) | **28.5** (+7.4) | **26.2** (+1.7) | 45.0 | 33.5 | 30.8 | 4.6 |
| `pt_mix-r05-deep-all` | **0.249** (+0.037) | **27.5** (+6.4) | **26.6** (+2.0) | **55.9** (+7.4) | 35.2 | **28.9** (-4.2) | **5.1** (+0.5) |
| `pt_mix-r05-deep-kdeg` | **0.247** (+0.035) | **27.0** (+5.9) | **26.7** (+2.1) | 51.1 | 33.6 | **28.9** (-4.2) | 4.8 |
| `pt_mix-r05-nodeep-all` | **0.246** (+0.034) | **27.4** (+6.3) | **26.1** (+1.6) | 45.7 | 34.7 | **29.7** (-3.3) | 3.9 |
| `pt_mix-r05-nodeep-tess` | **0.244** (+0.032) | **27.0** (+5.9) | **26.1** (+1.6) | **44.6** (-3.8) | 35.0 | **26.7** (-6.4) | **3.0** (-1.6) |
| `scratch-r00-deep-all` | **0.236** (+0.024) | **25.7** (+4.6) | **25.5** (+1.0) | 47.1 | **29.5** (-4.5) | **28.2** (-4.8) | 4.7 |
| `pt_mix-r16-deep-kdeg` | **0.233** (+0.021) | **26.3** (+5.1) | 24.5 | 51.7 | 32.7 | 34.4 | **5.2** (+0.7) |
| `pt_mix-r16-nodeep-tess` | **0.233** (+0.021) | **26.6** (+5.4) | 24.4 | 44.0 | 31.0 | 33.0 | 4.5 |
| `pt_mix-r16-deep-tess` | **0.229** (+0.017) | **26.2** (+5.1) | **23.8** (-0.7) | **53.5** (+5.1) | 32.5 | 32.0 | 4.6 |
| `pt_mix-r16-nodeep-kdeg` | **0.221** (+0.009) | 22.2 | **25.4** (+0.8) | 45.5 | 32.6 | 33.0 | 4.7 |
| `pt_mix-r16-deep-all` | 0.216 | 22.8 | 23.9 | **56.9** (+8.5) | 34.2 | 33.1 | **5.5** (+0.9) |
| `pt_ffi-r16-deep-all` | 0.210 | 21.1 | **24.1** (-0.5) | **55.0** (+6.5) | 33.3 | 34.0 | **5.2** (+0.6) |
| `scratch-r16-nodeep-all` | 0.209 | 19.8 | 25.0 | **39.5** (-8.9) | **30.7** (-3.3) | 33.1 | 4.3 |
| `pt_tess-r16-deep-all` | 0.208 | 22.8 | **22.6** (-2.0) | **55.9** (+7.5) | 31.6 | 33.1 | **5.4** (+0.8) |
| `scratch-r16-deep-all` | 0.205 | 19.8 | 24.1 | 51.0 | 32.1 | 34.0 | 5.1 |
| `classical` | 0.003 | 0.3 | 0.3 | 29.9 | 0.4 | 3.4 | 11.7 |
| `box_filter` | 0.004 | 0.6 | 0.1 | 52.6 | 3.7 | 0.5 | 8.8 |

## Pretraining

Validation loss as a fraction of predicting zero (each variant on its own validation windows): `pt_mix` 0.607,
`pt_tess` 0.625, `pt_ffi` 0.636 after 12,000 steps; most of the gain comes in the first 1,000 steps. Details in
`pretraining.json`.

## Realism of the batman injections

At a centre and duration fitted to each cutout, a classifier separates real transits from batman injections at ROC
AUC 0.56, below the 0.58 of the transplant control (real profiles in other stars). Normalized by each star's
central-transit duration, batman durations are 0.94-0.96 against 0.89-0.93 for confirmed Kepler planets in every period
bin; the long-period TOI gap comes from duotransit placeholder periods. Details in `realism.json` and the paper.

## Against the published papers (lc-benchmarks, `pt_mix-r00-deep-all`, 3 seeds, run 44442425451)

| Benchmark | Paper | Their data | Their result | Ours (mean ± sd over seeds) | Test set covered |
|---|---|---|---|---|---|
| `astronet_dr24` | Shallue & Vanderburg 2018 (AstroNet) | Kepler DR24, PC vs AFP+NTP | AUC 0.988; accuracy 0.960 at threshold 0.5 (1,523 test TCEs) | AUC 0.860 ± 0.002; AP 0.608 ± 0.006; PC vs NTP AUC 0.931 ± 0.002; PC vs AFP AUC 0.841 ± 0.003 | 1,559 of 1,574 (99%) |
| `astronet_triage` | Yu et al. 2019 (Astronet-Triage) | TESS QLP sectors 1-5 | Triage (PC+EB vs J): AUC 0.992, AP 0.970. Vetting (PC vs rest): AUC 0.984, AP 0.693 | Triage: AUC 0.579 ± 0.021, AP 0.587 ± 0.010. Vetting: AUC 0.908 ± 0.016, AP 0.568 ± 0.012 | 660 of 1,658 (40%) |
| `astronet_triage_v2` | Tey et al. 2023 (Astronet-Triage-v2) | TESS QLP cycles 1-3, E vs rest | AUC-PR 0.965; recall 97.2% at precision 75.7% | AUC-PR 0.313 ± 0.005; AUC 0.425 ± 0.012 | 1,013 of 2,516 (40%) |
| `salinas2025` | Salinas et al. 2025 | TESS-SPOC FFI sectors 1-26 | AUC 0.88, F1 0.82 on an unreleased 10% split; 213 published candidates | Candidates detected: multi_planet_candidate 14.3%, periodic_candidate 74.0%, single_candidate 85.0%. AUC against TESS non-transit signals 0.968 ± 0.006 (single 0.986 ± 0.004, periodic 0.959 ± 0.005); against lc-eval random windows 0.961 ± 0.005 | 209 of 213 (98%) |

Detection rate by label at 1% random-window false alarms on lc-eval val:

| Benchmark | Label | Signals | Detected |
|---|---|---:|---:|
| `astronet_dr24` | AFP | 967 | 9.7% |
| `astronet_dr24` | NTP | 249 | 0.7% |
| `astronet_dr24` | PC | 343 | 38.8% |
| `astronet_triage` | EB | 139 | 12.7% |
| `astronet_triage` | J | 481 | 1.6% |
| `astronet_triage` | PC | 40 | 51.7% |
| `astronet_triage_v2` | B | 20 | 0.0% |
| `astronet_triage_v2` | E | 168 | 9.9% |
| `astronet_triage_v2` | J | 594 | 0.8% |
| `astronet_triage_v2` | N | 5 | 0.0% |
| `astronet_triage_v2` | S | 21 | 12.7% |
| `astronet_triage_v2` | none | 205 | 0.3% |
| `salinas2025` | multi_planet_candidate | 7 | 14.3% |
| `salinas2025` | periodic_candidate | 122 | 74.0% |
| `salinas2025` | single_candidate | 80 | 85.0% |

How to read this:

- **It is not the same task.** The three AstroNet papers classify a signal they are given (period and epoch) from
  phase-folded views of all its transits. This model scores raw 27-day windows with no ephemeris; a signal's
  score is its probability at the predicted mid-time patches, averaged over its visible transits. Salinas et al.
  are closest: their transformer also reads unfolded single-sector FFI light curves, cut or padded to 1,000
  points at 30-minute cadence (about 21 days).
- **The model is a detector, not a vetter.** It was trained to call eclipsing binaries negative and has never seen
  the papers' other false-positive classes. Benchmarks whose positive class contains eclipsing binaries (Yu et al.
  triage, Tey et al. "E") come out near or below chance for that reason, not because transits are missed. The
  closer comparisons are PC against non-transiting phenomena on DR24 (AUC 0.93), PC against everything in Yu et al.'s
  vetting task (AUC 0.91) and the share of Salinas et al.'s candidates detected.
- **Coverage differs.** Yu et al. and Tey et al. used QLP light curves; only 40% of their test signals have a window in
  our TESS-SPOC data, so the class balance of what we score differs from theirs, which moves AP. DR24 and Salinas et al.
  are almost fully covered.
- **Salinas et al. 2025 is the closest task** (a single-transit search of TESS-SPOC FFIs). At the experiments'
  operating point the winner detects 85.0% of their single-transit candidates and 74.0% of their periodic candidates.
  They publish candidates only, so an AUC needs negatives from elsewhere: on the peak score (highest probability
  within one patch of a transit, the tolerance every experiment uses), their candidates rank above TESS
  non-transit signals from the triage benchmarks at AUC 0.968 (single-transit 0.986, periodic 0.959) and above
  lc-eval random windows at 0.961 (EB windows 0.938). Their own AUC of 0.88 was measured on their held-out
  split, a different question, so the two are not directly comparable. On the exact mid-time patch alone the
  AUC against TESS non-transit signals drops to about 0.81, mostly for single-transit candidates (median
  probability 0.44 at the exact patch, 0.997 within one patch), whose published times are less precise than the
  periodic candidates' fitted ephemerides.
