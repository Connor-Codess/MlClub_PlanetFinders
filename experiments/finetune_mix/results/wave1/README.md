# Wave 1 results (2026-11-17)

Workflow run 44000179513, commits efc3705 and 8813549 on `reference-catalogs`: 13 arms × 3 seeds, 10,000 steps
each, scored on lc-eval val. Files here: `summary.md` / `summary.json` / `runs.csv` from `aggregate.py`, and
`extra.json` from `analyze.py`. Each run's weights and per-window scores are in the run's
`result-<arm>-s<seed>` artifacts, which expire after 30 days.

## Findings

1. **The v2 mix (baseline) is the best-balanced arm.** None of the twelve changes improved injection recall,
   real-planet recall and EB false alarms together beyond seed noise.
2. **v2's stacked-SNR rule helps where intended.** On short-period injections whose single transits are
   below SNR 4 but combine to at least 7, baseline finds 20.3% and `v1_rule` 14.3%, a gain of 6.0 points
   (about 3 standard errors).
3. **Very deep real planets are missed.** Real long-period windows above stacked SNR 25 are found 28.7% of
   the time, against 69.6% at SNR 15-25; without EB negatives the figure is 51.0%. Training injections stop
   near combined SNR 50 while EB negatives are deep, so the model treats very deep dips as eclipsing
   binaries. The proposed fix is a tail of deep injections (combined SNR 60-300).
4. **Injections are essential.** Without them injection recall falls from 34.4% to 22.6% and EB false alarms
   double. Real rows add 5.1 points of real short-period recall.
5. **Batman and transplants do different jobs.** Transplants are best on faint repeated short-period
   transits (33.0%); batman covers long periods. Keep both.
6. **EB negatives halve EB false alarms** (11.6% without them, 5.4% with).
7. **TESS-only rows** add 1.1 points of injection recall and 5.5 points on faint short-period transits, but
   lose 1.7 points of real long-period recall, since every real long-period training row is Kepler.
8. **Crops have no measurable effect.** Gains from more rows stop at 80k for this model and compute.
9. **Every trained arm beats the box-filter baseline** (0.6% injection recall at 1% false alarms).

## Headline numbers at 1% random false alarms (%, mean of 3 seeds)

Bold values differ from baseline by more than two standard errors; the difference is in parentheses.

| Arm | Injected | Faint (SNR 4-10) | Strong (SNR ≥ 10) | Real short | Real long | EB false alarms | Patch AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| `box_filter` | 0.6 | 0.3 | 1.1 | 3.9 | 14.2 | 15.7 | 0.147 |
| `baseline` | 34.4 | 17.9 | 74.7 | 36.7 | 31.3 | 5.4 | 0.471 |
| `v1_rule` | **32.9** (-1.5) | 17.0 | **71.7** (-3.0) | 34.9 | 30.2 | 4.6 | **0.460** (-0.011) |
| `no_real` | **35.6** (+1.2) | **19.6** (+1.7) | **76.4** (+1.6) | **31.6** (-5.1) | **33.5** (+2.2) | 4.8 | **0.468** (-0.003) |
| `no_injected` | **22.6** (-11.8) | **7.0** (-10.9) | **53.1** (-21.6) | **49.9** (+13.2) | **39.2** (+7.9) | **10.7** (+5.3) | **0.408** (-0.063) |
| `batman_only` | 33.9 | 17.7 | 73.8 | 33.6 | 32.2 | 4.2 | **0.466** (-0.005) |
| `transplant_only` | **29.6** (-4.7) | **15.3** (-2.6) | **64.4** (-10.3) | **46.7** (+10.0) | **28.1** (-3.3) | **8.0** (+2.6) | **0.462** (-0.009) |
| `no_eb` | **33.5** (-0.9) | **15.2** (-2.7) | 74.7 | **47.3** (+10.6) | **39.4** (+8.0) | **11.6** (+6.2) | 0.471 |
| `no_crop` | 34.0 | 18.0 | **73.7** (-1.0) | 36.1 | 31.5 | **4.7** (-0.7) | 0.469 |
| `tess_only` | **35.5** (+1.1) | **19.8** (+1.9) | **75.9** (+1.2) | 36.6 | **29.6** (-1.7) | 5.5 | 0.473 |
| `strong_injections` | 33.3 | **16.1** (-1.7) | 73.6 | **32.8** (-3.9) | **29.5** (-1.8) | **4.2** (-1.1) | **0.462** (-0.009) |
| `rows_20k` | **29.3** (-5.1) | **12.3** (-5.6) | **66.2** (-8.5) | **31.8** (-4.9) | **25.1** (-6.2) | 5.0 | **0.425** (-0.045) |
| `rows_40k` | **32.8** (-1.6) | **16.6** (-1.2) | **71.6** (-3.1) | 34.1 | **29.6** (-1.8) | 5.1 | **0.457** (-0.013) |
| `rows_160k` | 34.6 | 18.1 | 75.0 | 37.2 | 32.3 | 5.3 | 0.473 |

## Subset analysis (`analyze.py`, %, mean ± sd of 3 seeds)

Subset sizes in lc-eval val:

- faint short-period injections: 1269
- single-transit injections: 12137
- real long-period windows at SNR >= 25: 561

"Matched" columns set the threshold so that 1% of random and EB windows together raise an alarm.

| Arm | Faint short-period | Single-transit injections | Real long, SNR >= 25 | Injected (matched) | Real short (matched) | Real long (matched) | EB false alarms (matched) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `box_filter` | 0.6 | 0.8 | 52.6 | 0.1 | 0.5 | 3.7 | 8.8 |
| `baseline` | 20.3 ± 2.8 | 35.7 ± 0.2 | 28.7 ± 2.7 | 32.6 ± 0.1 | 31.9 ± 1.5 | 26.9 ± 1.4 | 4.0 ± 0.5 |
| `v1_rule` | 14.3 ± 1.9 | 34.5 ± 0.8 | 25.6 ± 0.4 | 31.2 ± 0.4 | 31.0 ± 1.2 | 26.5 ± 1.6 | 3.6 ± 0.1 |
| `no_real` | 25.3 ± 1.2 | 36.7 ± 0.5 | 31.2 ± 2.3 | 33.9 ± 0.4 | 27.9 ± 0.5 | 29.5 ± 1.0 | 3.9 ± 0.1 |
| `no_injected` | 7.8 ± 1.5 | 18.1 ± 1.4 | 38.7 ± 6.6 | 16.9 ± 1.2 | 42.2 ± 0.2 | 28.3 ± 0.9 | 8.2 ± 0.4 |
| `batman_only` | 15.8 ± 2.2 | 36.8 ± 0.8 | 29.7 ± 5.1 | 32.2 ± 0.4 | 29.2 ± 2.5 | 28.0 ± 1.4 | 3.5 ± 0.6 |
| `transplant_only` | 33.0 ± 2.2 | 21.6 ± 1.6 | 15.0 ± 1.3 | 26.4 ± 0.7 | 40.7 ± 1.0 | 20.6 ± 1.7 | 6.6 ± 0.7 |
| `no_eb` | 18.7 ± 3.7 | 33.9 ± 0.1 | 51.0 ± 4.2 | 28.9 ± 0.5 | 38.5 ± 0.9 | 29.4 ± 2.0 | 7.1 ± 0.6 |
| `no_crop` | 18.4 ± 1.6 | 35.7 ± 0.6 | 29.5 ± 0.6 | 32.4 ± 0.1 | 32.6 ± 1.6 | 27.6 ± 0.9 | 3.5 ± 0.3 |
| `tess_only` | 25.8 ± 1.8 | 36.5 ± 0.5 | 26.6 ± 1.2 | 33.6 ± 0.4 | 31.8 ± 0.1 | 25.4 ± 0.9 | 4.0 ± 0.1 |
| `strong_injections` | 16.9 ± 3.4 | 35.1 ± 0.7 | 25.8 ± 2.5 | 31.6 ± 0.8 | 29.4 ± 1.3 | 26.5 ± 1.2 | 3.3 ± 0.2 |
| `rows_20k` | 12.2 ± 1.3 | 30.7 ± 0.3 | 25.0 ± 3.4 | 27.5 ± 0.5 | 28.1 ± 0.4 | 22.2 ± 1.3 | 3.6 ± 0.3 |
| `rows_40k` | 17.1 ± 0.9 | 34.3 ± 0.3 | 26.7 ± 1.4 | 30.9 ± 0.9 | 29.4 ± 2.8 | 25.1 ± 0.8 | 3.6 ± 0.2 |
| `rows_160k` | 20.2 ± 1.6 | 36.1 ± 0.6 | 29.7 ± 3.1 | 32.6 ± 0.3 | 33.2 ± 2.7 | 27.5 ± 3.1 | 4.0 ± 0.2 |

## Caveats

- **Proxy model.** The detector has 0.35M parameters, no pretraining and 10,000 steps on CPU. Rerun the best
  mixes once the pretrained model exists.
- **Real-planet numbers.** lc-eval val has 1,641 real short-period windows from 238 stars and 2,129
  long-period windows from 344 stars. Windows of one star are correlated, so differences of 2 points or less
  there are unresolved.
- **Operating point.** Thresholds are set on random windows, so read real-planet recall alongside EB false
  alarms.
