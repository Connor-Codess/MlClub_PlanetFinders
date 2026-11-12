# Fine-tuning plan: classify Kepler TCEs and TESS TOIs into 4 classes

## Goal
Start from the pretrained encoder (`runs/base/ckpt-040000.pt`). Train a classifier that labels each **transit signal** (a Kepler TCE or a TESS TOI, each with its own period and epoch) as **planet / candidate / false positive / no planet**. Evaluate on held-out TESS stars, including the newest sectors (≥100), which stand in for live data.

Labels belong to a signal, not to a window. A star can have several signals, and one signal appears in many windows. So each training example is **one signal, built from all of that star's windows in its split**.

## Label sources (checked 2026-11-11)
| Source | Rows | Has period/epoch? |
|---|---|---|
| `nyu-planet-finders/TESS_TCES` (TOI table, Nov 9) | 8,151 TOIs | **No.** Only toi, tid, disposition |
| NASA Exoplanet Archive `toi` table (public TAP) | 8,151 TOIs | Yes: period, epoch (BJD), duration, depth, Tmag, Teff, R★. 111 have no period |
| Kepler DR24 TCE (`kepler-raw-lightcurves/tces`) | 20,367 TCEs | Yes |
| NASA Exoplanet Archive `cumulative` KOI table | 9,564 KOIs | Yes, and dispositions CONFIRMED / CANDIDATE / FALSE POSITIVE |

The two archive tables are saved in `data/labels/`. Suggestion: upload the TOI table with ephemerides to `TESS_TCES` so the club has one copy.

## Class mapping
| Class | Kepler (DR24 + KOI match on kepid and period within 0.1%) | TESS (TFOPWG) |
|---|---|---|
| **planet** | DR24 PC whose KOI is CONFIRMED (2,654) | CP + KP (1,425) |
| **candidate** | the other DR24 PCs (~946) | PC + APC (5,307) |
| **false positive** | DR24 AFP (9,596) | FP (1,319) |
| **no planet** | DR24 NTP (2,541) | FA (98) + random-ephemeris negatives (see below) |

DR24 `UNK` rows (902) are left out.

**TESS has almost no "no planet" labels.** There are only 98 FA. By default I'll add negatives drawn from TESS stars that have **no TOI**: give each one a random period (0.5–20 d), epoch and duration, and fold it. These are easy negatives, so they will make the "no planet" scores look better than they really are. I'll report them separately. A better source for later is the full SPOC DV TCE lists on MAST, which include signals that never became TOIs.

## Coverage of TESS in our shards
- 6,984 of 8,151 TOIs (86%) are on stars in the 2-min shards. The other 1,167 are FFI-only; `tess-spoc-ffi-binned` could add them later.
- The median TOI star has 4 windows (sectors).

| TESS split | planet | candidate | FP | FA |
|---|---|---|---|---|
| train | 1,189 | 3,725 | 895 | 70 |
| val | 64 | 214 | 46 | 7 |
| **test** | 146 | 453 | 93 | 14 |
| **test_future** (star first seen at sector ≥100) | 5 | 59 | 1 | 1 |

The splits are the **same star-level splits as pretraining**, so no test star was ever seen in pretraining or fine-tuning. The TESS test set is small: 93 FPs means the FP metrics will have wide error bars, so I'll bootstrap confidence intervals. For training stars, windows from sectors ≥100 are not used at all.

**Kepler / TESS overlap:** some TOIs are on Kepler stars, including known Kepler planets. A TOI within 2″ of a KIC star in `kepler-raw-lightcurves` moves to "excluded" if that Kepler star is in Kepler train. I'll use RA/Dec from the archive and from the light-curve rows for this. It closes the overlap hook that is still open in `splits.py`.

## Inputs per signal
Built by a new `build_examples.py` from the existing shards. No new downloads are needed.

1. **Encoder windows.** Take up to 8 of the star's windows that contain at least one transit of this signal. Prefer windows with the most transits. Scale them exactly as in pretraining, and pass the transit-centre bin indices along with them.
2. **Folded views**, built from all of the star's windows in its split:
   - **Global view:** 201 phase bins over the full orbit.
   - **Local view:** 81 bins over ±3 transit durations.
   - Each view has detrended flux and centroid shift, plus a count of points per bin. The detrending masks the in-transit bins first, so long transits aren't eaten by the 1-day running median.
   - **Odd/even local views** (to catch eclipsing binaries at twice the period).
   - **Secondary-eclipse view:** a local view at phase 0.5.
   - Each view is normalised by the measured depth, and log depth is given separately.
3. **Scalar features:**
   - log period, log duration, log depth (ppm)
   - depth / σ, number of transits observed
   - magnitude, Teff, R★
   - the pretraining side tokens (log σ, centroid scale)

Folding window-binned data loses nothing important: bins are 30 min and transits last 1–15 h.

## Model
```
encoder windows (≤8) -> pretrained PatchEncoder -> for each window: [CLS, tokens of the transit patches]
                     -> attention pooling across windows ---------------------------------> 256
global/local/odd/even/secondary views -> small 1-D CNN (Astronet-style) -------------------> 256
scalar features -> MLP ----------------------------------------------------------------------> 64
concat -> MLP(576 -> 256 -> 4) -> softmax over planet / candidate / FP / no planet
```
- Classification-only additions: about 1.5M parameters on top of the 6.4M encoder.
- **No mission token.** Instead there is a "has centroid" flag, and Kepler noise augmentation stays on, so the model treats both missions the same way.

## Training
- **Stage 1, linear and head warm-up:** freeze the encoder and train everything else for 5 epochs at lr 1e-3. This is fast, because the encoder outputs can be cached once (a few minutes).
- **Stage 2, full fine-tune:**
  - Unfreeze the encoder with layer-wise lr decay of 0.75 (top layer 1e-4) and 10 epochs.
  - Use AdamW, weight decay 0.05, cosine schedule, early stopping on TESS val macro-F1.
- **Batches:** 50% Kepler / 50% TESS signals, with class-balanced sampling inside each mission.
  - Cross-entropy with label smoothing 0.05.
  - "candidate" counts with weight 0.5, because those labels are uncertain by definition.
- **Augmentation:**
  - Random subset and order of encoder windows.
  - Small epoch jitter (±10% of the duration).
  - Kepler noise injection to TESS σ, as in pretraining.
  - Randomly drop the centroid channel 10% of the time.
- **Cost estimate:**
  - About 25k signals per epoch, with up to 8 windows each.
  - Stage 1 cached: minutes. Stage 2: about 15–20 min per epoch in bf16, so about 3–4 h in total.
  - I'll measure on a short run first.

## Evaluation (TESS only, never used for any choice)
- Headline numbers on **TESS test** and **test_future**:
  - Macro-F1 and balanced accuracy over the 4 classes.
  - One-vs-rest PR-AUC for each class.
  - The confusion matrix.
  - Bootstrap 95% CIs.
- The practical question: **FP vs (planet + candidate)**, as recall at 90% precision.
- "No planet" scored two ways: on real FA labels only, and with the random-ephemeris negatives.
- Also scored on Kepler test stars, as a secondary number.

## Comparison runs (same data and settings; only the encoder differs)
1. **Pretrained encoder, fine-tuned** (main).
2. **Random-init encoder**, same architecture. Is pretraining worth it?
3. **No encoder:** folded views and scalars only (Astronet-like). Does the time-series encoder add anything at all?
4. **Pretrained encoder, kept frozen** (stage 1 only).

If (1) doesn't beat (3) on TESS test, the encoder isn't earning its place, and that is worth knowing before scaling up.

## Files to write
- `labels.py`: load and merge DR24 + KOI + TOI tables, map them to 4 classes, cross-match KIC↔TIC, and assign splits.
- `build_examples.py`: write the per-signal example shards (window references + folded views + scalars) to `data/examples/`.
- `finetune_model.py`: classifier on top of `PatchEncoder`.
- `finetune.py`: training loop. It reuses the checkpointing and logging from `pretrain.py`, and runs stage 1 then stage 2.
- `evaluate.py`: test-set metrics with bootstrap CIs, confusion matrices and plots.

## Checks before the long runs
- Fold a few known planets (e.g., a hot Jupiter TOI and a Kepler CONFIRMED planet) and plot the local view. The transit should sit at phase 0 with the expected depth.
- No signal's star appears in more than one split, and no TESS test star is within 2″ of a Kepler training star.
- Overfit 64 signals to near-zero loss.
- A short stage-2 run to measure speed.

## Defaults I'll use unless told otherwise
- Kepler PCs that later became CONFIRMED count as **planet**.
- TESS "no planet" = FA + random-ephemeris negatives on non-TOI stars, about 2,000 of them, matching the Kepler NTP share.
- FFI-only TOIs (1,167) are left out for now.
- The headline metric is TESS-test macro-F1, with FP-vs-planet recall at 90% precision reported next to it.
