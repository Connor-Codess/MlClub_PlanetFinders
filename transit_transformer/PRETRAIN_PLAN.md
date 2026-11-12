# Pretraining plan: joint Kepler + TESS masked-patch model

## Goal
Pretrain one encoder on unlabeled 28-day windows from both missions, so that it can later be fine-tuned on planet / candidate / false-positive / no-planet labels and evaluated on TESS. The only pretraining task is reconstructing hidden patches.

## Data (built and verified)
| | Train windows | Val | Test |
|---|---|---|---|
| TESS | 1,383,062 | 86,876 | 172,148 + 87,712 test_future |
| Kepler | 348,199 | 20,224 | 41,099 |

- `dataset.MixedPretrainDataset` produces batches like this: `values [B,1344,4]`, `mask [B,1344]`, `time`, `side [B,10]`, `mission`.
- The mission mix is 50/50. Half of the Kepler windows get extra noise added so they match TESS noise levels.
- Loader cost is about 1.7 ms per window per worker, which is about 4–5k windows/s with 8 workers. That is fast enough for the model below. The 200/s seen earlier was worker startup.

## Model: PatchTST-style encoder with masked-patch pretraining
**Tokens**
- **Patches:** 1344 bins / 16 = 84 patches of 8 hours each. A patch's input has 16 bins × 4 channels plus 16 mask bits, giving 80 features. A linear layer maps them to `d_model`. Unfilled bins are zeroed, and the mask bits tell the model they are missing.
- **Side token:** an MLP turns the 10 side features (log σ, centroid scale, has_c, magnitude, Teff, radius, missing flags, fill) into one token placed in front.
- **CLS token:** learned. It is not used in pretraining but is ready for fine-tuning.
- **Position:** a learned embedding over the 84 patch positions. The time grid is uniform, so the patch index is the time.
- **No mission token.**

**Encoder**
- Pre-LayerNorm transformer: `d_model=256`, 8 layers, 8 heads, FFN 1024, dropout 0.1.
- About 6.5M parameters over 86 tokens, which is cheap.

**Masking**
- 50% of the patches that have data are replaced by a learned [MASK] token.
- Half the time the mask is random patches. The other half it is contiguous spans of 2–6 patches (16–48 h), so the model has to infer structure rather than copy from neighbours.
- Patches that are entirely empty are never targets.

**Head and loss**
- A linear layer maps each token to 16 × 3 values: the flux, detrended flux and centroid channels. The err channel is input only.
- Loss: MSE on masked patches, counting only filled bins. The centroid channel is weighted 0.5 so it doesn't dominate. Loss is also logged per channel and per mission.
- Reconstructing a hidden transit isn't possible (it isn't predictable from its neighbours), so a low loss means the model has learned stellar variability, systematics and noise level. That is the useful prior for the classifier.

## Training
- **Optimiser:** AdamW, peak learning rate 5e-4, betas (0.9, 0.95), weight decay 0.05.
- **Schedule:** 2k warmup steps, then cosine decay to 5% of the peak. Gradients clipped at 1.0.
- **Batch:** 256, for 40k steps, which is about 10M windows: about 14 passes over Kepler and 3.6 over TESS. (Cut from 80k after measuring speed: the M5 manages about 520 windows/s in bf16.)
- **Device:** MPS on this M5 (24 GB). bf16 autocast, with the loss computed in fp32. Measured: about 520 windows/s for the model step alone, so roughly 6 hours.
- **Checkpoints:** every 2k steps, saving model, optimiser, scheduler, step and RNG. Runs can resume, and run under `caffeinate -i` like the builds.
- **Logging:** CSV plus TensorBoard, kept local, with no account needed. Weights & Biases can be added later if the club wants shared dashboards.

## Validation (every 2k steps)
1. Masked loss on fixed val batches with the same mask seed. Reported separately for TESS, Kepler, and Kepler with added noise.
2. **Baselines to beat:**
   - Predict 0 (the window median).
   - Linearly interpolate between the nearest visible patches.

   If the model doesn't beat interpolation, it isn't learning anything.
3. **Linear probe every 10k steps:** freeze the encoder, mean-pool it, then fit logistic regression on DR24 PC/AFP/NTP using the Kepler windows that contain a TCE transit.
   - The probe uses Kepler train stars to fit and Kepler val stars to score. Kepler test stars stay untouched.
   - This is the real signal of whether pretraining helps classification. It needs the DR24 TCE table (period, epoch, duration), which is already in `kepler-raw-lightcurves`.
4. Plots: reconstructions of 8 fixed val windows (4 per mission), saved as PNGs.

## Ablations (good use for the second computer)
Each one changes a single setting against the main run:
- **A. TESS-only pretraining.** Was adding Kepler worth it?
- **B. Joint pretraining without centroids** (`zero_centroids=True`). How much do centroids add?
- **C. No noise augmentation.** Does closing the domain gap matter?

All of them are compared on the same linear probe and, later, the same TESS fine-tuning.

## Files to write
- `model.py`: `PatchEncoder`, masking utilities, `MaskedPatchModel` (encoder + head + loss).
- `pretrain.py`: training loop, config through argparse/YAML, checkpointing, resume, logging, validation.
- `probe.py`: TCE window extraction and the linear probe.
- `configs/base.yaml`, plus the ablation configs.

## Verification before the long run
1. Unit checks:
   - Patchify followed by unpatchify returns the input.
   - Masked bins and empty patches add nothing to the loss.
   - The parameter count is about 6.5M.
2. Overfit check: on 64 windows, the loss should drop by more than 10× in a few hundred steps.
3. Smoke run: 500 steps on the real loader. Measure windows/s and MPS memory, then kill and resume to check the loss curve continues.
4. Only then start the 80k-step run.

## Open decisions (defaults above unless changed)
- Hardware on the second computer, which decides whether ablations run in parallel.
- Model size: 6.5M parameters is the default. Scale up to `d_model=384`, 12 layers (about 21M) only if the val loss is still dropping at the end.
- Shared experiment tracking (Weights & Biases) or local logs.
