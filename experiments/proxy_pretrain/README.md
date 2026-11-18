# Proxy pretraining

Pretrains the encoder of the small detector used in the data mix experiments
([`experiments/finetune_mix/model.py`](../finetune_mix/model.py), about 0.35M parameters) by masked
reconstruction of real light curves from [lc-pretrain](https://huggingface.co/datasets/nyu-planet-finders/lc-pretrain).
Fine-tuning runs then start from these weights instead of random ones.

This is a proxy for the full pretraining run, not a replacement. It answers three questions cheaply:

1. Do the data mix rankings change once the model is pretrained?
2. How much does pretraining help at this size?
3. Which pretraining data works best?

It also tests the masking code and the checkpoint handoff before the expensive run.

## Variants

| Variant | Windows by source | Question |
|---|---|---|
| `pt_mix` | 50% TESS FFI, 25% TESS 2-min, 25% Kepler (lc-pretrain's stored weights) | Reference |
| `pt_ffi` | 100% FFI | Does matching the search data help most? |
| `pt_tess` | 67% FFI, 33% 2-min | Is Kepler useful or a mismatch? |

Each variant packs 400,000 windows (2% held out for validation). Windows are drawn in proportion to
lc-pretrain's stored weights, from:

- 8 FFI sectors spread over cycles 1-6 (3, 13, ..., 73)
- 20 2-min sectors (5, 10, ..., 100)
- Kepler quarters 4 and 12

Every star in lc-benchmarks is excluded. lc-eval stars are absent from lc-pretrain by construction.

## The pretraining task

This follows the build guide's Stage 0 default loader:

1. Flux is asinh-compressed (`10 asinh(z / 10)`), the same input transform fine-tuning uses.
2. Half of the Kepler windows get Gaussian noise so their noise matches a random FFI star in the
   pack, plus a 1-day gap in the middle (`experiments/finetune_mix/augment.py`).
3. 20% of windows lose one or two extra spans of 0.5-2 days.
4. 35% of the 4-hour patches are hidden, in runs of 1-6 patches. In 25% of batches only the flux is
   hidden, so centroids and background stay visible; otherwise all channels are hidden.
5. The model predicts the flux of every observed slot in the hidden patches (mean squared error).

Training: AdamW, learning rate 1e-3 with 500 warmup steps and cosine decay, batch 64, 12,000 steps.
That is 1.9 passes over the training windows, about 2-2.5 hours on a 4-CPU runner.

Validation reports the masked loss on the held-out windows, next to the loss of predicting zero (the
flux median). A model that has learned anything must beat that floor.

The per-window metadata token from the build guide (noise level, magnitude, temperature, source) is
not used; the proxy keeps the fine-tuning model's architecture unchanged.

## Files

| File | What it does |
|---|---|
| `config.py` | Variants, files, window count, masking and loader settings, training budget |
| `pack.py` | Samples and packs one variant's windows (`x.npy`, `observed.npy`, `meta.parquet`) |
| `pretrain.py` | Masked-reconstruction training; writes `<variant>.pt` and `metrics.json` |
| `run_request.json` | Pushing a change to this file starts `.github/workflows/proxy-pretrain.yml` |

## How to run

On GitHub Actions: edit `run_request.json` and push, or start "Proxy pretraining" from the Actions
tab. The request can limit the variants, steps or windows. Each variant is one job: about 30-45
minutes of packing, then the training. The checkpoints are artifacts named `pretrained-<variant>`.
To use them, set `pretrain_run_id` in `experiments/finetune_mix/run_request.json` to that run's ID.

Locally (needs about 6 GB of disk per variant and `HF_TOKEN` with access to lc-benchmarks):

```bash
pip install -r experiments/finetune_mix/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m experiments.proxy_pretrain.pack --variant pt_mix --out build/pretrain/pt_mix
python -m experiments.proxy_pretrain.pretrain --data build/pretrain/pt_mix --variant pt_mix --out build/pretrain/out
```

Tests: `python -m pytest tests/test_proxy_pretrain.py -q`.

## Change log

| Date | Change |
|---|---|
| 2026-11-18 | First version for wave 2 of the data mix experiments |
