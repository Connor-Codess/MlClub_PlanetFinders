"""Every number of proxy pretraining.

What it is
    The pretraining-data variants, the lc-pretrain files each one reads, how many windows
    are packed, the masking task and the loader steps from the build guide (Stage 0 default
    loader), and the training budget.

Why these numbers
    The proxy encoder has about 0.35M parameters. 400,000 windows (65M patch tokens) and
    12,000 steps of batch 64 (1.9 passes) are plenty for it and fit one 6-hour GitHub job
    together with the packing. Files are spread across TESS cycles and Kepler quarters so the
    sample covers different pointings, cameras and years.

How to use
    Imported by pack.py and pretrain.py; nothing to run.
"""

from __future__ import annotations

PRETRAIN_REPO = "nyu-planet-finders/lc-pretrain"
BENCHMARK_REPO = "nyu-planet-finders/lc-benchmarks"

FFI_FILES = [f"data/ffi/train/s{s:04d}.parquet" for s in (3, 13, 23, 33, 43, 53, 63, 73)]
TWO_MIN_FILES = [f"data/2min/train/s{s:04d}.parquet" for s in range(5, 101, 5)]
KEPLER_FILES = ["data/kepler/train/q04.parquet", "data/kepler/train/q12.parquet"]
FILES = {"ffi": FFI_FILES, "2min": TWO_MIN_FILES, "kepler": KEPLER_FILES}

WINDOWS = 400_000
VALIDATION_SHARE = 0.02
SAMPLE_SEED = 20261118

# Share of packed windows per source. pt_mix follows lc-pretrain's stored weights (50/25/25).
VARIANTS = {
    "pt_mix": {"ffi": 0.50, "2min": 0.25, "kepler": 0.25},
    "pt_ffi": {"ffi": 1.00},
    "pt_tess": {"ffi": 2 / 3, "2min": 1 / 3},
}

# Loader steps (build guide, Stage 0). flux is asinh-compressed by the shared model input.
KEPLER_DEGRADE_SHARE = 0.5          # step 4: Kepler windows given FFI-level noise ...
KEPLER_GAP_SLOTS = 48               # ... and a 1-day gap in the middle
EXTRA_GAP_SHARE = 0.2               # step 5: windows given one or two extra gaps
EXTRA_GAP_SLOTS = (24, 96)          #         of 0.5-2 days
MASK_RATIO = 0.35                   # step 6: share of patches hidden ...
MASK_SPAN_PATCHES = (1, 6)          #         in runs of 1-6 patches (4-24 h)
FLUX_ONLY_SHARE = 0.25              #         hiding only the flux channel in 25% of batches

STEPS = 12_000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
EVAL_EVERY = 1_000
SEED = 0
