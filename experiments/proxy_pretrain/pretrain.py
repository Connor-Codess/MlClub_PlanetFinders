"""Pretrain the proxy detector's encoder by masked reconstruction of real light curves.

What it does
    Loads the windows packed by pack.py and trains the encoder of
    experiments.finetune_mix.model.Detector (the same 0.35M-parameter network the data mix
    experiments fine-tune) plus a small reconstruction head. Each batch goes through the
    build guide's default loader steps (Stage 0):

        Kepler windows    half get Gaussian noise to the level of a random FFI star in the
                          pack, and a 1-day gap in the middle
        extra gaps        20% of windows lose one or two spans of 0.5-2 days
        masking           35% of patches are hidden in runs of 1-6 patches (4-24 hours); in
                          25% of batches only the flux channel is hidden, otherwise all
                          channels and the observed mask
        target            the flux (asinh-compressed, as the model sees it) of every observed
                          slot in a hidden patch; mean squared error

    No labels, catalog tags or injected transits are used.

Outputs (in --out)
    <variant>.pt   encoder and head weights; finetune_mix's load_encoder() keeps the encoder
    metrics.json   settings, training curve, and validation loss next to the loss of
                   predicting zero (the flux median), the floor any useful model must beat

How to run
    python -m experiments.proxy_pretrain.pretrain --data build/pretrain/pt_mix --variant pt_mix \
        --out build/pretrain/out [--steps 12000] [--max-minutes 280]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from experiments.finetune_mix import config as mix
from experiments.finetune_mix.augment import degrade_kepler
from experiments.finetune_mix.data import compress, model_input
from experiments.finetune_mix.model import Detector, n_parameters

from . import config

LOG = logging.getLogger("pretrain.train")


class Pretrainer(Detector):
    """The detector's encoder with a head that predicts the flux of every slot in a patch."""

    def __init__(self) -> None:
        super().__init__()
        self.recon = nn.Linear(mix.D_MODEL, mix.PATCH_SLOTS)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.recon(self.encode(x)).reshape(len(x), mix.WINDOW_SLOTS)


def patch_mask(rng: np.random.Generator, n: int) -> np.ndarray:
    """(n, 162) bool: about MASK_RATIO of the patches, in runs of 1-6."""
    mask = np.zeros((n, mix.N_PATCHES), dtype=bool)
    lo, hi = config.MASK_SPAN_PATCHES
    for i in range(n):
        while mask[i].mean() < config.MASK_RATIO:
            start = int(rng.integers(mix.N_PATCHES))
            mask[i, start:start + int(rng.integers(lo, hi + 1))] = True
    return mask


def add_gaps(observed: np.ndarray, sources: np.ndarray, degraded: np.ndarray, rng: np.random.Generator) -> None:
    """In place: the 1-day middle gap of degraded Kepler windows, and extra gaps in 20% of windows."""
    mid = mix.WINDOW_SLOTS // 2
    half = config.KEPLER_GAP_SLOTS // 2
    observed[(sources == "kepler") & degraded, mid - half:mid + half] = 0
    lo, hi = config.EXTRA_GAP_SLOTS
    for i in np.flatnonzero(rng.uniform(size=len(observed)) < config.EXTRA_GAP_SHARE):
        for _ in range(int(rng.integers(1, 3))):
            length = int(rng.integers(lo, hi + 1))
            start = int(rng.integers(0, mix.WINDOW_SLOTS - length))
            observed[i, start:start + length] = 0


def make_batch(pack: dict, idx: np.ndarray, rng: np.random.Generator, ffi_sigmas: np.ndarray, flux_only: bool):
    x = pack["x"][idx].astype(np.float32)
    observed = pack["observed"][idx].astype(np.uint8)
    sources = pack["sources"][idx]
    factors = degrade_kepler(x, observed, sources, pack["sigmas"][idx], ffi_sigmas, rng,
                             share=config.KEPLER_DEGRADE_SHARE)
    add_gaps(observed, sources, factors > 1, rng)
    target = compress(x)[:, 0] * observed
    hidden = np.repeat(patch_mask(rng, len(idx)), mix.PATCH_SLOTS, axis=1)
    weight = (hidden & (observed > 0)).astype(np.float32)
    inputs = model_input(x, observed)
    if flux_only:
        inputs[:, 0] = np.where(hidden, 0.0, inputs[:, 0])
    else:
        inputs[:, :-1] = np.where(hidden[:, None, :], 0.0, inputs[:, :-1])
        inputs[:, -1] = np.where(hidden, 0.0, inputs[:, -1])
    return torch.from_numpy(inputs), torch.from_numpy(target.astype(np.float32)), torch.from_numpy(weight)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2 * weight).sum() / weight.sum().clamp(min=1)


@torch.no_grad()
def validate(model: Pretrainer, pack: dict, idx: np.ndarray, ffi_sigmas: np.ndarray) -> dict:
    """Loss on held-out windows with fixed masks, and the loss of predicting zero."""
    model.eval()
    rng = np.random.default_rng(12345)
    totals = {"model": 0.0, "zero": 0.0, "weight": 0.0}
    for start in range(0, len(idx), 256):
        x, target, weight = make_batch(pack, idx[start:start + 256], rng, ffi_sigmas, flux_only=False)
        pred = model.reconstruct(x)
        totals["model"] += float(((pred - target) ** 2 * weight).sum())
        totals["zero"] += float((target ** 2 * weight).sum())
        totals["weight"] += float(weight.sum())
    model.train()
    return {"loss": totals["model"] / max(totals["weight"], 1), "zero_loss": totals["zero"] / max(totals["weight"], 1)}


def learning_rate(step: int, total: int) -> float:
    if step < config.WARMUP_STEPS:
        return config.LEARNING_RATE * (step + 1) / config.WARMUP_STEPS
    progress = (step - config.WARMUP_STEPS) / max(total - config.WARMUP_STEPS, 1)
    return config.LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=config.STEPS)
    parser.add_argument("--max-minutes", type=float, default=280.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(mix.TORCH_THREADS)
    torch.manual_seed(config.SEED)
    rng = np.random.default_rng(config.SEED)

    meta = pd.read_parquet(args.data / "meta.parquet")
    pack = {"x": np.load(args.data / "x.npy", mmap_mode="r"), "observed": np.load(args.data / "observed.npy", mmap_mode="r"),
            "sources": meta["source"].to_numpy(), "sigmas": meta["sigma_p2p_ppm"].to_numpy(dtype=np.float64)}
    ffi_sigmas = pack["sigmas"][(pack["sources"] == "ffi") & np.isfinite(pack["sigmas"])]
    if len(ffi_sigmas) == 0:
        raise SystemExit("no FFI windows in the pack to take noise levels from")
    validation = np.flatnonzero(meta["validation"].to_numpy())
    training = np.flatnonzero(~meta["validation"].to_numpy())
    LOG.info("%s: %d training and %d validation windows; sources %s", args.variant, len(training), len(validation),
             meta["source"].value_counts().to_dict())

    model = Pretrainer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    curve, evals, started, step, running, count = [], [], time.time(), 0, 0.0, 0
    stopped_early = False
    while step < args.steps and not stopped_early:
        order = rng.permutation(training)
        for b in range(len(order) // config.BATCH_SIZE):
            if step >= args.steps:
                break
            idx = np.sort(order[b * config.BATCH_SIZE:(b + 1) * config.BATCH_SIZE])
            x, target, weight = make_batch(pack, idx, rng, ffi_sigmas, rng.uniform() < config.FLUX_ONLY_SHARE)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, args.steps)
            loss = masked_mse(model.reconstruct(x), target, weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
            count += 1
            step += 1
            if step % 200 == 0:
                minutes = (time.time() - started) / 60
                curve.append({"step": step, "loss": running / count, "minutes": minutes})
                LOG.info("step %d/%d loss %.4f (%.1f min)", step, args.steps, running / count, minutes)
                running, count = 0.0, 0
                if minutes > args.max_minutes:
                    stopped_early = True
                    break
            if step % config.EVAL_EVERY == 0:
                evals.append({"step": step, **validate(model, pack, validation, ffi_sigmas)})
                LOG.info("validation at step %d: %s", step, evals[-1])
    evals.append({"step": step, **validate(model, pack, validation, ffi_sigmas)})

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / f"{args.variant}.pt")
    (args.out / "metrics.json").write_text(json.dumps({
        "variant": args.variant, "sources": meta["source"].value_counts().to_dict(),
        "training_windows": int(len(training)), "validation_windows": int(len(validation)),
        "parameters": n_parameters(model), "steps": step, "planned_steps": args.steps,
        "stopped_early": stopped_early, "train_minutes": (time.time() - started) / 60,
        "settings": {k: getattr(config, k) for k in ("MASK_RATIO", "MASK_SPAN_PATCHES", "FLUX_ONLY_SHARE",
                                                     "KEPLER_DEGRADE_SHARE", "EXTRA_GAP_SHARE", "BATCH_SIZE",
                                                     "LEARNING_RATE")},
        "curve": curve, "validation": evals}, indent=2))
    LOG.info("final validation: %s", evals[-1])


if __name__ == "__main__":
    main()
