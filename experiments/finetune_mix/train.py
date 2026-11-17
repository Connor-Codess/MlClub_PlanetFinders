"""Train the detector on one arm with one seed, then score it on lc-eval val.

What it does
    Loads an arm's packed rows (pack.py --arm), trains model.Detector for config.TRAIN_STEPS
    optimizer steps with AdamW and a warmup-plus-cosine learning rate, using a weighted binary
    cross-entropy on patches whose label is 0 or 1 (255 is ignored). Then it scores the
    packed lc-eval val rows (pack.py --eval) with evaluate.score().

Outputs (in --out)
    metrics.json   run settings, training curve, timing and every score from evaluate.py
    scores.npz     per eval window: highest patch probability anywhere and on its transits
    model.pt       final weights

How to run
    python -m experiments.finetune_mix.train --arm-data build/mix/arm --eval build/mix/eval \
        --arm baseline --seed 0 --out build/mix/results/baseline-s0 [--max-minutes 300] [--steps 10000]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import config
from .data import load_pack, model_input
from .evaluate import average_precision, score, window_scores
from .model import Detector, n_parameters

LOG = logging.getLogger("mix.train")


def learning_rate(step: int, total: int) -> float:
    if step < config.WARMUP_STEPS:
        return config.LEARNING_RATE * (step + 1) / config.WARMUP_STEPS
    progress = (step - config.WARMUP_STEPS) / max(total - config.WARMUP_STEPS, 1)
    return config.LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def patch_loss(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    valid = label != config.LABEL_IGNORE
    target = (label == config.LABEL_TRANSIT).float()
    weight = torch.where(target > 0, torch.full_like(target, config.POS_WEIGHT), torch.ones_like(target)) * valid
    loss = F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="sum")
    return loss / valid.sum().clamp(min=1)


@torch.no_grad()
def predict(model: Detector, pack: dict, batch: int = 256) -> np.ndarray:
    model.eval()
    out = np.empty((len(pack["x"]), config.N_PATCHES), dtype=np.float32)
    for start in range(0, len(out), batch):
        sl = slice(start, start + batch)
        x = torch.from_numpy(model_input(pack["x"][sl], pack["observed"][sl]))
        out[sl] = torch.sigmoid(model(x)).numpy()
    return out


def train(arm: dict, steps: int, seed: int, max_minutes: float) -> tuple[Detector, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Detector()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    n = len(arm["x"])
    per_epoch = n // config.BATCH_SIZE
    curve, step, epoch, started, stopped_early = [], 0, 0, time.time(), False
    running, count = 0.0, 0
    model.train()
    while step < steps and not stopped_early:
        order = rng.permutation(n)
        for b in range(per_epoch):
            if step >= steps:
                break
            idx = np.sort(order[b * config.BATCH_SIZE:(b + 1) * config.BATCH_SIZE])
            x = torch.from_numpy(model_input(arm["x"][idx], arm["observed"][idx]))
            label = torch.from_numpy(arm["label"][idx].astype(np.int64))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, steps)
            loss = patch_loss(model(x), label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
            count += 1
            step += 1
            if step % 200 == 0:
                minutes = (time.time() - started) / 60
                curve.append({"step": step, "epoch": epoch, "loss": running / count, "minutes": minutes})
                LOG.info("step %d/%d epoch %d loss %.4f (%.1f min)", step, steps, epoch, running / count, minutes)
                running, count = 0.0, 0
                if minutes > max_minutes:
                    stopped_early = True
                    break
        epoch += 1
    info = {"steps": step, "planned_steps": steps, "epochs_seen": step * config.BATCH_SIZE / n,
            "train_minutes": (time.time() - started) / 60, "stopped_early": stopped_early, "curve": curve}
    return model, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm-data", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=config.TRAIN_STEPS)
    parser.add_argument("--max-minutes", type=float, default=300.0, help="stop training after this long")
    parser.add_argument("--eval-limit", type=int, help="score only the first N eval rows (smoke tests)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(config.TORCH_THREADS)

    arm = load_pack(args.arm_data)
    LOG.info("arm %s seed %d: %d training rows", args.arm, args.seed, len(arm["x"]))
    rows = len(arm["x"])
    model, info = train(arm, args.steps, args.seed, args.max_minutes)
    del arm

    evals = load_pack(args.eval)
    if args.eval_limit:
        evals = {k: v[: args.eval_limit] for k, v in evals.items()}
    started = time.time()
    patch = predict(model, evals)
    any_score, on_target = window_scores(patch, evals["target"], evals["label"])
    result = score(evals["meta"], any_score, on_target, average_precision(patch, evals["label"]))
    info["eval_minutes"] = (time.time() - started) / 60
    result["run"] = {"arm": args.arm, "seed": args.seed, "rows": rows, "parameters": n_parameters(model),
                     **info}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(args.out / "scores.npz", any_score=any_score.astype(np.float32),
                        on_target=on_target.astype(np.float32))
    torch.save(model.state_dict(), args.out / "model.pt")
    point = result["operating_points"]["0.01"]
    LOG.info("result at 1%% false alarms: %s", json.dumps({k: v for k, v in point.items() if k != "bins"}))


if __name__ == "__main__":
    main()
