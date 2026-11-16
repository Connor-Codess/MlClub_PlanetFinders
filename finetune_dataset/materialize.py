"""Stage 3 (one job per source file): turn recipes into finished samples.

What it does
    For every recipe whose window comes from this source file: cut the 27-day
    window out of the pass-1 row, apply the category's change (inject a batman or
    transplanted transit, or hide all but one transit), measure every event in the
    final window, compute the per-patch targets and write the row.

    Output: rows_<dataset>_<split>_<source>_<stem>.parquet, one per dataset and
    split present. The assemble stage shuffles and uploads them.

Why
    Each sample's random draws come from its own seed (fixed in recipes.parquet),
    so any job can be re-run and gives identical rows.

How to run
    python -m finetune_dataset.materialize --plan build/finetune/plan --source ffi --stem s0001 \
        --out build/finetune/rows [--local-root build/trial]
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pretrain_dataset import config as pretrain

from . import config
from .injection import (KIND_FOR_CLASS, Event, apply_model, batman_injection, cut_window,
                        encode, hide_outside, observed_mask, patch_labels, transplant_injection)
from .schema import INJECTION_COLUMNS, SAMPLE_SCHEMA
from .storage import fetch

LOG = logging.getLogger("finetune.materialize")

ROW_COLUMNS = ["segment_id", "star_key", "source", "sector", "quarter", "slot0", "n_slots", "sigma_p2p_ppm",
               "tmag", "kepmag", "teff", "radius", "quality_tier", *pretrain.CHANNELS_FLOAT16,
               *pretrain.CHANNELS_UINT8]
REPO_NAMES = {"pretrain": pretrain.PRETRAIN_REPO, "heldout": pretrain.HELDOUT_REPO}


def load_templates(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    return {r["template_id"]: {**r, "dt_h": np.asarray(r["dt_h"]), "flux": np.asarray(r["flux"])}
            for r in frame.to_dict("records")}


def known_events(recipe: dict) -> list[Event]:
    events = []
    for e in recipe["events"] if recipe["events"] is not None else []:
        events.append(Event(float(e["center_slot"]), float(e["duration_slots"]), KIND_FOR_CLASS[e["cls"]],
                            e["kind"], e["ref"], e["cls"], e["period_d"]))
    return events


def single_transit(window: dict, events: list[Event], rng: np.random.Generator) -> bool:
    """Keep one clearly visible planet transit; hide the slots nearer to any other planet transit.

    Returns False (window unchanged) when no transit reaches SINGLE_TRANSIT_CROP_MIN_SNR, so a crop
    never leaves only a transit too faint to see on its own.
    """
    planets = sorted((e for e in events if e.is_planet), key=lambda e: e.center_slot)
    strong = [e for e in planets if e.observed_fraction >= config.MIN_VISIBLE_FRACTION
              and e.snr >= config.SINGLE_TRANSIT_CROP_MIN_SNR]
    if len(planets) < 2 or not strong:
        return False
    keep = strong[int(rng.integers(len(strong)))]
    index = planets.index(keep)
    lo = (planets[index - 1].center_slot + keep.center_slot) / 2.0 if index > 0 else 0.0
    hi = (planets[index + 1].center_slot + keep.center_slot) / 2.0 if index + 1 < len(planets) else float(config.WINDOW_SLOTS)
    hide_outside(window, lo, hi)
    return True


def stacked_visible(events: list[Event]) -> dict:
    """Stacked measured SNR, sum(snr_i)/sqrt(n), of the strongest planet's visible transits."""
    groups: dict[str, list[float]] = {}
    for e in events:
        if e.is_planet and e.observed_fraction >= config.MIN_VISIBLE_FRACTION:
            groups.setdefault(e.ref, []).append(float(e.snr))
    if not groups:
        return {"stacked_snr": None, "n_visible_transits": 0}
    best = max(groups.values(), key=lambda x: sum(x) / np.sqrt(len(x)))
    return {"stacked_snr": float(sum(best) / np.sqrt(len(best))), "n_visible_transits": len(best)}


def build_sample(recipe: dict, row: dict, templates: dict[str, dict]) -> dict:
    start = int(recipe["start_offset"])
    window = cut_window(row, start)
    observed = observed_mask(window["flags"])
    s = float(row["sigma_p2p_ppm"]) * 1e-6
    rng = np.random.default_rng(int(recipe["seed"]))
    snr_range = config.INJ_SNR_RANGE["finetune" if recipe["dataset"] == "finetune" else "eval"]
    category = recipe["category"]
    params = {name: None for name in INJECTION_COLUMNS}

    events = known_events(recipe)
    for e in events:
        e.measure(window["flux_z"], observed)
    cropped = bool(recipe["single_transit_crop"]) and single_transit(window, events, rng)
    if cropped:
        observed = observed_mask(window["flags"])
        for e in events:
            e.measure(window["flux_z"], observed)

    injection = None
    if category == "injected_batman":
        injection = batman_injection(rng, observed, s, row["radius"], row["teff"], snr_range)
    elif category == "injected_transplant":
        injection = transplant_injection(rng, observed, s, templates[recipe["template_id"]], snr_range)
    window_slot0 = int(row["slot0"]) + start
    t_start = pretrain.GRID_EPOCH_BTJD + window_slot0 * pretrain.SLOT_DAYS
    if injection is not None:
        window["flux_z"] = apply_model(window["flux_z"], observed, injection.model, s).astype(np.float32)
        events = injection.events
        for e in events:
            e.measure(window["flux_z"], observed)
        params.update(injection.params)
        params["inj_t0_btjd"] = t_start + params.pop("inj_t0_slot") * pretrain.SLOT_DAYS

    labels = patch_labels(events)
    sample = {
        "sample_id": recipe["sample_id"], "dataset": recipe["dataset"], "split": recipe["split"],
        "category": category, "star_key": row["star_key"], "source": row["source"],
        "segment_id": row["segment_id"], "sector": row["sector"], "quarter": row["quarter"],
        "window_slot0": window_slot0, "t_start_btjd": t_start, "start_offset": start,
        "source_repo": REPO_NAMES[recipe["repo"]], "source_path": recipe["path"], "source_row": int(recipe["row"]),
        "sigma_p2p_ppm": row["sigma_p2p_ppm"], "tmag": row["tmag"], "kepmag": row["kepmag"],
        "teff": row["teff"], "radius": row["radius"], "quality_tier": row["quality_tier"],
        "coverage": float(observed.mean()), "single_transit_crop": cropped,
        **encode(window),
        "patch_label": labels["patch_label"].tobytes(),
        "patch_kind": labels["patch_kind"].tobytes(),
        "patch_overlap": np.round(labels["patch_overlap"] * 255).astype(np.uint8).tobytes(),
        "patch_center_offset": labels["patch_center_offset"].astype("<f2").tobytes(),
        "n_transit_patches": int((labels["patch_label"] == config.LABEL_TRANSIT).sum()),
        "has_transit": bool((labels["patch_label"] == config.LABEL_TRANSIT).any()),
        **stacked_visible(events),
        "events": [e.as_record() for e in events if -e.duration_slots / 2 <= e.center_slot < config.WINDOW_SLOTS + e.duration_slots / 2],
        "benchmark_tce_id": recipe.get("tce_id"),
        "build_version": config.BUILD_VERSION,
        **params,
    }
    return sample


def materialize(recipes: pd.DataFrame, files, templates: dict[str, dict]) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}
    for f in files:
        mine = recipes[(recipes["repo"] == f.repo) & (recipes["path"] == f.path)]
        if mine.empty:
            continue
        by_row: dict[int, list[dict]] = {}
        for r in mine.to_dict("records"):
            by_row.setdefault(int(r["row"]), []).append(r)
        parquet = pq.ParquetFile(f.local)
        first = 0
        for group in range(parquet.metadata.num_row_groups):
            n = parquet.metadata.row_group(group).num_rows
            wanted = [i for i in range(first, first + n) if i in by_row]
            if wanted:
                table = parquet.read_row_group(group, columns=ROW_COLUMNS)
                for i in wanted:
                    row = {name: table.column(name)[i - first].as_py() for name in ROW_COLUMNS}
                    for recipe in by_row[i]:
                        if recipe["segment_id"] != row["segment_id"]:
                            raise RuntimeError(f"row mismatch for {recipe['sample_id']}")
                        sample = build_sample(recipe, row, templates)
                        out.setdefault((recipe["dataset"], recipe["split"]), []).append(sample)
            first += n
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local-root", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    started = time.time()
    recipes = pd.read_parquet(args.plan / "recipes.parquet")
    recipes = recipes[(recipes["source"] == args.source)
                      & (recipes["path"].str.rsplit("/", n=1).str[-1] == f"{args.stem}.parquet")]
    for name in ("template_id", "tce_id"):
        recipes[name] = recipes[name].astype(object).where(recipes[name].notna(), None)
    templates = load_templates(args.plan / "templates.parquet")
    args.out.mkdir(parents=True, exist_ok=True)
    needed = set(recipes["repo"] + ":" + recipes["path"])
    with tempfile.TemporaryDirectory() as tmp:
        files = fetch(args.source, args.stem, args.local_root, Path(tmp), only=needed)
        produced = materialize(recipes, files, templates)
    counts = {}
    for (dataset, split), samples in produced.items():
        table = pa.Table.from_pylist(samples, schema=SAMPLE_SCHEMA)
        pq.write_table(table, args.out / f"rows_{dataset}_{split}_{args.source}_{args.stem}.parquet",
                       compression="zstd", row_group_size=500)
        counts[f"{dataset}/{split}"] = table.num_rows
    if sum(counts.values()) != len(recipes):
        raise RuntimeError(f"wrote {sum(counts.values())} samples for {len(recipes)} recipes")
    LOG.info("%s %s: %s in %.0f s", args.source, args.stem, counts, time.time() - started)
    print(f"::notice title=materialize {args.source} {args.stem}::{counts}")


if __name__ == "__main__":
    main()
