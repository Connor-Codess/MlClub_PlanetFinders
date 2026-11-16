"""Stage 2 (one job): choose every sample of lc-finetune and lc-eval.

What it does
    Reads all scan outputs, builds the transplant template library, and decides
    which window becomes which sample. Writes:

    recipes.parquet     one row per output sample: dataset, split, category, the
                        pass-1 file and row it comes from, window start, the
                        sample's random seed, its template (transplants) and its
                        measured known events (real and EB windows)
    templates.parquet   accepted transit templates (profile, period, duration, pool)
    plan.json           counts per dataset, split and category, and the matrix of
                        source files for the materialize stage

Rules (finetune_dataset/config.py has the numbers)
    lc-finetune train/validation  stars of the train/validation splits.
        real_planet_short / real_planet_long   default windows of verified planets whose
            visible transits stack to SNR >= 4 inside the window (60% TESS, 40% Kepler
            when available; at most 30 windows per star). A planet is verified when all
            of its visible transits on the star, stacked, reach SNR 7; transits of
            planets that fail are labeled ignore. TESS planets with P >= 20 d are all
            held out, so real long-period transits here come from Kepler DR24 planet
            candidates.
        negative_eb   windows of non-planet stars with an EB eclipse or a false-positive
            TOI/TCE event of SNR >= 3 (at most 10 per star).
        injected_batman, injected_transplant, negative_random   windows of stars with no
            catalog entry (60% FFI, 20% 2-min, 20% Kepler; at most 3 per star and
            category). Shortfalls in any category are filled with batman injections.
        25% of real_planet_short rows with two or more transits, one of them at
            per-transit SNR >= 4, keep that transit visible and hide the rest as a data gap.

Stacked SNR
    The stacked SNR of n transits of one planet is sum(snr_i) / sqrt(n). For transits of
    equal depth it equals sqrt(sum(snr_i^2)) of the noiseless signal, and unlike that sum
    it has zero mean on pure noise, so many faint noise dips cannot add up to a detection.
    lc-eval val/test  held-out stars only: scoreboard_hosts halves and list_excluded
        real_validation/real_test.
        real_planet_long / real_planet_short   every window with a planet transit at
            least half observed (no SNR cut; at most 10 per star).
        negative_eb   up to 5,000 per split.
        injected_batman (20,000), injected_transplant (5,000), negative_random (20,000)
            FFI windows of scoreboard stars with no catalog entry.
        Transplant templates come from train-split planets for lc-finetune and
        validation-split planets for lc-eval.
    lc-benchmarks  one split per published test set (benchmarks.py): up to 8 default
        windows per signal that contain one of its predicted transits. Every star of
        every benchmark is removed from lc-finetune, including its transit templates.

How to run
    python -m finetune_dataset.choose_samples --scan build/finetune/scan --out build/finetune/plan \
        --benchmarks build/finetune/benchmarks/benchmark_targets.parquet [--scale 0.01]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .pools import eval_split_for_pool, is_long_period
from .scan import EVENT_STRUCT
from .templates import build_templates

LOG = logging.getLogger("finetune.choose")

RECIPE_SCHEMA = pa.schema([
    ("sample_id", pa.string()), ("dataset", pa.string()), ("split", pa.string()), ("category", pa.string()),
    ("repo", pa.string()), ("path", pa.string()), ("row", pa.int64()), ("segment_id", pa.string()),
    ("star_key", pa.string()), ("source", pa.string()), ("pool", pa.string()), ("window_index", pa.int16()),
    ("start_offset", pa.int32()), ("seed", pa.int64()), ("template_id", pa.string()),
    ("single_transit_crop", pa.bool_()), ("tce_id", pa.string()), ("events", pa.list_(EVENT_STRUCT)),
])


def read_all(scan_dir: Path, kind: str) -> pd.DataFrame:
    files = sorted(scan_dir.glob(f"*_{kind}.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def stacked_snr(snrs) -> float:
    snrs = [float(x) for x in snrs]
    return sum(snrs) / np.sqrt(len(snrs)) if snrs else 0.0


def planet_fold_snr(events: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Stacked SNR of every visible transit of each planet, per star; best source wins.

    Overlapping windows repeat a transit, so each one counts once by (segment, absolute
    centre). TESS FFI and 2-min data cover the same sectors, so sources are stacked
    separately and the best is kept rather than adding the same transit twice.
    """
    seen = {}
    for star, segment, source, offset, evs in zip(events["star_key"], events["segment_id"], events["source"],
                                                  events["start_offset"], events["events"]):
        for e in evs:
            if e["cls"] != "planet" or e["observed_fraction"] < config.MIN_VISIBLE_FRACTION:
                continue
            key = (star, e["ref"], source, segment, int(round(offset + e["center_slot"])))
            seen[key] = float(e["snr"])
    groups: dict[tuple[str, str, str], list[float]] = {}
    for (star, ref, source, _, _), snr in seen.items():
        groups.setdefault((star, ref, source), []).append(snr)
    best: dict[tuple[str, str], float] = {}
    for (star, ref, _), snrs in groups.items():
        best[(star, ref)] = max(best.get((star, ref), -np.inf), stacked_snr(snrs))
    return best


def mark_unverified(events: pd.DataFrame, fold: dict[tuple[str, str], float]) -> pd.DataFrame:
    """Planet transits of planets whose stacked SNR is below MIN_PLANET_FOLD_SNR become cls 'unknown'."""
    def relabel(star, evs):
        out = []
        for e in evs:
            if e["cls"] == "planet" and fold.get((star, e["ref"]), 0.0) < config.MIN_PLANET_FOLD_SNR:
                e = {**e, "cls": "unknown"}
            out.append(e)
        return out
    out = events.copy()
    out["events"] = [relabel(star, list(evs)) for star, evs in zip(events["star_key"], events["events"])]
    return out


def window_features(events: pd.DataFrame) -> pd.DataFrame:
    """Per candidate window: visible planets, best and stacked planet SNR, long/short, EB/FP signal.

    combined_planet_snr is the best stacked SNR of any one planet's visible transits in the
    window. strong_long marks windows where a long-period planet alone reaches the combined
    threshold, so it decides long versus short in lc-finetune; any_long does it in lc-eval.
    """
    def summarize(evs) -> tuple:
        planet = [e for e in evs if e["cls"] == "planet"]
        visible = [e for e in planet if e["observed_fraction"] >= config.MIN_VISIBLE_FRACTION]
        other = [e for e in evs if e["cls"] in ("eclipsing_binary", "false_positive")
                 and e["observed_fraction"] >= config.MIN_VISIBLE_FRACTION]
        best = max((e["snr"] for e in visible), default=0.0)
        per_ref: dict[str, list] = {}
        for e in visible:
            per_ref.setdefault(e["ref"], []).append(e)
        stacked = {ref: stacked_snr(x["snr"] for x in group) for ref, group in per_ref.items()}
        combined = max(stacked.values(), default=0.0)
        strong_long = any(is_long_period(group[0]["period_d"]) and stacked[ref] >= config.MIN_REAL_COMBINED_SNR
                          for ref, group in per_ref.items())
        any_long = any(is_long_period(e["period_d"]) for e in visible)
        return (len(visible), best, combined, strong_long, any_long,
                max((e["snr"] for e in other), default=0.0))

    values = [summarize(list(evs)) for evs in events["events"]]
    out = events.copy()
    out[["n_visible_planet", "best_planet_snr", "combined_planet_snr", "strong_long", "any_long",
         "best_eb_snr"]] = pd.DataFrame(values, index=events.index)
    return out


def capped(frame: pd.DataFrame, rng: np.random.Generator, per_star: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    shuffled = frame.iloc[rng.permutation(len(frame))]
    return shuffled[shuffled.groupby("star_key").cumcount() < per_star]


def take_mixed(frame: pd.DataFrame, quota: int, rng: np.random.Generator, per_star: int,
               tess_share: float | None = None, source_shares: dict | None = None) -> pd.DataFrame:
    """Take up to quota rows, following a TESS/Kepler or per-source mix where possible."""
    pool = capped(frame, rng, per_star)
    if quota <= 0 or pool.empty:
        return pool.iloc[:0]
    if source_shares is not None:
        groups = {s: pool[pool["source"] == s] for s in source_shares}
        shares = source_shares
    elif tess_share is not None:
        tess = pool["source"].isin(["ffi", "2min"])
        groups = {"tess": pool[tess], "kepler": pool[~tess]}
        shares = {"tess": tess_share, "kepler": 1.0 - tess_share}
    else:
        return pool.iloc[:quota]
    wanted = {k: int(round(quota * v)) for k, v in shares.items()}
    taken = {k: groups[k].iloc[: wanted[k]] for k in groups}
    short = quota - sum(len(t) for t in taken.values())
    for k in groups:
        if short <= 0:
            break
        extra = groups[k].iloc[len(taken[k]): len(taken[k]) + short]
        taken[k] = pd.concat([taken[k], extra])
        short -= len(extra)
    return pd.concat(list(taken.values()))


def finetune_split(split: str, pool: str, feats: pd.DataFrame, hosts: pd.DataFrame, templates: pd.DataFrame,
                   rng: np.random.Generator, scale: float, excluded: set[str]) -> list[pd.DataFrame]:
    total = int(round(config.FINETUNE_ROWS[split] * scale))
    quota = {c: int(round(total * share)) for c, share in config.FINETUNE_MIX.items()}
    mine = feats[(feats["pool"] == pool) & ~feats["star_key"].isin(excluded)]
    hosts = hosts[~hosts["star_key"].isin(excluded)]
    templates = templates[~templates["star_key"].isin(excluded)]
    chosen = []

    real = mine[(mine["n_visible_planet"] > 0) & (mine["combined_planet_snr"] >= config.MIN_REAL_COMBINED_SNR)]
    for category, frame in (("real_planet_long", real[real["strong_long"]]),
                            ("real_planet_short", real[~real["strong_long"]])):
        picked = take_mixed(frame, quota[category], rng, config.REAL_WINDOWS_PER_STAR,
                            tess_share=config.FINETUNE_REAL_TESS_SHARE)
        picked = picked.assign(category=category)
        if category == "real_planet_short":
            eligible = ((picked["n_visible_planet"] >= 2)
                        & (picked["best_planet_snr"] >= config.SINGLE_TRANSIT_CROP_MIN_SNR))
            picked["single_transit_crop"] = eligible & (rng.uniform(size=len(picked)) < config.SINGLE_TRANSIT_CROP_SHARE)
        chosen.append(picked)

    eb = mine[~mine["planet_star"] & (mine["best_eb_snr"] >= config.MIN_EB_SNR)]
    chosen.append(take_mixed(eb, quota["negative_eb"], rng, config.EB_WINDOWS_PER_STAR,
                             tess_share=config.FINETUNE_REAL_TESS_SHARE).assign(category="negative_eb"))

    filled = {df["category"].iloc[0]: len(df) for df in chosen if len(df)}
    shortfall = sum(quota[c] - filled.get(c, 0) for c in ("real_planet_long", "real_planet_short", "negative_eb"))
    quota[config.FINETUNE_TOP_UP] += max(shortfall, 0)

    host_pool = hosts[hosts["pool"] == pool]
    # batman last: it is the top-up category, so a host shortage shrinks it first
    chosen += assign_hosts(host_pool, {c: quota[c] for c in ("negative_random", "injected_transplant",
                                                            "injected_batman")},
                           config.FINETUNE_HOST_SOURCES, rng)
    attach_templates(chosen, templates[templates["pool"] == config.TEMPLATE_POOLS["finetune"]], rng)
    return [df.assign(dataset="finetune", split=split) for df in chosen if len(df)]


def assign_hosts(hosts: pd.DataFrame, quotas: dict[str, int], shares: dict[str, float],
                 rng: np.random.Generator) -> list[pd.DataFrame]:
    """Give each category distinct host windows; injections avoid giants."""
    out = []
    used = set()
    for category, quota in quotas.items():
        candidates = hosts[~hosts.index.isin(used)]
        if category.startswith("injected"):
            radius = candidates["radius"]
            candidates = candidates[radius.isna() | (radius <= config.HOST_MAX_RADIUS_RSUN)]
        picked = take_mixed(candidates, quota, rng, config.HOST_WINDOWS_PER_STAR, source_shares=shares)
        used.update(picked.index)
        out.append(picked.assign(category=category, events=[[] for _ in range(len(picked))]))
    return out


def attach_templates(chosen: list[pd.DataFrame], templates: pd.DataFrame, rng: np.random.Generator) -> None:
    for df in chosen:
        if len(df) and df["category"].iloc[0] == "injected_transplant":
            if templates.empty:
                raise RuntimeError("no transit templates passed the quality cuts")
            df["template_id"] = templates["template_id"].to_numpy()[rng.integers(0, len(templates), len(df))]


def eval_split(split: str, feats: pd.DataFrame, hosts: pd.DataFrame, templates: pd.DataFrame,
               rng: np.random.Generator, scale: float) -> list[pd.DataFrame]:
    pools = config.EVAL_POOLS[split]
    mine = feats[feats["pool"].isin(pools)]
    chosen = []
    real = mine[mine["n_visible_planet"] > 0]
    for category, frame in (("real_planet_long", real[real["any_long"]]),
                            ("real_planet_short", real[~real["any_long"]])):
        picked = capped(frame, rng, config.EVAL_REAL_WINDOWS_PER_STAR)
        chosen.append(picked.assign(category=category))
    eb = mine[~mine["planet_star"] & (mine["best_eb_snr"] >= config.MIN_EB_SNR)]
    eb_quota = int(round(config.EVAL_ROWS["negative_eb"] * scale))
    chosen.append(capped(eb, rng, config.EVAL_EB_WINDOWS_PER_STAR).iloc[:eb_quota].assign(category="negative_eb"))

    host_pool = hosts[hosts["pool"].isin(pools)]
    quotas = {c: int(round(config.EVAL_ROWS[c] * scale))
              for c in ("negative_random", "injected_transplant", "injected_batman")}
    chosen += assign_hosts(host_pool, quotas, config.EVAL_HOST_SOURCES, rng)
    attach_templates(chosen, templates[templates["pool"] == config.TEMPLATE_POOLS["eval"]], rng)
    return [df.assign(dataset="eval", split=split) for df in chosen if len(df)]


def benchmark_samples(bench: pd.DataFrame, rng: np.random.Generator) -> list[pd.DataFrame]:
    if bench.empty:
        return []
    shuffled = bench.iloc[rng.permutation(len(bench))]
    picked = shuffled[shuffled.groupby(["benchmark", "tce_id"]).cumcount() < config.BENCH_WINDOWS_PER_SIGNAL]
    return [picked.assign(dataset="benchmark", split=picked["benchmark"], category=picked["label"])]


def to_recipes(frames: list[pd.DataFrame], rng: np.random.Generator) -> pd.DataFrame:
    recipes = pd.concat(frames, ignore_index=True)
    for name, default in (("template_id", None), ("single_transit_crop", False), ("tce_id", None)):
        if name not in recipes:
            recipes[name] = default
    recipes["single_transit_crop"] = recipes["single_transit_crop"].fillna(False).astype(bool)
    for name in ("template_id", "tce_id"):
        recipes[name] = recipes[name].astype(object).where(recipes[name].notna(), None)
    recipes["seed"] = rng.integers(0, 2**62, len(recipes))
    recipes = recipes.sort_values(["dataset", "split", "category", "star_key", "segment_id", "window_index"])
    recipes["sample_id"] = (recipes["dataset"] + ":" + recipes["split"] + ":" + recipes["category"] + ":"
                            + recipes.groupby(["dataset", "split", "category"]).cumcount().astype(str).str.zfill(6))
    return recipes[RECIPE_SCHEMA.names].reset_index(drop=True)


def check_pools(recipes: pd.DataFrame, excluded: set[str]) -> None:
    """lc-finetune only from train/validation stars and no benchmark star; lc-eval only from held-out stars."""
    finetune = recipes[recipes["dataset"] == "finetune"]
    bad = finetune[finetune["pool"] != finetune["split"]]
    evals = recipes[recipes["dataset"] == "eval"]
    bad_eval = evals[evals["pool"].map(eval_split_for_pool) != evals["split"]]
    overlap = set(finetune["star_key"]) & set(evals["star_key"])
    bench = set(finetune["star_key"]) & excluded
    if len(bad) or len(bad_eval) or overlap or bench:
        raise RuntimeError(f"pool violation: {len(bad)} finetune, {len(bad_eval)} eval, {len(overlap)} shared stars, "
                           f"{len(bench)} benchmark stars in lc-finetune")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0, help="multiply every quota (trial runs)")
    parser.add_argument("--benchmarks", type=Path, help="benchmark_targets.parquet; its stars stay out of lc-finetune")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(config.SEED)

    events = read_all(args.scan, "events")
    hosts = read_all(args.scan, "hosts")
    cutouts = read_all(args.scan, "cutouts")
    bench = read_all(args.scan, "bench")
    excluded = set(pd.read_parquet(args.benchmarks)["star_key"]) if args.benchmarks else set()
    LOG.info("Benchmarks: %d windows; %d benchmark stars kept out of lc-finetune", len(bench), len(excluded))
    LOG.info("Scan: %d event windows, %d host windows, %d cutouts", len(events), len(hosts), len(cutouts))
    templates = build_templates(cutouts)
    LOG.info("Templates: %s", templates.groupby(["pool", "mission"]).size().to_dict() if len(templates) else {})
    feats = window_features(events)
    fold = planet_fold_snr(events[events["pool"].isin([config.POOL_TRAIN, config.POOL_VALIDATION])])
    verified = sum(v >= config.MIN_PLANET_FOLD_SNR for v in fold.values())
    LOG.info("Planets on train/validation stars: %d with visible transits, %d verified (stacked SNR >= %.0f)",
             len(fold), verified, config.MIN_PLANET_FOLD_SNR)
    tuning = window_features(mark_unverified(events, fold))

    frames = []
    for split, pool in (("train", config.POOL_TRAIN), ("validation", config.POOL_VALIDATION)):
        frames += finetune_split(split, pool, tuning, hosts, templates, rng, args.scale, excluded)
    for split in ("val", "test"):
        frames += eval_split(split, feats, hosts, templates, rng, args.scale)
    frames += benchmark_samples(bench, rng)
    recipes = to_recipes(frames, rng)
    check_pools(recipes, excluded)

    args.out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(recipes, schema=RECIPE_SCHEMA, preserve_index=False),
                   args.out / "recipes.parquet", compression="zstd")
    templates.to_parquet(args.out / "templates.parquet", index=False)

    recipes["file"] = recipes["source"] + "/" + recipes["path"].str.rsplit("/", n=1).str[-1].str.removesuffix(".parquet")
    files = sorted(recipes["file"].unique())
    plan = {
        "counts": {f"{d}/{s}/{c}": int(n) for (d, s, c), n in
                   recipes.groupby(["dataset", "split", "category"]).size().items()},
        "templates": {f"{pool}/{mission}": int(n) for (pool, mission), n in
                      templates.groupby(["pool", "mission"]).size().items()},
        "matrix": [{"source": f.split("/")[0], "stem": f.split("/")[1]} for f in files],
        "planets": {"with_visible_transits": len(fold), "verified": int(verified)},
        "scale": args.scale,
    }
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2))
    LOG.info("Recipes: %s", json.dumps(plan["counts"], indent=1))


if __name__ == "__main__":
    main()
