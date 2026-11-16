"""Star pools and known-event classes shared by every stage.

Pools decide which dataset a star can feed:
    train, validation              -> lc-finetune (train, validation)
    scoreboard_val, list_val       -> lc-eval val
    scoreboard_test, list_test     -> lc-eval test
    unused                         -> search_holdout, never used here

scoreboard_hosts stars are halved by SHA-256("scoreboard:" + star_key); list_excluded
stars keep the real_validation / real_test assignment from the exclusion list.

Known events (the known_events column of lc-pretrain and lc-heldout) are classed as
planet, eclipsing_binary, false_positive or unknown from their catalog and disposition.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

from pretrain_dataset.catalogs import ReferenceCatalogs

from . import config


def scoreboard_half(star_key: str) -> str:
    digest = hashlib.sha256(f"scoreboard:{star_key}".encode()).digest()
    return "val" if int.from_bytes(digest[:8], "big") % 2 == 0 else "test"


def pool_for(star_key: str, split: str, list_sets: dict[str, str]) -> str:
    if split == "train":
        return config.POOL_TRAIN
    if split == "validation":
        return config.POOL_VALIDATION
    if split == "scoreboard_hosts":
        return config.POOL_SCOREBOARD_VAL if scoreboard_half(star_key) == "val" else config.POOL_SCOREBOARD_TEST
    if split == "list_excluded":
        return config.POOL_LIST_VAL if list_sets.get(star_key) == "real_validation" else config.POOL_LIST_TEST
    return config.POOL_UNUSED


def eval_split_for_pool(pool: str) -> str | None:
    for split, pools in config.EVAL_POOLS.items():
        if pool in pools:
            return split
    return None


@dataclass(frozen=True)
class EventInfo:
    cls: str                  # planet, eclipsing_binary, false_positive, unknown
    period_d: float | None
    depth_ppm: float | None


def classify(kind: str, ref: str) -> str:
    if kind.startswith("tess_eb") or kind.startswith("kepler_eb"):
        return "eclipsing_binary"
    if kind == "hawthorn_duotransit":
        return "planet"
    label = ref.rsplit(" ", 1)[-1] if " " in ref else ""
    if kind == "toi":
        if label in config.PLANET_DISPOSITIONS:
            return "planet"
        if label in config.FALSE_POSITIVE_DISPOSITIONS:
            return "false_positive"
        return "unknown"
    if kind == "kepler_tce":
        if label == "PC":
            return "planet"
        if label in ("AFP", "NTP"):
            return "false_positive"
        return "unknown"
    return "unknown"


class EventCatalog:
    """Period and depth for every known-event ref, plus star-level sets."""

    def __init__(self, catalogs: ReferenceCatalogs):
        self.info: dict[tuple[str, str], EventInfo] = {}
        for ephemerides in (*catalogs.tess_ephemerides.values(), *catalogs.kepler_ephemerides.values()):
            for eph in ephemerides:
                self.info[(eph.kind, eph.ref)] = EventInfo(classify(eph.kind, eph.ref), eph.period_d, None)
        for row in catalogs.toi.itertuples(index=False):
            key = ("toi", f"TOI-{row.toi:.2f} {row.disposition}")
            if key in self.info and pd.notna(row.depth_ppm):
                self.info[key] = EventInfo(self.info[key].cls, self.info[key].period_d, float(row.depth_ppm))
        for row in catalogs.dr24_tce.itertuples(index=False):
            key = ("kepler_tce", f"KIC {int(row.kepid)} TCE {int(row.tce_plnt_num)} {row.av_training_set}")
            if key in self.info and pd.notna(row.tce_depth):
                self.info[key] = EventInfo(self.info[key].cls, self.info[key].period_d, float(row.tce_depth))

        self.tagged_stars = set(catalogs.tags)
        self.planet_stars = set()
        for key, tags in catalogs.tags.items():
            dispositions = set((tags["toi_disposition"] or "").split(","))
            tce = set((tags["kepler_tce_labels"] or "").split(","))
            if dispositions & set(config.PLANET_DISPOSITIONS) or "PC" in tce or tags["hawthorn_candidate"] is not None:
                self.planet_stars.add(key)
        self.list_sets = dict(zip(catalogs.exclusions["star_key"], catalogs.exclusions["scoreboard_set"]))

    def lookup(self, kind: str, ref: str) -> EventInfo:
        return self.info.get((kind, ref), EventInfo(classify(kind, ref), None, None))


def is_long_period(period_d: float | None) -> bool:
    return period_d is None or period_d != period_d or period_d >= config.LONG_PERIOD_DAYS
