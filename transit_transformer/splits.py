"""Deterministic, star-level train/val/test splits for Kepler and TESS.

A star's split depends only on a hash of (mission, star_id), so both computers
and every rerun agree without sharing a file. Splits:

    train        pre-training and fine-tuning
    val          pre-training loss monitoring / fine-tuning model selection
    test         held out entirely: ~10% of stars, all their windows
    test_future  TESS sectors >= FUTURE_SECTOR for every non-test star
                 (mimics live data; never used for pre-training)
    excluded     TESS stars also observed by Kepler (pass overlap_tic) when they
                 would otherwise be test -- Kepler pre-training saw them
"""

from __future__ import annotations

import hashlib

TEST_FRAC = 0.10
VAL_FRAC = 0.05
FUTURE_SECTOR = 100
PRETRAIN_SPLITS = ("train",)        # val is used only to watch the loss


def star_unit(mission: str, star_id: int) -> float:
    """Uniform number in [0, 1) fixed for a given star."""
    h = hashlib.sha256(f"{mission}:{int(star_id)}".encode()).hexdigest()
    return int(h[:12], 16) / 16 ** 12


def assign(mission: str, star_id: int, segment: int | None = None,
           overlap_tic: frozenset[int] = frozenset()) -> str:
    u = star_unit(mission, star_id)
    if u < TEST_FRAC:
        if mission == "tess" and int(star_id) in overlap_tic:
            return "excluded"
        return "test"
    if u < TEST_FRAC + VAL_FRAC:
        return "val"
    if mission == "tess" and segment is not None and segment >= FUTURE_SECTOR:
        return "test_future"
    return "train"


def load_overlap(path: str | None) -> frozenset[int]:
    """CSV with a `tic_id` column listing TESS stars that are also Kepler targets
    (from a KIC<->TIC crossmatch). Optional; empty set if not given."""
    if not path:
        return frozenset()
    import pandas as pd
    return frozenset(pd.read_csv(path)["tic_id"].astype(int))
