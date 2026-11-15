"""Steps 1-2 of the build guide: one identity, one split and one exclusion rule per star.

star_key
    "tic:<TIC ID>" for every TESS row and for Kepler stars with a TIC match in
    the KIC-TIC crossmatch; "kic:<KIC ID>" for the rest.

hash_bucket
    SHA-256 of the star_key string, first 8 bytes as an unsigned big-endian
    integer, modulo 1000. Reproducible in any language from the key alone.

split
    From the bucket (config.SPLIT_BUCKETS): train, validation, search_holdout,
    scoreboard_hosts. A star on the exclusion list gets split "list_excluded"
    whatever its bucket. Only train and validation rows go to the public
    pretraining repo.

Exclusion list (Step 2)
    - every host of the 85 Hawthorn et al. 2024 duotransit candidates;
    - every TOI dispositioned PC, APC, CP or KP with period above 20 days or
      no period. These are divided into real_validation and real_test halves
      by a second hash (of "real:" + star_key), so the halves are reproducible.

Run (writes exclusions.parquet for the record; the build computes the same set itself)
    python -m pretrain_dataset.star_identity --out build/index
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config


def tic_key(tic_id: int) -> str:
    return f"tic:{int(tic_id)}"


def kic_key(kic_id: int) -> str:
    return f"kic:{int(kic_id)}"


def hash_bucket(star_key: str) -> int:
    digest = hashlib.sha256(star_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 1000


def bucket_split(bucket: int) -> str:
    for low, high, name in config.SPLIT_BUCKETS:
        if low <= bucket < high:
            return name
    raise ValueError(f"bucket {bucket} outside 0-999")


def real_test_half(star_key: str) -> str:
    """Divide list-excluded planets into two reproducible halves."""
    return "real_validation" if hash_bucket("real:" + star_key) % 2 == 0 else "real_test"


@dataclass(frozen=True)
class StarIdentity:
    star_key: str
    hash_bucket: int
    split: str


class IdentityResolver:
    """Maps a TIC or KIC ID to its star_key, bucket and split."""

    def __init__(self, kic_to_tic: dict[int, int], exclusions: dict[str, str]):
        self.kic_to_tic = kic_to_tic
        self.exclusions = exclusions

    def resolve(self, *, tic_id: int | None = None, kic_id: int | None = None) -> StarIdentity:
        if tic_id is not None:
            key = tic_key(tic_id)
        elif kic_id is not None:
            matched = self.kic_to_tic.get(int(kic_id))
            key = tic_key(matched) if matched is not None else kic_key(kic_id)
        else:
            raise ValueError("Need a TIC or KIC ID")
        bucket = hash_bucket(key)
        split = "list_excluded" if key in self.exclusions else bucket_split(bucket)
        return StarIdentity(key, bucket, split)


def build_exclusions(toi: pd.DataFrame, duotransits: pd.DataFrame) -> pd.DataFrame:
    """Return star_key, reason, scoreboard_set for every list-excluded star."""
    rows: dict[str, dict] = {}
    for tic in duotransits["tic_id"].dropna().astype("int64"):
        key = tic_key(tic)
        rows[key] = {"star_key": key, "reason": "hawthorn2024_duotransit_host"}

    planets = toi[toi["disposition"].isin(config.PLANET_DISPOSITIONS)]
    long_period = planets[planets["period_d"].isna() | (planets["period_d"] > config.LONG_PERIOD_DAYS)]
    for tic in long_period["tic_id"].dropna().astype("int64"):
        key = tic_key(tic)
        if key not in rows:
            rows[key] = {"star_key": key, "reason": "long_period_or_single_transit_toi"}

    out = pd.DataFrame(list(rows.values()))
    out["scoreboard_set"] = [real_test_half(k) for k in out["star_key"]]
    out["hash_bucket"] = [hash_bucket(k) for k in out["star_key"]]
    return out.sort_values("star_key").reset_index(drop=True)


def main() -> None:
    from .catalogs import ReferenceCatalogs

    parser = argparse.ArgumentParser(description="Write the Step 2 exclusion list")
    parser.add_argument("--out", type=Path, default=Path("build/index"))
    parser.add_argument("--revision", default=None, help="reference-catalogs revision to pin")
    args = parser.parse_args()
    catalogs = ReferenceCatalogs.load(args.revision)
    exclusions = build_exclusions(catalogs.toi, catalogs.duotransits)
    args.out.mkdir(parents=True, exist_ok=True)
    exclusions.to_parquet(args.out / "exclusions.parquet", index=False)
    print(exclusions["reason"].value_counts().to_string())
    print(exclusions["scoreboard_set"].value_counts().to_string())


if __name__ == "__main__":
    main()
