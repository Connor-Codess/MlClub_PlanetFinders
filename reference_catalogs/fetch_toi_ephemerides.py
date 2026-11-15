"""Download the TESS Objects of Interest (TOI) table with ephemerides.

What it does
    Queries the NASA Exoplanet Archive TAP service (table ``toi``) for every TOI,
    keeps the identification, disposition, ephemeris and stellar columns, and
    adds standardized columns used by the pretraining build:

        toi, tic_id, disposition, period_d, t0_btjd, duration_h, depth_ppm,
        planet_radius_re, tmag, teff_k, star_radius_rsun, ra_deg, dec_deg,
        sectors, period_known, long_period_or_unknown

    t0_btjd is pl_tranmid (BJD) minus 2457000. period_known is False when the
    archive lists no period or a period of zero, which is how single-transit
    TOIs usually appear. long_period_or_unknown marks TOIs with P > 20 d or no
    period; Step 2 of the build guide uses it to pick held-out test planets.

Why
    The TESS_TCES repo holds only TOI, TIC and disposition. Labeling known
    transits and choosing long-period test planets need the ephemerides.

Output
    toi_ephemerides.parquet (one row per TOI). Raw archive columns are kept
    under their archive names next to the standardized ones.

Run
    python -m reference_catalogs.fetch_toi_ephemerides --out build/reference_catalogs
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    LOG,
    CatalogResult,
    configure_logging,
    http_request,
    to_btjd,
    utc_now,
    write_parquet,
)

NAME = "toi_ephemerides"
FILENAME = f"{NAME}.parquet"
TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
SOURCE = "NASA Exoplanet Archive TAP, table toi"
LONG_PERIOD_DAYS = 20.0

WANTED_COLUMNS = [
    "toi", "toipfx", "tid", "ctoi_alias", "pl_pnum", "tfopwg_disp",
    "ra", "dec", "st_tmag", "st_teff", "st_logg", "st_rad",
    "pl_orbper", "pl_orbpererr1", "pl_orbpererr2",
    "pl_tranmid", "pl_tranmiderr1", "pl_tranmiderr2",
    "pl_trandurh", "pl_trandurherr1", "pl_trandurherr2",
    "pl_trandep", "pl_trandeperr1", "pl_trandeperr2",
    "pl_rade", "pl_eqt", "sectors", "toi_created", "rowupdate",
]
REQUIRED_COLUMNS = {"toi", "tid", "tfopwg_disp", "pl_orbper", "pl_tranmid", "pl_trandurh", "pl_trandep"}


def _tap_csv(query: str) -> pd.DataFrame:
    response = http_request("GET", TAP_URL, params={"query": query, "format": "csv"})
    return pd.read_csv(io.StringIO(response.text), low_memory=False)


def available_columns() -> list[str]:
    """Ask the archive which columns the toi table has, so the main query never names a missing one."""
    sample = _tap_csv("select top 1 * from toi")
    return [c.lower() for c in sample.columns]


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def standardize(raw: pd.DataFrame) -> pd.DataFrame:
    """Add the standardized columns; keep every raw archive column as well."""
    df = raw.copy()
    df.columns = [c.lower() for c in df.columns]

    period = _numeric(df, "pl_orbper")
    depth = _numeric(df, "pl_trandep")
    median_depth = float(depth.median())
    if median_depth < 1.0:
        LOG.warning("Median pl_trandep %.4g looks fractional; converting to ppm", median_depth)
        depth = depth * 1e6

    out = pd.DataFrame({
        "toi": pd.to_numeric(df["toi"], errors="coerce"),
        "tic_id": pd.to_numeric(df["tid"], errors="coerce").astype("Int64"),
        "disposition": df["tfopwg_disp"].astype("string"),
        "period_d": period.where(period > 0),
        "t0_btjd": to_btjd(df["pl_tranmid"], "BJD"),
        "duration_h": _numeric(df, "pl_trandurh"),
        "depth_ppm": depth,
        "planet_radius_re": _numeric(df, "pl_rade"),
        "tmag": _numeric(df, "st_tmag"),
        "teff_k": _numeric(df, "st_teff"),
        "star_radius_rsun": _numeric(df, "st_rad"),
        "ra_deg": _numeric(df, "ra"),
        "dec_deg": _numeric(df, "dec"),
        "sectors": df["sectors"].astype("string") if "sectors" in df else pd.Series(pd.NA, index=df.index, dtype="string"),
    })
    out["period_known"] = out["period_d"].notna()
    out["long_period_or_unknown"] = (~out["period_known"]) | (out["period_d"] > LONG_PERIOD_DAYS)

    raw_renamed = df.add_prefix("archive_")
    combined = pd.concat([out, raw_renamed], axis=1)
    return combined.sort_values("toi", kind="stable").reset_index(drop=True)


def build(out_dir: Path) -> CatalogResult:
    retrieved = utc_now()
    columns = available_columns()
    missing_required = REQUIRED_COLUMNS - set(columns)
    if missing_required:
        raise RuntimeError(f"TOI table is missing required columns: {sorted(missing_required)}")
    selected = [c for c in WANTED_COLUMNS if c in columns]
    skipped = [c for c in WANTED_COLUMNS if c not in columns]
    if skipped:
        LOG.warning("Columns not offered by the archive and skipped: %s", skipped)

    raw = _tap_csv(f"select {', '.join(selected)} from toi")
    LOG.info("Downloaded %d TOIs", len(raw))
    df = standardize(raw)

    counts = df["disposition"].value_counts(dropna=False).to_dict()
    notes = {
        "disposition_counts": {str(k): int(v) for k, v in counts.items()},
        "period_known": int(df["period_known"].sum()),
        "long_period_or_unknown": int(df["long_period_or_unknown"].sum()),
        "skipped_columns": skipped,
        "long_period_threshold_days": LONG_PERIOD_DAYS,
    }
    rows = write_parquet(df, out_dir / FILENAME, {
        "name": NAME, "source": SOURCE, "retrieved_utc": retrieved,
        "time_reference": "t0_btjd = pl_tranmid - 2457000", **notes,
    })
    return CatalogResult(NAME, FILENAME, rows, SOURCE, retrieved, notes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("build/reference_catalogs"))
    args = parser.parse_args()
    configure_logging()
    result = build(args.out)
    LOG.info("Done: %s", result.to_dict())


if __name__ == "__main__":
    main()
