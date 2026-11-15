"""Download the TESS eclipsing-binary catalog of Prsa et al. (2022).

What it does
    Retrieves VizieR table J/ApJS/258/16/tess-ebs (4,584 eclipsing binaries
    from 2-minute data in sectors 1-26) and adds standardized columns:

        tic_id, signal_id, period_d, t0_btjd, morphology, tmag, ra_deg, dec_deg,
        primary_width_phase, primary_depth, secondary_width_phase,
        secondary_depth, secondary_phase, primary_duration_h, sectors

    The table is read through TAPVizieR, with the CDS archive files (ReadMe
    plus tess-ebs.dat) as a fallback. BJD0 in this catalog is already
    BJD - 2457000, i.e. BTJD. Eclipse widths
    are in phase units; primary_duration_h = width x period x 24. The
    polychain-fit values are used, falling back to the 2-Gaussian fit when the
    polychain value is missing.

Why
    The pretraining build uses this file only as tags: an "eb_catalog" flag on
    every segment of these stars, and predicted eclipse times as known events.

Output
    tess_eb_catalog_prsa2022.parquet (one row per TIC and signal; a TIC with
    two independent eclipse signals has two rows). All VizieR columns are kept
    in snake_case under a "vizier_" prefix.

Reference
    Prsa et al. 2022, ApJS 258, 16, doi:10.3847/1538-4365/ac324a

Run
    python -m reference_catalogs.fetch_tess_eb_catalog --out build/reference_catalogs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    LOG,
    CatalogResult,
    configure_logging,
    fetch_vizier_table,
    find_column,
    snake_case,
    to_btjd,
    utc_now,
    write_parquet,
)

NAME = "tess_eb_catalog_prsa2022"
FILENAME = f"{NAME}.parquet"
VIZIER_CATALOG = "J/ApJS/258/16"
VIZIER_TABLE = "J/ApJS/258/16/tess-ebs"
SOURCE = f"VizieR {VIZIER_TABLE} (Prsa et al. 2022, ApJS 258, 16)"
EXPECTED_ROWS = 4584


CDS_DIRS = [
    "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/258/16",
    "https://cdsarc.cds.unistra.fr/VizieR/ftp/J/ApJS/258/16",
    "https://cdsarc.cds.unistra.fr/ftp/cats/J/ApJS/258/16",
]


def download() -> tuple[pd.DataFrame, str]:
    return fetch_vizier_table(VIZIER_TABLE, CDS_DIRS, "tess-ebs.dat", min_rows=EXPECTED_ROWS - 100)


def _first_numeric(df: pd.DataFrame, *candidates: str) -> pd.Series:
    column = find_column(list(df.columns), *candidates, required=False)
    if column is None:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def standardize(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    cols = list(df.columns)

    tic = pd.to_numeric(df[find_column(cols, "TIC")], errors="coerce").astype("Int64")
    signal_col = find_column(cols, "m_TIC", required=False)
    signal = (pd.to_numeric(df[signal_col], errors="coerce").fillna(1).astype("int16")
              if signal_col else pd.Series(1, index=df.index, dtype="int16"))
    period = _first_numeric(df, "Per", "period")

    primary_width = _first_numeric(df, "Wp-pf").fillna(_first_numeric(df, "Wp-2g"))
    primary_depth = _first_numeric(df, "Dp-pf").fillna(_first_numeric(df, "Dp-2g"))
    secondary_width = _first_numeric(df, "Ws-pf").fillna(_first_numeric(df, "Ws-2g"))
    secondary_depth = _first_numeric(df, "Ds-pf").fillna(_first_numeric(df, "Ds-2g"))
    secondary_phase = _first_numeric(df, "Phis-pf").fillna(_first_numeric(df, "Phis-2g"))
    sectors_col = find_column(cols, "Sectors", required=False)

    out = pd.DataFrame({
        "tic_id": tic,
        "signal_id": signal,
        "period_d": period.where(period > 0),
        "t0_btjd": to_btjd(_first_numeric(df, "BJD0"), "BTJD"),
        "morphology": _first_numeric(df, "Morph"),
        "tmag": _first_numeric(df, "Tmag"),
        "ra_deg": _first_numeric(df, "RAdeg", "_RA"),
        "dec_deg": _first_numeric(df, "DEdeg", "_DE"),
        "primary_width_phase": primary_width,
        "primary_depth": primary_depth,
        "secondary_width_phase": secondary_width,
        "secondary_depth": secondary_depth,
        "secondary_phase": secondary_phase,
        "sectors": (df[sectors_col].astype("string").str.strip()
                    if sectors_col else pd.Series(pd.NA, index=df.index, dtype="string")),
    })
    out["primary_duration_h"] = out["primary_width_phase"] * out["period_d"] * 24.0

    vizier = df.copy()
    vizier.columns = ["vizier_" + snake_case(c) for c in vizier.columns]
    for column in vizier.columns:
        if vizier[column].dtype == object:
            vizier[column] = vizier[column].astype("string")
    combined = pd.concat([out, vizier], axis=1)
    return combined.sort_values(["tic_id", "signal_id"], kind="stable").reset_index(drop=True)


def build(out_dir: Path) -> CatalogResult:
    retrieved = utc_now()
    raw, origin = download()
    df = standardize(raw)
    if len(df) != EXPECTED_ROWS:
        LOG.warning("Expected %d rows, got %d", EXPECTED_ROWS, len(df))
    notes = {
        "unique_tic": int(df["tic_id"].nunique()),
        "with_period": int(df["period_d"].notna().sum()),
        "expected_rows": EXPECTED_ROWS,
        "downloaded_from": origin,
    }
    rows = write_parquet(df, out_dir / FILENAME, {
        "name": NAME, "source": SOURCE, "retrieved_utc": retrieved,
        "time_reference": "t0_btjd = BJD0 (catalog already BJD - 2457000)", **notes,
    })
    return CatalogResult(NAME, FILENAME, rows, SOURCE, retrieved, notes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("build/reference_catalogs"))
    args = parser.parse_args()
    configure_logging()
    LOG.info("Done: %s", build(args.out).to_dict())


if __name__ == "__main__":
    main()
