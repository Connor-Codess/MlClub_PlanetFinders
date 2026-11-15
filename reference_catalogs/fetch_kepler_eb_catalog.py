"""Download the Kepler eclipsing-binary catalog of Kirk et al. (2016).

What it does
    Retrieves the main catalog table of VizieR J/AJ/151/68 (the final Kepler
    EB catalog, about 2,900 systems from Q0-Q17) from the CDS archive files
    (ReadMe plus catalog.dat), falling back to TAPVizieR, and adds standardized columns:

        kic_id, period_d, t0_btjd, t0_reference, t0_suspect, morphology, kepmag,
        primary_width_phase, primary_depth, secondary_width_phase,
        secondary_depth, primary_duration_h

    The catalog's epoch column (BJD0) is converted to BTJD. Its time reference
    (full BJD, BJD - 2400000 or BKJD) is detected from the median epoch and
    recorded in t0_reference and in the file metadata, so the conversion can be
    checked by hand.

Why
    Used only as tags in the pretraining build: an "eb_catalog" flag on every
    Kepler segment of these stars, and predicted eclipse times as known events.

Output
    kepler_eb_catalog_kirk2016.parquet. All VizieR columns are kept in
    snake_case under a "vizier_" prefix.

Reference
    Kirk et al. 2016, AJ 151, 68, doi:10.3847/0004-6256/151/3/68

Run
    python -m reference_catalogs.fetch_kepler_eb_catalog --out build/reference_catalogs
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
    infer_kepler_time_reference,
    snake_case,
    to_btjd,
    utc_now,
    write_parquet,
)

NAME = "kepler_eb_catalog_kirk2016"
FILENAME = f"{NAME}.parquet"
VIZIER_CATALOG = "J/AJ/151/68"
SOURCE = f"VizieR {VIZIER_CATALOG} (Kirk et al. 2016, AJ 151, 68)"
MIN_EXPECTED_ROWS = 2500
# Kepler ran from BTJD -2048 to -577; epochs propagated a few periods outside that
# window are normal, so the plausible range is generous.
T0_PLAUSIBLE_BTJD = (-5000.0, 2000.0)


VIZIER_TABLE = f"{VIZIER_CATALOG}/catalog"
CDS_DIRS = [
    "https://cdsarc.cds.unistra.fr/ftp/J/AJ/151/68",
    "https://cdsarc.cds.unistra.fr/VizieR/ftp/J/AJ/151/68",
    "https://cdsarc.cds.unistra.fr/ftp/cats/J/other/AJ/151/68",
    "https://cdsarc.cds.unistra.fr/ftp/cats/J/AJ/151/68",
]


def download() -> tuple[pd.DataFrame, str]:
    return fetch_vizier_table(VIZIER_TABLE, CDS_DIRS, "catalog.dat", min_rows=MIN_EXPECTED_ROWS)


def _numeric(df: pd.DataFrame, *candidates: str) -> pd.Series:
    column = find_column(list(df.columns), *candidates, required=False)
    if column is None:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def standardize(raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    df = raw.copy()
    cols = list(df.columns)
    kic = pd.to_numeric(df[find_column(cols, "KIC", "KIC_ID", "KICID")], errors="coerce").astype("Int64")
    period = _numeric(df, "Per", "Period", "P")
    epoch = _numeric(df, "BJD0", "bjd0", "T0", "Epoch")
    if epoch.notna().any():
        reference = infer_kepler_time_reference(epoch.dropna())
        t0 = to_btjd(epoch, reference)
    else:
        LOG.warning("No epoch column found among %s; t0_btjd left empty", cols)
        reference = "unknown"
        t0 = pd.Series(np.nan, index=df.index, dtype="float64")
    LOG.info("Kepler EB epochs interpreted as %s", reference)

    out = pd.DataFrame({
        "kic_id": kic,
        "period_d": period.where(period > 0),
        "t0_btjd": t0,
        "t0_reference": reference,
        "morphology": _numeric(df, "morph", "Morph"),
        "kepmag": _numeric(df, "Kpmag", "kpmag", "kmag", "Kmag", "Kp"),
        "primary_width_phase": _numeric(df, "pwidth", "Wp"),
        "primary_depth": _numeric(df, "pdepth", "Dp"),
        "secondary_width_phase": _numeric(df, "swidth", "Ws"),
        "secondary_depth": _numeric(df, "sdepth", "Ds"),
    })
    out["primary_duration_h"] = out["primary_width_phase"] * out["period_d"] * 24.0

    # A few catalog epochs fall nowhere near the mission (for example BJD0 = 5500.4
    # in a BJD - 2400000 column). Keep the raw value under vizier_bjd0, blank the
    # converted epoch and flag the row, rather than guess the intended value.
    suspect = (out["t0_btjd"] < T0_PLAUSIBLE_BTJD[0]) | (out["t0_btjd"] > T0_PLAUSIBLE_BTJD[1])
    out["t0_suspect"] = suspect.fillna(False)
    out.loc[out["t0_suspect"], "t0_btjd"] = np.nan

    vizier = df.copy()
    vizier.columns = ["vizier_" + snake_case(c) for c in vizier.columns]
    for column in vizier.columns:
        if vizier[column].dtype == object:
            vizier[column] = vizier[column].astype("string")
    combined = pd.concat([out, vizier], axis=1)
    return combined.sort_values("kic_id", kind="stable").reset_index(drop=True), reference


def build(out_dir: Path) -> CatalogResult:
    retrieved = utc_now()
    raw, table_name = download()
    LOG.info("Kepler EB columns: %s", list(raw.columns))
    df, reference = standardize(raw)
    if len(df) < MIN_EXPECTED_ROWS:
        raise RuntimeError(f"Only {len(df)} rows in {table_name}; expected at least {MIN_EXPECTED_ROWS}")
    notes = {
        "downloaded_from": table_name,
        "source_columns": list(raw.columns),
        "unique_kic": int(df["kic_id"].nunique()),
        "with_period": int(df["period_d"].notna().sum()),
        "epoch_reference_detected": reference,
        "t0_suspect": int(df["t0_suspect"].sum()),
    }
    rows = write_parquet(df, out_dir / FILENAME, {
        "name": NAME, "source": SOURCE, "retrieved_utc": retrieved,
        "time_reference": f"t0_btjd converted from BJD0 interpreted as {reference}", **notes,
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
