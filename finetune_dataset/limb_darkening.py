"""TESS-band quadratic limb-darkening coefficients from Claret (2017), for batman injections.

What it does
    fetch()        downloads the Claret (2017, A&A 600, A30; VizieR J/A+A/600/A30) tables
                   with astroquery, finds the quadratic-law table computed from ATLAS models
                   (columns logg, Teff, Z, xi and two coefficients), keeps solar metallicity
                   and xi = 2 km/s, and writes it as a small Parquet table.
    load_table()   reads that table back (or returns None).
    coefficients() the (u1, u2) of the grid point nearest to a star's Teff and log g.

Why
    v2 drew limb darkening uniformly over the physical range (Kipping 2013). v3 ties it to
    the host star, so each injected transit has the shape a planet would show on that star.
    If the table cannot be fetched, injections fall back to the v2 draw and the row records
    inj_ld_source = "kipping2013" instead of "claret2017".

Outputs
    ld_claret2017_tess.parquet with columns teff, logg, u1, u2 (written by --fetch)

How to run (needs network access to VizieR, so it runs in GitHub Actions)
    python -m finetune_dataset.limb_darkening --fetch --out build/finetune/ld_claret2017_tess.parquet
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("finetune.ld")
CATALOG = "J/A+A/600/A30"
SOLAR_Z = 0.0
MICROTURBULENCE = 2.0


def _pick_quadratic(tables) -> pd.DataFrame | None:
    """The quadratic ATLAS table: logg, Teff, Z, xi plus exactly two coefficient columns."""
    best = None
    for table in tables:
        cols = {c.lower(): c for c in table.colnames}
        meta = " ".join(str(v) for v in table.meta.values()).lower()
        LOG.info("table %s: %d rows, columns %s | %s", table.meta.get("name", "?"), len(table), table.colnames,
                 meta[:160])
        needed = ("logg", "teff", "z", "xi")
        if not all(n in cols for n in needed):
            continue
        coeff = [c for c in table.colnames if c.lower() not in needed and c.lower() not in ("met", "mod", "sys")]
        coeff = [c for c in coeff if np.issubdtype(np.asarray(table[c]).dtype, np.floating)]
        is_quadratic = "quadratic" in meta and "atlas" in meta
        if len(coeff) == 2 and is_quadratic:
            frame = table.to_pandas()
            frame = frame.rename(columns={cols["logg"]: "logg", cols["teff"]: "teff", cols["z"]: "z",
                                          cols["xi"]: "xi", coeff[0]: "u1", coeff[1]: "u2"})
            if best is None or len(frame) > len(best):
                best = frame
    return best


def fetch(out: Path) -> bool:
    from astroquery.vizier import Vizier

    vizier = Vizier(row_limit=-1, columns=["**"])
    tables = vizier.get_catalogs(CATALOG)
    frame = _pick_quadratic(tables)
    if frame is None:
        LOG.warning("no quadratic ATLAS table found in %s", CATALOG)
        return False
    frame = frame[np.isclose(frame["z"], SOLAR_Z) & np.isclose(frame["xi"], MICROTURBULENCE)]
    frame = frame[["teff", "logg", "u1", "u2"]].dropna().astype(float)
    if frame.empty:
        LOG.warning("quadratic table has no rows at Z = 0, xi = 2 km/s")
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    LOG.info("wrote %d grid points (Teff %.0f-%.0f K, log g %.1f-%.1f) to %s", len(frame), frame.teff.min(),
             frame.teff.max(), frame.logg.min(), frame.logg.max(), out)
    return True


def load_table(path: Path | None) -> pd.DataFrame | None:
    if path is None or not Path(path).exists():
        return None
    return pd.read_parquet(path)


def surface_gravity(mass_msun: float, radius_rsun: float) -> float:
    return 4.438 + math.log10(mass_msun) - 2.0 * math.log10(radius_rsun)


def coefficients(table: pd.DataFrame, teff: float, logg: float) -> tuple[float, float]:
    """(u1, u2) at the grid point nearest in Teff (scaled by 250 K) and log g (scaled by 0.5 dex)."""
    distance = ((table["teff"].to_numpy() - teff) / 250.0) ** 2 + ((table["logg"].to_numpy() - logg) / 0.5) ** 2
    row = table.iloc[int(np.argmin(distance))]
    return float(row["u1"]), float(row["u2"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.fetch and not fetch(args.out):
        LOG.warning("limb-darkening table unavailable; v3 injections will use the Kipping (2013) draw")


if __name__ == "__main__":
    main()
