"""TESS-band quadratic limb-darkening coefficients from Claret (2017), for batman injections.

What it does
    fetch()        downloads the Claret (2017, A&A 600, A30; VizieR J/A+A/600/A30) tables
                   (CDS archive files read with their ReadMe, else astroquery), finds the
                   quadratic-law table computed from ATLAS models
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
    ld_claret2017_tess.parquet with columns teff, logg, u1, u2 (written by --fetch). A copy is
    kept in finetune_dataset/data/ so builds do not depend on VizieR being reachable;
    materialize.py uses the plan directory's copy first, then that one.

How to run (needs network access to CDS, so it runs in GitHub Actions: fetch-limb-darkening.yml)
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
QUADRATIC_ATLAS_TABLE = f"{CATALOG}/table25"   # "a, b quadratic LDCs, ATLAS models, ... for TESS"
CDS_HOSTS = ("cdsarc.cds.unistra.fr", "cdsarc.u-strasbg.fr")
VIZIER_MIRRORS = ("vizier.cds.unistra.fr", "vizier.cfa.harvard.edu", "vizier.iucaa.in", "vizieridia.saao.ac.za")
SOLAR_Z = 0.0
MICROTURBULENCE = 2.0


def _pick_quadratic(tables) -> pd.DataFrame | None:
    """The quadratic ATLAS table: logg, Teff, Z, xi plus two coefficient columns (least-squares fit)."""
    best = None
    for table in tables:
        cols = {c.lower(): c for c in table.colnames}
        meta = " ".join(str(v) for v in table.meta.values()).lower()
        LOG.info("table %s: %d rows, columns %s | %s", table.meta.get("name", "?"), len(table), table.colnames,
                 meta[:160])
        needed = ("logg", "teff", "z", "xi")
        if not all(n in cols for n in needed):
            continue
        coeff = [c for c in table.colnames if c.lower() not in needed
                 and c.lower() not in ("met", "mod", "sys", "recno", "chi2", "type")]
        coeff = [c for c in coeff if np.issubdtype(np.asarray(table[c]).dtype, np.floating)]
        least_squares = [c for c in coeff if c.upper().endswith("LSM")]
        if len(least_squares) == 2:
            coeff = least_squares        # Claret's least-squares fit (aLSM, bLSM), not flux conservation
        is_quadratic = "quadratic" in meta and "atlas" in meta
        if len(coeff) == 2 and is_quadratic:
            frame = table.to_pandas()
            frame = frame.rename(columns={cols["logg"]: "logg", cols["teff"]: "teff", cols["z"]: "z",
                                          cols["xi"]: "xi", coeff[0]: "u1", coeff[1]: "u2"})
            if best is None or len(frame) > len(best):
                best = frame
    return best


def _from_cds_ftp(work: Path, host: str) -> pd.DataFrame | None:
    """Read the quadratic ATLAS table from the CDS archive files and their ReadMe."""
    import requests
    from astropy.io import ascii as astro_ascii

    base = f"https://{host}/ftp/{CATALOG}/"
    work.mkdir(parents=True, exist_ok=True)
    readme = requests.get(base + "ReadMe", timeout=60)
    readme.raise_for_status()
    (work / "ReadMe").write_text(readme.text)
    summary = readme.text.split("File Summary:")[1].split("Byte-by-byte")[0] if "File Summary:" in readme.text else ""
    candidates = []
    for line in summary.splitlines():
        parts = line.split()
        if not parts or not parts[0].startswith("table"):
            continue
        text = line.lower()
        LOG.info("CDS file: %s", line.strip())
        if "quadratic" in text and ("atlas" in text or "plane" in text):
            candidates.append((0 if "lsm" in text or "least" in text else 1, parts[0]))
    for _, name in sorted(candidates):
        data = requests.get(base + name, timeout=300)
        if data.status_code != 200 and not name.endswith(".gz"):
            data = requests.get(base + name + ".gz", timeout=300)
            name = name + ".gz"
        data.raise_for_status()
        (work / name).write_bytes(data.content)
        table = astro_ascii.read(str(work / name), readme=str(work / "ReadMe"), format="cds")
        frame = _pick_quadratic([_with_meta(table, "quadratic atlas " + name)])
        if frame is not None:
            LOG.info("using %s", name)
            return frame
    return None


def _with_meta(table, text: str):
    table.meta = {**table.meta, "description": text}
    return table


def _from_table25(server: str) -> pd.DataFrame | None:
    """Query Claret's quadratic ATLAS table directly (solar metallicity, xi = 2 km/s), with retries."""
    import time

    from astroquery.vizier import Vizier

    for attempt in range(3):
        vizier = Vizier(columns=["logg", "Teff", "Z", "xi", "aLSM", "bLSM"], row_limit=-1)
        vizier.VIZIER_SERVER = server
        tables = vizier.get_catalogs(QUADRATIC_ATLAS_TABLE)
        if len(tables):
            frame = tables[0].to_pandas()
            LOG.info("%s from %s: %d rows", QUADRATIC_ATLAS_TABLE, server, len(frame))
            return frame.rename(columns={"Teff": "teff", "Z": "z", "aLSM": "u1", "bLSM": "u2"})
        LOG.info("%s returned no table (attempt %d)", server, attempt + 1)
        time.sleep(10)
    return None


def _from_astroquery(server: str) -> pd.DataFrame | None:
    from astroquery.vizier import Vizier

    vizier = Vizier(row_limit=-1, columns=["**"])
    vizier.VIZIER_SERVER = server
    return _pick_quadratic(vizier.get_catalogs(CATALOG))


def fetch(out: Path) -> bool:
    frame = None
    sources = [(f"{QUADRATIC_ATLAS_TABLE} at {m}", lambda m=m: _from_table25(m)) for m in VIZIER_MIRRORS]
    sources += [(f"CDS archive files at {h}", lambda h=h: _from_cds_ftp(out.parent / "ld_cds", h)) for h in CDS_HOSTS]
    sources += [(f"astroquery VizieR at {m}", lambda m=m: _from_astroquery(m)) for m in VIZIER_MIRRORS]
    for name, method in sources:
        try:
            frame = method()
        except Exception as exc:  # noqa: BLE001 - try the next source, log why this one failed
            LOG.warning("%s failed: %s", name, exc)
            frame = None
        if frame is not None:
            LOG.info("limb-darkening table from %s", name)
            break
    if frame is None:
        LOG.warning("no quadratic ATLAS table found in %s", CATALOG)
        return False
    frame = frame[np.isclose(frame["z"].astype(float), SOLAR_Z) & np.isclose(frame["xi"].astype(float), MICROTURBULENCE)]
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
