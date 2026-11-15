"""Build the KIC-to-TIC crossmatch for every star in the Kepler light-curve set.

What it does
    1. Reads every kic_id from manifest.parquet in the Hugging Face dataset
       nyu-planet-finders/kepler-llc (one row per original Kepler FITS file,
       about 200,000 unique stars). A local Parquet or CSV file with a kic_id
       column can be given instead with --kic-list.
    2. Queries the TESS Input Catalog v8.2 at MAST (service
       Mast.Catalogs.Filtered.Tic) on its KIC column, in batches.
    3. Writes one row per Kepler star, including stars with no TIC match, so
       coverage is explicit:

        kic_id, tic_id, n_tic_matches, tic_disposition, tic_ra_deg,
        tic_dec_deg, tmag, matched

    When several TIC entries carry the same KIC ID, the entry without a TIC
    disposition (that is, not flagged as a duplicate, split or artifact) is
    preferred, then the brightest; n_tic_matches records how many there were.

Why
    Splits and exclusions are keyed on one star identity. Kepler-field stars
    that TESS also observed must map to the same star_key (tic:<id>) so they
    land in the same split in both missions.

Output
    kic_tic_crossmatch.parquet

Run
    python -m reference_catalogs.build_kic_tic_crossmatch --out build/reference_catalogs
    python -m reference_catalogs.build_kic_tic_crossmatch --kic-list my_kics.parquet --batch-size 500
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import LOG, CatalogResult, configure_logging, http_request, utc_now, write_parquet

NAME = "kic_tic_crossmatch"
FILENAME = f"{NAME}.parquet"
MAST_INVOKE_URL = "https://mast.stsci.edu/api/v0/invoke"
MAST_SERVICE = "Mast.Catalogs.Filtered.Tic"
KEPLER_REPO = "nyu-planet-finders/kepler-llc"
KEPLER_MANIFEST = "manifest.parquet"
SOURCE = "MAST TIC v8.2 (Mast.Catalogs.Filtered.Tic, KIC column)"
MIN_MATCH_FRACTION = 0.5


def load_kic_ids(kic_list: Path | None) -> np.ndarray:
    if kic_list is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(KEPLER_REPO, KEPLER_MANIFEST, repo_type="dataset")
        frame = pd.read_parquet(path, columns=["kic_id"])
    elif kic_list.suffix == ".csv":
        frame = pd.read_csv(kic_list, usecols=["kic_id"])
    else:
        frame = pd.read_parquet(kic_list, columns=["kic_id"])
    ids = pd.to_numeric(frame["kic_id"], errors="coerce").dropna().astype("int64").unique()
    ids.sort()
    LOG.info("Loaded %d unique KIC IDs", len(ids))
    return ids


COLUMN_SPECS = ("c.*", "*")


def query_mast(kic_batch: list[int], values_as_strings: bool, columns: str = "c.*",
               page_size: int = 5000) -> list[dict[str, Any]]:
    """Return every TIC row whose KIC is in the batch, following MAST paging."""
    values: list[Any] = [str(k) for k in kic_batch] if values_as_strings else list(kic_batch)
    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        request = {
            "service": MAST_SERVICE,
            "format": "json",
            "params": {"columns": columns, "filters": [{"paramName": "KIC", "values": values}]},
            "pagesize": page_size,
            "page": page,
            "removenullcolumns": False,
            "timeout": 60,
        }
        for _ in range(30):
            reply = http_request("POST", MAST_INVOKE_URL, data={"request": json.dumps(request)}).json()
            if reply.get("status") == "COMPLETE":
                break
            if reply.get("status") == "ERROR":
                raise RuntimeError(f"MAST error: {reply.get('msg')}")
            time.sleep(2)
        else:
            raise RuntimeError("MAST query did not complete after 30 polls")
        rows.extend(reply.get("data", []))
        paging = reply.get("paging", {})
        if page >= int(paging.get("pagesFiltered", 1) or 1):
            return rows
        page += 1


def detect_query_format(kic_ids: np.ndarray) -> tuple[str, bool]:
    """Find a column spec and value type that MAST accepts for this filter.

    Tries the first 200 IDs with each combination and keeps the first that
    returns matches. Kepler targets are almost all in the TIC, so an empty or
    failed reply means the request format was not understood.
    """
    probe = [int(k) for k in kic_ids[:200]]
    for columns in COLUMN_SPECS:
        for as_strings in (False, True):
            try:
                matched = query_mast(probe, as_strings, columns)
            except Exception as exc:
                LOG.warning("Probe columns=%r strings=%s failed: %s", columns, as_strings, exc)
                continue
            LOG.info("Probe columns=%r strings=%s: %d rows", columns, as_strings, len(matched))
            if matched:
                return columns, as_strings
    raise RuntimeError("MAST returned no TIC rows for 200 Kepler targets in any request format")


def pick_best(rows: pd.DataFrame) -> pd.DataFrame:
    """One TIC per KIC: no disposition first, then brightest Tmag."""
    rows = rows.copy()
    rows["_has_disposition"] = rows["tic_disposition"].notna() & (rows["tic_disposition"].astype(str).str.len() > 0)
    rows["_tmag_sort"] = rows["tmag"].fillna(99.0)
    rows = rows.sort_values(["kic_id", "_has_disposition", "_tmag_sort", "tic_id"])
    counts = rows.groupby("kic_id")["tic_id"].transform("count")
    rows["n_tic_matches"] = counts.astype("int16")
    best = rows.drop_duplicates("kic_id", keep="first")
    return best.drop(columns=["_has_disposition", "_tmag_sort"])


def build(out_dir: Path, kic_list: Path | None = None, batch_size: int = 500, workers: int = 4) -> CatalogResult:
    retrieved = utc_now()
    kic_ids = load_kic_ids(kic_list)
    columns_spec, as_strings = detect_query_format(kic_ids)

    batches = [kic_ids[i:i + batch_size].tolist() for i in range(0, len(kic_ids), batch_size)]
    LOG.info("Querying MAST in %d batches of up to %d with %d workers", len(batches), batch_size, workers)

    collected: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(query_mast, batch, as_strings, columns_spec): i for i, batch in enumerate(batches)}
        for done, future in enumerate(as_completed(futures), start=1):
            collected.extend(future.result())
            if done % 50 == 0 or done == len(batches):
                LOG.info("Batches done: %d/%d, TIC rows so far: %d", done, len(batches), len(collected))

    raw = pd.DataFrame.from_records(collected)
    if raw.empty:
        raise RuntimeError("MAST returned no rows")
    lower = {c.lower(): c for c in raw.columns}

    def column(name: str) -> pd.Series:
        key = lower.get(name.lower())
        return raw[key] if key else pd.Series(pd.NA, index=raw.index)

    matches = pd.DataFrame({
        "kic_id": pd.to_numeric(column("KIC"), errors="coerce").astype("Int64"),
        "tic_id": pd.to_numeric(column("ID"), errors="coerce").astype("Int64"),
        "tic_disposition": column("disposition").astype("string"),
        "tic_ra_deg": pd.to_numeric(column("ra"), errors="coerce"),
        "tic_dec_deg": pd.to_numeric(column("dec"), errors="coerce"),
        "tmag": pd.to_numeric(column("Tmag"), errors="coerce"),
    })
    requested = set(int(k) for k in kic_ids)
    matches = matches[matches["kic_id"].isin(requested)]
    best = pick_best(matches)

    result = pd.DataFrame({"kic_id": pd.array(kic_ids, dtype="Int64")}).merge(best, on="kic_id", how="left")
    result["matched"] = result["tic_id"].notna()
    result["n_tic_matches"] = result["n_tic_matches"].fillna(0).astype("int16")

    fraction = float(result["matched"].mean())
    LOG.info("Matched %d of %d KIC IDs (%.1f%%)", int(result["matched"].sum()), len(result), 100 * fraction)
    if fraction < MIN_MATCH_FRACTION:
        raise RuntimeError(f"Only {fraction:.1%} of KIC IDs matched a TIC entry; check the query")

    notes = {
        "kic_ids_requested": len(result),
        "matched": int(result["matched"].sum()),
        "match_fraction": round(fraction, 4),
        "kic_with_multiple_tic": int((result["n_tic_matches"] > 1).sum()),
        "kic_source": str(kic_list) if kic_list else f"hf://datasets/{KEPLER_REPO}/{KEPLER_MANIFEST}",
        "mast_values_as_strings": as_strings,
        "mast_columns_spec": columns_spec,
    }
    rows = write_parquet(result, out_dir / FILENAME, {
        "name": NAME, "source": SOURCE, "retrieved_utc": retrieved, **notes,
    })
    return CatalogResult(NAME, FILENAME, rows, SOURCE, retrieved, notes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("build/reference_catalogs"))
    parser.add_argument("--kic-list", type=Path, default=None,
                        help="Parquet or CSV with a kic_id column; default is the kepler-llc manifest")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    configure_logging()
    LOG.info("Done: %s", build(args.out, args.kic_list, args.batch_size, args.workers).to_dict())


if __name__ == "__main__":
    main()
