"""Shared helpers for the reference-catalog builders.

Every builder in this package downloads one external catalog, standardizes it,
and writes a single Parquet file. The helpers here keep HTTP retries, time
conversions, column naming and Parquet metadata identical across builders, so
that the files on Hugging Face can be joined without per-file special cases.

Time convention used everywhere in this project:
    BTJD = BJD_TDB - 2457000.0   (TESS Barycentric Julian Date)
    BKJD = BJD_TDB - 2454833.0   (Kepler Barycentric Julian Date)
    BTJD = BKJD - 2167.0
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

LOG = logging.getLogger("reference_catalogs")

BTJD_OFFSET = 2457000.0
BKJD_OFFSET = 2454833.0
MJD_LIKE_OFFSET = 2400000.0

USER_AGENT = (
    "nyu-planet-finders-reference-catalogs/1.0 "
    "(+https://github.com/Connor-Codess/MlClub_PlanetFinders)"
)

PARQUET_METADATA_KEY = b"reference_catalog"


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=level,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class CatalogResult:
    """Summary of one built catalog, collected into build_manifest.json."""

    name: str
    filename: str
    rows: int
    source: str
    retrieved_utc: str
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def http_request(
    method: str,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    timeout: float = 180.0,
    retries: int = 5,
    backoff_s: float = 5.0,
) -> requests.Response:
    """Send an HTTP request with retries on network errors and 429/5xx replies."""
    headers = {"User-Agent": USER_AGENT}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.request(
                method, url, params=params, data=data, headers=headers, timeout=timeout
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(
                    f"HTTP {response.status_code} from {url}", response=response
                )
            response.raise_for_status()
            return response
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            last_error = exc
            wait = backoff_s * 2 ** (attempt - 1)
            LOG.warning("Request failed (%s), attempt %d/%d; retrying in %.0f s",
                        exc, attempt, retries, wait)
            time.sleep(wait)
    raise RuntimeError(f"Giving up on {url} after {retries} attempts") from last_error


def to_btjd(values: pd.Series | np.ndarray, reference: str) -> pd.Series:
    """Convert times in a known reference frame to BTJD.

    reference must be one of: "BJD", "BJD-2400000", "BKJD", "BTJD".
    """
    v = pd.to_numeric(pd.Series(values), errors="coerce").astype("float64")
    if reference == "BJD":
        return v - BTJD_OFFSET
    if reference == "BJD-2400000":
        return v + MJD_LIKE_OFFSET - BTJD_OFFSET
    if reference == "BKJD":
        return v + BKJD_OFFSET - BTJD_OFFSET
    if reference == "BTJD":
        return v
    raise ValueError(f"Unknown time reference: {reference!r}")


def infer_kepler_time_reference(values: pd.Series) -> str:
    """Guess whether Kepler-era epochs are full BJD, BJD-2400000 or BKJD.

    Kepler observed from 2009 to 2013, i.e. BJD 2454953-2456424. The three
    conventions in use put the median epoch in clearly separated ranges.
    """
    median = float(pd.to_numeric(values, errors="coerce").median())
    if median > 2_000_000:
        return "BJD"
    if 40_000 < median < 100_000:
        return "BJD-2400000"
    if -100 < median < 3_000:
        return "BKJD"
    raise ValueError(f"Cannot infer the time reference from median epoch {median}")


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def snake_case(name: str) -> str:
    """Turn catalog labels such as 'Wp-pf', 'e_BJD0' or 'RAdeg' into snake_case."""
    text = _CAMEL_BOUNDARY.sub("_", str(name).strip())
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def find_column(columns: list[str], *candidates: str, required: bool = True) -> str | None:
    """Return the first column whose name matches a candidate, ignoring case."""
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if required:
        raise KeyError(f"None of {candidates} found among columns {columns}")
    return None


def write_parquet(df: pd.DataFrame, path: Path, metadata: Mapping[str, Any]) -> int:
    """Write a DataFrame to zstd Parquet with provenance stored in the file footer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    existing = table.schema.metadata or {}
    table = table.replace_schema_metadata(
        {**existing, PARQUET_METADATA_KEY: json.dumps(dict(metadata), default=str).encode()}
    )
    pq.write_table(table, path, compression="zstd")
    LOG.info("Wrote %s (%d rows, %d columns)", path, len(df), len(df.columns))
    return len(df)


TAPVIZIER_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"


def fetch_vizier_table(table: str, cds_dirs: list[str], data_file: str, min_rows: int) -> tuple[pd.DataFrame, str]:
    """Download a whole VizieR table from the CDS archive files, or from TAPVizieR as a fallback.

    table      VizieR table name, e.g. "J/ApJS/258/16/tess-ebs"
    cds_dirs   archive directories holding ReadMe and the data file
    data_file  data file name inside those directories, e.g. "tess-ebs.dat"
    min_rows   a result with fewer rows is treated as truncated and the next source is tried

    The static archive files are tried first because they are small and fast;
    every request uses a short timeout so an unresponsive service fails over
    within minutes. Returns the table and a short description of its origin.
    """
    import gzip
    import io
    import tempfile

    from astropy.io import ascii as astropy_ascii

    errors: list[str] = []
    for base in cds_dirs:
        try:
            readme = http_request("GET", f"{base}/ReadMe", timeout=60, retries=2).content
            content = None
            for name in (f"{data_file}.gz", data_file):
                try:
                    raw = http_request("GET", f"{base}/{name}", timeout=120, retries=2).content
                    content = gzip.decompress(raw) if name.endswith(".gz") else raw
                    break
                except Exception as exc:
                    errors.append(f"{base}/{name}: {exc}")
            if content is None:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                readme_path, data_path = Path(tmp) / "ReadMe", Path(tmp) / data_file
                readme_path.write_bytes(readme)
                data_path.write_bytes(content)
                parsed = astropy_ascii.read(str(data_path), format="cds", readme=str(readme_path))
            df = parsed.to_pandas()
            if len(df) >= min_rows:
                LOG.info("CDS archive %s returned %d rows", base, len(df))
                return df, f"CDS archive {base}/{data_file}"
            errors.append(f"{base}: only {len(df)} rows")
        except Exception as exc:
            errors.append(f"{base}: {exc}")
        LOG.warning("CDS archive %s did not deliver %s: %s", base, data_file, errors[-1])

    try:
        response = http_request("GET", TAPVIZIER_URL, timeout=120, retries=2, params={
            "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "MAXREC": "2000000",
            "QUERY": f'SELECT * FROM "{table}"',
        })
        df = pd.read_csv(io.StringIO(response.text), low_memory=False)
        if len(df) >= min_rows:
            LOG.info("TAPVizieR returned %d rows for %s", len(df), table)
            return df, f"TAPVizieR {table}"
        errors.append(f"TAPVizieR returned only {len(df)} rows")
    except Exception as exc:
        errors.append(f"TAPVizieR: {exc}")
    raise RuntimeError(f"Could not download {table}: " + " | ".join(errors))


def github_annotation(level: str, title: str, message: str) -> None:
    """Print a GitHub Actions annotation; plain log line when run elsewhere."""
    print(f"::{level} title={title}::{message}", flush=True)
