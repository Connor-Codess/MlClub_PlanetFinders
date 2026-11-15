"""Extract the 85 duotransit candidates of Hawthorn et al. (2024) from arXiv.

What it does
    Downloads the arXiv HTML version of "TESS Duotransit Candidates from the
    Southern Ecliptic Hemisphere" (arXiv:2310.17268) and parses appendix
    Tables A1-A3, which list every candidate by ID (1-85) and TIC ID:

        A1  host star: TOI/CTOI, RA, Dec, Tmag, radius, Teff, sectors
        A2  the two transits: Tc1 and Tc2 (TBJD) with their sectors, depths, duration
        A3  derived: planet radius, period range, number of aliases, most probable period

    Rows are recognised by the shape of their cells rather than by header text,
    because the tables are split into continuation blocks and their headers
    contain math markup. The three tables are merged on candidate ID. The build
    fails unless exactly 85 candidates with two transit times are found.

Why
    These stars become real-transit test stars (Step 2 of the build guide) and
    are excluded from pretraining. The two transit times are also the reference
    answers for the duotransit-matching stage.

Output
    duotransits_hawthorn2024.parquet with columns:
        candidate_id, tic_id, toi_or_ctoi, ra_deg, dec_deg, tmag,
        star_radius_rsun, teff_k, n_sectors,
        tc1_btjd, tc1_sector, tc2_btjd, tc2_sector,
        depth1_ppt, depth2_ppt, duration_h,
        planet_radius_re, period_min_d, period_max_d, n_alias, period_marg_d, total_snr

    TBJD in the paper is the same as BTJD (BJD - 2457000).

Reference
    Hawthorn et al. 2024, MNRAS 528, 1841, doi:10.1093/mnras/stad3783

Run
    python -m reference_catalogs.fetch_hawthorn2024_duotransits --out build/reference_catalogs
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .common import LOG, CatalogResult, configure_logging, http_request, utc_now, write_parquet

NAME = "duotransits_hawthorn2024"
FILENAME = f"{NAME}.parquet"
HTML_URLS = [
    "https://arxiv.org/html/2310.17268v2",
    "https://arxiv.org/html/2310.17268",
]
SOURCE = "arXiv:2310.17268v2 HTML, Tables A1-A3 (Hawthorn et al. 2024, MNRAS 528, 1841)"
EXPECTED_CANDIDATES = 85

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
_TIME_SECTOR = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*\[\s*(\d+)\s*\]\s*$")
_DEPTH_SECTOR = re.compile(r"(\d+(?:\.\d+)?)\s*\[\s*(\d+)\s*\]")
_PERIOD_RANGE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*$")
_INTEGER = re.compile(r"^\s*(\d+)\s*$")


def leading_number(text: str | None) -> float | None:
    """First number in a cell such as '9.6 ± 1.0' or '382.06^{+12.79}_{-9.96}'."""
    if text is None:
        return None
    match = _NUMBER.search(text.replace("−", "-"))
    return float(match.group()) if match else None


def _integer(text: str) -> int | None:
    match = _INTEGER.match(text)
    return int(match.group(1)) if match else None


def table_rows_from_html(html: str) -> list[list[str]]:
    """Return the text of every table row, with each <math> replaced by its TeX alttext."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for math in soup.find_all("math"):
        math.replace_with(" " + (math.get("alttext") or math.get_text(" ")) + " ")
    rows: list[list[str]] = []
    for tr in soup.find_all("tr"):
        cells = [" ".join(td.get_text(" ").split()) for td in tr.find_all(["td", "th"])]
        if cells:
            rows.append(cells)
    return rows


def classify_rows(rows: list[list[str]]) -> dict[str, dict[int, dict]]:
    """Sort candidate rows into tables A1, A2 and A3 by the shape of their cells."""
    found: dict[str, dict[int, dict]] = {"A1": {}, "A2": {}, "A3": {}}
    for cells in rows:
        if len(cells) < 6:
            continue
        candidate_id, tic_id = _integer(cells[0]), _integer(cells[1])
        if candidate_id is None or tic_id is None or not 1 <= candidate_id <= 500:
            continue

        tc1, tc2 = _TIME_SECTOR.match(cells[2]), _TIME_SECTOR.match(cells[3])
        if tc1 and tc2:
            depths = _DEPTH_SECTOR.findall(cells[4])
            found["A2"][candidate_id] = {
                "tic_id": tic_id,
                "tc1_btjd": float(tc1.group(1)), "tc1_sector": int(tc1.group(2)),
                "tc2_btjd": float(tc2.group(1)), "tc2_sector": int(tc2.group(2)),
                "depth1_ppt": float(depths[0][0]) if len(depths) > 0 else None,
                "depth2_ppt": float(depths[1][0]) if len(depths) > 1 else None,
                "duration_h": leading_number(cells[5]),
            }
            continue

        if len(cells) >= 10:
            period_range = _PERIOD_RANGE.match(cells[3])
            if period_range:
                found["A3"][candidate_id] = {
                    "tic_id": tic_id,
                    "planet_radius_re": leading_number(cells[2]),
                    "period_min_d": float(period_range.group(1)),
                    "period_max_d": float(period_range.group(2)),
                    "n_alias": _integer(cells[4]),
                    "period_marg_d": leading_number(cells[5]),
                    "total_snr": leading_number(cells[9]),
                }
                continue

            ra, dec, tmag = (leading_number(cells[i]) for i in (3, 4, 5))
            if (ra is not None and 0 <= ra <= 360 and dec is not None and -90 <= dec <= 90
                    and tmag is not None and 3 <= tmag <= 20):
                toi = cells[2].strip()
                found["A1"][candidate_id] = {
                    "tic_id": tic_id,
                    "toi_or_ctoi": None if toi in {"_", "-", "–", "—", ""} else toi,
                    "ra_deg": ra, "dec_deg": dec, "tmag": tmag,
                    "star_radius_rsun": leading_number(cells[6]),
                    "teff_k": leading_number(cells[7]),
                    "n_sectors": _integer(cells[9]),
                }
    return found


def merge_tables(found: dict[str, dict[int, dict]]) -> pd.DataFrame:
    """Join A1-A3 on candidate ID; A2 (the transit times) defines the candidate list."""
    records = []
    for candidate_id, transit in sorted(found["A2"].items()):
        record = {"candidate_id": candidate_id, **transit}
        for table in ("A1", "A3"):
            extra = found[table].get(candidate_id)
            if extra is None:
                continue
            if extra["tic_id"] != transit["tic_id"]:
                raise ValueError(
                    f"Candidate {candidate_id}: TIC {extra['tic_id']} in {table} "
                    f"but {transit['tic_id']} in A2"
                )
            record.update({k: v for k, v in extra.items() if k != "tic_id"})
        records.append(record)

    columns = [
        "candidate_id", "tic_id", "toi_or_ctoi", "ra_deg", "dec_deg", "tmag",
        "star_radius_rsun", "teff_k", "n_sectors",
        "tc1_btjd", "tc1_sector", "tc2_btjd", "tc2_sector",
        "depth1_ppt", "depth2_ppt", "duration_h",
        "planet_radius_re", "period_min_d", "period_max_d", "n_alias", "period_marg_d", "total_snr",
    ]
    df = pd.DataFrame.from_records(records).reindex(columns=columns)
    for column in ("candidate_id", "tic_id", "tc1_sector", "tc2_sector", "n_sectors", "n_alias"):
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    df["toi_or_ctoi"] = df["toi_or_ctoi"].astype("string")
    return df


def download_html() -> tuple[str, str]:
    last_error: Exception | None = None
    for url in HTML_URLS:
        try:
            return http_request("GET", url).text, url
        except Exception as exc:  # try the next URL
            last_error = exc
            LOG.warning("Could not fetch %s: %s", url, exc)
    raise RuntimeError("No arXiv HTML version could be downloaded") from last_error


def build(out_dir: Path) -> CatalogResult:
    retrieved = utc_now()
    html, url = download_html()
    found = classify_rows(table_rows_from_html(html))
    LOG.info("Rows found: A1=%d, A2=%d, A3=%d", *(len(found[t]) for t in ("A1", "A2", "A3")))
    df = merge_tables(found)

    if len(df) != EXPECTED_CANDIDATES:
        raise RuntimeError(f"Parsed {len(df)} candidates with transit times; expected {EXPECTED_CANDIDATES}")
    if df["tic_id"].duplicated().any():
        raise RuntimeError("Duplicate TIC IDs among parsed candidates")
    if (df["tc2_btjd"] <= df["tc1_btjd"]).any():
        raise RuntimeError("Some candidates have Tc2 earlier than Tc1; parsing went wrong")

    notes = {
        "html_url": url,
        "rows_a1": len(found["A1"]), "rows_a2": len(found["A2"]), "rows_a3": len(found["A3"]),
        "with_toi_or_ctoi": int(df["toi_or_ctoi"].notna().sum()),
    }
    rows = write_parquet(df, out_dir / FILENAME, {
        "name": NAME, "source": SOURCE, "retrieved_utc": retrieved,
        "time_reference": "tc1_btjd and tc2_btjd are TBJD from the paper (BJD - 2457000)", **notes,
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
