"""Offline tests for the reference-catalog builders.

These cover parsing and standardization logic with small hand-made inputs, so
they run without network access:  python -m pytest tests -q
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from reference_catalogs import (
    build_kic_tic_crossmatch,
    fetch_hawthorn2024_duotransits as hawthorn,
    fetch_kepler_eb_catalog,
    fetch_tess_eb_catalog,
    fetch_toi_ephemerides,
)
from reference_catalogs.common import infer_kepler_time_reference, snake_case, to_btjd, write_parquet
from reference_catalogs.upload_reference_catalogs_to_hub import render_card


def test_snake_case_handles_catalog_labels():
    assert snake_case("Wp-pf") == "wp_pf"
    assert snake_case("e_BJD0") == "e_bjd0"
    assert snake_case("RAdeg") == "radeg"
    assert snake_case("pmRA") == "pm_ra"
    assert snake_case("m_TIC") == "m_tic"


def test_time_conversions():
    assert to_btjd(pd.Series([2459000.5]), "BJD").iloc[0] == pytest.approx(2000.5)
    assert to_btjd(pd.Series([55000.0]), "BJD-2400000").iloc[0] == pytest.approx(-2000.0)
    assert to_btjd(pd.Series([167.0]), "BKJD").iloc[0] == pytest.approx(-2000.0)
    with pytest.raises(ValueError):
        to_btjd(pd.Series([1.0]), "MJD")


def test_infer_kepler_time_reference():
    assert infer_kepler_time_reference(pd.Series([2455000.1, 2455100.2])) == "BJD"
    assert infer_kepler_time_reference(pd.Series([54965.1, 55001.3])) == "BJD-2400000"
    assert infer_kepler_time_reference(pd.Series([131.5, 170.9])) == "BKJD"


HAWTHORN_HTML = """
<html><body>
<table class="ltx_tabular">
<tr><th>ID</th><th>TIC ID</th><th>TOI/CTOI</th><th>RA (deg)</th><th>Dec (deg)</th>
<th><math alttext="T_{\\rm mag}">T</math></th><th>R</th><th>Teff (K)</th><th>Distance (pc)</th><th>N</th></tr>
<tr><td>1</td><td>1504460</td><td>_</td><td>131.9658</td><td>-23.9895</td><td>11.63</td><td>1.5</td>
<td>6218</td><td><math alttext="451.71^{+20.73}_{-15.90}">451.71 +20.73 -15.90</math></td><td>2</td></tr>
<tr><td>85</td><td>466206508</td><td>TOI-5542</td><td>302.7985</td><td>-61.1355</td><td>11.85</td><td>1.27</td>
<td>5393</td><td><math alttext="342.66^{+1.69}_{-1.51}">x</math></td><td>2</td></tr>
</table>
<table class="ltx_tabular">
<tr><th>ID</th><th>TIC ID</th><th>Tc1</th><th>Tc2</th><th>Depth</th><th>Duration</th></tr>
<tr><td>1</td><td>1504460</td><td>1528.39739 [08]</td><td>2261.11990 [35]</td><td>3.367 [08], 3.415 [35]</td><td>3.78</td></tr>
<tr><td>85</td><td>466206508</td><td>1679.34576 [13]</td><td>2054.97133 [27]</td><td>11.214 [13], 9.274 [27]</td><td>8.03</td></tr>
</table>
<table class="ltx_tabular">
<tr><td>1</td><td>1504460</td><td>9.6 ± 1.0</td><td>19.80333, 732.72251</td><td>24</td><td>20.0+19.0</td>
<td><math alttext="0.78^{+0.12}_{-0.37}">y</math></td><td>0</td><td>13.84</td><td>14.82</td></tr>
<tr><td>85</td><td>466206508</td><td><math alttext="13.55\\pm 0.82">z</math></td><td>25.04186, 375.62557</td><td>16</td>
<td><math alttext="31.0^{+16.0}_{-6.0}">w</math></td><td><math alttext="0.37\\pm 0.2">v</math></td><td>0</td><td>12.81</td><td>64.68</td></tr>
</table>
<table class="ltx_tabular">
<tr><td>82286</td><td>Asteroid</td><td>1606.56693 [11]</td><td>2337.34989 [38]</td></tr>
</table>
</body></html>
"""


def test_hawthorn_rows_are_classified_and_merged():
    rows = hawthorn.table_rows_from_html(HAWTHORN_HTML)
    found = hawthorn.classify_rows(rows)
    assert set(found["A1"]) == {1, 85}
    assert set(found["A2"]) == {1, 85}
    assert set(found["A3"]) == {1, 85}

    df = hawthorn.merge_tables(found)
    first = df.set_index("candidate_id").loc[1]
    assert first["tic_id"] == 1504460
    assert first["tc1_btjd"] == pytest.approx(1528.39739)
    assert first["tc1_sector"] == 8 and first["tc2_sector"] == 35
    assert first["depth2_ppt"] == pytest.approx(3.415)
    assert first["period_min_d"] == pytest.approx(19.80333)
    assert first["n_alias"] == 24
    assert pd.isna(first["toi_or_ctoi"])

    last = df.set_index("candidate_id").loc[85]
    assert last["toi_or_ctoi"] == "TOI-5542"
    assert last["planet_radius_re"] == pytest.approx(13.55)
    assert last["period_marg_d"] == pytest.approx(31.0)
    assert last["total_snr"] == pytest.approx(64.68)


def test_hawthorn_rejects_tic_mismatch_between_tables():
    found = {
        "A1": {1: {"tic_id": 2}},
        "A2": {1: {"tic_id": 1, "tc1_btjd": 1.0, "tc1_sector": 1, "tc2_btjd": 2.0, "tc2_sector": 2,
                   "depth1_ppt": 1.0, "depth2_ppt": 1.0, "duration_h": 3.0}},
        "A3": {},
    }
    with pytest.raises(ValueError):
        hawthorn.merge_tables(found)


def test_toi_standardize():
    raw = pd.DataFrame({
        "toi": [101.01, 5542.01, 700.02],
        "tid": [231663901, 466206508, 1],
        "tfopwg_disp": ["KP", "PC", "FP"],
        "pl_orbper": [1.43, 0.0, 30.5],
        "pl_tranmid": [2458325.5, 2459679.3, 2458400.0],
        "pl_trandurh": [1.6, 8.0, 5.0],
        "pl_trandep": [13000.0, 9000.0, 400.0],
        "sectors": ["1,2", "13,27", "5"],
    })
    df = fetch_toi_ephemerides.standardize(raw)
    by_toi = df.set_index("toi")
    assert by_toi.loc[101.01, "t0_btjd"] == pytest.approx(1325.5)
    assert not by_toi.loc[5542.01, "period_known"]
    assert by_toi.loc[5542.01, "long_period_or_unknown"]
    assert by_toi.loc[700.02, "long_period_or_unknown"]
    assert not by_toi.loc[101.01, "long_period_or_unknown"]
    assert np.isnan(by_toi.loc[101.01, "planet_radius_re"])
    assert "archive_pl_tranmid" in df.columns


def test_tess_eb_standardize():
    raw = pd.DataFrame({
        "TIC": [91961, 91961], "m_TIC": [1, 2], "BJD0": [1468.5, 1470.1], "Per": [8.0, 0.0],
        "Morph": [0.2, 0.9], "Tmag": [10.0, 10.0], "RAdeg": [10.0, 10.0], "DEdeg": [-5.0, -5.0],
        "Wp-pf": [0.01, np.nan], "Wp-2g": [0.02, 0.03], "Dp-pf": [0.3, 0.1], "Sectors": ["1,2", "1"],
    })
    df = fetch_tess_eb_catalog.standardize(raw)
    assert list(df["signal_id"]) == [1, 2]
    assert df.loc[0, "t0_btjd"] == pytest.approx(1468.5)
    assert df.loc[0, "primary_duration_h"] == pytest.approx(0.01 * 8.0 * 24)
    assert np.isnan(df.loc[1, "period_d"])
    assert df.loc[1, "primary_width_phase"] == pytest.approx(0.03)
    assert "vizier_wp_pf" in df.columns


def test_kepler_eb_standardize_converts_epochs():
    raw = pd.DataFrame({"KIC": [1026032, 1026957], "Per": [8.46, 21.76],
                        "BJD0": [54966.77, 54981.65], "morph": [0.1, 0.2]})
    df, reference = fetch_kepler_eb_catalog.standardize(raw)
    assert reference == "BJD-2400000"
    assert df.loc[0, "t0_btjd"] == pytest.approx(54966.77 + 2400000 - 2457000)


def test_kepler_eb_standardize_without_epoch_column():
    raw = pd.DataFrame({"KIC": [1026032], "Per": [8.46]})
    df, reference = fetch_kepler_eb_catalog.standardize(raw)
    assert reference == "unknown"
    assert np.isnan(df.loc[0, "t0_btjd"])


def test_crossmatch_pick_best_prefers_clean_then_bright():
    rows = pd.DataFrame({
        "kic_id": pd.array([10, 10, 10, 20], dtype="Int64"),
        "tic_id": pd.array([1, 2, 3, 4], dtype="Int64"),
        "tic_disposition": pd.array(["DUPLICATE", None, None, None], dtype="string"),
        "tic_ra_deg": [1.0, 1.0, 1.0, 2.0],
        "tic_dec_deg": [1.0, 1.0, 1.0, 2.0],
        "tmag": [9.0, 12.0, 11.0, 13.0],
    })
    best = build_kic_tic_crossmatch.pick_best(rows).set_index("kic_id")
    assert best.loc[10, "tic_id"] == 3
    assert best.loc[10, "n_tic_matches"] == 3
    assert best.loc[20, "n_tic_matches"] == 1


def test_write_parquet_stores_provenance(tmp_path):
    path = tmp_path / "x.parquet"
    write_parquet(pd.DataFrame({"a": [1, 2]}), path, {"name": "x", "source": "test"})
    import pyarrow.parquet as pq

    meta = json.loads(pq.read_schema(path).metadata[b"reference_catalog"])
    assert meta == {"name": "x", "source": "test"}


def test_render_card_lists_configs_and_failures():
    manifest = {
        "catalogs": {
            "toi_ephemerides": {"name": "toi_ephemerides", "filename": "toi_ephemerides.parquet",
                                "rows": 7700, "source": "s", "retrieved_utc": "2026-11-15T00:00:00Z",
                                "notes": {"period_known": 7000}},
        },
        "failed": {"kic_tic_crossmatch": "RuntimeError: boom"},
    }
    card = render_card(manifest, "nyu-planet-finders/reference-catalogs")
    assert "config_name: toi_ephemerides" in card
    assert "path: toi_ephemerides.parquet" in card
    assert "7,700" in card
    assert "kic_tic_crossmatch`: RuntimeError: boom" in card
