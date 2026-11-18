"""Offline tests for finetune_dataset: labels, event measurement, injections, templates, pools.

Run: python -m pytest tests -q
"""

import numpy as np
import pandas as pd
import pytest

from finetune_dataset import config
from finetune_dataset.injection import (Event, apply_model, batman_injection, measure_event, patch_labels,
                                        transplant_injection)
from finetune_dataset.pools import classify, is_long_period, pool_for, scoreboard_half
from finetune_dataset.templates import build_template, cutout

W = config.WINDOW_SLOTS


def planet(center, duration=6.0, observed_fraction=1.0, kind=config.KIND_REAL_PLANET):
    e = Event(center, duration, kind, "toi", "TOI-1.01 PC", "planet", 5.0)
    e.observed_fraction = observed_fraction
    return e


class TestLabels:
    def test_center_patch_is_positive(self):
        labels = patch_labels([planet(100.0)])
        assert labels["patch_label"][100 // 8] == config.LABEL_TRANSIT
        assert labels["patch_label"].sum() == 1
        assert labels["patch_center_offset"][12] == pytest.approx(100 / 8 - 12)
        assert labels["patch_overlap"][12] == pytest.approx(0.75)    # slots 97-103 of patch 96-104
        assert labels["patch_kind"][12] == config.KIND_REAL_PLANET

    def test_hidden_transit_is_ignored(self):
        labels = patch_labels([planet(100.0, observed_fraction=0.2)])
        assert labels["patch_label"][12] == config.LABEL_IGNORE

    def test_eclipse_is_negative_but_marked(self):
        eb = Event(100.0, 6.0, config.KIND_ECLIPSING_BINARY, "tess_eb_primary", "x", "eclipsing_binary")
        eb.observed_fraction = 1.0
        labels = patch_labels([eb])
        assert labels["patch_label"].max() == 0
        assert labels["patch_kind"][12] == config.KIND_ECLIPSING_BINARY

    def test_planet_wins_over_eclipse(self):
        eb = Event(101.0, 6.0, config.KIND_ECLIPSING_BINARY, "tess_eb_primary", "x", "eclipsing_binary")
        eb.observed_fraction = 1.0
        labels = patch_labels([eb, planet(100.0)])
        assert labels["patch_kind"][12] == config.KIND_REAL_PLANET
        assert labels["patch_label"][12] == config.LABEL_TRANSIT

    def test_edges(self):
        labels = patch_labels([planet(-2.0), planet(W + 2.0)])
        assert labels["patch_label"].max() == 0
        assert labels["patch_kind"][0] == config.KIND_REAL_PLANET


class TestMeasure:
    def test_snr_of_box_dip(self):
        rng = np.random.default_rng(0)
        flux = rng.normal(0, 1, W).astype(np.float32)
        flux[np.abs(np.arange(W) + 0.5 - 500.0) < 4] -= 3.0
        snr, frac, depth = measure_event(flux, np.ones(W, bool), 500.0, 8.0)
        assert frac == 1.0
        assert depth == pytest.approx(3.0, abs=0.8)
        assert snr == pytest.approx(3.0 * np.sqrt(8), abs=2.5)

    def test_unobserved(self):
        observed = np.ones(W, bool)
        observed[490:510] = False
        assert measure_event(np.zeros(W, np.float32), observed, 500.0, 8.0) == (0.0, 0.0, 0.0)

    def test_partial_at_edge(self):
        _, frac, _ = measure_event(np.zeros(W, np.float32), np.ones(W, bool), 1.0, 8.0)
        assert frac == pytest.approx(5 / 8)


class TestInjection:
    def test_apply_model_multiplies_relative_flux(self):
        s = 1e-3
        flux = np.zeros(W, np.float32)
        model = np.ones(W)
        model[10] = 0.99
        observed = np.ones(W, bool)
        observed[11] = False
        out = apply_model(flux, observed, model, s)
        assert out[10] == pytest.approx(-10.0)
        assert out[11] == 0.0

    def test_batman_reaches_target_combined_snr(self):
        pytest.importorskip("batman")
        rng = np.random.default_rng(3)
        observed = np.ones(W, bool)
        achieved = []
        for _ in range(20):
            inj = batman_injection(rng, observed, 1e-3, 1.0, 5800.0, (8.0, 8.0))
            assert inj.params["inj_period_d"] >= config.INJ_PERIOD_RANGE_D[0]
            assert inj.events and all(e.kind_code == config.KIND_INJECTED_BATMAN for e in inj.events)
            assert inj.params["inj_snr"] <= inj.params["inj_snr_total"] + 1e-6
            achieved.append(inj.params["inj_snr_total"])
        assert np.median(achieved) == pytest.approx(8.0, rel=0.35)

    def test_many_transits_are_individually_faint(self):
        pytest.importorskip("batman")
        rng = np.random.default_rng(5)
        observed = np.ones(W, bool)
        faint = 0
        for _ in range(200):
            inj = batman_injection(rng, observed, 1e-3, 1.0, 5800.0, (4.0, 4.0))
            if inj.params["inj_period_d"] < 3.0 and inj.params["inj_rp_rs"] > config.INJ_RP_RS_RANGE[0]:
                assert inj.params["inj_snr"] < 2.5
                faint += 1
        assert faint > 0

    def test_transplant_never_deepens(self):
        dt = np.arange(-6, 6.01, 1 / 6)
        template = {"dt_h": dt, "flux": np.where(np.abs(dt) < 1.5, 0.99, 1.0), "period_d": 3.0, "duration_h": 3.0,
                    "template_id": "t"}
        inj = transplant_injection(np.random.default_rng(1), np.ones(W, bool), 1e-4, template, (1000.0, 1000.0))
        assert inj.params["inj_depth_scale"] == 1.0
        assert inj.model.min() == pytest.approx(0.99)
        assert len(inj.events) >= 8


class TestStacking:
    def window(self, snrs, ref="TOI-1.01 PC", period=3.0, start=0, segment="seg"):
        evs = [{"center_slot": 50.0 * (i + 1), "duration_slots": 6.0, "kind": "toi", "ref": ref, "cls": "planet",
                "period_d": period, "snr": x, "observed_fraction": 1.0, "depth_sigma": 0.0}
               for i, x in enumerate(snrs)]
        return {"star_key": "tic:1", "segment_id": segment, "source": "ffi", "start_offset": start, "events": evs}

    def test_faint_repeats_stack_and_noise_does_not(self):
        from finetune_dataset.choose_samples import window_features
        frame = pd.DataFrame([self.window([2.0] * 9), self.window([1.0, -1.0] * 8)])
        feats = window_features(frame)
        assert feats["best_planet_snr"].iloc[0] == pytest.approx(2.0)
        assert feats["combined_planet_snr"].iloc[0] == pytest.approx(6.0)
        assert feats["combined_planet_snr"].iloc[1] == pytest.approx(0.0)

    def test_overlapping_windows_count_a_transit_once(self):
        from finetune_dataset.choose_samples import planet_fold_snr
        a = self.window([3.0, 3.0], start=0)
        b = self.window([3.0], start=50)          # second window starts on the first transit
        b["events"][0]["center_slot"] = 50.0      # absolute slot 100: the same transit as a's second
        fold = planet_fold_snr(pd.DataFrame([a, b]))
        assert fold[("tic:1", "TOI-1.01 PC")] == pytest.approx(6.0 / np.sqrt(2))

    def test_unverified_planets_become_unknown(self):
        from finetune_dataset.choose_samples import mark_unverified, planet_fold_snr
        frame = pd.DataFrame([self.window([1.0, 1.5]), self.window([8.0, 9.0], ref="TOI-2.01 PC")])
        frame.loc[1, "star_key"] = "tic:2"
        marked = mark_unverified(frame, planet_fold_snr(frame))
        assert {e["cls"] for e in marked["events"].iloc[0]} == {"unknown"}
        assert {e["cls"] for e in marked["events"].iloc[1]} == {"planet"}


class TestTemplates:
    def test_stacked_box_becomes_template(self):
        rng = np.random.default_rng(2)
        cutouts = []
        for k in range(6):
            n = 600
            centers = np.arange(n) + 0.5
            flux_z = rng.normal(0, 1, n).astype(np.float32)
            center = 300.0 + k / 6.0
            flux_z[np.abs(centers - center) < 3] -= 20.0
            dt, rel = cutout(flux_z, np.ones(n, bool), 1e-4, center, 6.0)
            cutouts.append({"dt_h": dt, "rel": rel})
        template = build_template(pd.DataFrame(cutouts), 5.0, 3.0)
        assert template is not None
        assert template["depth_ppm"] == pytest.approx(2000, rel=0.3)
        outside = np.abs(template["dt_h"]) > 1.5 + config.TEMPLATE_EDGE_HOURS
        assert np.all(template["flux"][outside] == 1.0)

    def test_noise_is_rejected(self):
        rng = np.random.default_rng(4)
        cutouts = []
        for _ in range(5):
            flux_z = rng.normal(0, 1, 600).astype(np.float32)
            dt, rel = cutout(flux_z, np.ones(600, bool), 1e-4, 300.0, 6.0)
            cutouts.append({"dt_h": dt, "rel": rel})
        assert build_template(pd.DataFrame(cutouts), 5.0, 3.0) is None


class TestPools:
    def test_pools(self):
        assert pool_for("tic:1", "train", {}) == config.POOL_TRAIN
        assert pool_for("tic:1", "search_holdout", {}) == config.POOL_UNUSED
        assert pool_for("tic:9", "list_excluded", {"tic:9": "real_validation"}) == config.POOL_LIST_VAL
        assert pool_for("tic:9", "list_excluded", {"tic:9": "real_test"}) == config.POOL_LIST_TEST

    def test_scoreboard_halves_are_balanced(self):
        halves = pd.Series([scoreboard_half(f"tic:{i}") for i in range(4000)]).value_counts(normalize=True)
        assert abs(halves["val"] - 0.5) < 0.03

    @pytest.mark.parametrize("kind,ref,cls", [
        ("toi", "TOI-5.01 PC", "planet"), ("toi", "TOI-5.01 FP", "false_positive"), ("toi", "TOI-5.01 nan", "unknown"),
        ("kepler_tce", "KIC 1 TCE 1 PC", "planet"), ("kepler_tce", "KIC 1 TCE 2 AFP", "false_positive"),
        ("kepler_tce", "KIC 1 TCE 3 UNK", "unknown"), ("tess_eb_secondary", "TIC 1 signal 1", "eclipsing_binary"),
        ("hawthorn_duotransit", "Hawthorn2024 #3", "planet"),
    ])
    def test_classify(self, kind, ref, cls):
        assert classify(kind, ref) == cls

    def test_long_period(self):
        assert is_long_period(None) and is_long_period(float("nan")) and is_long_period(25.0)
        assert not is_long_period(5.0)


def test_astronet_split_procedure():
    """Same shuffle and cut as Shallue & Vanderburg's generate_input_records.py."""
    from finetune_dataset.benchmarks import astronet_dr24

    class Catalogs:
        kic_to_tic = {}
        dr24_tce = pd.DataFrame({
            "kepid": np.arange(100), "tce_plnt_num": 1, "av_training_set": ["PC", "AFP", "NTP", "UNK"] * 25,
            "tce_period": 10.0, "tce_time0bk": 140.0, "tce_duration": 3.0, "tce_depth": 500.0,
        })

    test = astronet_dr24(Catalogs)
    labeled = Catalogs.dr24_tce[Catalogs.dr24_tce["av_training_set"] != "UNK"]
    np.random.seed(123)
    expected = labeled.iloc[np.random.permutation(len(labeled))].iloc[int(0.9 * len(labeled)):]
    assert list(test["kic_id"]) == list(expected["kepid"])
    assert test["epoch_btjd"].iloc[0] == pytest.approx(140.0 - 2167.0)


def test_card_headers_are_valid_yaml():
    yaml = pytest.importorskip("yaml")
    from finetune_dataset.cards import yaml_header

    header = yaml_header("lc-x: a title, with: colons", ["train", "validation"])
    meta = yaml.safe_load(header.split("---")[1])
    assert meta["pretty_name"] == "lc-x: a title, with: colons"
    assert [d["split"] for d in meta["configs"][0]["data_files"]] == ["train", "validation"]


class TestV3Injection:
    def test_eccentric_duration_reduces_to_circular(self):
        from finetune_dataset.injection import duration_days, duration_days_ecc
        assert duration_days_ecc(10.0, 20.0, 0.05, 0.3, 0.0, 90.0) == pytest.approx(duration_days(10.0, 20.0, 0.05, 0.3))

    def test_v3_reaches_target_and_records_physics(self):
        pytest.importorskip("batman")
        from finetune_dataset.injection import batman_injection_v3
        rng = np.random.default_rng(11)
        observed = np.ones(W, bool)
        ratios, eccentric = [], 0
        for _ in range(30):
            inj = batman_injection_v3(rng, observed, 1e-3, 1.0, 5800.0, (8.0, 8.0))
            if inj.params["inj_rp_rs"] < config.INJ_RP_RS_RANGE[1] - 1e-6:
                ratios.append(inj.params["inj_snr_total"] / inj.params["inj_snr_target"])
            eccentric += inj.params["inj_ecc"] > 0
            assert inj.params["inj_ld_source"] == "kipping2013"
        assert np.median(ratios) == pytest.approx(1.0, abs=0.03)
        assert 0 < eccentric < 30

    def test_limb_darkening_lookup_uses_nearest_grid_point(self):
        import pandas as pd
        from finetune_dataset.limb_darkening import coefficients, surface_gravity
        table = pd.DataFrame({"teff": [5000.0, 6000.0], "logg": [4.5, 4.5], "u1": [0.4, 0.3], "u2": [0.2, 0.25]})
        assert coefficients(table, 5900.0, 4.4) == (0.3, 0.25)
        assert surface_gravity(1.0, 1.0) == pytest.approx(4.438)
