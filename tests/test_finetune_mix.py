"""Offline tests for experiments/finetune_mix: arm quotas, row selection, labels, scoring, baseline, model.

Run: python -m pytest tests/test_finetune_mix.py -q
"""

import numpy as np
import pandas as pd
import pytest

from experiments.finetune_mix import config
from experiments.finetune_mix.data import target_mask, train_label
from experiments.finetune_mix.evaluate import average_precision, box_filter, score, window_scores
from experiments.finetune_mix.select import arm_quotas, select_arm

SHARES = {"real_planet_short": 0.133, "real_planet_long": 0.027, "injected_batman": 0.415,
          "injected_transplant": 0.15, "negative_random": 0.175, "negative_eb": 0.10}


def fake_meta(n_per=1500, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in config.CATEGORIES:
        for _ in range(n_per):
            rows.append({"category": c, "source": rng.choice(["ffi", "2min", "kepler"]),
                         "single_transit_crop": bool(rng.uniform() < 0.2), "inj_snr": rng.uniform(1, 30),
                         "inj_snr_total": rng.uniform(2, 60), "best_transit_snr": rng.uniform(0, 20),
                         "path": "data/train/part-0.parquet", "row": len(rows), "sample_id": f"s{len(rows)}"})
    return pd.DataFrame(rows)


class TestQuotas:
    def test_every_arm_sums_to_its_rows(self):
        for name, spec in {**config.WAVE1_ARMS, **config.WAVE2_ARMS}.items():
            q = arm_quotas(spec, SHARES)
            assert sum(q.values()) == spec.get("rows", config.ROWS_PER_ARM), name

    def test_moves_empty_the_source(self):
        q = arm_quotas(config.WAVE1_ARMS["no_real"], SHARES)
        assert q["real_planet_short"] == 0 and q["real_planet_long"] == 0
        base = arm_quotas({}, SHARES)
        assert q["negative_random"] == base["negative_random"]
        assert q["injected_batman"] > base["injected_batman"]


class TestWave2:
    def test_real_share_sets_real_rows(self):
        for share in config.WAVE2_REAL_SHARES:
            q = arm_quotas({"real_share": share, "rows": 100_000}, SHARES)
            real = q["real_planet_short"] + q["real_planet_long"]
            assert real == pytest.approx(share * 100_000, abs=2)
            assert sum(q.values()) == 100_000

    def test_twenty_six_arms_with_known_inits(self):
        assert len(config.WAVE2_ARMS) == 26
        assert config.REFERENCE_ARM in config.WAVE2_ARMS
        assert {s.get("init") for s in config.WAVE2_ARMS.values()} == {None, "pt_mix", "pt_ffi", "pt_tess"}

    def test_tess_hosts_keep_kepler_real_planets(self):
        meta = fake_meta()
        meta["inj_deep"] = False
        rng = np.random.default_rng(3)
        order = {c: rng.permutation(np.flatnonzero(meta["category"].to_numpy() == c)) for c in config.CATEGORIES}
        spec = {**config.WAVE2_ARMS["pt_mix-r16-nodeep-tess"], "rows": 1000}
        chosen = meta.set_index("sample_id").loc[select_arm("x", spec, meta, order, SHARES)["sample_id"]]
        hosts = chosen[chosen["category"].isin([*config.INJECTED, "negative_random"])]
        assert set(hosts["source"]) <= set(config.TESS_SOURCES)
        assert "kepler" in set(chosen[chosen["category"].isin(config.REAL)]["source"])


class TestSelection:
    def setup_method(self):
        self.meta = fake_meta()
        rng = np.random.default_rng(1)
        self.order = {c: rng.permutation(np.flatnonzero(self.meta["category"].to_numpy() == c))
                      for c in config.CATEGORIES}

    def pick(self, spec):
        return select_arm("x", {**spec, "rows": 1000}, self.meta, self.order, SHARES)

    def test_filters_and_sources_hold(self):
        chosen = self.meta.loc[self.meta["sample_id"].isin(self.pick(config.WAVE1_ARMS["v1_rule"])["sample_id"])]
        real = chosen[chosen["category"].isin(config.REAL)]
        assert (real["best_transit_snr"] >= 4).all()
        tess = self.meta.loc[self.meta["sample_id"].isin(self.pick(config.WAVE1_ARMS["tess_only"])["sample_id"])]
        assert set(tess["source"]) <= set(config.TESS_SOURCES)

    def test_smaller_arm_is_nested(self):
        small = select_arm("a", {"rows": 500}, self.meta, self.order, SHARES)
        large = select_arm("b", {"rows": 1000}, self.meta, self.order, SHARES)
        assert set(small["sample_id"]) <= set(large["sample_id"])

    def test_shortage_raises(self):
        with pytest.raises(RuntimeError):
            select_arm("x", {"rows": 1_000_000}, self.meta, self.order, SHARES)


def test_read_meta_paths_are_relative_to_root(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from experiments.finetune_mix.data import META_COLUMNS, read_meta
    shard = tmp_path / "data" / "train"
    shard.mkdir(parents=True)
    columns = {c: [None] for c in META_COLUMNS}
    columns["events"] = [[{"snr": 5.0, "is_planet": True, "visible": True}]]
    pq.write_table(pa.table(columns), shard / "part-0.parquet")
    meta = read_meta([(shard / "part-0.parquet").resolve()], tmp_path)
    assert meta["path"].iloc[0] == "data/train/part-0.parquet"
    assert meta["best_transit_snr"].iloc[0] == 5.0


class TestLabels:
    def test_covered_neighbours_are_ignored(self):
        label = np.zeros((1, 162), np.uint8)
        overlap = np.zeros((1, 162), np.uint8)
        label[0, 10] = 1
        overlap[0, 9:12] = 100
        out = train_label(label, overlap)
        assert out[0, 10] == 1 and out[0, 9] == 255 and out[0, 11] == 255 and out[0, 12] == 0

    def test_target_is_dilated(self):
        label = np.zeros((1, 162), np.uint8)
        label[0, 0] = 1
        assert target_mask(label)[0, :3].tolist() == [1, 1, 0]


class TestScoring:
    def test_threshold_matches_false_alarm_rate_and_recall(self):
        rng = np.random.default_rng(0)
        n_neg, n_inj = 2000, 500
        patch = rng.uniform(0, 0.5, (n_neg + n_inj, 162)).astype(np.float32)
        target = np.zeros_like(patch, dtype=np.uint8)
        target[n_neg:, 50] = 1
        patch[n_neg:, 50] = 0.9
        label = np.zeros_like(target)
        meta = pd.DataFrame({"category": ["negative_random"] * n_neg + ["injected_batman"] * n_inj,
                             "inj_snr_total": [np.nan] * n_neg + [8.0] * n_inj,
                             "inj_snr": [np.nan] * n_neg + [3.0] * n_inj,
                             "inj_period_d": [np.nan] * n_neg + [5.0] * n_inj,
                             "stacked_snr": [np.nan] * (n_neg + n_inj)})
        any_score, on_target = window_scores(patch, target, label)
        result = score(meta, any_score, on_target, average_precision(patch, target))
        point = result["operating_points"]["0.01"]
        assert point["random_false_alarm"] == pytest.approx(0.01, abs=0.003)
        assert point["injected_recall"] == 1.0 and point["faint_recall"] == 1.0
        assert result["patch_average_precision"] == pytest.approx(1.0)

    def test_box_filter_finds_a_dip(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1, (2, 4, config.WINDOW_SLOTS)).astype(np.float16)
        x[1, 0, 400:406] -= 5.0
        observed = np.ones((2, config.WINDOW_SLOTS), np.uint8)
        scores = box_filter(x, observed)
        assert scores[1].argmax() == 400 // 8 or scores[1].argmax() == 401 // 8
        assert scores[1].max() > scores[0].max() + 5


def test_model_shape():
    torch = pytest.importorskip("torch")
    from experiments.finetune_mix.model import Detector
    out = Detector()(torch.zeros(2, len(config.INPUT_CHANNELS) + 1, config.WINDOW_SLOTS))
    assert out.shape == (2, config.N_PATCHES)


def _transit_rows(category: str, n: int, offset: float, rng: np.random.Generator) -> list[dict]:
    rows = []
    slots = np.arange(config.WINDOW_SLOTS) + 0.5
    for _ in range(n):
        centre, duration, depth = float(rng.uniform(200, 1000)), 6.0, float(rng.uniform(1.5, 3.0))
        true_centre = centre + offset
        flux = rng.normal(0, 1, config.WINDOW_SLOTS)
        flux[np.abs(slots - true_centre) < duration / 2] -= depth
        rows.append({"category": category, "source": "ffi", "flux_z": flux.astype("<f2").tobytes(),
                     "flags": np.ones(config.WINDOW_SLOTS, np.uint8).tobytes(),
                     "events": [{"is_planet": True, "visible": True, "snr": 10.0, "center_slot": centre,
                                 "duration_slots": duration}]})
    return rows


class TestRealism:
    def test_recentring_removes_an_ephemeris_offset(self):
        pytest.importorskip("sklearn")
        from experiments.finetune_mix.realism import classify, cutouts

        rng = np.random.default_rng(0)
        table = pd.DataFrame(_transit_rows("real", 300, 2.0, rng) + _transit_rows("batman", 300, 0.0, rng))
        result = classify(cutouts(table), "real", "batman", np.random.default_rng(1))
        assert result["pairs"] >= 280
        assert result["shape_auc"] > 0.8
        assert result["recentred_auc"] < 0.65
        assert result["wings_auc"] < 0.65
        assert len(result["mean_shape"]["real"]) == 40
