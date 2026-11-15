"""Offline tests for pretrain_dataset.

Covers star identity and splits, segment construction on synthetic light curves, the
float16 round trip, default windows, event prediction and window weights. None of the
tests touch the network.

Run: python -m pytest tests -q
"""

import hashlib

import numpy as np
import pandas as pd
import pytest

from pretrain_dataset import config
from pretrain_dataset.build_index import add_weights, explode_windows
from pretrain_dataset.catalogs import Ephemeris, predict_events
from pretrain_dataset.segments import Dropped, build_segment, decode, default_windows, quality_tier
from pretrain_dataset.star_identity import (
    IdentityResolver,
    bucket_split,
    build_exclusions,
    hash_bucket,
    kic_key,
    tic_key,
)


def synthetic_ffi_row(n=1300, sigma=1e-3, seed=0, t_start=1500.0, gap=None):
    rng = np.random.default_rng(seed)
    time = t_start + (np.arange(n) + 0.5) * config.SLOT_DAYS
    if gap is not None:
        keep = (time < gap[0]) | (time > gap[1])
        time = time[keep]
    flux = 1.0 + rng.normal(0.0, sigma, time.size)
    return {
        "time": time,
        "flux": flux,
        "flux_err": np.full(time.size, sigma),
        "centr_x": 100.0 + rng.normal(0.0, 0.01, time.size),
        "centr_y": 200.0 + rng.normal(0.0, 0.01, time.size),
        "bkg": 50.0 + rng.normal(0.0, 1.0, time.size),
        "n_points": np.full(time.size, 3.0),
    }


class TestStarIdentity:
    def test_keys(self):
        assert tic_key(123) == "tic:123"
        assert kic_key(757076) == "kic:757076"

    def test_hash_bucket_matches_definition(self):
        key = "tic:261136679"
        digest = hashlib.sha256(key.encode()).digest()
        assert hash_bucket(key) == int.from_bytes(digest[:8], "big") % 1000

    @pytest.mark.parametrize(
        "bucket,split",
        [(0, "train"), (849, "train"), (850, "validation"), (899, "validation"),
         (900, "search_holdout"), (949, "search_holdout"), (950, "scoreboard_hosts"), (999, "scoreboard_hosts")],
    )
    def test_bucket_split(self, bucket, split):
        assert bucket_split(bucket) == split

    def test_split_fractions(self):
        splits = pd.Series([bucket_split(hash_bucket(tic_key(i))) for i in range(20000)])
        share = splits.value_counts(normalize=True)
        assert abs(share["train"] - 0.85) < 0.01
        assert abs(share["validation"] - 0.05) < 0.01

    def test_kepler_star_uses_tic_when_matched(self):
        resolver = IdentityResolver({757076: 1234}, {})
        assert resolver.resolve(kic_id=757076).star_key == "tic:1234"
        assert resolver.resolve(kic_id=999).star_key == "kic:999"

    def test_excluded_star_is_list_excluded(self):
        resolver = IdentityResolver({}, {"tic:5": "hawthorn2024_duotransit_host"})
        assert resolver.resolve(tic_id=5).split == "list_excluded"

    def test_build_exclusions(self):
        toi = pd.DataFrame({
            "tic_id": [1, 2, 3, 4],
            "disposition": ["PC", "PC", "FP", "CP"],
            "period_d": [5.0, 45.0, 60.0, np.nan],
        })
        duo = pd.DataFrame({"tic_id": [10]})
        out = build_exclusions(toi, duo).set_index("star_key")
        assert set(out.index) == {"tic:2", "tic:4", "tic:10"}
        assert out.loc["tic:10", "reason"] == "hawthorn2024_duotransit_host"
        assert set(out["scoreboard_set"]) <= {"real_validation", "real_test"}


class TestSegments:
    def test_noise_and_round_trip(self):
        row = synthetic_ffi_row(sigma=1e-3)
        seg = build_segment(row, "ffi", 30)
        assert abs(seg.stats["sigma_p2p_ppm"] - 1000) < 100
        assert 0.8 < seg.stats["var_ratio"] < 1.2
        channels = decode(seg.channels)
        assert channels["flux_z"].size == seg.n_slots == row["time"].size
        flags = channels["flags"]
        assert (flags & config.FLAG_OBSERVED).sum() == seg.n_obs
        assert np.all(flags & config.FLAG_CENTROID_VALID)
        # one point per slot here, so the stored channel must reproduce the input flux
        sigma = seg.stats["sigma_p2p_ppm"] * 1e-6 * seg.stats["flux_median"]
        rebuilt = channels["flux_z"] * sigma + seg.stats["flux_median"]
        assert np.max(np.abs(rebuilt - row["flux"])) / sigma < 0.01

    def test_fill_from_point_counts(self):
        # sector 30 FFIs are 10-minute cadence: 3 points per slot is full
        seg = build_segment(synthetic_ffi_row(), "ffi", 30)
        fill = decode(seg.channels)["fill"]
        assert np.allclose(fill, 1.0, atol=1 / 255)

    def test_gap_flags(self):
        seg = build_segment(synthetic_ffi_row(gap=(1510.0, 1512.0)), "ffi", 30)
        flags = decode(seg.channels)["flags"]
        observed = (flags & config.FLAG_OBSERVED) > 0
        assert not observed.all()
        post_gap = (flags & config.FLAG_POST_GAP) > 0
        first_after = np.flatnonzero(observed & ~np.roll(observed, 1))[-1]
        assert post_gap[first_after]

    def test_spike_flag(self):
        row = synthetic_ffi_row()
        row["flux"][600] += 0.02
        seg = build_segment(row, "ffi", 30)
        flags = decode(seg.channels)["flags"]
        assert flags[600] & config.FLAG_SPIKE

    def test_kepler_time_shift(self):
        row = synthetic_ffi_row(t_start=200.0)
        for key in ("bkg", "n_points"):
            row.pop(key)
        seg = build_segment(row, "kepler", 5)
        t0 = config.GRID_EPOCH_BTJD + seg.slot0 * config.SLOT_DAYS
        assert abs(t0 - (200.0 + config.BKJD_TO_BTJD)) < config.SLOT_DAYS

    def test_drops(self):
        with pytest.raises(Dropped):
            build_segment(synthetic_ffi_row(n=100), "ffi", 30)
        with pytest.raises(Dropped):
            build_segment(synthetic_ffi_row(sigma=0.2), "ffi", 30)

    def test_default_windows(self):
        tess = np.ones(1296 + 400, bool)
        offsets, coverage = default_windows(tess, "ffi")
        assert offsets == [0, 400]
        assert coverage == [1.0, 1.0]
        assert default_windows(np.ones(1296 + 100, bool), "ffi")[0] == [0]
        kepler = np.ones(1296 * 3 + 700, bool)
        assert default_windows(kepler, "kepler")[0] == [0, 1296, 2592, 700 + 1296 * 2]

    def test_quality_tier(self):
        assert quality_tier(0.9, 1500) == 1
        assert quality_tier(0.6, 1500) == 2
        assert quality_tier(0.9, 20000) == 3


class TestEvents:
    def test_periodic_events_land_on_slots(self):
        slot0 = 1000
        t_start = config.GRID_EPOCH_BTJD + slot0 * config.SLOT_DAYS
        eph = Ephemeris("toi", "TOI-1.01", t0_btjd=t_start + 1.0, period_d=3.0, duration_h=3.0)
        events = predict_events([eph], slot0, 1296)
        centers = [e["center_slot"] for e in events]
        assert centers[0] == pytest.approx(48.0)
        assert np.diff(centers) == pytest.approx(144.0)
        assert all(e["duration_slots"] == pytest.approx(6.0) for e in events)

    def test_single_event_and_short_period(self):
        slot0 = 0
        t_start = config.GRID_EPOCH_BTJD
        single = Ephemeris("hawthorn", "x", t0_btjd=t_start + 10.0, period_d=None, duration_h=5.0)
        short = Ephemeris("tess_eb", "y", t0_btjd=t_start, period_d=0.2, duration_h=1.0)
        events = predict_events([single, short], slot0, 1296)
        assert len(events) == 1 and events[0]["kind"] == "hawthorn"


class TestWeights:
    def test_source_shares_and_normalization(self):
        segments = pd.DataFrame({
            "file": ["a"] * 6,
            "row": range(6),
            "segment_id": [f"s{i}" for i in range(6)],
            "star_key": [f"tic:{i}" for i in range(6)],
            "source": ["ffi", "ffi", "ffi", "ffi", "2min", "kepler"],
            "split": ["train"] * 6,
            "quality_tier": [1, 1, 1, 3, 1, 1],
            "var_ratio": [1.0] * 6,
            "window_offsets": [[0]] * 5 + [[0, 1296]],
            "window_coverage": [[1.0]] * 5 + [[1.0, 1.0]],
        })
        windows, _ = add_weights(explode_windows(segments))
        assert len(windows) == 7
        assert windows["w_default"].mean() == pytest.approx(1.0)
        share = windows.groupby("source")["w_source"].sum() / windows["w_source"].sum()
        assert share["ffi"] == pytest.approx(0.50)
        assert share["2min"] == pytest.approx(0.25)
        assert share["kepler"] == pytest.approx(0.25)
        assert windows.loc[windows["quality_tier"] == 3, "w_tier"].iloc[0] == pytest.approx(0.5)
