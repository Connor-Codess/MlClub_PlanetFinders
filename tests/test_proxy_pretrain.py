"""Offline tests for experiments/proxy_pretrain and the shared Kepler degradation.

Run: python -m pytest tests/test_proxy_pretrain.py -q
"""

import numpy as np
import pandas as pd
import pytest

from experiments.finetune_mix import config as mix
from experiments.finetune_mix.augment import degrade_kepler, positive_caps
from experiments.proxy_pretrain import config


def test_variant_shares_sum_to_one():
    for name, shares in config.VARIANTS.items():
        assert sum(shares.values()) == pytest.approx(1.0), name
        assert set(shares) <= set(config.FILES)


def test_choose_respects_shares_and_exclusions():
    from experiments.proxy_pretrain.pack import choose
    rows = []
    for source, files in config.FILES.items():
        for f in files:
            for r in range(400):
                rows.append({"file": f, "row": r, "star_key": f"{source}:{f}:{r}", "source": source,
                             "start_offset": 0, "w_default": 1.0})
    index = pd.DataFrame(rows)
    excluded = {"ffi:" + config.FFI_FILES[0] + ":0"}
    chosen = choose(index, "pt_mix", 1000, excluded, np.random.default_rng(0))
    assert len(chosen) == 1000
    assert chosen["source"].value_counts().to_dict() == {"ffi": 500, "2min": 250, "kepler": 250}
    assert not chosen["star_key"].isin(excluded).any()
    assert not chosen.duplicated(["file", "row"]).any()


def test_degradation_matches_target_noise():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(0, 1, (n, 1, mix.WINDOW_SLOTS)).astype(np.float32)
    observed = np.ones((n, mix.WINDOW_SLOTS), np.uint8)
    sources = np.array(["kepler"] * n)
    factors = degrade_kepler(x, observed, sources, np.full(n, 100.0), np.array([400.0]), rng, share=1.0)
    assert np.allclose(factors, 4.0)
    # flux_z is in units of the new, larger noise: its point-to-point scatter stays near 1
    assert np.std(x) == pytest.approx(1.0, rel=0.05)


def test_positive_rows_keep_their_snr_floor():
    meta = pd.DataFrame({"category": ["injected_batman", "real_planet_long", "negative_random"],
                         "inj_snr_total": [12.0, np.nan, np.nan], "stacked_snr": [np.nan, 4.5, np.nan]})
    caps = positive_caps(meta)
    assert caps[0] == pytest.approx(4.0) and caps[1] == pytest.approx(1.5) and np.isinf(caps[2])


def test_patch_mask_ratio_and_spans():
    torch = pytest.importorskip("torch")  # noqa: F841
    from experiments.proxy_pretrain.pretrain import patch_mask
    mask = patch_mask(np.random.default_rng(1), 200)
    assert mask.shape == (200, mix.N_PATCHES)
    assert mask.mean() == pytest.approx(config.MASK_RATIO, abs=0.04)


def test_pretrained_encoder_loads_into_detector(tmp_path):
    torch = pytest.importorskip("torch")
    from experiments.finetune_mix.model import Detector, load_encoder
    from experiments.proxy_pretrain.pretrain import Pretrainer
    model = Pretrainer()
    torch.save(model.state_dict(), tmp_path / "pt.pt")
    detector = Detector()
    keys = load_encoder(detector, tmp_path / "pt.pt")
    assert keys and not any(k.startswith(("head.", "recon.")) for k in keys)
    assert torch.equal(detector.position, model.position)
