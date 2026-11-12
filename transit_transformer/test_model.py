"""Quick checks for model.py:  python3 test_model.py"""

import numpy as np
import torch

from model import ModelConfig, MaskedPatchModel, make_patch_mask, patchify, unpatchify, count_params


def batch(B=4, L=1344, seed=0):
    g = torch.Generator().manual_seed(seed)
    values = torch.randn(B, L, 4, generator=g)
    mask = torch.rand(B, L, generator=g) > 0.2
    mask[:, 160:320] = False                       # 10 empty patches
    values[~mask] = 0
    side = torch.randn(B, 10, generator=g)
    side[:, 2] = 1.0
    return values, mask, side


def test_patchify_roundtrip():
    v, m, _ = batch()
    x, pm = patchify(v, m, 16)
    v2, m2 = unpatchify(x, pm)
    assert torch.equal(v, v2) and torch.equal(m, m2)


def test_mask_ratio_and_eligibility():
    _, m, _ = batch(B=64)
    rng = np.random.default_rng(0)
    h = make_patch_mask(m.numpy(), 16, 0.5, rng=rng)
    elig = m.numpy().reshape(64, 84, 16).any(-1)
    assert not (h & ~elig).any(), "empty patches must never be hidden"
    frac = h.sum(1) / elig.sum(1)
    assert np.allclose(frac, 0.5, atol=0.02), frac


def test_loss_ignores_visible_and_unfilled():
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0)
    model = MaskedPatchModel(cfg).eval()
    v, m, s = batch()
    h = torch.from_numpy(make_patch_mask(m.numpy(), 16, 0.5, rng=np.random.default_rng(1)))
    loss, _ = model(v, m, s, h)
    # Change values in visible patches' targets? They feed the encoder, so instead check that
    # changing *unfilled* bins (zeroed by the mask) does not change anything.
    v2 = v.clone()
    v2[~m] = 123.0
    loss2, _ = model(v2, m, s, h)
    assert torch.allclose(loss, loss2), (loss, loss2)
    # Hidden patch inputs must not leak: change values inside hidden patches' *inputs*
    # while keeping targets - the prediction must be unchanged.
    hb = h.repeat_interleave(16, 1)
    _, p1 = model(v, m, s, h)
    v3 = v.clone()
    v3[hb] = v3[hb] * 0 + 7.0
    _, p3 = model(v3, m, s, h)
    assert torch.allclose(p1["pred"], p3["pred"], atol=1e-5), "hidden patch content leaked"


def test_param_count():
    n = count_params(MaskedPatchModel(ModelConfig()))
    print(f"default model: {n / 1e6:.2f}M parameters")
    assert 5e6 < n < 8e6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
