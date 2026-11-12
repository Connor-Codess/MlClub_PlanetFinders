"""PatchTST-style encoder with masked-patch pretraining.

Input (from dataset.collate):
    values [B, L, 4]  channels: 0 flux, 1 detrended flux, 2 err/sigma, 3 centroid shift
    mask   [B, L]     True where the 30-min bin has data
    side   [B, 10]    per-window scale / star features (preprocess._scale)

Tokens: [CLS] [SIDE] patch_1 ... patch_P   (P = L / patch = 84)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CHANNELS = 4
TARGET_CHANNELS = (0, 1, 3)          # err/sigma (2) is input only


@dataclass
class ModelConfig:
    seq_len: int = 1344
    patch: int = 16
    n_side: int = 10
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1

    @property
    def n_patches(self):
        return self.seq_len // self.patch

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- patching

def patchify(values, mask, patch):
    """values [B,L,C], mask [B,L] -> x [B,P,patch,C], m [B,P,patch]."""
    B, L, C = values.shape
    return values.reshape(B, L // patch, patch, C), mask.reshape(B, L // patch, patch)


def unpatchify(x, m):
    B, P, p, C = x.shape
    return x.reshape(B, P * p, C), m.reshape(B, P * p)


# ---------------------------------------------------------------- masking

def make_patch_mask(bin_mask: np.ndarray, patch: int, ratio: float = 0.5,
                    span_frac: float = 0.5, span_range=(2, 6), rng=None) -> np.ndarray:
    """Choose patches to hide. bin_mask [B,L] bool -> hidden [B,P] bool.

    Only patches with at least one filled bin are eligible. Per window, with
    probability span_frac hide contiguous runs of span_range patches, otherwise
    independent random patches, until round(ratio * eligible) are hidden.
    """
    rng = rng or np.random.default_rng()
    B, L = bin_mask.shape
    P = L // patch
    eligible = bin_mask.reshape(B, P, patch).any(-1)
    hidden = np.zeros((B, P), bool)
    for i in range(B):
        idx = np.flatnonzero(eligible[i])
        n = int(round(ratio * len(idx)))
        if n == 0:
            continue
        if rng.random() < span_frac:
            h = np.zeros(P, bool)
            tries = 0
            while h[idx].sum() < n and tries < 200:
                ln = int(rng.integers(span_range[0], span_range[1] + 1))
                s = int(rng.integers(0, P))
                h[s:s + ln] = True
                tries += 1
            h &= eligible[i]
            on = np.flatnonzero(h)
            if len(on) > n:                    # trim overshoot from the ends of the last span
                h[on[n:]] = False
            hidden[i] = h
        else:
            hidden[i, rng.choice(idx, n, replace=False)] = True
    return hidden


# ---------------------------------------------------------------- model

class PatchEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.patch_embed = nn.Linear(cfg.patch * (N_CHANNELS + 1), d)
        self.side_embed = nn.Sequential(nn.Linear(cfg.n_side, d), nn.GELU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_patches + 2, d))
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, cfg.d_ff, cfg.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        for p in (self.cls, self.mask_token, self.pos):
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, values, mask, side, hidden=None):
        """Returns tokens [B, 2+P, d]: index 0 = CLS, 1 = SIDE, 2.. = patches.

        hidden [B,P] bool: patches replaced by the mask token (None = none hidden).
        """
        x, m = patchify(values * mask[..., None], mask, self.cfg.patch)
        B, P = m.shape[:2]
        tok = self.patch_embed(torch.cat([x.reshape(B, P, -1), m.float()], -1))
        if hidden is not None:
            tok = torch.where(hidden[..., None], self.mask_token.expand(B, P, -1), tok)
        tok = torch.cat([self.cls.expand(B, -1, -1), self.side_embed(side)[:, None], tok], 1)
        return self.norm(self.blocks(tok + self.pos))

    def pooled(self, values, mask, side):
        """Fine-tuning / probe features: [CLS, mean over patches with data] -> [B, 2d]."""
        t = self.forward(values, mask, side)
        has = patchify(values, mask, self.cfg.patch)[1].any(-1).float()[..., None]
        mean = (t[:, 2:] * has).sum(1) / has.sum(1).clamp(min=1)
        return torch.cat([t[:, 0], mean], -1)


class MaskedPatchModel(nn.Module):
    def __init__(self, cfg: ModelConfig, channel_weights=(1.0, 1.0, 0.5)):
        super().__init__()
        self.cfg = cfg
        self.encoder = PatchEncoder(cfg)
        self.head = nn.Linear(cfg.d_model, cfg.patch * len(TARGET_CHANNELS))
        self.register_buffer("cw", torch.tensor(channel_weights, dtype=torch.float32))

    def forward(self, values, mask, side, hidden):
        """Returns (loss, parts) where parts holds per-channel and per-sample MSE.

        Loss = weighted mean over channels of the MSE on filled bins inside
        hidden patches. The centroid channel only counts where has_c (side[:,2]).
        """
        cfg = self.cfg
        t = self.encoder(values, mask, side, hidden)[:, 2:]
        B, P = hidden.shape
        pred = self.head(t).float().reshape(B, P, cfg.patch, len(TARGET_CHANNELS))   # loss in fp32 under autocast
        x, m = patchify(values, mask, cfg.patch)
        tgt = x[..., list(TARGET_CHANNELS)]
        w = (m & hidden[..., None]).float()                         # [B,P,p]
        wc = w[..., None].expand(-1, -1, -1, len(TARGET_CHANNELS)).clone()
        wc[..., 2] *= side[:, 2].view(B, 1, 1)                      # no centroid target if has_c = 0
        se = (pred - tgt) ** 2 * wc
        per_ch = se.sum((0, 1, 2)) / wc.sum((0, 1, 2)).clamp(min=1)
        loss = (per_ch * self.cw).sum() / self.cw.sum()
        per_sample = se.sum((1, 2, 3)) / wc.sum((1, 2, 3)).clamp(min=1)
        return loss, dict(per_channel=per_ch.detach(), per_sample=per_sample.detach(), pred=pred)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
