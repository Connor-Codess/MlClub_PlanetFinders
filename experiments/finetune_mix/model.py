"""The small per-patch transit detector used to compare data mixes.

What it is
    Input: (batch, 5, 1296) = flux_z, cen_x, cen_y, bkg_z and the observed mask, one
    27-day window on the 30-minute grid. Output: (batch, 162) logits, one per 4-hour patch.

        two 1-D convolutions over slots (local transit shape, 3.5 h receptive field)
        -> a strided convolution that turns each 8-slot patch into one token
        -> a 4-layer transformer encoder over the 162 tokens (lets the model compare
           and stack repeated transits anywhere in the window)
        -> a linear head per token

    About 0.35M parameters, so it trains on a 4-CPU runner. It has the same interface as
    the planned pretrained model (patch tokens in, one logit per patch out), so the data
    comparisons are made on the right kind of model, only much smaller and with no
    pretraining.

How to use
    model = Detector(); logits = model(torch.randn(8, 5, 1296))   # (8, 162)
    load_encoder(model, "pretrained.pt")    # start from a proxy_pretrain checkpoint (head stays new)
"""

from __future__ import annotations

import torch
from torch import nn

from . import config


class Detector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        c, d = config.STEM_CHANNELS, config.D_MODEL
        in_channels = len(config.INPUT_CHANNELS) + 1
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(c, c, kernel_size=7, padding=3), nn.GELU(),
        )
        self.patchify = nn.Conv1d(c, d, kernel_size=config.PATCH_SLOTS, stride=config.PATCH_SLOTS)
        self.position = nn.Parameter(torch.zeros(1, config.N_PATCHES, d))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=config.N_HEADS, dim_feedforward=config.FF_DIM,
                                           dropout=config.DROPOUT, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.N_LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        nn.init.normal_(self.position, std=0.02)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, 162, d) patch features before the head."""
        tokens = self.patchify(self.stem(x)).transpose(1, 2) + self.position
        return self.norm(self.encoder(tokens))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x)).squeeze(-1)


ENCODER_PREFIXES = ("stem.", "patchify.", "position", "encoder.", "norm.")


def load_encoder(model: Detector, path) -> list[str]:
    """Load pretrained encoder weights (everything but the head); returns the keys loaded."""
    state = torch.load(path, map_location="cpu")
    keep = {k: v for k, v in state.items() if k.startswith(ENCODER_PREFIXES)}
    missing, unexpected = model.load_state_dict(keep, strict=False)
    if unexpected or any(not k.startswith("head.") for k in missing):
        raise RuntimeError(f"encoder checkpoint does not match: missing {missing}, unexpected {unexpected}")
    return sorted(keep)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
