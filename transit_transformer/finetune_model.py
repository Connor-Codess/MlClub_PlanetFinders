"""Signal classifier: pretrained PatchEncoder over transit windows + folded-view CNN + scalars.

    encoder windows [B,W,L,4] -> PatchEncoder -> per window [CLS, mean of transit-patch tokens]
                              -> Linear -> attention pool over windows -------------------> 256
    global view [B,3,201] -> 1-D CNN -> 128
    local views [B,6,81]  -> 1-D CNN -> 128
    scalars [B,NS] -> MLP -> 64
    concat (576) -> MLP -> 4 logits (planet, candidate, false positive, no planet)

encoder_mode: "pretrained" | "random" (same architecture, no pretrained weights)
              | "frozen" (pretrained, never updated) | "none" (views + scalars only)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import ModelConfig, PatchEncoder

N_CLASSES = 4


def _conv(cin, cout):
    return nn.Sequential(nn.Conv1d(cin, cout, 5, padding=2), nn.GELU(),
                         nn.Conv1d(cout, cout, 5, padding=2), nn.GELU(), nn.MaxPool1d(2))


class ViewCNN(nn.Module):
    def __init__(self, cin, length, widths, out=128):
        super().__init__()
        layers, c, n = [], cin, length
        for w in widths:
            layers.append(_conv(c, w)); c, n = w, n // 2
        self.net = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(c * n, out), nn.GELU())

    def forward(self, x):
        return self.fc(self.net(x))


class SignalClassifier(nn.Module):
    def __init__(self, enc_cfg: ModelConfig, n_scal: int, encoder_mode="pretrained", dropout=0.2):
        super().__init__()
        self.mode = encoder_mode
        d = enc_cfg.d_model
        self.encoder = PatchEncoder(enc_cfg) if encoder_mode != "none" else None
        self.win_proj = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.LayerNorm(256))
        self.win_score = nn.Linear(256, 1)
        self.glob_cnn = ViewCNN(3, 201, [16, 32, 64, 128])
        self.loc_cnn = ViewCNN(6, 81, [32, 64])
        self.scal_mlp = nn.Sequential(nn.Linear(n_scal, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(576), nn.Dropout(dropout), nn.Linear(576, 256), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(256, N_CLASSES))

    def encoder_trainable(self, on: bool):
        if self.encoder is None:
            return
        on = on and self.mode != "frozen"
        for p in self.encoder.parameters():
            p.requires_grad_(on)

    def encode_windows(self, values, mask, side, tpatch, wmask):
        B, W = wmask.shape
        out = values.new_zeros(B, W, 256)
        if self.encoder is None:
            return values.new_zeros(B, 256)
        sel = wmask.reshape(-1)
        v = values.reshape(B * W, *values.shape[2:])[sel]
        m = mask.reshape(B * W, -1)[sel]
        s = side.reshape(B * W, -1)[sel]
        tp = tpatch.reshape(B * W, -1)[sel].float()
        frozen = not any(p.requires_grad for p in self.encoder.parameters())
        if frozen:
            self.encoder.eval()
            with torch.no_grad():
                t = self.encoder(v, m, s)
        else:
            t = self.encoder(v, m, s)
        tr = (t[:, 2:] * tp[..., None]).sum(1) / tp.sum(1, keepdim=True).clamp(min=1)
        feat = self.win_proj(torch.cat([t[:, 0], tr], -1).float())
        out = out.reshape(B * W, 256)
        out[sel] = feat.to(out.dtype)
        out = out.reshape(B, W, 256)
        score = self.win_score(out).squeeze(-1).masked_fill(~wmask, float("-inf"))
        a = torch.softmax(score, -1).nan_to_num(0.0)
        return (a[..., None] * out).sum(1)

    def forward(self, b):
        e = self.encode_windows(b["values"], b["mask"], b["side"], b["tpatch"], b["wmask"])
        z = torch.cat([e, self.glob_cnn(b["glob"]), self.loc_cnn(b["loc"]), self.scal_mlp(b["scal"])], -1)
        return self.head(z.float())


def param_groups(model: SignalClassifier, head_lr, enc_lr, decay, wd):
    """Layer-wise lr decay on the encoder; everything else at head_lr."""
    groups = []
    enc_ids = set()
    if model.encoder is not None:
        enc = model.encoder
        n = len(enc.blocks.layers)
        def add(params, lr):
            ps = [p for p in params]
            enc_ids.update(id(p) for p in ps)
            groups.append(dict(params=ps, lr=lr, base_lr=lr, weight_decay=wd))
        add(list(enc.norm.parameters()), enc_lr)
        for i, layer in enumerate(enc.blocks.layers):
            add(list(layer.parameters()), enc_lr * decay ** (n - 1 - i))
        add([enc.cls, enc.mask_token, enc.pos, *enc.patch_embed.parameters(), *enc.side_embed.parameters()],
            enc_lr * decay ** n)
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    groups.append(dict(params=rest, lr=head_lr, base_lr=head_lr, weight_decay=wd))
    return groups
