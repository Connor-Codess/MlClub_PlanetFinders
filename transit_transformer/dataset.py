"""PyTorch data loading for joint Kepler + TESS pre-training.

Reads shards from build_shards.py, scales each window with
preprocess.scale_window at load time, mixes missions at a fixed ratio and
applies Kepler noise augmentation.

    from dataset import MixedPretrainDataset, collate
    ds = MixedPretrainDataset("shards", split="train")
    dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=8, collate_fn=collate)
    for batch in dl:  # dict of tensors: values [B,L,4], mask [B,L], time [B,L], side [B,F], mission [B]
        ...
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

import preprocess as P

MISSIONS = ("kepler", "tess")


def shard_paths(root, mission, split):
    return sorted(glob.glob(os.path.join(root, mission, split, "*.npz")))


def _iter_shard(path, rng):
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    for i in rng.permutation(len(d["flux_rel"])):
        has_c = bool(d["has_c"][i])
        meta = dict(star_id=int(d["star_id"][i]), segment=int(d["segment"][i]),
                    window=int(d["window"][i]), t0_bjd=float(d["t0_bjd"][i]),
                    fill_fraction=float(d["fill"][i]))

        def opt(v):
            return None if np.isnan(v) else float(v)

        b = P.Binned(d["flux_rel"][i], d["err_rel"][i].astype(np.float32), d["mask"][i],
                     d["cx"][i] if has_c else None, d["cy"][i] if has_c else None,
                     opt(d["magnitude"][i]), opt(d["teff"][i]), opt(d["radius"][i]), meta)
        yield b, float(d["sigma"][i])


class MissionStream:
    """Endless shuffled stream of (Binned, sigma) for one mission."""

    def __init__(self, paths, rng):
        if not paths:
            raise FileNotFoundError("no shards found")
        self.paths, self.rng = list(paths), rng
        self.it = iter(())

    def __next__(self):
        while True:
            try:
                return next(self.it)
            except StopIteration:
                path = self.paths[self.rng.integers(len(self.paths))]
                self.it = _iter_shard(path, self.rng)


class MixedPretrainDataset(IterableDataset):
    def __init__(self, root, split="train", kepler_frac=0.5, augment_frac=0.5,
                 zero_centroids=False, seed=0):
        """
        kepler_frac:    share of Kepler windows per batch (TESS gets the rest).
        augment_frac:   share of Kepler windows given extra noise so their sigma
                        matches a sigma drawn from the TESS shards.
        zero_centroids: blank channel 3 and has_centroid for both missions.
        """
        self.root, self.split, self.seed = root, split, seed
        self.kepler_frac, self.augment_frac = kepler_frac, augment_frac
        self.zero_centroids = zero_centroids
        self.paths = {m: shard_paths(root, m, split) for m in MISSIONS}

    def _tess_sigmas(self, rng, n_shards=20):
        paths = self.paths["tess"]
        pick = rng.choice(len(paths), min(n_shards, len(paths)), replace=False)
        return np.concatenate([np.load(paths[i])["sigma"] for i in pick])

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng([self.seed, wid])
        streams = {}
        for m in MISSIONS:
            mine = self.paths[m][wid::nw] or self.paths[m]   # split shards across workers
            streams[m] = MissionStream(mine, rng)
        target_sigmas = self._tess_sigmas(rng) if self.augment_frac > 0 else None

        while True:
            m = "kepler" if rng.random() < self.kepler_frac else "tess"
            b, sigma = next(streams[m])
            extra = 0.0
            if m == "kepler" and target_sigmas is not None and rng.random() < self.augment_frac:
                target = float(rng.choice(target_sigmas))
                extra = float(np.sqrt(max(target ** 2 - sigma ** 2, 0.0)))
            w = P.scale_window(b, extra, rng)
            if self.zero_centroids:
                w.values[:, 3] = 0
                w.side[1:3] = 0
            yield dict(values=w.values, mask=w.mask, time=w.time_rel, side=w.side,
                       mission=MISSIONS.index(m))


def collate(items):
    return {k: torch.as_tensor(np.stack([it[k] for it in items])) for k in items[0]}
