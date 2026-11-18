"""Kepler noise degradation, shared by fine-tuning (hosts "kdeg") and proxy pretraining.

What it does
    degrade_kepler() adds Gaussian noise to the flux of a share of Kepler windows so their
    noise matches a TESS FFI star drawn from a list of FFI noise levels (build guide, Stage 0
    loader, step 4). flux_z is in units of the star's own noise s_k; to look like a star of
    noise s_t = f * s_k the flux becomes z' = (z + N(0, f^2 - 1)) / f.

Why
    Kepler is about ten times more precise than FFI photometry. Training on undegraded Kepler
    windows teaches noise behaviour the FFI search never shows.

    For fine-tuning rows with a transit, f is capped so the transit's combined SNR stays at or
    above config.KEPLER_DEGRADE_MIN_SNR: the label says "transit", so it must stay findable.

How to use
    degrade_kepler(x, observed, sources, sigmas, ffi_sigmas, rng, max_factor=None)
    x is float32 (n, channels, 1296) with flux_z in channel 0 and is changed in place.
"""

from __future__ import annotations

import numpy as np

from . import config


def degrade_kepler(x: np.ndarray, observed: np.ndarray, sources: np.ndarray, sigmas: np.ndarray,
                   ffi_sigmas: np.ndarray, rng: np.random.Generator, max_factor: np.ndarray | None = None,
                   share: float = config.KEPLER_DEGRADE_SHARE) -> np.ndarray:
    """Degrade a random share of the Kepler rows; returns the noise factor used per row (1 = untouched)."""
    factors = np.ones(len(x), dtype=np.float32)
    if len(ffi_sigmas) == 0:
        return factors
    for i in np.flatnonzero((sources == "kepler") & (rng.uniform(size=len(x)) < share)):
        s_k = float(sigmas[i])
        if not np.isfinite(s_k) or s_k <= 0:
            continue
        f = float(rng.choice(ffi_sigmas)) / s_k
        if max_factor is not None and np.isfinite(max_factor[i]):
            f = min(f, float(max_factor[i]))
        if f <= 1.0:
            continue
        obs = observed[i].astype(bool)
        noise = rng.normal(0.0, np.sqrt(f * f - 1.0), size=int(obs.sum()))
        x[i, 0, obs] = (x[i, 0, obs] + noise) / f
        factors[i] = f
    return factors


def positive_caps(meta) -> np.ndarray:
    """Largest noise factor per fine-tuning row: rows with a transit keep combined SNR >= the floor."""
    snr = np.where(meta["category"].isin(config.INJECTED), meta["inj_snr_total"].astype(float),
                   meta["stacked_snr"].astype(float))
    has_transit = meta["category"].isin((*config.INJECTED, *config.REAL)).to_numpy()
    caps = np.full(len(meta), np.inf)
    caps[has_transit] = np.maximum(1.0, np.nan_to_num(snr[has_transit], nan=0.0) / config.KEPLER_DEGRADE_MIN_SNR)
    return caps
