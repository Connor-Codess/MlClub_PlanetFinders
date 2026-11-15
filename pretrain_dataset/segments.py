"""Steps 3-6 of the build guide for one light curve: clean, grid, normalize, build channels.

Input: one row of a source dataset (time, flux, flux_err, centr_x, centr_y, and
for FFI bkg and n_points). Output: a Segment holding the seven per-slot
channels as little-endian byte strings plus the per-segment statistics, or a
drop reason.

Step 3  clean     sort by time; drop non-finite time or flux, flux <= 0 and
                  repeated timestamps; Kepler BKJD -> BTJD; drop rows with
                  fewer than 200 good bins.
Step 4  grid      slot = floor((t - GRID_EPOCH_BTJD) / 30 min); bins sharing a
                  slot are combined with inverse-variance weights and flagged
                  "merged"; the segment spans first to last observed slot.
Step 5  normalize sigma_p2p from differences of neighbouring observed slots;
                  sigma_mad, var_ratio, sigma_6h; flux_z = (flux - median) / sigma_p2p.
Step 6  channels  flux_z, flux_err_z, cen_x, cen_y (arcsec), bkg_z as float16;
                  fill and flags as uint8. Missing values are 0 with the
                  matching validity bit cleared.

decode() reverses the byte encoding; it is what a data loader uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config

MAD_TO_SIGMA = 1.4826


@dataclass
class Segment:
    slot0: int
    n_slots: int
    n_obs: int
    stats: dict[str, float]
    channels: dict[str, bytes]
    window_offsets: list[int] = field(default_factory=list)
    window_coverage: list[float] = field(default_factory=list)


class Dropped(Exception):
    """Raised with a short reason when a light curve cannot become a segment."""


def _mad(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.median(np.abs(values - np.median(values))))


def _as_array(values, dtype=np.float64) -> np.ndarray:
    if values is None:
        return None
    return np.asarray(values, dtype=dtype)


def _weighted_slot_mean(slot_index: np.ndarray, values: np.ndarray, weights: np.ndarray, n: int):
    """Inverse-variance mean per slot, ignoring NaN values. Returns (mean, has_value)."""
    ok = np.isfinite(values)
    w = np.where(ok, weights, 0.0)
    sums = np.bincount(slot_index, weights=np.where(ok, values, 0.0) * w, minlength=n)
    wsum = np.bincount(slot_index, weights=w, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / wsum
    return mean, wsum > 0


def build_segment(row: dict, source: str, sector_or_quarter: int) -> Segment:
    time = _as_array(row["time"])
    flux = _as_array(row["flux"])
    flux_err = _as_array(row.get("flux_err"))
    cen_x = _as_array(row.get("centr_x"))
    cen_y = _as_array(row.get("centr_y"))
    bkg = _as_array(row.get("bkg")) if source == "ffi" else None
    n_points = _as_array(row.get("n_points")) if source == "ffi" else None

    # Step 3: clean
    if time is None or time.size == 0:
        raise Dropped("empty")
    if source == "kepler":
        time = time + config.BKJD_TO_BTJD
    good = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
    order = np.argsort(time[good], kind="stable")
    idx = np.flatnonzero(good)[order]
    time = time[idx]
    keep = np.concatenate([[True], np.diff(time) > 0])
    idx, time = idx[keep], time[keep]
    if idx.size < config.MIN_GOOD_BINS:
        raise Dropped("too_few_bins")

    def take(arr):
        return None if arr is None else arr[idx]

    flux, flux_err, cen_x, cen_y, bkg, n_points = map(take, (flux, flux_err, cen_x, cen_y, bkg, n_points))

    # Step 4: grid
    slots = np.floor((time - config.GRID_EPOCH_BTJD) / config.SLOT_DAYS).astype(np.int64)
    slot0 = int(slots[0])
    rel = slots - slot0
    n_slots = int(rel[-1]) + 1
    counts = np.bincount(rel, minlength=n_slots)
    observed = counts > 0

    if flux_err is not None and np.isfinite(flux_err).any():
        err = np.where(np.isfinite(flux_err) & (flux_err > 0), flux_err, np.nanmedian(flux_err[flux_err > 0]))
        weights = 1.0 / err**2
    else:
        err = np.full(flux.shape, np.nan)
        weights = np.ones_like(flux)

    f_slot, _ = _weighted_slot_mean(rel, flux, weights, n_slots)
    wsum = np.bincount(rel, weights=weights, minlength=n_slots)
    if np.isfinite(err).any():
        with np.errstate(divide="ignore", invalid="ignore"):
            err_slot = np.where(wsum > 0, 1.0 / np.sqrt(wsum), np.nan)
    else:
        err_slot = np.full(n_slots, np.nan)
    cx_slot, cx_ok = _weighted_slot_mean(rel, cen_x, weights, n_slots) if cen_x is not None else (None, None)
    cy_slot, cy_ok = _weighted_slot_mean(rel, cen_y, weights, n_slots) if cen_y is not None else (None, None)
    bkg_slot, bkg_ok = _weighted_slot_mean(rel, bkg, weights, n_slots) if bkg is not None else (None, None)
    npts_slot = np.bincount(rel, weights=n_points, minlength=n_slots) if n_points is not None else None

    # Step 5: normalization statistics
    obs_idx = np.flatnonzero(observed)
    f_obs = f_slot[obs_idx]
    median = float(np.median(f_obs))
    neighbours = np.diff(obs_idx) == 1
    diffs = np.diff(f_obs)[neighbours]
    sigma = MAD_TO_SIGMA * _mad(diffs) / np.sqrt(2.0) if diffs.size >= 50 else float("nan")
    if not np.isfinite(sigma) or sigma <= 0:
        raise Dropped("no_noise_estimate")
    sigma_mad = MAD_TO_SIGMA * _mad(f_obs)
    sigma_6h = _sigma_6h(f_slot, observed)

    bkg_sigma = float("nan")
    if bkg_slot is not None and bkg_ok.any():
        b_idx = np.flatnonzero(bkg_ok)
        b_diffs = np.diff(bkg_slot[b_idx])[np.diff(b_idx) == 1]
        if b_diffs.size >= 50:
            bkg_sigma = MAD_TO_SIGMA * _mad(b_diffs) / np.sqrt(2.0)

    rel_scale = median  # flux columns are already divided by the median, so this is ~1
    sigma_ppm = sigma / rel_scale * 1e6
    if sigma_ppm > config.DROP_SIGMA_PPM:
        raise Dropped("too_noisy")

    # Step 6: channels
    def to_f16(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        out = np.where(valid & np.isfinite(values), values, 0.0)
        return np.clip(out, -config.FLOAT16_MAX, config.FLOAT16_MAX).astype("<f2")

    flux_z = (f_slot - median) / sigma
    flux_err_z = err_slot / sigma
    flags = np.where(observed, config.FLAG_OBSERVED, 0).astype(np.uint8)
    flags |= np.where(counts > 1, config.FLAG_MERGED, 0).astype(np.uint8)

    scale = config.PLATE_SCALE_ARCSEC[source]
    cen_stats = {}
    if cx_slot is not None and cy_slot is not None:
        cen_valid = observed & cx_ok & cy_ok & np.isfinite(cx_slot) & np.isfinite(cy_slot)
        cx_med = float(np.median(cx_slot[cen_valid])) if cen_valid.any() else 0.0
        cy_med = float(np.median(cy_slot[cen_valid])) if cen_valid.any() else 0.0
        cen_x_arcsec = (cx_slot - cx_med) * scale
        cen_y_arcsec = (cy_slot - cy_med) * scale
        flags |= np.where(cen_valid, config.FLAG_CENTROID_VALID, 0).astype(np.uint8)
        cen_stats = {"cen_x_median_px": cx_med, "cen_y_median_px": cy_med}
    else:
        cen_valid = np.zeros(n_slots, bool)
        cen_x_arcsec = cen_y_arcsec = np.zeros(n_slots)
        cen_stats = {"cen_x_median_px": float("nan"), "cen_y_median_px": float("nan")}

    if bkg_slot is not None:
        bkg_valid = observed & bkg_ok & np.isfinite(bkg_slot)
        bkg_med = float(np.median(bkg_slot[bkg_valid])) if bkg_valid.any() else 0.0
        bkg_z = (bkg_slot - bkg_med) / sigma
        flags |= np.where(bkg_valid, config.FLAG_BACKGROUND_VALID, 0).astype(np.uint8)
    else:
        bkg_valid = np.zeros(n_slots, bool)
        bkg_z = np.zeros(n_slots)

    if source == "ffi":
        fill = np.clip(npts_slot / config.ffi_points_per_slot(sector_or_quarter), 0.0, 1.0)
    elif source == "2min":
        reference = np.nanmedian(err_slot[observed]) if np.isfinite(err_slot[observed]).any() else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            fill = np.clip((reference / err_slot) ** 2, 0.0, 1.0)
        fill = np.where(np.isfinite(fill), fill, 1.0)
        flags |= np.where(observed, config.FLAG_FILL_ESTIMATED, 0).astype(np.uint8)
    else:
        fill = np.ones(n_slots)
    fill = np.where(observed, fill, 0.0)

    flags |= _spike_flags(flux_z, observed)
    flags |= _post_gap_flags(observed)

    channels = {
        "flux_z": to_f16(flux_z, observed).tobytes(),
        "flux_err_z": to_f16(flux_err_z, observed).tobytes(),
        "cen_x": to_f16(cen_x_arcsec, cen_valid).tobytes(),
        "cen_y": to_f16(cen_y_arcsec, cen_valid).tobytes(),
        "bkg_z": to_f16(bkg_z, bkg_valid).tobytes(),
        "fill": np.round(fill * 255).astype(np.uint8).tobytes(),
        "flags": flags.astype(np.uint8).tobytes(),
    }

    n_obs = int(observed.sum())
    stats = {
        "flux_median": median,
        "sigma_p2p_ppm": float(sigma_ppm),
        "sigma_mad_ppm": float(sigma_mad / rel_scale * 1e6),
        "var_ratio": float(sigma_mad / sigma),
        "sigma_6h_ppm": float(sigma_6h / rel_scale * 1e6),
        "bkg_sigma": float(bkg_sigma),
        **cen_stats,
    }
    offsets, coverage = default_windows(observed, source)
    return Segment(slot0, n_slots, n_obs, stats, channels, offsets, coverage)


def _sigma_6h(f_slot: np.ndarray, observed: np.ndarray) -> float:
    """Scatter of 6-hour means after removing a 2-day running median."""
    import pandas as pd

    values = pd.Series(np.where(observed, f_slot, np.nan))
    trend = values.rolling(config.SIGMA_6H_DETREND_SLOTS, center=True,
                           min_periods=config.SIGMA_6H_BIN_SLOTS).median()
    resid = (values - trend).to_numpy()
    step = config.SIGMA_6H_BIN_SLOTS
    usable = resid[: resid.size // step * step].reshape(-1, step)
    counts = np.isfinite(usable).sum(axis=1)
    sums = np.nansum(usable, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    means = means[counts >= step // 2]
    if means.size < 8:
        return float("nan")
    return MAD_TO_SIGMA * _mad(means)


def _spike_flags(flux_z: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Isolated upward spikes: above +8 sigma with both neighbours below half that value."""
    z = np.where(observed, flux_z, np.nan)
    left = np.concatenate([[np.nan], z[:-1]])
    right = np.concatenate([z[1:], [np.nan]])
    neighbour_max = np.fmax(left, right)
    spike = (z > config.SPIKE_SIGMA) & ~(neighbour_max >= z / 2)
    return np.where(spike, config.FLAG_SPIKE, 0).astype(np.uint8)


def _post_gap_flags(observed: np.ndarray) -> np.ndarray:
    flags = np.zeros(observed.size, np.uint8)
    obs_idx = np.flatnonzero(observed)
    gaps = np.diff(obs_idx) - 1
    for gap_end_pos in np.flatnonzero(gaps >= config.POST_GAP_MIN_GAP_SLOTS):
        first_after = obs_idx[gap_end_pos + 1]
        flags[first_after:first_after + config.POST_GAP_FLAG_SLOTS] = config.FLAG_POST_GAP
    return flags


def default_windows(observed: np.ndarray, source: str) -> tuple[list[int], list[float]]:
    """Default window starts (relative to slot0) and their observed coverage."""
    n = observed.size
    w = config.WINDOW_SLOTS
    if source == "kepler":
        offsets = list(range(0, max(n - w, 0) + 1, w)) if n >= w else [0]
        remainder = n - (offsets[-1] + w)
        if n >= w and remainder >= config.KEPLER_EXTRA_WINDOW_MIN_SLOTS:
            offsets.append(n - w)
    else:
        offsets = [0]
        if n - w >= config.TESS_EXTRA_WINDOW_MIN_SLOTS:
            offsets.append(n - w)
    coverage = [float(observed[o:o + w].sum()) / w for o in offsets]
    return offsets, coverage


def quality_tier(coverage: float, sigma_ppm: float) -> int:
    if coverage >= config.TIER1["coverage"] and sigma_ppm <= config.TIER1["sigma_ppm"]:
        return 1
    if coverage >= config.TIER2["coverage"] and sigma_ppm <= config.TIER2["sigma_ppm"]:
        return 2
    return 3


def decode(row: dict) -> dict[str, np.ndarray]:
    """Turn the stored byte channels of one row back into numpy arrays.

    Returns float32 arrays for flux_z, flux_err_z, cen_x, cen_y, bkg_z; fill as
    a 0-1 float32 array; flags as uint8. All have length n_slots.
    """
    out = {}
    for name in config.CHANNELS_FLOAT16:
        out[name] = np.frombuffer(row[name], dtype="<f2").astype(np.float32)
    out["fill"] = np.frombuffer(row["fill"], dtype=np.uint8).astype(np.float32) / 255.0
    out["flags"] = np.frombuffer(row["flags"], dtype=np.uint8)
    return out
