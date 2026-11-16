"""Measured transit profiles of real planets, for transplant injections.

Scan stage (cutouts): for every transit of a known planet (TOI PC/APC/CP/KP or
Kepler DR24 PC) on a train or validation star, take the slots within
duration/2 + max(4 h, duration/2) of mid-transit, divide out a straight line fitted
to the out-of-transit slots, and keep (hours from mid-transit, relative flux).

Choose stage (templates): stack all cutouts of one planet, take the median in
10-minute bins, fill empty bins by interpolation, smooth with a 3-bin running
median and normalize to the out-of-transit median. A planet becomes a template
when it has at least 3 transits, at least 60% of its in-transit bins have data,
and the folded depth is at least 10 times the out-of-transit bin scatter. Beyond
duration/2 + 0.75 h the template is exactly 1, so no noise from the source star is
copied into the host.

Because every source light curve is binned to 30 minutes, the fold already holds
the 30-minute-smeared profile, which is what a 30-minute host slot sees. Kepler
templates are in the Kepler band; the band difference changes limb darkening a
little and is ignored.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config


def cutout(flux_z: np.ndarray, observed: np.ndarray, s: float, center: float,
           duration_slots: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Detrended (dt_hours, relative flux) around one transit, or None if unusable."""
    half = duration_slots / 2.0
    reach = half + max(8.0, half)
    slot_centers = np.arange(flux_z.size) + 0.5
    near = (np.abs(slot_centers - center) <= reach) & observed
    dt = (slot_centers[near] - center) * config.SLOT_HOURS
    rel = 1.0 + s * flux_z[near]
    out = np.abs(dt) > half * config.SLOT_HOURS + config.TEMPLATE_EDGE_HOURS
    inside = np.abs(dt) < half * config.SLOT_HOURS
    if out.sum() < 6 or inside.sum() < max(2, 0.6 * duration_slots):
        return None
    if (dt[out] < 0).sum() < 2 or (dt[out] > 0).sum() < 2:
        return None
    slope, intercept = np.polyfit(dt[out], rel[out], 1)
    trend = intercept + slope * dt
    if np.any(trend <= 0):
        return None
    return dt.astype(np.float32), (rel / trend).astype(np.float32)


def build_template(group: pd.DataFrame, period_d: float, duration_h: float) -> dict | None:
    """Stack one planet's cutouts into a template, or None if it fails the quality cuts."""
    if len(group) < config.TEMPLATE_MIN_TRANSITS or not duration_h or not math.isfinite(duration_h):
        return None
    dt = np.concatenate(group["dt_h"].to_numpy())
    rel = np.concatenate(group["rel"].to_numpy())
    half = duration_h / 2.0
    reach = half + max(4.0, half)
    edges = np.arange(-reach, reach + config.TEMPLATE_BIN_HOURS, config.TEMPLATE_BIN_HOURS)
    centers = (edges[:-1] + edges[1:]) / 2.0
    which = np.digitize(dt, edges) - 1
    values = np.full(centers.size, np.nan)
    for i in range(centers.size):
        sel = which == i
        if sel.any():
            values[i] = np.median(rel[sel])
    outside = np.abs(centers) > half + config.TEMPLATE_EDGE_HOURS
    inside = np.abs(centers) < half
    if np.isfinite(values[inside]).mean() < config.TEMPLATE_MIN_BIN_FILL or np.isfinite(values[outside]).sum() < 6:
        return None
    baseline = float(np.nanmedian(values[outside]))
    values = values / baseline
    scatter = 1.4826 * float(np.nanmedian(np.abs(values[outside] - 1.0)))
    values = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    smooth = pd.Series(values).rolling(3, center=True, min_periods=1).median().to_numpy()
    depth = 1.0 - float(np.min(smooth[inside]))
    # scatter is per bin; the template's noise is the bin scatter, its signal the depth
    if not 0 < depth <= config.TEMPLATE_MAX_DEPTH or depth < config.TEMPLATE_MIN_SNR * max(scatter, 1e-7):
        return None
    profile = np.where(outside, 1.0, np.minimum(smooth, 1.0))
    return {
        "dt_h": centers.astype(np.float32),
        "flux": profile.astype(np.float32),
        "period_d": float(period_d),
        "duration_h": float(duration_h),
        "depth_ppm": depth * 1e6,
        "fold_snr": depth / max(scatter, 1e-7),
        "n_transits": int(len(group)),
    }


def build_templates(cutouts: pd.DataFrame) -> pd.DataFrame:
    """One row per accepted template (columns of build_template plus identity)."""
    rows = []
    for (ref, star_key), group in cutouts.groupby(["ref", "star_key"], sort=True):
        first = group.iloc[0]
        template = build_template(group, first["period_d"], first["duration_h"])
        if template is None:
            continue
        template.update({
            "template_id": ref.replace(" ", "_"),
            "ref": ref,
            "star_key": star_key,
            "pool": first["pool"],
            "mission": "kepler" if first["source"] == "kepler" else "tess",
        })
        rows.append(template)
    columns = ["dt_h", "flux", "period_d", "duration_h", "depth_ppm", "fold_snr", "n_transits",
               "template_id", "ref", "star_key", "pool", "mission"]
    return pd.DataFrame(rows, columns=columns)
