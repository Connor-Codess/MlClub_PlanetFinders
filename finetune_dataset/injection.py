"""Window cutting, transit injection, event measurement and patch labels.

All functions work on one 27-day window of decoded lc-pretrain channels. Flux is
stored as flux_z = (flux / median - 1) / s, with s = sigma_p2p_ppm * 1e-6, so an
injected model m(t) multiplies relative flux:

    flux_z' = ((1 + s * flux_z) * m - 1) / s        on observed slots

The median and s of the host are kept, as the detector would compute nearly the
same values with the transit present. Unobserved slots stay at zero.

Two injection models:
    batman      Mandel-Agol transit (batman package) with quadratic limb darkening,
                integrated over the 30-minute slot.
    transplant  The phase-folded, measured transit of a real known planet
                (templates.py), placed at that planet's own period.
In both, the depth is chosen from a target per-transit SNR, so the set covers
the detection threshold evenly instead of being dominated by easy transits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pretrain_dataset import config as pretrain
from pretrain_dataset.segments import decode

from . import config

W = config.WINDOW_SLOTS
SLOT_CENTERS = np.arange(W) + 0.5


# ----------------------------------------------------------------------------- windows

def cut_window(row: dict, start: int) -> dict[str, np.ndarray]:
    """Decode a segment row and cut [start, start + W), padding past its end as unobserved."""
    arrays = decode(row)
    n = int(row["n_slots"])
    stop = min(start + W, n)
    out = {}
    for name, values in arrays.items():
        padded = np.zeros(W, dtype=values.dtype)
        if stop > start:
            padded[: stop - start] = values[start:stop]
        out[name] = padded
    return out


def observed_mask(flags: np.ndarray) -> np.ndarray:
    return (flags & pretrain.FLAG_OBSERVED) > 0


def encode(window: dict[str, np.ndarray]) -> dict[str, bytes]:
    out = {}
    for name in pretrain.CHANNELS_FLOAT16:
        values = np.clip(np.nan_to_num(window[name]), -pretrain.FLOAT16_MAX, pretrain.FLOAT16_MAX)
        out[name] = values.astype("<f2").tobytes()
    out["fill"] = np.clip(np.round(window["fill"] * 255.0), 0, 255).astype(np.uint8).tobytes()
    out["flags"] = window["flags"].astype(np.uint8).tobytes()
    return out


def hide_outside(window: dict[str, np.ndarray], lo: float, hi: float) -> None:
    """Mark slots whose centers fall outside [lo, hi) as unobserved, like a data gap."""
    hidden = (SLOT_CENTERS < lo) | (SLOT_CENTERS >= hi)
    for name, values in window.items():
        values[hidden] = 0


# ----------------------------------------------------------------------------- events

def measure_event(flux_z: np.ndarray, observed: np.ndarray, center: float,
                  duration_slots: float) -> tuple[float, float, float]:
    """(snr, observed fraction, depth in sigma) of a dip at a known position.

    Depth is the local out-of-transit median minus the in-transit mean; SNR is
    depth * sqrt(observed in-transit slots), since flux_z is in units of the
    slot-to-slot noise.
    """
    slot_centers = np.arange(flux_z.size) + 0.5
    inside = np.abs(slot_centers - center) < duration_slots / 2.0
    nominal = np.arange(math.floor(center - duration_slots / 2.0) - 1, math.ceil(center + duration_slots / 2.0) + 1) + 0.5
    expected = max(int((np.abs(nominal - center) < duration_slots / 2.0).sum()), 1)
    obs_in = inside & observed
    n_in = int(obs_in.sum())
    if n_in == 0:
        return 0.0, 0.0, 0.0
    margin = max(duration_slots, 12.0)
    near = (np.abs(slot_centers - center) < duration_slots / 2.0 + margin) & ~inside & observed
    baseline = float(np.median(flux_z[near])) if near.sum() >= 4 else 0.0
    depth = baseline - float(flux_z[obs_in].mean())
    return depth * math.sqrt(n_in), n_in / expected, depth


@dataclass
class Event:
    center_slot: float
    duration_slots: float
    kind_code: int
    kind: str
    ref: str
    cls: str
    period_d: float | None = None
    snr: float = 0.0
    observed_fraction: float = 0.0
    depth_sigma: float = 0.0

    @property
    def is_planet(self) -> bool:
        return self.kind_code in (config.KIND_REAL_PLANET, config.KIND_INJECTED_BATMAN,
                                  config.KIND_INJECTED_TRANSPLANT)

    def measure(self, flux_z: np.ndarray, observed: np.ndarray) -> "Event":
        self.snr, self.observed_fraction, self.depth_sigma = measure_event(
            flux_z, observed, self.center_slot, self.duration_slots)
        return self

    def as_record(self) -> dict:
        return {
            "center_slot": float(self.center_slot), "duration_slots": float(self.duration_slots),
            "kind": self.kind, "ref": self.ref, "cls": self.cls,
            "period_d": None if self.period_d is None else float(self.period_d),
            "snr": float(self.snr), "observed_fraction": float(self.observed_fraction),
            "visible": bool(self.observed_fraction >= config.MIN_VISIBLE_FRACTION),
            "is_planet": self.is_planet,
        }


KIND_FOR_CLASS = {
    "planet": config.KIND_REAL_PLANET,
    "eclipsing_binary": config.KIND_ECLIPSING_BINARY,
    "false_positive": config.KIND_FALSE_POSITIVE,
    "unknown": config.KIND_UNKNOWN_EVENT,
}


# ----------------------------------------------------------------------------- labels

def patch_labels(events: list[Event]) -> dict[str, np.ndarray]:
    """Per-patch targets for one window.

    patch_label          0 no transit, 1 a planet transit has its mid-time here and is
                         at least half observed, 255 ignore (planet transit mostly in a
                         gap, or a known event of unknown nature)
    patch_kind           code of the highest-priority event overlapping the patch
    patch_overlap        share of the patch inside a planet transit, 0-1
    patch_center_offset  position of the mid-time inside a labeled patch, 0-1, else NaN
    """
    label = np.zeros(config.N_PATCHES, dtype=np.uint8)
    kind = np.zeros(config.N_PATCHES, dtype=np.uint8)
    overlap = np.zeros(config.N_PATCHES, dtype=np.float32)
    offset = np.full(config.N_PATCHES, np.nan, dtype=np.float32)
    rank = {code: i for i, code in enumerate(config.KIND_PRIORITY)}
    edges = np.arange(config.N_PATCHES) * config.PATCH_SLOTS

    for event in events:
        lo = event.center_slot - event.duration_slots / 2.0
        hi = event.center_slot + event.duration_slots / 2.0
        covered = np.clip(np.minimum(hi, edges + config.PATCH_SLOTS) - np.maximum(lo, edges), 0, None)
        covered /= config.PATCH_SLOTS
        center_patch = int(math.floor(event.center_slot / config.PATCH_SLOTS))
        touched = covered > 0
        if 0 <= center_patch < config.N_PATCHES:
            touched[center_patch] = True
        for p in np.flatnonzero(touched):
            if kind[p] == config.KIND_NONE or rank[event.kind_code] < rank[kind[p]]:
                kind[p] = event.kind_code
        if event.is_planet:
            overlap = np.maximum(overlap, covered.astype(np.float32))
        if not 0 <= center_patch < config.N_PATCHES:
            continue
        visible = event.observed_fraction >= config.MIN_VISIBLE_FRACTION
        if event.is_planet and visible:
            label[center_patch] = config.LABEL_TRANSIT
            offset[center_patch] = event.center_slot / config.PATCH_SLOTS - center_patch
        elif (event.is_planet or event.kind_code == config.KIND_UNKNOWN_EVENT) and label[center_patch] != config.LABEL_TRANSIT:
            label[center_patch] = config.LABEL_IGNORE
    return {"patch_label": label, "patch_kind": kind, "patch_overlap": overlap,
            "patch_center_offset": offset}


# ----------------------------------------------------------------------------- injection

@dataclass
class Injection:
    model: np.ndarray                         # multiplicative flux model, length W
    events: list[Event]
    params: dict = field(default_factory=dict)


def apply_model(flux_z: np.ndarray, observed: np.ndarray, model: np.ndarray, s: float) -> np.ndarray:
    out = flux_z.copy()
    rel = (1.0 + s * flux_z[observed]) * model[observed]
    out[observed] = (rel - 1.0) / s
    return out


def transit_snr(model: np.ndarray, observed: np.ndarray, center: float, duration_slots: float, s: float) -> float:
    """Matched-filter SNR of one transit of the model on the observed slots."""
    span = (np.abs(SLOT_CENTERS - center) < duration_slots / 2.0 + 1.0) & observed
    return float(np.sqrt(np.sum((1.0 - model[span]) ** 2)) / s)


def star_mass(radius: float) -> float:
    if radius < 1.5:
        return float(np.clip(radius ** 1.25, 0.1, 1.6))
    return 1.2


def a_over_rs(period_d: float, mass: float, radius: float) -> float:
    return 215.032 * (mass * (period_d / 365.25) ** 2) ** (1.0 / 3.0) / radius


def duration_days(period_d: float, a_rs: float, rp: float, b: float) -> float:
    inc = math.acos(min(b / a_rs, 1.0))
    arg = math.sqrt(max((1.0 + rp) ** 2 - b ** 2, 0.0)) / (a_rs * math.sin(inc))
    return period_d / math.pi * math.asin(min(arg, 1.0))


def kipping_ld(rng: np.random.Generator) -> tuple[float, float]:
    """Quadratic limb darkening drawn uniformly over the physical range (Kipping 2013)."""
    q1, q2 = rng.uniform(0.0, 1.0, 2)
    return 2.0 * math.sqrt(q1) * q2, math.sqrt(q1) * (1.0 - 2.0 * q2)


def _anchor(rng: np.random.Generator, observed: np.ndarray) -> float:
    slots = np.flatnonzero(observed)
    return float(rng.choice(slots) + rng.uniform(0.0, 1.0))


def _centers(anchor: float, period_slots: float, half: float) -> list[float]:
    first = anchor - math.floor((anchor + half) / period_slots) * period_slots
    return [float(c) for c in np.arange(first, W + half, period_slots)]


def batman_injection(rng: np.random.Generator, observed: np.ndarray, s: float, radius: float | None,
                     teff: float | None, snr_range: tuple[float, float]) -> Injection:
    import batman

    radius = float(radius) if radius and radius > 0 and math.isfinite(radius) else config.DEFAULT_STAR_RADIUS
    mass = star_mass(radius)
    p_lo, p_hi = config.INJ_PERIOD_RANGE_D
    p_min = 365.25 * math.sqrt((config.INJ_MIN_A_RS * radius / 215.032) ** 3 / mass)
    p_lo = min(max(p_lo, p_min), p_hi)
    period = float(math.exp(rng.uniform(math.log(p_lo), math.log(p_hi))))
    a_rs = a_over_rs(period, mass, radius)
    b = float(rng.uniform(0.0, config.INJ_IMPACT_MAX))
    u1, u2 = kipping_ld(rng)
    snr_target = float(math.exp(rng.uniform(*np.log(snr_range))))

    mu = math.sqrt(max(1.0 - b * b, 0.0))
    ld_center = (1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2) / (1.0 - u1 / 3.0 - u2 / 6.0)
    rp = 0.05
    for _ in range(3):
        n_in = max(duration_days(period, a_rs, rp, b) / config.SLOT_DAYS, 1.0)
        depth = snr_target * s / math.sqrt(n_in)
        rp = float(np.clip(math.sqrt(max(depth, 0.0) / max(ld_center, 0.05)), *config.INJ_RP_RS_RANGE))
    duration = duration_days(period, a_rs, rp, b)

    anchor = _anchor(rng, observed)
    params = batman.TransitParams()
    params.t0 = anchor * config.SLOT_DAYS
    params.per = period
    params.rp = rp
    params.a = a_rs
    params.inc = math.degrees(math.acos(b / a_rs))
    params.ecc = 0.0
    params.w = 90.0
    params.u = [u1, u2]
    params.limb_dark = "quadratic"
    times = SLOT_CENTERS * config.SLOT_DAYS
    model = batman.TransitModel(params, times, supersample_factor=config.INJ_SUPERSAMPLE,
                                exp_time=config.INJ_EXPOSURE_DAYS).light_curve(params)

    dur_slots = duration / config.SLOT_DAYS
    events = [Event(c, dur_slots, config.KIND_INJECTED_BATMAN, "injected_batman", "batman", "planet", period)
              for c in _centers(anchor, period / config.SLOT_DAYS, dur_slots / 2.0)]
    snrs = [transit_snr(model, observed, e.center_slot, dur_slots, s) for e in events]
    return Injection(np.asarray(model, dtype=np.float64), events, {
        "inj_model": "batman", "inj_period_d": period, "inj_t0_slot": anchor, "inj_rp_rs": rp,
        "inj_a_rs": a_rs, "inj_b": b, "inj_u1": u1, "inj_u2": u2, "inj_duration_h": duration * 24.0,
        "inj_depth_ppm": float((1.0 - model.min()) * 1e6), "inj_snr_target": snr_target,
        "inj_snr": max(snrs) if snrs else 0.0, "inj_snr_total": float(np.sqrt(np.sum(np.square(snrs)))),
        "inj_star_radius": radius, "inj_star_mass": mass,
        "inj_star_teff": float(teff) if teff and math.isfinite(teff) else config.DEFAULT_TEFF,
        "inj_template_id": None, "inj_depth_scale": None,
    })


def template_model(template: dict, centers: list[float]) -> np.ndarray:
    """Evaluate a transplant template at every slot, using the nearest transit center."""
    centers_arr = np.asarray(centers)
    nearest = centers_arr[np.argmin(np.abs(SLOT_CENTERS[:, None] - centers_arr[None, :]), axis=1)]
    dt_hours = (SLOT_CENTERS - nearest) * config.SLOT_HOURS
    return np.interp(dt_hours, template["dt_h"], template["flux"], left=1.0, right=1.0)


def transplant_injection(rng: np.random.Generator, observed: np.ndarray, s: float, template: dict,
                         snr_range: tuple[float, float]) -> Injection:
    period = float(template["period_d"])
    dur_slots = float(template["duration_h"]) / config.SLOT_HOURS
    anchor = _anchor(rng, observed)
    centers = _centers(anchor, period / config.SLOT_DAYS, dur_slots / 2.0 + 2.0)
    full = template_model(template, centers)
    snr_target = float(math.exp(rng.uniform(*np.log(snr_range))))
    full_snr = max(transit_snr(full, observed, c, dur_slots, s) for c in centers)
    scale = min(1.0, snr_target / full_snr) if full_snr > 0 else 1.0
    model = 1.0 - scale * (1.0 - full)
    events = [Event(c, dur_slots, config.KIND_INJECTED_TRANSPLANT, "injected_transplant", template["template_id"],
                    "planet", period) for c in centers]
    snrs = [transit_snr(model, observed, e.center_slot, dur_slots, s) for e in events]
    return Injection(model, events, {
        "inj_model": "transplant", "inj_period_d": period, "inj_t0_slot": anchor, "inj_rp_rs": None,
        "inj_a_rs": None, "inj_b": None, "inj_u1": None, "inj_u2": None,
        "inj_duration_h": float(template["duration_h"]), "inj_depth_ppm": float((1.0 - model.min()) * 1e6),
        "inj_snr_target": snr_target, "inj_snr": max(snrs) if snrs else 0.0,
        "inj_snr_total": float(np.sqrt(np.sum(np.square(snrs)))), "inj_star_radius": None,
        "inj_star_mass": None, "inj_star_teff": None, "inj_template_id": template["template_id"],
        "inj_depth_scale": scale,
    })
