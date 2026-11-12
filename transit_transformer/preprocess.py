"""Shared light-curve preprocessing for Kepler and TESS.

Both computers must import this same file so the Kepler- and TESS-pretrained
models see identically prepared inputs. Pipeline, per light curve:

    quality/NaN mask -> flux / median -> 28-day windows -> 30-min bins
    -> per-window robust scaling -> fixed-length arrays + mask + side features

A window is the model's unit: L = 1344 bins (28 d at 30 min, = 84 patches of 16).
A TESS sector fits in one window; a Kepler quarter yields about 3.

Channels of `Window.values` (shape [L, 4]):
    0 flux       asinh(((flux - med) / sigma) / 5)        stellar variability kept
    1 flux_dt    same, after removing a 1-day running median  (transits stand out)
    2 flux_err   flux_err / sigma
    3 centroid   asinh((|centroid shift| / centroid_scale) / 5), 0 if unavailable
Empty bins are 0 in every channel and False in `mask`.

`Window.side` holds the scale information the per-window scaling removes, so the
model can still recover absolute depth (depth ~ sinh(5x)... * sigma):
    log10(sigma), log10(centroid_scale_arcsec), has_centroid,
    magnitude, has_magnitude, teff/5000, has_teff, log10(radius), has_radius,
    fill_fraction
No mission flag is included on purpose; it's kept in `meta` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---- locked constants: change only together on both computers ---------------
BIN_DAYS = 30.0 / 1440.0            # 30-minute bins (TESS HF set is already 30 min)
WINDOW_BINS = 1344                  # 28 days; divisible by 16 for patching
WINDOW_DAYS = WINDOW_BINS * BIN_DAYS
MIN_FILL = 0.5                      # drop windows with < 50% of bins filled
MIN_GOOD_POINTS = 100
DETREND_BINS = 48                   # 1-day running median for sigma / flux_dt
ASINH_SOFTNESS = 5.0                # linear within ~±5 sigma, compressed beyond
PIXEL_ARCSEC = {"kepler": 3.98, "tess": 21.0}
BJD_OFFSET = {"kepler": 2454833.0, "tess": 2457000.0}   # BKJD / BTJD -> BJD
N_SIDE = 10


@dataclass
class LightCurve:
    """One quarter (Kepler) or sector (TESS), before windowing."""
    mission: str                    # "kepler" or "tess"
    star_id: int                    # kepid or tic_id
    segment: int                    # quarter or sector
    time: np.ndarray                # mission time (BKJD / BTJD), days
    flux: np.ndarray                # any units; normalized here
    flux_err: np.ndarray
    good: np.ndarray                # bool: keep this cadence
    centr_x: np.ndarray | None = None   # pixels
    centr_y: np.ndarray | None = None
    magnitude: float | None = None  # kepmag / tmag
    teff: float | None = None
    radius: float | None = None


@dataclass
class Window:
    values: np.ndarray              # float32 [WINDOW_BINS, 4]
    mask: np.ndarray                # bool    [WINDOW_BINS]
    time_rel: np.ndarray            # float32 [WINDOW_BINS], days from window start (bin centres)
    side: np.ndarray                # float32 [N_SIDE]
    meta: dict = field(default_factory=dict)


# ---- loaders -----------------------------------------------------------------
def _finite(*arrays):
    ok = np.ones(len(arrays[0]), bool)
    for a in arrays:
        ok &= np.isfinite(a)
    return ok


def kepler_from_csv(path: str, kepmag: float | None = None, centroids=None) -> LightCurve:
    """Club CSV export (KEPLERID, QUARTER, TIME, ..., SAP_QUALITY, QUALITY_OK)."""
    df = pd.read_csv(path)
    return _kepler(
        star_id=int(df.KEPLERID.iloc[0]), quarter=int(df.QUARTER.iloc[0]),
        time=df.TIME.to_numpy(float), flux=df.PDCSAP_FLUX.to_numpy(float),
        flux_err=df.PDCSAP_FLUX_ERR.to_numpy(float),
        quality=df.SAP_QUALITY.to_numpy(), kepmag=kepmag,
        cadenceno=df.CADENCENO.to_numpy(), centroids=centroids,
    )


def kepler_from_hf_row(row, centroids=None) -> LightCurve:
    """Row of nyu-planet-finders/kepler-raw-lightcurves (one quarter).

    centroids: the matching row of the centroid dataset made by
    fetch_kepler_centroids.py (same shard, same row_in_shard), or None.
    """
    return _kepler(
        star_id=int(row["kepid"]), quarter=int(row["quarter"]),
        time=np.asarray(row["time"], float), flux=np.asarray(row["pdcsap_flux"], float),
        flux_err=np.asarray(row["pdcsap_flux_err"], float),
        quality=np.asarray(row["sap_quality"]), kepmag=row.get("kepmag"),
        cadenceno=np.asarray(row["cadenceno"]), centroids=centroids,
    )


def _kepler(star_id, quarter, time, flux, flux_err, quality, kepmag,
            cadenceno=None, centroids=None):
    # Strict rule, same as the TESS set: QUALITY == 0 *and* finite values.
    # QUALITY_OK alone is not enough: PDCSAP can be NaN on unflagged cadences.
    good = (np.nan_to_num(quality, nan=1) == 0) & _finite(time, flux, flux_err)
    cx = cy = None
    if centroids is not None:
        # Raw MOM_CENTR, like TESS centr_x/y; slow drift is removed later by
        # the running median. Cadence numbers must match exactly.
        cc = np.asarray(centroids["cadenceno"])
        if cadenceno is None or len(cc) != len(cadenceno) or not np.array_equal(cc, cadenceno):
            raise ValueError(f"kepid {star_id} Q{quarter}: centroid cadences do not match light curve")
        cx = np.asarray(centroids["mom_centr1"], float)
        cy = np.asarray(centroids["mom_centr2"], float)
    return LightCurve("kepler", star_id, quarter, time, flux, flux_err, good,
                      cx, cy, magnitude=_num(kepmag))


def tess_from_hf_row(row) -> LightCurve:
    """Row of nyu-planet-finders/tess-spoc-2min-binned (already QUALITY==0,
    median-normalized and 30-min binned; re-binning on the same grid is a no-op)."""
    time = np.asarray(row["time"], float)
    flux = np.asarray(row["flux"], float)
    err = np.asarray(row["flux_err"], float)
    cx = np.asarray(row["centr_x"], float)
    cy = np.asarray(row["centr_y"], float)
    good = _finite(time, flux, err)
    return LightCurve("tess", int(row["tic_id"]), int(row["sector"]), time, flux, err,
                      good, cx, cy, _num(row.get("tmag")), _num(row.get("teff")),
                      _num(row.get("radius")))


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


# ---- core --------------------------------------------------------------------
@dataclass
class Binned:
    """A 28-day window after masking and binning, before scaling.

    This is what the shards store, so noise augmentation can be applied at load
    time (scaling must be recomputed after adding noise).
    flux_rel: flux / light-curve median - 1 (float32 keeps ~ppm precision)
    err_rel:  flux_err / light-curve median
    cx, cy:   centroid offsets in arcsec (light-curve median removed), or None
    """
    flux_rel: np.ndarray
    err_rel: np.ndarray
    mask: np.ndarray
    cx: np.ndarray | None
    cy: np.ndarray | None
    magnitude: float | None
    teff: float | None
    radius: float | None
    meta: dict


def make_windows(lc: LightCurve, extra_noise: float = 0.0,
                 rng: np.random.Generator | None = None) -> list[Window]:
    """Turn one light curve into model-ready windows (bin_windows + scale_window).

    extra_noise: Gaussian noise (relative flux, e.g. 1e-3 = 1000 ppm per 30-min
    bin) added after binning. Use it to push quiet Kepler windows into the TESS
    noise range during pre-training; sigma and side features are updated to match.
    """
    return [scale_window(b, extra_noise, rng) for b in bin_windows(lc)]


def scale_window(b: Binned, extra_noise: float = 0.0,
                 rng: np.random.Generator | None = None) -> Window:
    """Per-window robust scaling -> model inputs. Cheap; run in the data loader."""
    cb = (b.cx, b.cy) if b.cx is not None else None
    values, mask, side = _scale(1.0 + b.flux_rel.astype(float), b.err_rel.astype(float),
                                b.mask, cb, b, extra_noise, rng)
    time_rel = (np.arange(WINDOW_BINS) * BIN_DAYS).astype(np.float32)
    return Window(values, mask, time_rel, side, dict(b.meta))


def bin_windows(lc: LightCurve) -> list[Binned]:
    """Mask, normalize, cut into 28-day windows and bin to 30 min."""
    g = lc.good
    if g.sum() < MIN_GOOD_POINTS:
        return []
    t, f, e = lc.time[g], lc.flux[g], lc.flux_err[g]
    order = np.argsort(t)
    t, f, e = t[order], f[order], e[order]
    med = np.median(f)
    f, e = f / med, e / med

    has_c = lc.centr_x is not None and lc.centr_y is not None
    if has_c:
        scale = PIXEL_ARCSEC[lc.mission]
        cx = lc.centr_x[g][order] * scale
        cy = lc.centr_y[g][order] * scale
        cok = np.isfinite(cx) & np.isfinite(cy)
        if cok.sum() < MIN_GOOD_POINTS:
            has_c = False
        else:   # per-quarter/sector median removal (Kepler rolls each quarter)
            cx = cx - np.nanmedian(cx)
            cy = cy - np.nanmedian(cy)

    windows = []
    start_idx = 0
    while start_idx < len(t):
        t0 = t[start_idx]
        end_idx = np.searchsorted(t, t0 + WINDOW_DAYS - BIN_DAYS / 2, side="left")
        sl = slice(start_idx, end_idx)
        w = _bin_window(t[sl] - t0, f[sl], e[sl],
                        cx[sl] if has_c else None, cy[sl] if has_c else None)
        if w is not None:
            fb, eb, mask, cb = w
            meta = dict(mission=lc.mission, star_id=lc.star_id, segment=lc.segment,
                        window=len(windows), t0_mission=float(t0),
                        t0_bjd=float(t0 + BJD_OFFSET[lc.mission]),
                        fill_fraction=float(mask.mean()))
            fr = np.where(mask, fb - 1.0, 0.0).astype(np.float32)
            er = np.where(mask, eb, 0.0).astype(np.float32)
            wcx = wcy = None
            if cb is not None:
                wcx, wcy = (c.astype(np.float32) for c in cb)   # NaN where no centroid
            windows.append(Binned(fr, er, mask, wcx, wcy, lc.magnitude, lc.teff,
                                  lc.radius, meta))
        start_idx = end_idx
    return windows


def _bin_window(trel, f, e, cx, cy):
    # Bins are centred on t0 + k*BIN_DAYS. TESS times already sit exactly on a
    # 30-min grid; edge-aligned bins would split them by float rounding.
    idx = np.floor(trel / BIN_DAYS + 0.5).astype(int)
    keep = idx < WINDOW_BINS
    idx, f, e = idx[keep], f[keep], e[keep]
    n = np.bincount(idx, minlength=WINDOW_BINS).astype(float)
    mask = n > 0
    if mask.mean() < MIN_FILL:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        fb = np.bincount(idx, f, WINDOW_BINS) / n
        eb = np.sqrt(np.bincount(idx, e ** 2, WINDOW_BINS)) / n
        cb = None
        if cx is not None:
            cx, cy = cx[keep], cy[keep]
            ok = np.isfinite(cx) & np.isfinite(cy)
            nc = np.bincount(idx[ok], minlength=WINDOW_BINS).astype(float)
            cb = (np.bincount(idx[ok], cx[ok], WINDOW_BINS) / nc,
                  np.bincount(idx[ok], cy[ok], WINDOW_BINS) / nc)
    return fb, eb, mask, cb


def _running_median(y, mask):
    """1-day running median over filled bins only (gaps are skipped, not filled)."""
    out = np.full_like(y, np.nan)
    out[mask] = (pd.Series(y[mask]).rolling(DETREND_BINS, center=True, min_periods=1)
                 .median().to_numpy())
    return out


def _robust_sigma(r):
    r = r[np.isfinite(r)]
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    return float(s) if s > 0 else float(np.std(r) or 1e-6)


def _soft(x):
    return np.arcsinh(x / ASINH_SOFTNESS)


def _scale(fb, eb, mask, cb, lc, extra_noise, rng):
    """lc: anything with .magnitude, .teff, .radius (LightCurve or Binned)."""
    if extra_noise > 0:
        rng = rng or np.random.default_rng()
        fb = fb + np.where(mask, rng.normal(0, extra_noise, fb.shape), 0)
        eb = np.sqrt(eb ** 2 + extra_noise ** 2)

    trend = _running_median(fb, mask)
    sigma = _robust_sigma((fb - trend)[mask])        # white-ish noise per 30-min bin
    med = np.median(fb[mask])

    vals = np.zeros((WINDOW_BINS, 4), np.float32)
    vals[mask, 0] = _soft((fb[mask] - med) / sigma)
    vals[mask, 1] = _soft((fb[mask] - trend[mask]) / sigma)
    vals[mask, 2] = eb[mask] / sigma

    c_scale, has_c = 1.0, 0.0
    if cb is not None:
        cx, cy = cb
        cm = mask & np.isfinite(cx) & np.isfinite(cy)
        if cm.mean() >= MIN_FILL:
            dx = cx - _running_median(np.where(cm, cx, 0), cm)   # remove slow drift
            dy = cy - _running_median(np.where(cm, cy, 0), cm)
            c_scale = _robust_sigma(np.concatenate([dx[cm], dy[cm]]))
            vals[cm, 3] = _soft(np.hypot(dx[cm], dy[cm]) / c_scale)
            has_c = 1.0

    def opt(v, fn):
        return (fn(v), 1.0) if v is not None and v > 0 else (0.0, 0.0)

    mag, has_mag = opt(lc.magnitude, lambda v: v / 10.0)
    teff, has_teff = opt(lc.teff, lambda v: v / 5000.0)
    rad, has_rad = opt(lc.radius, np.log10)
    side = np.array([np.log10(sigma), np.log10(c_scale), has_c, mag, has_mag,
                     teff, has_teff, rad, has_rad, mask.mean()], np.float32)
    return vals, mask, side


def stack(windows: list[Window]):
    """Batch helper: returns (values [B,L,4], mask [B,L], time_rel [B,L], side [B,F])."""
    return (np.stack([w.values for w in windows]), np.stack([w.mask for w in windows]),
            np.stack([w.time_rel for w in windows]), np.stack([w.side for w in windows]))
