"""Constants for the lc-pretrain build. Every number here is a decision from the build guide.

Changing anything marked "baked" changes the stored data and requires a new
build version. Everything else (window length, sampling weights) is a default
that loaders may override.
"""

from __future__ import annotations

BUILD_VERSION = "v1"

# Hugging Face repositories
ORG = "nyu-planet-finders"
PRETRAIN_REPO = f"{ORG}/lc-pretrain"          # public: pretrain-train and pretrain-val rows
HELDOUT_REPO = f"{ORG}/lc-heldout"            # private: search-holdout, scoreboard hosts, list exclusions
REFERENCE_REPO = f"{ORG}/reference-catalogs"
SOURCE_REPOS = {
    "ffi": f"{ORG}/tess-spoc-ffi-binned",
    "2min": f"{ORG}/tess-spoc-2min-binned",
    "kepler": f"{ORG}/kepler-llc",
}
KEPLER_TCE_REPO = f"{ORG}/kepler-raw-lightcurves"
KEPLER_TCE_FILE = "tces/dr24_tce.parquet"

# Step 4: global time grid (baked)
SLOT_DAYS = 30.0 / 1440.0
GRID_EPOCH_BTJD = -2200.0                     # slot index = floor((t - GRID_EPOCH_BTJD) / SLOT_DAYS)
BKJD_TO_BTJD = -2167.0                        # BTJD = BKJD - 2167

# Step 4: default windows (loader default, stored as an index)
WINDOW_SLOTS = 1296                           # 27 days
TESS_EXTRA_WINDOW_MIN_SLOTS = 288             # add an end-aligned window if >= 6 days remain
KEPLER_EXTRA_WINDOW_MIN_SLOTS = 648           # add an end-aligned window if >= half a window remains

# Step 3: cleaning (baked)
MIN_GOOD_BINS = 200
SPIKE_SIGMA = 8.0                             # isolated upward spike flag
POST_GAP_MIN_GAP_SLOTS = 24                   # gaps of 12 h or more
POST_GAP_FLAG_SLOTS = 12                      # flag the 6 h after such a gap

# Step 5: normalization (baked)
SIGMA_6H_BIN_SLOTS = 12
SIGMA_6H_DETREND_SLOTS = 96                   # 2-day running median removed before sigma_6h
FLOAT16_MAX = 65000.0

# Step 6: channels (baked)
PLATE_SCALE_ARCSEC = {"ffi": 21.0, "2min": 21.0, "kepler": 3.98}
TWO_MIN_POINTS_PER_SLOT = 15
CHANNELS_FLOAT16 = ("flux_z", "flux_err_z", "cen_x", "cen_y", "bkg_z")
CHANNELS_UINT8 = ("fill", "flags")
FLAG_OBSERVED = 1 << 0
FLAG_CENTROID_VALID = 1 << 1
FLAG_BACKGROUND_VALID = 1 << 2
FLAG_FILL_ESTIMATED = 1 << 3
FLAG_MERGED = 1 << 4
FLAG_SPIKE = 1 << 5
FLAG_POST_GAP = 1 << 6

# Step 8: quality (drop is baked; tiers are tags)
DROP_SIGMA_PPM = 50_000.0
TIER1 = {"coverage": 0.8, "sigma_ppm": 2_000.0}
TIER2 = {"coverage": 0.5, "sigma_ppm": 10_000.0}

# Step 1: splits (baked)
SPLIT_BUCKETS = [
    (0, 850, "train"),
    (850, 900, "validation"),
    (900, 950, "search_holdout"),
    (950, 1000, "scoreboard_hosts"),
]
PRETRAIN_SPLITS = ("train", "validation")

# Step 2: list exclusions (baked)
LONG_PERIOD_DAYS = 20.0
PLANET_DISPOSITIONS = ("PC", "APC", "CP", "KP")

# Step 7: known events
MIN_EVENT_PERIOD_DAYS = 0.5                   # shorter periods are tagged but not expanded into events
MAX_EVENTS_PER_SEGMENT = 200

# Step 9: default sampling weights (loader default)
SOURCE_SHARES = {"ffi": 0.50, "2min": 0.25, "kepler": 0.25}
STAR_WINDOW_CAP = 20
VARIABILITY_TOP_FRACTION = 0.20
VARIABILITY_WEIGHT = 2.0
TIER3_WEIGHT = 0.5


def native_cadence_s(source: str, sector_or_quarter: int) -> int:
    if source == "kepler":
        return 1765
    if source == "2min":
        return 120
    if sector_or_quarter <= 26:
        return 1800
    if sector_or_quarter <= 55:
        return 600
    return 200


def ffi_points_per_slot(sector: int) -> int:
    """Raw FFI cadences expected in one 30-minute bin."""
    return {1800: 1, 600: 3, 200: 9}[native_cadence_s("ffi", sector)]


def quality_policy(source: str) -> str:
    return "bitmask17087" if source == "ffi" else "quality0"
