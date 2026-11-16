"""Constants for the lc-finetune and lc-eval builds.

Every sample is one 27-day window (1,296 slots of 30 minutes) cut from a segment
of lc-pretrain or lc-heldout, with transit labels per 4-hour patch. Changing a
value here changes the stored data and needs a new BUILD_VERSION.
"""

from __future__ import annotations

from pretrain_dataset import config as pretrain

BUILD_VERSION = "v2"

ORG = pretrain.ORG
FINETUNE_REPO = f"{ORG}/lc-finetune"          # public: stars of the train and validation splits only
EVAL_REPO = f"{ORG}/lc-eval"                  # private: held-out stars only (scoreboard hosts, list exclusions)

# Window and patch layout
WINDOW_SLOTS = pretrain.WINDOW_SLOTS          # 27 days
PATCH_SLOTS = 8                               # 4 hours
N_PATCHES = WINDOW_SLOTS // PATCH_SLOTS       # 162
SLOT_DAYS = pretrain.SLOT_DAYS
SLOT_HOURS = SLOT_DAYS * 24.0

# Patch label values (patch_label channel)
LABEL_NEGATIVE = 0
LABEL_TRANSIT = 1                             # a planet transit (real or injected) has its mid-time in the patch
LABEL_IGNORE = 255                            # transit mostly unobserved, or a known event of unknown nature

# Patch kind codes (patch_kind channel): what overlaps the patch, highest priority first
KIND_NONE = 0
KIND_REAL_PLANET = 1
KIND_INJECTED_BATMAN = 2
KIND_INJECTED_TRANSPLANT = 3
KIND_ECLIPSING_BINARY = 4
KIND_FALSE_POSITIVE = 5                       # TOI FP/FA, Kepler DR24 AFP/NTP
KIND_UNKNOWN_EVENT = 6                        # TOI without disposition, Kepler DR24 UNK
KIND_PRIORITY = (KIND_REAL_PLANET, KIND_INJECTED_BATMAN, KIND_INJECTED_TRANSPLANT,
                 KIND_ECLIPSING_BINARY, KIND_FALSE_POSITIVE, KIND_UNKNOWN_EVENT)
MIN_VISIBLE_FRACTION = 0.5                    # share of in-transit slots observed for LABEL_TRANSIT
DEFAULT_EVENT_DURATION_H = 4.0                # used when a catalog gives no duration (Kepler EBs)

# Event classes from known_events
PLANET_DISPOSITIONS = pretrain.PLANET_DISPOSITIONS
FALSE_POSITIVE_DISPOSITIONS = ("FP", "FA")
LONG_PERIOD_DAYS = pretrain.LONG_PERIOD_DAYS  # real_planet_long: period >= 20 d or unknown

# Star pools. lc-finetune uses train/validation stars; lc-eval uses only held-out stars.
POOL_TRAIN = "train"
POOL_VALIDATION = "validation"
POOL_SCOREBOARD_VAL = "scoreboard_val"        # scoreboard_hosts, first hash half
POOL_SCOREBOARD_TEST = "scoreboard_test"
POOL_LIST_VAL = "list_val"                    # list_excluded, real_validation
POOL_LIST_TEST = "list_test"                  # list_excluded, real_test
POOL_UNUSED = "unused"                        # search_holdout stays untouched
EVAL_POOLS = {"val": (POOL_SCOREBOARD_VAL, POOL_LIST_VAL), "test": (POOL_SCOREBOARD_TEST, POOL_LIST_TEST)}

# lc-finetune size and mix (shares of rows in each split)
FINETUNE_ROWS = {"train": 400_000, "validation": 24_000}
FINETUNE_MIX = {
    "real_planet_short": 0.150,
    "real_planet_long": 0.075,
    "injected_batman": 0.350,
    "injected_transplant": 0.150,
    "negative_random": 0.175,
    "negative_eb": 0.100,
}
# Rows a category cannot fill go to this one
FINETUNE_TOP_UP = "injected_batman"
# Source mix of host and random-negative windows
FINETUNE_HOST_SOURCES = {"ffi": 0.6, "2min": 0.2, "kepler": 0.2}
# Source mix of real-transit and EB windows (TESS covers ffi and 2min)
FINETUNE_REAL_TESS_SHARE = 0.6
# Caps on windows per star in one category and split
# v2: 30 (v1: 12). The per-star cap, not the SNR cut, was what limited real short-period rows.
REAL_WINDOWS_PER_STAR = 30
EB_WINDOWS_PER_STAR = 10
HOST_WINDOWS_PER_STAR = 3
# Real planets in lc-finetune are verified as a whole: their transits, folded over every sector and
# quarter of the star, must reach this SNR (catches ephemerides that no longer land on the transits).
# Transits of planets that fail it are labeled ignore (255), not positive.
MIN_PLANET_FOLD_SNR = 7.0
# A real window enters lc-finetune when its visible verified planet transits reach this combined SNR,
# sqrt(sum of squared per-transit SNRs), so faint repeated transits that add up inside the window count.
MIN_REAL_COMBINED_SNR = 4.0
MIN_EB_SNR = 3.0
# Share of real short-period rows reduced to one visible transit (others hidden as a data gap); the kept
# transit must itself reach this per-transit SNR, so a crop never leaves only an invisible transit.
SINGLE_TRANSIT_CROP_SHARE = 0.25
SINGLE_TRANSIT_CROP_MIN_SNR = 4.0

# lc-eval sizes per split (val and test). Real categories take every qualifying window.
EVAL_ROWS = {"injected_batman": 20_000, "injected_transplant": 5_000, "negative_random": 20_000,
             "negative_eb": 5_000}
EVAL_HOST_SOURCES = {"ffi": 1.0}              # injections and random negatives: FFI, the search target
EVAL_REAL_WINDOWS_PER_STAR = 10
EVAL_EB_WINDOWS_PER_STAR = 5

# Host candidates kept by the scan (deterministic hash sampling), by pool and source
HOST_SAMPLE_RATE = {
    POOL_TRAIN: {"ffi": 0.05, "2min": 0.12, "kepler": 0.03},
    POOL_VALIDATION: {"ffi": 0.06, "2min": 0.15, "kepler": 0.03},
    POOL_SCOREBOARD_VAL: {"ffi": 0.40},
    POOL_SCOREBOARD_TEST: {"ffi": 0.40},
}
HOST_MIN_COVERAGE = 0.3
HOST_MAX_RADIUS_RSUN = 5.0                    # injections avoid giants (transits of days)

# Batman injections
INJ_PERIOD_RANGE_D = (1.0, 730.0)             # log-uniform; about half the draws exceed 27 d
INJ_IMPACT_MAX = 0.9
INJ_RP_RS_RANGE = (0.003, 0.3)
INJ_MIN_A_RS = 3.0
# Target combined SNR of the injected transits observed in the window, log-uniform. Each transit gets
# target / sqrt(transits observed), so a window with 13 transits at target 3 has per-transit SNR 0.8.
INJ_SNR_RANGE = {"finetune": (3.0, 60.0), "eval": (2.0, 40.0)}
INJ_EXPOSURE_DAYS = SLOT_DAYS                 # 30-minute integration
INJ_SUPERSAMPLE = 15
DEFAULT_STAR_RADIUS = 1.0
DEFAULT_TEFF = 5500.0

# Transplanted real transits (templates from phase-folded known planets)
TEMPLATE_BIN_HOURS = 1.0 / 6.0
TEMPLATE_MIN_TRANSITS = 3
TEMPLATE_MIN_SNR = 10.0                       # folded depth over out-of-transit scatter
TEMPLATE_MIN_CUTOUT_SNR = 2.0
TEMPLATE_MIN_BIN_FILL = 0.6                   # share of in-transit 10-minute bins with data
TEMPLATE_MAX_CUTOUTS_PER_SEGMENT = 20
TEMPLATE_EDGE_HOURS = 0.75                    # template is exactly 1 beyond duration/2 + this
TEMPLATE_MAX_DEPTH = 0.5
TEMPLATE_POOLS = {"finetune": POOL_TRAIN, "eval": POOL_VALIDATION}

# Published benchmarks (benchmarks.py): windows kept per signal, chosen at random
BENCHMARK_REPO = f"{ORG}/lc-benchmarks"      # private: our windows around published test signals
BENCH_WINDOWS_PER_SIGNAL = 8

SEED = 20261116
