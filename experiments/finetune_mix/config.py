"""Every number of the fine-tuning data mix experiments.

What it is
    The arms (data combinations), the training budget, the detector's size, and the
    bins and false-alarm rates used to score each run on lc-eval val. Changing a
    number here changes the experiment; record any change in the README's change log.

Why these numbers
    Each arm trains on the same number of rows (ROWS_PER_ARM) with the same model,
    optimizer and number of optimizer steps, so the only thing that differs between arms
    is which rows they see. Scaling arms change only the row count. Three seeds per arm
    give a measure of run-to-run noise to compare differences against.

How to use
    Imported by the other modules; nothing to run.
"""

from __future__ import annotations

ORG = "nyu-planet-finders"
FINETUNE_REPO = f"{ORG}/lc-finetune"
EVAL_REPO = f"{ORG}/lc-eval"
EVAL_SPLIT = "val"            # never "test": the test split is used once, at the end of the project

WINDOW_SLOTS = 1296
PATCH_SLOTS = 8
N_PATCHES = 162
LABEL_NEGATIVE, LABEL_TRANSIT, LABEL_IGNORE = 0, 1, 255

CATEGORIES = ("real_planet_short", "real_planet_long", "injected_batman", "injected_transplant",
              "negative_random", "negative_eb")
REAL = ("real_planet_short", "real_planet_long")
INJECTED = ("injected_batman", "injected_transplant")
TESS_SOURCES = ("ffi", "2min")

# ----------------------------------------------------------------------------- arms
ROWS_PER_ARM = 80_000
SELECTION_SEED = 20261117

# Each arm starts from the v2 build's own category shares, scaled to its row count. Then:
#   move     a category's quota goes to the listed categories, split by their base quotas
#   filters  rows of a category must pass a test (see select.ROW_FILTERS)
#   sources  only rows from these sources, in every category
#   rows     row count (default ROWS_PER_ARM)
WAVE1_ARMS: dict[str, dict] = {
    "baseline": {
        "question": "the v2 mix as built",
    },
    "v1_rule": {
        "question": "v1's rules: real windows need one transit at SNR >= 4, injections per-transit SNR >= 3",
        "filters": {"real_planet_short": "best_transit_snr_ge_4", "real_planet_long": "best_transit_snr_ge_4",
                    "injected_batman": "inj_snr_ge_3", "injected_transplant": "inj_snr_ge_3"},
    },
    "no_real": {
        "question": "are real transits worth their rows? (real rows replaced by injections)",
        "move": {"real_planet_short": INJECTED, "real_planet_long": INJECTED},
    },
    "no_injected": {
        "question": "are injections worth their rows? (injected rows replaced by real transits)",
        "move": {"injected_batman": REAL, "injected_transplant": REAL},
    },
    "batman_only": {
        "question": "does a real transit shape help? (transplants replaced by batman)",
        "move": {"injected_transplant": ("injected_batman",)},
    },
    "transplant_only": {
        "question": "does a real transit shape help? (batman replaced by transplants)",
        "move": {"injected_batman": ("injected_transplant",)},
    },
    "no_eb": {
        "question": "how much do EB negatives cut EB false alarms? (EB rows replaced by random negatives)",
        "move": {"negative_eb": ("negative_random",)},
    },
    "no_crop": {
        "question": "do single-transit crops help single-transit recall?",
        "filters": {"real_planet_short": "not_cropped"},
    },
    "tess_only": {
        "question": "does Kepler data help on TESS FFI? (TESS rows only; real long-period rows are all Kepler)",
        "sources": TESS_SOURCES,
        "move": {"real_planet_long": ("real_planet_short",)},
    },
    "strong_injections": {
        "question": "does training on faint injections help, or only add false alarms? (combined SNR >= 7 only)",
        "filters": {"injected_batman": "inj_snr_total_ge_7", "injected_transplant": "inj_snr_total_ge_7"},
    },
    "rows_20k": {"question": "data scale", "rows": 20_000},
    "rows_40k": {"question": "data scale", "rows": 40_000},
    "rows_160k": {"question": "data scale", "rows": 160_000},
}

# Wave 2 (2026-11-18). Every arm starts from lc-finetune v3 and the v2 category shares, then sets:
#   real_share  share of real planet rows (base 16.0%); the freed rows go to batman and transplants
#               in their base ratio
#   deep        False drops the deep-tail batman rows (combined SNR 60-300); True keeps them (~10%)
#   hosts       "all": every source; "tess": injected and random-negative rows from TESS only (real
#               planets and EBs keep Kepler); "kdeg": every source, with half of the Kepler rows given
#               extra noise at training time to match a random FFI star (train.py)
#   init        a proxy_pretrain checkpoint to start from, or None for training from scratch
WAVE2_REAL_SHARES = (0.0, 0.05, 0.16)


def _wave2_arm(init, share, deep, hosts) -> tuple[str, dict]:
    name = f"{init or 'scratch'}-r{round(share * 100):02d}-{'deep' if deep else 'nodeep'}-{hosts}"
    spec = {"question": f"init={init or 'none'}, real share {share:.0%}, deep tail {'on' if deep else 'off'}, "
                        f"hosts {hosts}",
            "real_share": share, "init": init, "hosts": hosts, "deep": deep, "filters": {}}
    if not deep:
        spec["filters"]["injected_batman"] = "not_deep"
    if hosts == "tess":
        spec["host_sources"] = TESS_SOURCES
    if hosts == "kdeg":
        spec["kepler_degrade"] = True
    return name, spec


WAVE2_ARMS: dict[str, dict] = dict(
    [_wave2_arm("pt_mix", share, deep, hosts)
     for share in WAVE2_REAL_SHARES for deep in (False, True) for hosts in ("all", "tess", "kdeg")]
    + [_wave2_arm(None, share, deep, "all") for share in (0.0, 0.16) for deep in (False, True)]
    + [_wave2_arm(init, share, True, "all") for init in ("pt_ffi", "pt_tess") for share in (0.0, 0.16)]
)
WAVE2_REFERENCE = "pt_mix-r16-nodeep-all"       # the v2 mix, pretrained
ARMS = WAVE2_ARMS
REFERENCE_ARM = WAVE2_REFERENCE

# Kepler noise degradation (hosts "kdeg"): share of Kepler rows degraded per batch, and the lowest
# combined SNR a positive row may be degraded to
KEPLER_DEGRADE_SHARE = 0.5
KEPLER_DEGRADE_MIN_SNR = 3.0

SEEDS = (0, 1, 2)

# ----------------------------------------------------------------------------- model and training
INPUT_CHANNELS = ("flux_z", "cen_x", "cen_y", "bkg_z")   # plus the observed mask as a fifth channel
INPUT_CLIP = 20.0
INPUT_SCALE = 4.0
# Wave 2 onward: flux_z is compressed as 10 * asinh(flux_z / 10) (build guide, Step 5), the same
# transform pretraining uses; the other channels keep the clip. Wave 1 clipped flux too.
FLUX_ASINH = 10.0
STEM_CHANNELS = 32
D_MODEL = 96
N_HEADS = 4
N_LAYERS = 4
FF_DIM = 192
DROPOUT = 0.1

# Every arm takes the same number of optimizer steps, so arms of different sizes get the same
# compute (baseline: 8 passes over its 80,000 rows; rows_160k: 4; rows_20k: 32).
TRAIN_STEPS = 10_000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 300
POS_WEIGHT = 5.0               # positive patches are about 1% of labeled patches
TORCH_THREADS = 4              # GitHub's public-repo runners have 4 vCPUs

# ----------------------------------------------------------------------------- scoring
# Thresholds are set on lc-eval val negative_random windows so that this share of them
# has any patch above threshold. The first is the headline operating point.
FALSE_ALARM_RATES = (0.01, 0.001)
TARGET_DILATION = 1            # a transit counts as found if a patch within 1 of its mid-time patch fires
COMBINED_SNR_BINS = (0, 4, 7, 10, 15, 25, 1e9)
PER_TRANSIT_SNR_BINS = (0, 2, 4, 7, 15, 1e9)
PERIOD_BINS_D = (0, 3, 10, 27, 1e9)
FAINT_RANGE = (4.0, 10.0)      # combined SNR range of the "faint" headline number
BOX_DURATIONS_SLOTS = (2, 4, 6, 10, 16, 26)   # 1 to 13 hours
BOX_DETREND_SLOTS = 49         # running-median window of the box-filter baseline (about 1 day)
