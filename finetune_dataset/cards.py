"""Dataset cards (README.md) for lc-finetune, lc-eval and lc-benchmarks.

Called by assemble.py after the checks; every number in a card comes from the
build itself (summary.json), not from this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config

CATEGORY_TEXT = {
    "real_planet_short": "Real transits of known planets with P < 20 d (TOI PC/APC/CP/KP, Kepler DR24 PC)",
    "real_planet_long": "Real transits with P >= 20 d or unknown period (Kepler DR24 PC in lc-finetune; long-period TOIs, Hawthorn et al. 2024 duotransits and Kepler PCs in lc-eval)",
    "injected_batman": "Batman transit model multiplied into a star with no catalog entry",
    "injected_transplant": "Measured transit profile of a real known planet multiplied into a star with no catalog entry",
    "negative_random": "Window of a star with no catalog entry, unchanged (no transit)",
    "negative_eb": "Eclipsing binary or false-positive event (TESS EB, Kepler EB, TOI FP/FA, DR24 AFP/NTP); labeled negative",
}

LABELS_TEXT = f"""| Value | `patch_label` meaning |
|---|---|
| 0 | No planet transit has its mid-time in this 4-hour patch |
| 1 | A planet transit (real or injected) has its mid-time here and at least {config.MIN_VISIBLE_FRACTION:.0%} of its in-transit slots are observed |
| 255 | Ignore in the loss: a planet transit is mostly in a data gap, or a known event of unknown nature |

`patch_kind` (uint8 per patch) says what overlaps the patch: 0 none, 1 real planet, 2 batman injection,
3 transplanted real transit, 4 eclipsing binary, 5 false positive (TOI FP/FA, DR24 AFP/NTP), 6 known event of
unknown disposition. `patch_overlap` is the share of the patch inside a planet transit (uint8, divide by 255)
and `patch_center_offset` the mid-time position inside a labeled patch (float16, 0-1, NaN elsewhere)."""

COLUMNS_TEXT = """| Column | Meaning |
|---|---|
| `sample_id`, `category`, `split` | Identity of the sample |
| `star_key`, `source`, `segment_id`, `sector`, `quarter` | Star (`tic:<id>` or `kic:<id>`), data source (ffi, 2min, kepler) and segment |
| `window_slot0`, `t_start_btjd`, `start_offset` | Window start on the global 30-minute grid, in BTJD, and inside its segment |
| `source_repo`, `source_path`, `source_row` | The lc-pretrain or lc-heldout row the window was cut from |
| `sigma_p2p_ppm`, `tmag`, `kepmag`, `teff`, `radius`, `quality_tier`, `coverage` | Noise, star parameters, quality tier, observed fraction of the window |
| `flux_z`, `flux_err_z`, `cen_x`, `cen_y`, `bkg_z` | float16 channels, 1,296 slots (little-endian bytes) |
| `fill`, `flags` | uint8 channels, 1,296 slots (bit 0 of `flags` = observed) |
| `patch_label`, `patch_kind`, `patch_overlap`, `patch_center_offset` | Per-patch targets, 162 patches |
| `n_transit_patches`, `has_transit` | Count of patches with label 1 |
| `events` | Every event in the window: center and duration in slots, kind, catalog reference, class, period, measured SNR, observed fraction, visible, is_planet |
| `single_transit_crop` | True when all but one transit were hidden as a data gap |
| `inj_*` | Injection parameters (model, period, epoch, Rp/R*, a/R*, b, limb darkening, duration, depth, intended and achieved SNR, template id, depth scale) |
| `benchmark_tce_id` | lc-benchmarks only: the published signal this window belongs to |"""

USAGE = """```python
import pandas as pd
from huggingface_hub import hf_hub_download
from finetune_dataset.schema import decode_sample   # github.com/Connor-Codess/MlClub_PlanetFinders

path = hf_hub_download("{repo}", "{example}", repo_type="dataset")
rows = pd.read_parquet(path)
sample = decode_sample(rows.iloc[0])
sample["flux_z"].shape, sample["patch_label"].shape   # (1296,), (162,)
```

`flux_z` is `(flux / median - 1) / sigma_p2p`. Mask slots where `observed` is False, and drop patches whose
label is 255 from the loss. The flux was not detrended; stellar variability is part of the task."""


def rows_table(measured: pd.DataFrame, dataset: str) -> str:
    sub = measured[measured["dataset"] == dataset]
    counts = sub.groupby(["category", "split"]).size().unstack(fill_value=0)
    splits = list(counts.columns)
    lines = ["| Category | " + " | ".join(splits) + " | What it is |", "|---|" + "---:|" * len(splits) + "---|"]
    for category, row in counts.iterrows():
        values = " | ".join(f"{int(row[s]):,}" for s in splits)
        lines.append(f"| `{category}` | {values} | {CATEGORY_TEXT.get(category, '')} |")
    total = " | ".join(f"**{int(counts[s].sum()):,}**" for s in splits)
    lines.append(f"| **Total** | {total} | |")
    return "\n".join(lines)


def source_table(measured: pd.DataFrame, dataset: str) -> str:
    sub = measured[measured["dataset"] == dataset]
    counts = sub.groupby(["source", "split"]).size().unstack(fill_value=0)
    splits = list(counts.columns)
    lines = ["| Source | " + " | ".join(splits) + " |", "|---|" + "---:|" * len(splits)]
    for source, row in counts.iterrows():
        lines.append(f"| {source} | " + " | ".join(f"{int(row[s]):,}" for s in splits) + " |")
    return "\n".join(lines)


def gates_text(summary: dict) -> str:
    return "\n".join(f"- {'PASS' if g['ok'] else 'FAIL'} {g['gate']}: {g['detail']}" for g in summary["gates"])


def yaml_header(pretty: str, splits: list[str], extra_tags: tuple[str, ...] = ()) -> str:
    tags = ["astronomy", "exoplanets", "tess", "kepler", "light-curves", "time-series", "transit-detection",
            *extra_tags]
    lines = ["---", "license: other", f"pretty_name: {json.dumps(pretty)}", "tags:"] + [f"- {t}" for t in tags]
    lines += ["configs:", "- config_name: default", "  data_files:"]
    for split in splits:
        lines += [f"  - split: {split}", f"    path: data/{split}/*.parquet"]
    return "\n".join(lines + ["---", ""])


def injection_text() -> str:
    lo, hi = config.INJ_PERIOD_RANGE_D
    return f"""**Batman injections** (Kreidberg 2015). Period log-uniform in {lo:g}-{hi:g} d (about half the draws
exceed 27 d, so most of those windows hold one transit); impact parameter uniform in 0-{config.INJ_IMPACT_MAX};
quadratic limb darkening drawn uniformly over the physical range (Kipping 2013); circular orbit; a/R* from the
period and the host's TIC radius with a main-sequence mass (stars above {config.HOST_MAX_RADIUS_RSUN:g} R_sun are
not used as hosts); integrated over the 30-minute slot ({config.INJ_SUPERSAMPLE}x supersampling). The depth is set
from a target per-transit SNR drawn log-uniformly (lc-finetune {config.INJ_SNR_RANGE['finetune'][0]:g}-{config.INJ_SNR_RANGE['finetune'][1]:g},
lc-eval {config.INJ_SNR_RANGE['eval'][0]:g}-{config.INJ_SNR_RANGE['eval'][1]:g}), so the set covers the detection threshold evenly.
One transit is always placed on observed data. The model multiplies relative flux:
`flux_z' = ((1 + s * flux_z) * m - 1) / s` with `s = sigma_p2p_ppm * 1e-6`.

**Transplanted real transits.** For known planets on train-split stars (lc-finetune) or validation-split stars
(lc-eval), every transit in our light curves is cut out, divided by a straight-line fit to its out-of-transit
slots, and all transits of the planet are stacked in 10-minute bins. A planet becomes a template when it has at
least {config.TEMPLATE_MIN_TRANSITS} transits, {config.TEMPLATE_MIN_BIN_FILL:.0%} of its in-transit bins have data and the folded depth is at least
{config.TEMPLATE_MIN_SNR:g} times the out-of-transit scatter; beyond duration/2 + {config.TEMPLATE_EDGE_HOURS} h the profile is exactly 1. The
template is placed at that planet's own period and multiplied into the host. Its depth is kept, or scaled down
(`inj_depth_scale` < 1) when needed to reach the target SNR, never scaled up. Kepler templates keep the Kepler
band's limb darkening."""


def finetune_card(summary: dict, measured: pd.DataFrame) -> str:
    return yaml_header("lc-finetune: labeled 27-day TESS and Kepler windows for transit detection",
                       ["train", "validation"]) + f"""
# lc-finetune {config.BUILD_VERSION}

Fine-tuning data for a per-patch transit detector. Each row is one 27-day window (1,296 slots of 30 minutes)
of a real TESS or Kepler light curve, with a target for each of its 162 four-hour patches. The windows are cut
from [lc-pretrain](https://huggingface.co/datasets/{config.ORG}/lc-pretrain), so channels, normalization and
the time grid are identical to pretraining.

Stars: only the `train` (hash bucket 0-849) and `validation` (850-899) splits of lc-pretrain. Held-out stars
(search holdout, scoreboard hosts, long-period TOIs and Hawthorn et al. 2024 duotransit hosts) never appear.
Every star of the published benchmark test sets in
[lc-benchmarks](https://huggingface.co/datasets/{config.BENCHMARK_REPO}) is removed too (including its
transit templates), so comparisons with those papers are fair.

## Contents

{rows_table(measured, "finetune")}

By source:

{source_table(measured, "finetune")}

Real long-period transits come from Kepler: every TESS planet candidate with P >= 20 d is held out for testing.
Real windows need a visible transit (per-transit SNR >= {config.MIN_REAL_SNR_FINETUNE:g}); at most {config.REAL_WINDOWS_PER_STAR} windows per star, 60%
TESS where available. In {config.SINGLE_TRANSIT_CROP_SHARE:.0%} of `real_planet_short` rows with two or more transits, all but one transit are
hidden as a data gap (`single_transit_crop`), so the model does not learn to expect repeats. Injection hosts
and random negatives are 60% FFI, 20% 2-min and 20% Kepler, at most {config.HOST_WINDOWS_PER_STAR} windows per star and category.

## Labels

{LABELS_TEXT}

## How the transits were made

{injection_text()}

## Columns

{COLUMNS_TEXT}

## Build checks

{gates_text(summary)}

## Usage

{USAGE.format(repo=config.FINETUNE_REPO, example="data/train/part-00000-of-" + _n_parts(measured, "finetune", "train") + ".parquet")}

Rows are shuffled across files, so reading a split in file order is already random.
`templates.parquet` holds the transplant templates used here.

Code: [finetune_dataset](https://github.com/Connor-Codess/MlClub_PlanetFinders/tree/main/finetune_dataset).

## Sources and acknowledgments

TESS and Kepler light curves from MAST (TESS-SPOC FFI, SPOC 2-minute, Kepler long cadence). Labels from the TOI
catalog (ExoFOP-TESS, Guerrero et al. 2021), Prsa et al. 2022 (TESS EBs), Kirk et al. 2016 (Kepler EBs), the
Kepler Q1-Q17 DR24 Autovetter training set (Catanzarite 2015) and Hawthorn et al. 2024. Injections use batman
(Kreidberg 2015, PASP 127, 1161) and the limb-darkening sampling of Kipping 2013 (MNRAS 435, 2152).
"""


def eval_card(summary: dict, measured: pd.DataFrame) -> str:
    return yaml_header("lc-eval: held-out test windows for transit detection (private)",
                       ["val", "test"]) + f"""
# lc-eval {config.BUILD_VERSION} (private)

Validation and test windows for the transit detector, in the same row format as
[lc-finetune](https://huggingface.co/datasets/{config.FINETUNE_REPO}). Every star here is a held-out star of
lc-pretrain (it lives in the private lc-heldout repo), so no model has seen it in pretraining or fine-tuning.

- `val`: scoreboard_hosts stars in the first SHA-256("scoreboard:" + star_key) half, and list-excluded stars
  assigned to `real_validation`. Use it to compare model and data variants.
- `test`: the other scoreboard half and the `real_test` list-excluded stars. Use it once, at the end.

## Contents

{rows_table(measured, "eval")}

By source:

{source_table(measured, "eval")}

Real categories hold every window with a planet transit at least half observed, with no SNR cut (at most
{config.EVAL_REAL_WINDOWS_PER_STAR} windows per star); `events[].snr` gives the measured SNR so recall can be binned. Injections and
random negatives use FFI windows only, since FFI is what the search runs on. Transplant templates come from
validation-split planets, which lc-finetune does not use as templates.

## Suggested metrics

- Recall of injected transits by per-transit SNR (`inj_snr`) and period (`inj_period_d`): the injection-recovery
  curve that ranks model variants.
- False alarms per window on `negative_random`, and on `negative_eb` (eclipses that must not be called planets).
- Recall on `real_planet_long` (the science target: single transits and duotransits) and `real_planet_short`.

## Labels

{LABELS_TEXT}

## How the transits were made

{injection_text()}

## Columns

{COLUMNS_TEXT}

## Build checks

{gates_text(summary)}

## Usage

{USAGE.format(repo=config.EVAL_REPO, example="data/val/part-00000-of-" + _n_parts(measured, "eval", "val") + ".parquet")}

Code: [finetune_dataset](https://github.com/Connor-Codess/MlClub_PlanetFinders/tree/main/finetune_dataset).
"""


def benchmark_card(summary: dict, measured: pd.DataFrame, coverage: pd.DataFrame, provenance: dict) -> str:
    names = sorted(provenance)
    sections = []
    for name in names:
        info = provenance[name]
        cov = coverage[coverage["benchmark"] == name]
        cov_lines = "\n".join(f"| {r.label} | {int(r.signals):,} | {int(r.with_windows):,} |" for r in cov.itertuples())
        metrics = "\n".join(f"| {k} | {v} |" for k, v in info["metrics"].items())
        sections.append(f"""### `{name}`: {info['paper']}

Data: {info['mission']}. The paper's positive class: {info['positive']}. Source: `{info['source']}`.

| Label | Signals | With our windows |
|---|---:|---:|
{cov_lines}

Published results on their test set:

| Metric | Value |
|---|---|
{metrics}
""")
    splits = names
    return yaml_header("lc-benchmarks: published exoplanet ML test sets in lc-finetune format (private)", splits,
                       ("benchmark",)) + f"""
# lc-benchmarks {config.BUILD_VERSION} (private)

The test sets of published exoplanet machine-learning papers, rebuilt on our own light curves, so our detector
can be scored on the same signals. Every star listed in `benchmark_targets.parquet` is excluded from
[lc-finetune](https://huggingface.co/datasets/{config.FINETUNE_REPO}). These stars are still in lc-pretrain,
which has no labels; to keep them out of pretraining as well, pass their star keys to the loader's exclusion
list.

One split per benchmark. Each row is one of our 27-day windows (lc-finetune format) containing at least one
predicted transit of a benchmark signal, at most {config.BENCH_WINDOWS_PER_SIGNAL} windows per signal; `benchmark_tce_id` links it to
`benchmark_targets.parquet` and `category` holds the paper's label. Kepler signals use our Kepler light curves;
TESS signals use our TESS-SPOC FFI light curves (the papers used QLP light curves except Salinas et al.), so
signals on stars that TESS-SPOC did not process have no windows; the coverage tables give the counts.

## How to compare

These papers classify a signal given its period and epoch; our model scores patches of raw windows. Score a
signal as the mean, over all of its visible predicted transits in its windows, of the model's transit
probability at the patch containing the transit mid-time. Then compute the paper's metrics on the signals that
have windows, with the paper's positive class (`positive_paper` in `benchmark_targets.parquet`;
`positive_planet` gives planet-only labels where the paper has them). Report the covered fraction next to every
number: this is a close comparison, not an identical one.

{chr(10).join(sections)}

## Build checks

{gates_text(summary)}

## Usage

{USAGE.format(repo=config.BENCHMARK_REPO, example="data/astronet_dr24/part-00000-of-" + _n_parts(measured, "benchmark", "astronet_dr24") + ".parquet")}

The signal table: `pd.read_parquet(hf_hub_download("{config.BENCHMARK_REPO}", "benchmark_targets.parquet", repo_type="dataset"))`,
one row per signal with its ephemeris, the paper's label, `positive_paper` and `positive_planet`; join on
`benchmark_tce_id` = `tce_id`.

Code: [finetune_dataset/benchmarks.py](https://github.com/Connor-Codess/MlClub_PlanetFinders/blob/main/finetune_dataset/benchmarks.py).
Please cite the original papers when using these test sets.
"""


def _n_parts(measured: pd.DataFrame, dataset: str, split: str) -> str:
    n = int(((measured["dataset"] == dataset) & (measured["split"] == split)).sum())
    return f"{max(1, -(-n // 10_000)):05d}"


def write_cards(out: Path, summary: dict, measured: pd.DataFrame, coverage: pd.DataFrame, provenance: dict) -> None:
    cards = {"finetune": finetune_card(summary, measured), "eval": eval_card(summary, measured),
             "benchmark": benchmark_card(summary, measured, coverage, provenance)}
    for dataset, text in cards.items():
        if (out / dataset).exists():
            (out / dataset / "README.md").write_text(text)
