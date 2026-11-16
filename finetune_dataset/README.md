# finetune_dataset

Builds the labeled datasets for fine-tuning and testing the transit detector, from the segments of
`lc-pretrain` and `lc-heldout`:

| Repo | Visibility | Stars | Contents |
|---|---|---|---|
| `nyu-planet-finders/lc-finetune` | public | train and validation splits, minus every benchmark star | `train` and `validation` |
| `nyu-planet-finders/lc-eval` | private | held-out only (scoreboard hosts, long-period TOIs, Hawthorn duotransit hosts) | `val` and `test` |
| `nyu-planet-finders/lc-benchmarks` | private | stars of published test sets | one split per paper |

Each row is one 27-day window (1,296 slots of 30 minutes) with a target for each of its 162 four-hour patches.
Channels and normalization are those of lc-pretrain.

## Categories

| Category | What it is |
|---|---|
| `real_planet_short` | Real transits of known planets with P < 20 d (TOI PC/APC/CP/KP, Kepler DR24 PC) |
| `real_planet_long` | Real transits with P >= 20 d or unknown period. In lc-finetune these are Kepler PCs, because every TESS candidate with P >= 20 d is held out |
| `injected_batman` | A batman transit multiplied into a star with no catalog entry |
| `injected_transplant` | The measured transit of a real known planet, multiplied into a star with no catalog entry |
| `negative_random` | An unchanged window of a star with no catalog entry |
| `negative_eb` | Eclipsing binaries and false-positive TOIs/TCEs, labeled negative |

lc-finetune mix: 15% real short, 7.5% real long, 35% batman, 15% transplant, 17.5% random negatives,
10% EB negatives; 400,000 train and 24,000 validation rows. Any category that cannot fill its share is
topped up with batman injections. Every number is in `config.py`.

## Published benchmarks

`benchmarks.py` rebuilds the test sets of AstroNet (Shallue & Vanderburg 2018, Kepler DR24), Astronet-Triage
(Yu et al. 2019, TESS sectors 1-5) and Astronet-Triage-v2 (Tey et al. 2023, TESS cycles 1-3), plus the
published candidates of Salinas et al. 2025, whose test set was not released. Their stars are removed from
lc-finetune. The lc-benchmarks card explains how to score our model on them.

## Files

| File | Stage | What it does |
|---|---|---|
| `config.py` | all | Sizes, mix, caps, injection distributions, label codes |
| `pools.py` | all | Star pools (which dataset a star may feed) and known-event classes |
| `injection.py` | 3 | Window cutting, batman and transplant injection, event measurement, patch labels |
| `templates.py` | 1, 2 | Transit cutouts and stacked templates for transplants |
| `benchmarks.py` | 0 | Published test sets as target tables |
| `plan_build.py` | 0 | Files to process and the pinned catalog revision |
| `scan.py` | 1 | Per source file: candidate windows, hosts, cutouts, benchmark windows |
| `choose_samples.py` | 2 | Picks every sample, builds templates, writes recipes |
| `materialize.py` | 3 | Per source file: turns recipes into labeled rows |
| `assemble.py` | 4 | Shuffles, runs the build checks, writes cards, uploads |
| `cards.py` | 4 | Dataset cards |
| `schema.py` | - | Row layout and `decode_sample()` |
| `build_request.json` | - | What the workflow builds; pushing a change starts it |

## Running

The build runs in GitHub Actions (`.github/workflows/build-finetune-dataset.yml`) and needs the `HF_TOKEN`
secret. Edit `build_request.json` (`files` and `scale` for a trial, `null` and `1.0` for the full build) and
push.

Local trial against a local pass-1 output:

```bash
pip install -r finetune_dataset/requirements.txt
python -m finetune_dataset.benchmarks --out build/finetune/benchmarks --catalog-revision <sha>
python -m finetune_dataset.scan --source ffi --stem s0001 --out build/finetune/scan --catalog-revision <sha> \
    --benchmarks build/finetune/benchmarks/benchmark_targets.parquet --local-root build/trial
python -m finetune_dataset.choose_samples --scan build/finetune/scan --out build/finetune/plan \
    --benchmarks build/finetune/benchmarks/benchmark_targets.parquet --scale 0.01
python -m finetune_dataset.materialize --plan build/finetune/plan --source ffi --stem s0001 \
    --out build/finetune/rows --local-root build/trial
python -m finetune_dataset.assemble --rows build/finetune/rows --plan build/finetune/plan \
    --benchmarks build/finetune/benchmarks --out build/finetune/final
```

## Reading a row

```python
from finetune_dataset.schema import decode_sample
sample = decode_sample(row)            # flux_z (1296,), observed (1296,), patch_label (162,), ...
```

Drop patches with label 255 from the loss and mask unobserved slots.
