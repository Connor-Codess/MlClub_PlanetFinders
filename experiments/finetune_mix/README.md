# Fine-tuning data mix experiments

Which combination of rows from [lc-finetune](https://huggingface.co/datasets/nyu-planet-finders/lc-finetune)
trains the best transit detector? Each **arm** is one combination. Every arm trains the same small detector
on the same number of rows for the same number of steps, three times with different seeds. Each run is then
scored on the held-out stars of `lc-eval` val. The only thing that changes between arms is the data.

These are proxy experiments. The model is small (0.35M parameters), trains from scratch on CPU, and has no
pretraining. They show which data choices matter and in which direction. The best two or three arms should be
rerun once the pretrained model exists.

## Arms

All arms start from the category shares of lc-finetune v2 train, scaled to 80,000 rows:

| Category | Share |
|---|---:|
| `real_planet_short` | 13.3% |
| `real_planet_long` | 2.7% |
| `injected_batman` | 41.5% |
| `injected_transplant` | 15.0% |
| `negative_random` | 17.5% |
| `negative_eb` | 10.0% |

| Arm | Change from baseline | Question |
|---|---|---|
| `baseline` | none | Reference |
| `v1_rule` | Real windows need one transit at SNR >= 4; injections need per-transit SNR >= 3 | Does v2's inclusion of faint, stackable transits help? |
| `no_real` | Real planet rows replaced by injections | Are real transits worth their rows? |
| `no_injected` | Injected rows replaced by real planet rows | Are injections worth their rows? |
| `batman_only` | Transplants replaced by batman | Does a real transit shape help? |
| `transplant_only` | Batman replaced by transplants | Does a real transit shape help? |
| `no_eb` | EB negatives replaced by random negatives | How much do EB negatives cut EB false alarms? |
| `no_crop` | No single-transit-cropped real rows | Do crops help single-transit recall? |
| `tess_only` | TESS rows only; long-period real rows (all Kepler) become short-period TESS rows | Does Kepler data help on TESS? |
| `strong_injections` | Injections with combined SNR >= 7 only | Does faint training data help or only add false alarms? |
| `rows_20k`, `rows_40k`, `rows_160k` | Baseline shares at 20k, 40k and 160k rows | Does more data help at fixed compute? |

Rows are taken from each category in one fixed random order, skipping rows that fail an arm's filters. Arms
therefore share as many rows as possible, and the scaling arms are nested.

## Model and training

`model.py` defines the detector:

- Input: one 27-day window as five channels of 1,296 slots: `flux_z`, `cen_x`, `cen_y`, `bkg_z` (clipped
  to ±20 and divided by 4, zero where unobserved) and the observed mask.
- Two 1-D convolutions over slots, then a strided convolution that makes one token per 4-hour patch.
- A 4-layer transformer encoder (d = 96, 4 heads) over the 162 tokens, then one logit per patch.

Training settings:

- AdamW, learning rate 1e-3 with 300 warmup steps and cosine decay, weight decay 0.01.
- Batch 64, 10,000 steps for every arm.
- Patch-level binary cross-entropy, with positive patches weighted 5x.
- Ignored patches: label 255, and patches a transit overlaps without containing its mid-time (their dip is
  real but they are not the labeled patch).

## Scoring

Thresholds are set on lc-eval val `negative_random` windows. At the headline operating point, 1% of random
27-day windows raise an alarm somewhere; the strict point is 0.1%. A transit window counts as found when a
patch within one patch of a transit mid-time is above threshold.

Headline numbers:

- Injection recall: overall, faint (combined SNR 4-10) and strong (>= 10), and separately for batman and
  transplants.
- Recall on real short- and long-period planets of held-out stars.
- The EB false-alarm rate.
- Patch-level average precision.

`metrics.json` also bins recall by combined SNR, per-transit SNR and period for injections, and by stacked
SNR for real planets. A classical single-event box search (`evaluate.box_filter`, BLS-style without
folding) is scored the same way as a reference.

`aggregate.py` reports each arm as the mean ± standard deviation over its seeds. It flags a difference from
baseline when the difference exceeds two standard errors.

## Files

| File | What it does |
|---|---|
| `config.py` | Every number: arms, model size, training budget, bins, false-alarm rates |
| `select.py` | Chooses each arm's rows from lc-finetune train metadata |
| `data.py` | Downloads shards, packs chosen rows into arrays, builds training and scoring targets |
| `pack.py` | Packs one arm, or lc-eval val (and runs the box-filter baseline) |
| `model.py` | The detector |
| `train.py` | Trains one arm and seed, then scores it on lc-eval val |
| `evaluate.py` | Thresholds, recall numbers, bins, average precision, box-filter baseline |
| `aggregate.py` | Combines all runs into `summary.json`, `summary.md` and `runs.csv` |
| `run_request.json` | Pushing a change to this file starts the workflow |

## How to run

On GitHub Actions (free for this public repository: 4 vCPUs and 16 GB per job, up to 20 jobs at once):
edit `run_request.json` and push, or start "Fine-tuning data mix experiments" from the Actions tab. The
request can limit the arms (`"arms": ["baseline", "no_real"]`) or change the step count (`"steps": 2000`).
The workflow needs the `HF_TOKEN` secret, because lc-eval is private. The results are in the run's
`summary` artifact and in one `result-<arm>-s<seed>` artifact per run.

Locally:

```bash
pip install -r experiments/finetune_mix/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m experiments.finetune_mix.select --out build/mix/select
python -m experiments.finetune_mix.pack --arm baseline --selection build/mix/select --out build/mix/arm
HF_TOKEN=... python -m experiments.finetune_mix.pack --eval --out build/mix/eval \
    --box-filter-out build/mix/results/box_filter
python -m experiments.finetune_mix.train --arm-data build/mix/arm --eval build/mix/eval \
    --arm baseline --seed 0 --out build/mix/results/baseline-s0
python -m experiments.finetune_mix.aggregate --results build/mix/results --out build/mix/summary
```

For a quick check on one shard, use `select.py --scale 0.02 --arms baseline` and `train.py --steps 500`.

Tests: `python -m pytest tests/test_finetune_mix.py -q`.

## Rules

- Score on lc-eval val only. The test split is used once, at the end of the project.
- Rank variants on injection recall at fixed false alarms, never on training loss.
- Record changes to `config.py` below, with the run IDs they apply to.

## Change log

| Date | Change |
|---|---|
| 2026-11-16 | First version: 13 arms × 3 seeds, 10,000 steps, lc-eval val |
