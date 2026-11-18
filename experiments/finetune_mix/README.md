# Fine-tuning data mix experiments

Which combination of rows from [lc-finetune](https://huggingface.co/datasets/nyu-planet-finders/lc-finetune)
trains the best transit detector? Each **arm** is one combination. Every arm trains the same small detector
on the same number of rows for the same number of steps, three times with different seeds. Each run is then
scored on the held-out stars of `lc-eval` val. The only thing that changes between arms is the data.

These are proxy experiments. The model is small (0.35M parameters) and trains on CPU. In wave 1 it had no
pretraining; wave 2 adds a pretrained version from `experiments/proxy_pretrain`. They show which data choices matter and in which direction. The best two or three arms should be
rerun once the pretrained model exists.

## Wave 2 (2026-11-18)

Wave 2 trains on lc-finetune v3 and adds pretraining. Each arm still trains 80,000 rows for 10,000 steps
and is scored on the same lc-eval val windows. lc-eval keeps the v2 injection physics on purpose, so the
two waves are comparable. Arm names read `<init>-r<real share>-<deep|nodeep>-<hosts>`.

| Group | Arms | Question |
|---|---|---|
| Grid, pretrained on `pt_mix` | real share 0%, 5%, 16% × deep tail off/on × hosts `all`, `tess`, `kdeg` (18 arms) | How much real data? Does the deep tail fix the depth shortcut? Do Kepler hosts hurt, and does noise degradation fix it? |
| From scratch | real share 0% and 16% × deep tail off/on, hosts `all` (4 arms) | What does pretraining add? |
| Pretraining data | `pt_ffi` and `pt_tess`, real share 0% and 16%, deep tail on, hosts `all` (4 arms) | Which pretraining data works best? |

How the factors are set:
- **Real share:** rows freed from real planets go to batman and transplants in their base ratio.
- **Deep tail:** `nodeep` drops the deep-tail batman rows (combined SNR 60-300, about 10% of batman rows).
- **Hosts:**
  - `tess` takes injected and random-negative rows from TESS only; real planets and EBs keep Kepler.
  - `kdeg` keeps every source but degrades half of the Kepler rows in each batch to the noise of a random FFI
    star (`augment.py`). Rows with a transit keep a combined SNR of at least 3.
- **Reference arm:** `pt_mix-r16-nodeep-all`, the v2 mix, pretrained.
- **Flux input:** flux is now asinh-compressed (`10 asinh(z / 10)`) in every arm, matching pretraining.
  Wave 1 clipped it.

Additions to the scoring:
- **Transit-hiding test** (`evaluate.hiding_test`). On injected windows with two or more visible transits, one
  transit is kept and the rest are hidden. Recall of that transit with and without its siblings shows whether
  the model combines repeated transits.
- **Classical baseline** (`classical.py`): wotan biweight detrending, then a single-event box search plus
  astropy BLS over 0.6-13.5 d, scored like the models. It replaces the box filter as the reference.
- **Realism checks** (`realism.py`):
  - duration and depth of batman injections against TOIs, by period
  - a classifier trying to tell real single transits from batman ones (and from transplants), matched in
    SNR and duration
- **Matched recall** (`analyze.py`): single-transit batman injections with P ≥ 20 d, weighted to the real
  long-period planets' SNR distribution, against the real planets' own recall.

The pretrained encoders come from [`experiments/proxy_pretrain`](../proxy_pretrain/README.md). Set
`pretrain_run_id` in `run_request.json` to that workflow's run. The `wait-pretrain` job holds the
pretrained arms until it has finished; arms from scratch, the classical baseline and the realism checks run
in the meantime.

## Wave 1 arms (2026-11-16, `config.WAVE1_ARMS`)

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
| `evaluate.py` | Thresholds, recall numbers, bins, average precision, box-filter baseline, transit-hiding test |
| `augment.py` | Kepler noise degradation (fine-tuning `kdeg` arms and proxy pretraining) |
| `classical.py` | wotan + box search + BLS baseline, in slices, then merged and scored |
| `realism.py` | Distribution comparison with TOIs and the real-versus-batman classifier |
| `aggregate.py` | Combines all runs into `summary.json`, `summary.md` and `runs.csv` |
| `analyze.py` | Recomputes recall on question-specific subsets from every run's scores (`extra.json`) |
| `results/wave1/` | Wave 1 results and findings (run 44000179513) |
| `run_request.json` | Pushing a change to this file starts the workflow |

## How to run

On GitHub Actions (free for this public repository: 4 vCPUs and 16 GB per job, up to 20 jobs at once):
edit `run_request.json` and push, or start "Fine-tuning data mix experiments" from the Actions tab. The
request can limit the arms (`"arms": ["baseline", "no_real"]`) or change the step count (`"steps": 2000`).
The request can also name the proxy pretraining run (`"pretrain_run_id"`) that pretrained arms load their
encoders from. The workflow needs the `HF_TOKEN` secret, because lc-eval is private. The results are in the run's
`summary` artifact and in one `result-<arm>-s<seed>` artifact per run.

## Results so far

Wave 1 (2026-11-17): the v2 mix is the best-balanced of 13 arms, and its stacked-SNR rule helps faint
short-period transits (20.3% against 14.3% recall). Real planets deeper than SNR 25 are mostly missed,
because training injections stop near SNR 50 while EB negatives are deep. See
[`results/wave1/README.md`](results/wave1/README.md).

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
| 2026-11-17 | Wave 1 finished (run 44000179513); findings in `results/wave1/README.md`. Added `analyze.py` to the summary job |
| 2026-11-18 | Wave 2: 26 arms on lc-finetune v3, pretrained encoders, Kepler degradation, asinh flux input, transit-hiding test, classical baseline, realism checks |
