# Transit transformer (Kepler + TESS)

A PatchTST-style transformer is pretrained on Kepler and TESS light-curve windows by filling in hidden patches. It is then fine-tuned to sort transit signals into planet, candidate, false positive or no planet, and tested on TESS stars it has never seen.

## Data (Hugging Face, `nyu-planet-finders`)
- `tess-spoc-2min-binned`: TESS light curves, binned to 30 minutes
- `kepler-raw-lightcurves`: Kepler light curves and Kepler DR24 (Data Release 24) labels
- `kepler-centroids`: Kepler centroid columns, re-fetched from MAST (Mikulski Archive for Space Telescopes) with `fetch_kepler_centroids.py`
- Labels: the NASA Exoplanet Archive TOI (TESS Object of Interest) table and the KOI (Kepler Object of Interest) cumulative table

## Pipeline (run in order)
1. `build_shards.py`: cleans each light curve and cuts it into 28-day windows of 30-minute bins, scaled per window. Uses `preprocess.py` and `splits.py`.
2. `pretrain.py`: masked-patch pretraining (`model.py`, `dataset.py`). The linear-probe check is in `probe.py`.
3. `labels.py`, then `build_examples.py`: builds labeled signals, folded views and scalar features.
4. `finetune.py` (`finetune_model.py`): two-stage fine-tuning. `run_finetunes.sh` runs all four comparison versions.
5. `evaluate.py`: test-set metrics with bootstrap confidence intervals, written to `results/finetune_comparison.md`.

`PRETRAIN_PLAN.md` and `FINETUNE_PLAN.md` give the full design. `runs/` holds the logs, plots and `eval.json` files; the checkpoints and data are not in git.
