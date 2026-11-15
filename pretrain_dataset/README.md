# pretrain_dataset

Builds the masked-reconstruction pretraining dataset described in
`docs/pretraining_dataset_build_guide.md` (Steps 1-11). It produces two Hugging Face datasets:

| Repo | Visibility | Contents |
|---|---|---|
| `nyu-planet-finders/lc-pretrain` | public | `train` and `validation` stars only, plus the window index |
| `nyu-planet-finders/lc-heldout` | private | `search_holdout`, `scoreboard_hosts` and `list_excluded` stars, in the same format |

Held-out stars are physically absent from `lc-pretrain`. The build fails if any star_key
appears in both repos.

## Files

| File | Build guide step | What it does |
|---|---|---|
| `config.py` | all | Every constant and decision: grid, windows, flags, splits, tiers, weights, repo names |
| `star_identity.py` | 1, 2 | `star_key`, SHA-256 hash bucket, split assignment, the list-excluded set |
| `catalogs.py` | 2, 7 | Loads `reference-catalogs` at a pinned revision; tags and predicted known events per segment |
| `segments.py` | 3-6 | One light curve to one segment: cleaning, 30-minute grid, noise statistics, 7 stored channels, default windows |
| `build_segments.py` | 1-7 (pass 1) | One source file to finished rows, routed by split; per-file stats JSON; upload |
| `build_index.py` | 8, 9 (pass 2) | Window index, star table, sampling weights and summary for both repos |
| `validate_build.py` | 11 | Hard gates (row accounting, leakage, round trip) and the validation report |
| `publish_cards.py` | 11 | Dataset cards for both repos |
| `plan_build.py` | - | Decides which source files still need pass 1; pins the catalog revision |
| `loader.py` | - | Reference sampler that returns model-ready window arrays |
| `build_request.json` | - | What the GitHub Actions build should process; pushing a change starts it |

## Running the build

The full build runs in GitHub Actions (`.github/workflows/build-pretrain-dataset.yml`), because
it streams about 1.2 TB of source Parquet and writes about 210 GB. It needs the `HF_TOKEN`
repository secret.

1. Edit `build_request.json` (for example, restrict `files` for a trial, or set it back to `null`).
2. Commit and push. The workflow plans the matrix (one job per source file, 20 at a time),
   runs pass 1, then rebuilds the index, runs the checks and publishes the cards.
3. Files that already have `index/pass1/<source>_<file>.json` in `lc-pretrain` are skipped, so
   pushing again resumes a partial build or adds newly uploaded Kepler quarters.

Local runs (small trials):

```bash
pip install -r pretrain_dataset/requirements.txt
python -m pretrain_dataset.build_segments --source ffi --file s0001.parquet --out build/pass1 --max-row-groups 2
python -m pretrain_dataset.build_index --local build/pass1 --out build/index
python -m pretrain_dataset.validate_build --index build/index --stats build/pass1/index/pass1
```

## Row layout (one row per star x sector or quarter)

Stored channels are little-endian byte strings, `n_slots` long, on a global 30-minute grid:
`slot = floor((BTJD + 2200) / (30 min))`, and the segment starts at `slot0`.

| Column | Type | Meaning |
|---|---|---|
| `flux_z`, `flux_err_z` | float16 | `(flux - median) / sigma_p2p`, and the binned error in the same units |
| `cen_x`, `cen_y` | float16 | Centroid offset from the segment median, arcsec |
| `bkg_z` | float16 | Background offset in units of `sigma_p2p` (FFI only; zero elsewhere) |
| `fill` | uint8 | Fraction of the slot covered by native cadences, times 255 |
| `flags` | uint8 | Bit 0 observed, 1 centroid valid, 2 background valid, 3 fill estimated, 4 merged, 5 spike above +8 sigma, 6 within 6 h after a gap of at least 12 h |

Use `segments.decode(row)` to get numpy arrays back. Other columns: identity (`segment_id`,
`star_key`, `tic_id`, `kic_id`, `hash_bucket`, `split`), observation (`source`, `sector`,
`quarter`, `native_cadence_s`, `quality_policy`, `slot0`, `n_slots`, `n_obs`, `coverage`),
noise statistics (`sigma_p2p_ppm`, `sigma_mad_ppm`, `var_ratio`, `sigma_6h_ppm`, `bkg_sigma`),
star parameters (`tmag`, `kepmag`, `teff`, `radius`), default windows (`window_offsets`,
`window_coverage`, `quality_tier`), tags (`toi_ids`, `toi_disposition`, `eb_catalog`,
`kepler_tce_labels`, `hawthorn_candidate`, `known_events`, `dup_group`) and provenance
(`source_repo`, `source_file`, `source_row_group`, `source_row`, `build_version`).

Tags and `known_events` are for exclusions, sampling, ablations and evaluation. They are never
model inputs or reconstruction targets; `loader.py` does not return them.

## Decisions taken from the build guide

All choices follow the guide's recommended options. One change from the guide is in Step 10. The
guide's separate global re-shuffle pass was dropped. Pass 1 writes each row once and visits source
row groups in a seeded random order. Sampling order comes from the window index, which already
carries the weights and tiers. This halves the upload volume, and the loader samples windows at
random from the index, so it never reads the files in their stored order.
