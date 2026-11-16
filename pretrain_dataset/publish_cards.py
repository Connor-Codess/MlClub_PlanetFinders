"""Write and upload the dataset cards for lc-pretrain and lc-heldout.

Reads index/summary.json and index/validation_report.md from the build folder
and uploads README.md to both repositories (needs HF_TOKEN).

Run
    HF_TOKEN=... python -m pretrain_dataset.publish_cards --index build/index [--from-hub]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import config

CODE_URL = "https://github.com/Connor-Codess/MlClub_PlanetFinders/tree/main/pretrain_dataset"
GUIDE_URL = "https://github.com/Connor-Codess/MlClub_PlanetFinders/blob/main/docs/pretraining_dataset_build_guide.md"

COLUMNS = """| Column | Meaning |
|---|---|
| `segment_id` | `<source>:<file>:<TIC or KIC>`, unique |
| `star_key` | `tic:<TIC>`, or `kic:<KIC>` for Kepler stars without a TIC match |
| `source`, `sector`, `quarter`, `native_cadence_s`, `quality_policy` | where the light curve came from |
| `slot0`, `n_slots`, `n_obs`, `coverage` | position on the global grid: slot = floor((BTJD + 2200) / 30 min) |
| `flux_median`, `sigma_p2p_ppm`, `sigma_mad_ppm`, `var_ratio`, `sigma_6h_ppm`, `bkg_sigma`, `cen_x_median_px`, `cen_y_median_px` | normalization statistics |
| `tmag`, `kepmag`, `teff`, `radius` | star parameters from the source files |
| `flux_z`, `flux_err_z`, `cen_x`, `cen_y`, `bkg_z` | little-endian float16 byte arrays, length n_slots |
| `fill`, `flags` | uint8 byte arrays, length n_slots |
| `window_offsets`, `window_coverage` | default 1,296-slot windows |
| `quality_tier` | 1 clean, 2 usable, 3 poor |
| `hash_bucket`, `split` | SHA-256(star_key) mod 1000 and the split it implies |
| `toi_ids`, `toi_disposition`, `eb_catalog`, `kepler_tce_labels`, `hawthorn_candidate`, `known_events` | catalog tags: never model inputs |
| `dup_group` | same TESS star and sector in FFI and 2-min form |
| `source_repo`, `source_file`, `source_row_group`, `source_row`, `build_version` | provenance |

flags bits: 0 observed, 1 centroid valid, 2 background valid, 3 fill estimated (2-min), 4 merged bins,
5 isolated spike above +8 sigma, 6 within 6 h after a gap of 12 h or more.

Decode a row:

```python
import numpy as np
flux_z = np.frombuffer(row["flux_z"], dtype="<f2").astype(np.float32)
flags = np.frombuffer(row["flags"], dtype=np.uint8)
observed = (flags & 1) > 0
flux = flux_z * row["sigma_p2p_ppm"] * 1e-6 * row["flux_median"] + row["flux_median"]
```"""

RELATED = """## Related datasets

| Dataset | What it is |
|---|---|
| [tess-spoc-ffi-binned](https://huggingface.co/datasets/nyu-planet-finders/tess-spoc-ffi-binned), [tess-spoc-2min-binned](https://huggingface.co/datasets/nyu-planet-finders/tess-spoc-2min-binned), [kepler-llc](https://huggingface.co/datasets/nyu-planet-finders/kepler-llc) | The source light curves |
| [reference-catalogs](https://huggingface.co/datasets/nyu-planet-finders/reference-catalogs) | TOIs, eclipsing binaries, Hawthorn et al. 2024 duotransits, KIC-TIC crossmatch used for tags and exclusions |
| [lc-pretrain](https://huggingface.co/datasets/nyu-planet-finders/lc-pretrain) | Pretraining segments, train and validation stars |
| lc-heldout (private) | The same format for held-out stars |
| [lc-finetune](https://huggingface.co/datasets/nyu-planet-finders/lc-finetune) | Labeled 27-day windows for fine-tuning (real, injected and transplanted transits, negatives) |
| lc-eval (private) | Labeled validation and test windows from held-out stars only |
| lc-benchmarks (private) | Test sets of published papers (AstroNet, Astronet-Triage, Astronet-Triage-v2, Salinas et al. 2025) on our light curves |
"""

USAGE = """## Usage

```python
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from pretrain_dataset.segments import decode   # github.com/Connor-Codess/MlClub_PlanetFinders

path = hf_hub_download("{repo}", "{example}", repo_type="dataset")
row = pq.ParquetFile(path).read_row_group(0).slice(0, 1).to_pylist()[0]
arrays = decode(row)            # flux_z, flux_err_z, cen_x, cen_y, bkg_z, fill (float32), flags (uint8)
observed = (arrays["flags"] & 1) > 0
```

The byte columns show as base64 text in the Hugging Face viewer; decode them as above. Read one row
group at a time: an FFI file holds about 135,000 rows. `pretrain_dataset/loader.py` samples weighted
27-day windows directly from the index.
"""


def pretrain_card(summary: dict, report: str) -> str:
    configs = []
    for source in ("ffi", "2min", "kepler"):
        configs += [f"- config_name: {source}", "  data_files:"]
        for split in config.PRETRAIN_SPLITS:
            configs += [f"  - split: {split}", f"    path: data/{source}/{split}/*.parquet"]
    configs += ["- config_name: windows", "  data_files:",
                "  - split: train", "    path: index/windows_train.parquet",
                "  - split: validation", "    path: index/windows_validation.parquet"]
    gates = "\n".join(line for line in report.splitlines() if line.startswith("- PASS") or line.startswith("- FAIL"))
    return f"""---
license: other
pretty_name: lc-pretrain, gridded TESS and Kepler light curves for transformer pretraining
tags:
- astronomy
- exoplanets
- tess
- kepler
- light-curves
- time-series
- self-supervised
configs:
{chr(10).join(configs)}
---

# lc-pretrain {config.BUILD_VERSION}

Every TESS-SPOC FFI, SPOC 2-minute and Kepler long-cadence light curve from the nyu-planet-finders
datasets, cleaned, placed on one global 30-minute grid, normalized by point-to-point noise and stored as
segments (one star in one sector or quarter). Built for self-supervised pretraining of light-curve
transformers. Design and every decision: [build guide]({GUIDE_URL}). Code: [{CODE_URL}]({CODE_URL}).

Only the **train** and **validation** splits are here. Stars used for testing (search holdout,
scoreboard hosts, Hawthorn et al. 2024 duotransit hosts and long-period TOIs) are in a separate private
repository and are absent from this one.

## Layout

- `data/<source>/<split>/<file>.parquet`, one file per source sector or quarter
- `index/windows_<split>.parquet`: one row per default 27-day window with `file`, `row`,
  `start_offset` and sampling weights (`w_default` gives 50/25/25 FFI/2-min/Kepler, at most 20 windows
  per star, variable stars doubled, poor-quality segments halved)
- `index/stars.parquet`, `index/summary.json`, `index/validation_report.md`, `index/pass1/*.json`

## Build checks

{gates}

## Columns

{COLUMNS}

## Counts

```json
{json.dumps({k: summary.get(k) for k in ("segments_by_split", "windows_by_split", "stars", "ffi_2min_duplicate_pairs")}, indent=2)}
```

{USAGE.format(repo=config.PRETRAIN_REPO, example="data/ffi/train/s0040.parquet")}
{RELATED}
## Sources and credits

Derived from TESS-SPOC FFI light curves (Caldwell et al. 2020, doi:10.17909/t9-wpz1-8s54), SPOC
2-minute light curves and Kepler long-cadence light curves, all from the Mikulski Archive for Space
Telescopes (MAST). Catalog tags come from
[reference-catalogs](https://huggingface.co/datasets/{config.REFERENCE_REPO}). Funding for the TESS and
Kepler missions is provided by NASA's Science Mission Directorate.
"""


def heldout_card(summary: dict) -> str:
    held = ("search_holdout", "scoreboard_hosts", "list_excluded")
    by_split = summary.get("segments_by_split", {})
    lines = ["| Split | " + " | ".join(by_split) + " | Total |", "|---|" + "---:|" * (len(by_split) + 1)]
    for split in held:
        values = [int(by_split[source].get(split, 0)) for source in by_split]
        lines.append(f"| `{split}` | " + " | ".join(f"{v:,}" for v in values) + f" | {sum(values):,} |")
    configs = []
    for source in ("ffi", "2min", "kepler"):
        configs += [f"- config_name: {source}", "  data_files:", "  - split: heldout",
                    f"    path: data/{source}/*.parquet"]
    return f"""---
license: other
pretty_name: lc-heldout (private), held-out stars of lc-pretrain
tags:
- astronomy
- exoplanets
- tess
- kepler
- light-curves
configs:
{chr(10).join(configs)}
---

# lc-heldout {config.BUILD_VERSION} (private)

The segments of the lc-pretrain build whose stars must never be pretrained or fine-tuned on. They are kept in
this separate private repository, so a pretraining or fine-tuning job cannot read them by accident; the build
checks that no star appears in both repositories.

Row format, channels and normalization are identical to
[lc-pretrain](https://huggingface.co/datasets/{config.PRETRAIN_REPO}). The `split` column says why a star is
held out.

| Split | How stars are chosen | Used for |
|---|---|---|
| `search_holdout` | SHA-256(star_key) mod 1000 in 900-949 (5% of stars) | Searched like every star, never trained on: compares results on seen and unseen stars to detect memorization |
| `scoreboard_hosts` | Hash bucket 950-999 (5%) | Hosts for injection-recovery validation and test windows (lc-eval), halved by SHA-256("scoreboard:" + star_key) |
| `list_excluded` | By name: Hawthorn et al. 2024 duotransit hosts and TOIs (PC/APC/CP/KP) with P > 20 d or no period | Real long-period benchmark, divided into `real_validation` and `real_test` in `index/exclusions.parquet` |

## Contents (segments: one star in one sector or quarter)

{chr(10).join(lines)}

## Layout

- `data/<source>/<file>.parquet`: one file per source sector or quarter, all three splits together
- `index/windows_heldout.parquet`: every default 27-day window (`file`, `row`, `start_offset`, `coverage`)
- `index/exclusions.parquet`: `star_key`, `reason` and `scoreboard_set` (real_validation or real_test) of
  every list-excluded star
- `index/summary.json`, `index/pass1/*.json`: counts and per-file build statistics

The catalog tags (`known_events`, `toi_ids`, `eb_catalog`, `hawthorn_candidate`) are filled in here too. They
mark where the real transits are when scoring a model on `list_excluded`, and which stars to avoid when
injecting into `scoreboard_hosts`.

## Columns

{COLUMNS}

{USAGE.format(repo=config.HELDOUT_REPO, example="data/ffi/s0040.parquet")}
Reading this repository needs a token with access to the nyu-planet-finders organization.

{RELATED}"""


def main() -> None:
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description="Upload dataset cards")
    parser.add_argument("--index", type=Path, default=Path("build/index"))
    parser.add_argument("--from-hub", action="store_true",
                        help="download summary.json and validation_report.md from lc-pretrain first")
    args = parser.parse_args()
    if args.from_hub:
        from huggingface_hub import hf_hub_download

        args.index.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "validation_report.md"):
            local = hf_hub_download(config.PRETRAIN_REPO, f"index/{name}", repo_type="dataset",
                                    token=os.environ.get("HF_TOKEN"))
            (args.index / name).write_bytes(Path(local).read_bytes())
    summary = json.loads((args.index / "summary.json").read_text())
    report_path = args.index / "validation_report.md"
    report = report_path.read_text() if report_path.exists() else ""
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_file(path_or_fileobj=pretrain_card(summary, report).encode(), path_in_repo="README.md",
                    repo_id=config.PRETRAIN_REPO, repo_type="dataset", commit_message="Dataset card")
    if report:
        api.upload_file(path_or_fileobj=str(report_path), path_in_repo="index/validation_report.md",
                        repo_id=config.PRETRAIN_REPO, repo_type="dataset", commit_message="Validation report")
    api.upload_file(path_or_fileobj=heldout_card(summary).encode(), path_in_repo="README.md",
                    repo_id=config.HELDOUT_REPO, repo_type="dataset", commit_message="Dataset card")


if __name__ == "__main__":
    main()
