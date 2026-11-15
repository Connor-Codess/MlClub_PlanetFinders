"""Write and upload the dataset cards for lc-pretrain and lc-heldout.

Reads index/summary.json and index/validation_report.md from the build folder
and uploads README.md to both repositories (needs HF_TOKEN).

Run
    HF_TOKEN=... python -m pretrain_dataset.publish_cards --index build/index
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

## Sources and credits

Derived from TESS-SPOC FFI light curves (Caldwell et al. 2020, doi:10.17909/t9-wpz1-8s54), SPOC
2-minute light curves and Kepler long-cadence light curves, all from the Mikulski Archive for Space
Telescopes (MAST). Catalog tags come from
[reference-catalogs](https://huggingface.co/datasets/{config.REFERENCE_REPO}). Funding for the TESS and
Kepler missions is provided by NASA's Science Mission Directorate.
"""


def heldout_card(summary: dict) -> str:
    return f"""---
license: other
pretty_name: lc-heldout (private), held-out stars of lc-pretrain
---

# lc-heldout {config.BUILD_VERSION}

The rows of the lc-pretrain build whose stars must never be pretrained on: `search_holdout`
(searched, used for the seen-versus-unseen check), `scoreboard_hosts` (hosts for injected validation and
test transits) and `list_excluded` (Hawthorn et al. 2024 duotransit hosts and TOIs with P > 20 d or no
period, divided into `real_validation` and `real_test` in `index/exclusions.parquet`).

Same row format as [lc-pretrain](https://huggingface.co/datasets/{config.PRETRAIN_REPO}).
`index/windows_heldout.parquet` lists every default window.

```json
{json.dumps({k: summary.get(k) for k in ("segments_by_split",)}, indent=2)}
```
"""


def main() -> None:
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description="Upload dataset cards")
    parser.add_argument("--index", type=Path, default=Path("build/index"))
    args = parser.parse_args()
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
