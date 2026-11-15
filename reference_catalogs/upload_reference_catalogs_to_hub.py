"""Publish the built reference catalogs to Hugging Face with a dataset card.

What it does
    1. Reads build_manifest.json from the build directory.
    2. Writes README.md (the dataset card): one config per catalog, row
       counts, sources, citations, column descriptions and the git commit
       that produced the files.
    3. Creates the dataset repo if needed and uploads the Parquet files,
       the manifest and the card in a single commit. Only catalogs listed as
       built in the manifest are uploaded.

Authentication
    Needs a Hugging Face token with write access to the nyu-planet-finders
    organization, in the HF_TOKEN environment variable (in GitHub Actions,
    the repository secret HF_TOKEN).

Run
    HF_TOKEN=... python -m reference_catalogs.upload_reference_catalogs_to_hub \\
        --src build/reference_catalogs --repo nyu-planet-finders/reference-catalogs
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .build_all_reference_catalogs import MANIFEST_NAME
from .common import LOG, configure_logging, utc_now

DEFAULT_REPO = "nyu-planet-finders/reference-catalogs"
CODE_URL = "https://github.com/Connor-Codess/MlClub_PlanetFinders/tree/main/reference_catalogs"

DESCRIPTIONS = {
    "toi_ephemerides": {
        "title": "TESS Objects of Interest with ephemerides",
        "source": "NASA Exoplanet Archive, TOI table (TAP service).",
        "citation": "NASA Exoplanet Archive TOI table; Guerrero et al. 2021, ApJS 254, 39.",
        "columns": "toi, tic_id, disposition (APC, CP, FA, FP, KP, PC), period_d (null when unknown), "
                   "t0_btjd (pl_tranmid - 2457000), duration_h, depth_ppm, planet_radius_re, tmag, teff_k, "
                   "star_radius_rsun, ra_deg, dec_deg, sectors, period_known, long_period_or_unknown "
                   "(P > 20 d or no period). Raw archive columns follow with an archive_ prefix.",
    },
    "tess_eb_catalog_prsa2022": {
        "title": "TESS eclipsing binaries, sectors 1-26",
        "source": "VizieR J/ApJS/258/16/tess-ebs.",
        "citation": "Prsa et al. 2022, ApJS 258, 16, doi:10.3847/1538-4365/ac324a.",
        "columns": "tic_id, signal_id, period_d, t0_btjd, morphology (0 detached to 1 ellipsoidal), tmag, "
                   "ra_deg, dec_deg, primary/secondary eclipse width (phase units) and depth, secondary_phase, "
                   "primary_duration_h, sectors. All VizieR columns follow with a vizier_ prefix.",
    },
    "kepler_eb_catalog_kirk2016": {
        "title": "Kepler eclipsing binaries, Q0-Q17",
        "source": "VizieR J/AJ/151/68, main catalog table.",
        "citation": "Kirk et al. 2016, AJ 151, 68, doi:10.3847/0004-6256/151/3/68.",
        "columns": "kic_id, period_d, t0_btjd, t0_reference (the detected epoch convention), morphology, "
                   "kepmag, primary/secondary eclipse width (phase units) and depth, primary_duration_h. "
                   "All VizieR columns follow with a vizier_ prefix.",
    },
    "duotransits_hawthorn2024": {
        "title": "TESS duotransit candidates (Cycles 1 and 3, south)",
        "source": "arXiv:2310.17268v2 HTML, appendix Tables A1-A3, parsed automatically.",
        "citation": "Hawthorn et al. 2024, MNRAS 528, 1841, doi:10.1093/mnras/stad3783.",
        "columns": "candidate_id (1-85), tic_id, toi_or_ctoi, ra_deg, dec_deg, tmag, star_radius_rsun, teff_k, "
                   "n_sectors, tc1_btjd, tc1_sector, tc2_btjd, tc2_sector, depth1_ppt, depth2_ppt, duration_h, "
                   "planet_radius_re, period_min_d, period_max_d, n_alias, period_marg_d, total_snr.",
    },
    "kic_tic_crossmatch": {
        "title": "KIC to TIC crossmatch for the Kepler light-curve set",
        "source": "MAST TIC v8.2 (Mast.Catalogs.Filtered.Tic) queried on its KIC column for every kic_id in "
                  "nyu-planet-finders/kepler-llc manifest.parquet.",
        "citation": "Stassun et al. 2019, AJ 158, 138; Paegert et al. 2021, arXiv:2108.04778.",
        "columns": "kic_id, tic_id (null when unmatched), n_tic_matches, tic_disposition, tic_ra_deg, "
                   "tic_dec_deg, tmag, matched.",
    },
}


def git_commit() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def render_card(manifest: dict, repo_id: str) -> str:
    built = manifest["catalogs"]
    config_lines = []
    for name, info in built.items():
        config_lines += [
            f"- config_name: {name}",
            "  data_files:",
            "  - split: train",
            f"    path: {info['filename']}",
        ]
    rows_table = "\n".join(
        f"| `{info['filename']}` | {DESCRIPTIONS.get(name, {}).get('title', name)} | {info['rows']:,} | {info['retrieved_utc']} |"
        for name, info in built.items()
    )
    sections = []
    for name, info in built.items():
        text = DESCRIPTIONS.get(name, {})
        notes = json.dumps(info.get("notes", {}), indent=2, default=str)
        sections.append(
            f"### {name}\n\n"
            f"{text.get('title', '')}\n\n"
            f"- **Source:** {text.get('source', info['source'])}\n"
            f"- **Cite:** {text.get('citation', '')}\n"
            f"- **Columns:** {text.get('columns', '')}\n\n"
            f"Build notes:\n\n```json\n{notes}\n```\n"
        )
    failed = manifest.get("failed", {})
    failed_text = "".join(f"- `{k}`: {v}\n" for k, v in failed.items()) or "None.\n"

    return f"""---
license: other
pretty_name: Reference catalogs for the TESS and Kepler planet search
tags:
- astronomy
- exoplanets
- tess
- kepler
- eclipsing-binaries
configs:
{chr(10).join(config_lines)}
---

# Reference catalogs for the TESS and Kepler planet search

Small catalog tables used by the nyu-planet-finders pretraining build (Step 0 of the build guide):
TOI ephemerides, eclipsing-binary catalogs for TESS and Kepler, the Hawthorn et al. 2024 duotransit
candidates, and a KIC-to-TIC crossmatch for every Kepler star in
[kepler-llc](https://huggingface.co/datasets/nyu-planet-finders/kepler-llc).

They are used for **tags, exclusions and evaluation only**. None of these labels is ever a model input
or a pretraining target.

| File | Contents | Rows | Retrieved (UTC) |
|---|---|---|---|
{rows_table}

All times are **BTJD = BJD_TDB - 2457000**. Kepler epochs were converted from their catalog convention
(recorded per file). Every Parquet file also stores its provenance as JSON in the file footer under the
key `reference_catalog`.

## Files

{chr(10).join(sections)}
## Not built in this version

{failed_text}
## How to load

```python
import pandas as pd

toi = pd.read_parquet("hf://datasets/{repo_id}/toi_ephemerides.parquet")
xmatch = pd.read_parquet("hf://datasets/{repo_id}/kic_tic_crossmatch.parquet")
```

## How it was built

Code: [{CODE_URL}]({CODE_URL}), commit `{git_commit()}`, card written {utc_now()}.
Rebuild with `python -m reference_catalogs.build_all_reference_catalogs` followed by
`python -m reference_catalogs.upload_reference_catalogs_to_hub`, or run the GitHub Actions workflow
"Build reference catalogs".

## Acknowledgments

This dataset uses the NASA Exoplanet Archive, operated by the California Institute of Technology under
contract with NASA; the VizieR catalogue access tool, CDS, Strasbourg; and the Mikulski Archive for
Space Telescopes (MAST) at the Space Telescope Science Institute. Each table remains subject to its
original source's terms; cite the papers listed above when using it.
"""


def upload(src: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set; cannot upload")
    manifest = json.loads((src / MANIFEST_NAME).read_text())
    if not manifest["catalogs"]:
        raise SystemExit("The manifest lists no built catalogs; nothing to upload")

    (src / "README.md").write_text(render_card(manifest, repo_id))
    allowed = [info["filename"] for info in manifest["catalogs"].values()] + [MANIFEST_NAME, "README.md"]

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    built = ", ".join(manifest["catalogs"])
    commit = api.upload_folder(
        folder_path=str(src),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allowed,
        commit_message=f"Update reference catalogs: {built}",
    )
    LOG.info("Uploaded %s to %s (%s)", allowed, repo_id, commit)
    print(f"::notice title=Hugging Face upload::{repo_id}: {built}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=Path("build/reference_catalogs"))
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--card-only", action="store_true",
                        help="write README.md into --src without uploading")
    args = parser.parse_args()
    configure_logging()
    if args.card_only:
        manifest = json.loads((args.src / MANIFEST_NAME).read_text())
        (args.src / "README.md").write_text(render_card(manifest, args.repo))
        LOG.info("Wrote %s", args.src / "README.md")
        return
    upload(args.src, args.repo, args.private)


if __name__ == "__main__":
    main()
