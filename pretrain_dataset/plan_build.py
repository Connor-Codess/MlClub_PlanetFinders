"""Plan a build: decide which source files pass 1 must process and pin the catalog revision.

What it does
    Reads pretrain_dataset/build_request.json, lists the source files on the
    Hub, removes files that already have a pass-1 stats file in lc-pretrain
    (when skip_existing is true, so re-running the workflow resumes a partial
    build or picks up newly uploaded Kepler quarters), pins the current
    revision of reference-catalogs, and writes the Step 2 exclusion list.

    In GitHub Actions it writes three outputs: matrix (JSON list of
    {"source", "file"}), count, and catalog_revision.

build_request.json fields
    sources          list of "ffi", "2min", "kepler"
    files            optional list of file names to restrict to (e.g. ["s0001.parquet"])
    skip_existing    skip files already processed in this build version
    max_row_groups   optional limit per file, for trial runs only
    note             free text, ignored

Run
    python -m pretrain_dataset.plan_build --request pretrain_dataset/build_request.json --out build/index
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import config


def plan(request: dict) -> tuple[list[dict], str]:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    done: set[str] = set()
    if request.get("skip_existing", True):
        try:
            done = {
                Path(name).stem
                for name in api.list_repo_files(config.PRETRAIN_REPO, repo_type="dataset")
                if name.startswith("index/pass1/")
            }
        except Exception:
            done = set()

    wanted_files = set(request.get("files") or [])
    matrix = []
    for source in request.get("sources", list(config.SOURCE_REPOS)):
        for name in sorted(api.list_repo_files(config.SOURCE_REPOS[source], repo_type="dataset")):
            if "/" in name or not name.endswith(".parquet") or name[:1] not in "sq":
                continue
            if wanted_files and name not in wanted_files:
                continue
            if f"{source}_{Path(name).stem}" in done:
                continue
            matrix.append({"source": source, "file": name})
    revision = api.dataset_info(config.REFERENCE_REPO).sha
    return matrix, revision


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan the pass-1 matrix")
    parser.add_argument("--request", type=Path, default=Path("pretrain_dataset/build_request.json"))
    parser.add_argument("--out", type=Path, default=Path("build/index"))
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    matrix, revision = plan(request)

    from .catalogs import ReferenceCatalogs

    catalogs = ReferenceCatalogs.load(revision)
    args.out.mkdir(parents=True, exist_ok=True)
    catalogs.exclusions.to_parquet(args.out / "exclusions.parquet", index=False)

    print(f"{len(matrix)} files to process; reference-catalogs revision {revision}")
    print(catalogs.exclusions["reason"].value_counts().to_string())
    if len(matrix) > 256:
        raise SystemExit("GitHub Actions allows at most 256 matrix jobs; narrow the request")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a") as handle:
            handle.write(f"matrix={json.dumps(matrix)}\n")
            handle.write(f"count={len(matrix)}\n")
            handle.write(f"catalog_revision={revision}\n")
            handle.write(f"max_row_groups={request.get('max_row_groups') or ''}\n")


if __name__ == "__main__":
    main()
