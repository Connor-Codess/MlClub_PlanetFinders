"""Plan a fine-tuning build: which source files to scan, which catalog revision, what scale.

Reads finetune_dataset/build_request.json:
    {"sources": ["ffi", "2min", "kepler"], "files": null or ["s0001.parquet", ...], "scale": 1.0,
     "reuse_run_id": null or <workflow run id>}

reuse_run_id reruns only the assemble job on the materialized rows of an earlier run
(artifacts are kept 5 days), for example after fixing a card or an upload problem.

Lists the source files that lc-pretrain's pass 1 finished (index/pass1/*.json) and pins
the reference-catalogs revision that pass 1 used, so known events and pools match the
stored rows. Writes to $GITHUB_OUTPUT: matrix (JSON list of {source, stem}), count,
catalog_revision, scale, reuse_run_id.

How to run
    python -m finetune_dataset.plan_build --request finetune_dataset/build_request.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from pretrain_dataset import config as pretrain


def plan(request: dict) -> tuple[list[dict], str]:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    names = [f for f in api.list_repo_files(pretrain.PRETRAIN_REPO, repo_type="dataset")
             if f.startswith("index/pass1/") and f.endswith(".json")]
    sources = set(request.get("sources") or ["ffi", "2min", "kepler"])
    wanted = set(request.get("files") or [])
    matrix = []
    for name in sorted(names):
        source, stem = Path(name).stem.split("_", 1)
        if source in sources and (not wanted or f"{stem}.parquet" in wanted):
            matrix.append({"source": source, "stem": stem})
    stats = json.loads(Path(hf_hub_download(pretrain.PRETRAIN_REPO, sorted(names)[0], repo_type="dataset",
                                            token=os.environ.get("HF_TOKEN"))).read_text())
    return matrix, stats["reference_catalogs_revision"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    matrix, revision = plan(request)
    if len(matrix) > 256:
        raise SystemExit(f"{len(matrix)} files exceed the 256-job matrix limit")
    scale = float(request.get("scale", 1.0))
    reuse = request.get("reuse_run_id")
    outputs = {"matrix": json.dumps(matrix), "count": str(len(matrix)), "catalog_revision": revision,
               "scale": str(scale), "reuse_run_id": str(reuse) if reuse else ""}
    print(json.dumps({k: v for k, v in outputs.items() if k != "matrix"}))
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as handle:
            for key, value in outputs.items():
                handle.write(f"{key}={value}\n")


if __name__ == "__main__":
    main()
