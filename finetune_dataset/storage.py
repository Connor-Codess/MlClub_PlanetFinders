"""Locating and reading the lc-pretrain and lc-heldout files of one source file.

Every original source file (for example ffi/s0040.parquet) produced up to three
pass-1 files: lc-pretrain data/<source>/train/<stem>, data/<source>/validation/<stem>
and lc-heldout data/<source>/<stem>. Stages work on all three at once.

With --local-root, files are read from a local pass-1 output directory
(build/pass1 has pretrain/ and heldout/ subdirectories); otherwise they are
downloaded from the Hub (HF_TOKEN is needed for the private lc-heldout).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from pretrain_dataset import config as pretrain


@dataclass(frozen=True)
class PassOneFile:
    repo: str            # "pretrain" or "heldout"
    path: str            # path inside the Hugging Face repo
    local: Path

    @property
    def key(self) -> str:
        return f"{self.repo}:{self.path}"


def repo_paths(source: str, stem: str) -> list[tuple[str, str]]:
    name = f"{stem}.parquet"
    return [
        ("pretrain", f"data/{source}/train/{name}"),
        ("pretrain", f"data/{source}/validation/{name}"),
        ("heldout", f"data/{source}/{name}"),
    ]


def repo_id(repo: str) -> str:
    return pretrain.PRETRAIN_REPO if repo == "pretrain" else pretrain.HELDOUT_REPO


def fetch(source: str, stem: str, local_root: Path | None, cache_dir: Path,
          only: set[str] | None = None) -> list[PassOneFile]:
    """Return the pass-1 files of one source file that exist, downloading if needed."""
    found = []
    for repo, path in repo_paths(source, stem):
        if only is not None and f"{repo}:{path}" not in only:
            continue
        if local_root is not None:
            local = local_root / repo / path
            if local.exists():
                found.append(PassOneFile(repo, path, local))
            continue
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            local = hf_hub_download(repo_id(repo), path, repo_type="dataset", local_dir=cache_dir / repo,
                                    token=os.environ.get("HF_TOKEN"))
        except EntryNotFoundError:
            continue
        found.append(PassOneFile(repo, path, Path(local)))
    return found


def iter_row_groups(path: Path, columns: list[str]):
    """Yield (first_row_index, table) for every row group of a file."""
    parquet = pq.ParquetFile(path)
    first = 0
    for group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(group, columns=columns)
        yield first, table
        first += table.num_rows
