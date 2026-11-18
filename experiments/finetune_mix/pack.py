"""Pack one arm's training rows, or lc-eval val, into arrays (see data.py for the layout).

What it does
    --arm NAME   reads selection.parquet from select.py, downloads the lc-finetune train
                 shards that hold the arm's rows and writes the arm's arrays.
    --eval       downloads lc-eval val (private: needs HF_TOKEN) and writes its arrays.
                 With --box-filter-out it also scores the box-filter baseline on them.

Outputs (in --out)
    x.npy, observed.npy, label.npy, target.npy, meta.parquet

How to run
    python -m experiments.finetune_mix.pack --arm baseline --selection build/mix/select --out build/mix/arm
    python -m experiments.finetune_mix.pack --eval --out build/mix/eval --box-filter-out build/mix/results/box_filter
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

from . import config
from .data import download, pack_rows, read_meta

LOG = logging.getLogger("mix.pack")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--selection", type=Path, help="select.py output directory")
    parser.add_argument("--data", type=Path, default=Path("build/mix/source"))
    parser.add_argument("--no-download", action="store_true", help="use shards already in --data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--box-filter-out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.eval:
        files = (sorted(args.data.glob(f"data/{config.EVAL_SPLIT}/*.parquet")) if args.no_download
                 else download(config.EVAL_REPO, config.EVAL_SPLIT, args.data))
        chosen = read_meta(files, args.data)
    else:
        if not args.arm or not args.selection:
            parser.error("--arm and --selection are required without --eval")
        selection = pd.read_parquet(args.selection / "selection.parquet")
        chosen = selection[selection["arm"] == args.arm].reset_index(drop=True)
        if chosen.empty:
            raise SystemExit(f"no rows selected for arm {args.arm}")
        paths = sorted(chosen["path"].unique())
        if not args.no_download:
            download(config.FINETUNE_REPO, "train", args.data, only=paths)
    LOG.info("packing %d rows from %d shards", len(chosen), chosen["path"].nunique())
    pack_rows(chosen, args.data, args.out, with_events=args.eval)
    LOG.info("wrote %s", args.out)
    if args.eval and args.box_filter_out:
        subprocess.run([sys.executable, "-m", "experiments.finetune_mix.evaluate", "--eval", str(args.out),
                        "--out", str(args.box_filter_out)], check=True)


if __name__ == "__main__":
    main()
