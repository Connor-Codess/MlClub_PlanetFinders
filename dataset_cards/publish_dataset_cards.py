"""Upload the hand-written dataset cards in this folder to Hugging Face.

Each <repo>.md here becomes README.md of nyu-planet-finders/<repo>. The cards of
generated datasets are written by their builds instead: pretrain_dataset/publish_cards.py
(lc-pretrain, lc-heldout), finetune_dataset/cards.py (lc-finetune, lc-eval,
lc-benchmarks) and reference_catalogs/upload_reference_catalogs_to_hub.py.

How to run
    HF_TOKEN=... python dataset_cards/publish_dataset_cards.py [--only tess-spoc-ffi-binned]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ORG = "nyu-planet-finders"
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="repository names to publish (default: every card here)")
    args = parser.parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])
    cards = sorted(HERE.glob("*.md"))
    for card in cards:
        repo = f"{ORG}/{card.stem}"
        if args.only and card.stem not in args.only:
            continue
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo, repo_type="dataset",
                        commit_message="Dataset card")
        print(f"::notice title=card::{repo}")


if __name__ == "__main__":
    main()
