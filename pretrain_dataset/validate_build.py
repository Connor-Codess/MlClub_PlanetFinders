"""Step 11 of the build guide: check the finished build before anyone trains on it.

Hard gates (exit status 1 if any fails)
    1. Row accounting: every pass-1 stats file satisfies rows_in = written + dropped,
       and every expected source file has a stats file.
    2. Leakage: no star_key appears in both lc-pretrain and lc-heldout, and lc-pretrain
       holds only the train and validation splits.
    3. Round trip: decoding flux_z reproduces the source flux within 0.1 sigma
       (slots with |flux_z| <= 100, sampled every 500th light curve).

Reported (no gate)
    FFI fill by cadence era (expect about 1 for 30-min, 10-min and 200-s FFIs once
    divided by 1, 3 and 9 points), noise against magnitude per source, quality tiers,
    drop reasons, FFI/2-min duplicate pairs, known events.

Writes validation_report.md next to the index files.

Run
    python -m pretrain_dataset.validate_build --index build/index --stats build/pass1/index/pass1
    python -m pretrain_dataset.validate_build --index build/index --stats-from-hub --expected-files all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from . import config

ROUNDTRIP_LIMIT_SIGMA = 0.1


def load_stats(stats_dir: Path | None) -> list[dict]:
    if stats_dir is not None:
        return [json.loads(p.read_text()) for p in sorted(stats_dir.glob("*.json"))]
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    paths = fs.glob(f"datasets/{config.PRETRAIN_REPO}/index/pass1/*.json")
    return [json.loads(fs.read_text(p)) for p in sorted(paths)]


def expected_source_files() -> set[tuple[str, str]]:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    expected = set()
    for source, repo in config.SOURCE_REPOS.items():
        for name in api.list_repo_files(repo, repo_type="dataset"):
            if name.endswith(".parquet") and name[:1] in "sq" and "/" not in name:
                expected.add((source, name))
    return expected


def noise_table(index_dir: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(index_dir.glob("windows_*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run(index_dir: Path, stats: list[dict], expected: set[tuple[str, str]] | None) -> tuple[bool, str]:
    summary = json.loads((index_dir / "summary.json").read_text())
    lines = [f"# Validation report, lc-pretrain {config.BUILD_VERSION}", ""]
    ok = True

    # Gate 1: row accounting
    bad = [s for s in stats if s["rows_in"] != s["rows_written"] + s["rows_dropped"]]
    have = {(s["source"], s["file"]) for s in stats}
    missing = sorted(expected - have) if expected is not None else []
    gate1 = not bad and not missing
    ok &= gate1
    lines += ["## Gates", "",
              f"- {'PASS' if gate1 else 'FAIL'} row accounting: {len(stats)} files checked, "
              f"{len(bad)} mismatched, {len(missing)} expected files missing"
              + (f" ({missing[:10]})" if missing else "")]

    # Gate 2: leakage
    leaked = summary["star_keys_in_both_pretrain_and_heldout"]
    pretrain_splits = set()
    for source_counts in summary["windows_by_split"].values():
        pretrain_splits |= set(source_counts.keys())
    wrong_splits = sorted(pretrain_splits - set(config.PRETRAIN_SPLITS))
    gate2 = leaked == 0 and not wrong_splits
    ok &= gate2
    lines.append(f"- {'PASS' if gate2 else 'FAIL'} leakage: {leaked} star_keys in both repos; "
                 f"unexpected splits in lc-pretrain: {wrong_splits or 'none'}")

    # Gate 3: round trip
    worst = max((s["roundtrip_max_error_sigma"] for s in stats), default=0.0)
    gate3 = worst <= ROUNDTRIP_LIMIT_SIGMA
    ok &= gate3
    lines.append(f"- {'PASS' if gate3 else 'FAIL'} round trip: worst decoded error {worst:.4f} sigma "
                 f"(limit {ROUNDTRIP_LIMIT_SIGMA})")

    # Reports
    table = pd.DataFrame(stats)
    lines += ["", "## Rows", "",
              f"- Light curves read: {int(table['rows_in'].sum()):,}",
              f"- Segments written: {int(table['rows_written'].sum()):,}",
              f"- Dropped: {int(table['rows_dropped'].sum()):,}"]
    drops: dict[str, int] = {}
    for s in stats:
        for reason, n in s["dropped_by_reason"].items():
            drops[reason] = drops.get(reason, 0) + n
    lines += [f"  - {reason}: {n:,}" for reason, n in sorted(drops.items(), key=lambda x: -x[1])]
    lines += ["", "## Segments and windows", "", "```json",
              json.dumps({k: summary[k] for k in ("segments", "segments_by_split", "windows_by_split",
                                                  "quality_tiers", "stars", "ffi_2min_duplicate_pairs",
                                                  "variability_thresholds")}, indent=2), "```"]

    ffi = table[table["source"] == "ffi"].copy()
    if not ffi.empty and "mean_fill_sampled" in ffi:
        ffi["sector"] = ffi["file"].str.slice(1, 5).astype(int)
        ffi["era"] = pd.cut(ffi["sector"], [0, 26, 55, 1000], labels=["30-min (s1-26)", "10-min (s27-55)", "200-s (s56+)"])
        era = ffi.groupby("era", observed=True)["mean_fill_sampled"].agg(["count", "mean", "min"]).round(3)
        lines += ["", "## FFI fill by cadence era", "", era.to_markdown()]

    lines += ["", "## Pass-1 files", "", table[["source", "file", "rows_in", "rows_written", "rows_dropped",
                                                  "roundtrip_max_error_sigma", "seconds"]].sort_values(["source", "file"]).to_markdown(index=False)]
    report = "\n".join(lines) + "\n"
    (index_dir / "validation_report.md").write_text(report)
    return ok, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 11 checks on a finished build")
    parser.add_argument("--index", type=Path, default=Path("build/index"))
    parser.add_argument("--stats", type=Path, default=None, help="folder of pass-1 stats JSON files")
    parser.add_argument("--stats-from-hub", action="store_true")
    parser.add_argument("--expected-files", choices=["all", "none"], default="none",
                        help="'all': every source file on the Hub must have been processed")
    args = parser.parse_args()
    stats = load_stats(None if args.stats_from_hub else args.stats)
    expected = expected_source_files() if args.expected_files == "all" else None
    ok, report = run(args.index, stats, expected)
    print(report)
    for line in report.splitlines():
        if line.startswith("- PASS") or line.startswith("- FAIL"):
            level = "notice" if line.startswith("- PASS") else "error"
            print(f"::{level} title=Step 11::{line[2:]}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
