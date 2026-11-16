"""Published test sets, rebuilt as target tables so our model can be scored on the same signals.

What it does
    Downloads each paper's released material, rebuilds its test set and writes
    benchmark_targets.parquet: one row per test signal (TCE or candidate) with the
    star, ephemeris, time range, the paper's label and its positive class. The scan
    stage cuts our windows around each signal's predicted transits; the choose stage
    removes every benchmark star from lc-finetune.

    astronet_dr24        Shallue & Vanderburg 2018 (AJ 155, 94). Kepler DR24 TCEs labeled PC,
                         AFP or NTP, shuffled with np.random.seed(123) and split 80/10/10 by TCE,
                         exactly as their generate_input_records.py does. Our DR24 table is in the
                         NASA Exoplanet Archive order (kepid, tce_plnt_num), which is assumed to be
                         the order of their CSV. 1,574 test TCEs; the paper evaluates 1,523, the ones
                         whose light curves their pipeline processed.
    astronet_triage      Yu et al. 2019 (AJ 158, 25). TESS QLP sectors 1-5. The test set is read
                         from test-00000-of-00001 in their repository (1,658 TCEs; the paper quotes
                         1,650). Labels PC, EB, J (J includes V and IS).
    astronet_triage_v2   Tey et al. 2023 (AJ 165, 95). TESS QLP, cycles 1-3. Rows with
                         Split == "test" in tces_with_labels_v3.csv (Zenodo 7411579). Labels E, S,
                         B, J, N, or blank when the vote was not unanimous.
    salinas2025          Salinas et al. 2025 (MNRAS 538, 2031). The test set was not released, so
                         this holds their published candidates instead (122 periodic, 82 single
                         transit, 9 multi-planet entries; TESS-SPOC FFI sectors 1-26). It measures
                         how many of their candidates our model also flags.

    These papers classify TCEs given a period and epoch; our model scores 4-hour patches of
    raw windows. A TCE score for our model is built from its patch outputs at the TCE's
    predicted transits (see the lc-benchmarks card), so the comparison is close but not
    identical.

How to run
    python -m finetune_dataset.benchmarks --out build/finetune/benchmarks --catalog-revision <sha>
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pretrain_dataset.catalogs import ReferenceCatalogs
from pretrain_dataset.star_identity import tic_key

LOG = logging.getLogger("finetune.benchmarks")

YU_COMMIT = "5675a57dd41dd0321df480453451096dc5a4a6b0"
SOURCES = {
    "astronet_triage": f"https://raw.githubusercontent.com/yuliang419/Astronet-Triage/{YU_COMMIT}/astronet/tfrecords/test-00000-of-00001",
    "astronet_triage_v2": "https://zenodo.org/api/records/7411579/files/tces_with_labels_v3.csv/content",
    "salinas2025": "https://arxiv.org/e-print/2502.07542v2",
}

# Results each paper reports on its own test set, shown in the lc-benchmarks card.
PUBLISHED = {
    "astronet_dr24": {
        "paper": "Shallue & Vanderburg 2018, AJ 155, 94", "mission": "Kepler",
        "positive": "PC", "metrics": {"AUC": 0.988, "accuracy (threshold 0.5)": 0.960},
        "test_size_paper": 1523,
    },
    "astronet_triage": {
        "paper": "Yu et al. 2019, AJ 158, 25", "mission": "TESS QLP FFI, sectors 1-5",
        "positive": "PC or EB (triage); PC (vetting)",
        "metrics": {"triage AUC": 0.992, "triage average precision": 0.970, "triage accuracy (0.5)": 0.974,
                    "triage precision at 0.1": 0.749, "triage recall at 0.1": 0.975,
                    "vetting AUC": 0.984, "vetting average precision": 0.693, "vetting accuracy": 0.978},
        "test_size_paper": 1650,
    },
    "astronet_triage_v2": {
        "paper": "Tey et al. 2023, AJ 165, 95", "mission": "TESS QLP FFI, cycles 1-3",
        "positive": "E (periodic eclipsing: planets and detached EBs)",
        "metrics": {"AUC-PR": 0.965, "recall at threshold 0.215": 0.972, "precision at threshold 0.215": 0.757,
                    "recall at threshold 0.0105": 0.996, "precision at threshold 0.0105": 0.397},
        "test_size_paper": None,
    },
    "salinas2025": {
        "paper": "Salinas et al. 2025, MNRAS 538, 2031", "mission": "TESS-SPOC FFI, sectors 1-26",
        "positive": "all rows are their new candidates",
        "metrics": {"AUC (own unreleased test set)": 0.88, "F1 (own unreleased test set)": 0.82},
        "test_size_paper": None,
    },
}

COLUMNS = ["benchmark", "tce_id", "star_key", "tic_id", "kic_id", "mission", "sector", "t_min_btjd", "t_max_btjd",
           "epoch_btjd", "period_d", "duration_h", "depth_ppm", "label", "cls", "positive_paper", "positive_planet"]


def download(url: str) -> bytes:
    response = requests.get(url, timeout=600, headers={"User-Agent": "nyu-planet-finders benchmark build"})
    response.raise_for_status()
    return response.content


def astronet_dr24(catalogs: ReferenceCatalogs) -> pd.DataFrame:
    table = catalogs.dr24_tce.sort_values(["kepid", "tce_plnt_num"], kind="stable")
    table = table[table["av_training_set"].isin(["PC", "AFP", "NTP"])]
    n = len(table)
    np.random.seed(123)
    table = table.iloc[np.random.permutation(n)]
    test = table.iloc[int(0.90 * n):]
    cls = {"PC": "planet", "AFP": "false_positive", "NTP": "false_positive"}
    rows = []
    for r in test.itertuples(index=False):
        tic = catalogs.kic_to_tic.get(int(r.kepid))
        rows.append({
            "tce_id": f"KIC {int(r.kepid)} TCE {int(r.tce_plnt_num)}",
            "star_key": tic_key(tic) if tic is not None else f"kic:{int(r.kepid)}",
            "tic_id": tic, "kic_id": int(r.kepid), "mission": "kepler",
            "epoch_btjd": float(r.tce_time0bk) - 2167.0, "period_d": float(r.tce_period),
            "duration_h": float(r.tce_duration), "depth_ppm": float(r.tce_depth),
            "label": r.av_training_set, "cls": cls[r.av_training_set],
            "positive_paper": r.av_training_set == "PC", "positive_planet": r.av_training_set == "PC",
        })
    return pd.DataFrame(rows)


def astronet_triage(raw: bytes) -> pd.DataFrame:
    from tfrecord.reader import tfrecord_loader

    with tempfile.NamedTemporaryFile(suffix=".tfrecord") as handle:
        handle.write(raw)
        handle.flush()
        records = list(tfrecord_loader(handle.name, None))
    cls = {"PC": "planet", "EB": "eclipsing_binary", "J": "false_positive"}
    rows = []
    for rec in records:
        label = rec["Disposition"].decode()
        tic = int(rec["tic_id"][0])
        rows.append({
            "tce_id": f"TIC {tic} S{int(rec['Sectors'][0])} row {int(rec['row_id'][0])}",
            "star_key": tic_key(tic), "tic_id": tic, "kic_id": None, "mission": "tess",
            "sector": int(rec["Sectors"][0]), "epoch_btjd": float(rec["Epoc"][0]),
            "period_d": float(rec["Period"][0]), "duration_h": float(rec["Duration"][0]) * 24.0,
            "depth_ppm": float(rec["Transit_Depth"][0]), "label": label, "cls": cls.get(label, "unknown"),
            "positive_paper": label in ("PC", "EB"), "positive_planet": label == "PC",
        })
    return pd.DataFrame(rows)


def astronet_triage_v2(raw: bytes) -> pd.DataFrame:
    table = pd.read_csv(io.BytesIO(raw), comment="#")
    table = table[table["TIC ID"].notna()]
    table = table[table["Split"] == "test"]
    cls = {"E": "unknown", "S": "unknown", "B": "eclipsing_binary", "J": "false_positive", "N": "unknown"}
    rows = []
    for i, r in enumerate(table.itertuples(index=False)):
        tic = int(float(r[0]))
        label = r[table.columns.get_loc("Consensus Label")]
        label = label if isinstance(label, str) else "none"
        rows.append({
            "tce_id": f"TIC {tic} Y{int(r.Year)} P{float(r.Period):.6f} E{float(r.Epoch):.4f}",
            "star_key": tic_key(tic), "tic_id": tic, "kic_id": None, "mission": "tess",
            "t_min_btjd": float(r.MinT), "t_max_btjd": float(r.MaxT), "epoch_btjd": float(r.Epoch),
            "period_d": float(r.Period), "duration_h": float(r.Duration) * 24.0, "depth_ppm": float(r.Depth),
            "label": label, "cls": cls.get(label, "unknown"),
            "positive_paper": label == "E", "positive_planet": False,
        })
    return pd.DataFrame(rows)


SALINAS_MULTI = [  # Table 5 of the paper: TIC, epoch (BTJD), depth (ppm)
    (193096383, "b", 1546.82, 2870), (193096383, "c", 1559.01, 9564), (221567884, "b", 1629.92, 868),
    (221567884, "c", 1648.66, 2169), (221567884, "d", 2372.65, 3037), (118798035, "b", 1656.35, 5970),
    (118798035, "c", 1663.91, 10660), (400049903, "b", 1669.72, 4041), (400049903, "c", 1778.60, 2424),
]


def salinas2025(raw: bytes) -> pd.DataFrame:
    rows = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        def read(name: str) -> pd.DataFrame:
            return pd.read_csv(tar.extractfile(f"candidates/{name}"))

        periodic = pd.concat([read("periodictransitcandidatepart1.csv"), read("periodictransitcandidatepart2.csv")])
        single = pd.concat([read("singletransitercandidatepart1d.csv"), read("singletransitercandidatepart2d.csv")])
    for r in periodic.itertuples(index=False):
        rows.append({"tce_id": f"TIC {int(r.TICID)} periodic", "tic_id": int(r.TICID), "epoch_btjd": float(r.epoch),
                     "period_d": float(r.orbitalperiod), "duration_h": float(r.duration),
                     "depth_ppm": float(r.tdepth), "label": "periodic_candidate"})
    for r in single.itertuples(index=False):
        rows.append({"tce_id": f"TIC {int(r.TICID)} single S{int(r.sector)}", "tic_id": int(r.TICID),
                     "sector": int(r.sector), "epoch_btjd": float(r.epoch), "period_d": None,
                     "duration_h": float(r.duration), "depth_ppm": float(r.tdepth), "label": "single_candidate"})
    for tic, letter, epoch, depth in SALINAS_MULTI:
        rows.append({"tce_id": f"TIC {tic} {letter}", "tic_id": tic, "epoch_btjd": epoch, "period_d": None,
                     "duration_h": None, "depth_ppm": float(depth), "label": "multi_planet_candidate"})
    out = pd.DataFrame(rows)
    out["star_key"] = out["tic_id"].map(tic_key)
    out["kic_id"] = None
    out["mission"] = "tess"
    out["cls"] = "planet"
    out["positive_paper"] = True
    out["positive_planet"] = True
    return out


def build(catalogs: ReferenceCatalogs) -> tuple[pd.DataFrame, dict]:
    provenance = {}
    frames = {"astronet_dr24": astronet_dr24(catalogs)}
    provenance["astronet_dr24"] = {"source": f"{catalogs.revision} reference-catalogs; DR24 TCE table of "
                                             "nyu-planet-finders/kepler-raw-lightcurves"}
    builders = {"astronet_triage": astronet_triage, "astronet_triage_v2": astronet_triage_v2,
                "salinas2025": salinas2025}
    for name, builder in builders.items():
        raw = download(SOURCES[name])
        provenance[name] = {"source": SOURCES[name], "sha256": hashlib.sha256(raw).hexdigest()}
        frames[name] = builder(raw)
    parts = []
    for name, frame in frames.items():
        frame = frame.assign(benchmark=name)
        for column in COLUMNS:
            if column not in frame:
                frame[column] = None
        parts.append(frame[COLUMNS])
        provenance[name].update({"rows": len(frame), "stars": int(frame["star_key"].nunique()),
                                 "labels": frame["label"].value_counts().to_dict(), **PUBLISHED[name]})
    targets = pd.concat(parts, ignore_index=True)
    for column in ("tic_id", "kic_id", "sector"):
        targets[column] = targets[column].astype("Int64")
    return targets, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--catalog-revision", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    targets, provenance = build(ReferenceCatalogs.load(args.catalog_revision))
    args.out.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(args.out / "benchmark_targets.parquet", index=False)
    (args.out / "benchmark_provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    for name, info in provenance.items():
        print(f"::notice title=benchmark {name}::{info['rows']} signals on {info['stars']} stars, labels {info['labels']}")


if __name__ == "__main__":
    main()
