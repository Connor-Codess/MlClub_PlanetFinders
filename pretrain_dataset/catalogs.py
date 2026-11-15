"""Step 7 inputs: catalog tags and known-event ephemerides for every star.

Loads, at one pinned revision of nyu-planet-finders/reference-catalogs:
    toi_ephemerides, tess_eb_catalog_prsa2022, kepler_eb_catalog_kirk2016,
    duotransits_hawthorn2024, kic_tic_crossmatch
plus the DR24 TCE table from nyu-planet-finders/kepler-raw-lightcurves, and
turns them into per-star lookups:

    tags_for(star_key, kic_id)            -> dict of tag columns
    events_for(star_key, kic_id, source, t_start, t_end)
                                          -> list of predicted transits/eclipses

Which ephemerides apply where
    TESS segments:   TOIs, TESS EBs and Hawthorn duotransits (keyed by TIC).
    Kepler segments: Kepler EBs and DR24 TCEs (keyed by KIC).
    TOI and EB ephemerides are not projected across the 5-10 years between the
    missions, where period errors make predicted times unreliable; the tags
    still mark those stars.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .star_identity import build_exclusions, tic_key

TESS_SOURCES = ("ffi", "2min")


@dataclass
class Ephemeris:
    kind: str
    ref: str
    t0_btjd: float
    period_d: float | None
    duration_h: float | None
    phase: float = 0.0          # secondary eclipses sit at t0 + phase * period


@dataclass
class ReferenceCatalogs:
    toi: pd.DataFrame
    tess_eb: pd.DataFrame
    kepler_eb: pd.DataFrame
    duotransits: pd.DataFrame
    crossmatch: pd.DataFrame
    dr24_tce: pd.DataFrame
    revision: str
    kic_to_tic: dict[int, int] = field(default_factory=dict)
    exclusions: pd.DataFrame | None = None

    @classmethod
    def load(cls, revision: str | None = None) -> "ReferenceCatalogs":
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        resolved = revision or api.dataset_info(config.REFERENCE_REPO).sha

        def read(name: str, repo: str = config.REFERENCE_REPO, rev: str | None = resolved) -> pd.DataFrame:
            return pd.read_parquet(hf_hub_download(repo, name, repo_type="dataset", revision=rev))

        catalogs = cls(
            toi=read("toi_ephemerides.parquet"),
            tess_eb=read("tess_eb_catalog_prsa2022.parquet"),
            kepler_eb=read("kepler_eb_catalog_kirk2016.parquet"),
            duotransits=read("duotransits_hawthorn2024.parquet"),
            crossmatch=read("kic_tic_crossmatch.parquet"),
            dr24_tce=read(config.KEPLER_TCE_FILE, config.KEPLER_TCE_REPO, None),
            revision=resolved,
        )
        catalogs.prepare()
        return catalogs

    def prepare(self) -> None:
        matched = self.crossmatch[self.crossmatch["matched"]]
        self.kic_to_tic = dict(zip(matched["kic_id"].astype("int64"), matched["tic_id"].astype("int64")))
        self.exclusions = build_exclusions(self.toi, self.duotransits)
        self._build_tags()
        self._build_ephemerides()

    def _kic_star_key(self, kic_id: int) -> str:
        tic = self.kic_to_tic.get(int(kic_id))
        return tic_key(tic) if tic is not None else f"kic:{int(kic_id)}"

    def _build_tags(self) -> None:
        toi_ids: dict[str, list[str]] = defaultdict(list)
        toi_disp: dict[str, set[str]] = defaultdict(set)
        for row in self.toi.itertuples(index=False):
            if pd.isna(row.tic_id):
                continue
            key = tic_key(row.tic_id)
            toi_ids[key].append(f"{row.toi:.2f}")
            if isinstance(row.disposition, str):
                toi_disp[key].add(row.disposition)

        eb: dict[str, set[str]] = defaultdict(set)
        for tic in self.tess_eb["tic_id"].dropna():
            eb[tic_key(tic)].add("tess_eb")
        for kic in self.kepler_eb["kic_id"].dropna():
            eb[self._kic_star_key(kic)].add("kepler_eb")

        tce_labels: dict[str, set[str]] = defaultdict(set)
        for kic, label in zip(self.dr24_tce["kepid"], self.dr24_tce["av_training_set"]):
            if pd.notna(kic) and isinstance(label, str):
                tce_labels[self._kic_star_key(kic)].add(label)

        hawthorn = {tic_key(t): int(c) for t, c in zip(self.duotransits["tic_id"], self.duotransits["candidate_id"])}

        self.tags: dict[str, dict] = {}
        for key in set(toi_ids) | set(eb) | set(tce_labels) | set(hawthorn):
            self.tags[key] = {
                "toi_ids": ",".join(toi_ids[key]) or None,
                "toi_disposition": ",".join(sorted(toi_disp[key])) or None,
                "eb_catalog": ",".join(sorted(eb[key])) or None,
                "kepler_tce_labels": ",".join(sorted(tce_labels[key])) or None,
                "hawthorn_candidate": hawthorn.get(key),
            }

    def _build_ephemerides(self) -> None:
        tess: dict[str, list[Ephemeris]] = defaultdict(list)
        kepler: dict[int, list[Ephemeris]] = defaultdict(list)

        def num(value) -> float | None:
            return None if value is None or pd.isna(value) else float(value)

        for r in self.toi.itertuples(index=False):
            if pd.notna(r.tic_id) and pd.notna(r.t0_btjd):
                tess[tic_key(r.tic_id)].append(Ephemeris(
                    "toi", f"TOI-{r.toi:.2f} {r.disposition}", float(r.t0_btjd), num(r.period_d), num(r.duration_h)))
        for r in self.tess_eb.itertuples(index=False):
            if pd.notna(r.tic_id) and pd.notna(r.t0_btjd) and pd.notna(r.period_d):
                ref = f"TIC {int(r.tic_id)} signal {int(r.signal_id)}"
                tess[tic_key(r.tic_id)].append(Ephemeris(
                    "tess_eb_primary", ref, float(r.t0_btjd), float(r.period_d), num(r.primary_duration_h)))
                if pd.notna(r.secondary_phase) and pd.notna(r.secondary_width_phase):
                    tess[tic_key(r.tic_id)].append(Ephemeris(
                        "tess_eb_secondary", ref, float(r.t0_btjd), float(r.period_d),
                        float(r.secondary_width_phase) * float(r.period_d) * 24.0, float(r.secondary_phase)))
        for r in self.duotransits.itertuples(index=False):
            ref = f"Hawthorn2024 #{int(r.candidate_id)}"
            for tc in (r.tc1_btjd, r.tc2_btjd):
                tess[tic_key(r.tic_id)].append(Ephemeris("hawthorn_duotransit", ref, float(tc), None, num(r.duration_h)))
        for r in self.kepler_eb.itertuples(index=False):
            if pd.notna(r.kic_id) and pd.notna(r.t0_btjd) and pd.notna(r.period_d):
                kepler[int(r.kic_id)].append(Ephemeris(
                    "kepler_eb_primary", f"KIC {int(r.kic_id)}", float(r.t0_btjd), float(r.period_d), None))
        for r in self.dr24_tce.itertuples(index=False):
            if pd.notna(r.kepid) and pd.notna(r.tce_time0bk) and pd.notna(r.tce_period):
                kepler[int(r.kepid)].append(Ephemeris(
                    "kepler_tce", f"KIC {int(r.kepid)} TCE {int(r.tce_plnt_num)} {r.av_training_set}",
                    float(r.tce_time0bk) + config.BKJD_TO_BTJD, float(r.tce_period), num(r.tce_duration)))
        self.tess_ephemerides = dict(tess)
        self.kepler_ephemerides = dict(kepler)

    def exclusion_map(self) -> dict[str, str]:
        return dict(zip(self.exclusions["star_key"], self.exclusions["reason"]))

    def tags_for(self, star_key: str) -> dict:
        return self.tags.get(star_key, {
            "toi_ids": None, "toi_disposition": None, "eb_catalog": None,
            "kepler_tce_labels": None, "hawthorn_candidate": None,
        })

    def events_for(self, star_key: str, kic_id: int | None, source: str,
                   slot0: int, n_slots: int) -> list[dict]:
        if source in TESS_SOURCES:
            ephemerides = self.tess_ephemerides.get(star_key, [])
        else:
            ephemerides = self.kepler_ephemerides.get(int(kic_id), []) if kic_id is not None else []
        return predict_events(ephemerides, slot0, n_slots)


def predict_events(ephemerides: list[Ephemeris], slot0: int, n_slots: int) -> list[dict]:
    """Predicted event centers inside a segment, as fractional slots relative to slot0.

    Slot j of the segment spans [j, j + 1); an event whose mid-time falls in the
    middle of slot j has center_slot j + 0.5. Events within half a duration of
    either edge are kept, so partial events are listed.
    """
    t_start = config.GRID_EPOCH_BTJD + slot0 * config.SLOT_DAYS
    t_end = t_start + n_slots * config.SLOT_DAYS
    events: list[dict] = []
    for eph in ephemerides:
        half = (eph.duration_h or 0.0) / 48.0
        if eph.period_d is None:
            centers = [eph.t0_btjd] if t_start - half <= eph.t0_btjd <= t_end + half else []
        else:
            if eph.period_d < config.MIN_EVENT_PERIOD_DAYS:
                continue
            ref_time = eph.t0_btjd + eph.phase * eph.period_d
            n_first = math.ceil((t_start - half - ref_time) / eph.period_d)
            n_last = math.floor((t_end + half - ref_time) / eph.period_d)
            centers = [ref_time + n * eph.period_d for n in range(n_first, n_last + 1)]
        for t in centers:
            events.append({
                "center_slot": np.float32((t - t_start) / config.SLOT_DAYS),
                "duration_slots": np.float32(eph.duration_h * 2.0) if eph.duration_h else np.float32("nan"),
                "kind": eph.kind,
                "ref": eph.ref,
            })
            if len(events) >= config.MAX_EVENTS_PER_SEGMENT:
                return events
    return events
