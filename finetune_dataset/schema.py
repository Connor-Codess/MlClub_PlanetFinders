"""Row layout of lc-finetune and lc-eval, and a decoder for one row.

One row is one 27-day window (1,296 slots of 30 minutes, 162 patches of 4 hours).
Channels use the lc-pretrain encoding (little-endian float16 or uint8 bytes).
Patch targets are byte strings of length 162.

    from finetune_dataset.schema import decode_sample
    arrays = decode_sample(row)        # row: dict from pyarrow / pandas / datasets
    arrays["flux_z"]                   # float32 (1296,)
    arrays["patch_label"]              # uint8 (162,): 0, 1 or 255 (ignore)
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from pretrain_dataset import config as pretrain

SAMPLE_EVENT = pa.struct([
    ("center_slot", pa.float32()), ("duration_slots", pa.float32()), ("kind", pa.string()),
    ("ref", pa.string()), ("cls", pa.string()), ("period_d", pa.float64()), ("snr", pa.float32()),
    ("observed_fraction", pa.float32()), ("visible", pa.bool_()), ("is_planet", pa.bool_()),
    ("expected_snr", pa.float32()),
])

SAMPLE_SCHEMA = pa.schema([
    ("sample_id", pa.string()), ("dataset", pa.string()), ("split", pa.string()), ("category", pa.string()),
    ("star_key", pa.string()), ("source", pa.string()), ("segment_id", pa.string()),
    ("sector", pa.int16()), ("quarter", pa.int16()),
    ("window_slot0", pa.int64()), ("t_start_btjd", pa.float64()), ("start_offset", pa.int32()),
    ("source_repo", pa.string()), ("source_path", pa.string()), ("source_row", pa.int64()),
    ("sigma_p2p_ppm", pa.float32()), ("tmag", pa.float32()), ("kepmag", pa.float32()),
    ("teff", pa.float32()), ("radius", pa.float32()), ("quality_tier", pa.int8()),
    ("coverage", pa.float32()), ("single_transit_crop", pa.bool_()),
    ("flux_z", pa.binary()), ("flux_err_z", pa.binary()), ("cen_x", pa.binary()), ("cen_y", pa.binary()),
    ("bkg_z", pa.binary()), ("fill", pa.binary()), ("flags", pa.binary()),
    ("patch_label", pa.binary()), ("patch_kind", pa.binary()), ("patch_overlap", pa.binary()),
    ("patch_center_offset", pa.binary()),
    ("n_transit_patches", pa.int16()), ("has_transit", pa.bool_()),
    ("stacked_snr", pa.float32()), ("n_visible_transits", pa.int16()),
    ("events", pa.list_(SAMPLE_EVENT)),
    ("inj_model", pa.string()), ("inj_period_d", pa.float64()), ("inj_t0_btjd", pa.float64()),
    ("inj_rp_rs", pa.float32()), ("inj_a_rs", pa.float32()), ("inj_b", pa.float32()),
    ("inj_u1", pa.float32()), ("inj_u2", pa.float32()), ("inj_duration_h", pa.float32()),
    ("inj_depth_ppm", pa.float32()), ("inj_snr_target", pa.float32()), ("inj_snr", pa.float32()),
    ("inj_snr_total", pa.float32()), ("inj_star_radius", pa.float32()), ("inj_star_mass", pa.float32()),
    ("inj_star_teff", pa.float32()), ("inj_template_id", pa.string()), ("inj_depth_scale", pa.float32()),
    ("benchmark_tce_id", pa.string()),
    ("build_version", pa.string()),
])

INJECTION_COLUMNS = [name for name in SAMPLE_SCHEMA.names if name.startswith("inj_")]


def decode_sample(row: dict) -> dict[str, np.ndarray]:
    """Channels as float32 arrays of 1,296 slots and patch targets as arrays of 162 patches."""
    out = {name: np.frombuffer(row[name], dtype="<f2").astype(np.float32)
           for name in pretrain.CHANNELS_FLOAT16}
    out["fill"] = np.frombuffer(row["fill"], dtype=np.uint8).astype(np.float32) / 255.0
    out["flags"] = np.frombuffer(row["flags"], dtype=np.uint8)
    out["observed"] = (out["flags"] & pretrain.FLAG_OBSERVED) > 0
    out["patch_label"] = np.frombuffer(row["patch_label"], dtype=np.uint8)
    out["patch_kind"] = np.frombuffer(row["patch_kind"], dtype=np.uint8)
    out["patch_overlap"] = np.frombuffer(row["patch_overlap"], dtype=np.uint8).astype(np.float32) / 255.0
    out["patch_center_offset"] = np.frombuffer(row["patch_center_offset"], dtype="<f2").astype(np.float32)
    return out
