"""
ROI-guided HU features for contrast-phase — the "Segment-and-Classify" signal.

Contrast phase is defined by *absolute HU enhancement in specific organs*
(aorta bright in arterial; portal/splenic vein + IVC bright in venous; kidney
cortico-medullary pattern; etc.). Hou 2025 (`CTPhase-XGBoost`) turns this into a
16-dim feature: the *median HU inside each organ mask*, fed to XGBoost.

This module reproduces that feature exactly, but reads a segmentation mask that
is already on disk (the repo's co-registered `*_deeds_seg_reg.nii.gz`) instead
of re-running TotalSegmentator per call — as long as that mask is a
TotalSegmentator *multilabel* volume (one integer label id per organ). If the
mask is binary (union of all organs) the per-organ signal is gone and the mask
cannot be used for this; `assert_multilabel` detects that up front.

Organ order and the `0.0 -> NaN` "organ absent" convention match
`CTPhase-XGBoost/totalseg_get_phase.py` verbatim so features are compatible with
the shipped `xgb_vindr.pkl`.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# EXACT organ list + order the shipped XGBoost model was trained on
# (CTPhase-XGBoost/totalseg_get_phase.py). Do not reorder.
ORGANS: List[str] = [
    "liver", "pancreas", "urinary_bladder", "gallbladder",
    "heart", "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein",
    "iliac_vena_left", "iliac_vena_right", "iliac_artery_left", "iliac_artery_right",
    "pulmonary_vein", "brain", "colon", "small_bowel",
]


def load_ts_label_map() -> Dict[str, int]:
    """{organ_name: integer label id} from the installed TotalSegmentator.

    This is the single source of truth for how organ names map to the integer
    values in a `*_seg_reg.nii.gz` multilabel mask — it must be the SAME package
    version that produced those masks. We refuse to guess: if TotalSegmentator
    isn't importable, raise (same no-silent-fallback stance as encoders.py),
    because a wrong id map silently produces garbage features.
    """
    try:
        from totalsegmentator.map_to_binary import class_map
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Cannot import totalsegmentator.map_to_binary.class_map — the organ "
            "name->label-id mapping for the seg masks is unknown. Run this on the "
            "machine/env that produced the masks (where TotalSegmentator is "
            f"installed). Original error: {e}"
        )
    total = class_map["total"]  # {id: name}
    name_to_id = {name: int(i) for i, name in total.items()}
    missing = [o for o in ORGANS if o not in name_to_id]
    if missing:
        raise RuntimeError(
            f"TotalSegmentator 'total' map is missing organs {missing} — the "
            f"installed version differs from the one the XGBoost model expects. "
            f"Available names sample: {sorted(name_to_id)[:8]}..."
        )
    return {o: name_to_id[o] for o in ORGANS}


def assert_multilabel(seg: np.ndarray, min_unique: int = 8) -> None:
    """Fail loudly if `seg` looks binary rather than a per-organ multilabel mask.

    The repo's synthesis loss only ever binarises these masks, so we can't assume
    they retain per-organ ids. A per-organ mask has many distinct positive labels;
    a binary/union mask has {0, 1}. Without distinct ids the ROI-HU feature is
    meaningless, so this must be checked before extraction — never silently.
    """
    uniq = np.unique(seg)
    pos = uniq[uniq > 0]
    if len(pos) < min_unique:
        raise ValueError(
            f"Segmentation looks binary/near-binary (only {len(pos)} positive "
            f"labels: {pos[:10].tolist()}). ROI-HU features need a TotalSegmentator "
            f"MULTILABEL mask (one id per organ). Either regenerate the masks with "
            f"`ml=True` or run TotalSegmentator on the raw volume (--totalseg in "
            f"eval_xgb_phase.py)."
        )


def organ_median_hu(vol: np.ndarray, seg: np.ndarray,
                    label_map: Dict[str, int]) -> np.ndarray:
    """16-vector of median HU per organ, aligned to ORGANS.

    Matches the reference exactly: an organ with no voxels (or a median of
    exactly 0.0) becomes NaN — XGBoost consumes NaN natively as "organ absent".
    """
    if vol.shape != seg.shape:
        raise ValueError(f"volume {vol.shape} and seg {seg.shape} shapes differ")
    feats = np.empty(len(ORGANS), dtype=np.float64)
    for k, organ in enumerate(ORGANS):
        lbl = label_map[organ]
        vals = vol[seg == lbl]
        m = float(np.median(vals)) if vals.size else 0.0
        feats[k] = np.nan if m == 0.0 else m
    return feats


def build_feature_matrix(vol_seg_pairs, label_map: Optional[Dict[str, int]] = None,
                         check_multilabel: bool = True):
    """(N,16) feature matrix from an iterable of (volume_np, seg_np) pairs."""
    if label_map is None:
        label_map = load_ts_label_map()
    rows = []
    for i, (vol, seg) in enumerate(vol_seg_pairs):
        if check_multilabel and i == 0:
            assert_multilabel(seg)
        rows.append(organ_median_hu(vol, seg, label_map))
    return np.vstack(rows) if rows else np.empty((0, len(ORGANS)))


__all__ = [
    'ORGANS', 'load_ts_label_map', 'assert_multilabel',
    'organ_median_hu', 'build_feature_matrix',
]
