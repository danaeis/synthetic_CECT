"""
Phase-classification data loading — deduplicated, patient-grouped.

Fixes the two data bugs found in the prior codebase (see RETRAIN_PLAN.md):
  1. NO generation pairs. The old pipeline fed the phase classifier the
     `input_path` of every directed phase→phase generation pair, so each unique
     volume was duplicated ~2-3× (same label). Here every volume contributes
     exactly ONE labelled sample.
  2. NO cross-split leakage. Train/val/test are split by scan_id (patient), so
     no patient's volumes ever straddle two splits — and because there are no
     duplicates, a volume can't leak into its own held-out fold either.

Volume discovery mirrors the *proven* logic in this repo's synthesis
`dataset.py` (`find_pairs_and_split`): same labels.csv schema
(StudyInstanceUID / SeriesInstanceUID / Label), same `*{file_tag}.nii.gz`
glob, same `series_id = stem.split('_')[1]` convention, same `_infer_phase`
filename fallback — so it reads the identical data the synthesis pipeline
already reads, just keeping ALL phase volumes as individual labelled samples
instead of only NCCT/target pairs.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Canonical phase vocabulary → integer id. Matches the label strings used by
# this repo's synthesis dataset.py / labels.csv (lowercased).
PHASE_TO_ID: Dict[str, int] = {
    'non-contrast': 0,
    'arterial':     1,
    'venous':       2,
    'delayed':      3,
}
ID_TO_PHASE: Dict[int, str] = {v: k for k, v in PHASE_TO_ID.items()}

# Filename-keyword fallback when a volume isn't found in labels.csv — mirrors
# synthesis dataset.py::_infer_phase (same keyword sets), collapsed to the
# canonical phase names above ('portal' folds into 'venous').
_PHASE_KW: Dict[str, List[str]] = {
    'non-contrast': ['noncontrast', 'non-contrast', 'pre', 'baseline', 'native', 'nc', 'nce', 'noncon'],
    'arterial':     ['arterial', 'art', 'early', 'phase1', 'p1'],
    'venous':       ['venous', 'portal', 'pv', 'phase2', 'p2', 'late'],
    'delayed':      ['delayed', 'delay', 'equilibrium', 'phase3', 'p3'],
}


def _infer_phase(name: str) -> Optional[str]:
    n = name.lower()
    for phase, kws in _PHASE_KW.items():
        if any(k in n for k in kws):
            return phase
    return None


def _normalize_label(label: str) -> Optional[str]:
    """Map a raw labels.csv Label string to a canonical phase name (or None)."""
    s = str(label).strip().lower()
    if s in PHASE_TO_ID:
        return s
    # tolerate common variants / synonyms
    aliases = {
        'nc': 'non-contrast', 'noncontrast': 'non-contrast', 'pre': 'non-contrast',
        'precontrast': 'non-contrast', 'non contrast': 'non-contrast',
        'art': 'arterial', 'a': 'arterial',
        'pv': 'venous', 'portal': 'venous', 'portal venous': 'venous',
        'portal-venous': 'venous', 'portal_venous': 'venous', 'v': 'venous',
        'delay': 'delayed', 'd': 'delayed',
    }
    return aliases.get(s)


# ---------------------------------------------------------------------------
# Volume discovery — ONE labelled sample per unique volume
# ---------------------------------------------------------------------------

def find_phase_volumes(
    data_dir: str,
    labels_csv: str = '',
    file_tag: str = '_deeds',
) -> List[Dict]:
    """
    Scan `data_dir` for per-case subdirectories and return one dict per unique
    labelled volume: {'volume_path', 'phase_id', 'phase_name', 'scan_id'}.

    A volume is included only if a phase label can be resolved for it (from
    labels.csv, else from the filename via `_infer_phase`); unlabelled or
    segmentation-mask files are skipped. No pairs, no duplication.
    """
    data_dir = Path(data_dir)

    phase_map: Dict[str, Dict[str, str]] = {}
    if labels_csv and Path(labels_csv).exists():
        import pandas as pd
        df = pd.read_csv(labels_csv)
        for _, row in df.iterrows():
            case  = str(row['StudyInstanceUID'])
            ser   = str(row['SeriesInstanceUID'])
            label = str(row['Label']).lower()
            phase_map.setdefault(case, {})[ser] = label
        log.info(f"Loaded labels CSV: {len(phase_map)} cases")
    else:
        log.warning(f"No labels CSV at '{labels_csv}' — falling back to filename phase inference")

    samples: List[Dict] = []
    n_skipped_unlabelled = 0
    seen_paths = set()   # guard against ever emitting the same volume twice

    for case_dir in sorted(data_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name

        for f in sorted(case_dir.glob(f'*{file_tag}.nii.gz')):
            if '_seg' in f.name:
                continue
            path = str(f)
            if path in seen_paths:
                continue

            stem = f.name.replace(f'{file_tag}.nii.gz', '').replace('.nii', '')
            parts = stem.split('_')
            series_id = parts[1] if len(parts) >= 2 else stem

            raw = None
            if case_id in phase_map and series_id in phase_map[case_id]:
                raw = phase_map[case_id][series_id]
            phase_name = _normalize_label(raw) if raw is not None else _infer_phase(f.name)

            if phase_name is None or phase_name not in PHASE_TO_ID:
                n_skipped_unlabelled += 1
                continue

            seen_paths.add(path)
            samples.append({
                'volume_path': path,
                'phase_id':    PHASE_TO_ID[phase_name],
                'phase_name':  phase_name,
                'scan_id':     case_id,
            })

    if not samples:
        raise RuntimeError(
            f"No labelled phase volumes found under {data_dir} "
            f"(file_tag='{file_tag}', labels loaded={bool(phase_map)}). "
            f"Check the data path, file_tag, and labels.csv columns "
            f"(StudyInstanceUID/SeriesInstanceUID/Label)."
        )

    # Report distribution + confirm no duplicate paths slipped through.
    assert len(seen_paths) == len(samples), "internal error: duplicate volume emitted"
    dist = {ID_TO_PHASE[i]: 0 for i in sorted(ID_TO_PHASE)}
    for s in samples:
        dist[s['phase_name']] += 1
    n_scans = len({s['scan_id'] for s in samples})
    log.info(f"Found {len(samples)} unique labelled volumes across {n_scans} scans")
    log.info(f"  phase distribution: {dist}")
    if n_skipped_unlabelled:
        log.info(f"  skipped {n_skipped_unlabelled} unlabelled/unknown-phase volumes")
    return samples


# ---------------------------------------------------------------------------
# Patient-grouped 3-way split (no scan_id in more than one split)
# ---------------------------------------------------------------------------

def split_by_patient(
    samples: List[Dict],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split samples into (train, val, test) at the scan_id (patient) level so no
    patient's volumes appear in more than one split. Splitting is on the set of
    unique scan_ids, then samples are assigned by their scan_id — guaranteeing
    zero patient leakage across splits.
    """
    scan_ids = sorted({s['scan_id'] for s in samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(scan_ids)

    n = len(scan_ids)
    n_test = max(1, int(round(n * test_frac)))
    n_val  = max(1, int(round(n * val_frac)))
    if n_test + n_val >= n:
        raise ValueError(
            f"val_frac+test_frac too large for {n} scans "
            f"(test={n_test}, val={n_val}); leaves no training scans."
        )

    test_ids  = set(scan_ids[:n_test])
    val_ids   = set(scan_ids[n_test:n_test + n_val])
    train_ids = set(scan_ids[n_test + n_val:])

    train = [s for s in samples if s['scan_id'] in train_ids]
    val   = [s for s in samples if s['scan_id'] in val_ids]
    test  = [s for s in samples if s['scan_id'] in test_ids]

    # Hard guarantee: the three scan-id sets are disjoint.
    assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids)
    log.info(
        f"Patient-grouped split: "
        f"train={len(train)} vols/{len(train_ids)} scans | "
        f"val={len(val)} vols/{len(val_ids)} scans | "
        f"test={len(test)} vols/{len(test_ids)} scans"
    )
    return train, val, test


# ---------------------------------------------------------------------------
# MONAI loader — resized whole volumes in [0, 1], with phase label + scan_id
# ---------------------------------------------------------------------------

def build_phase_loader(
    samples: List[Dict],
    spatial_size: Tuple[int, int, int] = (128, 128, 128),
    batch_size: int = 4,
    augment: bool = False,
    num_workers: int = 2,
    shuffle: bool = False,
    hu_min: float = -1000.0,
    hu_max: float = 1000.0,
):
    """
    Build a MONAI DataLoader over phase-labelled volumes. Each batch yields:
        {'volume': (B,1,D,H,W) float in [0,1], 'phase': (B,), 'scan_id': [...]}

    Validity of file paths is checked up front (missing files dropped with a
    warning) so a bad path can't crash mid-epoch during feature extraction.
    """
    import nibabel as nib
    from monai.data import Dataset, DataLoader
    from monai.transforms import (
        LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
        Resized, ToTensord, Compose,
        RandFlipd, RandShiftIntensityd,
    )

    # Drop unreadable/missing files before building the dataset.
    valid = []
    for s in samples:
        p = s['volume_path']
        if not Path(p).exists():
            log.warning(f"  missing volume, skipping: {p}")
            continue
        valid.append({'volume': p, 'phase': int(s['phase_id']), 'scan_id': s['scan_id']})
    if not valid:
        raise RuntimeError("build_phase_loader: no valid volumes to load")

    tfs = [
        LoadImaged(keys=['volume']),
        EnsureChannelFirstd(keys=['volume']),
        ScaleIntensityRanged(keys=['volume'], a_min=hu_min, a_max=hu_max,
                             b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=['volume'], spatial_size=spatial_size, mode='trilinear'),
        ToTensord(keys=['volume', 'phase']),
    ]
    if augment:
        tfs[4:4] = [
            RandFlipd(keys=['volume'], prob=0.2, spatial_axis=2),
            RandShiftIntensityd(keys=['volume'], prob=0.2, offsets=0.05),
        ]

    ds = Dataset(valid, transform=Compose(tfs))
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
    )


__all__ = [
    'PHASE_TO_ID', 'ID_TO_PHASE',
    'find_phase_volumes', 'split_by_patient', 'build_phase_loader',
]
