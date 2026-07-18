"""
Evaluate the shipped ROI-guided XGBoost phase classifier (Hou 2025) on the SAME
held-out test split the deep-encoder baseline uses — an apples-to-apples SOTA
number for the thesis.

Pipeline per test volume:
  1. load the RAW HU volume (NOT the [0,1]-normalised tensor the encoders use —
     XGBoost was trained on absolute HU),
  2. load its co-registered TotalSegmentator multilabel mask
     (`*_deeds_seg_reg.nii.gz`), or run TotalSegmentator if `--totalseg`,
  3. median HU per organ -> 16-dim feature (roi_hu_features),
  4. ensemble predict_proba across the 5 XGBoost folds in the .pkl, argmax.

Reuses `find_phase_volumes` + `split_by_patient` (same seed) so the test cases
are IDENTICAL to run_phase_detection.py's — the only thing that changes is the
model. Reports accuracy / macro-F1 / per-class report / confusion matrix.

Usage (on the box with the data + totalsegmentator + xgboost):
    python eval_xgb_phase.py \
        --data_dir  .../B2_deeds__aligned \
        --labels_csv .../labels.csv \
        --xgb_model  ../../phase_results/xgb_vindr.pkl \
        --output_dir ../../phase_results
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np

from phase_data import find_phase_volumes, split_by_patient, ID_TO_PHASE
from roi_hu_features import ORGANS, load_ts_label_map, organ_median_hu, assert_multilabel

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)


def _seg_path_for(volume_path: str, file_tag: str) -> Path:
    """Mirror dataset.py: '<x>_deeds.nii.gz' -> '<x>_deeds_seg_reg.nii.gz'."""
    return Path(volume_path.replace(f'{file_tag}.nii.gz', f'{file_tag}_seg_reg.nii.gz'))


def _load_nifti(path) -> np.ndarray:
    import nibabel as nib
    return np.asarray(nib.load(str(path)).get_fdata())


def _features_from_disk_mask(sample, file_tag, label_map, checked):
    seg_path = _seg_path_for(sample['volume_path'], file_tag)
    if not seg_path.exists():
        return None, f"no seg mask on disk ({seg_path.name})"
    vol = _load_nifti(sample['volume_path'])
    seg = _load_nifti(seg_path).round().astype(np.int32)
    if seg.shape != vol.shape:
        return None, f"seg/vol shape mismatch {seg.shape} vs {vol.shape}"
    if not checked['done']:
        assert_multilabel(seg)          # fail loudly once if masks are binary
        checked['done'] = True
    return organ_median_hu(vol, seg, label_map), None


def _features_from_totalseg(sample):
    """Fallback: segment the raw volume on the fly (reference's own path)."""
    import nibabel as nib
    from totalsegmentator.python_api import totalsegmentator
    ct_img = nib.load(sample['volume_path'])
    _, stats = totalsegmentator(ct_img, None, ml=True, fast=True, statistics=True,
                                roi_subset=None, statistics_exclude_masks_at_border=False,
                                quiet=True, stats_aggregation="median")
    feats = np.array([stats[o]["intensity"] for o in ORGANS], dtype=np.float64)
    feats[feats == 0.0] = np.nan
    return feats


def main():
    ap = argparse.ArgumentParser(description='Evaluate ROI-guided XGBoost phase classifier on the test split')
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--labels_csv', default='')
    ap.add_argument('--file_tag', default='_deeds')
    ap.add_argument('--xgb_model', required=True, help='path to xgb_*.pkl (list of {fold, model})')
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--test_frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split', default='test', choices=['test', 'trainval', 'all'])
    ap.add_argument('--totalseg', action='store_true',
                    help='segment each volume with TotalSegmentator instead of reading _seg_reg masks')
    ap.add_argument('--output_dir', default='phase_results')
    args = ap.parse_args()

    from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                                 confusion_matrix)

    # --- same split as the deep-encoder baseline ---
    samples = find_phase_volumes(args.data_dir, args.labels_csv, args.file_tag)
    train, val, test = split_by_patient(samples, args.val_frac, args.test_frac, args.seed)
    eval_samples = {'test': test, 'trainval': train + val, 'all': samples}[args.split]
    log.info(f"Evaluating XGBoost on split='{args.split}' ({len(eval_samples)} volumes)")

    label_map = None if args.totalseg else load_ts_label_map()
    checked = {'done': False}

    X, y, kept = [], [], []
    n_skipped = 0
    for s in eval_samples:
        try:
            if args.totalseg:
                feats = _features_from_totalseg(s)
            else:
                feats, err = _features_from_disk_mask(s, args.file_tag, label_map, checked)
                if feats is None:
                    n_skipped += 1
                    log.warning(f"  skip {Path(s['volume_path']).name}: {err}")
                    continue
        except Exception as e:  # noqa: BLE001
            n_skipped += 1
            log.warning(f"  skip {Path(s['volume_path']).name}: {e}")
            continue
        X.append(feats); y.append(s['phase_id']); kept.append(s)

    if not X:
        raise RuntimeError("No volumes produced features — check masks / --totalseg.")
    X = np.vstack(X); y = np.array(y)
    if n_skipped:
        log.warning(f"Skipped {n_skipped}/{len(eval_samples)} volumes without usable features")

    # --- ensemble the folds in the pkl ---
    # Each fold's predict_proba columns follow ITS OWN classes_ (a fold that
    # never saw 'delayed' emits 3 columns, not 4). Re-map every fold into a fixed
    # (N, 4) [non-contrast, arterial, venous, delayed] layout before averaging so
    # columns line up, then argmax gives a phase id directly.
    all_models = pickle.load(open(args.xgb_model, 'rb'))
    n_ph = len(ID_TO_PHASE)
    acc_probs = np.zeros((len(X), n_ph))
    for m in all_models:
        clf = m["model"]
        p = clf.predict_proba(X)                       # (N, len(classes_))
        aligned = np.zeros((len(X), n_ph))
        for col, cls in enumerate(clf.classes_):
            aligned[:, int(cls)] = p[:, col]
        acc_probs += aligned
    probs = acc_probs / len(all_models)
    y_pred = probs.argmax(axis=1)

    present = sorted(set(y.tolist()) | set(y_pred.tolist()))
    names = [ID_TO_PHASE[i] for i in present]
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro', zero_division=0)
    report = classification_report(y, y_pred, labels=present, target_names=names,
                                   zero_division=0, output_dict=True)
    conf = confusion_matrix(y, y_pred, labels=present).tolist()

    log.info(f"[xgb_roi] {args.split}  acc={acc:.4f}  macroF1={f1:.4f}  (n={len(y)})")
    log.info(f"  confusion (rows=true {names}):")
    for r in conf:
        log.info(f"    {r}")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    result = {
        'model': str(args.xgb_model), 'split': args.split, 'n': int(len(y)),
        'accuracy': float(acc), 'macro_f1': float(f1),
        'labels': names, 'confusion': conf, 'report': report,
        'n_skipped': int(n_skipped), 'used_totalseg': bool(args.totalseg),
    }
    (out / f'xgb_roi_eval_{args.split}.json').write_text(json.dumps(result, indent=2))
    log.info(f"Saved -> {out / f'xgb_roi_eval_{args.split}.json'}")


if __name__ == '__main__':
    main()
