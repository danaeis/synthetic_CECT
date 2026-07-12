"""
Phase-detection experiment runner (replaces the old phase_detector.py).

Pipeline:
  1. Discover unique labelled volumes (no generation-pair duplication).
  2. Patient-grouped train/val/test split (no scan_id spans two splits).
  3. Extract frozen-encoder features once per volume (MedViT and/or DINOv3).
  4. Compare three classifier heads (shrinkage LDA, PCA→LDA, linear NN) under
     GroupKFold CV + a single held-out test evaluation.
  5. Save features (cache), per-head metrics, and a comparison bar chart.

Usage (remote GPU box):
    python run_phase_detection.py \
        --data_dir /path/to/volumes \
        --labels_csv /path/to/labels.csv \
        --encoders medvit dinov3 \
        --medvit_pretrained_path /path/to/medvit_small.pth \
        --output_dir phase_results

See WEIGHTS_SETUP.md for obtaining the MedViT / DINOv3 weights first.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phase_data import (find_phase_volumes, split_by_patient, build_phase_loader,
                        ID_TO_PHASE)
from phase_classifier import compare_heads

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)


def extract_features(encoder, samples, spatial_size, batch_size, device, max_slices):
    """Return (features (N,D), labels (N,), groups (N,)) aligned to `samples`.

    scan_id is carried through the loader so features stay aligned to their
    patient group for GroupKFold — the loader does NOT shuffle here.
    """
    import torch
    loader = build_phase_loader(samples, spatial_size=spatial_size,
                                batch_size=batch_size, augment=False,
                                shuffle=False)
    encoder = encoder.to(device).eval()
    feats, labels, groups = [], [], []
    with torch.no_grad():
        for batch in loader:
            vol = batch['volume'].to(device)
            f = encoder(vol).cpu().numpy()
            feats.append(f)
            labels.extend([int(p) for p in batch['phase']])
            groups.extend([str(s) for s in batch['scan_id']])
    return np.vstack(feats), np.array(labels), np.array(groups)


def _feature_cache_path(cache_dir: Path, enc_name: str, split: str) -> Path:
    return cache_dir / f'{enc_name}_{split}_features.npz'


def get_features(encoder, enc_name, split, samples, args, device):
    """Extract features, using an on-disk cache keyed by encoder+split."""
    cache = _feature_cache_path(Path(args.output_dir) / 'feature_cache', enc_name, split)
    if args.use_cache and cache.exists():
        d = np.load(cache, allow_pickle=True)
        log.info(f"  [{enc_name}/{split}] loaded {len(d['y'])} cached features")
        return d['X'], d['y'], d['g']
    X, y, g = extract_features(encoder, samples, tuple(args.spatial_size),
                               args.batch_size, device, args.max_slices)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, y=y, g=g)
    log.info(f"  [{enc_name}/{split}] extracted + cached {len(y)} features")
    return X, y, g


def plot_comparison(all_results: dict, out_path: Path):
    encoders = list(all_results.keys())
    heads = ['lda', 'pca_lda', 'linear_nn']
    fig, (ax_cv, ax_te) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(encoders)); w = 0.25
    for i, h in enumerate(heads):
        cv = [all_results[e][h].cv_acc_mean for e in encoders]
        cverr = [all_results[e][h].cv_acc_std for e in encoders]
        te = [all_results[e][h].test_acc for e in encoders]
        ax_cv.bar(x + (i - 1) * w, cv, w, yerr=cverr, capsize=4, label=h)
        ax_te.bar(x + (i - 1) * w, te, w, label=h)
    for ax, title in ((ax_cv, 'GroupKFold CV accuracy'), (ax_te, 'Held-out test accuracy')):
        ax.set_xticks(x); ax.set_xticklabels(encoders); ax.set_ylim(0, 1)
        ax.set_ylabel('accuracy'); ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=120, bbox_inches='tight'); plt.close()


def main():
    ap = argparse.ArgumentParser(description='Phase detection: encoder × classifier-head comparison')
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--labels_csv', default='')
    ap.add_argument('--file_tag', default='_deeds')
    ap.add_argument('--encoders', nargs='+', default=['medvit'], choices=['medvit', 'dinov3'])
    ap.add_argument('--medvit_pretrained_path', default='')
    ap.add_argument('--medvit_size', default='small', choices=['small', 'base'])
    ap.add_argument('--spatial_size', type=int, nargs=3, default=[128, 128, 128])
    ap.add_argument('--max_slices', type=int, default=32)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--test_frac', type=float, default=0.15)
    ap.add_argument('--cv_splits', type=int, default=5)
    ap.add_argument('--pca_components', type=int, default=40)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_dir', default='phase_results')
    ap.add_argument('--use_cache', action='store_true', default=True)
    args = ap.parse_args()

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    class_names = [ID_TO_PHASE[i] for i in sorted(ID_TO_PHASE)]

    # 1-2. data + patient-grouped split
    samples = find_phase_volumes(args.data_dir, args.labels_csv, args.file_tag)
    train, val, test = split_by_patient(samples, args.val_frac, args.test_frac, args.seed)
    trainval = train + val   # CV (GroupKFold) is done inside compare_heads over train+val

    from encoders import build_encoder
    all_results = {}
    for enc_name in args.encoders:
        log.info(f"=== Encoder: {enc_name} ===")
        encoder = build_encoder(enc_name, medvit_pretrained_path=args.medvit_pretrained_path,
                                max_slices=args.max_slices, medvit_size=args.medvit_size)
        Xtv, ytv, gtv = get_features(encoder, enc_name, 'trainval', trainval, args, device)
        Xte, yte, _   = get_features(encoder, enc_name, 'test', test, args, device)
        del encoder
        if device == 'cuda':
            torch.cuda.empty_cache()

        results = compare_heads(Xtv, ytv, gtv, Xte, yte, class_names=class_names,
                                n_splits=args.cv_splits, pca_components=args.pca_components)
        all_results[enc_name] = results

    # 5. persist metrics + plot
    summary = {
        enc: {h: {k: v for k, v in r.__dict__.items() if k != 'test_report'}
              for h, r in heads.items()}
        for enc, heads in all_results.items()
    }
    (out / 'phase_detection_summary.json').write_text(json.dumps(summary, indent=2))
    full = {enc: {h: r.__dict__ for h, r in heads.items()} for enc, heads in all_results.items()}
    (out / 'phase_detection_full.json').write_text(json.dumps(full, indent=2))
    plot_comparison(all_results, out / 'phase_head_comparison.png')

    log.info("=== Summary (test accuracy) ===")
    for enc, heads in all_results.items():
        for h, r in heads.items():
            log.info(f"  {enc:8s} {h:10s}  CV={r.cv_acc_mean:.3f}±{r.cv_acc_std:.3f}  test={r.test_acc:.3f}")
    log.info(f"Saved results to {out}/")


if __name__ == '__main__':
    main()
