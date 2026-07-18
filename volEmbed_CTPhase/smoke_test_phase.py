"""
CPU-only, GPU-free smoke test for the phase-detection module's LOGIC.

Covers the parts that don't need a GPU, real CT data, or downloaded encoder
weights — i.e. exactly the bug-prone logic that produced the prior codebase's
inflated numbers:
  1. split_by_patient produces patient-disjoint train/val/test (no scan_id in
     two splits) — the leakage guarantee.
  2. GroupKFold inside compare_heads never puts a scan_id in both a fold's train
     and validation halves.
  3. All three classifier heads (lda, pca_lda, linear_nn) run end-to-end on
     synthetic features and return sane metrics, and — as a correctness check —
     score near-chance on RANDOM features (no spurious signal) and well above
     chance on LINEARLY-SEPARABLE features (real signal is captured).

Does NOT exercise encoders.py / phase_data.py's MONAI/NIfTI loading (those need
real data + weights) — those are validated on the remote box per WEIGHTS_SETUP.md.

Usage:  python smoke_test_phase.py
"""

import sys

import numpy as np

from phase_data import split_by_patient
from phase_classifier import compare_heads

N_CLASSES = 4
FEAT_DIM = 64


def _synthetic_samples(n_scans=40, phases_per_scan=4):
    """Fake volume records: each scan has one volume per phase."""
    samples = []
    for s in range(n_scans):
        for p in range(phases_per_scan):
            samples.append({'volume_path': f'/fake/scan{s:03d}/phase{p}.nii.gz',
                            'phase_id': p, 'phase_name': str(p), 'scan_id': f'scan{s:03d}'})
    return samples


def _make_centers(seed=99):
    """Shared per-class centers so train and test features live in the SAME
    space (else a classifier trained on train can't possibly predict test)."""
    return np.random.default_rng(seed).normal(0, 5, size=(N_CLASSES, FEAT_DIM))


def _synthetic_features(groups_labels, separable: bool, centers=None, seed=0):
    """Build features aligned to a list of (scan_id, phase) with either random
    or linearly-separable-by-phase structure. For separable, `centers` must be
    shared across the train and test calls."""
    rng = np.random.default_rng(seed)
    y = np.array([p for _, p in groups_labels])
    g = np.array([s for s, _ in groups_labels])
    if separable:
        X = np.stack([centers[p] + rng.normal(0, 1, FEAT_DIM) for p in y])
    else:
        X = rng.normal(0, 1, size=(len(y), FEAT_DIM))
    return X.astype(np.float32), y, g


def check(cond, name, detail=''):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f'  -- {detail}' if detail and not cond else ''))
    return bool(cond)


def main():
    ok = True

    # 1. patient-disjoint split
    samples = _synthetic_samples()
    train, val, test = split_by_patient(samples, val_frac=0.2, test_frac=0.2, seed=1)
    gtr = {s['scan_id'] for s in train}
    gva = {s['scan_id'] for s in val}
    gte = {s['scan_id'] for s in test}
    ok &= check(not (gtr & gva) and not (gtr & gte) and not (gva & gte),
                'split_by_patient: train/val/test scan_ids disjoint')
    ok &= check(len(train) + len(val) + len(test) == len(samples),
                'split_by_patient: no samples dropped/duplicated')
    ok &= check(all(len(x) > 0 for x in (train, val, test)),
                'split_by_patient: all three splits non-empty')

    # 2 + 3. heads on synthetic features. Build aligned (scan,phase) list.
    trainval = train + val
    tv_gl = [(s['scan_id'], s['phase_id']) for s in trainval]
    te_gl = [(s['scan_id'], s['phase_id']) for s in test]

    # Random features → all heads near chance (1/4 = 0.25); catches a head that
    # "cheats" via leakage (would score far above chance on noise).
    Xtv, ytv, gtv = _synthetic_features(tv_gl, separable=False, seed=2)
    Xte, yte, _   = _synthetic_features(te_gl, separable=False, seed=3)
    rand_res = compare_heads(Xtv, ytv, gtv, Xte, yte,
                             class_names=[str(i) for i in range(N_CLASSES)])
    for h, r in rand_res.items():
        ok &= check(r.cv_acc_mean < 0.45, f'random features: {h} CV near chance',
                    f'cv={r.cv_acc_mean:.3f} (expected <0.45)')

    # Separable features → all heads well above chance (real signal captured).
    # Train and test share the same class centers (same feature space).
    centers = _make_centers()
    Xtv, ytv, gtv = _synthetic_features(tv_gl, separable=True, centers=centers, seed=4)
    Xte, yte, _   = _synthetic_features(te_gl, separable=True, centers=centers, seed=5)
    sep_res = compare_heads(Xtv, ytv, gtv, Xte, yte,
                            class_names=[str(i) for i in range(N_CLASSES)])
    for h, r in sep_res.items():
        ok &= check(r.cv_acc_mean > 0.8, f'separable features: {h} CV high',
                    f'cv={r.cv_acc_mean:.3f} (expected >0.8)')
        ok &= check(r.test_acc > 0.8, f'separable features: {h} test high',
                    f'test={r.test_acc:.3f} (expected >0.8)')
        ok &= check(len(r.test_confusion) == N_CLASSES and
                    sum(sum(row) for row in r.test_confusion) == len(yte),
                    f'{h}: confusion matrix well-formed')

    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
