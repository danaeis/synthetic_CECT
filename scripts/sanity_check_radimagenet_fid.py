#!/usr/bin/env python
"""
Sanity check for perceptual.py's FID(Rad) column reading suspiciously flat
(0.0-0.1 for nearly every model in the master table).

Two things could produce that: (a) RadImageNet ResNet50 features are genuinely
much less sensitive to this failure mode than ImageNet InceptionV3's — plausible,
Woodland et al. 2024 found RadImageNet-FID rankings volatile vs ImageNet-FID; or
(b) the features are near-degenerate on grayscale CT (far outside RadImageNet's
own training distribution) and collapse toward a similar point for ANY input,
which would make FID(Rad) uninformative regardless of how different the two
models actually are.

This distinguishes the two by extracting RadImageNet features for TWO GENUINELY
DIFFERENT models' outputs (pick your two most different rows in the curated
table, e.g. cyclegan vs dea_gan) plus the shared real set, and reporting:

  1. Per-dimension feature variance, both backbones side by side. If RadImageNet's
     variance is orders of magnitude smaller than ImageNet's on the SAME slices,
     that is direct evidence of collapse, not just "less sensitive."
  2. Mean pairwise cosine similarity within each feature set. Near 1.0 means the
     features are barely distinguishing individual slices at all.
  3. FID under both backbones for the same two models, side by side, so the
     "0.0-0.1 for everyone" pattern is either reproduced (real finding) or not
     (something about the full benchmark run differs from this check).

Usage (on the GPU host, same env as benchmark.py --perceptual):
    python scripts/sanity_check_radimagenet_fid.py \
        --radimagenet_weights /path/to/ResNet50.pt \
        --manifest_a cyclegan=../bench_ncct2cect/CycleGAN/results/vindr_nifti/manifest.csv \
        --manifest_b dea_gan=../bench_ncct2cect/results/vindr_dea_gan_nifti/manifest.csv \
        --hu_min -200 --hu_max 400 --n_slices 8
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import metrics as M
from perceptual import _load_inception, _load_radimagenet_resnet50


def load_case_slices(manifest_path, hu_min, hu_max, n_slices, min_body_frac=0.1):
    """A handful of body-containing slices from the FIRST case in a manifest,
    for both gen and real — enough to see whether features collapse, not a
    full re-score."""
    import nibabel as nib
    rows = list(csv.DictReader(open(manifest_path)))
    row = rows[0]
    gen = np.asanyarray(nib.load(row['gen_path']).dataobj).astype(np.float32)
    real = np.asanyarray(nib.load(row['real_path']).dataobj).astype(np.float32)
    g01 = M.to_unit(gen, hu_min, hu_max)
    r01 = M.to_unit(real, hu_min, hu_max)
    bmask = M.body_mask(real)
    frac = bmask.reshape(-1, bmask.shape[-1]).mean(axis=0) if bmask.ndim == 3 else bmask.mean(axis=tuple(range(bmask.ndim - 1)))
    idx = np.flatnonzero(frac >= min_body_frac)
    if idx.size == 0:
        idx = np.arange(g01.shape[-1])
    idx = idx[np.linspace(0, idx.size - 1, min(n_slices, idx.size)).astype(int)]
    return g01[..., idx], r01[..., idx], row['gen_path']


def stack_rgb(vol01, torch):
    v = np.moveaxis(vol01, -1, 0)
    t = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
    return t.unsqueeze(1).repeat(1, 3, 1, 1)


def report_backbone(name, feats_fn, device, torch, sets):
    """sets: {label: (n,3,H,W) tensor}. Prints per-set variance/self-similarity
    and pairwise Frechet distances between every pair of sets."""
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    feats = {}
    with torch.no_grad():
        for label, t in sets.items():
            f = feats_fn(t.to(device)).cpu().numpy()
            feats[label] = f
            var = f.var(axis=0).mean()
            norm = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
            cos = (norm @ norm.T)
            iu = np.triu_indices(len(cos), k=1)
            mean_cos = cos[iu].mean() if len(iu[0]) else float('nan')
            print(f"  {label:14s} n={f.shape[0]:3d}  dim={f.shape[1]:5d}  "
                  f"mean per-dim variance={var:.6g}  mean pairwise cosine sim={mean_cos:.4f}")
    from perceptual import frechet_distance
    labels = list(feats)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = feats[labels[i]], feats[labels[j]]
            if a.shape[0] < 2 or b.shape[0] < 2:
                continue
            fid = frechet_distance(a.mean(0), np.cov(a, rowvar=False),
                                   b.mean(0), np.cov(b, rowvar=False))
            print(f"  FID {labels[i]} vs {labels[j]}: {fid:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--radimagenet_weights', type=Path, required=True)
    ap.add_argument('--manifest_a', required=True, help='name=path.csv')
    ap.add_argument('--manifest_b', required=True, help='name=path.csv')
    ap.add_argument('--hu_min', type=float, default=-200.0)
    ap.add_argument('--hu_max', type=float, default=400.0)
    ap.add_argument('--n_slices', type=int, default=8)
    ap.add_argument('--device', default=None)
    a = ap.parse_args()

    import torch
    device = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    name_a, _, path_a = a.manifest_a.partition('=')
    name_b, _, path_b = a.manifest_b.partition('=')

    ga, ra, case_a = load_case_slices(path_a, a.hu_min, a.hu_max, a.n_slices)
    gb, rb, case_b = load_case_slices(path_b, a.hu_min, a.hu_max, a.n_slices)
    print(f"case A ({name_a}): {Path(case_a).name}  |  case B ({name_b}): {Path(case_b).name}")

    sets = {
        f'{name_a}_gen': stack_rgb(ga, torch),
        f'{name_b}_gen': stack_rgb(gb, torch),
        'real_A': stack_rgb(ra, torch),
        'real_B': stack_rgb(rb, torch),
    }

    inception_feats, inception_backend = _load_inception(device)
    print(f"\nImageNet backend: {inception_backend}")
    report_backbone('ImageNet InceptionV3 (existing FID column)', inception_feats, device, torch, sets)

    rad_feats, rad_backend = _load_radimagenet_resnet50(a.radimagenet_weights, device)
    print(f"\nRadImageNet backend: {rad_backend}")
    report_backbone('RadImageNet ResNet50 (FID(Rad) column)', rad_feats, device, torch, sets)

    print(f"\n{'=' * 70}\nHOW TO READ THIS\n{'=' * 70}")
    print("If RadImageNet's per-dim variance is many orders of magnitude below")
    print("ImageNet's on these SAME slices, and/or its pairwise cosine similarity")
    print("is close to 1.0 across clearly different images (different models,")
    print("gen vs real), that's feature collapse -- FID(Rad) numbers are not")
    print("trustworthy at all, not just 'less sensitive'. If variance is")
    print("comparable and cosine similarity is well below 1.0, RadImageNet is")
    print("extracting real signal and the 0.0-0.1 FID(Rad) values in the master")
    print("table likely mean this backbone genuinely doesn't separate these")
    print("models much -- a substantive (if disappointing) finding, not a bug.")


if __name__ == '__main__':
    main()
