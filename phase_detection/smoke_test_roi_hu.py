"""
Synthetic smoke test for the ROI-HU phase losses — runs WITHOUT the real data,
TotalSegmentator, xgboost or nibabel (mirrors smoke_test_phase.py's pattern).

Validates: prototype construction, differentiable per-organ mean HU, Option A
(ROIHUPhaseLoss) shape + gradient flow + that a volume matching the target
prototype gives a near-zero loss, and Option B (distilled head + CDW-CE).
"""

import tempfile
from pathlib import Path

import numpy as np
import torch

from roi_hu_phase_loss import (compute_phase_prototypes, save_prototypes,
                               organ_mean_hu_torch, ROIHUPhaseLoss,
                               fit_distilled_head, PhaseClassifierLoss)

O = 16
LABEL_IDS = list(range(1, O + 1))          # fake organ ids 1..16
N_PHASES = 4
D = H = W = 8


def _fake_case(phase, rng):
    """A seg with all 16 organs present, and a volume whose per-organ mean HU is
    a phase-dependent signature + small noise (so prototypes are separable)."""
    seg = np.zeros((D, H, W), np.int32)
    vol = rng.normal(0, 5, size=(D, H, W)).astype(np.float32)
    flat = np.arange(D * H * W).reshape(D, H, W)
    for k, lbl in enumerate(LABEL_IDS):
        region = (flat % O) == k            # disjoint voxel sets per organ
        seg[region] = lbl
        base = 40 + 30 * phase + 8 * k      # phase-dependent organ HU
        vol[region] += base
    return vol, seg


def main():
    rng = np.random.default_rng(0)
    cases = [(*_fake_case(p, rng), p) for p in range(N_PHASES) for _ in range(6)]

    # 1) prototypes
    d = compute_phase_prototypes(iter(cases), LABEL_IDS, n_phases=N_PHASES)
    assert d['prototypes'].shape == (N_PHASES, O)
    assert not np.isnan(d['prototypes']).any(), "all organs present -> no NaN expected"
    tmp = Path(tempfile.mkdtemp())
    save_prototypes(d, tmp / 'proto.npz')

    # 2) differentiable per-organ mean HU
    vol, seg, _ = cases[0]
    vt = torch.tensor(vol)[None, None].requires_grad_(True)
    st = torch.tensor(seg)[None, None]
    means, present = organ_mean_hu_torch(vt, st, torch.tensor(LABEL_IDS))
    assert means.shape == (1, O) and present.all()
    means.sum().backward()
    assert vt.grad is not None and torch.isfinite(vt.grad).all()

    # 3) Option A — loss is ~0 when the volume matches the target prototype,
    #    and > 0 for a wrong phase.
    loss_fn = ROIHUPhaseLoss(str(tmp / 'proto.npz'), hu_scale=100.0)
    v0 = torch.tensor(cases[0][0])[None, None].requires_grad_(True)   # true phase 0
    s0 = torch.tensor(cases[0][1])[None, None]
    l_right = loss_fn(v0, s0, torch.tensor([0]))
    l_wrong = loss_fn(v0, s0, torch.tensor([3]))
    l_right.backward()
    assert v0.grad is not None and torch.isfinite(v0.grad).all()
    assert l_right.item() < l_wrong.item(), (l_right.item(), l_wrong.item())
    print(f"[A] loss(correct phase)={l_right.item():.4f} < loss(wrong)={l_wrong.item():.4f}  OK")

    # 3b) input_window path ([0,1] -> HU) runs and stays finite
    lw = ROIHUPhaseLoss(str(tmp / 'proto.npz'), input_window=(-160.0, 400.0))
    assert torch.isfinite(lw(torch.rand(1, 1, D, H, W), s0, torch.tensor([1]))).all()

    # 4) Option B — distilled head + CDW-CE loss
    X = np.vstack([np.array([vol[seg == l].mean() for l in LABEL_IDS])
                   for vol, seg, _ in cases])
    y = np.array([p for *_, p in cases])
    soft = np.eye(N_PHASES)[y]                       # one-hot as stand-in for xgb proba
    head = fit_distilled_head(X, soft, LABEL_IDS, epochs=200)
    torch.save(head, tmp / 'head.pt')
    clf = PhaseClassifierLoss(str(tmp / 'head.pt'))
    v = torch.tensor(cases[0][0])[None, None].requires_grad_(True)
    lc_right = clf(v, s0, torch.tensor([0]))
    lc_wrong = clf(v, s0, torch.tensor([3]))
    lc_right.backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert lc_right.item() < lc_wrong.item(), (lc_right.item(), lc_wrong.item())
    print(f"[B] CDW-CE(correct)={lc_right.item():.4f} < CDW-CE(wrong)={lc_wrong.item():.4f}  OK")

    print("\nAll ROI-HU smoke checks passed.")


if __name__ == '__main__':
    main()
