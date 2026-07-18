"""
Phase-consistency losses driven by ROI (organ) HU — a physically grounded,
fully-differentiable alternative to using a 0.84-accuracy black-box classifier
as a generator loss.

Why this instead of the deep classifier / sklearn-LDA bridge:
  * The mean HU inside a fixed organ mask is a differentiable function of the
    generated volume — gradients flow straight to the generator, no frozen 2-D
    encoder and its [0,1] preprocessing in the way.
  * It targets the exact physical quantity that DEFINES contrast phase (per-organ
    enhancement), so it injects a clean signal, not classifier noise.

Two losses, both keyed on a precomputed artifact (prototypes .npz) so the train
loop needs neither TotalSegmentator nor xgboost:

  Option A — ROIHUPhaseLoss (recommended, robust):
      smooth_L1( per-organ mean HU of G(x) , prototype[target_phase] )
    Organs absent from a case's mask (or from a phase prototype) are dropped from
    the term, so it never penalises against missing anatomy.

  Option B — PhaseClassifierLoss (optional, "does it CLASSIFY as target phase"):
      a small MLP fit on the 16-dim organ-HU features (distilled from the XGBoost
      soft labels), reconstructed as a frozen torch head, driven by a
      class-distance-weighted cross-entropy toward the requested phase. Ordinal
      distance is meaningful here (noncontrast<arterial<venous<delayed), so a
      wrong-by-one-phase mistake is penalised less than wrong-by-three.

Note: prototypes and the loss use per-organ *mean* HU (median has ~zero gradient
a.e.), unlike the XGBoost feature which uses median. The two are separate,
internally-consistent objects — don't mix a median prototype into this loss.
"""

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from roi_hu_features import ORGANS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prototype construction (offline, numpy) — run once on the box
# ---------------------------------------------------------------------------

def _organ_mean_hu_np(vol: np.ndarray, seg: np.ndarray, label_ids: List[int]) -> np.ndarray:
    """(16,) per-organ MEAN HU (NaN where the organ has no voxels)."""
    out = np.full(len(label_ids), np.nan, dtype=np.float64)
    for k, lbl in enumerate(label_ids):
        m = seg == lbl
        if m.any():
            out[k] = float(vol[m].mean())
    return out


def compute_phase_prototypes(
    vol_seg_phase: Iterable[Tuple[np.ndarray, np.ndarray, int]],
    label_ids: List[int],
    n_phases: int = 4,
) -> Dict[str, np.ndarray]:
    """Per-phase organ-HU prototype = median across training volumes of each
    organ's mean HU. Median across cases is robust to segmentation outliers.

    Returns {'prototypes': (n_phases,16), 'counts': (n_phases,16), 'label_ids', 'organs'}.
    A prototype entry is NaN if no training volume of that phase had the organ.
    """
    per_phase = {p: [] for p in range(n_phases)}
    for vol, seg, phase in vol_seg_phase:
        per_phase[int(phase)].append(_organ_mean_hu_np(vol, seg, label_ids))

    protos = np.full((n_phases, len(label_ids)), np.nan, dtype=np.float64)
    counts = np.zeros((n_phases, len(label_ids)), dtype=np.int64)
    for p, rows in per_phase.items():
        if not rows:
            continue
        M = np.vstack(rows)                       # (n_cases, 16), may contain NaN
        counts[p] = np.sum(~np.isnan(M), axis=0)
        protos[p] = np.nanmedian(M, axis=0) if M.shape[0] else np.nan
    return {'prototypes': protos, 'counts': counts,
            'label_ids': np.array(label_ids), 'organs': np.array(ORGANS)}


def save_prototypes(d: Dict[str, np.ndarray], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **d)
    log.info(f"Saved phase prototypes -> {path}")


# ---------------------------------------------------------------------------
# Differentiable per-organ mean HU (torch)
# ---------------------------------------------------------------------------

def organ_mean_hu_torch(vol: torch.Tensor, seg: torch.Tensor,
                        label_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-organ mean HU + presence mask, differentiable w.r.t. `vol`.

    vol/seg: (B,1,D,H,W) — vol is the generated volume in HU (or whatever space
    the prototypes were built in), seg is an integer multilabel mask on the same
    grid. Returns (means (B,O), present (B,O) bool). Absent organs -> mean 0,
    present False (caller must ignore them).
    """
    assert vol.shape == seg.shape, f"{vol.shape} vs {seg.shape}"
    B = vol.shape[0]
    O = len(label_ids)
    means = vol.new_zeros(B, O)
    present = torch.zeros(B, O, dtype=torch.bool, device=vol.device)
    v = vol.reshape(B, -1)
    s = seg.reshape(B, -1)
    for k in range(O):
        m = (s == label_ids[k]).to(v.dtype)       # (B, N)
        cnt = m.sum(dim=1)                          # (B,)
        ok = cnt > 0
        summed = (v * m).sum(dim=1)
        means[:, k] = torch.where(ok, summed / cnt.clamp_min(1.0), means[:, k])
        present[:, k] = ok
    return means, present


class ROIHUPhaseLoss(nn.Module):
    """Option A — smooth-L1 between G(x)'s per-organ mean HU and the target-phase
    prototype, over organs present in BOTH.

    Args:
        prototypes_npz: path produced by compute_phase_prototypes / save_prototypes.
        hu_scale: HU are divided by this before smooth_l1 so the loss is O(1)
                  (100 HU ~= a big enhancement difference).
        input_window: if the generator emits [0,1] instead of HU, pass the
                  (a_min, a_max) HU window it was normalised with; the loss maps
                  input back to HU as x*(a_max-a_min)+a_min before comparing.
    """

    def __init__(self, prototypes_npz: str, hu_scale: float = 100.0,
                 input_window: Optional[Tuple[float, float]] = None):
        super().__init__()
        d = np.load(prototypes_npz, allow_pickle=True)
        protos = d['prototypes'].astype(np.float32)             # (P,O) with NaN
        valid = ~np.isnan(protos)
        self.register_buffer('proto', torch.tensor(np.nan_to_num(protos)))
        self.register_buffer('proto_valid', torch.tensor(valid))
        self.register_buffer('label_ids', torch.tensor(d['label_ids'].astype(np.int64)))
        self.hu_scale = float(hu_scale)
        self.input_window = input_window
        log.info(f"ROIHUPhaseLoss | prototypes={Path(prototypes_npz).name} "
                 f"| phases={protos.shape[0]} | organs={protos.shape[1]} "
                 f"| window={input_window}")

    def _to_hu(self, vol: torch.Tensor) -> torch.Tensor:
        if self.input_window is None:
            return vol
        a, b = self.input_window
        return vol * (b - a) + a

    def forward(self, gen_volume: torch.Tensor, seg: torch.Tensor,
                target_phase: torch.Tensor) -> torch.Tensor:
        """gen_volume/seg: (B,1,D,H,W); target_phase: (B,) long. Scalar loss."""
        vol = self._to_hu(gen_volume)
        means, present = organ_mean_hu_torch(vol, seg, self.label_ids)   # (B,O)
        tgt = self.proto[target_phase]                                    # (B,O)
        tgt_valid = self.proto_valid[target_phase]                       # (B,O)
        use = present & tgt_valid                                        # (B,O)
        diff = F.smooth_l1_loss(means / self.hu_scale, tgt / self.hu_scale,
                                reduction='none')
        denom = use.sum().clamp_min(1)
        return (diff * use).sum() / denom


# ---------------------------------------------------------------------------
# Option B — distilled differentiable classifier head + CDW-CE loss
# ---------------------------------------------------------------------------

class _OrganHUMLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def fit_distilled_head(
    X: np.ndarray, soft_targets: np.ndarray, label_ids: List[int],
    hidden: int = 64, epochs: int = 400, lr: float = 1e-3, seed: int = 0,
) -> Dict:
    """Distill XGBoost soft probabilities into a small MLP on 16-dim organ-HU.

    X: (N,16) with NaN for absent organs (imputed with per-feature train mean).
    soft_targets: (N,P) XGBoost predict_proba (or one-hot labels). Returns a dict
    with the fitted state + imputation/standardisation buffers for reconstruction.
    """
    torch.manual_seed(seed)
    X = X.astype(np.float32)
    fmean = np.nanmean(X, axis=0)
    fmean = np.where(np.isnan(fmean), 0.0, fmean)
    Xf = np.where(np.isnan(X), fmean, X)
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    Xs = torch.tensor((Xf - mu) / sd)
    Y = torch.tensor(soft_targets.astype(np.float32))

    model = _OrganHUMLP(X.shape[1], soft_targets.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        logp = F.log_softmax(model(Xs), dim=1)
        loss = F.kl_div(logp, Y, reduction='batchmean')   # distillation
        loss.backward(); opt.step()
    log.info(f"Distilled head: final KL={loss.item():.4f} on {len(X)} samples")
    return {'state_dict': model.state_dict(), 'in_dim': X.shape[1],
            'n_classes': soft_targets.shape[1], 'hidden': hidden,
            'impute_mean': fmean, 'mu': mu, 'sd': sd,
            'label_ids': np.array(label_ids)}


class PhaseClassifierLoss(nn.Module):
    """Option B — class-distance-weighted CE toward the target phase, using the
    distilled organ-HU head. Reconstructs the frozen MLP from fit_distilled_head.
    """

    def __init__(self, head_ckpt: str, input_window: Optional[Tuple[float, float]] = None,
                 cdw_alpha: float = 1.0, hu_scale: float = 100.0):
        super().__init__()
        d = np.load(head_ckpt, allow_pickle=True)['head'].item() \
            if str(head_ckpt).endswith('.npz') \
            else torch.load(head_ckpt, map_location='cpu', weights_only=False)
        self.model = _OrganHUMLP(int(d['in_dim']), int(d['n_classes']), int(d['hidden']))
        self.model.load_state_dict(d['state_dict'])
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.register_buffer('impute_mean', torch.tensor(np.asarray(d['impute_mean'], np.float32)))
        self.register_buffer('mu', torch.tensor(np.asarray(d['mu'], np.float32)))
        self.register_buffer('sd', torch.tensor(np.asarray(d['sd'], np.float32)))
        self.register_buffer('label_ids', torch.tensor(np.asarray(d['label_ids'], np.int64)))
        n = int(d['n_classes'])
        # ordinal distance matrix |i-j| between phase classes
        idx = torch.arange(n).float()
        self.register_buffer('dist', (idx[None, :] - idx[:, None]).abs())
        self.cdw_alpha = float(cdw_alpha)
        self.hu_scale = float(hu_scale)
        self.input_window = input_window

    def _to_hu(self, vol):
        if self.input_window is None:
            return vol
        a, b = self.input_window
        return vol * (b - a) + a

    def forward(self, gen_volume: torch.Tensor, seg: torch.Tensor,
                target_phase: torch.Tensor) -> torch.Tensor:
        means, present = organ_mean_hu_torch(self._to_hu(gen_volume), seg, self.label_ids)
        feats = torch.where(present, means, self.impute_mean.expand_as(means))
        logits = self.model((feats - self.mu) / self.sd)
        probs = F.softmax(logits, dim=1)
        # Class-distance-weighted CE (Polat et al.): penalise probability mass on
        # each wrong class by its ordinal distance to the target, so a
        # wrong-by-one phase costs less than wrong-by-three. w is 0 at the target
        # class, so the target is excluded automatically.
        w = self.dist[target_phase] ** self.cdw_alpha              # (B, P)
        neg_log_1mp = -torch.log1p(-probs.clamp(0.0, 1.0 - 1e-6))  # -log(1 - p_c)
        return (w * neg_log_1mp).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# CLI: build prototypes (+ optionally the distilled head) from the train split
# ---------------------------------------------------------------------------

def _load_nifti(path):
    import nibabel as nib
    return np.asarray(nib.load(str(path)).get_fdata())


def _build_from_split(args):
    from phase_data import find_phase_volumes, split_by_patient
    from roi_hu_features import load_ts_label_map, assert_multilabel

    samples = find_phase_volumes(args.data_dir, args.labels_csv, args.file_tag)
    train, val, _ = split_by_patient(samples, args.val_frac, args.test_frac, args.seed)
    fit_samples = train + val
    label_map = load_ts_label_map()
    label_ids = [label_map[o] for o in ORGANS]

    def gen():
        checked = False
        for s in fit_samples:
            seg_p = Path(s['volume_path'].replace(f'{args.file_tag}.nii.gz',
                                                  f'{args.file_tag}_seg_reg.nii.gz'))
            if not seg_p.exists():
                log.warning(f"  skip (no mask): {seg_p.name}"); continue
            vol = _load_nifti(s['volume_path'])
            seg = _load_nifti(seg_p).round().astype(np.int32)
            if seg.shape != vol.shape:
                log.warning(f"  skip (shape): {seg_p.name}"); continue
            if not checked:
                assert_multilabel(seg); checked = True
            yield vol, seg, s['phase_id']

    d = compute_phase_prototypes(gen(), label_ids, n_phases=4)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    save_prototypes(d, out / 'phase_hu_prototypes.npz')
    log.info("Prototype per-phase organ HU (rows=phase, NaN=organ never seen):")
    for p in range(d['prototypes'].shape[0]):
        vals = ' '.join(f'{x:6.0f}' if not np.isnan(x) else '   nan'
                        for x in d['prototypes'][p])
        log.info(f"  phase {p}: {vals}")


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    ap = argparse.ArgumentParser(description='Build ROI-HU phase prototypes from the train split')
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--labels_csv', default='')
    ap.add_argument('--file_tag', default='_deeds')
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--test_frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_dir', default='phase_results')
    _build_from_split(ap.parse_args())


__all__ = [
    'compute_phase_prototypes', 'save_prototypes', 'organ_mean_hu_torch',
    'ROIHUPhaseLoss', 'fit_distilled_head', 'PhaseClassifierLoss',
]
