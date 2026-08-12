"""
FID and LPIPS for the NCCT→CECT benchmark — reported for cross-paper comparability.

WHY THIS IS A SEPARATE MODULE, AND WHY IT IS OPT-IN
---------------------------------------------------
`metrics.py` states the case against these two: both run ImageNet-pretrained
networks built for natural RGB photographs, so on grayscale CT their absolute
values carry no physical meaning and their ranking is not validated for this
domain. Nothing here retracts that. They are implemented because the NCCT→CECT
literature reports them and a table that omits them is hard to place next to
those papers — so they are reported as SECONDARY, comparability-only columns
next to the CT-native texture metrics (`raps_hf`, `grad_w1`) that actually
motivated them.

Consequences of that reasoning, which are design decisions here:

  * Torch stays out of `metrics.py`. That module is numpy/scipy only and every
    benchmark run depends on it; these two need torch + downloaded ImageNet
    weights, so they live behind `benchmark.py --perceptual` and an ImportError
    here degrades to NaN columns rather than killing the run.
  * LPIPS is PAIRED and per-case, so it joins the per-case rows and the paired
    t-tests like every other metric.
  * FID is DISTRIBUTIONAL and has no per-case value: it is one number per model,
    computed over the pooled slices of all its cases. It therefore cannot enter
    the paired block, and at n=20 volumes it is a small-sample estimate — see the
    bias caveat below.

WHAT IS COMPUTED, EXACTLY
-------------------------
Both operate on axial slices of the SAME shared HU[-200,400]→[0,1] domain as
every other metric (`metrics.to_unit`), replicated to 3 channels.

  * LPIPS(gen_slice, real_slice) averaged over a case's slices → per-case value
    → averaged over cases → the table cell. Slices are fed at NATIVE in-plane
    resolution (no resize); LPIPS values are resolution-dependent, so this is
    only comparable across models scored on the same grid, which is the case
    here since every model emits on the source grid.
  * FID between {all gen slices of all cases} and {all real slices of all cases},
    from 2048-d InceptionV3 pool features.

SLICE SELECTION. Only slices whose body mask covers at least `min_body_frac` of
the frame are used. This is not cosmetic: a CT volume's first and last slices are
mostly scanner air, and identical near-black frames in both distributions pull FID
toward 0 for every model equally, which would make the column unable to separate
anything. The same slice set is used for gen and real, and it is derived from the
REAL volume, so it is identical across models.

FID BIAS AT n=20 CASES. FID is biased upward at small sample size and the bias
depends on the number of samples, so FID values are comparable ACROSS MODELS HERE
(identical slice counts, identical real set) but should not be compared to a
published FID computed on a different sample size. Slices from one patient are
also not independent draws, so the effective sample size is nearer 20 than the
few thousand slices actually fed. Report it with that caveat or not at all.

BACKEND. FID features come from `pytorch_fid`'s InceptionV3 when installed — that
is the weight set the FID literature is calibrated on. `torchvision`'s
ImageNet-1k InceptionV3 is a fallback so the column can still be produced, but it
yields a DIFFERENT number: `fid_backend` in the output records which was used and
values from the two must never be pooled into one table.
"""

from typing import Dict, List, Optional

import numpy as np

__all__ = ['PerceptualScorer', 'frechet_distance', 'PerceptualUnavailable']

# Reported in the table footer so a reader knows which InceptionV3 produced the
# numbers; set on first use.
_FID_BACKEND = None


class PerceptualUnavailable(RuntimeError):
    """Raised when torch / lpips / an Inception backend is missing."""


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                     mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet distance between two Gaussians — the FID formula.

    ||mu1-mu2||^2 + Tr(S1 + S2 - 2*(S1 S2)^{1/2}).

    `eps` conditions the covariances before the matrix square root: with 2048
    features estimated from a few thousand correlated slices, S is near-singular
    and `sqrtm` returns a complex result with a small imaginary part. Adding
    eps*I on a retry is the standard fix (it is what pytorch-fid does); a
    residual imaginary component beyond a tolerance is a genuine numerical
    failure and raises rather than being silently discarded.
    """
    from scipy import linalg

    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError('FID: sqrtm returned a substantially complex result '
                             f'(max |imag| = {np.max(np.abs(covmean.imag)):.3g})')
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2)
                 - 2 * np.trace(covmean))


def _load_inception(device):
    """2048-d InceptionV3 pool features, canonical weights if available.

    Returns (callable(float tensor in [0,1], N3HW) -> (N,2048) tensor, backend name).
    """
    global _FID_BACKEND
    import torch

    try:
        from pytorch_fid.inception import InceptionV3
        # block_idx 3 = final average pool, 2048-d — the standard FID feature.
        # resize_input/normalize_input let us hand it [0,1] at native size.
        net = InceptionV3([3], resize_input=True, normalize_input=True).to(device).eval()

        def feats(x):
            return net(x)[0].flatten(1)

        _FID_BACKEND = 'pytorch-fid (TF-ported InceptionV3)'
        return feats, _FID_BACKEND
    except ImportError:
        pass

    try:
        import torch.nn.functional as F
        from torchvision.models import inception_v3, Inception_V3_Weights
    except ImportError as e:
        raise PerceptualUnavailable(
            'FID needs `pytorch-fid` (preferred) or `torchvision`; neither imported '
            f'({e}). pip install pytorch-fid') from e

    net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                       aux_logits=True, init_weights=False).to(device).eval()
    net.fc = torch.nn.Identity()          # expose the 2048-d pool instead of logits
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def feats(x):
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return net((x - mean) / std)

    _FID_BACKEND = 'torchvision (ImageNet-1k InceptionV3) — NOT the canonical FID weights'
    return feats, _FID_BACKEND


class PerceptualScorer:
    """Accumulating FID + per-case LPIPS over the benchmark's models.

    Real-side Inception features are cached by case key (the real CECT path, the
    same join key `benchmark.py` uses). Every model is scored against the same
    20 real volumes, so without the cache the real half of FID would be recomputed
    ~33 times for an identical result.
    """

    def __init__(self, device: Optional[str] = None, lpips_net: str = 'alex',
                 min_body_frac: float = 0.02, max_slices_per_case: Optional[int] = None,
                 batch_size: int = 16, slice_axis: int = -1):
        import torch

        self.torch = torch
        self.device = torch.device(
            device if device is not None
            else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.min_body_frac = float(min_body_frac)
        self.max_slices_per_case = max_slices_per_case
        self.batch_size = int(batch_size)
        self.slice_axis = slice_axis

        try:
            import lpips as _lpips
        except ImportError as e:
            raise PerceptualUnavailable(
                'LPIPS needs the `lpips` package (it ships the learned linear '
                f'weights, which cannot be reconstructed from torchvision): {e}. '
                'pip install lpips') from e
        self.lpips_net_name = lpips_net
        self._lpips = _lpips.LPIPS(net=lpips_net).to(self.device).eval()
        for p in self._lpips.parameters():
            p.requires_grad_(False)

        self._inception, self.fid_backend = _load_inception(self.device)

        # model -> list of (n_slices, 2048) arrays; and the shared real cache.
        self._gen_feats: Dict[str, List[np.ndarray]] = {}
        self._real_feats: Dict[str, np.ndarray] = {}
        self._real_order: Dict[str, List[str]] = {}

    # -- slice extraction ---------------------------------------------------

    def _select(self, bmask: np.ndarray) -> np.ndarray:
        """Indices of axial slices with enough body in them (see module docstring)."""
        m = np.moveaxis(bmask, self.slice_axis, 0)
        frac = m.reshape(m.shape[0], -1).mean(axis=1)
        idx = np.flatnonzero(frac >= self.min_body_frac)
        if idx.size == 0:                      # degenerate mask: fall back to all
            idx = np.arange(m.shape[0])
        if self.max_slices_per_case and idx.size > self.max_slices_per_case:
            idx = idx[np.linspace(0, idx.size - 1, self.max_slices_per_case).astype(int)]
        return idx

    def _stack(self, vol01: np.ndarray, idx: np.ndarray):
        """(len(idx), 3, H, W) float tensor in [0,1] from a [0,1] volume."""
        v = np.moveaxis(vol01, self.slice_axis, 0)[idx]
        t = self.torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
        return t.unsqueeze(1).repeat(1, 3, 1, 1)

    # -- metrics ------------------------------------------------------------

    def case_lpips(self, g01: np.ndarray, r01: np.ndarray, bmask: np.ndarray) -> float:
        """Mean LPIPS over a case's body-containing axial slices."""
        idx = self._select(bmask)
        gs, rs = self._stack(g01, idx), self._stack(r01, idx)
        vals = []
        with self.torch.no_grad():
            for i in range(0, gs.shape[0], self.batch_size):
                a = gs[i:i + self.batch_size].to(self.device) * 2 - 1   # LPIPS wants [-1,1]
                b = rs[i:i + self.batch_size].to(self.device) * 2 - 1
                vals.append(self._lpips(a, b).flatten().cpu().numpy())
        return float(np.concatenate(vals).mean()) if vals else float('nan')

    def _features(self, vol01: np.ndarray, idx: np.ndarray) -> np.ndarray:
        st = self._stack(vol01, idx)
        out = []
        with self.torch.no_grad():
            for i in range(0, st.shape[0], self.batch_size):
                out.append(self._inception(st[i:i + self.batch_size].to(self.device))
                           .cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2048), dtype=np.float32)

    def add_case(self, model: str, case_key: str, g01: np.ndarray, r01: np.ndarray,
                 bmask: np.ndarray) -> float:
        """Accumulate this case into `model`'s FID set; return its LPIPS.

        The real features are computed once per `case_key` and reused for every
        later model — correct because the slice selection comes from the real
        volume's body mask and so does not depend on the model.
        """
        idx = self._select(bmask)
        self._gen_feats.setdefault(model, []).append(self._features(g01, idx))
        if case_key not in self._real_feats:
            self._real_feats[case_key] = self._features(r01, idx)
        self._real_order.setdefault(model, []).append(case_key)
        return self.case_lpips(g01, r01, bmask)

    def fid(self, model: str) -> float:
        """FID for one model, against exactly the real slices of ITS OWN cases.

        Restricting the real set to the cases this model actually scored matters
        for the multi-phase arms, which each cover a different subset — pooling
        every real volume in the run would compare an arterial model against a
        real set that is mostly venous.
        """
        gf = self._gen_feats.get(model)
        if not gf:
            return float('nan')
        g = np.concatenate(gf)
        r = np.concatenate([self._real_feats[k] for k in self._real_order[model]])
        # 2048-d covariance from < ~2049 samples is rank-deficient; the estimate is
        # then dominated by that deficiency rather than by image quality.
        if g.shape[0] < 2 or r.shape[0] < 2:
            return float('nan')
        return frechet_distance(g.mean(0), np.cov(g, rowvar=False),
                                r.mean(0), np.cov(r, rowvar=False))

    def n_slices(self, model: str) -> int:
        gf = self._gen_feats.get(model)
        return int(sum(f.shape[0] for f in gf)) if gf else 0
