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

RADIMAGENET FID (`--radimagenet_weights`), OPT-IN ON TOP OF --perceptual
--------------------------------------------------------------------------
A second, independent FID column computed from a ResNet50 pretrained on
RadImageNet (1.35M CT/MRI/ultrasound images: Mustafa et al., "RadImageNet",
Radiology: AI 2022) instead of ImageNet photographs — the domain-matched
backbone every "FID has no meaning on grayscale CT" caveat above is arguing for.
It is reported as `fid_rad`, next to (never merged with) the ImageNet `fid`
column, because they are not the same quantity and a reader comparing this table
to another paper needs to know which backbone produced which number.

Caveat worth weighing before leaning on it: McKinley et al./Woodland et al.,
"Feature Extraction for Generative Medical Imaging Evaluation: New Evidence
Against an Evolving Trend" (2024), found RadImageNet-based FID rankings on
medical images were MORE volatile and LESS aligned with human judgment than
ImageNet-based ones in their tests — the domain-matched backbone is not a proven
upgrade, just a different, also-imperfect proxy that the NCCT→CECT literature
increasingly reports. `fid_rad` is included for that comparability, not because
it is known to be the more valid number.

No `RadImageNet-LPIPS` is computed. LPIPS is not "any backbone's features plus a
distance" — it is a backbone's features plus a LINEAR CALIBRATION LAYER trained
to match human 2AFC perceptual judgments (`richzhang/PerceptualSimilarity`), and
no such calibration has been published for a RadImageNet backbone. A cosine or L2
distance in RadImageNet feature space is a different, uncalibrated metric and
would be mislabeled if reported as "LPIPS".

The RadImageNet weights are not fetched by this code. Point `--radimagenet_weights`
at a local PyTorch state_dict (`.pt`/`.pth`) — official source:
github.com/BMEII-AI/RadImageNet (weights via their linked Google Drive). The
official PyTorch ResNet50.pt wraps the network (fc already stripped) in a plain
`nn.Sequential`, so its keys are `backbone.<child-index>...` rather than
`conv1...`/`layer1...`; `_remap_backbone_keys` translates that specific,
verified-by-inspection layout automatically. Loading is otherwise best-effort
(`strict=False`): a `state_dict` in some other layout (e.g. a from-scratch
Keras→PyTorch conversion) will load partially or not at all, and
`_load_radimagenet_resnet50` raises with a key-mismatch count rather than
silently scoring on a mostly-random network.
"""

from pathlib import Path
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


# backbone.<i> -> the torchvision resnet50 attribute name at children()[i]. The
# official BMEII-AI/RadImageNet PyTorch release wraps resnet50 minus its `fc` in
# a plain nn.Sequential (`backbone = nn.Sequential(*list(resnet50().children())
# [:-1])`), so its state_dict keys are `backbone.0.weight`, `backbone.4.0.conv1.
# weight`, etc. instead of `conv1.weight`/`layer1.0.conv1.weight`. Indices 2, 3, 8
# (relu/maxpool/avgpool) carry no parameters and so never appear as key prefixes.
# Confirmed by disassembling the actual .pt file's pickle stream, not assumed.
_RESNET50_CHILD_INDEX = {'0': 'conv1', '1': 'bn1', '4': 'layer1',
                         '5': 'layer2', '6': 'layer3', '7': 'layer4'}


def _remap_backbone_keys(raw: dict) -> dict:
    out = {}
    for k, v in raw.items():
        if not k.startswith('backbone.'):
            out[k] = v
            continue
        idx, _, tail = k[len('backbone.'):].partition('.')
        name = _RESNET50_CHILD_INDEX.get(idx)
        out[f'{name}.{tail}' if name else k] = v
    return out


def _load_radimagenet_resnet50(weights_path: Path, device):
    """2048-d ResNet50 pool features from a RadImageNet-pretrained state_dict.

    Returns (callable(float tensor in [0,1], N3HW) -> (N,2048) tensor, backend
    label). ResNet50's avgpool is already 2048-d — same dimensionality as the
    InceptionV3 pool features FID normally uses, so `frechet_distance` needs no
    changes to run on this backbone; it is simply a different Gaussian.

    torchvision's `resnet50` architecture is the load target because it is the
    best-aligned public PyTorch skeleton for a ResNet50 state_dict (clean 1:1
    conv/BN/fc blocks); it is NOT guaranteed to be what produced whatever file
    `weights_path` points at (community ports and any Keras->PyTorch conversion
    have made independent naming choices). `strict=False` plus an explicit
    mismatch count is the honest way to surface that rather than silently
    scoring on a partially-random network.
    """
    import torch
    from torchvision.models import resnet50

    net = resnet50(weights=None)
    n_params = len(list(net.state_dict()))
    net.fc = torch.nn.Identity()          # expose the 2048-d pool instead of logits

    raw = torch.load(weights_path, map_location=device)
    if isinstance(raw, dict) and 'state_dict' in raw:
        raw = raw['state_dict']
    # Common prefixes added by training wrappers (DataParallel, a Keras-conversion
    # script's own module nesting) that a bare torchvision resnet50 does not have.
    state = {k.removeprefix('module.').removeprefix('model.'): v for k, v in raw.items()}
    state = _remap_backbone_keys(state)

    missing, unexpected = net.load_state_dict(state, strict=False)
    # fc.{weight,bias} are expected-missing: we just replaced fc with Identity.
    missing = [m for m in missing if not m.startswith('fc.')]
    if len(missing) > n_params // 2:
        raise PerceptualUnavailable(
            f'--radimagenet_weights {weights_path}: {len(missing)}/{n_params} '
            f'torchvision resnet50 parameters had no match in this state_dict '
            f'(first few missing: {missing[:5]}; first few unused keys in the '
            f'file: {list(unexpected)[:5]}). This usually means the file is not '
            f'a torchvision-layout ResNet50 (e.g. a raw Keras conversion with '
            f'different layer names) — rename its keys to match '
            f'`torchvision.models.resnet50().state_dict()` before pointing this '
            f'flag at it.')
    if missing or unexpected:
        print(f'  [radimagenet] loaded with {len(missing)} unmatched param(s) and '
              f'{len(unexpected)} unused key(s) in the file — partial match, '
              f'not a clean load. Verify features look reasonable before trusting fid_rad.')
    net = net.to(device).eval()

    def feats(x):
        return net(x)

    return feats, 'RadImageNet ResNet50 (PyTorch state_dict, community port)'


class PerceptualScorer:
    """Accumulating FID + per-case LPIPS over the benchmark's models.

    Real-side Inception features are cached by case key (the real CECT path, the
    same join key `benchmark.py` uses). Every model is scored against the same
    20 real volumes, so without the cache the real half of FID would be recomputed
    ~33 times for an identical result.
    """

    def __init__(self, device: Optional[str] = None, lpips_net: str = 'alex',
                 min_body_frac: float = 0.02, max_slices_per_case: Optional[int] = None,
                 batch_size: int = 16, slice_axis: int = -1,
                 radimagenet_weights: Optional[Path] = None):
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
        # The import succeeding is not enough: LPIPS loads its learned LINEAR
        # weights from package data (`lpips/weights/v0.1/<net>.pth`), and an
        # install can be importable with that directory missing — pip caches,
        # partial copies, --no-binary builds. The failure then surfaces as a bare
        # FileNotFoundError from torch.load, AFTER torchvision has spent minutes
        # downloading the backbone, and looks like a benchmark bug rather than an
        # install one.
        try:
            self._lpips = _lpips.LPIPS(net=lpips_net).to(self.device).eval()
        except (FileNotFoundError, OSError) as e:
            wdir = Path(_lpips.__file__).parent / 'weights' / 'v0.1'
            raise PerceptualUnavailable(
                f'the `lpips` package is installed but its learned linear weights '
                f'are missing ({e}). These ship inside the package and are not the '
                f'torchvision backbone, so re-downloading the backbone will not fix '
                f'it. Repair with:\n'
                f'    pip install --force-reinstall --no-cache-dir lpips\n'
                f'  or fetch the one file (~6 KB):\n'
                f'    curl -L -o {wdir / (lpips_net + ".pth")} \\\n'
                f'      https://github.com/richzhang/PerceptualSimilarity/raw/'
                f'master/lpips/weights/v0.1/{lpips_net}.pth') from e
        for p in self._lpips.parameters():
            p.requires_grad_(False)

        inception_feats, self.fid_backend = _load_inception(self.device)
        # backbone name -> feature fn. 'imagenet' is always present; 'radimagenet'
        # only when weights were given, so add_case/fid degrade to NaN without it
        # rather than requiring every caller to branch on whether it was loaded.
        self._backbones: Dict[str, callable] = {'imagenet': inception_feats}
        self.radimagenet_backend: Optional[str] = None
        if radimagenet_weights is not None:
            rad_feats, self.radimagenet_backend = _load_radimagenet_resnet50(
                Path(radimagenet_weights), self.device)
            self._backbones['radimagenet'] = rad_feats

        # backbone -> model -> list of (n_slices, 2048) arrays; and the shared
        # per-backbone real-feature cache (real features don't depend on model).
        self._gen_feats: Dict[str, Dict[str, List[np.ndarray]]] = {
            b: {} for b in self._backbones}
        self._real_feats: Dict[str, Dict[str, np.ndarray]] = {
            b: {} for b in self._backbones}
        self._real_order: Dict[str, Dict[str, List[str]]] = {
            b: {} for b in self._backbones}

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

    def _features(self, vol01: np.ndarray, idx: np.ndarray, backbone: str) -> np.ndarray:
        st = self._stack(vol01, idx)
        feats_fn = self._backbones[backbone]
        out = []
        with self.torch.no_grad():
            for i in range(0, st.shape[0], self.batch_size):
                out.append(feats_fn(st[i:i + self.batch_size].to(self.device))
                           .cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2048), dtype=np.float32)

    def add_case(self, model: str, case_key: str, g01: np.ndarray, r01: np.ndarray,
                 bmask: np.ndarray) -> float:
        """Accumulate this case into `model`'s FID set(s) for every loaded backbone
        ('imagenet', and 'radimagenet' if `--radimagenet_weights` was given);
        return its (ImageNet-backbone) LPIPS — see the module docstring for why
        there is no RadImageNet-LPIPS.

        The real features are computed once per `case_key` and reused for every
        later model — correct because the slice selection comes from the real
        volume's body mask and so does not depend on the model.
        """
        idx = self._select(bmask)
        for b in self._backbones:
            self._gen_feats[b].setdefault(model, []).append(self._features(g01, idx, b))
            if case_key not in self._real_feats[b]:
                self._real_feats[b][case_key] = self._features(r01, idx, b)
            self._real_order[b].setdefault(model, []).append(case_key)
        return self.case_lpips(g01, r01, bmask)

    def fid(self, model: str, backbone: str = 'imagenet') -> float:
        """FID for one model on one backbone, against exactly the real slices of
        ITS OWN cases.

        Restricting the real set to the cases this model actually scored matters
        for the multi-phase arms, which each cover a different subset — pooling
        every real volume in the run would compare an arterial model against a
        real set that is mostly venous.

        NaN if `backbone` was never loaded (e.g. `fid('m', 'radimagenet')`
        without `--radimagenet_weights`) rather than a KeyError, so callers can
        request it unconditionally and get an absent column instead of a crash.
        """
        gf = self._gen_feats.get(backbone, {}).get(model)
        if not gf:
            return float('nan')
        g = np.concatenate(gf)
        real_feats, real_order = self._real_feats[backbone], self._real_order[backbone]
        r = np.concatenate([real_feats[k] for k in real_order[model]])
        # 2048-d covariance from < ~2049 samples is rank-deficient; the estimate is
        # then dominated by that deficiency rather than by image quality.
        if g.shape[0] < 2 or r.shape[0] < 2:
            return float('nan')
        return frechet_distance(g.mean(0), np.cov(g, rowvar=False),
                                r.mean(0), np.cov(r, rowvar=False))

    def fid_rad(self, model: str) -> float:
        """FID on the RadImageNet backbone — NaN when it was not loaded."""
        return self.fid(model, backbone='radimagenet')

    def n_slices(self, model: str) -> int:
        gf = self._gen_feats.get('imagenet', {}).get(model)
        return int(sum(f.shape[0] for f in gf)) if gf else 0
