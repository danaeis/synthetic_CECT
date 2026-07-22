"""
Loss functions for NCCT→CECT synthesis — 2-D and 3-D compatible.

All losses accept inputs of shape (B, C, H, W) [2-D] or (B, C, D, H, W) [3-D].
3-D handling strategy per loss:
  L1 / MSE / AdversarialLoss / FeatureMatchingLoss / OrganWeightedLoss /
  PhaseSaliencyLoss / CyclicConsistencyLoss
    → dimension-agnostic PyTorch ops, work unchanged.

  PerceptualLoss (VGG16)
    → reshapes (B, C, D, H, W) → (B*D, C, H, W), applies per-slice, averages.

  SSIMLoss
    → 2-D: standard Gaussian-window SSIM.
    → 3-D: applied per 2-D slice (depth loop), averaged.

  GradientLoss (Sobel)
    → 2-D: x- and y-gradients.
    → 3-D: x-, y-, and z-gradients.

  FrequencyLoss (FFT)
    → 2-D: torch.fft.fft2 amplitude.
    → 3-D: torch.fft.fftn over H×W per slice (same spatial frequency target).

  SegmentationConsistencyLoss
    → same as GradientLoss: extends to z-direction in 3-D.

Baseline references:
  Liu22, Yan24c, Yan22, Hau21, Cho21 — L1 + Adversarial + Perceptual + FeatureMatching
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  Adversarial loss
# ---------------------------------------------------------------------------

class AdversarialLoss(nn.Module):
    """LSGAN or BCE PatchGAN adversarial loss (dimension-agnostic)."""

    def __init__(self, mode: str = 'lsgan'):
        super().__init__()
        assert mode in ('bce', 'lsgan')
        self.mode = mode

    def disc_loss(self, pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
        if self.mode == 'lsgan':
            return 0.5 * (F.mse_loss(pred_real, torch.ones_like(pred_real)) +
                          F.mse_loss(pred_fake, torch.zeros_like(pred_fake)))
        real = F.binary_cross_entropy_with_logits(pred_real, torch.full_like(pred_real, 0.9))
        fake = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
        return 0.5 * (real + fake)

    def gen_loss(self, pred_fake: torch.Tensor) -> torch.Tensor:
        if self.mode == 'lsgan':
            return F.mse_loss(pred_fake, torch.ones_like(pred_fake))
        return F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))


# ---------------------------------------------------------------------------
# 2.  Perceptual loss (VGG16, 3-D aware via per-slice reshape)
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """
    Multi-level VGG16 perceptual loss.

    2-D input (B, 1, H, W):
        Repeat channel → 3-ch, apply VGG slices.

    3-D input (B, 1, D, H, W):
        Reshape to (B*D, 1, H, W), apply VGG per-slice, average over D.
    """

    _LAYERS = [3, 8, 15, 22]      # relu1_2, relu2_2, relu3_3, relu4_3

    def __init__(self):
        super().__init__()
        try:
            import torchvision.models as tvm
            try:
                vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
            except Exception:
                vgg = tvm.vgg16(pretrained=True)
            feats = list(vgg.features.children())
        except Exception as e:
            raise RuntimeError(
                f"Could not load VGG16 for PerceptualLoss: {e}. "
                "Set use_perceptual=False in config if offline."
            )

        self.slices = nn.ModuleList()
        prev = 0
        for idx in self._LAYERS:
            self.slices.append(nn.Sequential(*feats[prev: idx + 1]))
            prev = idx + 1
        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.w = [1.0 / len(self._LAYERS)] * len(self._LAYERS)

    def _to_vgg(self, x: torch.Tensor) -> torch.Tensor:
        """(B,1,H,W) or (B*D,1,H,W), assumed [0,1] → 3-ch ImageNet-normalised."""
        x = x.float().clamp(0.0, 1.0)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def _flatten_depth(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, D, H, W) → (B*D, 1, H, W)."""
        B, C, D, H, W = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        is_3d = pred.dim() == 5
        if is_3d:
            pred   = self._flatten_depth(pred)
            target = self._flatten_depth(target)

        p_in = self._to_vgg(pred)
        t_in = self._to_vgg(target)

        loss = pred.new_zeros(1).squeeze()
        pf, tf = p_in, t_in
        for w, sl in zip(self.w, self.slices):
            pf = sl(pf);  tf = sl(tf)
            loss = loss + w * F.l1_loss(pf, tf.detach())
        return loss


# ---------------------------------------------------------------------------
# 2b. DINO perceptual loss (drop-in alternative to VGG16 PerceptualLoss)
# ---------------------------------------------------------------------------

class DinoPerceptualLoss(nn.Module):
    """
    Multi-scale perceptual loss using a frozen DINO ViT's spatial patch
    tokens instead of VGG16's ImageNet conv features. Takes an already-
    constructed `DinoSpatialBackbone` so the (comparatively heavy) ViT can
    be shared with `DinoSaliencyLoss` and loaded exactly once.

    2-D input (B,1,H,W): direct.
    3-D input (B,1,D,H,W): reshaped to (B*D,1,H,W), per-slice, averaged.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    @staticmethod
    def _flatten_depth(x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 5:
            pred   = self._flatten_depth(pred)
            target = self._flatten_depth(target)

        pf = self.backbone(pred)
        with torch.no_grad():
            tf = self.backbone(target)

        loss = pred.new_zeros(1).squeeze()
        for p, t in zip(pf, tf):
            loss = loss + F.l1_loss(p, t) / len(pf)
        return loss


# ---------------------------------------------------------------------------
# 3.  Feature-matching loss (discriminator intermediate layers)
# ---------------------------------------------------------------------------

class FeatureMatchingLoss(nn.Module):
    """L1 on discriminator intermediate features (pix2pixHD / Hau21)."""

    def forward(
        self,
        real_features: List[torch.Tensor],
        fake_features: List[torch.Tensor],
    ) -> torch.Tensor:
        if not real_features:
            return torch.tensor(0.0)
        loss = sum(F.l1_loss(fk, re.detach())
                   for re, fk in zip(real_features, fake_features))
        return loss / len(real_features)


# ---------------------------------------------------------------------------
# 4.  SSIM loss (2-D; 3-D = per-slice average)
# ---------------------------------------------------------------------------

class SSIMLoss(nn.Module):
    """
    1 − SSIM loss with Gaussian window.
    3-D inputs are processed slice-by-slice (average over depth).
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.ws  = window_size
        self.pad = window_size // 2
        self.register_buffer('win', self._make_win(window_size, sigma))

    @staticmethod
    def _make_win(sz, sigma) -> torch.Tensor:
        c = torch.arange(sz, dtype=torch.float32) - sz // 2
        g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        w = g.unsqueeze(0) * g.unsqueeze(1)
        return w.unsqueeze(0).unsqueeze(0)           # (1, 1, sz, sz)

    def _ssim2d(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute 1-SSIM on 2-D tensors (B, C, H, W)."""
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        C  = p.size(1)
        w  = self.win.expand(C, 1, -1, -1)
        pad = self.pad

        mu_p  = F.conv2d(p, w, padding=pad, groups=C)
        mu_t  = F.conv2d(t, w, padding=pad, groups=C)
        mu_pp = mu_p ** 2; mu_tt = mu_t ** 2; mu_pt = mu_p * mu_t

        sig_pp = F.conv2d(p * p, w, padding=pad, groups=C) - mu_pp
        sig_tt = F.conv2d(t * t, w, padding=pad, groups=C) - mu_tt
        sig_pt = F.conv2d(p * t, w, padding=pad, groups=C) - mu_pt

        ssim_map = ((2*mu_pt + C1)*(2*sig_pt + C2)) / \
                   ((mu_pp + mu_tt + C1)*(sig_pp + sig_tt + C2))
        return 1.0 - ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 5:
            # 3-D: iterate over depth slices
            B, C, D, H, W = pred.shape
            losses = [self._ssim2d(pred[:, :, d], target[:, :, d]) for d in range(D)]
            return torch.stack(losses).mean()
        return self._ssim2d(pred, target)


# ---------------------------------------------------------------------------
# 5.  Gradient / Sobel loss (extended to 3-D)
# ---------------------------------------------------------------------------

class GradientLoss(nn.Module):
    """
    L1 loss on Sobel gradient magnitude.
    2-D: x- and y-directions.
    3-D: x-, y-, and z-directions (3-D Sobel approximation).
    """

    def __init__(self):
        super().__init__()
        # 2-D Sobel
        kx2 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky2 = kx2.t()
        self.register_buffer('kx2', kx2.view(1, 1, 3, 3))
        self.register_buffer('ky2', ky2.view(1, 1, 3, 3))

        # 3-D Sobel (x-direction kernel, others derived by permutation)
        kx3 = torch.zeros(3, 3, 3, dtype=torch.float32)
        kx3[:, :, 0] = -1; kx3[:, :, 2] = 1
        kx3[1, 1, 0] *= 2; kx3[1, 1, 2] *= 2
        ky3 = kx3.permute(0, 2, 1)
        kz3 = kx3.permute(2, 1, 0)
        self.register_buffer('kx3', kx3.view(1, 1, 3, 3, 3))
        self.register_buffer('ky3', ky3.view(1, 1, 3, 3, 3))
        self.register_buffer('kz3', kz3.view(1, 1, 3, 3, 3))

    def _grad2d(self, x: torch.Tensor) -> torch.Tensor:
        C = x.size(1)
        kx = self.kx2.expand(C, 1, -1, -1)
        ky = self.ky2.expand(C, 1, -1, -1)
        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    def _grad3d(self, x: torch.Tensor) -> torch.Tensor:
        C = x.size(1)
        kx = self.kx3.expand(C, 1, -1, -1, -1)
        ky = self.ky3.expand(C, 1, -1, -1, -1)
        kz = self.kz3.expand(C, 1, -1, -1, -1)
        gx = F.conv3d(x, kx, padding=1, groups=C)
        gy = F.conv3d(x, ky, padding=1, groups=C)
        gz = F.conv3d(x, kz, padding=1, groups=C)
        return torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 5:
            return F.l1_loss(self._grad3d(pred), self._grad3d(target))
        return F.l1_loss(self._grad2d(pred), self._grad2d(target))


# ---------------------------------------------------------------------------
# 6.  Frequency (FFT) loss
# ---------------------------------------------------------------------------

class FrequencyLoss(nn.Module):
    """
    L1 loss on FFT amplitude spectrum.
    2-D: fft2 over H×W.
    3-D: fftn over H×W per slice (spatial frequency target), averaged over D.
    """

    def _amp(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, C, D, H, W = x.shape
            x2d = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            return torch.abs(torch.fft.fft2(x2d, norm='ortho'))
        return torch.abs(torch.fft.fft2(x, norm='ortho'))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._amp(pred), self._amp(target))


# ---------------------------------------------------------------------------
# 7.  Organ-weighted loss (dimension-agnostic)
# ---------------------------------------------------------------------------

class OrganWeightedLoss(nn.Module):
    """Mask-weighted L1, in one of two modes.

    Per-organ (`organ_weights` given): a {label_id: weight} lookup table, so each
    TotalSegmentator label carries its own weight and a weight of 0 excludes that
    anatomy from the gradient entirely. Requires a MULTI-LABEL mask (raw label
    ids) — a binarised mask collapses every organ onto the label-1 weight.

    Uniform (`organ_weights` None): the legacy behaviour — every masked voxel
    gets `organ_weight`× the background, from a binarised mask.

    L1 rather than MSE: MSE penalises large errors quadratically and so regresses
    to the conditional mean harder than L1 does, which is precisely the blur this
    term exists to counteract.
    """

    def __init__(
        self,
        organ_weight:      float = 10.0,
        organ_weights:     Optional[Dict[int, float]] = None,
        default_weight:    float = 1.0,
        background_weight: float = 1.0,
        max_label:         int = 256,
    ):
        super().__init__()
        self.uniform_weight = organ_weight
        self.per_organ = bool(organ_weights)
        if self.per_organ:
            lut = torch.full((max_label,), float(default_weight))
            lut[0] = float(background_weight)
            for lid, w in organ_weights.items():
                if not 0 <= int(lid) < max_label:
                    raise ValueError(f"organ label id {lid} outside [0,{max_label})")
                lut[int(lid)] = float(w)
            if float(lut.sum()) == 0.0:
                raise ValueError(
                    "all organ weights are zero — the organ loss would be "
                    "identically 0 and contribute no gradient."
                )
            self.register_buffer('lut', lut)

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        mask:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            return F.l1_loss(pred, target)
        if self.per_organ:
            w = self.lut[mask.round().long().clamp(0, self.lut.numel() - 1)]
        else:
            w = 1.0 + (self.uniform_weight - 1.0) * mask.clamp(0, 1)
        # Normalise by w.sum(), not .mean(): with zero-weighted regions present a
        # plain mean shrinks the loss as more area is excluded, which would tie
        # the effective lambda_organ to whichever weight scheme is in use and make
        # scenarios incomparable.
        return (w * (pred - target).abs()).sum() / w.sum().clamp_min(1e-8)


# ---------------------------------------------------------------------------
# 7b. Organ HU-profile loss (dimension-agnostic)
# ---------------------------------------------------------------------------

class OrganHUProfileLoss(nn.Module):
    """Penalise each organ's MEAN intensity deviation, not its per-voxel error.

    Motivation, from the measured ablation: the XGBoost phase classifier reads
    per-organ *median HU* — nothing else. Contrast phase is defined by the
    absolute enhancement level of each organ, so that is what a phase-faithful
    generator has to get right. Per-voxel losses optimise it only indirectly, and
    the scenario that most improved per-organ HU error (organ_curriculum,
    -1.58 HU vs baseline, t=-4.22) did so as a side effect rather than by
    targeting it.

    This complements OrganWeightedLoss rather than replacing it: that one
    sharpens texture *within* an organ, this one fixes the organ's overall level.
    A patch can score 0 here while looking nothing like the target, so it must
    never be the only spatial term.

    Weight-0 organs (bowel) are skipped entirely, as in OrganWeightedLoss —
    their content is not inferable from NCCT, so their mean HU is not a
    meaningful target either.
    """

    def __init__(
        self,
        organ_weights:  Optional[Dict[int, float]] = None,
        default_weight: float = 1.0,
        min_voxels:     int = 16,
        max_label:      int = 256,
    ):
        super().__init__()
        self.min_voxels = min_voxels
        self.default_weight = float(default_weight)
        lut = torch.full((max_label,), float(default_weight))
        lut[0] = 0.0                      # background has no meaningful "level"
        for lid, w in (organ_weights or {}).items():
            lut[int(lid)] = float(w)
        self.register_buffer('lut', lut)

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        mask:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            return pred.new_zeros(())

        lbl = mask.round().long().clamp(0, self.lut.numel() - 1)
        total = pred.new_zeros(())
        wsum = pred.new_zeros(())
        # Loop over the labels actually present — typically ~10-40 per patch, far
        # fewer than the 117 possible, and each iteration is two masked means.
        for lid in torch.unique(lbl):
            if lid.item() == 0:
                continue
            w = self.lut[lid]
            if w == 0:
                continue
            sel = (lbl == lid)
            n = sel.sum()
            if n < self.min_voxels:       # too few voxels for a stable mean
                continue
            mu_p = pred[sel].mean()
            mu_t = target[sel].mean()
            total = total + w * (mu_p - mu_t).abs()
            wsum = wsum + w
        return total / wsum.clamp_min(1e-8)


# ---------------------------------------------------------------------------
# 8.  Phase saliency loss (dimension-agnostic)
# ---------------------------------------------------------------------------

class PhaseSaliencyLoss(nn.Module):
    """Up-weights voxels strongly enhanced by contrast agent (target−source)."""

    def __init__(self, saliency_weight: float = 5.0, threshold: float = 0.08):
        super().__init__()
        self.saliency_weight = saliency_weight
        self.threshold = threshold

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            sal = ((target - source).abs() > self.threshold).float()
        w = 1.0 + (self.saliency_weight - 1.0) * sal
        return (w * (pred - target) ** 2).mean()


# ---------------------------------------------------------------------------
# 8b. DINO-based phase saliency loss (learned alternative to the threshold
#     heuristic above)
# ---------------------------------------------------------------------------

class DinoSaliencyLoss(nn.Module):
    """
    Up-weights voxels where the frozen DINO backbone's own feature space
    changes most between source (NCCT) and target (CECT) — a semantic,
    learned stand-in for the raw-intensity-difference threshold used by
    `PhaseSaliencyLoss`. Reuses the same shared `DinoSpatialBackbone` as
    `DinoPerceptualLoss` (no extra model load if both are enabled).

    The weight map is built from source/target only (no_grad, non-
    differentiable input, like the heuristic version) — the actual loss
    term is a plain weighted pixel MSE against `pred`, so it stays cheap
    and gradients still flow straight to the generator.
    """

    def __init__(self, backbone, saliency_weight: float = 5.0):
        super().__init__()
        self.backbone = backbone
        self.saliency_weight = saliency_weight

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        is_3d = pred.dim() == 5
        with torch.no_grad():
            s2d = source[:, :, source.shape[2] // 2] if is_3d else source
            t2d = target[:, :, target.shape[2] // 2] if is_3d else target
            sf = self.backbone(s2d)[-1]              # deepest feature map
            tf = self.backbone(t2d)[-1]
            diff = (tf - sf).norm(dim=1, keepdim=True)          # (B,1,h,w)
            diff = diff / (diff.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1) + 1e-8)
            size = pred.shape[-2:]
            sal = F.interpolate(diff, size=size, mode='bilinear', align_corners=False)
            if is_3d:
                sal = sal.unsqueeze(2).expand(-1, -1, pred.shape[2], -1, -1)
            w = 1.0 + (self.saliency_weight - 1.0) * sal

        return (w * (pred - target) ** 2).mean()


# ---------------------------------------------------------------------------
# 9.  Cyclic consistency loss (dimension-agnostic)
# ---------------------------------------------------------------------------

class CyclicConsistencyLoss(nn.Module):
    """source → G(source) → G(G(source)) ≈ source (L1)."""

    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(reconstructed, original)


# ---------------------------------------------------------------------------
# 10. Segmentation consistency loss (2-D / 3-D Sobel edge matching)
# ---------------------------------------------------------------------------

class SegmentationConsistencyLoss(nn.Module):
    """
    L1 on Sobel edge maps within organ-masked regions.
    Reuses GradientLoss logic; mask restricts to organ boundaries.
    """

    def __init__(self):
        super().__init__()
        self._grad = GradientLoss()

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        mask:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ep = self._grad._grad3d(pred)   if pred.dim() == 5 else self._grad._grad2d(pred)
        et = self._grad._grad3d(target) if pred.dim() == 5 else self._grad._grad2d(target)
        if mask is not None:
            m = mask.clamp(0, 1)
            return F.l1_loss(ep * m, et * m)
        return F.l1_loss(ep, et)


# ---------------------------------------------------------------------------
# CompositeLoss — all losses combined
# ---------------------------------------------------------------------------

class CompositeLoss(nn.Module):
    """
    Configurable composite loss.

    Default (baseline): L1 + Adversarial (LSGAN) + Perceptual (VGG) + Feature matching.
    All extra losses are off; enable via config flags.

    Works identically for 2-D (B,1,H,W) and 3-D (B,1,D,H,W) inputs —
    each sub-loss handles the dimension check internally.

    Key config flags  (default → value):
      use_adversarial      True     PatchGAN adversarial
      use_perceptual       True     perceptual (backbone: perceptual_backbone='vgg'|'dino')
      use_feature_matching True     discriminator feature matching
      use_ssim             False
      use_gradient         False
      use_frequency        False
      use_organ            False    needs 'mask' in batch (auto-loaded by dataset.py
                                     when use_organ or use_seg_consistency is True)
      use_saliency         False    heuristic by default; saliency_mode='dino' for a
                                     learned DINO-feature-difference weight map
      use_cycle            False    single shared generator: G(G(source)) ~= source,
                                     ramped in via cycle_warmup_epochs
      use_seg_consistency  False    needs 'mask' in batch, see use_organ

    perceptual_backbone='dino' and saliency_mode='dino' share ONE frozen DINO
    backbone (loaded lazily, at most once, only if either is requested) —
    see dino_backbone.py for the priority-ordered DINOv3/DINOv2 loader.

    Key λ values  (defaults):
      lambda_l1=100 (parametric — auto-reduced to lambda_l1_reduced=25 whenever
      use_adversarial/use_perceptual/use_feature_matching is active, since those
      three specifically trade pixel fidelity for realism and need L1 to back
      off to have any real influence; see config.py's LAMBDA_L1 comment),
      lambda_adv=2, lambda_perceptual=10, lambda_feature_match=10
    """

    def __init__(self, config: Dict):
        super().__init__()
        c = config

        # L1 is parametric on which other losses are active: adversarial,
        # perceptual, and feature_matching all specifically trade realism for
        # exact pixel fidelity (unlike ssim/gradient/frequency/organ/saliency/
        # cycle/seg_consistency, which refine what fidelity means rather than
        # compete with it) — so L1 needs to back off from its full weight for
        # any of those three to have a real chance to influence the output.
        # See config.py's LAMBDA_L1 / LAMBDA_L1_REDUCED comment for the
        # empirical evidence (scenario_results_overview.md): at a flat 100:1
        # L1:adv ratio, every adversarial-inclusive scenario converged to
        # near-identical metrics regardless of which extra losses were added.
        # Resolve the three competitor flags ONCE, here, and reuse the resolved
        # values both for this decision and to build the terms below. Re-reading
        # the config dict with a different default in the two places is how the
        # two can disagree — e.g. a loss being active while L1 stays at 100,
        # or vice versa.
        self.use_adv  = c.get('use_adversarial', True)
        self.use_perc = c.get('use_perceptual', True)
        self.use_fm   = c.get('use_feature_matching', True)

        # L1 decay curriculum: hold lambda_l1 until l1_decay_start_epoch, then
        # ramp linearly down to lambda_l1_floor by l1_decay_end_epoch and hold.
        # The floor is deliberately non-zero — see config.py's USE_L1_DECAY note.
        self.use_l1_decay    = c.get('use_l1_decay', False)
        self.l1_decay_start  = c.get('l1_decay_start_epoch', 10)
        self.l1_decay_end    = c.get('l1_decay_end_epoch', 30)
        self.lambda_l1_floor = c.get('lambda_l1_floor', 25.0)

        # Where the schedule STARTS. When the decay is on it starts from the full
        # lambda_l1 and ignores lambda_l1_reduced: the decay IS the mechanism for
        # backing L1 off, so applying the static reduction as well is
        # double-counting. Applying both is what silently broke l1_adv_organ —
        # adversarial pinned the start to 25, the floor was also 25, and the
        # curriculum ran 25→25, i.e. did nothing for the entire run.
        if self.use_l1_decay:
            self.lambda_l1 = c.get('lambda_l1', 100.0)
        elif self.use_adv or self.use_perc or self.use_fm:
            self.lambda_l1 = c.get('lambda_l1_reduced', 25.0)
        else:
            self.lambda_l1 = c.get('lambda_l1', 100.0)

        if self.use_l1_decay:
            if self.lambda_l1_floor > self.lambda_l1:
                log.warning(f"lambda_l1_floor ({self.lambda_l1_floor}) > lambda_l1 "
                            f"({self.lambda_l1}) — L1 will ramp UP, not decay.")
            elif self.lambda_l1_floor == self.lambda_l1:
                log.warning(
                    f"use_l1_decay is ON but lambda_l1 == lambda_l1_floor "
                    f"({self.lambda_l1}) — the curriculum is a NO-OP and L1 will "
                    f"stay constant for the whole run. Lower lambda_l1_floor or "
                    f"raise lambda_l1.")

        self.lambda_adv = c.get('lambda_adv', 1.0)
        self.adv_warmup = c.get('adv_warmup_epochs', 5)
        self.adv_loss   = AdversarialLoss(mode=c.get('adv_mode', 'lsgan'))
        self._epoch     = 0

        # DINO backbone is loaded at most once and shared between the
        # perceptual loss and the saliency loss — never loaded at all if
        # neither wants it (see _get_dino_backbone below).
        self._dino_backbone = None

        self.lambda_perc= c.get('lambda_perceptual', 10.0)
        self.perceptual_backbone = c.get('perceptual_backbone', 'vgg')   # 'vgg' | 'dino'
        if self.use_perc:
            if self.perceptual_backbone == 'dino':
                self.perceptual = DinoPerceptualLoss(self._get_dino_backbone())
            else:
                self.perceptual = PerceptualLoss()

        self.lambda_fm  = c.get('lambda_feature_match', 10.0)
        if self.use_fm:
            self.feat_match = FeatureMatchingLoss()

        self.use_ssim   = c.get('use_ssim', False)
        self.lambda_ssim= c.get('lambda_ssim', 10.0)
        if self.use_ssim:
            self.ssim = SSIMLoss()

        self.use_grad   = c.get('use_gradient', False)
        self.lambda_grad= c.get('lambda_gradient', 5.0)
        if self.use_grad:
            self.gradient = GradientLoss()

        self.use_freq   = c.get('use_frequency', False)
        self.lambda_freq= c.get('lambda_frequency', 1.0)
        if self.use_freq:
            self.frequency = FrequencyLoss()

        self.use_organ  = c.get('use_organ', False)
        self.lambda_organ = c.get('lambda_organ', 5.0)
        if self.use_organ:
            self.organ = OrganWeightedLoss(
                organ_weight      = c.get('organ_weight', 10.0),
                organ_weights     = c.get('organ_weights'),
                default_weight    = c.get('organ_weight_default', 1.0),
                background_weight = c.get('organ_weight_background', 1.0),
            )

        self.use_hu_profile  = c.get('use_hu_profile', False)
        self.lambda_hu_profile = c.get('lambda_hu_profile', 10.0)
        if self.use_hu_profile:
            self.hu_profile = OrganHUProfileLoss(
                organ_weights  = c.get('organ_weights'),
                default_weight = c.get('organ_weight_default', 1.0),
            )
            if not c.get('organ_weights'):
                log.warning("use_hu_profile is on but organ_weights is None — every "
                            "organ gets the default weight. The term still works, but "
                            "the phase-critical vessels get no priority.")

        self.use_sal    = c.get('use_saliency', False)
        self.lambda_sal = c.get('lambda_saliency', 5.0)
        self.saliency_mode = c.get('saliency_mode', 'heuristic')         # 'heuristic' | 'dino'
        if self.use_sal:
            if self.saliency_mode == 'dino':
                self.saliency = DinoSaliencyLoss(self._get_dino_backbone(),
                                                  saliency_weight=c.get('saliency_weight', 5.0))
            else:
                self.saliency = PhaseSaliencyLoss(saliency_weight=c.get('saliency_weight', 5.0),
                                                   threshold=c.get('saliency_threshold', 0.08))

        self.use_cycle  = c.get('use_cycle', False)
        self.lambda_cycle = c.get('lambda_cycle', 10.0)
        # Warmup: with a single shared generator, G must learn to act as its
        # own inverse (G(G(source)) ~= source) *while* also learning
        # G(source) ~= target. Ramping lambda_cycle in avoids the two
        # objectives fighting each other before G has learned anything
        # useful — same warmup pattern as adversarial loss above.
        self.cycle_warmup = c.get('cycle_warmup_epochs', 5)
        if self.use_cycle:
            self.cycle = CyclicConsistencyLoss()

        self.use_seg    = c.get('use_seg_consistency', False)
        self.lambda_seg = c.get('lambda_seg', 2.0)
        if self.use_seg:
            self.seg_cons = SegmentationConsistencyLoss()

    def _get_dino_backbone(self):
        """Load the shared DINO backbone at most once, on first use."""
        if self._dino_backbone is None:
            from dino_backbone import DinoSpatialBackbone
            self._dino_backbone = DinoSpatialBackbone()
        return self._dino_backbone

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _adv_w(self) -> float:
        if not self.use_adv:
            return 0.0
        return self.lambda_adv * min(1.0, self._epoch / max(1, self.adv_warmup))

    def _cycle_w(self) -> float:
        if not self.use_cycle:
            return 0.0
        return self.lambda_cycle * min(1.0, self._epoch / max(1, self.cycle_warmup))

    def _l1_w(self) -> float:
        """Current lambda_l1 under the decay curriculum (constant if disabled)."""
        if not self.use_l1_decay or self._epoch <= self.l1_decay_start:
            return self.lambda_l1
        span = max(1, self.l1_decay_end - self.l1_decay_start)
        f = min(1.0, (self._epoch - self.l1_decay_start) / span)
        return self.lambda_l1 + f * (self.lambda_l1_floor - self.lambda_l1)

    def forward(
        self,
        pred:             torch.Tensor,
        target:           torch.Tensor,
        source:           Optional[torch.Tensor] = None,
        mask:             Optional[torch.Tensor] = None,
        adv_fake_logits:  Optional[torch.Tensor] = None,
        real_features:    Optional[List[torch.Tensor]] = None,
        fake_features:    Optional[List[torch.Tensor]] = None,
        cycle_pred:       Optional[torch.Tensor] = None,
    ):
        """Returns (total_loss, loss_dict)."""
        d: Dict[str, float] = {}
        total = pred.new_zeros(1).squeeze()

        _lam_l1 = self._l1_w()
        l1 = F.l1_loss(pred, target) * _lam_l1
        d['l1'] = l1.item();  total = total + l1
        d['lambda_l1'] = _lam_l1        # logged so the curriculum is auditable

        if self.use_adv and adv_fake_logits is not None:
            adv = self.adv_loss.gen_loss(adv_fake_logits) * self._adv_w()
            d['adversarial'] = adv.item();  total = total + adv
        else:
            d['adversarial'] = 0.0

        if self.use_perc:
            perc = self.perceptual(pred, target) * self.lambda_perc
            d['perceptual'] = perc.item();  total = total + perc
        else:
            d['perceptual'] = 0.0

        if self.use_fm and real_features and fake_features:
            fm = self.feat_match(real_features, fake_features) * self.lambda_fm
            d['feature_matching'] = fm.item();  total = total + fm
        else:
            d['feature_matching'] = 0.0

        if self.use_ssim:
            ssim = self.ssim(pred, target) * self.lambda_ssim
            d['ssim'] = ssim.item();  total = total + ssim
        else:
            d['ssim'] = 0.0

        if self.use_grad:
            grad = self.gradient(pred, target) * self.lambda_grad
            d['gradient'] = grad.item();  total = total + grad
        else:
            d['gradient'] = 0.0

        if self.use_freq:
            freq = self.frequency(pred, target) * self.lambda_freq
            d['frequency'] = freq.item();  total = total + freq
        else:
            d['frequency'] = 0.0

        if self.use_organ:
            org = self.organ(pred, target, mask) * self.lambda_organ
            d['organ'] = org.item();  total = total + org
        else:
            d['organ'] = 0.0

        if self.use_hu_profile:
            hup = self.hu_profile(pred, target, mask) * self.lambda_hu_profile
            d['hu_profile'] = hup.item();  total = total + hup
        else:
            d['hu_profile'] = 0.0

        if self.use_sal and source is not None:
            sal = self.saliency(pred, target, source) * self.lambda_sal
            d['saliency'] = sal.item();  total = total + sal
        else:
            d['saliency'] = 0.0

        if self.use_cycle and cycle_pred is not None and source is not None:
            cyc = self.cycle(cycle_pred, source) * self._cycle_w()
            d['cycle'] = cyc.item();  total = total + cyc
        else:
            d['cycle'] = 0.0

        if self.use_seg:
            seg = self.seg_cons(pred, target, mask) * self.lambda_seg
            d['seg_consistency'] = seg.item();  total = total + seg
        else:
            d['seg_consistency'] = 0.0

        d['total'] = total.item()
        return total, d
