"""
Frozen volume encoders for phase classification: MedViT and DINOv3.

Both take a 3-D volume (B,1,D,H,W) in [0,1] and return a pooled feature vector
(B, feat_dim) by encoding sampled 2-D slices and mean-pooling:
  - across the spatial dimensions (handled by each backbone's own pooling), and
  - across sampled slices (plain mean — deterministic, NO trainable/untrained
    aggregation layer).

Why plain mean pooling: the prior codebase's MedViT encoder ran a *randomly
initialised* LSTM + Linear projection over slices and used its output as
"frozen features" — injecting untrained noise into the feature space. A frozen
feature extractor must be fully deterministic and fully pretrained; mean
pooling has no parameters, so the only learned weights involved are the
backbone's genuine pretrained ones.

NO SILENT FALLBACKS: if a backbone's real pretrained weights can't be loaded,
these raise. (The old `dino_encoder.py` silently fell back through DINOv1 → a
supervised ViT → random weights while logging "✅ Loaded", so its reported
accuracy couldn't be attributed to any known model — see RETRAIN_PLAN.md.)
"""

import logging
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


def _sample_slice_indices(depth: int, max_slices: int) -> List[int]:
    if depth <= max_slices:
        return list(range(depth))
    return torch.linspace(0, depth - 1, max_slices).round().long().tolist()


class _VolumeEncoderBase(nn.Module):
    """Shared slice-sampling / preprocessing / mean-pool skeleton."""

    IMAGE_SIZE = 224
    imagenet_norm = False   # subclass sets True if its backbone needs it

    def __init__(self, max_slices: int = 32):
        super().__init__()
        self.max_slices = max_slices
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def _prep_slice(self, slc: torch.Tensor) -> torch.Tensor:
        """(B,1,H,W) in [0,1] → (B,3,224,224), optionally ImageNet-normalised."""
        slc = slc.float().clamp(0.0, 1.0)
        if slc.shape[-1] != self.IMAGE_SIZE or slc.shape[-2] != self.IMAGE_SIZE:
            slc = F.interpolate(slc, size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
                                mode='bilinear', align_corners=False)
        if slc.size(1) == 1:
            slc = slc.repeat(1, 3, 1, 1)
        if self.imagenet_norm:
            slc = (slc - self.mean) / self.std
        return slc

    def _encode_slice(self, slc: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        assert volume.dim() == 5, f"expected (B,1,D,H,W), got {tuple(volume.shape)}"
        idxs = _sample_slice_indices(volume.shape[2], self.max_slices)
        feats = []
        for i in idxs:
            feats.append(self._encode_slice(self._prep_slice(volume[:, :, i])))
        return torch.stack(feats, dim=1).mean(dim=1)   # (B, feat_dim)


# ---------------------------------------------------------------------------
# MedViT
# ---------------------------------------------------------------------------

class MedViTVolumeEncoder(_VolumeEncoderBase):
    imagenet_norm = False   # matches the prior runs' preprocessing ([0,1] RGB)

    def __init__(self, pretrained_path: str, model_size: str = 'small', max_slices: int = 32):
        super().__init__(max_slices)

        # Fail fast on the most common misconfiguration BEFORE importing/building
        # the backbone: no weights → refuse to run (never silently random-init).
        p = Path(pretrained_path)
        if not pretrained_path or not p.exists():
            raise FileNotFoundError(
                f"MedViT pretrained weights not found at '{pretrained_path}'. "
                f"Refusing to run on randomly-initialised weights — see "
                f"WEIGHTS_SETUP.md for how to obtain them."
            )

        sys.path.insert(0, str(Path(__file__).parent))
        from MedViT.MedViT import MedViT_small, MedViT_base

        if model_size == 'base':
            self.backbone = MedViT_base()
        else:
            self.backbone = MedViT_small()

        self._load_pretrained(str(p))

        # Drop the ImageNet classification head → forward returns pooled feats.
        self.backbone.proj_head = nn.Identity()
        self.feat_dim = 1024   # MedViT small & base both end at 1024 channels
        self._freeze()
        log.info(f"MedViTVolumeEncoder | size={model_size} | feat_dim={self.feat_dim} "
                 f"| weights={p.name}")

    def _load_pretrained(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) else ckpt
        model_sd = self.backbone.state_dict()
        # Keep only matching keys with matching shapes (ImageNet head won't match — fine).
        compat = {k: v for k, v in state.items()
                  if k in model_sd and v.shape == model_sd[k].shape}
        if len(compat) < 0.5 * len(model_sd):
            raise RuntimeError(
                f"MedViT checkpoint '{path}' matched only {len(compat)}/{len(model_sd)} "
                f"weights — wrong checkpoint or architecture mismatch. Aborting rather "
                f"than run on a mostly-random backbone."
            )
        self.backbone.load_state_dict(compat, strict=False)
        log.info(f"  loaded {len(compat)}/{len(model_sd)} MedViT weights from checkpoint")

    def _encode_slice(self, slc: torch.Tensor) -> torch.Tensor:
        out = self.backbone(slc)
        if isinstance(out, (tuple, list)):
            out = out[0]
        while out.dim() > 2:
            out = out.mean(dim=-1)
        return out


# ---------------------------------------------------------------------------
# DINOv3 (priority-ordered real weights; DINOv2 ungated fallback — never random)
# ---------------------------------------------------------------------------

_HF_DINOV3   = 'facebook/dinov3-vits16-pretrain-lvd1689m'
_TIMM_DINOV2 = 'vit_small_patch14_dinov2.lvd142m'


class DinoV3VolumeEncoder(_VolumeEncoderBase):
    imagenet_norm = True

    def __init__(self, max_slices: int = 32):
        super().__init__(max_slices)
        self.backbone = None
        self.source, self.feat_dim = None, None
        for loader in (self._try_timm_dinov3, self._try_timm_dinov2):
            try:
                loader()
                break
            except Exception as e:
                log.info(f"  DINO load attempt '{loader.__name__}' failed: {e}")
        if self.backbone is None:
            raise RuntimeError(
                "Could not load any real DINO backbone (tried timm DINOv3, timm "
                "DINOv2). Refusing to fall back to random/supervised weights — see "
                "WEIGHTS_SETUP.md. Fix the install (pip install -U timm) or use MedViT."
            )
        self._freeze()
        log.info(f"DinoV3VolumeEncoder | source={self.source} | feat_dim={self.feat_dim}")

    def _try_timm_dinov3(self):
        import timm
        name = 'vit_small_patch16_dinov3.lvd1689m'
        if name not in timm.list_models():
            raise RuntimeError(f'{name} not in this timm version')
        self.backbone = timm.create_model(name, pretrained=True, num_classes=0,
                                        dynamic_img_size=True)
        self.feat_dim = self.backbone.num_features
        self.source = f'timm:{name}'

    def _try_timm_dinov2(self):
        import timm
        self.backbone = timm.create_model(_TIMM_DINOV2, pretrained=True, num_classes=0,
                                        dynamic_img_size=True)
        self.feat_dim = self.backbone.num_features
        self.source = f'timm:{_TIMM_DINOV2}'

    def _encode_slice(self, slc: torch.Tensor) -> torch.Tensor:
        # num_classes=0 → forward returns the pooled embedding (B, num_features).
        return self.backbone(slc)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_encoder(name: str, medvit_pretrained_path: str = '', max_slices: int = 32,
                  medvit_size: str = 'small') -> _VolumeEncoderBase:
    name = name.lower()
    if name in ('medvit', 'med_vit'):
        return MedViTVolumeEncoder(medvit_pretrained_path, model_size=medvit_size,
                                   max_slices=max_slices)
    if name in ('dino', 'dinov3', 'dino_v3'):
        return DinoV3VolumeEncoder(max_slices=max_slices)
    raise ValueError(f"Unknown encoder '{name}' (expected 'medvit' or 'dinov3')")


__all__ = ['MedViTVolumeEncoder', 'DinoV3VolumeEncoder', 'build_encoder']
