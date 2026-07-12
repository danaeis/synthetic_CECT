"""
Shared, frozen DINO backbone loader for the perceptual / saliency losses.

Tries the best real DINO weights actually available on this machine, in
priority order, and logs exactly which one loaded — no silent fallback to a
mislabeled or randomly-initialised model (see phase_detection/dino_encoder.py
for the failure mode this is designed to avoid: it silently falls through to
an ImageNet-*supervised* ViT, or literally random weights, while still
calling itself "DINO v3").

Priority (checked once per process, then cached):
  1. transformers AutoModel  facebook/dinov3-vits16-pretrain-lvd1689m
     Real DINOv3. Requires `pip install transformers huggingface_hub`, a HF
     account, and accepting the gated license at the model page before it
     will download.
  2. timm                    first '*dinov3*' entry in the timm registry
     Real DINOv3, if your timm version ships it. Still HF-gated the first
     time it downloads weights.
  3. timm                    vit_small_patch14_dinov2.lvd142m
     Real DINOv2 (not v3). Ungated, works today with the timm you already
     have installed — the safe default.

Verify which of these will work on your machine BEFORE training, in this
same order:

    # 1) real DINOv3 via transformers (gated — needs `huggingface-cli login`
    #    and license acceptance at the model page first)
    pip install -U transformers huggingface_hub
    python -c "from transformers import AutoModel; \
        AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m'); print('OK')"

    # 2) real DINOv3 via timm (only if your timm ships it)
    python -c "import timm; print([m for m in timm.list_models() if 'dinov3' in m])"

    # 3) real DINOv2 via timm (ungated fallback, should work now)
    python -c "import timm; timm.create_model('vit_small_patch14_dinov2.lvd142m', \
        pretrained=True, num_classes=0); print('OK')"

Whichever succeeds first at runtime is what gets used; `DinoSpatialBackbone.source`
tells you which.
"""

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

_HF_DINOV3 = 'facebook/dinov3-vits16-pretrain-lvd1689m'
_TIMM_DINOV2 = 'vit_small_patch14_dinov2.lvd142m'


def _try_hf_dinov3():
    from transformers import AutoModel
    model = AutoModel.from_pretrained(_HF_DINOV3)
    patch = getattr(model.config, 'patch_size', 16)
    dim   = model.config.hidden_size
    return model, 'hf:' + _HF_DINOV3, patch, dim, 'hf'


def _try_timm_dinov3():
    import timm
    names = [m for m in timm.list_models() if 'dinov3' in m]
    if not names:
        raise RuntimeError('timm has no dinov3 models in this version')
    name = names[0]
    model = timm.create_model(name, pretrained=True, num_classes=0)
    patch = model.patch_embed.patch_size[0]
    dim   = model.embed_dim
    return model, 'timm:' + name, patch, dim, 'timm'


def _try_timm_dinov2():
    import timm
    model = timm.create_model(_TIMM_DINOV2, pretrained=True, num_classes=0)
    patch = model.patch_embed.patch_size[0]
    dim   = model.embed_dim
    return model, 'timm:' + _TIMM_DINOV2, patch, dim, 'timm'


class DinoSpatialBackbone(nn.Module):
    """
    Frozen DINO ViT backbone returning SPATIAL patch-token feature maps
    (not a pooled/CLS vector) so it can be used like VGG's conv feature
    maps in a perceptual loss, or spatially compared to build a saliency
    weight map.

    forward(x) : (B, 1, H, W) in [0, 1]  ->  List[(B, C, h, w)]
                 one feature map per probed depth (shallow/mid/deep).
    """

    IMAGE_SIZE = 224

    def __init__(self):
        super().__init__()
        model = source = None
        for loader in (_try_hf_dinov3, _try_timm_dinov3, _try_timm_dinov2):
            try:
                model, source, patch, dim, kind = loader()
                break
            except Exception as e:
                log.info(f"  DINO backbone attempt '{loader.__name__}' failed: {e}")
        if model is None:
            raise RuntimeError(
                "Could not load any DINO backbone (tried HF DINOv3, timm DINOv3, "
                "timm DINOv2). Set perceptual_backbone='vgg' / saliency_mode='heuristic' "
                "in config if you want to train without it, or fix the install — see "
                "the priority-ordered commands in this file's docstring."
            )
        log.warning(f"DINO backbone loaded: {source}  (dim={dim}, patch={patch})")

        self.model  = model
        self.source = source
        self.kind   = kind          # 'hf' or 'timm'
        self.patch  = patch
        self.dim    = dim
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().clamp(0.0, 1.0)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
                           mode='bilinear', align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        No @torch.no_grad() here on purpose: backbone weights are frozen
        (requires_grad_(False) above) so no gradient accumulates in them,
        but gradients w.r.t. `x` must still flow through for this to work
        as a perceptual loss on the generator's output. Callers that only
        need a static weight map (e.g. saliency) should wrap the call site
        in `with torch.no_grad():` themselves.
        """
        x = self._prep(x)
        grid = self.IMAGE_SIZE // self.patch

        if self.kind == 'timm':
            depth = len(self.model.blocks)
            idx = sorted(set([max(0, depth // 3 - 1), max(0, 2 * depth // 3 - 1), depth - 1]))
            feats = self.model.forward_intermediates(
                x, indices=idx, output_fmt='NCHW', intermediates_only=True,
            )
            return list(feats)

        # transformers (HF) path
        out = self.model(pixel_values=x, output_hidden_states=True)
        hs = out.hidden_states                     # tuple of (B, N+prefix, C)
        depth = len(hs) - 1
        idx = sorted(set([max(1, depth // 3), max(1, 2 * depth // 3), depth]))
        n_prefix = getattr(self.model.config, 'num_register_tokens', 0) + 1  # + CLS
        feats = []
        for i in idx:
            tok = hs[i][:, n_prefix:, :]            # drop CLS (+ registers)
            B, N, C = tok.shape
            tok = tok.transpose(1, 2).reshape(B, C, grid, grid)
            feats.append(tok)
        return feats
