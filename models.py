"""
2-D / 3-D model architectures for the literature baseline.

Both Generator and Discriminator accept a `dims` parameter (2 or 3).
Setting dims=2 gives the standard 2-D U-Net / PatchGAN used in the
literature (Liu22, Hau21, Cho21, Yan24c …).
Setting dims=3 swaps every Conv/Norm/Pool/Dropout layer to its 3-D
equivalent — the topology is identical.

Switching from 2-D to 3-D training requires only:
  1. config['patch_depth'] > 1   (e.g. 8 or 16)
  2. config['dims'] = 3
Everything else (loss, trainer, dataset) works unchanged.
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — select the right nn class for dims=2 or dims=3
# ---------------------------------------------------------------------------

def _conv(dims):       return getattr(nn, f'Conv{dims}d')
def _convT(dims):      return getattr(nn, f'ConvTranspose{dims}d')
def _inorm(dims):      return getattr(nn, f'InstanceNorm{dims}d')
def _bnorm(dims):      return getattr(nn, f'BatchNorm{dims}d')
def _pool(dims):       return getattr(nn, f'MaxPool{dims}d')
def _drop(dims):       return getattr(nn, f'Dropout{dims}d')


# ---------------------------------------------------------------------------
# U-Net building blocks (shared by 2-D and 3-D)
# ---------------------------------------------------------------------------

class _EncBlock(nn.Module):
    def __init__(self, dims, in_ch, out_ch, dropout=0.0):
        super().__init__()
        Conv  = _conv(dims)
        Norm  = _inorm(dims)
        Drop  = _drop(dims)
        layers = [
            Conv(in_ch, out_ch, 3, padding=1), Norm(out_ch), nn.LeakyReLU(0.2, inplace=True),
            Conv(out_ch, out_ch, 3, padding=1), Norm(out_ch), nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout > 0:
            layers.append(Drop(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _DecBlock(nn.Module):
    def __init__(self, dims, in_ch, out_ch, dropout=0.2):
        super().__init__()
        Conv = _conv(dims)
        Norm = _inorm(dims)
        Drop = _drop(dims)
        self.block = nn.Sequential(
            Conv(in_ch, out_ch, 3, padding=1), Norm(out_ch), nn.LeakyReLU(0.2, inplace=True),
            Drop(dropout),
            Conv(out_ch, out_ch, 3, padding=1), Norm(out_ch), nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# U-Net Generator
# ---------------------------------------------------------------------------

class UNetGenerator(nn.Module):
    """
    2-D or 3-D U-Net generator for NCCT → CECT translation.

    Args:
        dims:           2 for 2-D convolutions, 3 for 3-D convolutions.
        base_channels:  Feature maps at the first encoder level (default 64).
        dropout:        Dropout rate in decoder blocks (default 0.2).
        pool_stride:    Pooling / up-conv stride.
                        • dims=2: always 2 (ignored if set).
                        • dims=3: (2,2,2) for isotropic volumes,
                                  (1,2,2) for thin depth patches (8–16 slices).
                        Default when dims=3: (1,2,2) — safe for patch_depth<16.

    Input/Output:
        dims=2: (B, 1, H, W)       → (B, 1, H, W)
        dims=3: (B, 1, D, H, W)    → (B, 1, D, H, W)
    """

    def __init__(
        self,
        dims:          int = 2,
        base_channels: int = 64,
        dropout:       float = 0.2,
        pool_stride    = None,
    ):
        super().__init__()
        self.dims = dims
        ch  = base_channels
        bn  = ch * 8                   # bottleneck channels

        Conv  = _conv(dims)
        ConvT = _convT(dims)

        # Stride / kernel for pool and up-conv
        if dims == 2:
            stride = 2
        else:
            stride = pool_stride if pool_stride is not None else (1, 2, 2)

        # Encoder
        self.enc1 = _EncBlock(dims, 1,    ch,     dropout=0.0)
        self.enc2 = _EncBlock(dims, ch,   ch * 2, dropout=0.0)
        self.enc3 = _EncBlock(dims, ch*2, ch * 4, dropout=0.0)
        self.enc4 = _EncBlock(dims, ch*4, ch * 8, dropout=0.0)
        Pool = _pool(dims)
        self.pool = Pool(kernel_size=stride, stride=stride)

        # Bottleneck
        Drop = _drop(dims)
        Norm = _inorm(dims)
        self.bottleneck = nn.Sequential(
            Conv(bn, bn, 3, padding=1), Norm(bn), nn.LeakyReLU(0.2, inplace=True), Drop(dropout),
            Conv(bn, bn, 3, padding=1), Norm(bn), nn.LeakyReLU(0.2, inplace=True), Drop(dropout),
        )

        # Decoder (stride matches pool)
        self.up4 = ConvT(bn,   ch*4, kernel_size=stride, stride=stride)
        self.dec4 = _DecBlock(dims, ch*4 + ch*8, ch*4, dropout)

        self.up3 = ConvT(ch*4, ch*2, kernel_size=stride, stride=stride)
        self.dec3 = _DecBlock(dims, ch*2 + ch*4, ch*2, dropout)

        self.up2 = ConvT(ch*2, ch,   kernel_size=stride, stride=stride)
        self.dec2 = _DecBlock(dims, ch + ch*2, ch, dropout)

        self.up1 = ConvT(ch,   ch,   kernel_size=stride, stride=stride)
        self.dec1 = _DecBlock(dims, ch + ch, ch, dropout)

        # Sigmoid, not Tanh: dataset normalises HU -> [0, 1] (see dataset.py),
        # so the output range must match. Tanh's [-1, 1] range wastes its
        # negative half and saturates (vanishing gradient) near the positive
        # targets we actually train against.
        self.out_conv = nn.Sequential(Conv(ch, 1, 1), nn.Sigmoid())

        n = sum(p.numel() for p in self.parameters()) / 1e6
        log.info(f"UNetGenerator | dims={dims} | base_ch={base_channels} | "
                 f"pool_stride={stride} | {n:.2f}M params")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b  = self.bottleneck(self.pool(e4))

        d  = self.up4(b);  d = self.dec4(torch.cat([d, e4], dim=1))
        d  = self.up3(d);  d = self.dec3(torch.cat([d, e3], dim=1))
        d  = self.up2(d);  d = self.dec2(torch.cat([d, e2], dim=1))
        d  = self.up1(d);  d = self.dec1(torch.cat([d, e1], dim=1))
        return self.out_conv(d)


# ---------------------------------------------------------------------------
# PatchGAN Discriminator (with intermediate feature extraction)
# ---------------------------------------------------------------------------

class PatchGANDiscriminator(nn.Module):
    """
    2-D or 3-D 70×70 PatchGAN discriminator.

    forward(x, return_features=True) → (logits, List[feature_maps])
    Used for feature-matching loss (pix2pixHD style, Hau21).

    Args:
        dims:         2 or 3.
        ndf:          Base feature maps (default 64).
        n_layers:     Number of strided blocks (default 4).
        stride_3d:    Stride for strided blocks when dims=3.
                      (1,2,2) keeps depth, (2,2,2) downsamples all.
                      Default (1,2,2) — safe for patch_depth 8–16.

    Input shapes:
        dims=2: (B, 1, H, W)
        dims=3: (B, 1, D, H, W)
    """

    def __init__(self, dims: int = 2, ndf: int = 64, n_layers: int = 4, stride_3d=None):
        super().__init__()
        self.dims = dims
        Conv = _conv(dims)
        BN   = _bnorm(dims)

        # Strided-block stride: 2 for 2-D, anisotropic tuple for 3-D
        if dims == 2:
            s = 2
            s1 = 1
        else:
            s  = stride_3d if stride_3d is not None else (1, 2, 2)
            s1 = tuple(1 for _ in s) if isinstance(s, tuple) else 1

        self.blocks = nn.ModuleList()

        # Block 0 — no norm
        self.blocks.append(nn.Sequential(
            Conv(1, ndf, 4, stride=s, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ))

        nf = ndf
        for _ in range(1, n_layers - 1):
            nf_prev, nf = nf, min(nf * 2, 512)
            self.blocks.append(nn.Sequential(
                Conv(nf_prev, nf, 4, stride=s, padding=1),
                BN(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ))

        nf_prev, nf = nf, min(nf * 2, 512)
        self.blocks.append(nn.Sequential(
            Conv(nf_prev, nf, 4, stride=s1, padding=1),
            BN(nf),
            nn.LeakyReLU(0.2, inplace=True),
        ))

        self.output_conv = Conv(nf, 1, 4, stride=s1, padding=1)

        n = sum(p.numel() for p in self.parameters()) / 1e6
        log.info(f"PatchGANDiscriminator | dims={dims} | ndf={ndf} | "
                 f"stride={s} | {n:.2f}M params")

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        features: List[torch.Tensor] = []
        h = x
        for block in self.blocks:
            h = block(h)
            features.append(h)
        logits = self.output_conv(h)

        if return_features:
            return logits, features
        return logits, None
