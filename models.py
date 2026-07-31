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


NORM_KINDS = ('instance', 'group', 'batch')


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: y = x * (1 + gamma(c)) + beta(c).

    Used to tell the DECODER which contrast phase to synthesise while the encoder
    stays phase-agnostic — so one encoder learns anatomy from every phase's
    (NCCT, target) pairs and only the decoder specialises. Modulation is
    per-channel and spatially uniform, which is the point: contrast phase sets an
    organ's HU LEVEL, not its shape.

    Zero-init: the last layer starts at zero, so gamma=beta=0 and the block is an
    exact identity at step 0. The conditioned model therefore begins bitwise
    equal to the unconditioned one, can be fine-tuned from an existing
    checkpoint, and the learned |gamma| magnitude is itself a result — if it
    stays at zero, conditioning bought nothing.
    """

    def __init__(self, cond_dim: int, channels: int, hidden: int = 0):
        super().__init__()
        hidden = hidden or max(cond_dim, channels // 4)
        self.to_scale_shift = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2),
        )
        nn.init.zeros_(self.to_scale_shift[-1].weight)
        nn.init.zeros_(self.to_scale_shift[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_scale_shift(cond).chunk(2, dim=1)
        # Rank-agnostic broadcast: (B, C, 1, 1) in 2-D, (B, C, 1, 1, 1) in 3-D.
        # A literal .view(B, C, 1, 1) would work today and break silently the
        # first time this runs with dims=3 — see tests/test_phase_cond.py.
        shape = (x.shape[0], -1) + (1,) * (x.ndim - 2)
        return x * (1 + gamma.view(*shape)) + beta.view(*shape)


def _norm(dims: int, ch: int, kind: str = 'instance') -> nn.Module:
    """Normalisation layer, selectable because the choice decides whether a
    patch-based generator can tile seamlessly.

      instance — normalises each sample by ITS OWN spatial mean/std. Two
                 overlapping tiles covering the same voxel therefore apply two
                 different, content-dependent affine transforms to it, and the
                 disagreement is a DC offset that overlap-blending cannot cancel.
                 This is the default only because it is what the existing runs
                 were trained with; `norm_attribution.py` measures the cost.
      group    — normalises over channel groups within a sample. Still
                 sample-dependent, but far less sensitive to the tile's overall
                 intensity than per-channel spatial statistics.
      batch    — at eval() applies FIXED running statistics, so the transform is
                 completely independent of tile content. The only option here
                 that is tile-invariant by construction.

    Instance/batch are dims-specific; GroupNorm is dims-agnostic and works for
    both 2-D and 3-D unchanged.
    """
    if kind == 'instance':
        return _inorm(dims)(ch)
    if kind == 'batch':
        return _bnorm(dims)(ch)
    if kind == 'group':
        # 8 groups where possible; must divide ch (all channel counts here are
        # powers of two, so this holds for ch >= 8).
        return nn.GroupNorm(min(8, ch) if ch % min(8, ch) == 0 else 1, ch)
    raise ValueError(f'unknown norm kind {kind!r}; expected one of {NORM_KINDS}')


# ---------------------------------------------------------------------------
# U-Net building blocks (shared by 2-D and 3-D)
# ---------------------------------------------------------------------------

class _EncBlock(nn.Module):
    def __init__(self, dims, in_ch, out_ch, dropout=0.0, norm='instance'):
        super().__init__()
        Conv  = _conv(dims)
        Drop  = _drop(dims)
        layers = [
            Conv(in_ch, out_ch, 3, padding=1), _norm(dims, out_ch, norm), nn.LeakyReLU(0.2, inplace=True),
            Conv(out_ch, out_ch, 3, padding=1), _norm(dims, out_ch, norm), nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout > 0:
            layers.append(Drop(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _DecBlock(nn.Module):
    def __init__(self, dims, in_ch, out_ch, dropout=0.2, norm='instance'):
        super().__init__()
        Conv = _conv(dims)
        Drop = _drop(dims)
        self.block = nn.Sequential(
            Conv(in_ch, out_ch, 3, padding=1), _norm(dims, out_ch, norm), nn.LeakyReLU(0.2, inplace=True),
            Drop(dropout),
            Conv(out_ch, out_ch, 3, padding=1), _norm(dims, out_ch, norm), nn.LeakyReLU(0.2, inplace=True),
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
        in_channels:    Input feature maps (default 1). Set to 2k+1 to feed a
                        stack of adjacent axial slices as channels ("2.5-D") and
                        predict the centre slice — the output stays 1 channel.
        base_channels:  Feature maps at the first encoder level (default 64).
        dropout:        Dropout rate in decoder blocks (default 0.2).
        pool_stride:    Pooling / up-conv stride.
                        • dims=2: always 2 (ignored if set).
                        • dims=3: (2,2,2) for isotropic volumes,
                                  (1,2,2) for thin depth patches (8–16 slices).
                        Default when dims=3: (1,2,2) — safe for patch_depth<16.
        use_phase_cond: Modulate the DECODER with a contrast-phase embedding
                        (FiLM). Off by default; when off, no conditioning module
                        is constructed at all, so the parameter set and every RNG
                        draw are identical to a model built without this feature.
        n_phases:       Number of target phases the embedding can address.
        cond_dim:       Width of the phase embedding.

    Input/Output:
        dims=2: (B, in_channels, H, W)     → (B, 1, H, W)
        dims=3: (B, in_channels, D, H, W)  → (B, 1, D, H, W)
        forward(x, phase=None); `phase` is a LongTensor (B,) of phase ids and is
        required exactly when use_phase_cond is True.
    """

    def __init__(
        self,
        dims:          int = 2,
        base_channels: int = 64,
        dropout:       float = 0.2,
        pool_stride    = None,
        norm:          str = 'instance',
        in_channels:   int = 1,
        use_phase_cond: bool = False,
        n_phases:      int = 2,
        cond_dim:      int = 64,
    ):
        super().__init__()
        self.dims = dims
        self.norm = norm
        self.in_channels = in_channels
        self.use_phase_cond = use_phase_cond
        self.n_phases = n_phases
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
        self.enc1 = _EncBlock(dims, in_channels, ch, dropout=0.0, norm=norm)
        self.enc2 = _EncBlock(dims, ch,   ch * 2, dropout=0.0, norm=norm)
        self.enc3 = _EncBlock(dims, ch*2, ch * 4, dropout=0.0, norm=norm)
        self.enc4 = _EncBlock(dims, ch*4, ch * 8, dropout=0.0, norm=norm)
        Pool = _pool(dims)
        self.pool = Pool(kernel_size=stride, stride=stride)

        # Bottleneck
        Drop = _drop(dims)
        self.bottleneck = nn.Sequential(
            Conv(bn, bn, 3, padding=1), _norm(dims, bn, norm), nn.LeakyReLU(0.2, inplace=True), Drop(dropout),
            Conv(bn, bn, 3, padding=1), _norm(dims, bn, norm), nn.LeakyReLU(0.2, inplace=True), Drop(dropout),
        )

        # Decoder (stride matches pool)
        self.up4 = ConvT(bn,   ch*4, kernel_size=stride, stride=stride)
        self.dec4 = _DecBlock(dims, ch*4 + ch*8, ch*4, dropout, norm=norm)

        self.up3 = ConvT(ch*4, ch*2, kernel_size=stride, stride=stride)
        self.dec3 = _DecBlock(dims, ch*2 + ch*4, ch*2, dropout, norm=norm)

        self.up2 = ConvT(ch*2, ch,   kernel_size=stride, stride=stride)
        self.dec2 = _DecBlock(dims, ch + ch*2, ch, dropout, norm=norm)

        self.up1 = ConvT(ch,   ch,   kernel_size=stride, stride=stride)
        self.dec1 = _DecBlock(dims, ch + ch, ch, dropout, norm=norm)

        # Sigmoid, not Tanh: dataset normalises HU -> [0, 1] (see dataset.py),
        # so the output range must match. Tanh's [-1, 1] range wastes its
        # negative half and saturates (vanishing gradient) near the positive
        # targets we actually train against.
        self.out_conv = nn.Sequential(Conv(ch, 1, 1), nn.Sigmoid())

        # Phase conditioning — DECODER ONLY, so the encoder stays phase-agnostic
        # and learns anatomy from every phase's pairs.
        #
        # Built strictly inside this `if`. Constructing these modules draws from
        # the global RNG, so instantiating them unconditionally — even unused —
        # would shift the init of every layer created afterwards and silently
        # change baseline results under a fixed seed.
        if use_phase_cond:
            self.phase_emb = nn.Embedding(n_phases, cond_dim)
            self.film_bottleneck = FiLM(cond_dim, bn)
            self.film_dec4 = FiLM(cond_dim, ch * 4)
            self.film_dec3 = FiLM(cond_dim, ch * 2)
            self.film_dec2 = FiLM(cond_dim, ch)
            self.film_dec1 = FiLM(cond_dim, ch)

        n = sum(p.numel() for p in self.parameters()) / 1e6
        log.info(f"UNetGenerator | dims={dims} | in_ch={in_channels} | "
                 f"base_ch={base_channels} | pool_stride={stride} | norm={norm}"
                 + (f" | phase_cond({n_phases})" if use_phase_cond else "")
                 + f" | {n:.2f}M params")

    def film_stats(self) -> dict:
        """Mean |gamma| per injection point — log this to history.json.

        gamma converging to ~0 means the decoder ignored the phase input, which
        is a result in its own right: conditioning bought nothing.
        """
        if not self.use_phase_cond:
            return {}
        with torch.no_grad():
            cond = self.phase_emb(torch.arange(self.n_phases,
                                               device=self.phase_emb.weight.device))
            out = {}
            for name in ('bottleneck', 'dec4', 'dec3', 'dec2', 'dec1'):
                film = getattr(self, f'film_{name}')
                gamma, _ = film.to_scale_shift(cond).chunk(2, dim=1)
                out[f'gamma_{name}'] = float(gamma.abs().mean())
        return out

    def forward(self, x: torch.Tensor,
                phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Encoder is deliberately NOT conditioned: it sees every phase's pairs
        # and learns phase-agnostic anatomy.
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b  = self.bottleneck(self.pool(e4))

        if not self.use_phase_cond:
            if phase is not None:
                raise ValueError(
                    "phase was given but use_phase_cond=False — the model would "
                    "silently ignore it. Build with use_phase_cond=True.")
            d  = self.up4(b);  d = self.dec4(torch.cat([d, e4], dim=1))
            d  = self.up3(d);  d = self.dec3(torch.cat([d, e3], dim=1))
            d  = self.up2(d);  d = self.dec2(torch.cat([d, e2], dim=1))
            d  = self.up1(d);  d = self.dec1(torch.cat([d, e1], dim=1))
            return self.out_conv(d)

        if phase is None:
            raise ValueError("use_phase_cond=True requires a `phase` tensor (B,)")
        c = self.phase_emb(phase.reshape(-1).long())

        b  = self.film_bottleneck(b, c)
        d  = self.up4(b);  d = self.film_dec4(self.dec4(torch.cat([d, e4], dim=1)), c)
        d  = self.up3(d);  d = self.film_dec3(self.dec3(torch.cat([d, e3], dim=1)), c)
        d  = self.up2(d);  d = self.film_dec2(self.dec2(torch.cat([d, e2], dim=1)), c)
        d  = self.up1(d);  d = self.film_dec1(self.dec1(torch.cat([d, e1], dim=1)), c)
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
