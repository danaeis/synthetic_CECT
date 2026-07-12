"""
Trainer for the literature-baseline NCCT → CECT model.

Batch format (from CTPairDataset):
    {'source': Tensor(B, 1, H, W)  or (B, 1, D, H, W),
     'target': Tensor(B, 1, H, W)  or (B, 1, D, H, W)}

The trainer is completely independent of the dimensionality of the patches —
it delegates all shape-awareness to the models and losses.
"""

import gc
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import CompositeLoss
from models import PatchGANDiscriminator, UNetGenerator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().squeeze().numpy()


def _psnr(pred: np.ndarray, target: np.ndarray, data_range: float) -> float:
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return float(10.0 * np.log10(data_range ** 2 / mse))


def _ssim(pred: np.ndarray, target: np.ndarray, data_range: float) -> float:
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu1, mu2 = pred.mean(), target.mean()
    s1, s2   = pred.std(), target.std()
    s12      = np.mean((pred - mu1) * (target - mu2))
    return float(((2*mu1*mu2 + C1) * (2*s12 + C2)) /
                 ((mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2)))


def _psnr_ssim(pred: torch.Tensor, target: torch.Tensor):
    """Compute PSNR and SSIM on first item. For 3-D tensors use centre slice."""
    p = _np(pred[0])
    t = _np(target[0])
    if p.ndim == 3:                        # (D, H, W) — take centre slice
        mid = p.shape[0] // 2
        p, t = p[mid], t[mid]
    # Fixed data_range=1.0: both tensors are globally normalised to [0, 1] by
    # dataset.py (see hu_min/hu_max clip + rescale), so PSNR/SSIM must use
    # that same fixed range. Using the per-sample target min/max instead (as
    # this used to) makes the metric incomparable across patches/epochs —
    # a near-uniform patch (small true dynamic range) would get an arbitrary
    # data_range that has nothing to do with the network's actual output scale.
    return _psnr(p, t, 1.0), _ssim(p, t, 1.0)


class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best     = float('inf')
        self.counter  = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best:
            self.best = val_loss; self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Training loop for the literature baseline.

    Works for 2-D (patch_depth=1, dims=2) and 3-D (patch_depth>1, dims=3)
    without any code change — the models and losses handle the difference.
    """

    def __init__(self, config: Dict):
        self.cfg        = config
        self.device     = config['device']
        self.out        = Path(config['output_dir'])
        self.out.mkdir(parents=True, exist_ok=True)
        self.samples    = self.out / 'samples'
        self.samples.mkdir(exist_ok=True)

        dims = config.get('dims', 2)

        # ── models ──────────────────────────────────────────────────────────
        self.G = UNetGenerator(
            dims          = dims,
            base_channels = config['generator_base_channels'],
            dropout       = config.get('generator_dropout', 0.2),
        ).to(self.device)

        self.use_adv = config.get('use_adversarial', True)
        if self.use_adv:
            self.D = PatchGANDiscriminator(dims=dims, ndf=64).to(self.device)
        else:
            self.D = None

        # ── optimisers ──────────────────────────────────────────────────────
        betas = config.get('betas', (0.5, 0.999))
        self.opt_G = optim.Adam(
            self.G.parameters(),
            lr           = config.get('learning_rate', 2e-4),
            betas        = betas,
            weight_decay = config.get('weight_decay', 1e-5),
        )
        if self.D:
            self.opt_D = optim.Adam(
                self.D.parameters(),
                lr    = config.get('lr_disc', 1e-4),
                betas = betas,
            )

        # ── scheduler ───────────────────────────────────────────────────────
        if config.get('use_cosine_schedule', True):
            self.sched_G = sched.CosineAnnealingWarmRestarts(
                self.opt_G,
                T_0    = config.get('cosine_t0', 15),
                T_mult = config.get('cosine_tmult', 2),
                eta_min= config.get('cosine_eta_min', 5e-7),
            )
        else:
            self.sched_G = None

        # ── loss ────────────────────────────────────────────────────────────
        self.criterion  = CompositeLoss(config).to(self.device)
        self.disc_freq  = config.get('disc_update_freq', 1)
        self.use_cycle  = config.get('use_cycle', False)
        self.use_fm     = config.get('use_feature_matching', True)

        # ── AMP ─────────────────────────────────────────────────────────────
        # use_mixed_precision defaults True independent of device; gate it on
        # actual CUDA availability too, since GradScaler('cuda', enabled=True)
        # / autocast('cuda', enabled=True) assume a CUDA context and error out
        # on a CPU-only machine even though `enabled` looks like a runtime switch.
        self.use_amp  = config.get('use_mixed_precision', True) and self.device == 'cuda'
        self.scaler_G = GradScaler('cuda', enabled=self.use_amp)
        self.scaler_D = GradScaler('cuda', enabled=self.use_amp)

        # ── state ───────────────────────────────────────────────────────────
        self.global_step   = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stop    = EarlyStopping(config.get('early_stop_patience', 12))

        self.history: Dict[str, List] = {k: [] for k in [
            'epoch', 'lr_gen',
            'train_gen_total', 'train_l1', 'train_adv', 'train_perc', 'train_fm',
            'train_ssim', 'train_grad', 'train_freq', 'train_organ',
            'train_sal', 'train_cycle', 'train_seg', 'train_disc',
            'val_loss', 'val_psnr', 'val_ssim',
        ]}

        self._log_active_losses()

    # -----------------------------------------------------------------------
    def _log_active_losses(self):
        flags = ['adversarial', 'perceptual', 'feature_matching',
                 'ssim', 'gradient', 'frequency',
                 'organ', 'saliency', 'cycle', 'seg_consistency']
        active = ['L1'] + [f for f in flags if self.cfg.get(f'use_{f}')]
        log.info(f"Active losses: {' + '.join(active)}")
        log.info(f"Dims: {self.cfg.get('dims', 2)}-D  |  "
                 f"patch_depth={self.cfg.get('patch_depth', 1)}  |  "
                 f"patch_size={self.cfg.get('patch_size')}")

    # -----------------------------------------------------------------------
    def _disc_step(self, real, fake):
        """Train discriminator one step. Returns (loss_val, real_features)."""
        self.opt_D.zero_grad()
        with autocast('cuda', enabled=self.use_amp):
            logits_real, real_feats = self.D(real,          return_features=True)
            logits_fake, _          = self.D(fake.detach(), return_features=False)
            loss_D = self.criterion.adv_loss.disc_loss(logits_real, logits_fake)
        self.scaler_D.scale(loss_D).backward()
        self.scaler_D.unscale_(self.opt_D)
        nn.utils.clip_grad_norm_(self.D.parameters(), 10.0)
        self.scaler_D.step(self.opt_D)
        self.scaler_D.update()
        return loss_D.item(), real_feats

    # -----------------------------------------------------------------------
    def _train_step(self, batch: Dict) -> Dict:
        source = batch['source'].to(self.device)
        target = batch['target'].to(self.device)
        mask   = batch['mask'].to(self.device) if 'mask' in batch else None

        # Generator forward for the discriminator's fake input only — no_grad
        # because it is immediately detached; avoids building an autograd
        # graph that would just be discarded (halves G forward-passes/step).
        if self.D is not None and (self.global_step % self.disc_freq == 0):
            with torch.no_grad(), autocast('cuda', enabled=self.use_amp):
                fake_for_d = self.G(source)
            loss_D_val, real_feats = self._disc_step(target, fake_for_d)
        else:
            loss_D_val, real_feats = 0.0, None

        # Generator step (re-forward so gradients flow)
        self.opt_G.zero_grad()
        with autocast('cuda', enabled=self.use_amp):
            fake = self.G(source)

            adv_logits = None
            fake_feats = None
            if self.D is not None:
                adv_logits, fake_feats = self.D(fake, return_features=True)

            cycle_pred = None
            if self.use_cycle:
                cycle_pred = self.G(fake)            # fake → G → should ≈ source

            loss_G, ld = self.criterion(
                pred            = fake,
                target          = target,
                source          = source,
                mask            = mask,
                adv_fake_logits = adv_logits,
                real_features   = real_feats,
                fake_features   = fake_feats,
                cycle_pred      = cycle_pred,
            )

        self.scaler_G.scale(loss_G).backward()
        self.scaler_G.unscale_(self.opt_G)
        nn.utils.clip_grad_norm_(self.G.parameters(), 10.0)
        self.scaler_G.step(self.opt_G)
        self.scaler_G.update()

        self.global_step += 1
        return {
            'gen_total': ld['total'],
            'disc':      loss_D_val,
            'l1':        ld.get('l1', 0),
            'adv':       ld.get('adversarial', 0),
            'perc':      ld.get('perceptual', 0),
            'fm':        ld.get('feature_matching', 0),
            'ssim':      ld.get('ssim', 0),
            'grad':      ld.get('gradient', 0),
            'freq':      ld.get('frequency', 0),
            'organ':     ld.get('organ', 0),
            'sal':       ld.get('saliency', 0),
            'cycle':     ld.get('cycle', 0),
            'seg':       ld.get('seg_consistency', 0),
        }

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Dict:
        self.G.eval()
        losses, psnrs, ssims = [], [], []
        for batch in val_loader:
            src = batch['source'].to(self.device)
            tgt = batch['target'].to(self.device)
            with autocast('cuda', enabled=self.use_amp):
                fake = self.G(src)
            losses.append(torch.nn.functional.l1_loss(fake, tgt).item())
            p, s = _psnr_ssim(fake, tgt)
            psnrs.append(p); ssims.append(s)
        self.G.train()
        return {
            'val_loss': float(np.mean(losses)) if losses else 0.0,
            'val_psnr': float(np.mean(psnrs))  if psnrs  else 0.0,
            'val_ssim': float(np.mean(ssims))  if ssims  else 0.0,
        }

    # -----------------------------------------------------------------------
    def _save_samples(self, val_loader: DataLoader, epoch: int):
        """Save (source | fake | target) grid for the first validation batch."""
        self.G.eval()
        with torch.no_grad():
            batch = next(iter(val_loader))
            src = batch['source'].to(self.device)
            tgt = batch['target'].to(self.device)
            fake = self.G(src)

        n = min(4, src.size(0))

        def _mid(t):
            arr = _np(t)
            return arr[arr.shape[0] // 2] if arr.ndim == 3 else arr

        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        if n == 1: axes = axes[None]
        for i in range(n):
            for j, img in enumerate([src[i], fake[i], tgt[i]]):
                axes[i, j].imshow(_mid(img), cmap='gray', vmin=0, vmax=1)
                if i == 0:
                    axes[i, j].set_title(['NCCT', 'Generated', 'CECT'][j], fontsize=9)
                axes[i, j].axis('off')
        plt.suptitle(f'Epoch {epoch}', fontsize=11)
        plt.tight_layout()
        plt.savefig(self.samples / f'ep{epoch:03d}.png', dpi=120, bbox_inches='tight')
        plt.close()

        # Sort by mtime, not filename: a fresh (non-resumed) run restarts its
        # epoch counter at 1, and a filename/epoch-number sort would keep
        # stale high-numbered images left over from an earlier, longer run of
        # this same scenario instead of the current run's own latest samples.
        keep = self.cfg.get('keep_last_n_sample_epochs', 5)
        imgs = sorted(self.samples.glob('ep*.png'), key=lambda p: p.stat().st_mtime)
        for old in imgs[:-keep]:
            old.unlink(missing_ok=True)

        self.G.train()

    # -----------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, is_best: bool):
        state = {
            'epoch':    epoch,
            'G_state':  self.G.state_dict(),
            'opt_G':    self.opt_G.state_dict(),
            'best_val': self.best_val_loss,
        }
        if self.D:
            state['D_state'] = self.D.state_dict()
            state['opt_D']   = self.opt_D.state_dict()
        path = self.out / f'ckpt_ep{epoch:03d}.pth'
        torch.save(state, path)
        if is_best:
            torch.save(state, self.out / 'best_model.pth')
            log.info(f"  ★ best model saved (epoch {epoch})")
        # Same mtime-not-filename reasoning as _save_samples above.
        keep = self.cfg.get('keep_last_n_checkpoints', 3)
        ckpts = sorted(self.out.glob('ckpt_ep*.pth'), key=lambda p: p.stat().st_mtime)
        for old in ckpts[:-keep]:
            old.unlink(missing_ok=True)

    def load_checkpoint(self, path: str) -> int:
        state = torch.load(path, map_location=self.device)
        self.G.load_state_dict(state['G_state'])
        self.opt_G.load_state_dict(state['opt_G'])
        if 'D_state' in state and self.D:
            self.D.load_state_dict(state['D_state'])
            self.opt_D.load_state_dict(state['opt_D'])
        self.best_val_loss = state.get('best_val', float('inf'))
        ep = state.get('epoch', 0)
        log.info(f"Resumed from checkpoint epoch {ep}")
        return ep

    # -----------------------------------------------------------------------
    def _update_history(self, epoch, avgs, val):
        h = self.history
        h['epoch'].append(epoch)
        h['lr_gen'].append(self.opt_G.param_groups[0]['lr'])
        for k in ['gen_total', 'disc', 'l1', 'adv', 'perc', 'fm',
                  'ssim', 'grad', 'freq', 'organ', 'sal', 'cycle', 'seg']:
            h[f'train_{k}'].append(avgs.get(k, 0.0))
        h['val_loss'].append(val['val_loss'])
        h['val_psnr'].append(val['val_psnr'])
        h['val_ssim'].append(val['val_ssim'])

    def _save_history(self):
        hist = {k: [float(v) for v in vl] for k, vl in self.history.items()}
        with open(self.out / 'history.json', 'w') as f:
            json.dump(hist, f, indent=2)

    def _plot_history(self):
        if not self.history['epoch']:
            return
        ep = self.history['epoch']
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))

        ax = axes[0, 0]
        ax.plot(ep, self.history['train_gen_total'], label='Gen total')
        ax.plot(ep, self.history['train_disc'],      label='Disc')
        ax.set_title('Total losses'); ax.legend(); ax.grid(alpha=0.3)

        ax = axes[0, 1]
        for k, lbl in [('train_l1','L1'), ('train_adv','Adv'),
                        ('train_perc','Perc'), ('train_fm','FeatMatch')]:
            ax.plot(ep, self.history[k], label=lbl)
        ax.set_title('Core losses'); ax.legend(); ax.grid(alpha=0.3)

        ax = axes[0, 2]
        for k, lbl in [('train_ssim','SSIM'), ('train_grad','Grad'),
                        ('train_freq','Freq'), ('train_organ','Organ'),
                        ('train_sal','Sal'), ('train_cycle','Cycle'),
                        ('train_seg','Seg')]:
            if any(v > 0 for v in self.history[k]):
                ax.plot(ep, self.history[k], label=lbl)
        ax.set_title('Extra losses'); ax.legend(); ax.grid(alpha=0.3)

        axes[1, 0].plot(ep, self.history['val_loss']);  axes[1, 0].set_title('Val L1');   axes[1, 0].grid(alpha=0.3)
        axes[1, 1].plot(ep, self.history['val_psnr']);  axes[1, 1].set_title('Val PSNR'); axes[1, 1].grid(alpha=0.3)
        axes[1, 2].plot(ep, self.history['val_ssim']);  axes[1, 2].set_title('Val SSIM'); axes[1, 2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.out / 'curves.png', dpi=120, bbox_inches='tight')
        plt.close()

    # -----------------------------------------------------------------------
    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int,
        start_epoch:  int = 0,
    ):
        log.info(f"Training {start_epoch + 1} → {epochs}  |  device={self.device}")
        save_every = self.cfg.get('save_samples_interval', 1)

        for epoch in range(start_epoch + 1, epochs + 1):
            self.current_epoch = epoch
            self.criterion.set_epoch(epoch)
            self.G.train()
            if self.D: self.D.train()

            accum: Dict[str, List] = {k: [] for k in [
                'gen_total', 'disc', 'l1', 'adv', 'perc', 'fm',
                'ssim', 'grad', 'freq', 'organ', 'sal', 'cycle', 'seg'
            ]}

            pbar = tqdm(train_loader, desc=f'Ep {epoch}/{epochs}', leave=False)
            for batch in pbar:
                step = self._train_step(batch)
                for k in accum:
                    accum[k].append(step.get(k, 0.0))
                pbar.set_postfix(
                    total=f"{step['gen_total']:.4f}",
                    l1=f"{step['l1']:.4f}",
                )

            avgs = {k: float(np.mean(v)) if v else 0.0 for k, v in accum.items()}

            if self.sched_G:
                self.sched_G.step()

            val = self._validate(val_loader)

            log.info(
                f"Ep {epoch:3d}/{epochs} | "
                f"total={avgs['gen_total']:.4f}  l1={avgs['l1']:.4f}  "
                f"adv={avgs['adv']:.4f}  perc={avgs['perc']:.4f} | "
                f"val_loss={val['val_loss']:.4f}  PSNR={val['val_psnr']:.2f}  "
                f"SSIM={val['val_ssim']:.4f}  "
                f"lr={self.opt_G.param_groups[0]['lr']:.2e}"
            )

            self._update_history(epoch, avgs, val)
            self._save_history()
            if epoch % 5 == 0 or epoch == epochs:
                self._plot_history()

            is_best = val['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val['val_loss']
            self._save_checkpoint(epoch, is_best)

            if epoch % save_every == 0:
                self._save_samples(val_loader, epoch)

            if self.early_stop.step(val['val_loss']):
                log.info(f"Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        log.info(f"Done. Best val loss: {self.best_val_loss:.6f}")
