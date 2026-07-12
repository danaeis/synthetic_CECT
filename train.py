"""
Entry point for the literature-baseline NCCT → CECT training run.

Usage:
    cd train/literature_baseline

    # Baseline run (2-D, pix2pixHD losses)
    python train.py

    # Switch to 3-D patches (edit config.py: PATCH_DEPTH=8, DIMS=3, PATCH_SIZE=96)
    # or override from CLI:
    python train.py --patch_depth 8 --dims 3 --patch_size 96

    # Ablation: add extra losses
    python train.py --use_ssim --use_gradient

    # Disable a baseline loss
    python train.py --no_perceptual

    # Resume from latest checkpoint
    python train.py --resume
"""

import argparse
import json
import logging
import re
from pathlib import Path

import torch

from config import train_config
from dataset import build_loaders
from trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(description='Literature baseline NCCT→CECT')
    p.add_argument('--resume',      action='store_true')
    p.add_argument('--epochs',      type=int,   default=None)
    p.add_argument('--batch_size',  type=int,   default=None)
    p.add_argument('--lr',          type=float, default=None)
    p.add_argument('--output_dir',  type=str,   default=None)
    p.add_argument('--patch_size',  type=int,   default=None)
    p.add_argument('--patch_depth', type=int,   default=None,
                   help='1=2D (default), >1=3D (must also set --dims 3)')
    p.add_argument('--dims',        type=int,   default=None, choices=[2, 3])

    p.add_argument('--perceptual_backbone', type=str, default=None, choices=['vgg', 'dino'])
    p.add_argument('--saliency_mode',       type=str, default=None, choices=['heuristic', 'dino'])

    # Loss flags: --use_X / --no_X
    for flag in ['adversarial', 'perceptual', 'feature_matching',
                 'ssim', 'gradient', 'frequency',
                 'organ', 'saliency', 'cycle', 'seg_consistency']:
        g = p.add_mutually_exclusive_group()
        g.add_argument(f'--use_{flag}',  dest=f'use_{flag}', action='store_true', default=None)
        g.add_argument(f'--no_{flag}',   dest=f'use_{flag}', action='store_false')
    return p.parse_args()


def _apply(cfg: dict, args) -> dict:
    c = cfg.copy()
    if args.epochs      is not None: c['epochs']       = args.epochs
    if args.batch_size  is not None: c['batch_size']   = args.batch_size
    if args.lr          is not None: c['learning_rate']= args.lr
    if args.output_dir  is not None: c['output_dir']   = Path(args.output_dir)
    if args.patch_size  is not None: c['patch_size']   = args.patch_size
    if args.patch_depth is not None: c['patch_depth']  = args.patch_depth
    if args.dims        is not None: c['dims']         = args.dims
    if args.perceptual_backbone is not None: c['perceptual_backbone'] = args.perceptual_backbone
    if args.saliency_mode       is not None: c['saliency_mode']       = args.saliency_mode
    for flag in ['adversarial', 'perceptual', 'feature_matching',
                 'ssim', 'gradient', 'frequency',
                 'organ', 'saliency', 'cycle', 'seg_consistency']:
        v = getattr(args, f'use_{flag}', None)
        if v is not None:
            c[f'use_{flag}'] = v

    # Auto-derive dims from patch_depth if not explicitly given
    if args.dims is None and 'patch_depth' in c:
        c['dims'] = 3 if c['patch_depth'] > 1 else 2

    return c


def _latest_ckpt(out: Path):
    ckpts = sorted(out.glob('ckpt_ep*.pth'))
    if not ckpts:
        return None, 0
    m = re.search(r'ckpt_ep(\d+)', ckpts[-1].name)
    return ckpts[-1], int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = _parse()
    config = _apply(train_config, args)
    out    = Path(config['output_dir'])
    out.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    with open(out / 'run_config.json', 'w') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in config.items()}, f, indent=2)

    # Add file log
    fh = logging.FileHandler(out / 'train.log', mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logging.getLogger().addHandler(fh)

    log.info('=' * 65)
    log.info('LITERATURE BASELINE — NCCT → CECT')
    log.info('=' * 65)
    log.info(f"Output     : {out}")
    log.info(f"Device     : {config['device']}")
    log.info(f"Dims       : {config.get('dims', 2)}-D  "
             f"(patch_depth={config.get('patch_depth', 1)}, "
             f"patch_size={config.get('patch_size')})")
    log.info(f"Target     : NCCT → {config['target_phase']}")
    log.info(f"HU window  : [{config['hu_min']}, {config['hu_max']}] → [0,1]")
    log.info(f"RAM budget : {config['max_train_patches']:,} train / "
             f"{config['max_val_patches']:,} val patches")
    active = ['L1'] + [f for f in [
        'adversarial', 'perceptual', 'feature_matching',
        'ssim', 'gradient', 'frequency',
        'organ', 'saliency', 'cycle', 'seg_consistency',
    ] if config.get(f'use_{f}')]
    log.info(f"Losses     : {' + '.join(active)}")
    log.info('')

    # ── Data ────────────────────────────────────────────────────────────────
    log.info("Building loaders …")
    train_loader, val_loader = build_loaders(config)

    if len(train_loader) == 0:
        log.error("No training batches — check data path and HU/patch settings.")
        return

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = Trainer(config)
    start   = 0
    if args.resume:
        ckpt, start = _latest_ckpt(out)
        if ckpt:
            start = trainer.load_checkpoint(str(ckpt))
        else:
            log.warning("--resume: no checkpoint found, starting fresh")

    try:
        trainer.train(train_loader, val_loader, config['epochs'], start_epoch=start)
    except KeyboardInterrupt:
        log.info("Interrupted — saving emergency checkpoint …")
        trainer._save_checkpoint(trainer.current_epoch, is_best=False)

    log.info("Done.")


if __name__ == '__main__':
    main()
