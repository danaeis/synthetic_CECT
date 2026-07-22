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

from config import train_config, resolve_organ_weights
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
    # Resume is now the default: if the output dir already holds a checkpoint we
    # continue from it. --resume is kept for backwards compatibility (no-op),
    # and --fresh forces a clean start that ignores any existing checkpoint.
    p.add_argument('--resume',      action='store_true',
                   help='(default) continue from the latest checkpoint if one exists')
    p.add_argument('--fresh',       action='store_true',
                   help='ignore any existing checkpoint and start training from scratch')
    p.add_argument('--device',      type=str,   default=None,
                   help="override device, e.g. 'cuda', 'cuda:0', 'cpu'")
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

    # Organ-weighting / curriculum knobs (see config.py for the rationale)
    p.add_argument('--selection_metric', type=str, default=None,
                   choices=['val_org_ssim', 'val_ssim', 'val_loss'],
                   help='metric picking best_model.pth (default val_org_ssim)')
    p.add_argument('--organ_weight_preset', type=str, default=None,
                   choices=['tiered', 'gi_zero'],
                   help="'tiered' full scheme | 'gi_zero' control (GI excluded, rest 1.0)")
    p.add_argument('--sample_mode', type=str, default=None, choices=['random', 'fixed'],
                   help='val patches in the per-epoch sample grid (default random)')
    p.add_argument('--sample_n',    type=int, default=None,
                   help='rows in the sample grid (default 4)')
    p.add_argument('--lambda_organ',       type=float, default=None)
    p.add_argument('--lambda_hu_profile',  type=float, default=None)
    p.add_argument('--lambda_l1_floor',    type=float, default=None)
    p.add_argument('--l1_decay_start_epoch', type=int, default=None)
    p.add_argument('--l1_decay_end_epoch',   type=int, default=None)
    p.add_argument('--adv_warmup_epochs',  type=int,   default=None)
    p.add_argument('--lr_disc',            type=float, default=None)

    # Loss flags: --use_X / --no_X
    for flag in ['adversarial', 'perceptual', 'feature_matching',
                 'ssim', 'gradient', 'frequency',
                 'organ', 'saliency', 'cycle', 'seg_consistency',
                 'l1_decay', 'per_organ_weights', 'hu_profile']:
        g = p.add_mutually_exclusive_group()
        g.add_argument(f'--use_{flag}',  dest=f'use_{flag}', action='store_true', default=None)
        g.add_argument(f'--no_{flag}',   dest=f'use_{flag}', action='store_false')
    return p.parse_args()


def _apply(cfg: dict, args) -> dict:
    c = cfg.copy()
    if args.device      is not None: c['device']       = args.device
    if args.epochs      is not None: c['epochs']       = args.epochs
    if args.batch_size  is not None: c['batch_size']   = args.batch_size
    if args.lr          is not None: c['learning_rate']= args.lr
    if args.output_dir  is not None: c['output_dir']   = Path(args.output_dir)
    if args.patch_size  is not None: c['patch_size']   = args.patch_size
    if args.patch_depth is not None: c['patch_depth']  = args.patch_depth
    if args.dims        is not None: c['dims']         = args.dims
    if args.perceptual_backbone is not None: c['perceptual_backbone'] = args.perceptual_backbone
    if args.saliency_mode       is not None: c['saliency_mode']       = args.saliency_mode
    for k in ['selection_metric', 'sample_mode', 'sample_n',
              'lambda_organ', 'lambda_hu_profile', 'lambda_l1_floor',
              'l1_decay_start_epoch', 'l1_decay_end_epoch',
              'adv_warmup_epochs', 'lr_disc']:
        v = getattr(args, k, None)
        if v is not None:
            c[k] = v
    for flag in ['adversarial', 'perceptual', 'feature_matching',
                 'ssim', 'gradient', 'frequency',
                 'organ', 'saliency', 'cycle', 'seg_consistency',
                 'l1_decay', 'per_organ_weights', 'hu_profile']:
        v = getattr(args, f'use_{flag}', None)
        if v is not None:
            c[f'use_{flag}'] = v

    # Per-organ weights are resolved from organ NAMES against the TS label map,
    # so the LUT has to be rebuilt whenever the CLI toggles the flag.
    if args.use_per_organ_weights is not None or args.organ_weight_preset is not None:
        c['organ_weight_preset'] = args.organ_weight_preset or c.get('organ_weight_preset')
        c['organ_weights'] = resolve_organ_weights(
            enabled = args.use_per_organ_weights,
            preset  = args.organ_weight_preset,
        )

    # Auto-derive dims from patch_depth if not explicitly given
    if args.dims is None and 'patch_depth' in c:
        c['dims'] = 3 if c['patch_depth'] > 1 else 2

    return c


def _ckpt_epoch(p: Path) -> int:
    m = re.search(r'ckpt_ep(\d+)', p.name)
    return int(m.group(1)) if m else 0


def _resumable_ckpts(out: Path):
    """Checkpoints newest-first. Sorted by parsed epoch number, not filename, so
    ckpt_ep100 doesn't sort before ckpt_ep99."""
    return sorted(out.glob('ckpt_ep*.pth'), key=_ckpt_epoch, reverse=True)


def _latest_ckpt(out: Path):
    ckpts = _resumable_ckpts(out)
    if not ckpts:
        return None, 0
    return ckpts[0], _ckpt_epoch(ckpts[0])


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

    # If we asked for CUDA but it isn't usable, fall back to CPU rather than
    # crashing — and warn loudly, since this is exactly the "silently on CPU"
    # trap that makes an epoch take ~45 min instead of seconds.
    if str(config['device']).startswith('cuda') and not torch.cuda.is_available():
        log.warning("Requested CUDA but torch.cuda.is_available() is False — "
                    "falling back to CPU.")
        config['device'] = 'cpu'
    if config['device'] == 'cpu':
        log.warning('=' * 65)
        log.warning("RUNNING ON CPU — training will be VERY slow.")
        log.warning(f"torch {torch.__version__} | cuda build: {torch.version.cuda} "
                    f"| cuda available: {torch.cuda.is_available()} "
                    f"| visible GPUs: {torch.cuda.device_count()}")
        if torch.version.cuda is None:
            log.warning("This is a CPU-only torch build (torch.version.cuda is None).")
            log.warning("Install a CUDA build in the env, e.g.:")
            log.warning("  pip install --force-reinstall torch "
                        "--index-url https://download.pytorch.org/whl/cu121")
        else:
            log.warning("torch has CUDA support but no GPU is visible — check "
                        "`nvidia-smi`, drivers, and CUDA_VISIBLE_DEVICES.")
        log.warning("Then re-run with --resume (default) to continue from the "
                    "last saved epoch on the GPU.")
        log.warning('=' * 65)
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
        'organ', 'hu_profile', 'saliency', 'cycle', 'seg_consistency',
    ] if config.get(f'use_{f}')]
    log.info(f"Losses     : {' + '.join(active)}")
    log.info(f"Selection  : {config.get('selection_metric', 'val_org_ssim')}")
    if config.get('use_l1_decay'):
        log.info(f"L1 decay   : {config['lambda_l1']} → {config['lambda_l1_floor']} "
                 f"over epochs {config['l1_decay_start_epoch']}–{config['l1_decay_end_epoch']}")
    ow = config.get('organ_weights')
    if ow:
        zeroed = sorted(k for k, v in ow.items() if v == 0.0)
        log.info(f"Organ wts  : per-organ LUT over {len(ow)} labels "
                 f"({len(zeroed)} zero-weighted: {zeroed})")
    elif config.get('use_organ'):
        log.info(f"Organ wts  : uniform {config.get('organ_weight')}× on masked voxels")
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
    ckpt, _ = _latest_ckpt(out)
    if args.fresh:
        if ckpt:
            log.info(f"--fresh: ignoring existing checkpoint {ckpt.name}, "
                     f"starting from scratch")
    elif ckpt:
        # A checkpoint truncated by an interrupted write (or a full disk) raises
        # from torch.load — typically "PytorchStreamReader failed locating file
        # data/N". Rather than abort the whole run, walk back to the next-newest
        # checkpoint: keep_last_n_checkpoints normally leaves 2 older ones, so
        # the cost of a corrupt tail is a few epochs, not the entire scenario.
        start, used = 0, None
        for cand in _resumable_ckpts(out):
            try:
                start = trainer.load_checkpoint(str(cand))
                used = cand
                break
            except Exception as e:
                bad = cand.with_suffix('.pth.corrupt')
                log.warning(f"checkpoint {cand.name} is unreadable ({type(e).__name__}: {e}); "
                            f"renaming to {bad.name} and trying the next-newest")
                try:
                    cand.rename(bad)
                except OSError:
                    pass
        if used is None:
            log.warning("no readable checkpoint in the output dir — starting fresh. "
                        "(Any *.pth.corrupt files are the unreadable ones; a truncated "
                        "write usually means the job was killed mid-save or the disk filled.)")
        else:
            if start >= config['epochs']:
                log.info(f"Checkpoint epoch {start} ≥ target {config['epochs']} epochs — "
                         f"nothing left to train. Use --epochs to extend or --fresh to restart.")
                return
            log.info(f"Auto-resuming from {used.name}: will train epochs "
                     f"{start + 1} → {config['epochs']}")
    else:
        log.info("No checkpoint in output dir — starting fresh.")

    try:
        trainer.train(train_loader, val_loader, config['epochs'], start_epoch=start)
    except KeyboardInterrupt:
        log.info("Interrupted — saving emergency checkpoint …")
        trainer._save_checkpoint(trainer.current_epoch, is_best=False)

    log.info("Done.")


if __name__ == '__main__':
    main()
