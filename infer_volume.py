"""
Volume-level inference for the NCCT→CECT generator + phase-fidelity manifest.

Phase 1 of NCCT2CECT_PLAN.md: pixel metrics (MAE/PSNR/SSIM/NCC) reward L1 blur and
can't tell whether a generator produced the RIGHT contrast phase. To answer that we
must reconstruct a FULL synthetic CECT volume from a trained generator and score it
with the 97%-OOF CTPhase-XGBoost model (via CTPhase-XGBoost/phase_eval.py).

The generator only ever saw patches, so reconstruction goes through the SAME patch
pipeline it was trained on (dataset.py geometry + HU normalisation) — no whole-slice
or organ-crop shortcuts:
  • tile the NCCT volume into patches (patch_size / overlap / patch_depth from the
    run's own run_config.json),
  • normalise each patch exactly as dataset.py does (clip [hu_min,hu_max] → [0,1]),
  • run G (batched), stitch back with uniform overlap-averaging,
  • de-normalise [0,1] → HU, save as NIfTI on the source grid.

Dims-parametric: `patch_depth==1` → 2-D per-slice; `patch_depth>1` → 3-D sub-volume.
The 3-D path is exercised by the smoke test so the next (3-D patch) training phase
reuses this same code rather than a 2-D-only one-off.

Outputs per scenario:
  <out_dir>/<case_id>_syn.nii.gz   synthetic CECT (HU) for each held-out case
  <out_dir>/manifest.csv           gen_path,real_path,mask_path,target_phase
                                   → feed straight to phase_eval.py --gen_in_hu

Usage:
    python infer_volume.py --scenario_dir ../../simlified_train/literature_baseline_l1_only \
        --split test --out_dir <scenario_dir>/phase_infer
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import torch

from dataset import _load_vol, find_pairs_and_split
from models import UNetGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generator loading (no full Trainer — avoids building D / CompositeLoss/VGG)
# ---------------------------------------------------------------------------

def load_generator(ckpt_path: str, cfg: Dict, device: str) -> UNetGenerator:
    G = UNetGenerator(
        dims          = cfg.get('dims', 2),
        base_channels = cfg['generator_base_channels'],
        dropout       = cfg.get('generator_dropout', 0.2),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(state['G_state'])
    G.eval()
    log.info(f"Loaded generator from {ckpt_path} (epoch {state.get('epoch', '?')})")
    return G


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------

def _starts(length: int, patch: int, stride: int) -> List[int]:
    """Tile-start indices covering [0, length) with the given patch/stride,
    always including a final flush-to-edge patch so the whole extent is covered."""
    if length <= patch:
        return [0]
    s = list(range(0, length - patch + 1, stride))
    if s[-1] != length - patch:
        s.append(length - patch)
    return s


@torch.no_grad()
def infer_volume(G: UNetGenerator, vol_dhw: np.ndarray, cfg: Dict, device: str,
                 batch_size: int = 32) -> np.ndarray:
    """Reconstruct a full synthetic CECT volume (HU, (D,H,W)) from an NCCT volume.

    Uniform overlap-averaging. Patches are normalised/denormalised with the run's
    own HU window so the generator sees exactly the training input distribution.
    """
    dims = cfg.get('dims', 2)
    pd = int(cfg.get('patch_depth', 1))
    ps = cfg['patch_size']
    ph, pw = (int(ps), int(ps)) if isinstance(ps, int) else (int(ps[0]), int(ps[1]))
    hu_min = float(cfg.get('hu_min', -200)); hu_max = float(cfg.get('hu_max', 400))
    ratio = 1.0 - cfg.get('overlap', 0.5)
    sh = max(1, int(ph * ratio)); sw = max(1, int(pw * ratio))
    sd = max(1, int(pd * ratio)) if pd > 1 else 1
    use_amp = (device == 'cuda')

    D, H, W = vol_dhw.shape
    # Pad tiled dims up to patch size if a volume is smaller (rare; 3-D depth).
    pad_d = max(0, pd - D) if pd > 1 else 0
    pad_h = max(0, ph - H); pad_w = max(0, pw - W)
    if pad_d or pad_h or pad_w:
        vol_dhw = np.pad(vol_dhw, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='edge')
    Dp, Hp, Wp = vol_dhw.shape

    # Normalise once (elementwise → identical to per-patch clip+rescale).
    vn = np.clip(vol_dhw, hu_min, hu_max)
    vn = ((vn - hu_min) / (hu_max - hu_min)).astype(np.float32)

    out_sum = np.zeros((Dp, Hp, Wp), np.float32)
    out_cnt = np.zeros((Dp, Hp, Wp), np.float32)

    d_starts = _starts(Dp, pd, sd) if pd > 1 else list(range(Dp))
    h_starts = _starts(Hp, ph, sh)
    w_starts = _starts(Wp, pw, sw)

    buf_patch, buf_coord = [], []

    def _flush():
        if not buf_patch:
            return
        batch = torch.from_numpy(np.stack(buf_patch)).unsqueeze(1).to(device)  # (B,1,...)
        with torch.autocast('cuda', enabled=use_amp):
            out = G(batch)
        out = out.squeeze(1).float().cpu().numpy()   # (B, ...) in [0,1]
        for (d0, y0, x0), o in zip(buf_coord, out):
            if pd > 1:
                out_sum[d0:d0+pd, y0:y0+ph, x0:x0+pw] += o
                out_cnt[d0:d0+pd, y0:y0+ph, x0:x0+pw] += 1.0
            else:
                out_sum[d0, y0:y0+ph, x0:x0+pw] += o
                out_cnt[d0, y0:y0+ph, x0:x0+pw] += 1.0
        buf_patch.clear(); buf_coord.clear()

    for d0 in d_starts:
        for y0 in h_starts:
            for x0 in w_starts:
                if pd > 1:
                    patch = vn[d0:d0+pd, y0:y0+ph, x0:x0+pw]     # (pd,ph,pw)
                else:
                    patch = vn[d0, y0:y0+ph, x0:x0+pw]           # (ph,pw)
                buf_patch.append(patch); buf_coord.append((d0, y0, x0))
                if len(buf_patch) >= batch_size:
                    _flush()
    _flush()

    out_cnt[out_cnt == 0] = 1.0                       # guard (shouldn't happen)
    syn01 = out_sum / out_cnt
    syn_hu = syn01 * (hu_max - hu_min) + hu_min
    return syn_hu[:D, :H, :W]                          # crop away any padding


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(scenario_dir: str, split: str, out_dir: str, ckpt_name: str,
        batch_size: int, device: str):
    sdir = Path(scenario_dir)
    cfg = json.loads((sdir / 'run_config.json').read_text())
    ckpt = sdir / ckpt_name
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    G = load_generator(str(ckpt), cfg, device)

    train_pairs, val_pairs, test_pairs = find_pairs_and_split(cfg)
    pairs = {'val': val_pairs, 'test': test_pairs,
             'both': val_pairs + test_pairs}[split]
    log.info(f"Inferring {len(pairs)} {split} case(s) for scenario '{sdir.name}'")

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    target_phase = cfg.get('target_phase', 'venous')
    rows = []
    n_no_mask = 0
    for pair in pairs:
        case_id = pair['case_id']
        seg_path = pair.get('seg_path')
        if not seg_path:
            n_no_mask += 1
            log.warning(f"  {case_id}: no {cfg.get('seg_suffix','_seg_reg')} mask — skipping (can't phase-score)")
            continue
        src_nii = nib.load(pair['source_path'])
        src_dhw = _load_vol(pair['source_path'])
        syn_hu_dhw = infer_volume(G, src_dhw, cfg, device, batch_size=batch_size)
        # back to native (X,Y,Z) orientation to match real/mask that phase_eval
        # loads raw; save on the source grid/affine.
        syn_xyz = np.transpose(syn_hu_dhw, (2, 1, 0)).astype(np.float32)
        gen_path = out / f'{case_id}_syn.nii.gz'
        nib.save(nib.Nifti1Image(syn_xyz, src_nii.affine, src_nii.header), str(gen_path))
        rows.append({'gen_path': str(gen_path), 'real_path': pair['target_path'],
                     'mask_path': seg_path, 'target_phase': target_phase})
        log.info(f"  {case_id}: saved {gen_path.name}")

    manifest = out / 'manifest.csv'
    with open(manifest, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['gen_path', 'real_path', 'mask_path', 'target_phase'])
        w.writeheader(); w.writerows(rows)
    log.info(f"Wrote {len(rows)} rows → {manifest}"
             + (f"  ({n_no_mask} skipped, no mask)" if n_no_mask else ""))
    log.info(f"Next: python <CTPhase-XGBoost>/phase_eval.py --weights <xgb_vindr_full.pkl> "
             f"--manifest {manifest} --gen_in_hu --hu_min {cfg.get('hu_min',-200)} "
             f"--hu_max {cfg.get('hu_max',400)} --out_json {out/'phase_eval_report.json'}")


def main():
    ap = argparse.ArgumentParser(description='Volume inference + phase-fidelity manifest')
    ap.add_argument('--scenario_dir', required=True, help='a trained scenario output dir (has run_config.json + best_model.pth)')
    ap.add_argument('--split', default='test', choices=['val', 'test', 'both'])
    ap.add_argument('--out_dir', default=None, help='default: <scenario_dir>/phase_infer')
    ap.add_argument('--ckpt_name', default='best_model.pth')
    ap.add_argument('--batch_size', type=int, default=32)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = args.out_dir or str(Path(args.scenario_dir) / 'phase_infer')
    run(args.scenario_dir, args.split, out_dir, args.ckpt_name, args.batch_size, device)


if __name__ == '__main__':
    main()
