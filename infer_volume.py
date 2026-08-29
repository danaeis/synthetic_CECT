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
  • run G (batched), stitch back with weighted overlap-blending (--blend),
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
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import torch

from dataset import _load_vol, find_pairs_and_split, standardised_level
from models import UNetGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generator loading (no full Trainer — avoids building D / CompositeLoss/VGG)
# ---------------------------------------------------------------------------

def load_generator(ckpt_path: str, cfg: Dict, device: str) -> UNetGenerator:
    _phases = cfg.get('target_phases') or [cfg.get('target_phase', 'venous')]
    G = UNetGenerator(
        dims          = cfg.get('dims', 2),
        base_channels = cfg['generator_base_channels'],
        dropout       = cfg.get('generator_dropout', 0.2),
        norm          = cfg.get('generator_norm', 'instance'),
        in_channels   = cfg.get('in_channels', 1),
        use_phase_cond= cfg.get('use_phase_cond', False),
        n_phases      = max(2, len(_phases)),
        cond_dim      = cfg.get('phase_cond_dim', 64),
        n_levels      = len(cfg.get('cond_organs') or []),
    ).to(device)
    # weights_only=False: checkpoints also carry non-tensor state (history,
    # early_stop, ...); we only read G_state but must be able to unpickle them.
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
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


def _axis_window(n: int, mode: str, margin: int,
                 at_lo: bool, at_hi: bool) -> np.ndarray:
    """1-D blend weight along one tile axis.

    `margin` discards the outer band of the tile, where every output voxel was
    computed partly from the zero-padding the convolutions invented outside the
    tile (padding=1 nested 9 deep in UNetGenerator ⇒ a systematically corrupted
    band ~20-30 px wide). The margin is NOT applied on a side that lies on the
    volume boundary — there is no neighbouring tile to supply those voxels, so
    trimming them would leave the volume's own outer shell uncovered.
    """
    if mode == 'uniform':
        w = np.ones(n, np.float32)
    elif mode == 'hann':
        # n+2 then trim: a plain Hann is exactly 0 at both ends, which would make
        # the outermost row of every tile weightless and, on a border tile,
        # unrecoverable.
        w = np.hanning(n + 2)[1:-1].astype(np.float32)
    elif mode == 'gaussian':
        x = (np.arange(n, dtype=np.float32) - (n - 1) / 2) / max(1.0, (n - 1) / 2)
        w = np.exp(-0.5 * (x / 0.35) ** 2).astype(np.float32)
    else:
        raise ValueError(f'unknown blend mode: {mode}')

    if margin > 0 and n > 2 * margin:
        if not at_lo:
            w[:margin] = 0.0
        if not at_hi:
            w[-margin:] = 0.0
    return w


@torch.no_grad()
def infer_volume(G: UNetGenerator, vol_dhw: np.ndarray, cfg: Dict, device: str,
                 batch_size: int = 32, blend: str = 'hann',
                 edge_margin: int = 0, overlap: Optional[float] = None,
                 phase_id: Optional[int] = None,
                 level: Optional[np.ndarray] = None) -> np.ndarray:
    """Reconstruct a full synthetic CECT volume (HU, (D,H,W)) from an NCCT volume.

    Weighted overlap-blending. Patches are normalised/denormalised with the run's
    own HU window so the generator sees exactly the training input distribution.

    `blend='uniform'` reproduces the original flat average, whose blend weight
    jumps discontinuously wherever a tile starts — measured at seam=1.27 against
    a seam=1.04 floor from a real untouched scan (`metrics.seam_energy`). A
    tapered window removes that discontinuity.

    NOTE this cannot fix a seam caused by the generator itself: InstanceNorm
    rescales each tile by that tile's own statistics, so overlapping tiles
    disagree by a content-dependent DC offset that no blending weight can
    reconcile. Blending is necessary, not sufficient; see the norm ablation.

    `overlap` overrides the training-time value from the config — inference
    overlap is free to be higher than what training used.
    """
    pd = int(cfg.get('patch_depth', 1))
    # 2.5-D: gather 2k+1 adjacent slices as channels, still emit one slice. Output
    # depth is unchanged, so all the blending/windowing below is untouched.
    n_in = int(cfg.get('n_input_slices', 1))
    k_sl = (n_in - 1) // 2
    ps = cfg['patch_size']
    ph, pw = (int(ps), int(ps)) if isinstance(ps, int) else (int(ps[0]), int(ps[1]))
    hu_min = float(cfg.get('hu_min', -200)); hu_max = float(cfg.get('hu_max', 400))
    ov = cfg.get('overlap', 0.5) if overlap is None else float(overlap)
    ratio = 1.0 - ov
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

    # Blend windows depend only on which volume edges a tile touches, so there are
    # at most a handful of distinct ones — build each once and reuse.
    _wcache: Dict = {}

    def _window(d0, y0, x0):
        key = (d0 == 0, d0 + pd >= Dp, y0 == 0, y0 + ph >= Hp, x0 == 0, x0 + pw >= Wp)
        if key not in _wcache:
            wh = _axis_window(ph, blend, edge_margin, key[2], key[3])
            ww = _axis_window(pw, blend, edge_margin, key[4], key[5])
            if pd > 1:
                wd = _axis_window(pd, blend, edge_margin, key[0], key[1])
                w = wd[:, None, None] * wh[None, :, None] * ww[None, None, :]
            else:
                w = wh[:, None] * ww[None, :]
            _wcache[key] = w.astype(np.float32)
        return _wcache[key]

    buf_patch, buf_coord = [], []

    def _flush():
        if not buf_patch:
            return
        arr = np.stack(buf_patch)
        # 2.5-D patches already carry the slice stack in the channel axis.
        batch = torch.from_numpy(arr).to(device)
        if k_sl == 0:
            batch = batch.unsqueeze(1)                                  # (B,1,...)
        # Phase is per-tile and constant across the volume, so it is broadcast to
        # the batch here rather than carried through the tiling bookkeeping.
        ph_t = (torch.full((batch.shape[0],), int(phase_id),
                           dtype=torch.long, device=device)
                if getattr(G, 'use_phase_cond', False) else None)
        # Level is constant across the volume, so it is broadcast here rather
        # than carried through the tiling bookkeeping.
        lv_t = (torch.as_tensor(level, dtype=torch.float32, device=device)
                .reshape(1, -1).expand(batch.shape[0], -1)
                if getattr(G, 'n_levels', 0) else None)
        # Only pass conditioning that actually exists, so infer_volume still
        # works with any plain generator whose forward is just forward(x).
        gkw = {}
        if ph_t is not None:
            gkw['phase'] = ph_t
        if lv_t is not None:
            gkw['level'] = lv_t
        with torch.autocast('cuda', enabled=use_amp):
            out = G(batch, **gkw)
        out = out.squeeze(1).float().cpu().numpy()   # (B, ...) in [0,1]
        for (d0, y0, x0), o in zip(buf_coord, out):
            w = _window(d0, y0, x0)
            if pd > 1:
                out_sum[d0:d0+pd, y0:y0+ph, x0:x0+pw] += o * w
                out_cnt[d0:d0+pd, y0:y0+ph, x0:x0+pw] += w
            else:
                out_sum[d0, y0:y0+ph, x0:x0+pw] += o * w
                out_cnt[d0, y0:y0+ph, x0:x0+pw] += w
        buf_patch.clear(); buf_coord.clear()

    for d0 in d_starts:
        for y0 in h_starts:
            for x0 in w_starts:
                if pd > 1:
                    patch = vn[d0:d0+pd, y0:y0+ph, x0:x0+pw]     # (pd,ph,pw)
                elif k_sl:
                    # Edge-clamped, matching dataset.py's _crop_src exactly, so
                    # the model sees the same boundary behaviour it trained on.
                    zs = np.clip(np.arange(d0 - k_sl, d0 + k_sl + 1), 0, Dp - 1)
                    patch = vn[zs, y0:y0+ph, x0:x0+pw]           # (2k+1,ph,pw)
                else:
                    patch = vn[d0, y0:y0+ph, x0:x0+pw]           # (ph,pw)
                buf_patch.append(patch); buf_coord.append((d0, y0, x0))
                if len(buf_patch) >= batch_size:
                    _flush()
    _flush()

    # With edge-aware margins every voxel keeps positive weight, so a zero here
    # means a real coverage bug rather than a benign edge case — say so instead of
    # silently emitting hu_min.
    n_zero = int((out_cnt <= 0).sum())
    if n_zero:
        log.error(f"  {n_zero} voxel(s) received no tile weight "
                  f"(blend={blend}, edge_margin={edge_margin}) — output invalid there; "
                  f"reduce --edge_margin or raise --overlap")
        out_cnt[out_cnt <= 0] = 1.0
    syn01 = out_sum / out_cnt
    syn_hu = syn01 * (hu_max - hu_min) + hu_min
    return syn_hu[:D, :H, :W]                          # crop away any padding


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(scenario_dir: str, split: str, out_dir: str, ckpt_name: str,
        batch_size: int, device: str, blend: str = 'hann',
        edge_margin: int = 0, overlap: Optional[float] = None,
        level_mode: str = 'oracle', level: Optional[float] = None,
        level_z: Optional[float] = None):
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

    # ── level conditioning at inference ──────────────────────────────────────
    # Same checkpoint, three modes. The REPORTABLE RESULT is the gap between
    # 'oracle' and 'population' — oracle reads the answer off the real CECT and
    # can never be a headline number on its own.
    #
    #   oracle      the case's true standardised level      -> the ceiling
    #   population  all zeros == the training-set mean      -> must match baseline
    #   fixed       --level L, standardised, for every case -> the deployable mode
    #   fixed_z     --level_z Z standard deviations         -> the sweep mode
    #
    # WHY fixed_z EXISTS. `fixed` applies ONE HU number to EVERY conditioned
    # organ, then standardises each by its own organ's stats. That is fine for a
    # single-organ model (--cond_organs aorta), but for the 8-organ model those
    # organs occupy wildly different HU ranges: with --level 350, aorta lands at
    # a sensible z=+1.4 while gallbladder (train mean 29.2, sd 12.4) lands at
    # z=+25.9 — an enhancement the model has never seen and cannot render.
    # Sweeping a calibration curve that way measures the model's behaviour far
    # outside its training distribution, not its controllability. fixed_z moves
    # every organ by the SAME number of standard deviations instead, so each one
    # stays in its own plausible range and the sweep tests what it means to.
    cond_organs = list(cfg.get('cond_organs') or [])
    levels_db = None
    if cond_organs:
        if level_mode not in ('oracle', 'population', 'population_phase',
                              'fixed', 'fixed_z'):
            raise ValueError(f"unknown level_mode {level_mode!r}")
        lp = Path(cfg.get('levels_json', 'splits/levels.json'))
        if level_mode in ('oracle', 'population_phase') and not lp.exists():
            raise FileNotFoundError(f"level_mode={level_mode} needs {lp}")
        if lp.exists():
            levels_db = json.loads(lp.read_text())
        log.info(f"level conditioning: {len(cond_organs)} organ(s), mode={level_mode}"
                 + (f", level={level}" if level_mode == 'fixed' else "")
                 + (f", level_z={level_z}" if level_mode == 'fixed_z' else ""))
        if level_mode == 'fixed' and len(cond_organs) > 1:
            st = levels_db['standardize']
            idx = [levels_db['organs'].index(o) for o in cond_organs]
            zs = [(level - st['mean'][i]) / (st['std'][i] or 1.0) for i in idx]
            worst = max(range(len(zs)), key=lambda k: abs(zs[k]))
            if abs(zs[worst]) > 4:
                log.warning(
                    f"--level {level} puts '{cond_organs[worst]}' at z={zs[worst]:+.1f} "
                    f"(train mean {st['mean'][idx[worst]]:.1f}, sd {st['std'][idx[worst]]:.1f}). "
                    f"One HU value cannot suit {len(cond_organs)} organs with different "
                    f"ranges — use --level_mode fixed_z --level_z Z for a calibration sweep.")

    # Per-phase training mean, expressed in the POOLED standardised units the
    # model was trained with. Cached because it is identical for every case.
    #
    # Why this mode exists. dump_levels.py standardises over all phases pooled,
    # so for a multiphase run `standardize.mean` is the average of venous and
    # arterial. The aorta's pooled mean is 207 HU — the venous median is ~145 and
    # the arterial ~294, so z=0 ('population') asks for a level that belongs to
    # NEITHER phase. A model that faithfully obeys its level input then emits a
    # phase-inappropriate volume and scores terribly on phase fidelity, while a
    # model that ignores the input scores well; the metric rewards exactly the
    # wrong thing. This asks each phase for ITS OWN training mean instead.
    #
    # Leak-free: it reads train-split statistics only, never the test case.
    _phase_z_cache: Dict[str, np.ndarray] = {}

    def _phase_population_z(phase: str) -> np.ndarray:
        n = len(cond_organs)
        if phase in _phase_z_cache:
            return _phase_z_cache[phase]
        st = levels_db['standardize']
        idx = [levels_db['organs'].index(o) for o in cond_organs]
        z = np.zeros(n, np.float32)
        for j, o in enumerate(cond_organs):
            vals = [rec[phase]['cect'].get(o) for rec in levels_db['cases'].values()
                    if rec.get('split') == 'train' and phase in rec]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            if not vals:
                continue                      # 0.0 == pooled mean, the old behaviour
            mu, sd = st['mean'][idx[j]], (st['std'][idx[j]] or 1.0)
            z[j] = (float(np.mean(vals)) - mu) / sd
        _phase_z_cache[phase] = z
        log.info(f"  population_phase[{phase}]: z = "
                 + ", ".join(f"{o}={v:+.2f}" for o, v in zip(cond_organs, z)))
        return z

    def _level_for(case_id: str, phase: str):
        if not cond_organs:
            return None
        n = len(cond_organs)
        if level_mode == 'population':
            return np.zeros(n, np.float32)      # standardised mean (POOLED)
        if level_mode == 'population_phase':
            return _phase_population_z(phase)
        if level_mode == 'fixed_z':
            # Every organ shifted by the same number of its own SDs: z=0 is
            # exactly `population`, z=+1 is "one sd more enhanced than average".
            return np.full(n, float(level_z), np.float32)
        if level_mode == 'fixed':
            st = levels_db['standardize']
            idx = [levels_db['organs'].index(o) for o in cond_organs]
            return np.array([(level - st['mean'][i]) / (st['std'][i] or 1.0)
                             for i in idx], np.float32)
        idx = [levels_db['organs'].index(o) for o in cond_organs]
        st = levels_db['standardize']
        return np.array(standardised_level(
            levels_db, case_id, phase, cond_organs, idx,
            [st['mean'][i] for i in idx],
            [st['std'][i] or 1.0 for i in idx]), np.float32)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    default_phase = cfg.get('target_phase', 'venous')
    # One manifest per phase. A multi-phase model emits several volumes per case,
    # which would collide on '{case_id}_syn.nii.gz' in a single directory, and
    # each phase has to be phase-scored against its own target anyway. Runs with
    # one phase keep exactly the previous flat layout.
    phases = sorted({p.get('phase', default_phase) for p in pairs})
    multi = len(phases) > 1
    rows_by_phase = {ph: [] for ph in phases}
    n_no_mask = 0

    for pair in pairs:
        case_id = pair['case_id']
        seg_path = pair.get('seg_path')
        ph = pair.get('phase', default_phase)
        if not seg_path:
            n_no_mask += 1
            log.warning(f"  {case_id} [{ph}]: no {cfg.get('seg_suffix','_seg_reg')} mask — skipping (can't phase-score)")
            continue
        pdir = (out / ph) if multi else out
        pdir.mkdir(parents=True, exist_ok=True)
        src_nii = nib.load(pair['source_path'])
        src_dhw = _load_vol(pair['source_path'])
        syn_hu_dhw = infer_volume(G, src_dhw, cfg, device, batch_size=batch_size,
                                  blend=blend, edge_margin=edge_margin, overlap=overlap,
                                  phase_id=pair.get('phase_id', 0),
                                  level=_level_for(case_id, ph))
        # back to native (X,Y,Z) orientation to match real/mask that phase_eval
        # loads raw; save on the source grid/affine.
        syn_xyz = np.transpose(syn_hu_dhw, (2, 1, 0)).astype(np.float32)
        gen_path = pdir / f'{case_id}_syn.nii.gz'
        nib.save(nib.Nifti1Image(syn_xyz, src_nii.affine, src_nii.header), str(gen_path))
        rows_by_phase[ph].append({'gen_path': str(gen_path),
                                  'real_path': pair['target_path'],
                                  'mask_path': seg_path, 'target_phase': ph})
        log.info(f"  {case_id} [{ph}]: saved {gen_path.name}")

    for ph, rows in rows_by_phase.items():
        pdir = (out / ph) if multi else out
        manifest = pdir / 'manifest.csv'
        with open(manifest, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['gen_path', 'real_path', 'mask_path', 'target_phase'])
            w.writeheader(); w.writerows(rows)
        log.info(f"Wrote {len(rows)} rows → {manifest}"
                 + (f"  ({n_no_mask} skipped, no mask)" if n_no_mask else ""))
        log.info(f"Next: python <CTPhase-XGBoost>/phase_eval.py --weights <xgb_vindr_full.pkl> "
                 f"--manifest {manifest} --gen_in_hu --hu_min {cfg.get('hu_min',-200)} "
                 f"--hu_max {cfg.get('hu_max',400)} --out_json {pdir/'phase_eval_report.json'}")


def main():
    ap = argparse.ArgumentParser(description='Volume inference + phase-fidelity manifest')
    ap.add_argument('--scenario_dir', required=True, help='a trained scenario output dir (has run_config.json + best_model.pth)')
    ap.add_argument('--split', default='test', choices=['val', 'test', 'both'])
    ap.add_argument('--out_dir', default=None, help='default: <scenario_dir>/phase_infer')
    ap.add_argument('--ckpt_name', default='best_model.pth')
    ap.add_argument('--level_mode', default='oracle',
                    choices=['oracle', 'population', 'population_phase', 'fixed', 'fixed_z'],
                    help="oracle = the case's true level (the CEILING, leaks the "
                         "target); population = training mean (must reproduce the "
                         "unconditioned baseline, but POOLED across phases); population_phase = "
                         "each phase's own training mean (use this for a multiphase run); "
                         "fixed = --level HU for every case; "
                         "fixed_z = --level_z standard deviations for every case. "
                         "Report the oracle-vs-population GAP, never oracle alone.")
    ap.add_argument('--level', type=float, default=None,
                    help='HU value for --level_mode fixed. One HU number is applied '
                         'to EVERY conditioned organ, so this only makes sense for a '
                         'single-organ model — use fixed_z for a multi-organ sweep.')
    ap.add_argument('--level_z', type=float, default=None,
                    help='standard deviations from the training mean, for '
                         '--level_mode fixed_z. Every conditioned organ is shifted by '
                         'this many of ITS OWN sds, so all stay in range. z=0 is '
                         'identical to --level_mode population; try -1.5 -1 0 1 1.5 '
                         'for a calibration sweep.')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--blend', default='hann', choices=['uniform', 'hann', 'gaussian'],
                    help="tile blend window. 'uniform' is the original flat average "
                         "(measured seam=1.27 vs a 1.04 floor); default 'hann' tapers "
                         "the weight so it is continuous across a tile boundary.")
    ap.add_argument('--edge_margin', type=int, default=0,
                    help='discard this many voxels from each tile side before '
                         'blending (the conv zero-padding band). Not applied on '
                         'sides lying on the volume boundary. Needs enough overlap '
                         'to stay covered: margin < patch_size*overlap/2.')
    ap.add_argument('--overlap', type=float, default=None,
                    help='override the training-time overlap for inference only')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = args.out_dir or str(Path(args.scenario_dir) / 'phase_infer')
    run(args.scenario_dir, args.split, out_dir, args.ckpt_name, args.batch_size, device,
        blend=args.blend, edge_margin=args.edge_margin, overlap=args.overlap,
        level_mode=args.level_mode, level=args.level, level_z=args.level_z)


if __name__ == '__main__':
    main()
