#!/usr/bin/env python
"""
Is the patch seam a STITCHING bug or a WEIGHTS bug?

Reconstructed volumes show visible tile borders. `metrics.seam_energy` measures
them (1.27 on an l1 run, against a 1.04 floor from a real untouched scan), but not
their cause, and the two candidate causes need opposite fixes:

  STITCHING — uniform overlap-averaging makes the blend weight jump where a tile
              starts. Fixed at inference, no retraining: `infer_volume --blend hann`.
  WEIGHTS   — InstanceNorm rescales each patch by ITS OWN spatial mean/std, so the
              same voxel in two overlapping tiles receives two different affine
              transforms. The tiles then disagree by a content-dependent DC offset
              that NO blending weight can reconcile. Fixing it requires retraining
              with a different norm.

This script measures the second directly. Take a patch, shift the crop window by a
few voxels, and compare the two predictions ON THE REGION THEY SHARE. Those voxels
have identical ground truth and near-identical local input, so a well-behaved
generator should predict them near-identically. Whatever disagreement remains is
caused purely by the change in surrounding context.

The disagreement is then split:

  DC       = |mean(A - B)|      a constant offset over the shared region.
                                THIS is the InstanceNorm signature, and this is
                                what survives overlap-averaging as a seam.
  residual = mean|A - B - DC|   everything else — ordinary boundary effects,
                                which blending does attenuate.

A high DC fraction means the norm must change (`config.GEN_NORM`). A low one means
blending alone is enough and no retraining is warranted.

With --compare_norms the same measurement runs on group/batch variants so the
choice is made on numbers rather than on the argument above. Those variants are
randomly initialised unless a matching checkpoint exists, so read them as a
comparison of ARCHITECTURES at equal training state — the trained run's own number
is the one reported first, and it is the one that decides.

Usage:
    # trained model from a run (the number that matters)
    python norm_attribution.py --scenario_dir ../out_synthesis_train/literature_baseline_l1_adv_organ

    # architecture comparison at equal (random) init, no checkpoint needed
    python norm_attribution.py --data_dir <dir> --compare_norms
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from dataset import _load_vol, find_pairs_and_split
from models import NORM_KINDS, UNetGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

def _norm01(patch: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    return ((np.clip(patch, hu_min, hu_max) - hu_min) / (hu_max - hu_min)).astype(np.float32)


@torch.no_grad()
def _predict(G, patches: List[np.ndarray], device: str) -> np.ndarray:
    x = torch.from_numpy(np.stack(patches)).unsqueeze(1).to(device)
    return G(x).squeeze(1).float().cpu().numpy()


@torch.no_grad()
def context_shift_drift(G, vols: List[np.ndarray], cfg: Dict, device: str,
                        shifts=(4, 8, 16, 32), n_sites: int = 24,
                        seed: int = 0) -> Dict[int, Dict[str, float]]:
    """Disagreement between two crops of the same voxels, by context shift.

    Returned values are in HU so they are read against the per-organ HU errors the
    rest of the project reports (portal vein ~36 HU on the best run) — a DC drift
    of comparable size is not a rounding detail.
    """
    ps = cfg['patch_size']
    p = int(ps) if isinstance(ps, (int, float)) else int(ps[0])
    hu_min = float(cfg.get('hu_min', -200)); hu_max = float(cfg.get('hu_max', 400))
    hu_range = hu_max - hu_min
    rng = np.random.default_rng(seed)

    # Sample anchors on tissue: an air-only patch is constant, so every model
    # agrees trivially there and the drift would be understated.
    sites = []
    guard = p + max(shifts) + 1
    tries = 0
    while len(sites) < n_sites and tries < n_sites * 200:
        tries += 1
        vi = int(rng.integers(len(vols)))
        D, H, W = vols[vi].shape
        if H < guard or W < guard or D < 1:
            continue
        z = int(rng.integers(0, D))
        y = int(rng.integers(0, H - guard))
        x = int(rng.integers(0, W - guard))
        crop = vols[vi][z, y:y + p, x:x + p]
        if crop.std() < 50.0 or crop.mean() < -600.0:
            continue
        sites.append((vi, z, y, x))
    if not sites:
        raise SystemExit('no tissue-bearing sites found — check the volumes')
    log.info(f'  {len(sites)} sample sites ({tries} draws)')

    # CONTROL — the model's own output variability. Without this the drift number
    # is unreadable: a model that emits a constant image scores a perfect zero
    # drift while being useless. (Not hypothetical — an untrained BatchNorm
    # generator in eval() does exactly this, because eval-mode BN applies
    # running_var=1, which does not match an untrained network's activations, and
    # the output collapses to ~0.25 HU std. It measured 0.02 HU drift and looked
    # 700x better than InstanceNorm.)
    out_stds = []
    for vi, z, y, x in sites:
        a = _norm01(vols[vi][z, y:y + p, x:x + p], hu_min, hu_max)
        out_stds.append(float(_predict(G, [a], device)[0].std()) * hu_range)
    out_std = float(np.mean(out_stds))

    out = {}
    for s in shifts:
        dcs, resids, totals = [], [], []
        for vi, z, y, x in sites:
            a = _norm01(vols[vi][z, y:y + p, x:x + p], hu_min, hu_max)
            b = _norm01(vols[vi][z, y + s:y + s + p, x:x + p], hu_min, hu_max)
            pa, pb = _predict(G, [a, b], device)
            # Shared voxels: rows [y+s, y+p) of the volume.
            #   in A's frame -> [s, p)      in B's frame -> [0, p-s)
            d = pa[s:, :] - pb[:p - s, :]
            dc = float(d.mean())
            dcs.append(abs(dc) * hu_range)
            resids.append(float(np.abs(d - dc).mean()) * hu_range)
            totals.append(float(np.abs(d).mean()) * hu_range)
        out[s] = {
            'dc_hu': float(np.mean(dcs)),
            'residual_hu': float(np.mean(resids)),
            'total_hu': float(np.mean(totals)),
            'dc_frac': float(np.mean(dcs) / max(1e-9, np.mean(totals))),
            # Drift as a fraction of the model's own output variation. This is the
            # comparable number across models; raw HU drift is not.
            'total_rel': float(np.mean(totals) / max(1e-9, out_std)),
            'out_std_hu': out_std,
        }
    return out


DEGENERATE_STD_HU = 5.0    # below this the generator is emitting ~a constant


def _report(name: str, res: Dict[int, Dict[str, float]], lines: List[str]):
    out_std = next(iter(res.values()))['out_std_hu']
    lines.append(f'\n### {name}')
    lines.append(f'\noutput std = {out_std:.2f} HU  (control: drift is only '
                 f'meaningful relative to this)')
    if out_std < DEGENERATE_STD_HU:
        lines.append(f'\n> **DEGENERATE — ignore the drift numbers below.** The '
                     f'generator emits a near-constant image ({out_std:.2f} HU std), '
                     f'which is trivially context-invariant and so scores an '
                     f'artificially perfect drift. Expected for an *untrained* '
                     f'BatchNorm model in eval(): running_var=1 does not match an '
                     f'untrained network\'s activations. Compare trained '
                     f'checkpoints instead.')
    lines.append(f"\n{'shift':>6}{'total HU':>11}{'DC HU':>10}{'resid HU':>11}"
                 f"{'DC frac':>10}{'drift/std':>11}")
    for s, r in sorted(res.items()):
        lines.append(f"{s:>6}{r['total_hu']:>11.2f}{r['dc_hu']:>10.2f}"
                     f"{r['residual_hu']:>11.2f}{r['dc_frac']:>10.1%}"
                     f"{r['total_rel']:>11.1%}")


def _load_volumes(cfg: Dict, n_cases: int) -> List[np.ndarray]:
    train, val, test = find_pairs_and_split(cfg)
    pairs = (test or val or train)[:n_cases]
    vols = []
    for pr in pairs:
        try:
            vols.append(_load_vol(pr['source_path']))       # NCCT input
        except Exception as e:
            log.warning(f"  skip {pr['case_id']}: {e}")
    if not vols:
        raise SystemExit('no volumes loaded')
    log.info(f'Loaded {len(vols)} NCCT volume(s)')
    return vols


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario_dir', default=None,
                    help='trained run dir (run_config.json + checkpoint)')
    ap.add_argument('--ckpt_name', default='best_model.pth')
    ap.add_argument('--data_dir', default=None,
                    help='override cfg data_dir (needed when there is no scenario)')
    ap.add_argument('--labels_csv', default=None, help='override cfg labels_csv')
    ap.add_argument('--volumes', nargs='+', default=None,
                    help='explicit NCCT NIfTI paths, bypassing pair discovery '
                         '(which needs pandas + labels.csv). The measurement only '
                         'ever reads the INPUT volume, so no pairing is required.')
    ap.add_argument('--compare_norms', action='store_true',
                    help='also measure group/batch variants at equal init')
    ap.add_argument('--n_cases', type=int, default=3)
    ap.add_argument('--n_sites', type=int, default=24)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='write the report here as well')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.scenario_dir:
        sdir = Path(args.scenario_dir)
        cfg = json.loads((sdir / 'run_config.json').read_text())
    else:
        import config as C
        cfg = dict(C.train_config)
    if args.data_dir:
        cfg['data_dir'] = args.data_dir
    if args.labels_csv:
        cfg['labels_csv'] = args.labels_csv

    if args.volumes:
        vols = [_load_vol(v) for v in args.volumes]
        log.info(f'Loaded {len(vols)} NCCT volume(s) from --volumes')
    else:
        vols = _load_volumes(cfg, args.n_cases)
    lines = ['# Norm attribution — does the tile seam come from the weights?',
             '',
             'Two crops of the SAME voxels, differing only in surrounding context.',
             'DC = constant offset between the two predictions (the InstanceNorm',
             'signature, and the part that survives overlap-averaging as a seam).',
             'residual = everything else (ordinary boundary effects, which blending',
             'does attenuate). Values in HU.']

    trained_norm = cfg.get('generator_norm', 'instance')
    if args.scenario_dir:
        from infer_volume import load_generator
        ckpt = Path(args.scenario_dir) / args.ckpt_name
        G = load_generator(str(ckpt), cfg, device)
        res = context_shift_drift(G, vols, cfg, device, n_sites=args.n_sites,
                                  seed=args.seed)
        _report(f'TRAINED — {Path(args.scenario_dir).name} (norm={trained_norm})',
                res, lines)
        worst = max(r['dc_hu'] for r in res.values())
        frac = np.mean([r['dc_frac'] for r in res.values()])
        lines.append(
            f'\n**Verdict:** DC drift up to **{worst:.2f} HU** '
            f'({frac:.0%} of the total disagreement). '
            + ('A DC-dominated disagreement is the InstanceNorm signature and cannot '
               'be removed by blending — retrain with `GEN_NORM=\'batch\'` (or '
               '`\'group\'`) to fix the seam at its source.'
               if frac > 0.5 else
               'The disagreement is mostly non-DC, i.e. ordinary boundary effects '
               'that a tapered blend window does attenuate — `--blend hann` should '
               'be sufficient and a norm retrain is not justified by this evidence.'))

    if args.compare_norms:
        lines.append('\n---\n\n## Architecture comparison at equal (random) init')
        lines.append('\nSame measurement, untrained weights, one seed — isolates the '
                     'effect of the norm layer itself from anything training added. '
                     'Read the RELATIVE ordering, not the absolute HU values.')
        lines.append('\n**`batch` is not interpretable here** and is printed only so '
                     'the degeneracy is visible rather than silently flattering: '
                     'eval-mode BatchNorm on untrained weights collapses to a near-'
                     'constant output (see the control line under it). Its '
                     'tile-invariance argument is structural — eval-mode BN applies '
                     'fixed running statistics, so its transform cannot depend on '
                     'the tile — but the NUMBER must come from a trained checkpoint.')
        for kind in NORM_KINDS:
            torch.manual_seed(args.seed)
            G = UNetGenerator(
                dims=cfg.get('dims', 2),
                base_channels=cfg.get('generator_base_channels', 64),
                dropout=0.0,
                norm=kind,
            ).to(device).eval()
            res = context_shift_drift(G, vols, cfg, device, n_sites=args.n_sites,
                                      seed=args.seed)
            _report(f'random init — norm={kind}', res, lines)

    text = '\n'.join(lines)
    print('\n' + text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + '\n')
        log.info(f'wrote {args.out}')


if __name__ == '__main__':
    main()
