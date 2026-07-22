#!/usr/bin/env python
"""
Is the residual synthesis error the MODEL's fault, or the DATA's?

Motivation. Across five scenarios with different losses, per-case
`feature_l1_hu` correlates at r = 0.957-0.983 (mean r^2 = 0.94). Every model
fails on the same cases and succeeds on the same cases, and the between-case
spread (6.2 HU std) is about twice the between-model spread on a fixed case
(3.2 HU). Something about the *case* — not the loss — sets ~94% of the error.

This script tests the leading explanation: NCCT->CECT pairs are two separate
scans, so the "ground truth" is only as good as the registration between them
and as stable as the patient's anatomy in between. Two proxies, both computed
from files already on disk:

  1. REGISTRATION QUALITY — agreement between the NCCT and CECT inside BONE.
     Bone is contrast-invariant: its HU does not change when contrast is
     injected. So any NCCT-vs-CECT disagreement in bone is misregistration (or
     noise), never a real contrast effect. This is the cleanest registration
     proxy available without re-running the registration itself.

  2. ANATOMY CHANGE — how much the bowel gas pattern differs between the two
     scans. Gas moves between acquisitions and is not inferable from the NCCT,
     so this bounds what any model could possibly predict.

If either correlates strongly with per-case error, the pipeline has hit a DATA
ceiling and further loss/architecture work has little headroom — which is a
thesis finding, not a failure. If neither does, the residual is model capacity
and scaling up is justified.

Usage:
    python audit_data_ceiling.py \
        --manifest ../out_synthesis_train/literature_baseline_l1_organ_curriculum/phase_infer/manifest.csv \
        --report   ../out_synthesis_train/literature_baseline_l1_organ_curriculum/phase_infer/phase_eval_report.json
"""

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import nibabel as nib
except ImportError:                                          # pragma: no cover
    raise SystemExit("audit_data_ceiling.py needs nibabel (pip install nibabel)")

# TotalSegmentator label groups. Bone is the contrast-invariant reference.
BONE_PREFIXES = ('vertebrae_', 'rib_', 'hip_', 'femur_', 'scapula_', 'humerus_',
                 'clavicula_', 'sacrum', 'skull', 'sternum')
GI_ORGANS = ('stomach', 'small_bowel', 'duodenum', 'colon')
GAS_HU = -400.0          # below this inside a GI organ is gas, not tissue


def _load(p: str) -> np.ndarray:
    return np.asanyarray(nib.load(p).dataobj).astype(np.float32)


def _label_ids(name_to_id: Dict[str, int], prefixes) -> List[int]:
    return [i for n, i in name_to_id.items() if n.startswith(tuple(prefixes))]


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 100:
        return float('nan')
    a = a - a.mean(); b = b - b.mean()
    d = (a.std() * b.std())
    return float((a * b).mean() / d) if d > 0 else float('nan')


def _find_ncct(real_path: str) -> Optional[str]:
    """The NCCT is the other *_deeds.nii.gz in the same case directory."""
    p = Path(real_path)
    cands = [c for c in sorted(p.parent.glob('*_deeds.nii.gz'))
             if c.name != p.name and 'seg' not in c.name and 'dvf' not in c.name]
    return str(cands[0]) if cands else None


def pearson(a: List[float], b: List[float]) -> float:
    pairs = [(x, y) for x, y in zip(a, b)
             if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 3:
        return float('nan')
    xs, ys = zip(*pairs)
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True,
                    help='phase_eval_report.json for the same run')
    ap.add_argument('--label_map', type=Path,
                    default=Path(__file__).resolve().parent /
                            'orgFeatXGB_CTPhase' / 'retrain_out_full' /
                            'ts_label_map_total.json')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    name_to_id = json.loads(args.label_map.read_text())
    bone_ids = _label_ids(name_to_id, BONE_PREFIXES)
    gi_ids = [name_to_id[o] for o in GI_ORGANS if o in name_to_id]

    rows = list(csv.DictReader(args.manifest.open()))
    report = json.loads(args.report.read_text())
    errs = [c['feature_l1_hu'] for c in report['cases']]
    if len(errs) != len(rows):
        raise SystemExit(f'manifest ({len(rows)}) and report ({len(errs)}) disagree '
                         f'on case count — are they from the same run?')

    print(f'Auditing {len(rows)} cases from {args.manifest.parent.parent.name}\n')
    print(f"{'case':>5}{'bone NCC':>10}{'bone MAE':>10}{'gas Dice':>10}{'err HU':>9}")

    bone_ncc, bone_mae, gas_dice, keep_err = [], [], [], []
    for i, (row, err) in enumerate(zip(rows, errs)):
        ncct_path = _find_ncct(row['real_path'])
        if not ncct_path or not Path(row['mask_path']).exists():
            print(f'{i:5d}  -- missing NCCT or mask, skipped --')
            continue
        try:
            cect = _load(row['real_path'])
            ncct = _load(ncct_path)
            mask = _load(row['mask_path'])
        except Exception as e:
            print(f'{i:5d}  -- load failed: {e} --')
            continue
        if not (cect.shape == ncct.shape == mask.shape):
            print(f'{i:5d}  -- shape mismatch {cect.shape}/{ncct.shape}/{mask.shape} --')
            continue

        lbl = np.round(mask).astype(np.int32)
        bone = np.isin(lbl, bone_ids)
        gi = np.isin(lbl, gi_ids)

        # 1. Registration proxy: bone is contrast-invariant, so disagreement
        #    there is misregistration rather than a real enhancement difference.
        if bone.sum() > 100:
            n = _ncc(ncct[bone], cect[bone])
            m = float(np.abs(ncct[bone] - cect[bone]).mean())
        else:
            n, m = float('nan'), float('nan')

        # 2. Anatomy change: how differently the gas is arranged between scans.
        if gi.sum() > 100:
            g1, g2 = (ncct < GAS_HU) & gi, (cect < GAS_HU) & gi
            inter = float((g1 & g2).sum()); tot = float(g1.sum() + g2.sum())
            d = 2 * inter / tot if tot > 0 else float('nan')
        else:
            d = float('nan')

        bone_ncc.append(n); bone_mae.append(m); gas_dice.append(d); keep_err.append(err)
        print(f'{i:5d}{n:10.3f}{m:10.1f}{d:10.3f}{err:9.1f}')

    if len(keep_err) < 5:
        raise SystemExit('\ntoo few usable cases to correlate')

    print('\n' + '=' * 72)
    print('CORRELATION with per-case synthesis error (feature_l1_hu)')
    print('=' * 72)
    results = [
        ('bone NCC  (higher = better registration)', bone_ncc, 'negative'),
        ('bone MAE  (higher = worse registration)', bone_mae, 'positive'),
        ('gas Dice  (higher = anatomy more stable)', gas_dice, 'negative'),
    ]
    verdict = []
    for label, vals, expect in results:
        r = pearson(vals, keep_err)
        if math.isnan(r):
            print(f'  {label:44s} r = n/a')
            continue
        agrees = (r < 0) if expect == 'negative' else (r > 0)
        print(f'  {label:44s} r = {r:+.3f}   r^2 = {r*r:.3f}'
              f'{"   <-- direction as predicted" if agrees and abs(r) > 0.3 else ""}')
        if agrees and abs(r) > 0.3:
            verdict.append((label.split()[0] + ' ' + label.split()[1], r))

    print('\n' + '-' * 72)
    if verdict:
        best = max(verdict, key=lambda x: abs(x[1]))
        print(f'VERDICT: data quality explains a real share of the error — '
              f'{best[0]} at r={best[1]:+.3f} (r^2={best[1]**2:.2f}).')
        print('The pipeline is at least partly at a DATA ceiling. Report this as a')
        print('finding: further loss/architecture work has limited headroom, and')
        print('improving registration would move the metric more than any loss.')
    else:
        print('VERDICT: neither proxy explains the per-case error (all |r| < 0.3).')
        print('The r=0.97 cross-model agreement is therefore NOT explained by')
        print('registration quality or bowel-gas change. Scaling the model (2.5-D /')
        print('3-D context) is justified — but consider other case-level factors')
        print('first: contrast dose/timing, scanner, patient habitus, slice spacing.')
    print('-' * 72)
    print(f'\nn = {len(keep_err)} cases. With this n, |r| < 0.44 is not significant')
    print('at p<0.05 — treat a weak correlation as inconclusive, not as absence.')

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            'n': len(keep_err),
            'bone_ncc': bone_ncc, 'bone_mae': bone_mae,
            'gas_dice': gas_dice, 'feature_l1_hu': keep_err,
            'r_bone_ncc': pearson(bone_ncc, keep_err),
            'r_bone_mae': pearson(bone_mae, keep_err),
            'r_gas_dice': pearson(gas_dice, keep_err),
        }, indent=2))
        print(f'\n[written] {args.out}')


if __name__ == '__main__':
    main()
