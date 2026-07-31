#!/usr/bin/env python
"""
What physical size is a patch, actually?

Motivation. Nothing in this pipeline reads voxel spacing — `dataset.py:48` is
`nib.load(path).get_fdata()` and the affine is only carried through to the output
file. `PATCH_SIZE` is therefore in VOXELS, so a 128-px patch covers a different
number of millimetres in every case, and the model sees the same anatomy at
different scales without being told.

Two open questions depend on the answer, and they point opposite ways:

  * PROJECT_PLAN.md §1.5 argues in-plane context is already sufficient — at
    1.5 mm a 128-px patch spans 192 mm, more than the aorta's 36x81 mm
    cross-section — so the real gap is Z (`patch_depth=1`). That figure came from
    the SEGMENTATION masks, not the CT volumes.
  * If the volumes are nearer 0.7-0.8 mm, a 128-px patch spans only ~100 mm, and
    the in-plane argument changes.

This settles it. Header-only: `get_zooms()` does not read voxel data, so the whole
dataset scans in seconds.

Reads the frozen split (`splits/split.json`) when present so the numbers describe
exactly the cases that were trained on; otherwise re-derives it from config.

Usage:
    python scripts/spacing_stats.py
    python scripts/spacing_stats.py --patch 256 --split all
    python scripts/spacing_stats.py --out_json analysis/spacing.json
"""
import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import nibabel as nib


def _pct(vals, p):
    if not vals:
        return float('nan')
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
    return s[i]


def _describe(name, vals, unit='mm'):
    if not vals:
        print(f"  {name:<18} (none)")
        return {}
    d = {'n': len(vals), 'min': min(vals), 'p25': _pct(vals, 25),
         'median': st.median(vals), 'p75': _pct(vals, 75), 'max': max(vals),
         'mean': st.fmean(vals)}
    d['ratio_max_min'] = d['max'] / d['min'] if d['min'] else float('inf')
    print(f"  {name:<18} min {d['min']:7.3f}  p25 {d['p25']:7.3f}  "
          f"median {d['median']:7.3f}  p75 {d['p75']:7.3f}  max {d['max']:7.3f} {unit}"
          f"   (max/min = {d['ratio_max_min']:.2f}x)")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='train',
                    choices=['train', 'val', 'test', 'all'])
    ap.add_argument('--patch', type=int, default=None,
                    help='patch size in voxels for the FOV column (default: config)')
    ap.add_argument('--split_json', type=Path, default=Path('splits/split.json'),
                    help='frozen split; falls back to find_pairs_and_split if absent')
    ap.add_argument('--out_json', type=Path, default=None)
    a = ap.parse_args()

    from config import train_config as cfg
    patch = a.patch or (cfg['patch_size'] if isinstance(cfg['patch_size'], int)
                        else cfg['patch_size'][0])

    # Prefer the frozen split: these are the exact cases that were trained on.
    # dump_split.py writes {case_id, ncct, cect, seg}; find_pairs_and_split
    # returns {case_id, source_path, target_path, seg_path}. Normalise both.
    def _norm(p):
        return {'case_id': p.get('case_id'),
                'source_path': p.get('source_path') or p.get('ncct'),
                'target_path': p.get('target_path') or p.get('cect')}

    if a.split_json.exists():
        js = json.loads(a.split_json.read_text())
        keys = ['train', 'val', 'test'] if a.split == 'all' else [a.split]
        pairs = [_norm(p) for k in keys for p in js.get(k, [])]
        src = str(a.split_json)
    else:
        from dataset import find_pairs_and_split
        tr, va, te = find_pairs_and_split(cfg)
        pairs = [_norm(p) for p in {'train': tr, 'val': va, 'test': te,
                                    'all': tr + va + te}[a.split]]
        src = 'find_pairs_and_split(config)'

    print(f"split '{a.split}' from {src}: {len(pairs)} pairs   patch={patch} voxels\n")
    if not pairs:
        raise SystemExit("no pairs found")

    rows, missing = [], 0
    for p in pairs:
        for role in ('source_path', 'target_path'):
            path = p.get(role)
            if not path or not Path(path).exists():
                missing += 1
                continue
            h = nib.load(path).header
            zx, zy, zz = (float(v) for v in h.get_zooms()[:3])
            shape = nib.load(path).shape
            rows.append({'case_id': p.get('case_id'), 'role': role,
                         'sx': zx, 'sy': zy, 'sz': zz, 'shape': list(shape[:3])})
    if missing:
        print(f"⚠ {missing} volume(s) not readable at their recorded path — "
              f"is this the training host?\n")
    if not rows:
        raise SystemExit("no readable volumes")

    inplane = [r['sx'] for r in rows] + [r['sy'] for r in rows]
    sx = [r['sx'] for r in rows]
    sy = [r['sy'] for r in rows]
    sz = [r['sz'] for r in rows]
    fov = [patch * r['sx'] for r in rows]

    print(f"voxel spacing over {len(rows)} volumes")
    out = {'n_volumes': len(rows), 'patch': patch, 'split': a.split}
    out['sx'] = _describe('in-plane X', sx)
    out['sy'] = _describe('in-plane Y', sy)
    out['sz'] = _describe('through-plane Z', sz)
    print()
    out['patch_fov_mm'] = _describe(f'{patch}px FOV', fov)

    aniso = [r['sz'] / r['sx'] for r in rows if r['sx']]
    print()
    out['z_over_xy'] = _describe('Z / in-plane', aniso, unit='x')

    # Anisotropy within a single volume would break an in-plane-only resample.
    xy_mismatch = [r['case_id'] for r in rows if abs(r['sx'] - r['sy']) > 1e-4]
    print(f"\n  volumes with sx != sy: {len(xy_mismatch)}"
          + (f"  e.g. {xy_mismatch[:3]}" if xy_mismatch else ''))

    common = Counter((round(r['sx'], 3), round(r['sy'], 3), round(r['sz'], 3))
                     for r in rows).most_common(5)
    print("\n  most common (sx, sy, sz):")
    for k, n in common:
        print(f"    {k}   x{n}")

    # A source/target pair scanned at different spacings is a separate problem
    # from between-case variation: the model would be learning a resample.
    by_case = {}
    for r in rows:
        by_case.setdefault(r['case_id'], {})[r['role']] = r
    pair_mismatch = [c for c, d in by_case.items()
                     if 'source_path' in d and 'target_path' in d
                     and abs(d['source_path']['sx'] - d['target_path']['sx']) > 1e-3]
    print(f"\n  cases where NCCT and CECT differ in-plane: {len(pair_mismatch)}"
          + (f"  e.g. {pair_mismatch[:3]}" if pair_mismatch else ''))
    out['pair_spacing_mismatch'] = pair_mismatch

    # ── verdict ──────────────────────────────────────────────────────────────
    med = st.median(inplane)
    ratio = max(inplane) / min(inplane)
    print(f"\n{'=' * 72}")
    print(f"median in-plane spacing : {med:.3f} mm")
    print(f"→ a {patch}px patch spans {patch * med:.0f} mm at the median, "
          f"but {patch * min(inplane):.0f}-{patch * max(inplane):.0f} mm across the set")
    print(f"→ in-plane spacing varies {ratio:.2f}x between volumes")
    out['median_inplane'] = med
    out['inplane_ratio'] = ratio
    out['suggested_target_spacing'] = round(med, 2)

    if ratio < 1.15:
        print("\nVERDICT: spacing is effectively uniform. Resampling would buy little —\n"
              "  deprioritise it, and read the FOV number above as reliable.")
    else:
        print(f"\nVERDICT: spacing varies materially ({ratio:.2f}x). The same 128-px patch\n"
              f"  is a different anatomical extent per case, so the model sees anatomy at\n"
              f"  inconsistent scale. Resample in-plane to target_spacing="
              f"{round(med, 2)} mm\n"
              f"  (order=1 volumes, order=0 masks; leave Z alone — dims=2).")

    print(f"\nPROJECT_PLAN.md §1.5 assumed 1.5 mm → 192 mm for a 128px patch.")
    print(f"  Measured: {patch * med:.0f} mm. "
          + ("§1.5 holds — in-plane context is sufficient, Z is the gap."
             if patch * med >= 170 else
             "§1.5 does NOT hold — in-plane FOV is smaller than assumed;\n"
             "  reconsider patch size alongside the Z work."))

    if a.out_json:
        a.out_json.parent.mkdir(parents=True, exist_ok=True)
        a.out_json.write_text(json.dumps({**out, 'per_volume': rows}, indent=2))
        print(f"\nwrote {a.out_json}")


if __name__ == '__main__':
    main()
