#!/usr/bin/env python
"""Write a benchmark manifest whose "prediction" is the NCCT input itself.

WHY THIS EXISTS. Every number in the benchmark is an error against the real
CECT, and none of them says how much error there was to remove in the first
place. `featHU = 36.73` for an arterial arm is unreadable without that: it could
be a model that removed most of the enhancement gap, or one that removed almost
none of it. The identity — predict the source, change nothing — is the floor
that makes every other row a fraction of a known budget, and it needs no
training, no checkpoint and no inference pass.

It is also the cleanest test of an external baseline's export path. If ResViT's
exported volumes score like garbage but the identity pushed through the SAME
PNG -> HU -> NIfTI conversion also scores like garbage, the bug is in the
conversion, not the model.

The split comes from `find_pairs_and_split` driven by a run's own
`run_config.json`, so the case list is identical to that run's by construction
rather than by convention — anything else risks scoring the identity on a
different 20 cases than the models it anchors.

    python scripts/make_identity_manifest.py \
        --run ../out_synthesis_train/literature_baseline_l1_only \
        --out analysis/manifests

    # multi-phase: one manifest per phase, matching the run's target_phases
    python scripts/make_identity_manifest.py \
        --run ../out_synthesis_train/literature_baseline_multiphase_film \
        --out analysis/manifests

Then score them alongside everything else:

    python benchmark.py --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
        --runs_dir ../out_synthesis_train \
        --manifest identity_venous=analysis/manifests/identity_venous.csv \
        --manifest identity_arterial=analysis/manifests/identity_arterial.csv \
        --baseline l1_organ_groupnorm_s43 --out analysis/benchmark
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import find_pairs_and_split          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True,
                    help='run directory holding run_config.json — its split is reused')
    ap.add_argument('--out', type=Path, default=Path('analysis/manifests'))
    ap.add_argument('--split', default='test', choices=['test', 'val', 'train'])
    args = ap.parse_args()

    cfg = json.loads((args.run / 'run_config.json').read_text())
    # data_dir in run_config.json is relative to the TRAINING machine's cwd; on a
    # laptop holding only the logs this scan fails deep inside a directory walk.
    data_dir = Path(cfg.get('data_dir', ''))
    if not data_dir.is_dir():
        raise SystemExit(f"data_dir {str(data_dir)!r} (from {args.run}/run_config.json) "
                         f"is not reachable from {Path.cwd()} — run this on the machine "
                         f"holding the registered volumes, or cd to the directory that "
                         f"path is relative to")
    train, val, test = find_pairs_and_split(cfg)
    pairs = {'train': train, 'val': val, 'test': test}[args.split]
    if not pairs:
        raise SystemExit(f'no {args.split} pairs found for {args.run} — check '
                         f"data_dir={cfg.get('data_dir')!r} is reachable from here")

    # One manifest per phase. A single pooled file would let the paired test
    # compare an arterial identity against a venous model, which is exactly the
    # mistake the per-phase model rows already avoid.
    by_phase = {}
    for p in pairs:
        by_phase.setdefault(p['phase'], []).append(p)

    args.out.mkdir(parents=True, exist_ok=True)
    for phase, ps in sorted(by_phase.items()):
        # A pair with no segmentation has no organ mask, so every organ-region
        # metric would be NaN for that case and the identity would silently be
        # scored on fewer cases than the models it is supposed to anchor.
        usable = [p for p in ps if p['seg_path']]
        dropped = len(ps) - len(usable)
        path = args.out / f'identity_{phase}.csv'
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['gen_path', 'real_path',
                                              'mask_path', 'target_phase'])
            w.writeheader()
            for p in usable:
                w.writerow({'gen_path':     p['source_path'],   # the NCCT, unchanged
                            'real_path':    p['target_path'],
                            'mask_path':    p['seg_path'],
                            'target_phase': p['phase']})
        print(f'[written] {path}  n={len(usable)}'
              + (f'  ({dropped} dropped: no seg_path)' if dropped else ''))


if __name__ == '__main__':
    main()
