"""
Diagnostic: how much real segmentation-mask coverage does the actual data
directory have, for use_organ / use_seg_consistency.

`OrganWeightedLoss` and `SegmentationConsistencyLoss` are only meaningful if
a case actually has a `_seg_reg.nii.gz` file AND that file has non-trivial
(non-all-zero) coverage in the patches actually sampled. dataset.py silently
falls back to an all-zero mask when `seg_path` is missing or fails to load
(no error) — see `CTPairDataset.__init__` — so a broken/missing-mask data
directory would train `use_organ`/`use_seg_consistency` against an
effectively-degenerate loss without ever raising.

This script re-uses `find_pairs_and_split` (the exact same pairing logic
`train.py` uses) and reports, per split:
  - how many pairs have no `seg_path` at all (file missing on disk)
  - how many pairs have a `seg_path` that exists but fails to load or has a
    shape mismatch against the source volume (dataset.py falls back to
    all-zero for these too)
  - for pairs with a loadable mask: the fraction of non-zero voxels
    (min/median/max across cases) — near-zero means the mask is present but
    covers almost nothing, which would still make the organ/seg-consistency
    loss nearly a no-op even though nothing looks broken in the logs.

Usage (run on the machine that has the real data, e.g. the remote box):
    python check_seg_masks.py
    python check_seg_masks.py --data_dir /path/to/other/data_dir
"""

import argparse
import logging

import numpy as np

from config import train_config
from dataset import _load_vol, find_pairs_and_split

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


def _check_pairs(pairs, split_name: str):
    n_missing = 0
    n_load_fail = 0
    n_shape_mismatch = 0
    coverage = []  # fraction of non-zero voxels, one per successfully-loaded mask

    for pair in pairs:
        case_id  = pair['case_id']
        seg_path = pair.get('seg_path')
        if seg_path is None:
            n_missing += 1
            log.info(f"  [{split_name}] {case_id}: no seg_path (file not found on disk)")
            continue
        try:
            src_vol = _load_vol(pair['source_path'])
            seg_vol = _load_vol(seg_path)
        except Exception as e:
            n_load_fail += 1
            log.info(f"  [{split_name}] {case_id}: failed to load seg volume — {e}")
            continue
        if seg_vol.shape != src_vol.shape:
            n_shape_mismatch += 1
            log.info(f"  [{split_name}] {case_id}: shape mismatch "
                      f"seg={seg_vol.shape} vs src={src_vol.shape} — dataset.py "
                      f"falls back to all-zero mask for this case")
            continue
        frac_nonzero = float((seg_vol > 0).mean())
        coverage.append(frac_nonzero)
        log.info(f"  [{split_name}] {case_id}: OK, {frac_nonzero:.4%} non-zero voxels")

    n_total = len(pairs)
    n_ok = len(coverage)
    log.info(f"\n[{split_name}] summary: {n_total} pairs total")
    log.info(f"  {n_ok} loadable with a shape-matched mask")
    log.info(f"  {n_missing} missing seg_path (no _seg_reg.nii.gz on disk)")
    log.info(f"  {n_load_fail} present but failed to load")
    log.info(f"  {n_shape_mismatch} present but shape-mismatched")
    if coverage:
        arr = np.array(coverage)
        log.info(f"  non-zero voxel coverage: min={arr.min():.4%}  "
                  f"median={np.median(arr):.4%}  max={arr.max():.4%}")
        n_near_empty = int((arr < 0.001).sum())
        if n_near_empty:
            log.warning(f"  {n_near_empty}/{n_ok} loadable masks are <0.1% non-zero — "
                        f"near-degenerate, organ/seg-consistency loss will barely "
                        f"see these cases even though nothing errors")
    degenerate = n_missing + n_load_fail + n_shape_mismatch
    if degenerate:
        log.warning(f"  {degenerate}/{n_total} pairs ({degenerate/n_total:.1%}) will silently "
                    f"train use_organ/use_seg_consistency against an all-zero mask")
    return {
        'total': n_total, 'ok': n_ok, 'missing': n_missing,
        'load_fail': n_load_fail, 'shape_mismatch': n_shape_mismatch,
        'coverage': coverage,
    }


def main():
    p = argparse.ArgumentParser(description='Check real segmentation-mask coverage')
    p.add_argument('--data_dir', type=str, default=None)
    args = p.parse_args()

    cfg = dict(train_config)
    if args.data_dir:
        cfg['data_dir'] = args.data_dir

    train_pairs, val_pairs, test_pairs = find_pairs_and_split(cfg)

    results = {}
    for name, pairs in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
        if not pairs:
            log.info(f"[{name}] no pairs, skipping")
            continue
        results[name] = _check_pairs(pairs, name)

    log.info("\n" + "=" * 65)
    ok_ratio = sum(r['ok'] for r in results.values()) / max(1, sum(r['total'] for r in results.values()))
    if ok_ratio > 0.9:
        log.info(f"Overall: {ok_ratio:.1%} of pairs have a usable mask — "
                 f"use_organ/use_seg_consistency should be meaningful.")
    else:
        log.warning(f"Overall: only {ok_ratio:.1%} of pairs have a usable mask — "
                    f"use_organ/use_seg_consistency will be trained on mostly "
                    f"all-zero masks for the rest. Check data_dir's *_seg_reg.nii.gz "
                    f"coverage before trusting those scenarios' results.")


if __name__ == '__main__':
    main()
