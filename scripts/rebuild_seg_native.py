#!/usr/bin/env python
"""
Rebuild downstream_seg.py's per-arm `_seg_full` masks in NATIVE TotalSegmentator
label ids, from the per-organ files that the segment stage already wrote.

WHY. The reference masks stored beside the split are numbered with the full
117-class TotalSegmentator map (config.py:TS_LABEL_MAP_JSON) — spleen=1,
kidney_right=2, kidney_left=3, gallbladder=4, liver=5, aorta=52, IVC=63 ...
`downstream_seg.py --stage segment` instead passed TotalSegmentator's output
through data_reg_pipeline/segmentation/combine_masks.py, which renumbers into a
COMPACT map (liver=1, spleen=2, aorta=13 ...). Scoring one against the other
gives Dice = 0 for every organ except kidney_left, which is id 3 in both maps by
coincidence — a result that looks like a catastrophic segmentation failure and
is really a units error.

The fix does not need the GPU again. TotalSegmentator's own per-organ output is
still cached at <work>/<arm>/<case>_segs/<organ>.nii.gz (unless the segment
stage was run with --cleanup), and building a native-id multi-label volume from
it is exactly what `TotalSegmentator --ml` would have written.

The previous mask is renamed to <case>_seg_combine.nii.gz rather than deleted —
it is the only record of what was scored before, and its presence is what marks
a case as already rebuilt.

USAGE
    python scripts/rebuild_seg_native.py --work ../out_downstream_seg
    python scripts/downstream_seg.py --stage score --work ../out_downstream_seg ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

DEFAULT_TS_LABEL_MAP = "orgFeatXGB_CTPhase/retrain_out_full/ts_label_map_total.json"


def rebuild_case(seg_dir: Path, out_path: Path, name_to_id: dict) -> tuple[int, int]:
    """Combine <organ>.nii.gz files into one native-id volume. -> (n_organs, n_unknown)"""
    organ_files = sorted(seg_dir.glob("*.nii.gz"))
    if not organ_files:
        raise RuntimeError(f"no organ files in {seg_dir}")

    ref = nib.load(str(organ_files[0]))
    out = np.zeros(ref.shape, dtype=np.int16)

    # Ascending id, so that on the (rare) overlap the higher id wins
    # deterministically instead of depending on directory order.
    todo, unknown = [], []
    for f in organ_files:
        name = f.name[: -len(".nii.gz")]
        if name in name_to_id:
            todo.append((int(name_to_id[name]), f))
        else:
            unknown.append(name)

    for lab, f in sorted(todo):
        img = nib.load(str(f))
        if img.shape != ref.shape:
            raise RuntimeError(f"{f.name} shape {img.shape} != {ref.shape}")
        m = np.asanyarray(img.dataobj) > 0
        if m.any():
            out[m] = lab

    img = nib.Nifti1Image(out, ref.affine, ref.header)
    img.set_data_dtype(np.int16)
    nib.save(img, str(out_path))
    return len(todo), len(unknown)


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="../out_downstream_seg")
    ap.add_argument("--ts_label_map", default=str(here / DEFAULT_TS_LABEL_MAP))
    ap.add_argument("--arm", action="append", dest="arms", default=[],
                    help="repeatable; default = every arm directory under --work")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if a _seg_combine.nii.gz backup already exists")
    args = ap.parse_args()

    lm = Path(args.ts_label_map)
    if not lm.is_file():
        raise SystemExit(f"label map not found: {lm}\n"
                         "Generate it with: python orgFeatXGB_CTPhase/dump_ts_label_map.py")
    name_to_id = json.loads(lm.read_text())
    print(f"label map: {lm}  ({len(name_to_id)} classes)")

    work = Path(args.work).resolve()
    if not work.is_dir():
        raise SystemExit(f"--work not found: {work}")
    arm_dirs = ([work / a for a in args.arms] if args.arms
                else sorted(d for d in work.iterdir() if d.is_dir()))

    n_done = n_skip = n_fail = 0
    for arm_dir in arm_dirs:
        if not arm_dir.is_dir():
            print(f"! missing arm dir: {arm_dir}", file=sys.stderr)
            n_fail += 1
            continue
        seg_dirs = sorted(arm_dir.glob("*_segs"))
        print(f"\n=== {arm_dir.name} — {len(seg_dirs)} cached cases")
        if not seg_dirs:
            print("  ! nothing to rebuild: the per-organ output is gone (--cleanup?). "
                  "This arm must be re-segmented.", file=sys.stderr)
            n_fail += 1
            continue

        for seg_dir in seg_dirs:
            cid = seg_dir.name[: -len("_segs")]
            out_path = arm_dir / f"{cid}_seg_full.nii.gz"
            backup = arm_dir / f"{cid}_seg_combine.nii.gz"

            if backup.exists() and not args.force:
                print(f"  {cid}: already rebuilt")
                n_skip += 1
                continue
            try:
                if out_path.exists() and not backup.exists():
                    out_path.rename(backup)
                n_org, n_unk = rebuild_case(seg_dir, out_path, name_to_id)
            except Exception as e:                      # noqa: BLE001
                print(f"  ! {cid}: {e}", file=sys.stderr)
                n_fail += 1
                continue
            extra = f", {n_unk} unmapped organ files skipped" if n_unk else ""
            print(f"  {cid}: {n_org} organs -> {out_path.name}{extra}")
            n_done += 1

    print(f"\nrebuilt {n_done}, already done {n_skip}, failed {n_fail}")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
