#!/usr/bin/env python
"""
Does synthesising a CECT actually close the multi-organ SEGMENTATION gap?

Motivation. Every number this project reports so far is image fidelity —
featHU, org_mae, the texture ratios — plus a phase classifier that PROJECT_PLAN
1.6 records as saturated (phase_acc = 1.000 for all seed runs, so it can no
longer rank models). None of them is the quantity the thesis title promises.
This script measures that quantity directly.

The design is a three-arm comparison against ONE fixed reference:

    reference   TotalSegmentator on the REAL CECT   (= the stored _seg_full mask
                every organ loss and every organ metric in this repo already uses)

    arm 'ncct'  TotalSegmentator on the REAL NCCT   -> this is THE GAP
    arm '<m>'   TotalSegmentator on model m's SYNTHETIC CECT
    arm 'cect'  TotalSegmentator on the REAL CECT   -> integrity check, see below

Because the split's volumes are already resampled onto a common grid and
co-registered (data_dir is `.../B2_deeds__aligned`), fixed and moving masks live
in the same voxel space and Dice is a fair voxel-wise comparison with no further
warping.

Three falsifiable predictions, and the script reports all three:

  1. THE GAP IS REAL AND ORGAN-DEPENDENT. Dice(ncct) < Dice(reference-by-
     construction = 1.0), and the deficit is concentrated in the organs whose
     boundary is defined by contrast — aorta, IVC, portal vein, pancreas,
     gallbladder — not spread uniformly.

  2. SKELETAL STRUCTURES SHOW NO GAP. Vertebrae, ribs and hips are visible
     identically with and without contrast. If the 'ncct' arm loses Dice on bone
     too, the measurement is picking up general image-quality differences (noise,
     dose, reconstruction kernel) rather than contrast, and prediction 1 does not
     mean what it appears to mean. THIS IS THE NEGATIVE CONTROL AND IT IS THE
     MOST IMPORTANT ROW IN THE OUTPUT.

  3. SYNTHESIS RECOVERS SOME FRACTION OF 1, BOUNDED BY THE ALEATORIC CEILING.
     Given the aleatoric finding (PROJECT_PLAN 1.9: held-out R^2 = -0.105 for
     predicting aortic CECT HU from the NCCT), a synthetic CECT cannot restore
     the case-specific enhancement LEVEL. It can restore the population-average
     level and the texture. If organ *shape* recovery depends mainly on the
     average level, synthesis helps; if it depends on the case-specific level,
     it cannot. Include the oracle-conditioned run (`level_all8`) as an arm and
     the difference between it and a normal run IS that ceiling, measured.

The 'cect' arm is an integrity check, not a result. Running TotalSegmentator on
the real CECT and scoring it against the stored _seg_full mask must return Dice
= 1.0 exactly. Anything else means the installed TotalSegmentator differs from
the one that produced the reference masks, and in that case EVERY number in this
script (and every organ metric elsewhere in the repo) is measured against a
reference the current environment cannot reproduce. Run it once. It is cheap.

Reference masks are pseudo-ground-truth: TotalSegmentator's own output on the
real CECT, not a human annotation. That is a real limitation and it must be
stated wherever these numbers appear, because it makes the comparison partly
circular — a synthetic volume can score well by looking like a CECT *to
TotalSegmentator* specifically. The skeletal negative control (prediction 2) and
the ncct arm's floor are what keep the comparison interpretable despite it. The
same caveat already applies to featHU, which reads organ medians out of these
same masks, so this script does not introduce a new dependency — it inherits the
existing one.

USAGE
-----
Two stages, because only the first needs a GPU and it is the slow one.

    # 1. segment  (GPU; resumable — existing outputs are skipped)
    #    Masks are written with NATIVE TotalSegmentator ids, the same numbering
    #    the reference _seg_full masks use. --label_space combine renumbers
    #    through combine_masks.py instead, which does NOT match the reference and
    #    exists only to reproduce old runs.
    python scripts/downstream_seg.py --stage segment \
        --work ../out_downstream_seg \
        --arm ncct --arm cect \
        --arm ours=../out_synthesis_train/literature_baseline_l1_organ_groupnorm_s43/phase_infer/manifest.csv \
        --arm oracle=../out_synthesis_train/literature_baseline_level_all8/phase_infer/manifest.csv \
        --arm resvit=../bench_ncct2cect/ResViT/results/vindr_nifti/manifest.csv

    # 2. score  (CPU, no GPU needed, minutes)
    #    --reference cect removes the offset between the stored masks and what
    #    this environment segments; --metric dice is the readable metric for
    #    vessels (6 mm erosion is near-degenerate below ~12 mm diameter).
    python scripts/downstream_seg.py --stage score \
        --work ../out_downstream_seg --out analysis/downstream_seg \
        --reference cect --metric dice --ceiling oracle \
        --arm ncct --arm cect --arm ours=... --arm oracle=... --arm resvit=...

Arms are `name=manifest.csv`, plus the two reserved names `ncct` and `cect`
which read their volumes straight from splits/split.json.

Metric definitions (eroded Dice at 6 mm, symmetric HD95, centroid displacement)
are byte-for-byte the same quantities as
`data_reg_pipeline/evaluate_pipeline/evaluate_registration.py`, so the
segmentation chapter and the registration chapter report one Dice, not two.
They are reimplemented here rather than imported so this script has no
cross-repository dependency on a host where only one repo may be checked out.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Organ groups.
#
# The stored `_seg_full` reference masks are in NATIVE TotalSegmentator
# numbering — the full 117-class map at config.py:TS_LABEL_MAP_JSON, the same
# ids resolve_organ_weights() and every per-organ metric in this repo already
# read. Groups are therefore keyed by NAME and resolved to ids through that map
# at run time, so a TotalSegmentator version bump fails loudly here instead of
# silently scoring the wrong anatomy. The path is duplicated rather than
# imported because config.py imports torch and the score stage must run on a
# CPU-only host — keep it in sync with config.py:TS_LABEL_MAP_JSON.
#
# Do NOT hardcode the COMPACT ids that data_reg_pipeline/segmentation/
# combine_masks.py assigns (liver=1, spleen=2, aorta=13, ...). That is a
# DIFFERENT label space from the reference masks. Scoring a combine_masks mask
# against a native-id reference gives Dice = 0 for every organ except
# kidney_left, which is id 3 in both maps by coincidence. That is what the first
# run of this script reported, and its integrity check blamed the installed
# TotalSegmentator for what was actually a label-space mismatch.
# ---------------------------------------------------------------------------

DEFAULT_TS_LABEL_MAP = "orgFeatXGB_CTPhase/retrain_out_full/ts_label_map_total.json"

GROUP_NAMES: Dict[str, List[str]] = {
    "parenchymal": ["liver", "spleen", "kidney_left", "kidney_right",
                    "pancreas", "gallbladder"],
    "vascular":    ["aorta", "inferior_vena_cava",
                    "portal_vein_and_splenic_vein",
                    "iliac_artery_left", "iliac_artery_right",
                    "iliac_vena_left", "iliac_vena_right"],
    "muscular":    ["autochthon_left", "autochthon_right",
                    "iliopsoas_left", "iliopsoas_right"],
    # The negative control. Bone is visible identically with and without
    # contrast, so any Dice deficit here is NOT a contrast effect.
    "skeletal":    ["vertebrae_L1", "vertebrae_L2", "vertebrae_L3",
                    "vertebrae_L4", "vertebrae_L5",
                    "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
                    "sacrum", "hip_left", "hip_right"],
}

# Filled by init_labels(); GROUPS keeps the group order above.
GROUPS: Dict[str, Dict[int, str]] = {g: {} for g in GROUP_NAMES}
LABEL_NAME: Dict[int, str] = {}
LABEL_GROUP: Dict[int, str] = {}
TS_NAME_TO_ID: Dict[str, int] = {}     # the full 117-class map, for --label_space native

# Composite endpoints, reported alongside the anatomical groups. They exist
# because a group mean dilutes the signal: liver, spleen and the kidneys have
# ~0.008 Dice of headroom between the ncct and cect arms and outvote pancreas
# and gallbladder 4-to-2 inside `parenchymal`. `contrast_critical` is the set of
# organs whose BOUNDARY is defined by contrast, chosen from the physics and from
# prediction 1 in this file's header — not from the results.
COMPOSITE_NAMES: Dict[str, List[str]] = {
    "contrast_critical": ["pancreas", "gallbladder", "aorta",
                          "inferior_vena_cava", "portal_vein_and_splenic_vein"],
    "bulk_parenchyma":   ["liver", "spleen", "kidney_left", "kidney_right"],
}

# endpoint name -> set of labels. Filled by init_labels(); anatomical groups
# first, then the composites, which deliberately overlap them.
ENDPOINTS: Dict[str, set] = {}


def init_labels(ts_label_map: str) -> None:
    """Resolve GROUP_NAMES to TotalSegmentator ids. Must run before scoring."""
    p = Path(ts_label_map)
    if not p.is_file():
        raise SystemExit(
            f"TotalSegmentator label map not found: {p}\n"
            "It is the name->id map the reference _seg_full masks are numbered "
            "with (config.py:TS_LABEL_MAP_JSON). Regenerate it with:\n"
            "  python orgFeatXGB_CTPhase/dump_ts_label_map.py")
    name_to_id = json.loads(p.read_text())
    missing = sorted({n for names in GROUP_NAMES.values()
                      for n in names if n not in name_to_id})
    if missing:
        raise SystemExit(
            f"these organs are not in {p}: {missing}\n"
            "The installed TotalSegmentator names its classes differently from "
            "the one that produced the reference masks. Fix that before scoring.")
    TS_NAME_TO_ID.update({k: int(v) for k, v in name_to_id.items()})
    for g, names in GROUP_NAMES.items():
        for n in names:
            lab = int(name_to_id[n])
            GROUPS[g][lab] = n
            LABEL_NAME[lab] = n
            LABEL_GROUP[lab] = g

    for g in GROUP_NAMES:
        ENDPOINTS[g] = set(GROUPS[g])
    for c, names in COMPOSITE_NAMES.items():
        missing = [n for n in names if n not in LABEL_NAME.values()]
        if missing:
            raise SystemExit(f"composite '{c}' names organs that no group scores: "
                             f"{missing}. Add them to GROUP_NAMES first.")
        ENDPOINTS[c] = {lab for lab, nm in LABEL_NAME.items() if nm in names}


EROSION_MM = 6.0          # matches evaluate_pipeline/config.py:EROSION_MM
RESERVED_ARMS = ("ncct", "cect")


# ===========================================================================
# METRICS — mirrors evaluate_pipeline/evaluate_registration.py
# ===========================================================================

def erode_mask(binary_mask: np.ndarray,
               spacing_zyx: Tuple[float, float, float],
               erosion_mm: float = EROSION_MM) -> np.ndarray:
    """Euclidean erosion in physical space via distance transform."""
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(binary_mask.astype(bool), sampling=spacing_zyx)
    return (dist >= erosion_mm).astype(np.uint8)


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> Optional[float]:
    """Plain Dice on the raw masks. None if either is empty."""
    a, b = mask_a.astype(bool), mask_b.astype(bool)
    sa, sb = int(a.sum()), int(b.sum())
    if sa == 0 or sb == 0:
        return None
    return 2.0 * float((a & b).sum()) / (sa + sb)


def eroded_dice(mask_fixed: np.ndarray,
                mask_moving: np.ndarray,
                spacing_zyx: Tuple[float, float, float],
                erosion_mm: float = EROSION_MM) -> Optional[float]:
    """Dice on 6 mm-eroded masks. None if either erodes to nothing."""
    ef = erode_mask(mask_fixed, spacing_zyx, erosion_mm)
    em = erode_mask(mask_moving, spacing_zyx, erosion_mm)
    if ef.sum() == 0 or em.sum() == 0:
        return None
    return 2.0 * float((ef & em).sum()) / (int(ef.sum()) + int(em.sum()))


def hd95_mm(mask_fixed: np.ndarray,
            mask_moving: np.ndarray,
            spacing_zyx: Tuple[float, float, float]) -> Optional[float]:
    """
    Symmetric 95th-percentile Hausdorff distance in mm.

    Sampling note carried over from the registration evaluator: arrays are
    (Z, Y, X) and `spacing_zyx` is already (z, y, x), so it is passed to
    `distance_transform_edt` UN-reversed. Reversing it silently swaps the Z and
    X physical scales.
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt

    mf, mm = mask_fixed.astype(bool), mask_moving.astype(bool)
    if mf.sum() == 0 or mm.sum() == 0:
        return None

    surf_f = mf & ~binary_erosion(mf)
    surf_m = mm & ~binary_erosion(mm)
    if surf_f.sum() == 0 or surf_m.sum() == 0:
        return None

    dt_f = distance_transform_edt(~surf_f, sampling=spacing_zyx)
    dt_m = distance_transform_edt(~surf_m, sampling=spacing_zyx)
    both = np.concatenate([dt_f[surf_m], dt_m[surf_f]])
    return float(np.percentile(both, 95))


def centroid_displacement_mm(mask_fixed: np.ndarray,
                             mask_moving: np.ndarray,
                             spacing_zyx: Tuple[float, float, float]) -> Optional[float]:
    """Euclidean distance between organ centroids, mm. Full (non-eroded) masks."""
    if mask_fixed.sum() == 0 or mask_moving.sum() == 0:
        return None
    cf = np.argwhere(mask_fixed.astype(bool)).mean(axis=0)
    cm = np.argwhere(mask_moving.astype(bool)).mean(axis=0)
    return float(np.linalg.norm((cf - cm) * np.asarray(spacing_zyx)))


def volume_error_pct(mask_fixed: np.ndarray, mask_moving: np.ndarray) -> Optional[float]:
    """Signed relative volume error, %. Negative = the arm under-segments."""
    sf = int(mask_fixed.astype(bool).sum())
    if sf == 0:
        return None
    sm = int(mask_moving.astype(bool).sum())
    return 100.0 * (sm - sf) / sf


# ===========================================================================
# IO
# ===========================================================================

def load_nifti(path: str):
    """Return (array (Z,Y,X), spacing (z,y,x)). Uses nibabel; falls back to SITK."""
    try:
        import nibabel as nib
        img = nib.load(path)
        arr = np.asanyarray(img.dataobj)
        zooms = img.header.get_zooms()[:3]          # (x, y, z)
        # nibabel gives (X, Y, Z); transpose to (Z, Y, X) to match the
        # registration evaluator's SimpleITK array convention.
        return np.transpose(arr, (2, 1, 0)), (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    except ImportError:
        import SimpleITK as sitk
        img = sitk.ReadImage(path)
        sp = img.GetSpacing()                        # (x, y, z)
        return sitk.GetArrayFromImage(img), (sp[2], sp[1], sp[0])


def read_split(split_path: Path) -> Tuple[List[dict], dict]:
    with open(split_path) as fh:
        d = json.load(fh)
    return d["test"], d


def read_manifest(path: Path) -> List[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def case_id_from_path(p: str) -> str:
    """
    Recover the case id from a manifest path.

    Generated volumes are named `<case_id>_syn.nii.gz` by infer_volume.py, and
    real/mask paths live in a directory named for the case. Try the filename
    first, then the parent directory.
    """
    stem = Path(p).name
    for suffix in ("_syn.nii.gz", ".nii.gz"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if stem.endswith("_syn"):
        stem = stem[:-4]
    return stem


# ===========================================================================
# STAGE 1 — SEGMENT
# ===========================================================================

def resolve_arm_volumes(arm: str,
                        spec: Optional[str],
                        test_cases: Sequence[dict],
                        split_dir: Path) -> List[Tuple[str, str, str]]:
    """
    Return [(case_id, volume_path, reference_mask_path)] for one arm.

    Paths inside split.json and the manifests are relative to the repo root
    (the directory this script's parent lives in), matching how every other
    script in this repo resolves them.
    """
    if arm in RESERVED_ARMS:
        key = "ncct" if arm == "ncct" else "cect"
        return [(c["case_id"], c[key], c["seg"]) for c in test_cases]

    if not spec:
        raise SystemExit(f"arm '{arm}' needs a manifest: --arm {arm}=path/to/manifest.csv")

    rows = read_manifest(Path(spec))
    out: List[Tuple[str, str, str]] = []
    by_case = {c["case_id"]: c for c in test_cases}
    for r in rows:
        cid = case_id_from_path(r["gen_path"])
        if cid not in by_case:
            # Manifests from bench_ncct2cect may name files differently; fall
            # back to matching on the real_path, which is always a split path.
            cid = case_id_from_path(Path(r["real_path"]).parent.name)
        if cid not in by_case:
            print(f"  ! {arm}: cannot map manifest row to a test case: {r['gen_path']}",
                  file=sys.stderr)
            continue
        out.append((cid, r["gen_path"], r.get("mask_path") or by_case[cid]["seg"]))
    return out


def run_totalsegmentator(vol: str, seg_dir: Path, fast: bool, extra: Sequence[str]) -> None:
    cmd = ["TotalSegmentator", "-i", vol, "-o", str(seg_dir)]
    if fast:
        cmd.append("--fast")
    cmd.extend(extra)
    subprocess.run(cmd, check=True)


def combine_native(seg_dir: Path, out_path: Path,
                   name_to_id: Dict[str, int]) -> Tuple[int, int]:
    """
    Merge TotalSegmentator's per-organ files into one native-id volume.

    Shared with scripts/rebuild_seg_native.py, which does the same thing after
    the fact for masks that were built through combine_masks.py. Imported rather
    than duplicated so the two can never drift.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from rebuild_seg_native import rebuild_case
    except ImportError as e:                                # pragma: no cover
        raise SystemExit(
            f"--label_space native needs scripts/rebuild_seg_native.py: {e}") from e
    return rebuild_case(seg_dir, out_path, name_to_id)


def stage_segment(args, test_cases: Sequence[dict], split_dir: Path) -> None:
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)

    combine = Path(args.combine_masks).resolve()
    if args.label_space == "combine" and not combine.is_file():
        raise SystemExit(
            f"--combine_masks not found: {combine}\n"
            "--label_space combine renumbers through it. Note that its compact map "
            "is NOT the label space of the reference masks — use the default "
            "--label_space native unless you know why you want otherwise."
        )

    for arm, spec in args.arms:
        vols = resolve_arm_volumes(arm, spec, test_cases, split_dir)
        arm_dir = work / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== arm '{arm}' — {len(vols)} volumes -> {arm_dir}")

        for i, (cid, vol, _ref) in enumerate(vols, 1):
            prefix = arm_dir / cid
            out_mask = Path(f"{prefix}_seg_full.nii.gz")
            if out_mask.is_file() and not args.force:
                print(f"  [{i}/{len(vols)}] {cid}: cached")
                continue
            if not Path(vol).is_file():
                print(f"  [{i}/{len(vols)}] {cid}: MISSING volume {vol}", file=sys.stderr)
                continue

            seg_dir = arm_dir / f"{cid}_segs"
            seg_dir.mkdir(parents=True, exist_ok=True)

            # Resume at the organ-file level too, not just the final mask.
            # TotalSegmentator is ~100 s/volume and is by far the expensive step;
            # if a previous attempt segmented this case but failed to combine,
            # there is no reason to pay for it again.
            n_organs = len(list(seg_dir.glob("*.nii.gz")))
            if n_organs > 0 and not args.force:
                print(f"  [{i}/{len(vols)}] {cid}: TotalSegmentator cached "
                      f"({n_organs} organ files)")
            else:
                print(f"  [{i}/{len(vols)}] {cid}: TotalSegmentator")
                run_totalsegmentator(str(Path(vol).resolve()), seg_dir,
                                     args.fast, args.ts_args)

            n_organs = len(list(seg_dir.glob("*.nii.gz")))
            if n_organs == 0:
                raise SystemExit(
                    f"TotalSegmentator wrote no organ files into {seg_dir}.\n"
                    "combine_masks.py expects one <organ>.nii.gz per organ, so a "
                    "multi-label run (--ml) or a task other than 'total' will not "
                    "work here. Check the TotalSegmentator invocation."
                )
            if args.label_space == "native":
                # Write the ids the reference masks actually use. Going through
                # combine_masks.py instead renumbers into its own compact map,
                # and scoring that against a native-id reference returns Dice = 0
                # for every organ except the ids the two maps share.
                print(f"  [{i}/{len(vols)}] {cid}: combine native ({n_organs} organ files)")
                n_mapped, n_unknown = combine_native(seg_dir, out_mask, TS_NAME_TO_ID)
                extra = f", {n_unknown} unmapped files skipped" if n_unknown else ""
                print(f"  [{i}/{len(vols)}] {cid}: {n_mapped} organs -> {out_mask.name}{extra}")
            else:
                print(f"  [{i}/{len(vols)}] {cid}: combine_masks ({n_organs} organ files)")

                # combine_masks.py runs with cwd=its own directory (that is where
                # its pipeline_logs/ live), so BOTH paths must be absolute — a
                # path relative to this repo would resolve against
                # data_reg_pipeline/ instead and the organ files would silently
                # not be found.
                subprocess.run([sys.executable, str(combine),
                                str(seg_dir.resolve()), str(prefix.resolve())],
                               check=True, cwd=str(combine.parent))

            if not out_mask.is_file():
                raise SystemExit(
                    f"the combine step reported success but {out_mask} does not "
                    "exist. Refusing to continue with a missing mask."
                )

            if args.cleanup:
                for f in seg_dir.glob("*.nii.gz"):
                    f.unlink()
                try:
                    seg_dir.rmdir()
                except OSError:
                    pass

    print("\nsegment stage done.")


# ===========================================================================
# STAGE 2 — SCORE
# ===========================================================================

def score_arm(arm: str,
              vols: Sequence[Tuple[str, str, str]],
              work: Path) -> List[dict]:
    rows: List[dict] = []
    arm_dir = work / arm

    for cid, _vol, ref_path in vols:
        pred_path = arm_dir / f"{cid}_seg_full.nii.gz"
        if not pred_path.is_file():
            print(f"  ! {arm}/{cid}: no segmentation, run --stage segment first",
                  file=sys.stderr)
            continue
        if not Path(ref_path).is_file():
            print(f"  ! {arm}/{cid}: reference mask missing: {ref_path}", file=sys.stderr)
            continue

        ref, sp_ref = load_nifti(ref_path)
        pred, sp_pred = load_nifti(str(pred_path))

        if ref.shape != pred.shape:
            # A shape mismatch silently becomes an all-zero mask elsewhere in
            # this repo (dataset.py:452). Refuse rather than repeat that.
            print(f"  ! {arm}/{cid}: shape {pred.shape} != reference {ref.shape}; skipped",
                  file=sys.stderr)
            continue
        if not np.allclose(sp_ref, sp_pred, atol=1e-3):
            print(f"  ! {arm}/{cid}: spacing {sp_pred} != reference {sp_ref}",
                  file=sys.stderr)

        for lab, name in LABEL_NAME.items():
            mf = ref == lab
            mm = pred == lab
            if mf.sum() == 0:
                continue                    # organ not in the reference FOV
            rows.append({
                "arm": arm,
                "case_id": cid,
                "label": lab,
                "organ": name,
                "group": LABEL_GROUP[lab],
                "dice": dice(mf, mm),
                "eroded_dice": eroded_dice(mf, mm, sp_ref),
                "hd95_mm": hd95_mm(mf, mm, sp_ref),
                "centroid_mm": centroid_displacement_mm(mf, mm, sp_ref),
                "vol_err_pct": volume_error_pct(mf, mm),
                "detected": int(mm.sum() > 0),
            })
        print(f"  {arm}/{cid}: scored")
    return rows


def _nanmean(xs: Sequence[Optional[float]]) -> float:
    a = np.asarray([x for x in xs if x is not None], dtype=np.float64)
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else float("nan")


def _sig(t: float, n: int) -> str:
    if not np.isfinite(t) or n < 3:
        return "ns"
    try:
        from scipy import stats
        p = float(2 * (1 - stats.t.cdf(abs(t), df=n - 1)))
    except ImportError:
        return "?"
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def _mde(sd: float, n: int, power: float = 0.80, alpha: float = 0.05) -> float:
    """
    Smallest paired difference this many cases can detect.

    Without it a `ns` is ambiguous: it can mean 'the arms are equivalent' or
    'n = 20 cannot see a difference this size'. Reported next to every test so
    the two readings are distinguishable.
    """
    if not np.isfinite(sd) or n < 3:
        return float("nan")
    try:
        from scipy import stats
    except ImportError:
        return float("nan")
    return float((stats.t.ppf(1 - alpha / 2, n - 1)
                  + stats.t.ppf(power, n - 1)) * sd / np.sqrt(n))


def _per_case(all_rows: Sequence[dict], arm: str, labels: set,
              metric: str) -> Dict[str, float]:
    """Mean of `metric` over the organs of one endpoint, per case."""
    acc: Dict[str, List[float]] = defaultdict(list)
    for r in all_rows:
        if r["arm"] != arm or r["label"] not in labels:
            continue
        v = r[metric]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            acc[r["case_id"]].append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def _paired_delta(a: Dict[str, float], b: Dict[str, float]) -> np.ndarray:
    """Per-case a - b over the cases both have. Order is stable."""
    keys = sorted(set(a) & set(b))
    d = np.asarray([a[k] - b[k] for k in keys], dtype=np.float64)
    return d[~np.isnan(d)]


def _t_of(d: np.ndarray) -> Tuple[float, float, int]:
    """(t, sd, n) of a paired difference vector."""
    n = int(d.size)
    if n < 2:
        return float("nan"), float("nan"), n
    sd = float(d.std(ddof=1))
    t = float(d.mean() / (sd / np.sqrt(n))) if sd > 0 else float("nan")
    return t, sd, n


def stage_score(args, test_cases: Sequence[dict], split_dir: Path) -> None:
    work = Path(args.work).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    metric = args.metric
    metric_label = "eroded Dice" if metric == "eroded_dice" else "Dice"
    ceilings = set(args.ceilings)
    unknown_ceiling = ceilings - {a for a, _ in args.arms}
    if unknown_ceiling:
        raise SystemExit(f"--ceiling names arms that are not scored: "
                         f"{sorted(unknown_ceiling)}")

    if args.reference == "cect" and not (work / "cect").is_dir():
        raise SystemExit("--reference cect needs the 'cect' arm segmented: "
                         f"{work/'cect'} does not exist.")

    all_rows: List[dict] = []
    for arm, spec in args.arms:
        vols = resolve_arm_volumes(arm, spec, test_cases, split_dir)
        if args.reference == "cect":
            vols = [(cid, vol, str(work / "cect" / f"{cid}_seg_full.nii.gz"))
                    for cid, vol, _ref in vols]
        print(f"\n=== scoring arm '{arm}'")
        all_rows.extend(score_arm(arm, vols, work))

    if not all_rows:
        raise SystemExit("no rows scored — run --stage segment first")

    fields = ["arm", "case_id", "label", "organ", "group",
              "dice", "eroded_dice", "hd95_mm", "centroid_mm", "vol_err_pct", "detected"]
    with open(out / "per_case.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    arms = [a for a, _ in args.arms]
    others = [a for a in arms if a not in RESERVED_ARMS]
    endpoints = list(ENDPOINTS)                     # groups first, then composites
    tag = lambda a: f"{a}†" if a in ceilings else a  # noqa: E731
    width = max([14] + [len(tag(a)) + 2 for a in arms])

    lines: List[str] = []
    lines.append("# Downstream multi-organ segmentation — does synthesis close the gap?\n")
    ref_desc = ("the stored `_seg_full` masks shipped with the split"
                if args.reference == "stored" else
                "THIS environment's TotalSegmentator run on the real CECT "
                "(the `cect` arm), so no version or resampling offset separates the "
                "reference from the arms")
    lines.append(f"Reference: TotalSegmentator on the REAL CECT — {ref_desc}. "
                 "Every arm is scored against it on the same test cases and the same voxel "
                 f"grid. Tables report **{metric_label}**"
                 + (" (6 mm erosion, matching the registration chapter)."
                    if metric == "eroded_dice" else ".") + "\n")
    if metric == "eroded_dice":
        lines.append("6 mm erosion removes a 6 mm shell from structures whose radius is "
                     "often smaller than that, so the iliac vessels and the portal vein "
                     "read near zero here while their plain Dice is 0.7-0.8. Re-run with "
                     "`--metric dice` to read the vascular rows.\n")
    if ceilings:
        lines.append("† marks a CEILING, not a competitor: an arm given information no "
                     "deployable model has. Its margin over a normal run is a measurement "
                     "of that information's value, never a win.\n")

    if "cect" in arms and args.reference == "cect":
        lines.append("\n**Integrity check: not applicable.** With `--reference cect` the "
                     "`cect` arm is scored against itself and is 1.0 by construction. The "
                     "offset it would have measured is instead removed from every arm.\n")
    elif "cect" in arms:
        c = _nanmean([r["dice"] for r in all_rows if r["arm"] == "cect"])
        if c > 0.999:
            verdict = "PASS — the installed TotalSegmentator reproduces the reference masks."
        else:
            # Two failure modes look identical in the mean and are told apart by
            # the SHAPE of the per-label Dice: a version/settings difference
            # degrades every organ a little; a label-space mismatch zeroes all
            # of them but leaves whichever ids the two maps share intact.
            per_lab = [(_nanmean([r["dice"] for r in all_rows
                                  if r["arm"] == "cect" and r["label"] == lab]), lab)
                       for lab in sorted(LABEL_NAME)]
            per_lab = [(v, lab) for v, lab in per_lab if np.isfinite(v)]
            n_zero = sum(1 for v, _ in per_lab if v < 0.01)
            n_high = sum(1 for v, _ in per_lab if v > 0.8)
            if per_lab and n_zero >= 0.5 * len(per_lab) and n_high:
                agree = ", ".join(f"{LABEL_NAME[lab]} ({lab})"
                                  for v, lab in per_lab if v > 0.8)
                verdict = (
                    "**FAIL — LABEL-SPACE MISMATCH, not a TotalSegmentator version "
                    f"problem.** {n_zero}/{len(per_lab)} organs score Dice ~0 while "
                    f"these score >0.8: {agree}. A degraded model loses a little Dice "
                    "on every organ; only two different numberings produce exact zeros "
                    "everywhere except the ids the two maps happen to share. Re-run the "
                    "segment stage with the default `--label_space native`, or rebuild "
                    "the existing masks with scripts/rebuild_seg_native.py.")
            else:
                verdict = (
                    "**FAIL** — the installed TotalSegmentator does NOT reproduce the "
                    "reference masks. Every number below, and every organ metric elsewhere "
                    "in this repo, is measured against a reference this environment cannot "
                    "regenerate. Consider `--reference cect`, which removes the offset by "
                    "scoring every arm against this environment's own CECT segmentation.")
        lines.append(f"\n**Integrity check (`cect` arm): mean Dice = {c:.4f}.** {verdict}\n")

    # ---- per endpoint ----------------------------------------------------
    lines.append(f"\n## By endpoint ({metric_label})\n")
    lines.append("| endpoint | " + " | ".join(tag(a) for a in arms) + " |")
    lines.append("|---" * (len(arms) + 1) + "|")
    for g in endpoints:
        cells = []
        for a in arms:
            v = _nanmean([r[metric] for r in all_rows
                          if r["arm"] == a and r["label"] in ENDPOINTS[g]])
            cells.append(f"{v:.4f}" if np.isfinite(v) else "—")
        mark = "**" if g in COMPOSITE_NAMES else ""
        lines.append(f"| {mark}{g}{mark} | " + " | ".join(cells) + " |")
    lines.append("\n`skeletal` is the NEGATIVE CONTROL: bone is contrast-independent, so all "
                 "arms should agree there.\n")
    lines.append("The two bold rows cut across the anatomical groups. "
                 "`contrast_critical` is the organs whose boundary contrast defines "
                 "(pancreas, gallbladder, aorta, IVC, portal vein); `bulk_parenchyma` is "
                 "liver, spleen and the kidneys, which are already well seen without "
                 "contrast. The group means mix the two and dilute the effect — "
                 "`parenchymal` outvotes pancreas and gallbladder 4-to-2.\n")

    # ---- per organ -------------------------------------------------------
    lines.append(f"\n## By organ ({metric_label})\n")
    lines.append("| organ | group | " + " | ".join(tag(a) for a in arms) + " |")
    lines.append("|---" * (len(arms) + 2) + "|")
    for lab, name in sorted(LABEL_NAME.items(), key=lambda kv: (LABEL_GROUP[kv[0]], kv[1])):
        cells = []
        for a in arms:
            v = _nanmean([r[metric] for r in all_rows
                          if r["arm"] == a and r["label"] == lab])
            cells.append(f"{v:.4f}" if np.isfinite(v) else "—")
        if all(c == "—" for c in cells):
            continue
        lines.append(f"| {name} | {LABEL_GROUP[lab]} | " + " | ".join(cells) + " |")

    # ---- boundary metrics ------------------------------------------------
    # Computed for every row and, until now, never reported. They do not
    # degenerate on thin structures the way eroded Dice does, so they are the
    # readable metric for the vessels.
    lines.append("\n## Boundary accuracy — HD95 (mm), lower is better\n")
    lines.append("| endpoint | " + " | ".join(tag(a) for a in arms) + " |")
    lines.append("|---" * (len(arms) + 1) + "|")
    for g in endpoints:
        cells = []
        for a in arms:
            v = _nanmean([r["hd95_mm"] for r in all_rows
                          if r["arm"] == a and r["label"] in ENDPOINTS[g]])
            cells.append(f"{v:.2f}" if np.isfinite(v) else "—")
        lines.append(f"| {g} | " + " | ".join(cells) + " |")

    lines.append("\n## Centroid displacement (mm) / signed volume error (%)\n")
    lines.append("| endpoint | " + " | ".join(tag(a) for a in arms) + " |")
    lines.append("|---" * (len(arms) + 1) + "|")
    for g in endpoints:
        cells = []
        for a in arms:
            c_ = _nanmean([r["centroid_mm"] for r in all_rows
                           if r["arm"] == a and r["label"] in ENDPOINTS[g]])
            v_ = _nanmean([r["vol_err_pct"] for r in all_rows
                           if r["arm"] == a and r["label"] in ENDPOINTS[g]])
            cells.append(f"{c_:.2f} / {v_:+.1f}" if np.isfinite(c_) else "—")
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    lines.append("\nA negative volume error means the arm under-segments the organ.\n")

    # ---- gap closed ------------------------------------------------------
    if "ncct" in arms:
        # The ceiling is the `cect` arm, NOT 1.0. Segmenting the real CECT does
        # not reproduce the stored masks exactly (resampling of the reference
        # onto the aligned grid, plus any TotalSegmentator version difference),
        # and that offset applies to every arm equally. Normalising by 1.0 would
        # divide the achievable headroom by ~4.
        has_ceiling = "cect" in arms
        lines.append("\n## Fraction of the gap closed\n")
        lines.append("Per endpoint: `(arm - ncct) / (ceiling - ncct)`. 0% = no better than "
                     "segmenting the non-contrast scan directly; 100% = indistinguishable "
                     "from segmenting the real CECT. The ceiling is "
                     + ("the `cect` arm — segmenting the real CECT in this environment — "
                        "which is the highest score any arm can reach here.\n"
                        if has_ceiling else
                        "1.0, because no `cect` arm was scored. Add `--arm cect`: without "
                        "it the denominator is wrong and every percentage is understated.\n"))
        lines.append("| endpoint | ncct | ceiling | headroom | "
                     + " | ".join(tag(a) for a in others) + " |")
        lines.append("|---" * (len(others) + 4) + "|")
        for g in endpoints:
            base = _nanmean([r[metric] for r in all_rows
                             if r["arm"] == "ncct" and r["label"] in ENDPOINTS[g]])
            top = (_nanmean([r[metric] for r in all_rows
                             if r["arm"] == "cect" and r["label"] in ENDPOINTS[g]])
                   if has_ceiling else 1.0)
            cells = []
            for a in others:
                v = _nanmean([r[metric] for r in all_rows
                              if r["arm"] == a and r["label"] in ENDPOINTS[g]])
                if np.isfinite(v) and np.isfinite(base) and np.isfinite(top) and top > base:
                    cells.append(f"{100.0 * (v - base) / (top - base):+.1f}%")
                else:
                    cells.append("—")
            head = f"{top - base:+.4f}" if np.isfinite(top) and np.isfinite(base) else "—"
            lines.append(f"| {g} | {base:.4f} | {top:.4f} | {head} | " + " | ".join(cells) + " |")
        lines.append("\n`headroom` is what is actually there to win. Where it is a few "
                     "thousandths — the skeletal and muscular rows — the percentage column "
                     "is a ratio of two noise terms and must not be read as a result.\n")

        # ---- paired tests vs ncct ---------------------------------------
        lines.append("\n## Paired per-case tests vs the `ncct` arm (positive = better)\n")
        lines.append("```")
        lines.append(f"{'arm':<{width}}{'endpoint':<19}{'d':>9}{'t':>8}  {'sig':<5}"
                     f"{'MDE':>8}  better")
        for a in others:
            for g in endpoints:
                d = _paired_delta(_per_case(all_rows, a, ENDPOINTS[g], metric),
                                  _per_case(all_rows, "ncct", ENDPOINTS[g], metric))
                t, sd, n = _t_of(d)
                if n < 2:
                    continue
                mde = _mde(sd, n)
                flag = ""
                if _sig(t, n) == "ns" and np.isfinite(mde) and mde > abs(d.mean()):
                    flag = "  <- underpowered, not evidence of equivalence"
                lines.append(f"{tag(a):<{width}}{g:<19}{d.mean():>+9.4f}{t:>8.2f}  "
                             f"{_sig(t, n):<5}{mde:>8.4f}  {int((d > 0).sum())}/{n}{flag}")
        lines.append("```")
        lines.append("\nEach case contributes the mean " + metric_label + " over the organs "
                     "of that endpoint, so the test is paired on cases (n = test cases), not "
                     "on organs. Between-case variance dominates between-arm differences on "
                     "this data — the same reason the synthesis benchmark uses paired "
                     "tests.\n")
        lines.append("`MDE` is the smallest difference detectable at 80% power, alpha = 0.05, "
                     "with this many cases. When `MDE` exceeds the observed `d`, an `ns` "
                     "means the experiment could not see an effect of that size — it is not "
                     "evidence that the arms are equivalent.\n")

        # ---- generation tax ---------------------------------------------
        if "skeletal" in ENDPOINTS:
            lines.append("\n## Generation tax, and the contrast effect net of it\n")
            lines.append("Bone is contrast-independent, so an arm's deficit on `skeletal` "
                         "is what SYNTHESIS ITSELF costs: blur, texture and artefacts that "
                         "damage anatomy contrast was never going to change. Subtracting it "
                         "from the vascular gain answers the objection prediction 2 raises — "
                         "that the vascular row might be general image quality rather than "
                         "contrast — using each arm as its own control.\n")
            lines.append("| arm | Δ vascular | Δ skeletal (tax) | net | t(net) | sig | "
                         "Δ contrast_critical | sig |")
            lines.append("|---" * 8 + "|")
            nc_v = _per_case(all_rows, "ncct", ENDPOINTS["vascular"], metric)
            nc_s = _per_case(all_rows, "ncct", ENDPOINTS["skeletal"], metric)
            nc_c = _per_case(all_rows, "ncct", ENDPOINTS["contrast_critical"], metric)
            for a in others:
                dv = _paired_delta(_per_case(all_rows, a, ENDPOINTS["vascular"], metric), nc_v)
                ds = _paired_delta(_per_case(all_rows, a, ENDPOINTS["skeletal"], metric), nc_s)
                dc = _paired_delta(_per_case(all_rows, a, ENDPOINTS["contrast_critical"],
                                             metric), nc_c)
                if dv.size < 2 or ds.size != dv.size:
                    continue
                net = dv - ds
                tn, _, nn = _t_of(net)
                tc, _, ncn = _t_of(dc)
                lines.append(f"| {tag(a)} | {dv.mean():+.4f} | {ds.mean():+.4f} | "
                             f"**{net.mean():+.4f}** | {tn:.2f} | {_sig(tn, nn)} | "
                             f"{dc.mean():+.4f} | {_sig(tc, ncn)} |")
            lines.append("\nA negative tax on every arm is expected and is itself a result: "
                         "synthesis pays a uniform cost on anatomy it should have left alone. "
                         "The `net` column is only interpretable while that tax is roughly "
                         "constant across arms — if one arm's tax is several times another's, "
                         "compare the taxes first.\n")

    lines.append("\n**Limitation, to be repeated wherever these numbers appear.** The "
                 "reference is TotalSegmentator's own output on the real CECT, not a human "
                 "annotation. A synthetic volume can therefore score well by resembling a "
                 "CECT *to TotalSegmentator specifically*. The skeletal negative control and "
                 "the `ncct` floor are what make the remaining signal interpretable.\n")

    (out / "summary.md").write_text("\n".join(lines))
    print(f"\nwrote {out/'per_case.csv'} and {out/'summary.md'}")


# ===========================================================================

def parse_arm(s: str) -> Tuple[str, Optional[str]]:
    if "=" in s:
        name, spec = s.split("=", 1)
        return name.strip(), spec.strip()
    return s.strip(), None


def main() -> None:
    here = Path(__file__).resolve().parent.parent

    ap = argparse.ArgumentParser(
        description="Downstream multi-organ segmentation evaluation for NCCT->CECT synthesis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("USAGE")[-1])
    ap.add_argument("--stage", choices=["segment", "score", "all"], default="all")
    ap.add_argument("--split", default=str(here / "splits" / "split.json"))
    ap.add_argument("--work", default="../out_downstream_seg",
                    help="where per-arm segmentations are cached")
    ap.add_argument("--out", default="analysis/downstream_seg")
    ap.add_argument("--arm", action="append", dest="arm_specs", default=[],
                    metavar="NAME[=MANIFEST]",
                    help="repeatable; 'ncct' and 'cect' read from split.json")
    ap.add_argument("--label_space", choices=["native", "combine"], default="native",
                    help="'native' writes the segment stage's masks with the same "
                         "TotalSegmentator ids the reference masks use — this is "
                         "what makes the two comparable. 'combine' renumbers "
                         "through combine_masks.py's compact map, which does NOT "
                         "match the reference; kept only to reproduce old runs.")
    ap.add_argument("--ceiling", action="append", dest="ceilings", default=[],
                    metavar="NAME",
                    help="repeatable; arms that are ceilings rather than "
                         "competitors (e.g. an oracle-conditioned run). Marked "
                         "with a dagger in the tables.")
    ap.add_argument("--reference", choices=["stored", "cect"], default="stored",
                    help="'stored' scores against the _seg_full masks shipped with "
                         "the split. 'cect' scores against THIS environment's "
                         "TotalSegmentator run on the real CECT (the 'cect' arm), "
                         "which removes any offset between the two — use it when "
                         "the integrity check does not return 1.0.")
    ap.add_argument("--metric", choices=["eroded_dice", "dice"], default="eroded_dice",
                    help="metric for the summary tables. 6 mm erosion matches the "
                         "registration chapter but is near-degenerate on vessels "
                         "thinner than ~12 mm (iliacs, portal vein) — use 'dice' "
                         "to read those.")
    ap.add_argument("--ts_label_map", default=str(here / DEFAULT_TS_LABEL_MAP),
                    help="name->id map the reference _seg_full masks use "
                         "(config.py:TS_LABEL_MAP_JSON)")
    ap.add_argument("--combine_masks", default="../data_reg_pipeline/segmentation/combine_masks.py",
                    help="path to the combine_masks.py that built the reference masks")
    ap.add_argument("--fast", action="store_true",
                    help="TotalSegmentator --fast (3 mm). Only if the reference "
                         "masks were also built with it — otherwise the integrity "
                         "check will fail, correctly.")
    ap.add_argument("--ts_args", nargs=argparse.REMAINDER, default=[],
                    help="extra args passed through to TotalSegmentator")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete per-organ TS output after combining (saves disk)")
    ap.add_argument("--force", action="store_true", help="re-segment even if cached")
    args = ap.parse_args()

    init_labels(args.ts_label_map)

    if not args.arm_specs:
        raise SystemExit("give at least one --arm (e.g. --arm ncct --arm ours=.../manifest.csv)")
    args.arms = [parse_arm(s) for s in args.arm_specs]

    seen = set()
    for a, _ in args.arms:
        if a in seen:
            raise SystemExit(f"duplicate arm name: {a}")
        seen.add(a)

    split_path = Path(args.split)
    test_cases, _meta = read_split(split_path)
    print(f"split: {split_path}  ({len(test_cases)} test cases)")
    print(f"arms : {', '.join(a for a, _ in args.arms)}")

    os.chdir(here)          # every path in split.json / manifests is repo-relative

    if args.stage in ("segment", "all"):
        stage_segment(args, test_cases, split_path.parent)
    if args.stage in ("score", "all"):
        stage_score(args, test_cases, split_path.parent)


if __name__ == "__main__":
    main()
