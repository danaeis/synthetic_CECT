#!/usr/bin/env python3
"""
Qualitative comparison panels for the NCCT→CECT tables.

Produces the figure that sits next to the comparison table (and, with a
different model list, next to the ablation table): one column per model, one
figure per anatomical level, with

    row 1   zoomed inset of the ROI at that level
    row 2   the synthetic CECT slice (display window)
    row 3   the difference map  gen − real  in HU, shared diverging scale

and column (a) reserved for the reference pair (real CECT on top, NCCT at the
bottom), so the reader can see what the models were given and what they had to
reproduce.

WHY IT READS MANIFESTS
----------------------
Models are addressed exactly the way benchmark.py addresses them: a manifest
CSV with `gen_path,real_path,mask_path,target_phase`, either passed explicitly
(`--manifest name=path`) or auto-discovered under `--runs_dir`. A figure built
from a different source than the table is a figure that can disagree with the
table; this one cannot. `discover()` is a local copy of benchmark.discover()
rather than an import, for the same reason scripts/downstream_seg.py keeps its
own DEFAULT_TS_LABEL_MAP: importing benchmark pulls in the XGBoost phase
evaluator, and figures must render on a CPU-only host.

SLICE SELECTION IS ANATOMICAL, NOT UNIFORM
------------------------------------------
"A random slice" from a uniform draw is usually a mid-liver slice, because that
is where most of the volume is. Each level below instead scores every axial
slice from the organ mask (native TotalSegmentator ids), keeps the slices whose
score is within `--band` of the best, and draws ONE of those with the seeded
RNG. So the slice is random, but it is random *within* the requested region,
and the same --seed reproduces the figure exactly.

Levels: aorta_cardia (aorta + gastric cardia / distal oesophagus), liver
(mid-liver), kidney (renal hila), pelvis (iliac wings / bladder).

USAGE
-----
    # comparison table figure — every discovered run plus the external baselines
    python scripts/qualitative_panel.py \
        --runs_dir ../out_synthesis_train \
        --manifest pix2pix=../ncct2cect/pix2pix/results/vindr_nifti/manifest.csv \
        --manifest resvit=../ncct2cect/ResViT/results/vindr_nifti/manifest.csv \
        --order 'pix2pix=Pix2pix,pix2pixhd_baseline=Pix2pixHD,l1_only=U-Net,
                 cyclegan=CycleGAN,resvit=ResViT,transunet=TransUNet,
                 swinunetr=SwinUNETR,eagan=Ea-GAN,
                 l1_organ_groupnorm_s42=Ours' \
        --out analysis/figures/comparison --seed 11

    # ablation table figure — same case and same slices, ablation arms only
    python scripts/qualitative_panel.py --runs_dir ../out_synthesis_train \
        --order 'l1_only=L1,l1_adv=+adv,l1_adv_organ=+organ,
                 l1_organ_groupnorm_s42=+GroupNorm (full)' \
        --case <case_id_printed_by_the_run_above> \
        --out analysis/figures/ablation --seed 11

    # segmentation-overlay figure from the downstream_seg work dir
    python scripts/qualitative_panel.py --mode seg \
        --seg_work analysis/downstream_seg \
        --order 'cect=CECT,l1_only=U-Net,l1_organ_groupnorm_s42=Ours' \
        --case <case_id> --out analysis/figures/seg
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import ConnectionPatch, Rectangle

REPO = Path(__file__).resolve().parent.parent

# Kept in sync with config.py:TS_LABEL_MAP_JSON and downstream_seg.py — the
# NATIVE 117-class TotalSegmentator numbering the `mask_path` volumes use.
DEFAULT_TS_LABEL_MAP = REPO / 'orgFeatXGB_CTPhase' / 'retrain_out_full' / 'ts_label_map_total.json'

PANEL_LETTERS = 'abcdefghijklmnopqrstuvwxyz'


# ===========================================================================
# Anatomical levels
# ===========================================================================

@dataclass(frozen=True)
class Level:
    key: str
    title: str
    # every inner tuple is an OR-group; a candidate slice must satisfy all of them
    require: Tuple[Tuple[str, ...], ...]
    score: Tuple[str, ...]      # organs whose combined slice area ranks candidates
    roi: Tuple[str, ...]        # organs that define the zoom box
    # restrict candidates to slices superior/inferior to this organ's centroid
    superior_to: Optional[str] = None
    inferior_to: Optional[str] = None


LEVELS: Dict[str, Level] = {l.key: l for l in [
    Level('aorta_cardia', 'Aorta / gastric cardia',
          require=(('aorta',), ('stomach', 'esophagus')),
          score=('stomach', 'esophagus', 'aorta'),
          roi=('aorta', 'stomach', 'esophagus'),
          superior_to='liver'),
    Level('liver', 'Mid-liver',
          require=(('liver',),),
          score=('liver',),
          roi=('liver',)),
    Level('kidney', 'Renal hila',
          require=(('kidney_left',), ('kidney_right',)),
          score=('kidney_left', 'kidney_right', 'aorta'),
          roi=('kidney_left', 'kidney_right')),
    Level('pelvis', 'Pelvis',
          require=(('hip_left', 'hip_right'),),
          score=('hip_left', 'hip_right', 'urinary_bladder',
                 'iliac_artery_left', 'iliac_artery_right'),
          roi=('hip_left', 'hip_right', 'urinary_bladder'),
          inferior_to='liver'),
]}

# Organ overlay palette for --mode seg. Anything not listed is left uncoloured,
# which keeps the overlay readable instead of painting all 117 classes.
SEG_COLORS: Dict[str, str] = {
    'liver':                        '#d98b83',
    'spleen':                       '#7fb069',
    'stomach':                      '#4c9f70',
    'pancreas':                     '#e0c341',
    'gallbladder':                  '#8fd694',
    'kidney_left':                  '#5b8cba',
    'kidney_right':                 '#5b8cba',
    'aorta':                        '#c1666b',
    'inferior_vena_cava':           '#6a8eae',
    'portal_vein_and_splenic_vein': '#b08bbb',
    'autochthon_left':              '#cdb380',
    'autochthon_right':             '#cdb380',
    'urinary_bladder':              '#79addc',
}


# ===========================================================================
# Model sources — same contract as benchmark.py
# ===========================================================================

def discover(runs_dir: Path) -> Dict[str, Path]:
    """Runs that went through infer_volume.py. Mirrors benchmark.discover()."""
    found: Dict[str, Path] = {}
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name.replace('literature_baseline_', '')
        pi = d / 'phase_infer'
        if (pi / 'manifest.csv').exists():
            found[name] = pi / 'manifest.csv'
            continue
        for m in (sorted(pi.glob('*/manifest.csv')) if pi.is_dir() else []):
            found[f'{name}/{m.parent.name}'] = m
    return found


def read_manifest(path: Path) -> List[dict]:
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def case_key(row: dict) -> str:
    """Cases are matched across models on the REAL volume, as benchmark.py does."""
    return str(Path(row['real_path']).resolve())


def case_id_from_path(p: str) -> str:
    """Mirrors downstream_seg.case_id_from_path()."""
    stem = Path(p).name
    for suffix in ('_syn.nii.gz', '.nii.gz'):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem[:-4] if stem.endswith('_syn') else stem


def parse_order(spec: Optional[str]) -> List[Tuple[str, str]]:
    """'key=Label,key2=Label2' → [(key, label)]. A bare key labels itself."""
    if not spec:
        return []
    out = []
    for tok in spec.replace('\n', ' ').split(','):
        tok = tok.strip()
        if not tok:
            continue
        key, _, label = tok.partition('=')
        out.append((key.strip(), (label.strip() or key.strip())))
    return out


# ===========================================================================
# Volume IO / orientation
# ===========================================================================

def load_vol(path: str, canonical: bool = True) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    img = nib.load(str(path))
    if canonical:
        img = nib.as_closest_canonical(img)      # → RAS: +x right, +y anterior, +z superior
    arr = np.asanyarray(img.dataobj).astype(np.float32)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return arr, zooms


def axial(vol: np.ndarray, z: int) -> np.ndarray:
    """One axial slice in radiological display orientation.

    In RAS the slice `vol[:, :, z]` is indexed (x→patient right, y→anterior).
    Transposing puts anterior on the row axis, and the 180° rotation puts
    anterior at the top and the patient's right on the image's left — the
    convention every CT figure in this thesis uses.
    """
    return np.rot90(vol[:, :, z].T, 2)


def to_display(img_hu: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((img_hu - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


# ===========================================================================
# Slice selection
# ===========================================================================

def organ_area_profile(mask: np.ndarray, ids: Sequence[int]) -> np.ndarray:
    """Voxels of `ids` per axial slice, as a (Z,) array."""
    if not ids:
        return np.zeros(mask.shape[2], dtype=np.int64)
    return np.isin(mask, list(ids)).sum(axis=(0, 1)).astype(np.int64)


def _centroid_z(profile: np.ndarray) -> Optional[float]:
    tot = profile.sum()
    if tot <= 0:
        return None
    return float((profile * np.arange(profile.size)).sum() / tot)


@dataclass
class SlicePick:
    z: int
    rule: str                      # which rule actually fired, for the manifest
    n_candidates: int


def pick_slice(mask: np.ndarray, name2id: Dict[str, int], level: Level,
               rng: random.Random, band: float = 0.85) -> SlicePick:
    """A random slice from the band of slices that best represent `level`.

    Falls back, loudly and in order: full rule → drop the superior/inferior
    constraint → drop the `require` groups → the body-extent fraction. A silent
    fallback would put a pelvis label on a liver slice.
    """
    nz = mask.shape[2]
    prof = {o: organ_area_profile(mask, [name2id[o]]) for o in
            set(level.score) | set(level.roi) | {n for g in level.require for n in g}
            | ({level.superior_to} if level.superior_to else set())
            | ({level.inferior_to} if level.inferior_to else set())
            if o in name2id}

    score = np.zeros(nz, dtype=np.float64)
    for o in level.score:
        if o in prof:
            score += prof[o].astype(np.float64)

    ok = np.ones(nz, dtype=bool)
    for group in level.require:
        present = np.zeros(nz, dtype=bool)
        for o in group:
            if o in prof:
                present |= prof[o] > 0
        ok &= present

    zlim = np.ones(nz, dtype=bool)
    if level.superior_to and level.superior_to in prof:
        c = _centroid_z(prof[level.superior_to])
        if c is not None:
            zlim &= np.arange(nz) >= c
    if level.inferior_to and level.inferior_to in prof:
        c = _centroid_z(prof[level.inferior_to])
        if c is not None:
            zlim &= np.arange(nz) <= c

    for rule, valid in (('full', ok & zlim), ('no_zlimit', ok), ('score_only', score > 0)):
        cand = np.where(valid & (score > 0))[0]
        if cand.size:
            best = score[cand].max()
            keep = cand[score[cand] >= band * best]
            return SlicePick(int(rng.choice(list(keep))), rule, int(keep.size))

    # Nothing of this level is segmented in this case: fall back to a fixed
    # fraction of the body extent so the figure still renders, and say so.
    body = np.where((mask > 0).sum(axis=(0, 1)) > 0)[0]
    frac = {'aorta_cardia': 0.85, 'liver': 0.6, 'kidney': 0.45, 'pelvis': 0.12}
    lo, hi = (int(body.min()), int(body.max())) if body.size else (0, nz - 1)
    z = int(round(lo + frac.get(level.key, 0.5) * (hi - lo)))
    return SlicePick(z, 'fallback_body_fraction', 0)


def roi_box(mask_slice: np.ndarray, name2id: Dict[str, int], level: Level,
            size: int, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """(r0, r1, c0, c1) zoom window centred on the level's ROI in DISPLAY coords."""
    ids = [name2id[o] for o in level.roi if o in name2id]
    m = np.isin(mask_slice, ids) if ids else np.zeros_like(mask_slice, bool)
    if m.any():
        rr, cc = np.nonzero(m)
        r, c = int(rr.mean()), int(cc.mean())
    else:
        r, c = shape[0] // 2, shape[1] // 2
    half = size // 2
    r0 = int(np.clip(r - half, 0, max(shape[0] - size, 0)))
    c0 = int(np.clip(c - half, 0, max(shape[1] - size, 0)))
    return r0, r0 + min(size, shape[0]), c0, c0 + min(size, shape[1])


# ===========================================================================
# Figure 1 — synthesis comparison (zoom / image / difference)
# ===========================================================================

def render_synth_figure(case_id: str, level: Level, z: int,
                        ncct: np.ndarray, real: np.ndarray,
                        gens: List[Tuple[str, np.ndarray]],
                        box: Tuple[int, int, int, int],
                        win: Tuple[float, float], diff_clip: float,
                        out_base: Path, fmts: Sequence[str], dpi: int,
                        title: Optional[str] = None) -> None:
    r0, r1, c0, c1 = box
    ncols = 1 + len(gens)
    cell = 1.9
    fig = plt.figure(figsize=(cell * ncols + 0.9, cell * 3 + 0.75))
    gs = fig.add_gridspec(3, ncols, wspace=0.015, hspace=0.015,
                          left=0.005, right=0.955, top=0.965, bottom=0.055)

    def show(ax, img, **kw):
        ax.imshow(img, **kw)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return ax

    gray = dict(cmap='gray', vmin=0.0, vmax=1.0, interpolation='nearest')
    dnorm = Normalize(vmin=-diff_clip, vmax=diff_clip)

    columns: List[Tuple[str, np.ndarray, Optional[np.ndarray]]] = [
        ('NCCT & CECT', real, None)] + [(lbl, g, g - real) for lbl, g in gens]

    im_diff = None
    for j, (label, img_hu, diff) in enumerate(columns):
        # --- row 2: full slice ------------------------------------------------
        ax_full = show(fig.add_subplot(gs[1, j]), to_display(img_hu, *win), **gray)
        ax_full.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0,
                                    fill=False, ec='#e8a56b', lw=0.9))
        # --- row 1: zoom ------------------------------------------------------
        ax_zoom = show(fig.add_subplot(gs[0, j]),
                       to_display(img_hu[r0:r1, c0:c1], *win), **gray)
        for s in ax_zoom.spines.values():
            s.set_visible(True); s.set_color('#e8a56b'); s.set_linewidth(0.9)
        # connector from the box on the full slice to the inset above it
        for xyA, xyB in (((0, r1), (0, c1 - c0 - 1)), ((c1 - c0 - 1, r1), (c1 - c0 - 1, 0))):
            fig.add_artist(ConnectionPatch(
                xyA=(xyA[0] + c0, r0), coordsA=ax_full.transData,
                xyB=(xyB[0], (r1 - r0) - 1), coordsB=ax_zoom.transData,
                color='#e8a56b', lw=0.7, alpha=0.9))
        # --- row 3: difference (or the NCCT input in the reference column) ----
        ax_d = fig.add_subplot(gs[2, j])
        if diff is None:
            show(ax_d, to_display(ncct, *win), **gray)
        else:
            d = diff.copy()
            d[real < win[0]] = np.nan              # air: nothing to compare
            im_diff = ax_d.imshow(d, cmap='RdBu_r', norm=dnorm, interpolation='nearest')
            ax_d.set_xticks([]); ax_d.set_yticks([])
            for s in ax_d.spines.values():
                s.set_visible(False)
        ax_d.set_xlabel(f'({PANEL_LETTERS[j]}) {label}', fontsize=8, labelpad=3)

    if im_diff is not None:
        cax = fig.add_axes([0.962, 0.06, 0.010, 0.26])
        cb = fig.colorbar(im_diff, cax=cax)
        cb.set_label('Difference (HU)', fontsize=7)
        cb.ax.tick_params(labelsize=6)

    for row, txt in ((0, 'Zoom'), (1, 'CECT images'), (2, 'Difference')):
        fig.text(0.9585, 1 - (0.055 + (row + 0.5) * 0.303), txt, rotation=270,
                 va='center', ha='left', fontsize=7.5)

    if title:
        fig.suptitle(title, fontsize=9, y=0.995)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in fmts:
        fig.savefig(f'{out_base}.{fmt}', dpi=dpi, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)


# ===========================================================================
# Figure 2 — segmentation overlay
# ===========================================================================

def overlay(img01: np.ndarray, seg: np.ndarray, name2id: Dict[str, int],
            alpha: float = 0.45) -> np.ndarray:
    rgb = np.repeat(img01[..., None], 3, axis=2)
    for organ, hexcol in SEG_COLORS.items():
        lid = name2id.get(organ)
        if lid is None:
            continue
        m = seg == lid
        if not m.any():
            continue
        col = np.array(matplotlib.colors.to_rgb(hexcol))
        rgb[m] = (1 - alpha) * rgb[m] + alpha * col
    return np.clip(rgb, 0, 1)


def render_seg_figure(case_id: str, level: Level, z: int,
                      arms: List[Tuple[str, np.ndarray, np.ndarray]],
                      box: Tuple[int, int, int, int], win: Tuple[float, float],
                      name2id: Dict[str, int], per_row: int,
                      out_base: Path, fmts: Sequence[str], dpi: int) -> None:
    r0, r1, c0, c1 = box
    blocks = math.ceil(len(arms) / per_row)
    ncols = min(per_row, len(arms))
    cell = 2.1
    fig = plt.figure(figsize=(cell * ncols, cell * 2 * blocks + 0.35 * blocks))
    gs = fig.add_gridspec(2 * blocks, ncols, wspace=0.015, hspace=0.02,
                          left=0.005, right=0.995, top=0.98, bottom=0.03)

    for k, (label, img_hu, seg) in enumerate(arms):
        b, j = divmod(k, per_row)
        img01 = to_display(img_hu, *win)
        full = overlay(img01, seg, name2id)

        ax_z = fig.add_subplot(gs[2 * b, j])
        ax_z.imshow(full[r0:r1, c0:c1], interpolation='nearest')
        ax_z.set_xticks([]); ax_z.set_yticks([])
        for s in ax_z.spines.values():
            s.set_color('#e8a56b'); s.set_linewidth(0.9)

        ax_f = fig.add_subplot(gs[2 * b + 1, j])
        ax_f.imshow(full, interpolation='nearest')
        ax_f.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0,
                                 fill=False, ec='#e8a56b', lw=0.9))
        ax_f.set_xticks([]); ax_f.set_yticks([])
        for s in ax_f.spines.values():
            s.set_visible(False)
        ax_f.set_xlabel(f'({PANEL_LETTERS[k]}) {label}', fontsize=8, labelpad=3)

        fig.add_artist(ConnectionPatch(
            xyA=(c0, r0), coordsA=ax_f.transData,
            xyB=(0, (r1 - r0) - 1), coordsB=ax_z.transData,
            color='#e8a56b', lw=0.7))

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in fmts:
        fig.savefig(f'{out_base}.{fmt}', dpi=dpi, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)


# ===========================================================================
# Drivers
# ===========================================================================

def build_sources(args) -> Dict[str, Path]:
    src: Dict[str, Path] = {}
    if args.runs_dir:
        src.update(discover(Path(args.runs_dir)))
    for spec in args.manifest:
        name, _, path = spec.partition('=')
        if not path:
            raise SystemExit(f'--manifest expects name=path, got: {spec}')
        src[name.strip()] = Path(path.strip())
    return src


def select_models(src: Dict[str, Path], order: List[Tuple[str, str]]
                  ) -> List[Tuple[str, str, Path]]:
    if not order:
        return [(k, k, v) for k, v in src.items()]
    out, missing = [], []
    for key, label in order:
        if key in src:
            out.append((key, label, src[key]))
        else:
            missing.append(key)
    if missing:
        print(f'  ! --order names models with no manifest: {", ".join(missing)}\n'
              f'    available: {", ".join(sorted(src))}', file=sys.stderr)
    if not out:
        raise SystemExit('no model in --order has a manifest; nothing to draw')
    return out


def run_synth(args, name2id: Dict[str, int], rng: random.Random) -> None:
    src = build_sources(args)
    if not src:
        raise SystemExit('no models: pass --runs_dir and/or --manifest')
    models = select_models(src, parse_order(args.order))

    rows: Dict[str, Dict[str, dict]] = {}     # model → case_key → row
    for key, _label, man in models:
        rows[key] = {case_key(r): r for r in read_manifest(man)}

    common = set.intersection(*(set(v) for v in rows.values()))
    if not common:
        raise SystemExit('the selected models share no case; check the manifests')

    if args.case:
        chosen = [k for k in sorted(common) if args.case in k]
        if not chosen:
            raise SystemExit(f'--case {args.case} not among the {len(common)} shared cases')
    else:
        chosen = rng.sample(sorted(common), k=min(args.n_cases, len(common)))

    record = {'seed': args.seed, 'display_window': list(args.window),
              'diff_clip': args.diff_clip, 'models': [m[0] for m in models],
              'figures': []}

    for ckey in chosen:
        ref = rows[models[0][0]][ckey]
        cid = case_id_from_path(ref['real_path'])
        real, _ = load_vol(ref['real_path'], not args.no_canonical)
        mask, _ = load_vol(ref['mask_path'], not args.no_canonical)
        mask = np.rint(mask).astype(np.int32)
        ncct_path = args.ncct_from and Path(args.ncct_from) / f'{cid}.nii.gz'
        ncct = None
        if ncct_path and ncct_path.is_file():
            ncct, _ = load_vol(str(ncct_path), not args.no_canonical)
        if ncct is None:
            # No NCCT given: show the real CECT again rather than inventing one.
            ncct = real

        gens: List[Tuple[str, np.ndarray]] = []
        for key, label, _man in models:
            g, _ = load_vol(rows[key][ckey]['gen_path'], not args.no_canonical)
            if g.shape != real.shape:
                print(f'  ! {key}: shape {g.shape} != real {real.shape} — skipped',
                      file=sys.stderr)
                continue
            gens.append((label, g))

        print(f'\ncase {cid}  ({len(gens)} models, {real.shape})')
        for lkey in args.levels:
            level = LEVELS[lkey]
            pick = pick_slice(mask, name2id, level, rng, args.band)
            ms = axial(mask, pick.z)
            box = roi_box(ms, name2id, level, args.zoom_size, ms.shape)
            print(f'  {lkey:<13} z={pick.z:<4} rule={pick.rule:<22} '
                  f'({pick.n_candidates} candidate slices)')

            out_base = Path(args.out) / f'{cid}__{lkey}'
            render_synth_figure(
                cid, level, pick.z,
                axial(ncct, pick.z), axial(real, pick.z),
                [(lbl, axial(g, pick.z)) for lbl, g in gens],
                box, tuple(args.window), args.diff_clip,
                out_base, args.fmt, args.dpi,
                title=(f'{level.title} — {cid}  (z={pick.z})' if args.title else None))
            record['figures'].append(
                {'case': cid, 'level': lkey, 'z': pick.z, 'rule': pick.rule,
                 'box': list(box), 'files': [f'{out_base}.{f}' for f in args.fmt]})

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / 'figure_manifest.json').write_text(json.dumps(record, indent=2))
    print(f'\nwrote {len(record["figures"])} figure(s) → {args.out}')


def run_seg(args, name2id: Dict[str, int], rng: random.Random) -> None:
    """Overlay figure from the downstream_seg work dir.

    Each arm directory holds `<case_id>_seg_full.nii.gz` (downstream_seg.py
    stage 1). The underlying grey image is the arm's own volume: the real CECT
    for the `cect` arm, the synthetic volume for every model arm.
    """
    work = Path(args.seg_work)
    order = parse_order(args.order) or [(d.name, d.name) for d in sorted(work.iterdir())
                                        if d.is_dir()]
    src = build_sources(args)

    if not args.case:
        cands = sorted({p.name.replace('_seg_full.nii.gz', '')
                        for p in (work / order[0][0]).glob('*_seg_full.nii.gz')})
        if not cands:
            raise SystemExit(f'no segmentations under {work / order[0][0]}')
        cid = rng.choice(cands)
    else:
        cid = args.case

    arms: List[Tuple[str, np.ndarray, np.ndarray]] = []
    mask_ref = None
    for key, label in order:
        seg_p = work / key / f'{cid}_seg_full.nii.gz'
        if not seg_p.is_file():
            print(f'  ! {key}: no {seg_p.name}', file=sys.stderr)
            continue
        seg, _ = load_vol(str(seg_p), not args.no_canonical)
        vol_p = None
        if key in src:
            hit = [r for r in read_manifest(src[key]) if case_id_from_path(r['gen_path']) == cid]
            if hit:
                vol_p = hit[0]['gen_path']
                mask_ref = mask_ref or hit[0]['mask_path']
        if vol_p is None:
            vol_p = args.seg_image_tpl.format(arm=key, case=cid) if args.seg_image_tpl else None
        if vol_p is None or not Path(vol_p).is_file():
            print(f'  ! {key}: no grey volume (give --manifest {key}=... or --seg_image_tpl)',
                  file=sys.stderr)
            continue
        img, _ = load_vol(vol_p, not args.no_canonical)
        arms.append((label, img, np.rint(seg).astype(np.int32)))

    if not arms:
        raise SystemExit('no arm could be assembled')

    ref_seg = arms[0][2]
    for lkey in args.levels:
        level = LEVELS[lkey]
        pick = pick_slice(ref_seg, name2id, level, rng, args.band)
        ms = axial(ref_seg, pick.z)
        box = roi_box(ms, name2id, level, args.zoom_size, ms.shape)
        print(f'  {lkey:<13} z={pick.z:<4} rule={pick.rule}')
        render_seg_figure(
            cid, level, pick.z,
            [(lbl, axial(img, pick.z), axial(seg, pick.z)) for lbl, img, seg in arms],
            box, tuple(args.window), name2id, args.per_row,
            Path(args.out) / f'{cid}__{lkey}__seg', args.fmt, args.dpi)
    print(f'\nwrote → {args.out}')


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Qualitative panels for the comparison / ablation tables',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['synth', 'seg'], default='synth')
    ap.add_argument('--runs_dir', type=Path, default=None,
                    help='auto-discover <run>/phase_infer/manifest.csv, like benchmark.py')
    ap.add_argument('--manifest', action='append', default=[], metavar='NAME=PATH')
    ap.add_argument('--order', default=None,
                    help="column order and display names: 'key=Label,key2=Label2'")
    ap.add_argument('--case', default=None, help='case id (substring); default: random')
    ap.add_argument('--n_cases', type=int, default=1)
    ap.add_argument('--levels', default='aorta_cardia,liver,kidney,pelvis')
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--band', type=float, default=0.85,
                    help='keep slices scoring >= band*best, then draw one at random')
    ap.add_argument('--zoom_size', type=int, default=112, help='inset size in voxels')
    ap.add_argument('--window', type=float, nargs=2, default=[-150.0, 250.0],
                    metavar=('LO', 'HI'), help='display HU window')
    ap.add_argument('--diff_clip', type=float, default=150.0, help='+/- HU on the diff map')
    ap.add_argument('--ncct_from', default=None,
                    help='directory of <case_id>.nii.gz NCCT volumes for column (a)')
    ap.add_argument('--label_map', type=Path, default=DEFAULT_TS_LABEL_MAP)
    ap.add_argument('--no_canonical', action='store_true',
                    help='do not reorient to RAS (use only if the grids are already RAS)')
    ap.add_argument('--seg_work', type=Path, default=REPO / 'analysis' / 'downstream_seg')
    ap.add_argument('--seg_image_tpl', default=None,
                    help='template for an arm grey volume, e.g. "/data/{arm}/{case}.nii.gz"')
    ap.add_argument('--per_row', type=int, default=6, help='--mode seg: panels per block')
    ap.add_argument('--out', type=Path, default=REPO / 'analysis' / 'figures')
    ap.add_argument('--fmt', default='png,pdf')
    ap.add_argument('--dpi', type=int, default=400)
    ap.add_argument('--title', action='store_true', help='draw a title (drafts only)')
    args = ap.parse_args()

    args.fmt = [f.strip() for f in args.fmt.split(',') if f.strip()]
    args.levels = [l.strip() for l in args.levels.split(',') if l.strip()]
    bad = [l for l in args.levels if l not in LEVELS]
    if bad:
        raise SystemExit(f'unknown level(s) {bad}; choose from {sorted(LEVELS)}')

    if not Path(args.label_map).is_file():
        raise SystemExit(f'label map not found: {args.label_map}\n'
                         'regenerate with orgFeatXGB_CTPhase/dump_ts_label_map.py')
    name2id = {k: int(v) for k, v in json.loads(Path(args.label_map).read_text()).items()}

    rng = random.Random(args.seed)
    (run_seg if args.mode == 'seg' else run_synth)(args, name2id, rng)


if __name__ == '__main__':
    main()
