#!/usr/bin/env python3
"""
Qualitative comparison figures for the NCCT->CECT benchmark.

Reproduces the paper-style panels (NCCT / real CECT / every model's synthetic
CECT + subtraction maps) for a RANDOM held-out case, sampled at several
anatomical levels (aorta & cardia, mid-liver, kidney, pelvis). Two figure sets
are produced:

  * comparison  -- the external / architecture-swap baselines vs. our method,
                   i.e. the rows of the main comparison table.
  * ablation    -- our own loss / capacity ablation variants vs. our method.

For each set it emits one axial figure per anatomical level plus, unless
--no-multiplane, a combined multi-view figure (axial + coronal by default; add
sagittal via --planes).

Layout is square-first: instead of one long strip, NCCT | CECT | every model
are laid out as panels wrapped into blocks (auto-sized to keep the figure
roughly square via --per_row), so each slice is drawn large enough to read.
Every panel carries a magnified organ ZOOM inset over the level's anatomy with
the true organ boundary drawn on it (--zoom / --zoom_size / --no-zoom-contour),
so boundary blur and mis-registration stand out model-by-model; the full IMAGE
marks the zoom box; and a synth-ref DIFFERENCE row (--no-diff to drop it)
carries the HU residual.

Everything is driven off the SAME manifests benchmark.py already scores from
(`<run>/phase_infer[/<phase>]/manifest.csv` under --runs_dir, plus any
`**/manifest.csv` under --bench_dir), so a model appears in the figure exactly
when it appears in the table. No model checkpoints are loaded and no torch is
imported: we only read the NIfTI volumes the manifests point at.

Manifest columns (see infer_volume.py): gen_path, real_path, mask_path,
target_phase. The NCCT source is not stored in the manifest but lives in the
same case directory as real_path (the registered `_deeds` volumes share a grid),
so it is recovered by globbing that directory.

Slice selection is anatomy-aware: the segmentation mask (TotalSegmentator label
ids, orgFeatXGB_CTPhase/retrain_out_full/ts_label_map_total.json) is used to
find the axial slice that best represents each requested body region. If an
organ is missing for a case, that level falls back to a fixed depth quantile.

The two default sets are the PLC-Net paper tables — `comparison` = Table 3
(published methods), `ablation` = Table 6 (generator architectures / diffusion
arms) — so the standard invocation needs no member flags:

Usage
-----
    cd synthetic_CECT
    python scripts/make_comparison_figures.py \
        --runs_dir ../out_synthesis_train \
        --bench_dir ../bench_ncct2cect \
        --set both --seed 0 --out_dir analysis/figures

    # pin PLC-Net's inference variant / phase (defaults: E=infer_best,
    # F=infer_popphase, phase=venous):
    python scripts/make_comparison_figures.py \
        --plcnet_f_infer infer_best --phase venous

    # one explicit case / custom member list still works:
    python scripts/make_comparison_figures.py --case <StudyInstanceUID> \
        --comparison "Pix2pixHD=pix2pixhd,ResViT=resvit,PLC-Net (F)=plcnet_f"

Run `--list` to print every model handle (paper preset + discovered) and exit.
Override any preset path with --manifest handle=/exact/path.
"""

import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')                     # headless
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import nibabel as nib
import numpy as np

# Accent used for every zoom box / inset frame / organ contour so the eye reads
# the box on the full slice and the magnified inset above it as one unit.
_ACCENT = '#f0a202'          # amber box + inset frame
_CONTOUR = '#22d3ee'         # cyan organ boundary (high contrast on grey CT)

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

# --------------------------------------------------------------------------- #
# TotalSegmentator label ids (mirrors config.TS_LABEL_MAP_JSON).              #
# --------------------------------------------------------------------------- #
_LABEL_JSON = _REPO / 'orgFeatXGB_CTPhase' / 'retrain_out_full' / 'ts_label_map_total.json'


def load_label_map() -> Dict[str, int]:
    try:
        return {k: int(v) for k, v in json.loads(_LABEL_JSON.read_text()).items()}
    except Exception as e:                 # keep going with the hardcoded fallback
        print(f'[warn] could not read {_LABEL_JSON} ({e}); using builtin ids')
        return {'aorta': 52, 'heart': 51, 'esophagus': 15, 'stomach': 6,
                'liver': 5, 'kidney_left': 3, 'kidney_right': 2,
                'iliac_artery_left': 65, 'iliac_artery_right': 66,
                'urinary_bladder': 21}


# Each anatomical level = (title, [label names whose presence marks the level],
# fallback depth quantile in [0,1] measured head->foot after orientation).
# The slice chosen is the one MAXIMISING the summed area of the level's labels,
# except pelvis which takes the LOWEST slice carrying its labels.
ANATOMICAL_LEVELS = [
    ('Aorta & cardia', ['heart', 'aorta', 'esophagus'], 0.20, 'max'),
    ('Mid-liver',      ['liver'],                        0.40, 'max'),
    ('Kidney',         ['kidney_left', 'kidney_right'],  0.55, 'max'),
    ('Pelvis',         ['urinary_bladder', 'iliac_artery_left',
                        'iliac_artery_right'],           0.85, 'low'),
]

# --------------------------------------------------------------------------- #
# Volume IO (mirrors dataset._load_vol: on-disk (X,Y,Z) -> (Z,H,W) axial).    #
# --------------------------------------------------------------------------- #

def load_axial(path: str) -> np.ndarray:
    """Return (D, H, W) float32; axis 0 is the axial (head->foot) slice axis."""
    vol = np.asanyarray(nib.load(path).dataobj).astype(np.float32)
    return np.transpose(vol, (2, 1, 0))


_PHASE_KW = {
    'non-contrast': ['noncontrast', 'non-contrast', 'pre', 'baseline',
                     'native', 'nc', 'nce', 'noncon'],
    'arterial':     ['arterial', 'art', 'early', 'phase1', 'p1'],
    'venous':       ['venous', 'portal', 'pv', 'phase2', 'p2', 'late'],
    'delayed':      ['delayed', 'delay', 'equilibrium', 'phase3', 'p3'],
}


def infer_phase(name: str) -> Optional[str]:
    n = name.lower()
    for phase, kws in _PHASE_KW.items():
        if any(k in n for k in kws):
            return phase
    return None


def find_ncct(real_path: str) -> Optional[str]:
    """Recover the NCCT volume paired with a CECT real_path: the non-contrast
    `_deeds` volume in the same case directory (segmentation masks excluded)."""
    real = Path(real_path)
    cands = [f for f in sorted(real.parent.glob('*_deeds.nii.gz'))
             if '_seg' not in f.name and f.resolve() != real.resolve()]
    if not cands:
        return None
    for f in cands:                         # prefer an explicit non-contrast tag
        if infer_phase(f.name) == 'non-contrast':
            return str(f)
    return str(cands[0])                     # else the only remaining volume


# --------------------------------------------------------------------------- #
# Manifest discovery (mirrors benchmark.discover + the --manifest adapters).  #
# --------------------------------------------------------------------------- #

_EXTERNAL_KEYWORDS = ('pix2pixhd', 'resvit', 'resnet', 'cyclegan', 'gan_ext',
                      'cytran', 'syndiff', 'gea_gan', 'dea_gan', 'eagan')


def model_family(name: str) -> str:
    base = name.split('/')[0].lower()
    if base.startswith('identity') or 'oracle' in base:
        return 'Reference / floor'
    if any(k in base for k in _EXTERNAL_KEYWORDS):
        return 'External baseline'
    if base.startswith('diff'):
        return 'Diffusion'
    return 'UNet + PatchGAN (this repo)'


def discover_runs(runs_dir: Path) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    if not runs_dir.is_dir():
        return found
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name.replace('literature_baseline_', '')
        pi = d / 'phase_infer'
        if (pi / 'manifest.csv').exists():
            found[name] = pi / 'manifest.csv'
            continue
        for m in sorted(pi.glob('*/manifest.csv')) if pi.is_dir() else []:
            found[f'{name}/{m.parent.name}'] = m       # multi-phase: one row/phase
    return found


def discover_bench(bench_dir: Path) -> Dict[str, Path]:
    """Every `**/manifest.csv` under a vendored-model repo. Model name = the repo
    directory (the child of bench_dir on the manifest's path)."""
    found: Dict[str, Path] = {}
    if not bench_dir.is_dir():
        return found
    for m in sorted(bench_dir.glob('**/manifest.csv')):
        try:
            rel = m.relative_to(bench_dir)
        except ValueError:
            continue
        # Two layouts coexist: <Repo>/results/vindr_nifti/... (name = repo) and a
        # shared results/vindr_<model>_nifti/... (name = <model>). Prefer the
        # embedded model tag so the shared-results models don't all collide on
        # the literal 'results' directory.
        name = rel.parts[0].lower()
        for p in rel.parts:
            mm = re.match(r'vindr_(.+?)_nifti$', p)
            if mm:
                name = mm.group(1).lower()
                break
        found.setdefault(name, m)            # first manifest wins per name
    return found


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def case_key(real_path: str) -> str:
    """Join key across models = resolved real CECT path (same as benchmark.py)."""
    return str(Path(real_path).resolve())


# --------------------------------------------------------------------------- #
# Slice selection                                                             #
# --------------------------------------------------------------------------- #

def choose_slices(mask: np.ndarray, labels: Dict[str, int]) -> Dict[str, int]:
    """Pick one axial index per anatomical level from the segmentation volume."""
    D = mask.shape[0]
    # Orient head->foot along axis 0 so depth quantiles are anatomically stable:
    # more of the liver+heart mass sits toward one end; if the aorta/heart center
    # of mass is past the midpoint, flip so it lands near the top (small index).
    picks: Dict[str, int] = {}
    for title, names, q, mode in ANATOMICAL_LEVELS:
        ids = [labels[n] for n in names if n in labels]
        area = np.zeros(D)
        for i in ids:
            area += (mask == i).sum(axis=(1, 2))
        if area.any():
            present = np.where(area > 0)[0]
            if mode == 'low':                # most-inferior slice carrying label
                # 'low' anatomically = largest axial index only if foot is at
                # high index; use the present extremum on the side away from heart
                idx = int(present.max()) if _foot_is_high(mask, labels) else int(present.min())
            else:                            # slice with the most of this organ
                idx = int(np.argmax(area))
        else:
            idx = int(round(q * (D - 1)))    # fallback: fixed depth quantile
            if not _foot_is_high(mask, labels):
                idx = (D - 1) - idx
        picks[title] = int(np.clip(idx, 0, D - 1))
    return picks


_FOOT_CACHE: Dict[int, bool] = {}


def _foot_is_high(mask: np.ndarray, labels: Dict[str, int]) -> bool:
    """True when larger axial index == more inferior (toward the feet). Decided
    by whether the heart's centroid sits at a smaller index than the bladder's;
    falls back to True (nibabel's usual axial ordering for these volumes)."""
    key = id(mask)
    if key in _FOOT_CACHE:
        return _FOOT_CACHE[key]
    res = True
    top_ids = [labels[n] for n in ('heart', 'aorta') if n in labels]
    bot_ids = [labels[n] for n in ('urinary_bladder', 'iliac_artery_left',
                                   'iliac_artery_right') if n in labels]

    def _centroid(ids):
        a = np.zeros(mask.shape[0])
        for i in ids:
            a += (mask == i).sum(axis=(1, 2))
        return float(np.average(np.arange(mask.shape[0]), weights=a)) if a.any() else None

    ct, cb = _centroid(top_ids), _centroid(bot_ids)
    if ct is not None and cb is not None:
        res = cb > ct
    _FOOT_CACHE[key] = res
    return res


# --------------------------------------------------------------------------- #
# Multi-plane (axial / coronal / sagittal) slice selection + extraction       #
# --------------------------------------------------------------------------- #

def choose_plane_indices(mask: np.ndarray, labels: Dict[str, int]) -> Dict[str, int]:
    """Coronal (y along H) and sagittal (x along W) indices through the great
    vessels/kidneys, so the reformats cut the enhancing structures. Falls back
    to the volume mid-plane when the mask is empty."""
    _, Hh, Ww = mask.shape

    def _centroid(ids, axis, default):
        m = np.isin(mask, ids) if ids else np.zeros_like(mask, bool)
        if not m.any():
            return default
        return int(round(np.where(m)[axis].mean()))

    aorta = [labels[n] for n in ('aorta',) if n in labels]
    kidney = [labels[n] for n in ('kidney_left', 'kidney_right') if n in labels]
    spine = [labels[n] for n in labels if n.startswith('vertebrae')]
    y = _centroid(aorta + kidney, 1, Hh // 2)          # coronal plane (H index)
    x = _centroid(aorta + spine, 2, Ww // 2)           # sagittal plane (W index)
    return {'coronal': int(np.clip(y, 0, Hh - 1)),
            'sagittal': int(np.clip(x, 0, Ww - 1))}


def extract_plane(vol: np.ndarray, plane: str, idx: int, foot_high: bool) -> np.ndarray:
    """Return a display-oriented 2-D slice from a (Z,H,W) volume.

    axial    -> (H,W), anterior up (flipud).
    coronal  -> (Z,W), head up (row 0 = head when foot_high).
    sagittal -> (Z,H), head up and anterior to the left.
    """
    if plane == 'axial':
        return np.flipud(vol[idx, :, :])
    if plane == 'coronal':
        sl = vol[:, idx, :]                             # (Z, W)
        return sl if foot_high else np.flipud(sl)
    if plane == 'sagittal':
        sl = vol[:, :, idx]                             # (Z, H)
        sl = sl if foot_high else np.flipud(sl)
        return sl[:, ::-1]                              # anterior to the left
    raise ValueError(f'unknown plane {plane!r}')


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #

def to_display(sl: np.ndarray) -> np.ndarray:
    """Orient one axial slice for radiological display (patient left on image
    right, anterior up). np.flipud puts row 0 (posterior) at the bottom."""
    return np.flipud(sl)


def auto_per_row(n_panels: int, rows_per_block: int) -> int:
    """Columns that make a block-wrapped panel grid as square as possible.

    A block is `rows_per_block` tall; wrapping `n_panels` into `per_row` columns
    gives ceil(n/per_row) blocks, so height ~= rows_per_block*ceil(n/per_row) and
    width ~= per_row. Setting the two equal gives per_row ~= sqrt(rows*n)."""
    pr = int(round(math.sqrt(max(rows_per_block, 1) * max(n_panels, 1))))
    return int(min(max(pr, 3), n_panels)) if n_panels else 1


def roi_box(mask_disp: np.ndarray, ids: Sequence[int], size: int,
            shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """(r0, r1, c0, c1) square zoom window centred on the level's organ(s).

    `mask_disp` is the segmentation slice in the SAME display orientation as the
    images (so the box lands on the organ in every panel). Falls back to the
    slice centre when none of `ids` are present."""
    m = np.isin(mask_disp, list(ids)) if len(ids) else np.zeros(shape, bool)
    if m.any():
        rr, cc = np.nonzero(m)
        r, c = int(round(rr.mean())), int(round(cc.mean()))
    else:
        r, c = shape[0] // 2, shape[1] // 2
    sz_r, sz_c = min(size, shape[0]), min(size, shape[1])
    r0 = int(np.clip(r - sz_r // 2, 0, shape[0] - sz_r))
    c0 = int(np.clip(c - sz_c // 2, 0, shape[1] - sz_c))
    return r0, r0 + sz_r, c0, c0 + sz_c


def window(img: np.ndarray, center: float, width: float) -> np.ndarray:
    lo, hi = center - width / 2.0, center + width / 2.0
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def diff_map(a_hu: np.ndarray, b_hu: np.ndarray,
             clip_lo: float, clip_hi: float) -> np.ndarray:
    """Difference of two HU slices AFTER clamping both to [clip_lo, clip_hi].

    The generators are trained on HU clipped to a soft-tissue domain (default
    [-200, 400]), so in a synthetic volume air and lung sit at the clip floor
    while in the real scan they are ~-1000/-700. A raw synth-real subtraction is
    then +800 HU across the whole background and saturates the colormap, hiding
    the soft-tissue detail. Clamping both operands to the same domain makes
    air/lung/bone cancel to ~0 and leaves only the differences that fall inside
    the window we actually care about (parenchyma + vessels)."""
    return np.clip(a_hu, clip_lo, clip_hi) - np.clip(b_hu, clip_lo, clip_hi)


def render_level(level_title: str, sidx: int,
                 columns: List[Tuple[str, np.ndarray]],
                 ncct: np.ndarray, cect: np.ndarray,
                 mask_disp: np.ndarray, roi_ids: Sequence[int], zoom_size: int,
                 show_zoom: bool, show_diff: bool, show_contour: bool,
                 win_c: float, win_w: float, diff_max: float,
                 clip_lo: float, clip_hi: float, diff_mode: str,
                 per_row: int, out_path: Path, case_id: str) -> None:
    """One square-ish figure per anatomical level.

    Each of NCCT | CECT | every model is one *panel*; a panel stacks up to three
    rows — an organ-boundary ZOOM (magnified crop over the level's organ, with
    the true boundary drawn on so blur / mis-registration is obvious), the full
    IMAGE (with the zoom box marked), and the synth−ref DIFFERENCE map. Panels
    are wrapped into blocks of `per_row` columns so many models still lay out
    close to square, with each slice drawn large enough to read. Model
    subtraction is synth−CECT (error) or synth−NCCT (enhancement); the NCCT panel
    always shows the CECT−NCCT target."""
    cect_d = to_display(cect[sidx])
    ncct_d = to_display(ncct[sidx]) if ncct is not None else None
    ref_d = ncct_d if (diff_mode == 'enhancement' and ncct_d is not None) else cect_d

    # panel = (label, image_display_or_None, (minuend, subtrahend)_or_None)
    panels: List[Tuple[str, Optional[np.ndarray],
                       Optional[Tuple[np.ndarray, np.ndarray]]]] = []
    panels.append(('NCCT', ncct_d, (cect_d, ncct_d) if ncct_d is not None else None))
    panels.append(('CECT (real)', cect_d, None))
    for label, vol in columns:
        syn_d = to_display(vol[sidx])
        panels.append((label, syn_d, (syn_d, ref_d)))

    roles = (['zoom'] if show_zoom else []) + ['image'] + (['diff'] if show_diff else [])
    rpb = len(roles)
    npan = len(panels)
    pr = min(per_row, npan) if per_row and per_row > 0 else auto_per_row(npan, rpb)
    nblocks = math.ceil(npan / pr)
    ncols, nrows = min(pr, npan), rpb * nblocks

    r0, r1, c0, c1 = roi_box(mask_disp, roi_ids, zoom_size, cect_d.shape)
    organ_crop = (np.isin(mask_disp[r0:r1, c0:c1], list(roi_ids))
                  if show_contour and len(roi_ids) else None)

    cell = 2.6
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(cell * ncols, cell * nrows + 0.4),
                             gridspec_kw={'wspace': 0.02, 'hspace': 0.05})
    for ax in axes.ravel():
        ax.axis('off')

    def _gray(ax, arr):
        ax.imshow(window(arr, win_c, win_w), cmap='gray', vmin=0, vmax=1,
                  interpolation='nearest')

    im = None
    for k, (label, img_d, dspec) in enumerate(panels):
        b, j = divmod(k, pr)
        base, ri = b * rpb, 0
        if show_zoom:                                   # magnified organ crop
            axz = axes[base + ri, j]; ri += 1
            if img_d is not None:
                _gray(axz, img_d[r0:r1, c0:c1])
                if organ_crop is not None and organ_crop.any():
                    axz.contour(organ_crop.astype(float), levels=[0.5],
                                colors=[_CONTOUR], linewidths=1.1)
                axz.axis('on'); axz.set_xticks([]); axz.set_yticks([])
                for s in axz.spines.values():
                    s.set_color(_ACCENT); s.set_linewidth(1.3)
            axz.set_title(label, fontsize=11, pad=3)
        axi = axes[base + ri, j]; ri += 1               # full slice + zoom box
        if img_d is not None:
            _gray(axi, img_d)
            axi.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0,
                                    fill=False, ec=_ACCENT, lw=1.1))
        if not show_zoom:
            axi.set_title(label, fontsize=11, pad=3)
        if show_diff:                                   # synth − ref residual
            axd = axes[base + ri, j]
            if dspec is not None:
                im = axd.imshow(diff_map(dspec[0], dspec[1], clip_lo, clip_hi),
                                cmap='bwr', vmin=-diff_max, vmax=diff_max,
                                interpolation='nearest')

    # role label down the left edge of every block
    for b in range(nblocks):
        for ri, role in enumerate(roles):
            ax = axes[b * rpb + ri, 0]
            ax.axis('on'); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                if not (role == 'zoom'):
                    s.set_visible(False)
            ax.set_ylabel(role, fontsize=10, labelpad=4)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.012,
                            pad=0.01, aspect=40)
        cbar.set_label('HU difference', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(f'{level_title}  |  case {case_id[:24]}…  |  axial slice {sidx}',
                 fontsize=13, y=0.997)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out_path}')


def render_multiplane(planes: List[Tuple[str, int]],
                      columns: List[Tuple[str, np.ndarray]],
                      ncct: np.ndarray, cect: np.ndarray, foot_high: bool,
                      win_c: float, win_w: float, diff_max: float,
                      clip_lo: float, clip_hi: float, diff_mode: str,
                      per_row: int, out_path: Path, case_id: str) -> None:
    """One figure spanning several planes (e.g. axial + coronal). Each of
    NCCT | CECT | every model is a *panel* holding, per plane, an image row over
    a synth−ref subtraction row. Panels wrap into blocks of `per_row` columns so
    the multi-view stays close to square instead of stretching into one long
    strip. The NCCT panel's subtraction is always the CECT−NCCT target."""
    # Pre-extract every plane for every panel: panel = (label, [(img, diff|None)])
    panels: List[Tuple[str, List[Tuple[Optional[np.ndarray],
                                       Optional[Tuple[np.ndarray, np.ndarray]]]]]] = []

    def _panel(label, vol, is_ncct=False, is_ref=False):
        rows = []
        for plane, idx in planes:
            cect_d = extract_plane(cect, plane, idx, foot_high)
            ncct_d = extract_plane(ncct, plane, idx, foot_high) if ncct is not None else None
            ref_d = ncct_d if (diff_mode == 'enhancement' and ncct_d is not None) else cect_d
            if is_ncct:
                img = ncct_d
                dif = (cect_d, ncct_d) if ncct_d is not None else None
            elif is_ref:
                img, dif = cect_d, None
            else:
                img = extract_plane(vol, plane, idx, foot_high)
                dif = (img, ref_d)
            rows.append((img, dif))
        return label, rows

    panels.append(_panel('NCCT', None, is_ncct=True))
    panels.append(_panel('CECT (real)', None, is_ref=True))
    for label, vol in columns:
        panels.append(_panel(label, vol))

    rpb = 2 * len(planes)                          # image+diff per plane
    npan = len(panels)
    pr = min(per_row, npan) if per_row and per_row > 0 else auto_per_row(npan, rpb)
    nblocks = math.ceil(npan / pr)
    ncols, nrows = min(pr, npan), rpb * nblocks

    cell = 2.4
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(cell * ncols, cell * nrows + 0.4),
                             gridspec_kw={'wspace': 0.02, 'hspace': 0.05})
    for ax in axes.ravel():
        ax.axis('off')

    def _img(ax, arr):
        # aspect='auto' fills the cell: coronal/sagittal reformats have far fewer
        # slices than in-plane pixels, so 'equal' would render a thin sliver.
        ax.imshow(window(arr, win_c, win_w), cmap='gray', vmin=0, vmax=1,
                  aspect='auto', interpolation='nearest')

    im = None
    for k, (label, rows) in enumerate(panels):
        b, j = divmod(k, pr)
        base = b * rpb
        for p, (img, dif) in enumerate(rows):
            r_img, r_dif = base + 2 * p, base + 2 * p + 1
            if img is not None:
                _img(axes[r_img, j], img)
            if dif is not None:
                im = axes[r_dif, j].imshow(diff_map(dif[0], dif[1], clip_lo, clip_hi),
                                           cmap='bwr', vmin=-diff_max, vmax=diff_max,
                                           aspect='auto', interpolation='nearest')
        axes[base, j].set_title(label, fontsize=11, pad=3)

    # plane / row labels down the left edge of every block
    for b in range(nblocks):
        for p, (plane, _idx) in enumerate(planes):
            for ri, role in ((2 * p, f'{plane}\nimage'), (2 * p + 1, f'{plane}\ndiff')):
                ax = axes[b * rpb + ri, 0]
                ax.axis('on'); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.set_ylabel(role, fontsize=9, labelpad=4)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.010,
                            pad=0.01, aspect=50)
        cbar.set_label('HU difference', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    view = ' + '.join(pl for pl, _ in planes)
    fig.suptitle(f'{view} views  |  case {case_id[:24]}…', fontsize=13, y=0.997)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out_path}')


# --------------------------------------------------------------------------- #
# Set assembly                                                                #
# --------------------------------------------------------------------------- #

# PLC-Net paper preset. `PAPER_MODELS` maps each handle to (root, relative
# manifest path) where root is 'runs' (--runs_dir) or 'bench' (--bench_dir), so
# the two default sets below resolve straight off --runs_dir/--bench_dir with no
# --manifest flags. The two PLC-Net entries carry a {infer} slot filled from
# --plcnet_e_infer / --plcnet_f_infer and a {phase} slot from --phase. Override
# any path with --manifest handle=/exact/path, or replace whole sets with
# --comparison / --ablation.
PAPER_MODELS = {
    # competing / published methods
    'pix2pixhd':    ('runs',  'literature_baseline_pix2pixhd_baseline/phase_infer/manifest.csv'),
    'cyclegan':     ('bench', 'CycleGAN/results/vindr_nifti/manifest.csv'),
    'resvit':       ('bench', 'ResViT/results/vindr_nifti/manifest.csv'),
    'cytran':       ('bench', 'CyTran/results/vindr_nifti/manifest.csv'),
    'geagan':       ('bench', 'results/vindr_gea_gan_nifti/manifest.csv'),
    'deagan':       ('bench', 'results/vindr_dea_gan_nifti/manifest.csv'),
    'swinunetr':    ('bench', 'results/vindr_swinunetr_s5_nifti/manifest.csv'),
    'transunet':    ('bench', 'results/vindr_transunet_nifti/manifest.csv'),
    # diffusion arms
    'ddpm_x0':      ('runs',  'literature_baseline_diff_x0/phase_infer/manifest.csv'),
    'diff_v':       ('runs',  'literature_baseline_diff_v/phase_infer/manifest.csv'),
    'diff_v_organ': ('runs',  'literature_baseline_diff_v_organ/phase_infer/manifest.csv'),
    'diff_hnll':    ('runs',  'literature_baseline_diff_hetero_nll/phase_infer/manifest.csv'),
    # UNet + PatchGAN reference points
    'l1_only':      ('runs',  'literature_baseline_l1_only/phase_infer/manifest.csv'),
    'l1gn':         ('runs',  'literature_baseline_l1_organ_groupnorm_s43/phase_infer/manifest.csv'),
    # ours (proposed)
    'plcnet_e':     ('runs',  'literature_baseline_multiphase_film_level_adv/{infer}/{phase}/manifest.csv'),
    'plcnet_f':     ('runs',  'literature_baseline_multiphase_film_level_adv_slices11/{infer}/{phase}/manifest.csv'),
}

# Table 3 — comparison with published methods.
DEFAULT_COMPARISON = [
    ('pix2pixHD',   'pix2pixhd'),
    ('CycleGAN',    'cyclegan'),
    ('ResViT',      'resvit'),
    ('CyTran',      'cytran'),
    ('gEa-GAN',     'geagan'),
    ('dEa-GAN',     'deagan'),
    ('SwinUNETR',   'swinunetr'),
    ('TransUNet',   'transunet'),
    ('DDPM (x0)',   'ddpm_x0'),
    ('PLC-Net (E)', 'plcnet_e'),
    ('PLC-Net (F)', 'plcnet_f'),
]
# Table 6 — comparison with other generator architectures / diffusion arms.
DEFAULT_ABLATION = [
    ('UNet+L1',      'l1_only'),
    ('UNet+L1+adv',  'l1gn'),
    ('DDPM x0',      'ddpm_x0'),
    ('DDPM v',       'diff_v'),
    ('DDPM v+adv',   'diff_v_organ'),
    ('DDPM het-var', 'diff_hnll'),
    ('PLC-Net (E)',  'plcnet_e'),
    ('PLC-Net (F)',  'plcnet_f'),
]


def resolve_paper_models(runs_dir: Path, bench_dir: Path,
                         e_infer: str, f_infer: str, phase: str) -> Dict[str, Path]:
    """Resolve the PAPER_MODELS registry to concrete manifest paths under the
    given run/bench dirs. Non-existent paths are still returned (the loader warns
    and skips), so a partially-inferred tree degrades gracefully."""
    roots = {'runs': runs_dir, 'bench': bench_dir}
    infer_for = {'plcnet_e': e_infer, 'plcnet_f': f_infer}
    out: Dict[str, Path] = {}
    for handle, (root, rel) in PAPER_MODELS.items():
        rel = rel.format(infer=infer_for.get(handle, ''), phase=phase)
        out[handle] = roots[root] / rel
    return out


def parse_spec(spec: str) -> List[Tuple[str, str]]:
    """'Label=key,key2,Other=key3' -> [(label, key), ...]. A bare key uses the
    key itself as its display label."""
    out = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '=' in part:
            label, key = part.split('=', 1)
            out.append((label.strip(), key.strip()))
        else:
            out.append((part, part))
    return out


def resolve_members(members: List[Tuple[str, str]],
                    manifests: Dict[str, Path]) -> List[Tuple[str, str]]:
    """Keep only members present in the discovered manifests; warn on the rest."""
    keep, miss = [], []
    for label, key in members:
        if key in manifests:
            keep.append((label, key))
        else:
            miss.append(key)
    if miss:
        print(f'  [skip] not discovered: {", ".join(miss)}')
    return keep


def build_case_index(manifests: Dict[str, Path]) -> Dict[str, Dict[str, Dict]]:
    """{case_key: {model_name: manifest_row}} across all discovered models."""
    index: Dict[str, Dict[str, Dict]] = {}
    for name, mpath in manifests.items():
        # Preset handles are added for every paper model; a tree that only has
        # some of them yields many missing paths, which are not worth warning
        # about. Only a path that exists but fails to parse is a real problem.
        if not mpath.exists():
            continue
        try:
            rows = read_manifest(mpath)
        except Exception as e:
            print(f'  [warn] cannot read {name} manifest ({e})')
            continue
        for r in rows:
            if not r.get('real_path'):
                continue
            index.setdefault(case_key(r['real_path']), {})[name] = r
    return index


def pick_case(index: Dict[str, Dict[str, Dict]], need: List[str],
              rng: random.Random, forced: Optional[str]) -> Optional[str]:
    """A case covered by every required model. `forced` matches a case_id substring."""
    ok = [ck for ck, models in index.items() if all(n in models for n in need)]
    if forced:
        ok = [ck for ck in ok if forced in ck]
        if not ok:
            print(f'  [error] --case {forced!r} matches no case covered by all '
                  f'required models')
            return None
    if not ok:
        return None
    return rng.choice(sorted(ok))


def run_set(set_name: str, members: List[Tuple[str, str]],
            manifests: Dict[str, Path], index: Dict[str, Dict[str, Dict]],
            labels: Dict[str, int], args, rng: random.Random) -> None:
    ours_label, ours_key = 'Ours', args.ours
    members = list(members)
    if ours_key:
        members = [(l, k) for (l, k) in members if k != ours_key]
        members.append((ours_label, ours_key))     # our method last, as in the paper
    members = resolve_members(members, manifests)
    if not members:
        print(f'[{set_name}] no members discovered — skipped')
        return

    need = [k for _, k in members]
    case = pick_case(index, need, rng, args.case)
    if case is None:
        print(f'[{set_name}] no case is covered by ALL of: {", ".join(need)}\n'
              f'          relax the member list or pass --case; skipped')
        return

    row0 = index[case][need[0]]
    real_path, mask_path = row0['real_path'], row0['mask_path']
    case_id = Path(real_path).parent.name
    print(f'[{set_name}] case {case_id}  ({len(members)} models)')

    cect = load_axial(real_path)
    ncct_path = find_ncct(real_path)
    ncct = load_axial(ncct_path) if ncct_path else None
    if ncct is None:
        print('  [warn] NCCT source not found; NCCT column will be blank')
    try:
        mask = load_axial(mask_path)
    except Exception as e:
        print(f'  [warn] mask unreadable ({e}); using depth-quantile slices')
        mask = np.zeros_like(cect)

    slices = choose_slices(mask, labels)

    # Load every model's synthetic volume once; reuse across all levels.
    cols: List[Tuple[str, np.ndarray]] = []
    for label, key in members:
        gen_path = index[case][key]['gen_path']
        try:
            vol = load_axial(gen_path)
        except Exception as e:
            print(f'  [warn] {label} ({key}) gen unreadable ({e}); skipped')
            continue
        if vol.shape != cect.shape:
            print(f'  [warn] {label} shape {vol.shape} != CECT {cect.shape}; skipped')
            continue
        cols.append((label, vol))

    for level_title, _names, _q, _mode in ANATOMICAL_LEVELS:
        sidx = slices[level_title]
        roi_ids = [labels[n] for n in _names if n in labels]
        mask_disp = to_display(mask[sidx])
        safe = re.sub(r'[^a-z0-9]+', '_', level_title.lower()).strip('_')
        out = Path(args.out_dir) / set_name / f'{set_name}_{safe}_slice{sidx}.png'
        render_level(level_title, sidx, cols, ncct, cect,
                     mask_disp, roi_ids, args.zoom_size,
                     args.zoom, args.diff, args.zoom_contour,
                     args.win_center, args.win_width, args.diff_max,
                     args.diff_clip_lo, args.diff_clip_hi, args.diff_mode,
                     args.per_row, out, case_id)

    # Multi-plane figure (axial + coronal [+ sagittal]) — one panel, all planes.
    if args.multiplane:
        planes_want = [p.strip().lower() for p in args.planes.split(',') if p.strip()]
        cs = choose_plane_indices(mask, labels)
        axial_idx = slices.get(args.axial_level, slices['Mid-liver'])
        idx_for = {'axial': axial_idx,
                   'coronal': cs['coronal'], 'sagittal': cs['sagittal']}
        foot_high = _foot_is_high(mask, labels)
        planes = [(p, idx_for[p]) for p in planes_want if p in idx_for]
        if planes:
            tag = '_'.join(p for p, _ in planes)
            out = Path(args.out_dir) / set_name / f'{set_name}_multiplane_{tag}.png'
            render_multiplane(planes, cols, ncct, cect, foot_high,
                              args.win_center, args.win_width, args.diff_max,
                              args.diff_clip_lo, args.diff_clip_hi, args.diff_mode,
                              args.per_row, out, case_id)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description='Qualitative NCCT->CECT comparison / ablation figures',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--runs_dir', default='../out_synthesis_train',
                    help='our trained runs (<run>/phase_infer[/<phase>]/manifest.csv)')
    ap.add_argument('--bench_dir', default='../bench_ncct2cect',
                    help='vendored competing-model repos (**/manifest.csv)')
    ap.add_argument('--manifest', action='append', default=[],
                    metavar='name=path.csv',
                    help='extra manifest(s); repeatable')
    ap.add_argument('--set', choices=['comparison', 'ablation', 'both'],
                    default='both')
    ap.add_argument('--ours', default='',
                    help='handle of a proposed method to append to every set; '
                         'empty by default because the paper presets already '
                         'list PLC-Net (E)/(F) as members')
    ap.add_argument('--comparison', default=None,
                    help='override comparison members: "Label=key,key2,..."')
    ap.add_argument('--ablation', default=None,
                    help='override ablation members: "Label=key,..."')
    ap.add_argument('--plcnet_e_infer', default='infer_best',
                    help='inference subfolder for PLC-Net (E)')
    ap.add_argument('--plcnet_f_infer', default='infer_popphase',
                    help='inference subfolder for PLC-Net (F)')
    ap.add_argument('--phase', default='venous',
                    help='target phase subfolder for the multiphase PLC-Net runs')
    ap.add_argument('--case', default=None,
                    help='force a case (StudyInstanceUID substring); default random')
    ap.add_argument('--seed', type=int, default=0, help='RNG seed for case pick')
    ap.add_argument('--out_dir', default='analysis/figures')
    ap.add_argument('--win_center', type=float, default=40.0,
                    help='display window center (HU); 40/400 = soft-tissue')
    ap.add_argument('--win_width', type=float, default=400.0,
                    help='display window width (HU)')
    ap.add_argument('--diff_max', type=float, default=100.0,
                    help='subtraction colormap range is +/- this (HU); lower = '
                         'more sensitive to soft-tissue detail')
    ap.add_argument('--diff_clip_lo', type=float, default=-200.0,
                    help='clamp both volumes to >= this HU before differencing '
                         '(the model training floor; kills air/lung saturation)')
    ap.add_argument('--diff_clip_hi', type=float, default=400.0,
                    help='clamp both volumes to <= this HU before differencing')
    ap.add_argument('--diff_mode', choices=['error', 'enhancement'],
                    default='error',
                    help="'error' = synth-CECT residual; 'enhancement' = "
                         "synth-NCCT (contrast the model added)")
    ap.add_argument('--per_row', type=int, default=0,
                    help='panels per block before wrapping to a new row-block; '
                         '0 = auto (chosen to make the figure roughly square)')
    ap.add_argument('--zoom', action=argparse.BooleanOptionalAction, default=True,
                    help='add a magnified organ-boundary inset above each panel')
    ap.add_argument('--zoom_size', type=int, default=96,
                    help='side length (in voxels) of the organ zoom window')
    ap.add_argument('--zoom_contour', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='draw the real organ boundary on every zoom inset so '
                         'boundary blur / mis-registration is visible per model')
    ap.add_argument('--diff', action=argparse.BooleanOptionalAction, default=True,
                    help='include the synth-ref subtraction row in each figure')
    ap.add_argument('--multiplane', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='also emit a combined multi-view figure per set')
    ap.add_argument('--planes', default='axial,coronal',
                    help='planes for the multi-view figure (of axial,coronal,sagittal)')
    ap.add_argument('--axial_level', default='Mid-liver',
                    choices=[t for t, *_ in ANATOMICAL_LEVELS],
                    help="which anatomical level supplies the multi-view axial row")
    ap.add_argument('--list', action='store_true',
                    help='print discovered models + table family and exit')
    args = ap.parse_args()

    manifests: Dict[str, Path] = {}
    # Paper-preset handles first, then discovery (which may add more runs under
    # their own names), then explicit --manifest wins over everything.
    manifests.update(resolve_paper_models(Path(args.runs_dir), Path(args.bench_dir),
                                          args.plcnet_e_infer, args.plcnet_f_infer,
                                          args.phase))
    manifests.update(discover_runs(Path(args.runs_dir)))
    manifests.update(discover_bench(Path(args.bench_dir)))
    for spec in args.manifest:
        if '=' not in spec:
            print(f'[error] --manifest expects name=path, got {spec!r}')
            return 2
        name, path = spec.split('=', 1)
        manifests[name.strip()] = Path(path.strip())

    if not manifests:
        print('[error] no manifests discovered. Check --runs_dir / --bench_dir, '
              'and run infer_volume.py / the vendored repos first.')
        return 2

    if args.list:
        print(f'Discovered {len(manifests)} models:\n')
        for name in sorted(manifests):
            print(f'  {model_family(name):32s}  {name:34s}  {manifests[name]}')
        return 0

    labels = load_label_map()
    index = build_case_index(manifests)
    if not index:
        print('[error] manifests contained no readable rows.')
        return 2

    comp = parse_spec(args.comparison) if args.comparison else DEFAULT_COMPARISON
    abla = parse_spec(args.ablation) if args.ablation else DEFAULT_ABLATION

    # One RNG, seeded once, so --set both draws the SAME case for both figures
    # whenever their required-model sets overlap on a case.
    if args.set in ('comparison', 'both'):
        run_set('comparison', comp, manifests, index, labels, args,
                random.Random(args.seed))
    if args.set in ('ablation', 'both'):
        run_set('ablation', abla, manifests, index, labels, args,
                random.Random(args.seed))
    return 0


if __name__ == '__main__':
    sys.exit(main())
