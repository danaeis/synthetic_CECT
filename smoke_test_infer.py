"""
CPU-only smoke test for infer_volume.py (Phase 1 volume inference).

Two parts, no GPU / real data / trained weights needed:
  A) infer_volume() on a tiny volume with an untrained generator — checks full
     voxel coverage (no gaps), output shape == input shape, HU range sane, for
     BOTH 2-D (patch_depth=1) and 3-D (patch_depth>1) paths.
  B) the run() driver end-to-end on a synthetic scenario (fake data dir + labels
     + run_config.json + a saved generator) — checks it writes synthetic NIfTIs
     and a manifest whose gen/real/mask volumes are all the SAME shape (the exact
     requirement phase_eval.py enforces).

Usage:  python smoke_test_infer.py
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from models import UNetGenerator
from infer_volume import infer_volume, load_generator, run


def check(cond, name, detail=''):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f'  -- {detail}' if detail and not cond else ''))
    return bool(cond)


def _save(path, arr, affine=None):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine if affine is not None else np.eye(4)), str(path))


def part_a():
    ok = True
    hu_min, hu_max = -200.0, 400.0

    # 2-D: volume (D=6, H=64, W=64), patch 32, patch_depth=1
    G2 = UNetGenerator(dims=2, base_channels=8, dropout=0.0).eval()
    cfg2 = dict(dims=2, patch_depth=1, patch_size=32, overlap=0.5, hu_min=hu_min, hu_max=hu_max)
    vol2 = np.random.uniform(-500, 500, size=(6, 64, 64)).astype(np.float32)
    syn2 = infer_volume(G2, vol2, cfg2, device='cpu', batch_size=8)
    ok &= check(syn2.shape == vol2.shape, '2D: output shape == input', f'{syn2.shape} vs {vol2.shape}')
    ok &= check(not np.isnan(syn2).any(), '2D: no NaN (full coverage, no div-by-zero)')
    ok &= check(syn2.min() >= hu_min - 1 and syn2.max() <= hu_max + 1,
                '2D: output within HU window', f'[{syn2.min():.0f},{syn2.max():.0f}]')

    # 3-D: volume (D=8, H=32, W=32), patch 32, patch_depth=4
    G3 = UNetGenerator(dims=3, base_channels=8, dropout=0.0).eval()
    cfg3 = dict(dims=3, patch_depth=4, patch_size=32, overlap=0.5, hu_min=hu_min, hu_max=hu_max)
    vol3 = np.random.uniform(-500, 500, size=(8, 32, 32)).astype(np.float32)
    syn3 = infer_volume(G3, vol3, cfg3, device='cpu', batch_size=4)
    ok &= check(syn3.shape == vol3.shape, '3D: output shape == input', f'{syn3.shape} vs {vol3.shape}')
    ok &= check(not np.isnan(syn3).any(), '3D: no NaN (full coverage)')

    # tiny-volume padding path (D<patch_depth for 3D)
    vol3s = np.random.uniform(-500, 500, size=(2, 32, 32)).astype(np.float32)
    syn3s = infer_volume(G3, vol3s, cfg3, device='cpu', batch_size=4)
    ok &= check(syn3s.shape == vol3s.shape, '3D: sub-patch depth padded+cropped back', f'{syn3s.shape}')
    return ok


def part_b():
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix='infer_smoke_'))
    try:
        data = tmp / 'deeds'; data.mkdir()
        # 4 studies so a val+test split is non-empty; each: NC + venous + venous seg
        rows = []
        XYZ = (48, 48, 6)   # -> _load_vol -> (D=6,H=48,W=48)
        for i in range(4):
            st = f'study{i}'
            cdir = data / st; cdir.mkdir()
            for ser, phase in [(f'{st}_nc', 'non-contrast'), (f'{st}_pv', 'venous')]:
                base = f'{st}_{ser}_deeds'
                _save(cdir / f'{base}.nii.gz', np.random.uniform(-500, 500, XYZ))
                rows.append({'StudyInstanceUID': st, 'SeriesInstanceUID': ser, 'Label': phase})
                if phase == 'venous':   # seg_full mask only needed for the target
                    _save(cdir / f'{base}_seg_full.nii.gz',
                          np.random.randint(0, 5, XYZ).astype(np.float32))
        with open(data.parent / 'labels.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['StudyInstanceUID', 'SeriesInstanceUID', 'Label'])
            w.writeheader(); w.writerows(rows)

        # scenario dir: run_config.json + best_model.pth
        sdir = tmp / 'scenario'; sdir.mkdir()
        cfg = dict(dims=2, patch_depth=1, patch_size=32, overlap=0.5, hu_min=-200, hu_max=400,
                   generator_base_channels=8, generator_dropout=0.0, target_phase='venous',
                   data_dir=str(data), labels_csv=str(data.parent / 'labels.csv'),
                   file_tag='_deeds', seg_suffix='_seg_full',
                   val_split=0.25, test_split=0.25, seed=42)
        (sdir / 'run_config.json').write_text(json.dumps(cfg))
        G = UNetGenerator(dims=2, base_channels=8, dropout=0.0)
        torch.save({'G_state': G.state_dict(), 'epoch': 1}, sdir / 'best_model.pth')

        out = sdir / 'phase_infer'
        run(str(sdir), 'both', str(out), 'best_model.pth', batch_size=8, device='cpu')

        manifest = out / 'manifest.csv'
        ok &= check(manifest.exists(), 'driver wrote manifest.csv')
        with open(manifest) as f:
            mrows = list(csv.DictReader(f))
        ok &= check(len(mrows) > 0, 'manifest has rows', f'{len(mrows)}')
        # the phase_eval requirement: gen/real/mask all same shape
        allmatch = True
        for r in mrows:
            g = nib.load(r['gen_path']).get_fdata().shape
            rl = nib.load(r['real_path']).get_fdata().shape
            mk = nib.load(r['mask_path']).get_fdata().shape
            if not (g == rl == mk):
                allmatch = False
                print(f"    shape mismatch: gen{g} real{rl} mask{mk}")
        ok &= check(allmatch, 'gen/real/mask shapes match (phase_eval requirement)')
        # target_phase written as name
        ok &= check(all(r['target_phase'] == 'venous' for r in mrows), "manifest target_phase='venous'")
    finally:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)
    return ok


def main():
    print("--- Part A: infer_volume coverage/shape (2D + 3D) ---")
    a = part_a()
    print("\n--- Part B: run() driver end-to-end ---")
    b = part_b()
    ok = a and b
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
