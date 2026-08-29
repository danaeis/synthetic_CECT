#!/usr/bin/env python
"""
Does a level-conditioned model actually DELIVER the level it was asked for?

This is the evidence for the controllability claim, and it is the one number in
the project that cannot leak: it never reads the real CECT's level. You ask the
model for a level, you measure what it produced, and you compare those two. The
ground truth is the REQUEST, not the patient.

Run infer_volume.py once per requested level first, e.g.

    for Z in -1.5 -1 -0.5 0 0.5 1 1.5; do
      python infer_volume.py --scenario_dir <run> --ckpt_name <ckpt> \
        --level_mode fixed_z --level_z $Z --out_dir <run>/infer_z$Z
    done

then point this at those directories:

    python scripts/level_calibration.py \
        --run ../out_synthesis_train/literature_baseline_multiphase_film_level_adv \
        --pattern 'infer_z*' --phase venous --out analysis/level_calibration

USE fixed_z, NOT fixed, for a multi-organ model. `--level L` applies one HU
number to every conditioned organ, and those organs occupy different HU ranges:
with 8 organs, --level 350 puts the aorta at a sensible z=+1.4 but the
gallbladder at z=+25.9, far outside anything the model ever saw. fixed_z shifts
each organ by the same number of ITS OWN standard deviations, so the sweep
measures controllability instead of extrapolation behaviour.

A perfectly controllable model gives slope 1.0: ask for one sd more enhancement,
get one sd more. Slope < 1 means the request is being damped (the model partly
ignores you); slope > 1 means it overshoots. The intercept says whether z=0
reproduces the population mean, which it must, since z=0 IS --level_mode
population.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:                                          # pragma: no cover
    raise SystemExit("level_calibration.py needs nibabel (pip install nibabel)")


def _load(p):
    return np.asanyarray(nib.load(p).dataobj).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True,
                    help='scenario dir (holds run_config.json and the infer_* dirs)')
    ap.add_argument('--pattern', default='infer_z*',
                    help="glob for the swept inference dirs (default 'infer_z*')")
    ap.add_argument('--phase', default='venous',
                    help='which phase subdir to read for a multiphase run; '
                         'ignored when the manifest sits directly in the dir')
    ap.add_argument('--levels_json', type=Path, default=Path('splits/levels.json'))
    ap.add_argument('--label_map', type=Path,
                    default=Path(__file__).resolve().parent.parent /
                            'orgFeatXGB_CTPhase' / 'retrain_out_full' /
                            'ts_label_map_total.json')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--no_plot', action='store_true')
    a = ap.parse_args()

    cfg = json.loads((a.run / 'run_config.json').read_text())
    cond_organs = list(cfg.get('cond_organs') or [])
    if not cond_organs:
        raise SystemExit(f"{a.run.name} has no cond_organs — nothing to calibrate")
    name_to_id = json.loads(a.label_map.read_text())
    ldb = json.loads(a.levels_json.read_text())
    st = ldb['standardize']
    idx = {o: ldb['organs'].index(o) for o in cond_organs}

    # requested z is parsed from the directory name, so the sweep is
    # self-describing and the script needs no second list of levels to stay in
    # step with. 'infer_z-1.5' -> -1.5
    dirs = sorted(a.run.glob(a.pattern))
    if not dirs:
        raise SystemExit(f"no directories match {a.run}/{a.pattern}")

    per_z = {}
    for d in dirs:
        m = re.search(r'(-?\d+(?:\.\d+)?)\s*$', d.name)
        if not m:
            print(f"  skip {d.name}: no z in the name")
            continue
        z_req = float(m.group(1))
        mf = d / a.phase / 'manifest.csv'
        if not mf.exists():
            mf = d / 'manifest.csv'
        if not mf.exists():
            print(f"  skip {d.name}: no manifest.csv (looked in {a.phase}/ and ./)")
            continue
        rows = list(csv.DictReader(mf.open()))
        got = {o: [] for o in cond_organs}
        for r in rows:
            try:
                gen = _load(r['gen_path']); mask = _load(r['mask_path'])
            except Exception as e:                           # noqa: BLE001
                print(f"    {d.name}: skip a case ({e})")
                continue
            if gen.shape != mask.shape:
                continue
            lbl = np.round(mask).astype(np.int32)
            for o in cond_organs:
                sel = lbl == name_to_id[o]
                if int(sel.sum()) >= 64:
                    got[o].append(float(np.median(gen[sel])))
        per_z[z_req] = {o: (float(np.mean(v)) if v else float('nan'))
                        for o, v in got.items()}
        print(f"  {d.name}: z={z_req:+.2f}, {len(rows)} cases")

    if len(per_z) < 3:
        raise SystemExit(f"need at least 3 swept levels to fit a slope, got {len(per_z)}")

    zs = np.array(sorted(per_z))
    print(f"\n{'=' * 78}\nCALIBRATION — requested vs achieved, per conditioned organ\n{'=' * 78}")
    print(f"  {'organ':<32}{'slope':>8}{'r2':>7}{'z=0 got':>10}{'train mean':>12}")
    out = {'run': a.run.name, 'phase': a.phase, 'z_requested': zs.tolist(), 'organs': {}}
    slopes = []
    for o in cond_organs:
        i = idx[o]
        mu, sd = st['mean'][i], (st['std'][i] or 1.0)
        # achieved level expressed in the SAME standardised units as the request,
        # so slope is dimensionless and 1.0 is the target for every organ.
        got_hu = np.array([per_z[z][o] for z in zs])
        ok = np.isfinite(got_hu)
        if ok.sum() < 3:
            print(f"  {o:<32}{'—':>8}{'—':>7}  (too few valid points)")
            continue
        got_z = (got_hu[ok] - mu) / sd
        slope, intercept = np.polyfit(zs[ok], got_z, 1)
        r = float(np.corrcoef(zs[ok], got_z)[0, 1])
        at0 = float(np.interp(0.0, zs[ok], got_hu[ok]))
        slopes.append(slope)
        out['organs'][o] = {'slope': float(slope), 'intercept': float(intercept),
                            'r2': r * r, 'achieved_hu': got_hu.tolist(),
                            'train_mean': mu, 'train_sd': sd}
        print(f"  {o:<32}{slope:>8.3f}{r*r:>7.2f}{at0:>10.1f}{mu:>12.1f}")

    med = float(np.median(slopes)) if slopes else float('nan')
    out['median_slope'] = med
    print(f"\n  median slope over {len(slopes)} organ(s): {med:.3f}")
    if med > 0.75:
        print("  => CONTROLLABLE. The model follows the requested level closely.")
    elif med > 0.35:
        print("  => PARTIALLY controllable: the request is damped. Report the slope,")
        print("     do not claim full control.")
    else:
        print("  => NOT controllable. The conditioning input is being largely ignored;")
        print("     the model emits about the same level whatever you ask for.")
    print("\n  This number never reads the real CECT's level — the ground truth is the")
    print("  REQUEST. It is therefore leak-free, unlike featHU/beta-lev under oracle.")

    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / 'calibration.json').write_text(json.dumps(out, indent=2))
        print(f"\n[written] {a.out / 'calibration.json'}")
        if not a.no_plot:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                n = len(out['organs'])
                cols = min(4, n); rows = (n + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows),
                                         squeeze=False)
                for k, (o, d) in enumerate(out['organs'].items()):
                    ax = axes[k // cols][k % cols]
                    mu, sd = d['train_mean'], d['train_sd']
                    req_hu = mu + zs * sd
                    ax.plot(req_hu, req_hu, '--', color='#a89c82', lw=1, label='ideal')
                    ax.plot(req_hu, d['achieved_hu'], 'o-', color='#b5541f', ms=4,
                            label=f"slope={d['slope']:.2f}")
                    ax.set_title(o, fontsize=9)
                    ax.set_xlabel('requested HU', fontsize=8)
                    ax.set_ylabel('achieved HU', fontsize=8)
                    ax.tick_params(labelsize=7)
                    ax.legend(fontsize=7)
                for k in range(n, rows * cols):
                    axes[k // cols][k % cols].set_visible(False)
                fig.suptitle(f"Level controllability — {a.run.name} ({a.phase})", fontsize=11)
                fig.tight_layout()
                fig.savefig(a.out / 'calibration.png', dpi=160, bbox_inches='tight')
                print(f"[written] {a.out / 'calibration.png'}")
            except ImportError:
                print("(matplotlib absent — JSON written, plot skipped)")


if __name__ == '__main__':
    main()
