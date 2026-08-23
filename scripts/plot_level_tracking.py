#!/usr/bin/env python
"""Render the real-vs-generated median-HU scatter from audit_enhancement.py's
--out JSONs, side by side across models.

Usage:
    python plot_level_tracking.py \
        --json l1_only=l1_only.json level_all8=level_all8.json diff_hetero_nll=diff_hetero_nll.json \
        --organs aorta portal_vein_and_splenic_vein inferior_vena_cava liver \
        --out level_tracking.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LEVEL_ORGANS_DEFAULT = ['aorta', 'portal_vein_and_splenic_vein',
                        'inferior_vena_cava', 'liver']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', nargs='+', required=True,
                    help='label=path.json (repeatable) — one per model, in display order')
    ap.add_argument('--organs', nargs='*', default=LEVEL_ORGANS_DEFAULT)
    ap.add_argument('--out', default='level_tracking.png')
    a = ap.parse_args()

    models = []
    for spec in a.json:
        label, _, path = spec.partition('=')
        models.append((label, json.loads(Path(path).read_text())))

    organs = a.organs
    n_rows, n_cols = len(organs), len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.2 * n_rows),
                             squeeze=False)

    for col, (label, data) in enumerate(models):
        for row, organ in enumerate(organs):
            ax = axes[row][col]
            if organ not in data.get('real_hu', {}):
                ax.set_visible(False)
                continue
            real = np.array(data['real_hu'][organ])
            gen = np.array(data['gen_hu'][organ])
            ok = np.isfinite(real) & np.isfinite(gen)
            real, gen = real[ok], gen[ok]
            stats = data['organs'][organ]
            # audit_enhancement.py's --out JSON keeps beta ('slope') and r2 per
            # organ but never writes the fit's intercept, only the raw real_hu/
            # gen_hu arrays it was computed from — refit here rather than assume
            # a key that was never saved.
            beta, r2 = stats['beta'], stats['r2']
            intercept = float(np.polyfit(real, gen, 1)[1]) if len(real) >= 3 else float('nan')

            ax.scatter(real, gen, s=22, alpha=0.75, color='#b5541f', edgecolors='none')
            lo, hi = min(real.min(), gen.min()), max(real.max(), gen.max())
            pad = 0.06 * (hi - lo if hi > lo else 1)
            lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], '--', color='#a89c82', lw=1, label='y=x (beta=1)')
            if np.isfinite(beta) and np.isfinite(intercept):
                xs = np.array([lo, hi])
                ax.plot(xs, intercept + beta * xs, '-', color='#3a4249', lw=1.5,
                       label=f'fit (beta={beta:.2f})')
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(label, fontsize=11, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{organ}\ngenerated median HU', fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel('real median HU', fontsize=8)
            ax.text(0.04, 0.92, f'$\\beta$={beta:.2f}  r$^2$={r2:.2f}',
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round', fc='white', ec='#ddd4c5', alpha=0.9))

    fig.suptitle('Does the generator track each case\'s contrast level, or emit the population mean?',
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(a.out, dpi=160, bbox_inches='tight')
    print(f'[written] {a.out}')


if __name__ == '__main__':
    main()
