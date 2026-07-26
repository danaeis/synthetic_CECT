#!/usr/bin/env python
"""scripts/seed_stats.py — mean ± std across seed replicates.

Reads benchmark.py's master_table.csv, groups rows whose model name differs only
by a trailing _s<N>, and reports the spread. The std column is the whole point:
it is the noise floor, and any scenario-to-scenario delta smaller than 2x it is
not a result.
"""
import argparse, re, sys
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--csv', default='analysis/benchmark/master_table.csv')
ap.add_argument('--metrics', nargs='*', default=[
    'feature_l1_hu', 'org_mae', 'org_ssim', 'body_mae', 'psnr',
    'raps_hf', 'grad_w1', 'seam', 'zflicker', 'phase_acc', 'gen_prob'])
a = ap.parse_args()

df = pd.read_csv(a.csv)
df['group'] = df['model'].str.replace(r'_s\d+$', '', regex=True)
df['seed'] = df['model'].str.extract(r'_s(\d+)$')

for grp, sub in df.groupby('group'):
    if len(sub) < 2:
        continue
    print(f"\n=== {grp}   (n_seeds={len(sub)}: {', '.join(sub['seed'].dropna())})")
    print(f"{'metric':16s}{'mean':>10}{'std':>10}{'2*std':>10}{'min':>10}{'max':>10}")
    for m in a.metrics:
        if m not in sub:
            continue
        v = sub[m].astype(float)
        print(f"{m:16s}{v.mean():10.4f}{v.std(ddof=1):10.4f}"
              f"{2*v.std(ddof=1):10.4f}{v.min():10.4f}{v.max():10.4f}")


# Pooled within-group sigma. With n=3 per group the per-group std is itself very
# noisy (~±40%); pooling the two scenarios doubles the degrees of freedom and
# gives a steadier noise floor to set the gate against.
print("\n=== POOLED (within-group, across all groups)")
print(f"{'metric':16s}{'pooled_std':>12}{'2*pooled':>12}{'df':>6}")
for m in a.metrics:
    if m not in df:
        continue
    ss, dof = 0.0, 0
    for _, sub in df.groupby('group'):
        v = sub[m].astype(float).dropna()
        if len(v) < 2:
            continue
        ss  += float(((v - v.mean()) ** 2).sum())
        dof += len(v) - 1
    if dof:
        s = (ss / dof) ** 0.5
        print(f"{m:16s}{s:12.4f}{2*s:12.4f}{dof:6d}")
