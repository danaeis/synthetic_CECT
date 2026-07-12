"""
Data-driven HU clipping window analysis.

The trainer currently hard-clips both NCCT and CECT volumes to a fixed,
literature-typical soft-tissue window (config.py: HU_MIN=-200, HU_MAX=300)
before rescaling to [0, 1]. That window is a reasonable prior but wasn't
fit to this dataset — clipping always discards information (bone, air,
metal, and anything past the bounds), so the right question is "where do
we actually lose the least of what matters" rather than "what does the
textbook usually use".

This script scans your real NCCT/CECT pairs (same file discovery as
dataset.py), builds pooled HU histograms restricted to tissue-containing
voxels (crude air/background exclusion so the giant air peak at ~-1000
doesn't drown out the tissue distribution), and recommends a percentile-
based window instead of a guessed one.

Usage:
    cd train/literature_baseline
    python analyze_hu_range.py                      # uses config.py paths
    python analyze_hu_range.py --n_cases 40          # subsample for speed
    python analyze_hu_range.py --low_pct 1 --high_pct 99
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config import train_config
from dataset import _load_vol, find_pairs_and_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

AIR_FLOOR = -900.0   # crude air/background exclusion, not a proposed clip bound


def _parse():
    p = argparse.ArgumentParser(description='Data-driven HU clip window analysis')
    p.add_argument('--n_cases', type=int, default=30, help='Number of cases to sample (0 = all)')
    p.add_argument('--low_pct',  type=float, default=0.5, help='Lower percentile for recommended clip')
    p.add_argument('--high_pct', type=float, default=99.5, help='Upper percentile for recommended clip')
    p.add_argument('--output',   type=str, default='hu_range_analysis.png')
    return p.parse_args()


def main():
    args = _parse()
    cfg = train_config

    train_pairs, val_pairs, test_pairs = find_pairs_and_split(cfg)
    pairs = train_pairs + val_pairs + test_pairs
    if not pairs:
        log.error("No NCCT/target pairs found — check config.py data_dir/labels_csv/file_tag.")
        return

    rng = np.random.default_rng(cfg.get('seed', 42))
    if args.n_cases and len(pairs) > args.n_cases:
        idx = rng.choice(len(pairs), args.n_cases, replace=False)
        pairs = [pairs[i] for i in idx]
    log.info(f"Sampling {len(pairs)} case(s) for HU histogram")

    src_vals, tgt_vals = [], []
    for pair in pairs:
        try:
            src = _load_vol(pair['source_path'])
            tgt = _load_vol(pair['target_path'])
        except Exception as e:
            log.warning(f"  Skip {pair['case_id']}: {e}")
            continue
        # crude tissue mask: drop obvious air/background so the histogram
        # isn't dominated by the huge -1000 HU air peak
        src_vals.append(src[src > AIR_FLOOR].ravel())
        tgt_vals.append(tgt[tgt > AIR_FLOOR].ravel())

    src_vals = np.concatenate(src_vals)
    tgt_vals = np.concatenate(tgt_vals)
    all_vals = np.concatenate([src_vals, tgt_vals])

    lo, hi = np.percentile(all_vals, [args.low_pct, args.high_pct])
    lo_round = float(np.round(lo / 10) * 10)
    hi_round = float(np.round(hi / 10) * 10)

    cur_lo, cur_hi = cfg['hu_min'], cfg['hu_max']
    cur_frac = float(np.mean((all_vals >= cur_lo) & (all_vals <= cur_hi)))
    rec_frac = float(np.mean((all_vals >= lo_round) & (all_vals <= hi_round)))

    log.info("=" * 60)
    log.info(f"Pooled tissue-voxel HU percentiles (n={all_vals.size:,}):")
    for p in (0.1, 0.5, 1, 5, 25, 50, 75, 95, 99, 99.5, 99.9):
        log.info(f"  p{p:>5}: {np.percentile(all_vals, p):8.1f} HU")
    log.info("-" * 60)
    log.info(f"Current config window : [{cur_lo}, {cur_hi}]  "
             f"covers {cur_frac*100:.2f}% of tissue voxels")
    log.info(f"Recommended window    : [{lo_round}, {hi_round}]  "
             f"(p{args.low_pct}-p{args.high_pct}), covers {rec_frac*100:.2f}%")

    # Low-end contamination check: AIR_FLOOR is a crude per-voxel air cut, not
    # a real tissue boundary. If the recommended low bound lands close to
    # AIR_FLOOR, that means a large mass of voxels sits just above the floor
    # (lung parenchyma / fat / partial-volume air-tissue edges, none of which
    # carries NCCT<->CECT contrast-enhancement signal) and is dominating the
    # low percentiles. Blindly widening hu_min down to that value would spend
    # most of the normalised [0,1] range on that non-informative mass and
    # compress the real soft-tissue/vessel enhancement band into a smaller
    # slice of it — a regression, not an improvement, versus a tighter window.
    contamination_margin = 100.0  # HU
    if lo_round <= AIR_FLOOR + contamination_margin:
        p_at_floor = float(np.mean(all_vals <= AIR_FLOOR + contamination_margin)) * 100
        log.warning(
            f"Recommended low bound ({lo_round:.0f}) is within "
            f"{contamination_margin:.0f} HU of AIR_FLOOR ({AIR_FLOOR:.0f}) — "
            f"{p_at_floor:.1f}% of pooled voxels sit in that narrow band just "
            f"above the crude air cut. This is very likely lung/fat, not "
            f"diagnostic tissue — do NOT adopt the recommended low bound "
            f"as-is. Inspect {args.output} for a bimodal low-HU peak before "
            f"trusting this window; consider keeping (or raising) the "
            f"current hu_min instead of lowering it."
        )
    log.info("=" * 60)
    log.info("To apply: set HU_MIN / HU_MAX in config.py to the recommended values, "
             "or pass hu_min/hu_max via a config override. Read the warning above "
             "(if any) before doing so blindly.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(AIR_FLOOR, 2000, 300)
    for ax, vals, title in [(axes[0], src_vals, 'NCCT (source)'),
                             (axes[1], tgt_vals, f"{cfg.get('target_phase')} (target)")]:
        ax.hist(vals, bins=bins, density=True, alpha=0.7, color='steelblue')
        ax.axvline(cur_lo, color='red', linestyle='--', label=f'current [{cur_lo:.0f}, {cur_hi:.0f}]')
        ax.axvline(cur_hi, color='red', linestyle='--')
        ax.axvline(lo_round, color='green', linestyle='-', label=f'recommended [{lo_round:.0f}, {hi_round:.0f}]')
        ax.axvline(hi_round, color='green', linestyle='-')
        ax.set_yscale('log')
        ax.set_xlabel('HU'); ax.set_ylabel('density (log)')
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=130)
    log.info(f"Saved histogram plot to {args.output}")


if __name__ == '__main__':
    main()
