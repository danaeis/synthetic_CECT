#!/usr/bin/env bash
# Sample driver for scripts/make_comparison_figures.py — square block layout
# with organ-boundary zoom insets. Run from the repo root:
#
#     bash scripts/run_comparison_figures.sh
#
# Adjust the paths / knobs below to match your tree.
set -euo pipefail
cd "$(dirname "$0")/.."          # -> repo root

RUNS_DIR="../out_synthesis_train"     # your trained runs (phase_infer manifests)
BENCH_DIR="../bench_ncct2cect"        # vendored competing-model repos
OUT_DIR="analysis/figures"

python scripts/make_comparison_figures.py \
    --runs_dir "$RUNS_DIR" \
    --bench_dir "$BENCH_DIR" \
    --set both \
    --seed 0 \
    --out_dir "$OUT_DIR" \
    --per_row 0 \
    --zoom --zoom_size 110 --zoom_contour \
    --diff \
    --multiplane --planes axial,coronal \
    --win_center 40 --win_width 400 \
    --diff_max 100

echo
echo "Figures written under $OUT_DIR/{comparison,ablation}/"
