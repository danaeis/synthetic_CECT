#!/usr/bin/env bash
# Runs a sequence of loss-flag ablation scenarios through train.py, one at a
# time. Each scenario reuses the on-disk patch cache (see dataset.py /
# config.py CACHE_DIR) — only the first scenario in the queue pays the full
# preload cost, later ones with the same data config load in seconds.
#
# Edit the SCENARIOS array below to add/remove/reorder scenarios. Flags are
# train.py's existing --use_X / --no_X CLI overrides (see `python train.py
# --help`); leave the flag list empty for the L1-only baseline.
#
# Usage:
#   ./run_scenarios.sh              # run every scenario in order
#   ./run_scenarios.sh ssim cycle   # run only the named scenario(s)

set -uo pipefail

BASE_OUT="../out_synthesis_train/literature_baseline"
STOP_ON_ERROR=1   # set to 0 to keep going after a scenario fails

SCENARIOS=(
  "l1_only|"
  "l1_adv| --use_adversarial"
  "pix2pixhd_baseline|--use_adversarial --use_perceptual"
  "pix2pixhd_baseline_feature|--use_adversarial --use_perceptual --use_feature_matching"
  "extra_ssim|--use_adversarial --use_perceptual --use_feature_matching --use_ssim"

  # ── Organ-weighting curriculum ladder ──────────────────────────────────────
  # Run B before A: it is the control. B changes exactly one thing (GI tract
  # excluded from the organ term); A adds the full tiered weight vector on top.
  # If B recovers most of A's gain, the tiered vector is unnecessary complexity.
  #
  # NOTE: these turn on per-organ weights, which flips the train split's mask to
  # MULTI-LABEL and therefore changes the patch-cache key — the first of these to
  # run pays a one-time full re-preload. That is expected, not a hang.
  "l1_bowel_zero|--use_organ --use_per_organ_weights --organ_weight_preset gi_zero --use_l1_decay"
  "l1_organ_curriculum|--use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay"
  # C: the adversarial branch was not worse, it destabilised (best ep27, then
  # decayed). Longer warmup + slower discriminator, on top of A.
  # NOTE: the first l1_adv_organ run trained with a FLAT lambda_l1=25 — the decay
  # was a no-op because adversarial pinned the start to lambda_l1_reduced (25),
  # which equalled the floor. Fixed in losses.py; re-run to get the real curriculum.
  "l1_adv_organ|--use_adversarial --use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --adv_warmup_epochs 15 --lr_disc 5e-5"

  # D: HU-profile loss on top of A. The measured win from tiered weighting was in
  # per-organ HU error (-1.58 HU, t=-4.22), which is exactly what the XGBoost
  # phase model reads — so optimise it directly instead of as a side effect.
  "l1_organ_huprofile|--use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --use_hu_profile"
  # E: HU-profile without the per-voxel organ term — isolates how much of the
  # gain is the LEVEL constraint vs the texture weighting.
  "l1_huprofile_only|--use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --use_hu_profile"

  # ── Gate B: capacity probes ────────────────────────────────────────────────
  # Why these exist. Reading the 45-epoch seed runs: raw train L1 flattens at
  # 0.0139 (~8.3 HU) while val MAE sits at 0.0148 (~8.9 HU) — a 6% gap. There is
  # no overfitting, so regularisation/augmentation is not the lever. The model
  # cannot fit its OWN training data below ~8 HU, and these separate the three
  # candidate reasons: capacity, optimisation, or the data (registration error).
  #
  # B1 is decisive. 5 cases, no dropout, L1 only, 200 epochs, no early stop:
  #   train MAE  -> <2 HU   capacity is sufficient; the ceiling is data/optimisation,
  #                         and attention's role would be inductive bias, not params
  #   train MAE plateaus >10 HU   capacity or optimisation IS the bottleneck; fix
  #                         width/depth before adding anything
  # Run scripts/audit_data_ceiling.py FIRST — if per-case error tracks bone-HU
  # misregistration, an 8.3 HU floor is the data's and none of this has headroom.
  "capacity_overfit|--max_train_cases 5 --dropout 0.0 --epochs 200 --no_early_stop"

  # B2/B4: width sweep. base_ch 64 already exists as l1_organ_curriculum, so only
  # 32 and 96 are new. Flat 32≈64≈96 means capacity is not binding. Either way,
  # base_ch=96 (~30.0M vs 13.3M) is the parameter-matched control that any
  # attention claim must beat — a reviewer asks for it first.
  "width32|--use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --base_ch 32"
  "width96|--use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --base_ch 96"

  # ── Gate D: pre-measured architecture hypothesis ──────────────────────────
  # GroupNorm addresses TWO measured symptoms of one cause, at the cost of one
  # flag: seam is 1.34-1.67 for every model and never near 1.0
  # (norm_attribution.py: instance drift@shift32 13.98 HU vs group 6.83 HU), and
  # scripts/erf.py shows instance/group norm couple the whole patch into every
  # output pixel (r50 14 px -> 4 px moving to group). See
  # analysis/texture_consistency_findings.md.
  "l1_organ_groupnorm|--use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --generator_norm group"
  # add more scenarios here, format: "name|--flag1 --flag2 ..."
)

SEEDS="${SEEDS:-42}"

run_one() {
  local name="$1" flags="$2"
  for seed in $SEEDS; do
    local out="${BASE_OUT}_${name}"
    [[ "$seed" != "42" || "$SEEDS" != "42" ]] && out="${out}_s${seed}"
    mkdir -p "$out"
    echo "=== [$(date '+%F %T')] Scenario: $name  seed=$seed  ->  $out ==="
    # shellcheck disable=SC2086
    python train.py --output_dir "$out" --seed "$seed" $flags \
      2>&1 | tee -a "$out/run_scenarios.log"
    local status="${PIPESTATUS[0]}"
    if [[ "$status" -ne 0 ]]; then
      echo "!!! Scenario '$name' seed $seed FAILED (exit $status)"
      [[ "$STOP_ON_ERROR" -eq 1 ]] && exit "$status"
    fi
  done
}

if [[ $# -gt 0 ]]; then
  for want in "$@"; do
    found=0
    for entry in "${SCENARIOS[@]}"; do
      name="${entry%%|*}"; flags="${entry#*|}"
      if [[ "$name" == "$want" ]]; then
        run_one "$name" "$flags"
        found=1
      fi
    done
    [[ "$found" -eq 0 ]] && echo "!!! No scenario named '$want' in SCENARIOS" >&2
  done
else
  for entry in "${SCENARIOS[@]}"; do
    name="${entry%%|*}"; flags="${entry#*|}"
    run_one "$name" "$flags"
  done
fi
