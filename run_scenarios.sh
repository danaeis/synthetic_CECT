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
  "l1_adv_organ|--use_adversarial --use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --adv_warmup_epochs 15 --lr_disc 5e-5"
  # add more scenarios here, format: "name|--flag1 --flag2 ..."
)

run_one() {
  local name="$1" flags="$2"
  local out="${BASE_OUT}_${name}"
  mkdir -p "$out"
  echo "=== [$(date '+%F %T')] Scenario: $name  ->  $out ==="
  # shellcheck disable=SC2086
  python train.py --output_dir "$out" $flags 2>&1 | tee -a "$out/run_scenarios.log"
  local status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "!!! Scenario '$name' FAILED (exit $status)"
    if [[ "$STOP_ON_ERROR" -eq 1 ]]; then
      echo "Stopping (STOP_ON_ERROR=1). Remaining scenarios not run."
      exit "$status"
    fi
  else
    echo "=== Scenario '$name' done ==="
  fi
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
