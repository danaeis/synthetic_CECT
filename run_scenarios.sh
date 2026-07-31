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

  # ── Phase conditioning: a 3-arm design ────────────────────────────────────
  # Why. Four measurements have ruled out overfitting, registration error, voxel
  # scale and capacity as the cause of the ~8 HU per-organ floor, leaving the
  # aleatoric explanation: enhancement level depends on injection dose and bolus
  # timing, which an NCCT cannot show. Arterial and venous share the SAME NCCT
  # input but have different targets — a known, controlled instance of exactly
  # that ambiguity. These two arms measure what an explicit conditioning
  # variable recovers.
  #
  #   M1  venous-only            = l1_organ_curriculum_s{42,43,44}, already run
  #   M2  multi-phase, NO input  <- below
  #   M3  multi-phase + FiLM     <- below
  #
  # M2 IS NOT OPTIONAL. Without it an M3 gain cannot be attributed to
  # conditioning rather than to 2x the pairs. It also has its own prediction: if
  # the aleatoric story holds, M2 should be WORSE than M1 on venous featHU,
  # because pooling two phases without saying which one adds ambiguity.
  #
  # NOTE both change the patch-cache key (the pair list gains arterial targets),
  # so the first to run pays a one-time full re-preload. Expected, not a hang.
  #
  # Run M2 and M3 at seed 42 first; expand to 43/44 only if the featHU gap
  # exceeds the measured 2-sigma gate of 0.82 HU.
  "multiphase_uncond|--use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --target_phases venous arterial"
  "multiphase_film|--use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --target_phases venous arterial --use_phase_cond"

  # ── B0: the texture baseline everything below builds on ────────────────────
  # GroupNorm is the validated best (val 8.50 vs 8.86 HU at 3 seeds, ~13x the
  # 2-sigma gate). Adversarial restores texture: on the seed-42 table it is 2.1x
  # better on gradW1, 2.4x on organ-region gradW1 and 1.7x closer to 1.0 on
  # raps_hf, for +0.30 HU featHU — INSIDE the 0.82 HU gate, i.e. free.
  #
  # The discriminator is now CONDITIONAL: cat([source, image]) rather than the
  # image alone (pix2pix's D(x,y)), which is also a plausible fix for the
  # observed instability (train_adv 0.52->0.96 while train_disc 0.245->0.178 —
  # the discriminator was winning).
  "b0_groupnorm_adv|--use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --generator_norm group --use_adversarial --use_cond_disc --adv_warmup_epochs 15 --lr_disc 5e-5"

  # ── Level-conditioning ablation, on top of B0 ──────────────────────────────
  # audit_enhancement.py: the model recovers only 18% of case-to-case level
  # variation and is 5.7x under-dispersed. These ask how much of the residual
  # that accounts for, by telling it the level outright.
  #
  # Run `python scripts/dump_levels.py` FIRST — these need splits/levels.json.
  #
  # Evaluate each with infer_volume.py --level_mode {oracle,population}; the
  # RESULT IS THE GAP. Oracle reads the answer off the real CECT and is never a
  # headline number. featHU is partly circular here (it IS per-organ median-HU
  # error), so also run scripts/heldout_feathu.py — and note the ablation exists
  # precisely to make that contamination visible as it grows.
  #
  #   L1  1 scalar  -> 15 held-out organs
  #   L2  2 scalars -> 14
  #   L3  8 scalars ->  8
  # If L1 already recovers most of it, contrast level is a single global latent
  # factor. If recovery grows with each scalar, each organ carries independent
  # unpredictable variation. Either way it is a result.
  "level_aorta|--use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --generator_norm group --use_adversarial --use_cond_disc --adv_warmup_epochs 15 --lr_disc 5e-5 --cond_organs aorta"
  "level_aorta_pv|--use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --generator_norm group --use_adversarial --use_cond_disc --adv_warmup_epochs 15 --lr_disc 5e-5 --cond_organs aorta portal_vein_and_splenic_vein"
  "level_all8|--use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --generator_norm group --use_adversarial --use_cond_disc --adv_warmup_epochs 15 --lr_disc 5e-5 --cond_organs aorta portal_vein_and_splenic_vein inferior_vena_cava heart liver pancreas gallbladder colon"

  # ── 2.5-D: the one geometric gap that survived measurement ─────────────────
  # Spacing is uniform 1.5 mm isotropic and 128 px already spans 192 mm in-plane,
  # but patch_depth=1 shows the model 1.5 mm of a 258 mm aorta. k=2 buys 7.5 mm,
  # k=5 buys 16.5 mm, for essentially no extra parameters. Both change the
  # patch-cache key -> one full re-preload each.
  "slices5_k2|--use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --generator_norm group --n_input_slices 5"
  "slices11_k5|--use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --generator_norm group --n_input_slices 11"
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
