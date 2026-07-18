# NCCT→CECT — plan to completion

State as of 2026-07-17 (verified from result files):

- **Phase classifier (CTPhase-XGBoost)**: DONE. Full TS masks (`_seg_full`) →
  **97.07% OOF** (was 87.8% with the vessel-incomplete `_seg_reg`). Aorta coverage
  0→410/410; per-organ HU shows correct dynamics (aorta NC=39/art=277/ven=132,
  portal-vein & IVC peak in venous). This is a trustworthy phase judge now.
- **Synthesis**: `l1_only` and `l1_adv` trained (~80 ep). `l1_only` wins every
  pixel/organ metric — the expected L1-regression-to-mean blur; pixel metrics
  can't tell which model makes the *right contrast phase*.
- **Per-organ synthesis metrics**: working, but show `label_<id>` (no name map
  loaded) and use `_seg_reg` (no aorta).
- **phase_eval.py**: built, not yet applied to generated volumes.

The single most important missing piece is **phase-fidelity evaluation** — running
each trained generator over full volumes and scoring the synthetic CECT with the 97%
XGBoost model. Everything else is ablation breadth and polish.

---

## Phase 0 — housekeeping (cheap, unblocks names + full-mask organ metrics)

0.1 **Re-run the XGBoost retrain with the current code** so it emits
`organ_report.csv`, `organ_importance`, and **`organ_label_map.json`** (the old
run predates these):
```bash
python retrain_xgb.py --data_dir <B2_deeds__aligned> --labels_csv <labels.csv> \
    --file_tag _deeds --seg_suffix _seg_full \
    --out_weights xgb_vindr_full.pkl --out_dir retrain_out_full
```
0.2 **Point the synthesis at the name map + full masks** (`config.py`):
`ORGAN_LABEL_MAP_JSON = '../CTPhase-XGBoost/retrain_out_full/organ_label_map.json'`
and `SEG_SUFFIX = '_seg_full'`. New runs then report per-organ metrics **by name,
including the aorta/vessels**. (Switching `seg_suffix` invalidates the patch cache
→ one-time re-preload; the running `l1_only`/`l1_adv` stay valid for pixel metrics.)

---

## Phase 1 — phase-fidelity evaluation (CRITICAL PATH) — BUILT

The metric that actually validates contrast synthesis, and the only one that can
rank `l1_only` vs `l1_adv` vs the rest on *phase correctness* rather than blur.

**Built (verified locally, `smoke_test_infer.py` 10/10):**
- `infer_volume.py` — loads a scenario's `run_config.json` + `best_model.pth`,
  tiles each held-out NCCT volume with the run's OWN patch geometry (batched,
  overlap-averaged), stitches a full synthetic CECT, de-normalises `[0,1]→HU`,
  saves NIfTI on the source grid, and writes `phase_infer/manifest.csv`. Dims-
  parametric (2-D now, 3-D-ready — both smoke-tested).
- `CTPhase-XGBoost/phase_eval.py` — added `--gen_in_hu`: treats the saved
  synthetic volume as already-HU while still clipping the real CECT to the window
  for a fair per-organ comparison. (Only the `literature_baseline/CTPhase-XGBoost`
  copy was edited; the `phase-detection/CTPhase-XGBoost` copy still lacks it.)

**Run per scenario (remote):**
```bash
# 1. reconstruct + manifest (GPU)
python infer_volume.py --scenario_dir <.../literature_baseline_l1_only> --split test
# 2. score (prints the exact command at the end of step 1)
python CTPhase-XGBoost/phase_eval.py --weights CTPhase-XGBoost/xgb_vindr_full.pkl \
    --manifest <scenario_dir>/phase_infer/manifest.csv --gen_in_hu \
    --out_json <scenario_dir>/phase_infer/phase_eval_report.json
```
Yields `gen_phase_accuracy_vs_target`, `gen_agreement_with_real`, per-organ HU
error — the headline "does model X actually produce venous-phase-looking CT?" row.
Depends on Phase 0's `xgb_vindr_full.pkl` (full-mask retrain).

---

## Phase 2 — finish the ablation ladder + master comparison

2.1 Run the remaining scenarios on top of what's done (`l1_only`, `l1_adv`):
`pix2pixhd_baseline` (adv+perc+fm), `+ssim`, `+gradient`, `+frequency`, `+organ`,
`+saliency`, `+seg_consistency` — via `run_scenarios.sh` (parametric λ already in
place; use `_seg_full`).
2.2 **Master comparison table** across all scenarios with the full metric suite:
global pixel (MAE/PSNR/SSIM/NCC), organ-region, per-organ (names), and **Phase-1
phase-fidelity**. Expectation to state up front: adversarial/perceptual runs will
*lose* on pixel metrics but should *win* on phase-fidelity + look sharper — that
trade-off is the thesis's core finding, and only Phase 1 can show it.

---

## Phase 3 — improve generation (optional, higher effort)

3.1 **Differentiable organ-enhancement loss** (the deferred "3b"): using the full
masks + each case's real per-organ target-HU profile, penalise the generated
image's per-organ mean HU deviation — differentiable, optimises exactly the signal
the XGBoost reads. Natural next lever once the ablation shows where pixel losses
plateau.
3.2 **Conditional multi-phase generation** (arterial + venous by condition) per
`phase_conditioning_plan.md` — one generator conditioned on target phase; bigger
architectural change, do only if single-phase venous is solid.

---

## Phase 4 — consolidate

Pick the best config by phase-fidelity + radiologist-plausible sharpness (not
lowest MAE), write the comparison up, note the metric-methodology finding (why
pixel metrics mislead here).

---

## Recommended order

Phase 0 (hours) → **Phase 1 (the missing decisive metric)** → Phase 2 (breadth) →
Phase 3 (if time) → Phase 4. Phase 1's volume-inference script is the main new
build and the critical dependency for a defensible comparison.
