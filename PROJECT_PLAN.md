# NCCT→CECT synthesis — master plan

State as of 2026-07-19. Supersedes `NCCT2CECT_PLAN.md` and
`scenario_results_overview.md` (both deleted; recoverable via git).
Companion docs: `IMPLEMENTATION.md` (what the code actually does),
`phase_conditioning_plan.md` (literature + multi-phase conditioning design).

---

## 1. Where things stand

### 1.1 Phase classifier — done, trustworthy
`orgFeatXGB_CTPhase/retrain_out_full/metrics.json`: **97.07% OOF** (410 cases,
137 studies). Non-contrast recall 1.00; residual error is arterial↔venous (11
cases). `organ_report.csv` shows physiologically correct dynamics (aorta
NC=39 / art=277 / ven=132 HU), so it reads real contrast kinetics.

Feature importance is highly concentrated:
**aorta 0.293 + liver 0.290 + heart 0.235 = 82%**; bowel ≈1.3%.

### 1.2 Phase fidelity — the decisive metric, now run

`out_synthesis_train/*/phase_infer/phase_eval_report.json`, n=20 test cases:

| | l1_only | l1_adv |
|---|---|---|
| PSNR / SSIM | **29.75 / 0.9837** | 29.33 / 0.9824 |
| organ-region PSNR / SSIM | **26.13 / 0.9381** | 25.72 / 0.9340 |
| gen phase accuracy vs target | 0.95 | **1.00** |
| agreement with real | 0.95 | **1.00** |
| mean gen target prob | 0.948 | **0.975** |
| mean feature L1 (HU) | **17.32** | 17.85 |

**The adversarial model loses every pixel metric and wins on phase correctness.**
That is the thesis's core methodological claim, and it now has evidence: PSNR/SSIM
reward the L1-blur that makes an image *look* close while getting the contrast
state less reliably right.

Three caveats to keep attached to this result:

1. **0.95 vs 1.00 on n=20 is a one-case difference.** The confidence margin
   (0.948 → 0.975, consistent across cases) is the sturdier signal. This needs
   more test cases before it can carry a chapter.
2. `l1_adv` is *worse* on mean HU error (17.85 vs 17.32). The gain is in phase
   *class* and confidence, not raw HU regression — state it that way.
3. `l1_adv` did not lose so much as **destabilize**: best epoch 27
   (29.33 / 0.9824, within 0.4 dB of `l1_only`), decaying to 27.74 / 0.9761 by
   ep57. That is a GAN-stability curve. Compounding it, checkpoint selection ran
   on global val MAE, which structurally prefers the blurriest epoch.

### 1.3 The loss budget is misallocated

Voxel shares from a sample `_seg_full` mask vs XGBoost importance:

| group | voxel share | phase importance |
|---|---|---|
| bone + muscle | 36.6% | ~0 |
| GI tract (stomach/bowel/colon/duodenum/esophagus) | 27.1% | ~1.3% |
| lungs | 15.5% | ~0 |
| solid organs | 13.9% | ~33% |
| heart | 2.7% | 23.5% |
| **phase-critical vessels** | **1.8%** | **~35%** |

Aorta: **0.91% of voxels, 0.293 importance** — a ~30:1 mismatch. The consequence
shows in both metric sets. Worst per-organ SSIM is bowel/duodenum/esophagus
(0.55–0.69, on the largest areas); worst HU errors are the vessels (portal vein
40.3, pulmonary vein 33.3, iliac arteries ~28). The model spends its capacity on
stochastic bowel gas and contrast-free bone, and under-fits the structures that
determine phase.

---

## 2. What changed in the code

### 2.1 Full TS label map — the `label_<id>` mystery, solved

`load_organ_label_map()` returns only the **16 organs the XGBoost consumes**, in
its trained feature order — by design; it must not grow. Your `_seg_full` masks
carry **79 of TS's 117 labels**, so that map could never name them.

New: `orgFeatXGB_CTPhase/dump_ts_label_map.py` → `ts_label_map_total.json`
(all 117, generated and committed). The separate root cause for the missing names
was that `ORGAN_LABEL_MAP_JSON` was a **relative** path pointing at a
`../CTPhase-XGBoost/` directory that doesn't hold your retrain output; it silently
resolved to `{}`. Now absolute, anchored to `config.py`'s own location.

### 2.2 Per-organ weighted L1 (`losses.py`)

`OrganWeightedLoss` rewritten:
- **Per-label LUT** — each TS label gets its own weight; weight 0 excludes that
  anatomy from the gradient entirely.
- **L1 instead of MSE.** The old term was mask-weighted MSE, which regresses to
  the conditional mean harder than L1 — working directly against the sharpness
  this term exists to improve.
- **Normalised by `w.sum()`**, not `.mean()`, so excluding area doesn't silently
  shrink the loss and couple `lambda_organ` to the weight scheme.
- Raises on an all-zero scheme rather than training on a constant 0 loss.

Weights are declared **by organ name** in `config.py` and resolved through the
label map, so a TS version bump fails loudly instead of mis-weighting. This caught
a real trap: **ids 54–62 are contrast-carrying vessels** (carotids, SVC,
brachiocephalics) that an id-range scheme would have missed.

### 2.3 L1 decay curriculum

Three stages: **structure (L1) → contrast (organ) → texture (adv)**.
`lambda_l1` holds at 100 through `l1_decay_start_epoch`, ramps linearly to
`lambda_l1_floor` by `l1_decay_end_epoch`, then holds. Implemented with the same
`set_epoch` / `_w()` pattern the adversarial and cycle warmups already use, and
logged per-epoch to `history.json`.

**The floor is deliberately non-zero, and this is load-bearing.** The organ term
only sees *labelled* voxels — background (air, fat, skin, table) and any
zero-weighted label have no other constraint anywhere in the composite loss. At
`lambda_l1 = 0` they would receive zero gradient for the rest of training and
could drift into artifacts that no organ-region metric would report. The floor
keeps them anchored while still handing a 4× priority shift to the organ term.
Verified: with bowel at weight 0, it still receives global-L1 gradient.

### 2.4 Checkpoint selection (`trainer.py`)

Was `val_loss` (global MAE) — which selects the blurriest epoch *by construction*,
and is dominated by background (global PSNR runs ~3.6 dB above organ-region PSNR
here). Now `val_org_ssim` via `_selection_score`, configurable through
`SELECTION_METRIC`, with automatic fallback to `val_loss` when no mask exists.

### 2.5 Sample grids

Previously `next(iter(val_loader))` on an unshuffled loader — the same 4 patches
from the same case, every epoch, for every scenario. Now `SAMPLE_MODE='random'`
draws fresh patches each epoch, **stratified across distinct validation cases**,
seeded by epoch so any given grid is reproducible. Each row is labelled with case
id + source slice and annotated with its own PSNR/SSIM, and a 4th `|error|` column
(fixed scale 0–0.25, so rows and epochs are comparable) shows *where* the model
fails. `SAMPLE_MODE='fixed'` restores the old behaviour.

Trade-off to be aware of: random rows mean you can no longer watch one specific
patch sharpen across epochs. `curves.png` and the per-row metrics cover progress;
the grid is now for spotting failure modes and checking generalisation across
patients.

Per-patch provenance (`case_ids`, `patch_z`) is recorded in the dataset and
written into new caches. It is **deliberately not in the cache key** — existing
caches stay valid and simply produce unlabelled (still random) grids. Delete a
cache to regenerate it with labels.

### 2.6 Robustness: checkpoint writes

A truncated `ckpt_*.pth` (job killed mid-save, full disk) previously aborted the
whole scenario on resume. Now: `_atomic_save` writes to a temp file, `fsync`s,
and `os.replace`s into position, so a checkpoint that exists is always complete;
and the resume path walks newest→oldest, quarantining unreadable files as
`.pth.corrupt` instead of failing. With `keep_last_n_checkpoints=3`, a corrupt
tail costs a few epochs rather than the run.

### 2.7 Two bugs fixed along the way

- **Train masks were binary.** `mask_multilabel` was gated on conditions that are
  val/test-only, so the train split got a binarised mask and per-organ weights
  would have silently collapsed every organ onto one weight. (This corrects a
  claim in the previous revision of this document, which said no dataset change
  was needed — that was wrong.)
- **`per_organ_history` wasn't checkpointed**, so `organ_metrics.json` restarted
  empty on resume and lost every pre-resume epoch.

Also: per-organ metrics derived the label id with `np.round` but selected with
exact float equality; both now use the rounded array. And `CompositeLoss` read the
three "L1 competitor" flags twice with different defaults — now resolved once, so
the active losses and the chosen `lambda_l1` can't disagree.

---

## 3. Experiment ladder

`l1_only` / `l1_adv` are trained — do not re-run. Added to `run_scenarios.sh`:

| # | scenario | flags | isolates |
|---|---|---|---|
| B | `l1_bowel_zero` | `--use_organ --use_per_organ_weights --organ_weight_preset gi_zero --use_l1_decay` | GI exclusion alone |
| A | `l1_organ_curriculum` | as B, `--organ_weight_preset tiered` | full tiered scheme |
| C | `l1_adv_organ` | A + `--use_adversarial --adv_warmup_epochs 15 --lr_disc 5e-5` | stabilized adv branch |

**Run B first — it is the control.** It changes exactly one thing. If B recovers
most of A's gain, the tiered vector is unnecessary complexity and the simpler
intervention is the better thesis claim. C tests whether the adversarial branch
(which already wins on phase fidelity) survives past ep27 with a slower
discriminator and selection on organ-SSIM rather than MAE.

> These flip the train mask to multi-label, which changes the patch-cache key —
> the first of them pays a one-time full re-preload. Expected, not a hang.

---

## 4. How to run and verify

```bash
# 0. label map (already generated; regenerate only if TS version changes)
python orgFeatXGB_CTPhase/dump_ts_label_map.py

# 1. unit-level checks
python smoke_test_organ_weights.py     # 18 checks: LUT, zero-grad, decay, mask, anchor
python smoke_test_organ_focus.py

# 2. the ladder
./run_scenarios.sh l1_bowel_zero
./run_scenarios.sh l1_organ_curriculum
./run_scenarios.sh l1_adv_organ

# 3. phase fidelity — the metric that decides
python infer_volume.py --scenario_dir ../out_synthesis_train/literature_baseline_l1_bowel_zero --split test
python orgFeatXGB_CTPhase/phase_eval.py \
    --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
    --manifest ../out_synthesis_train/literature_baseline_l1_bowel_zero/phase_infer/manifest.csv \
    --gen_in_hu --out_json .../phase_infer/phase_eval_report.json
```

**During a run, check:**
- `history.json` has `lambda_l1` decaying 100 → 25 and a non-zero `train_organ`.
- The startup log prints the zero-weighted label ids and the selection metric.
- `organ_metrics.json` now names organs (`liver`, `aorta`) not `label_5`.
- Bowel still *appears* in `organ_metrics.json` — excluded from the loss, still
  evaluated. Keep it that way so the exclusion stays transparent.

**Success criterion**: beat `l1_only`'s `gen_phase_accuracy_vs_target` 0.95 /
`mean_gen_target_prob` 0.948, and bring the vessel HU errors (portal vein 40.3,
pulmonary vein 33.3) down. Not PSNR — PSNR going *down* is expected and fine.

**One check no metric will do for you:** visually inspect bowel and background in
a synthetic volume. Zero-weighted regions constrained only by the L1 floor are
exactly where drift would appear, and every organ-region metric is blind to it.
If artifacts show, raise `LAMBDA_L1_FLOOR` (25 → 40) before touching anything else.

---

## 5. Open risks

- **n=20 is thin** for the headline phase-fidelity claim. Expanding the test split
  is the cheapest way to harden the thesis's central result.
- **Zero-weighting bowel needs defending.** Frame it as measured: 27% of voxels,
  ~1.3% of phase importance, stochastic gas/content not inferable from NCCT.
  Report bowel metrics anyway to show it didn't degrade.
- **`_seg_reg` and `_seg_full` use different label conventions** (`_seg_reg`
  tops out at id 83 with ids absent from `_seg_full`). The weights assume
  `_seg_full`. Do not mix mask sets — `SEG_SUFFIX` is `'_seg_full'` and should stay.
- **Selection-metric change breaks comparability** with the two completed runs,
  which were selected on MAE. When comparing, either say so or re-select the old
  runs' best epoch from their `history.json`.

---

## 6. Next, after the ladder

Per-organ **HU-profile loss** (penalise deviation from each case's real per-organ
target HU) optimises exactly what the XGBoost reads, and is the natural next lever
once these ablations show where weighted L1 plateaus. Multi-phase conditioning
(`phase_conditioning_plan.md`) only after single-phase venous is solid.
