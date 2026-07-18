# ROI-HU phase detection + phase-consistency loss

Three additions, all run from `phase_detection/` on the box (has the data,
`totalsegmentator`, `xgboost`, `nibabel`).

## 1. Narrowed HU window (fair deep-encoder number)

`run_phase_detection.py` now takes `--hu_min/--hu_max` (default **-160/400**,
was effectively -1000/1000). The window is part of the feature-cache key, so the
old cached features are NOT reused — it re-extracts. Re-run exactly as before:

```bash
python run_phase_detection.py \
  --labels_csv .../vindr_ds/labels.csv \
  --data_dir   .../B2_deeds__aligned \
  --encoders medvit dinov3 \
  --medvit_pretrained_path ../../MedViT_small_im1k.pth \
  --output_dir ../../phase_results
# optionally sweep, e.g. --hu_min -100 --hu_max 300
```

## 2. XGBoost SOTA baseline on the SAME test split

Reuses `find_phase_volumes` + `split_by_patient` (seed 42) so the 63 test
volumes are identical to the deep baseline; only the model changes.

```bash
python eval_xgb_phase.py \
  --data_dir   .../B2_deeds__aligned \
  --labels_csv .../vindr_ds/labels.csv \
  --xgb_model  ../../phase_results/xgb_vindr.pkl \
  --output_dir ../../phase_results          # -> xgb_roi_eval_test.json
```

Reads per-organ median HU from the co-registered `*_deeds_seg_reg.nii.gz` masks.
**If those masks are binary (not TotalSegmentator multilabel)** the script fails
loudly — add `--totalseg` to segment each raw volume on the fly instead.

## 3. ROI-HU phase-consistency loss (Option A) + distilled classifier (Option B)

Build per-phase organ-HU prototypes from the train split (once):

```bash
python roi_hu_phase_loss.py \
  --data_dir   .../B2_deeds__aligned \
  --labels_csv .../vindr_ds/labels.csv \
  --output_dir ../../phase_results          # -> phase_hu_prototypes.npz
```

Then in the generator training loop:

```python
from phase_detection.roi_hu_phase_loss import ROIHUPhaseLoss
phase_loss = ROIHUPhaseLoss('.../phase_hu_prototypes.npz',
                            input_window=(-160, 400))  # if G emits [0,1]
# gen_vol,(B,1,D,H,W); seg = co-registered MULTILABEL mask; target_phase (B,) long
L = phase_loss(gen_vol, seg, target_phase)
```

`seg` must keep per-organ label ids (the same multilabel mask, NOT binarised as
the current `use_organ`/`use_seg_consistency` path does). Organs absent from a
case's mask or a phase prototype are dropped from the term automatically.

**Option B** (explicit "classifies as target phase", class-distance-weighted CE)
needs the distilled head fit on the 16-dim organ-HU features vs the XGBoost soft
labels via `fit_distilled_head(...)`, saved with `torch.save`, then:

```python
from phase_detection.roi_hu_phase_loss import PhaseClassifierLoss
clf_loss = PhaseClassifierLoss('.../head.pt', input_window=(-160, 400))
```

Start with Option A alone at a low weight (soft nudge); add B only if you want an
explicit classification term. Wire both into `CompositeLoss` behind required
checkpoint-path flags (raise if unset), per `phase_conditioning_plan.md` Step 7.

Local synthetic check (no data needed): `python smoke_test_roi_hu.py`.

## Prototypes vs XGBoost feature — one subtlety

The XGBoost feature uses per-organ **median** HU. The loss/prototypes use per-organ
**mean** HU (median has ~zero gradient a.e., so it can't drive a generator). They
are separate, internally-consistent objects — don't feed a median prototype into
the loss.
