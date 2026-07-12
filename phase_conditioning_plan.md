# Conditional multi-phase generation + phase-detection loss — plan

Continues the direction already sketched in `train/loss_plan.md` ("Day 4 — Phase-aware
weighting", "Conditioning your auxiliary models"), now grounded in the literature
review (`thesis_latex/LR_sheets/ncct_lr_updated.csv`, 45 papers) and a direct audit
of `phase_detection/`'s actual code state.

---

## Part 1 — Direction: per-phase generation "by condition"

**Don't train separate models per phase.** The literature converges on one
conditioned model over N separate ones, and it matches your stated goal directly:

- **Zha25 (MAN-GAN)**: single generator, NCCT → {arterial, PV, delayed}, phase
  conditioning via mask-adaptive normalization; L1 + adversarial + cycle + mask loss.
- **Pin23**: simplest precedent — a phase-conditioned 3D U-Net / Pix2Pix
  "contrast-switching framework". Good minimal baseline to copy first.
- **Pin24 (HyperPix2Pix)**: conditions via a HyperNetwork generating
  *time-dependent conv kernels* — for continuous/arbitrary phase, not just discrete
  classes. Worth revisiting once discrete conditioning works.
- **Zhe25 (CFPS-Diff)**: phase triplet embeddings + a Phase Distinction Network +
  **phase classification loss (class-distance-weighted) + feature consistency loss**
  — closest existing precedent to what you asked for (classifier as loss), on a
  diffusion backbone rather than GAN.
- **Uhm22 (DiagnosisGAN)**: generator + classifier trained together, classification
  cross-entropy as an auxiliary loss term — direct precedent for "does the output
  classify as the target phase" as a loss, on a GAN (closer to this repo's setup).
- **Li26**: text-prompt phase conditioning — interesting but diffusion-specific and
  a bigger architectural jump; deprioritize for now.

**Recommended staged rollout** (matches `loss_plan.md`'s own week-by-week structure):

1. **Conditioning mechanism, cheapest first**: concatenate a one-hot phase vector
   (broadcast spatially) as extra input channels to `UNetGenerator` — smallest
   change to the existing architecture, easy to validate with the same
   `smoke_test.py` synthetic-tensor pattern already in place. Condition the
   discriminator too (standard cGAN practice — otherwise it can't tell if the
   *wrong* phase was generated, only whether *some* realistic CT was). Only invest
   in FiLM/AdaIN or Pin24's hypernetwork approach if plain concatenation proves too
   weak empirically — validate cheap before expensive.
2. **Data**: `dataset.py`/`config.py` currently hard-code one `TARGET_PHASE`
   ('venous'). `find_pairs_and_split` needs generalizing to yield
   `(NCCT, phase_id, matching_CECT)` triples across all phases present per case
   (`find_pairs_and_split` already scans for `'arterial'`/`'venous'`/`'delayed'`
   keywords via `_infer_phase`, so the phase-matching logic mostly exists — it just
   currently throws away everything except the one configured target phase).
3. **Loss**: once a validated phase classifier exists (Part 2), add a
   classification-consistency term pushing the conditioned generator's output
   toward actually producing the *requested* phase (Zhe25/Uhm22-style), not just a
   plausible contrast image.

---

## Part 2 — Phase detection: get back to a trained, usable classifier

### Step 0 — Blocker (found during this session, must resolve first)

`phase_detector.py:26` does `from data import prepare_dataset_from_folders,
prepare_data` — **`data.py` does not exist anywhere in this repo checkout**
(checked `phase_detection/` and the whole repo tree). Nothing in `phase_detection/`
can currently run, including the retraining you want, until this is found or
rebuilt. It almost certainly still exists wherever `phase_detector.py` was actually
run before (to produce `encoder_comparison.png`) — check that machine/environment
first before reconstructing it from the call-site signature
(`prepare_dataset_from_folders(data_path, labels_csv, validation_split=0.2,
skip_prep=True) -> (train_data_dicts, val_data_dicts)`,
`prepare_data(data_dicts, batch_size=..., augmentation=False, spatial_size=img_size)
-> DataLoader`).

### Step 1 — Confirm MedViT's pretrained weights are actually present

`--medvit_pretrained_path` defaults to the relative path
`'pretrained_medvit_small.pth'`, and `phase_detector.py:981` silently passes
`None` instead if that file doesn't exist at that path — **no warning, no error**,
MedViT would just train from a randomly-initialized backbone. This is the same
class of silent-fallback bug already flagged for `dino_encoder.py`. Run
`download_medvit.py` and confirm the weight file is actually at the path you pass
(or pass `--medvit_pretrained_path <explicit path>`) *before* retraining — otherwise
you can't trust a rerun to reproduce `encoder_comparison.png`'s MedViT numbers.

### Step 2 — Pick the encoder to standardize on: MedViT

From `encoder_comparison.png` (your prior run):

| Encoder | Test acc | **CV acc (real generalization estimate)** |
|---|---|---|
| Dino_v3 | 0.973 | 0.687 ± **0.064** |
| MedViT | 0.962 | 0.685 ± **0.005** |
| TimmViT | 0.752 | 0.325 ± 0.028 |

Two things worth noticing:
- **Train/test-split accuracy (96–97%) is misleadingly high** — 5-fold CV accuracy
  for the top two is only ~69%. Small dataset + LDA on high-dim pooled features
  overfits the train/test split. Treat **~69%, not ~97%,** as the realistic accuracy
  this classifier will have — that matters a lot for how you use it as a loss (see
  Step 4).
- **MedViT's CV variance is 13× lower** than Dino_v3's (0.005 vs 0.064) — far more
  stable across folds. Combined with `dino_encoder.py`'s known bug (every one of its
  four fallback loading paths — real DINOv3, `torch.hub` DINOv1, ImageNet-supervised
  ViT, or literally random `SimpleViT` weights — logs `"✅ Loaded ... model"`
  regardless of which fired, so you can't even be certain what the "Dino_v3" bars
  above actually measured), **MedViT is the safer encoder to standardize on** for
  the classification loss.

### Step 3 — Retrain (reproduces `encoder_comparison.png` + saves checkpoints this time)

```bash
python phase_detector.py \
  --data_path <path_to_ct_data> \
  --labels_csv <labels.csv> \
  --output_dir contrast_phase_results \
  --mode train
```

This retrains all three encoders (the script hard-codes all three in
`encoder_configs`, `phase_detector.py:965-988` — no CLI flag to run just one without
a small edit) and saves `.pkl` checkpoints to
`contrast_phase_results/{encoder_name}/{encoder_name}_trained_model.pkl` this time —
confirm whatever deleted the previous weights (manual cleanup? a scratch/tmp
directory that got wiped?) won't happen again to this output dir.

One thing to verify, not assume: whether the same `labels.csv` already used by this
training pipeline's `config.py` (`LABELS_CSV`) has columns compatible with what
`prepare_dataset_from_folders` expects — can't confirm without `data.py` (Step 0).

### Step 4 — Use the ~69% CV accuracy honestly

A ~69%-accurate classifier used as a *hard* cross-entropy target could inject real
noise into generator gradients. Prefer something closer to Zhe25's **class-distance-
weighted (CDW)** loss (penalizes proportional to how far off the predicted phase is,
not a strict argmax match) over plain cross-entropy, and start with a low loss
weight — treat it as a soft nudge, not ground truth, until retrained numbers (and
ideally more labeled data) improve the CV accuracy.

### Step 5 — Bridge: sklearn LDA isn't differentiable, but it doesn't need to be

`ContrastPhaseClassifier.predict()` takes pre-extracted numpy feature vectors, and
the LDA head is a fitted `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`
— not part of the torch autograd graph. But mathematically, LDA's decision function
is just a linear projection: extract the fitted `coef_`/`intercept_` and rebuild it
as a frozen `torch.nn.Linear` loaded with those exact weights — numerically
identical to the sklearn model's decision function, but now differentiable, so
gradients can flow generator-output → frozen MedViT encoder → this reconstructed
linear head → loss. (The encoder itself is already a normal frozen `nn.Module`, so
only the LDA step needs this treatment.) The encoder's existing `_preprocess_slice`
already handles resize-to-224 + ImageNet-normalize + RGB-repeat from a `(B,1,H,W)`
or `(B,1,D,H,W)` `[0,1]` tensor — no new preprocessing bridge needed there.

### Step 6 — Saliency map: this needs real new work, nothing exists yet

Confirmed: nothing in `phase_detection/` produces a spatial map — MedViT's LDA
operates on a single pooled feature vector per volume, no spatial structure survives
pooling. Two options, in order of effort:

- **(a) Vanilla-gradient saliency** — backprop the target-phase logit (via the
  Step 5 linear-head reconstruction) w.r.t. the input tensor. No architecture
  changes needed, works immediately once Step 5 exists. Known limitation: vanilla
  gradients are often noisy/scattered compared to Grad-CAM-style methods.
- **(b) DINO spatial-token attention** — unlike MedViT's pooled-only output, DINO's
  `forward_intermediates`/`hidden_states` retain per-patch spatial tokens *before*
  pooling (the parent dir's `dino_backbone.py` already exposes this, and
  `losses.py`'s existing `DinoSaliencyLoss` already uses it for a generic
  feature-difference heatmap). You could compute a genuine per-region importance
  map as (target-phase class direction) · (spatial patch tokens) — no Grad-CAM
  machinery needed — but only if the DINO path is used for this specific purpose
  even if MedViT is what's used for the classification loss.

Start with (a) as a working v1; only invest in (b) if (a) doesn't actually improve
results over the existing heuristic `PhaseSaliencyLoss` already in `losses.py`.

### Step 7 — Wire into the training pipeline (once 5–6 are validated, not before)

Follow the exact pattern every other loss in `losses.py`/`CompositeLoss` already
uses: new `PhaseClassifierLoss`/`PhaseSaliencyLoss` classes, `use_phase_classifier`/
`use_phase_saliency` config flags, gated behind a **required** checkpoint-path
config (raise a clear error if unset — same approach discussed and shelved earlier
this session for the same reason: don't let it silently run on an untrained/missing
checkpoint). Add coverage to `smoke_test.py`. Write a `check_phase_classifier.py`
sanity script (same spirit as `check_seg_masks.py`) reporting real held-out accuracy
before trusting it as a loss signal on a real training run.

---

## Summary — what actually blocks starting

Everything in Part 2 is blocked on **Step 0** (`data.py` missing) and **Step 1**
(confirming MedViT's pretrained weights are actually on disk) — nothing else can
proceed until those two are resolved. Part 1's Steps 1–2 (conditioning mechanism +
data restructuring) don't depend on phase detection at all and could start in
parallel.
