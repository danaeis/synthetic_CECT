# Phase-detection retrain + improvement plan

Written after auditing the prior, messy codebase at
`/Users/macbook/develope/thesis/contrastPhaseGen_disentangledApproach` and the
result artifacts it left behind (`contrast_phase_results*/experiment_summary.json`,
`test_results.json`). Goal (user's): own this module, retrain on the **full**
dataset (previous runs were on a subset), evaluate honestly on held-out data, make
the input compatible with the synthesis training pipeline (patch input, with a 2D →
3D → bigger-patch fallback ladder), and improve on the train=100% / generalization≈68%
overfitting gap.

---

## 1. What the old numbers actually are (verified, not relayed)

| experiment | encoder | train | 5-fold CV | test | note |
|---|---|---|---|---|---|
| results (Sep 3) | Dino_v3 | 100% | 68.7% ±6.4 | 97.3% | 4-class? see below |
| results (Sep 3) | MedViT | 99.3% | 68.5% ±0.5 | 96.2% | |
| results (Sep 3) | TimmViT | 84% | 32.5% | 75.2% | **3 classes, support 175/174/175** |
| results_2 (Sep 2) | MedViT | 100% | 73.3% ±5.2 | 98.5% | |

**The ~68% CV is real and consistent — but it is *optimistic*, not the floor.**
Two compounding methodology problems make both the 68% CV and the 96% test
untrustworthy as generalization estimates:

### 1a. Volume duplication (the big one)

The classifier is trained on features extracted from `batch['input_path']` over the
**generation-pair** dataset produced by `data.py:prepare_dataset_from_folders` — which
emits every directed `phase→phase` pair per scan. With 3 phases/scan, each volume is
the "input" of exactly 2 pairs, so every unique volume's feature vector is duplicated
~2× (more with 4 phases) carrying the same label. Evidence: the TimmViT test set has
support **175/174/175** — a *perfectly balanced* 3-class test set, which real phase
data never is; that balance is the arithmetic signature of "each of ~262 unique
volumes counted twice." So:
- `train=100%` is partly trivial — the LDA memorizes near-identical duplicates.
- `StratifiedKFold` (sample-level, used in `phase_detector.py:155`) scatters a
  volume's duplicate twin across folds → **the held-out fold contains copies of
  training volumes** → the 68% CV is inflated by within-fold leakage.
- Real deduplicated, patient-grouped accuracy is therefore likely **below 68%**.

### 1b. Frozen encoder + LDA with p ≈ n

256 encoder features vs ~140–260 unique volumes and 3–4 classes. LDA on that many
features with that few samples separates the training set almost perfectly by
chance (high variance, `cv_std` up to ±6.4). This is classic small-sample
high-dimensional overfitting, independent of 1a.

**Bottom line:** ~68% is not a hard ceiling *or* a trustworthy score — it's an
optimistic estimate of a pipeline with two fixable flaws. Fixing them will likely
*lower* the reported number first (honest), then the improvements below aim to raise
the real one.

---

## 2. Improvement roadmap (priority order)

Cheapest/highest-leverage first. Items 1–4 are classifier-head changes (no encoder
retraining, fast to iterate); 5–6 are heavier.

1. **Deduplicate to one feature vector per unique volume.** The single most important
   fix. `data.py:create_phase_detection_dataset` already exists for this but has a
   schema bug (expects `volume`/`phase` keys, but `prepare_dataset_from_folders`
   emits `input_path`/`input_phase`) — so it was likely never used. Dedup by volume
   file path, keep one (features, phase, scan_id) per volume.
2. **Patient-grouped CV** (`GroupKFold(groups=scan_id)`) instead of `StratifiedKFold`.
   Removes cross-fold patient/duplicate leakage → the honest number. Do this and (1)
   together; expect the reported CV to drop, that's correct.
3. **Shrinkage LDA**: `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')`.
   Directly regularizes the p≈n overfitting (item 1b) — usually the biggest single
   accuracy win for LDA on high-dim features, one-line change.
4. **Dimensionality reduction before LDA**: PCA 256 → ~30–50 comps (fit on train
   only). Complements shrinkage; also makes the classifier cheaper to use later as a
   loss.
5. **Fine-tune the encoder** (unfreeze, train end-to-end with a small classification
   head + cross-entropy) instead of frozen+LDA. The largest potential lever, since
   the frozen encoders were pretrained on ImageNet/generic-medical data, never on
   *this* phase task. Heavier (needs GPU, careful LR/regularization), do after 1–4
   establish an honest baseline.
6. **Use the full dataset** (user's own instinct). The old runs used a subset; the
   full vindr abdomen set (the old `data_path` was
   `../ncct_cect/vindr_ds/original_volumes/Abdomen/raw_image/`) has more scans/phases
   than the 137 NCCT/venous pairs the *synthesis* pipeline filters down to. More
   unique volumes directly attacks the p≈n problem (item 1b).

Also worth checking, cheap: per-class support / imbalance (use balanced LDA priors if
skewed), and whether Delayed phase is present (the TimmViT run had only 3 classes).

---

## 3. Input compatibility with the synthesis pipeline (the patch ladder)

The synthesis trainer consumes `[0,1]`-normalized patches (2D 128×128 today, 3D
`8×64×64` planned). The eventual goal is to use this classifier **as a loss** on the
generator's output, so it must accept the same patch tensors. Two things to resolve,
empirically, in the order the user specified:

- **Phase is a partly *global* property.** Contrast enhancement is read from
  aorta/portal-vein/liver/kidney enhancement patterns. A single 128×128 2D patch of,
  say, muscle or fat may carry almost no phase signal — so a patch-level classifier
  may have a low accuracy ceiling *by construction*, not from a code bug. This is
  exactly why the user's fallback ladder is the right experiment:
  1. **2D 128×128 patches** — cheapest, reuses the synthesis `dataset.py` pipeline
     directly. Try first; measure per-patch accuracy honestly (GroupKFold by scan).
  2. **3D patches** (`8×64×64` etc.) — if 2D is too low, depth gives cross-slice
     enhancement context. The synthesis `dataset.py` + smoke test already support
     `dims=3`.
  3. **Bigger patches / whole-slice / whole-volume** — if 3D patches still too low,
     fall back toward the old whole-volume approach (which is where the ~68% came
     from). At the limit this converges to "classify the whole volume," i.e. the
     original method.
- **Reuse the synthesis `dataset.py`, don't port the old `data.py`.** The synthesis
  repo's `find_pairs_and_split` already loads this exact dataset + labels cleanly and
  is tested; building patch-level phase labels on top of it (source patch → NCCT=phase
  0, target patch → its phase id) is less code and guarantees the classifier sees the
  identical input distribution the generator produces. The old `data.py` is
  volume/pair-oriented, has hardcoded paths and a `'registered'`-filename filter, and
  carries the schema bug from §2.1 — port only ideas from it, not the file.

---

## 4. Concrete next steps

**Build (I can do locally, verifiable without GPU):**
- A clean `phase_classifier.py` in `phase_detection/` implementing items 1–4 (dedup,
  GroupKFold, shrinkage LDA, optional PCA) as the classification head, decoupled from
  feature extraction so the same head works on volume features *or* patch features.
- Smoke-test the head logic on synthetic features (same pattern as `smoke_test.py` /
  `test_patch_cache.py`): assert dedup collapses duplicates, GroupKFold never puts a
  scan_id in both folds, shrinkage/PCA path runs — all CPU-only, no data needed.

**Run (needs the remote GPU + full data):**
- Feature extraction with a real encoder (MedViT recommended: same CV as Dino_v3 but
  13× lower variance, and no silent-fallback bug — see §12 of `../IMPLEMENTATION.md`).
- The 2D→3D→bigger patch ladder for the loss-compatible variant.

**Not yet resolvable (blockers to flag):**
- MedViT pretrained weights must actually be present at `--medvit_pretrained_path`
  (`phase_detector.py` silently trains from random init if missing).
- No trained classifier checkpoints survive from either codebase — a fresh train is
  required regardless.

---

## 5. Honest expectation-setting

Doing items 1–2 first will likely make the reported number **drop from 68%** to
whatever the deduplicated, patient-grouped truth is — that's the point, it replaces an
inflated number with a real one. Items 3–6 then try to raise the *real* number. If
even the whole-volume, full-data, fine-tuned version lands around 70–80%, that may
simply be the achievable accuracy for 3–4 phase discrimination on this dataset — in
which case using it as a *soft*, class-distance-weighted loss with a low weight (per
`../phase_conditioning_plan.md`) is the right way to consume an imperfect classifier,
rather than as a hard argmax target.
