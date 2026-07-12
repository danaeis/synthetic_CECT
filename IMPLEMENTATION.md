# literature_baseline — implementation reference

This documents what is **actually implemented** in this directory right now
(not the design intent). Re-verify against the code before trusting a claim
here after further edits — this file is a snapshot, not a live view.

Scope: 2-D NCCT → CECT image translation (single target contrast phase),
U-Net generator + PatchGAN discriminator, a configurable composite loss with
10 optional loss terms. 3-D code paths exist throughout but are untested
(default config is `patch_depth=1`, `dims=2`).

---

## 1. Data pipeline — `dataset.py`

### File discovery (`find_pairs_and_split`)
- Scans `cfg['data_dir']` for per-case subdirectories. In each, globs
  `*{file_tag}.nii.gz` (default `file_tag='_deeds'`) and classifies each file
  into a phase via `labels_csv` (matched on `StudyInstanceUID`/
  `SeriesInstanceUID`) or, if not found there, via keyword matching on the
  filename (`_infer_phase`, `dataset.py:57-70`).
- A case becomes a training pair only if it has both a `'non-contrast'` and
  the configured `target_phase` volume. `pairs.append(...)` (`dataset.py:130-140`)
  builds `{'source_path', 'target_path', 'seg_path', 'case_id'}`.
- `seg_path` is derived by string substitution: `target_path` with
  `{file_tag}.nii.gz` replaced by `{file_tag}_seg_reg.nii.gz`. This assumes
  the on-disk convention `..._deeds.nii.gz` / `..._deeds_seg_reg.nii.gz`
  (confirmed present in `sample_data_reg/.../B2_deeds__aligned/<case>/`). If
  that file doesn't exist, `seg_path` is `None` — no error, just no mask for
  that pair.
- Case-level split (not slice-level) into train/val/test using
  `np.random.default_rng(seed).permutation` — `test_split`/`val_split`
  fractions of *cases*, remainder is train.

### `CTPairDataset`
Three-phase construction, all in `__init__`:
1. **Index** (`dataset.py:232-295`): for every pair, load both volumes
   (`_load_vol`, LRU-cached, `maxsize=80`), slide a window over
   `(z, y, x)` at `stride = patch_size * (1 - overlap)`, keep a coordinate
   only if the **source (NCCT)** patch passes `std >= min_patch_std`,
   `mean >= min_patch_mean`, `max >= min_patch_max` (all in raw HU, before
   normalization). Target/mask validity is not checked independently.
2. **Sub-sample**: if more valid coordinates than `max_patches`, uniformly
   random-subsample (`rng.choice`, no replacement) down to `max_patches`.
3. **Preload** (`dataset.py:310-353`): for every surviving coordinate, crop
   source/target, `np.clip(hu_min, hu_max)` then rescale to `[0, 1]`, append
   to `self.src_patches` / `self.tgt_patches` (plain Python lists, fully in
   RAM). If `self.load_mask` is true (see below), also crop the segmentation
   volume at the same coordinate and binarize with `> 0` → `self.mask_patches`;
   if `seg_path` was `None` or the load/shape-check fails, that patch's mask
   is an all-zero array instead of raising.

`self.load_mask = cfg.get('use_organ', False) or cfg.get('use_seg_consistency', False)`
(`dataset.py:228`) — segmentation volumes are only read from disk and held in
RAM when one of those two loss flags is on. With both off, mask I/O is
skipped entirely and `self.mask_patches = None`.

`__getitem__` (`dataset.py:359-365`) returns `{'source': (1,H,W), 'target':
(1,H,W)}`, plus `'mask': (1,H,W)` **only if** `self.mask_patches` is
non-empty (a `CTPairDataset` instance either always has masks or never does —
consistent within one DataLoader, so default batch collation works).

A patch grid PNG (`{split}_patch_grid.png`) is saved once per split for
visual sanity-checking, in `cfg.get('out_dir', Path('.'))/patch_grids/`.

### On-disk patch cache
`CTPairDataset.__init__` checks a cache file before indexing (`dataset.py`,
right after the geometry/threshold setup, before Step 1). The cache key is an
md5 hash of everything that determines the preloaded content: sorted
`(source_path, target_path)` pairs, patch geometry, validity thresholds,
`hu_min`/`hu_max`, `max_patches`, `seed`, `load_mask`, `split_name` — see
`_cache_path`. Cache files live under `cfg['cache_dir']`
(`config.py`'s `CACHE_DIR`, default `../../simlified_train/patch_cache`), a
location **fixed independent of `output_dir`**, so scenario runs that only
change loss flags (different `output_dir` per scenario, same data config)
share one cache instead of re-preloading from scratch. On a cache hit,
indexing and preload are skipped entirely; on a miss, the existing
index→subsample→preload path runs as before and then writes the cache
(`np.savez`, uncompressed — files are multi-GB for large patch counts, by
design, since reading them beats re-preloading).

### Preload performance fix (formerly a known limitation)
`_load_vol`'s LRU cache was `maxsize=80`, but a typical train split touches
far more distinct volumes (e.g. 97 pairs → 194 volumes). Because the
preload loop iterated `coords` in the random order produced by sub-sampling
(`rng.choice`), it constantly evicted and reloaded full volumes from disk —
the actual cause of a 3-hour preload for 20,000 patches in one observed run
(vs. ~1.4s for a val split whose volume count fit inside 80 slots). Fixed by
sorting `coords` by `(src_path, tgt_path)` right after sub-sampling, so the
preload loop touches each volume in one contiguous run regardless of cache
size, plus bumping `maxsize` to 256 as headroom.

---

## 2. Model architecture — `models.py`

### `UNetGenerator`
- 4-level 2-D/3-D U-Net (`dims` parameter selects `Conv2d`/`Conv3d` etc. via
  the `_conv`/`_convT`/`_inorm`/`_pool`/`_drop` helpers, `models.py:29-34`).
- Encoder: `InstanceNorm` + `LeakyReLU(0.2)` blocks, channels
  `1→64→128→256→512` (`base_channels=64` default), no dropout in the encoder
  (`dropout=0.0` hardcoded per `_EncBlock` call).
- Bottleneck: 2× conv at 512 channels with dropout (`generator_dropout`,
  default `0.2`).
- Decoder: transposed-conv upsampling with skip concatenation at every
  level, dropout in every decoder block.
- **Output activation is `nn.Sigmoid()`** (`models.py:152`), matching the
  dataset's `[0, 1]` normalization. (Previously `nn.Tanh()` — changed
  because the data was never `[-1, 1]`; see §7.)
- `dims=3` uses `pool_stride=(1,2,2)` by default (keeps depth, downsamples
  H/W only) — appropriate for thin patches (8–16 slices), not exercised by
  the current 2-D default config.

### `PatchGANDiscriminator`
- 70×70-receptive-field PatchGAN, 4 strided blocks (`n_layers=4` default),
  `ndf=64`, channels double each block up to 512, `BatchNorm` (except block
  0, which has no norm, standard PatchGAN convention).
- `forward(x, return_features=True)` returns `(logits, [block_outputs])` —
  the feature list is every intermediate block's output (used by
  `FeatureMatchingLoss`), always computed and returned when
  `return_features=True` regardless of whether feature matching is actually
  enabled (minor: the training loop calls this with `return_features=True`
  during the D-step precisely so `FeatureMatchingLoss` has real features to
  consume; see §4).

---

## 3. DINO backbone — `dino_backbone.py`

`DinoSpatialBackbone` is a **frozen** ViT feature extractor returning
*spatial* patch-token maps `List[(B, C, h, w)]` (not a pooled/CLS vector) —
built specifically so it can substitute for VGG's conv feature maps in a
perceptual loss, or be spatially diffed for a saliency map. It is **not**
the same object as `phase_detection/dino_encoder.py`'s `DinoV3Encoder`,
which is a separate, unrelated whole-volume classification encoder (see §8).

Loading is a priority chain, tried once at construction (`dino_backbone.py:97-113`):
1. `transformers.AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m')`
   — real DINOv3. Requires `transformers` installed and the gated HF license
   accepted; **not installed** in the environment checked during this
   session (`transformers` absent).
2. First `timm.list_models()` entry containing `'dinov3'` — real DINOv3 if
   the installed `timm` ships it. **Not verified** whether `timm==1.0.22`
   (the version confirmed installed) has any such entry — the check command
   is in this file's docstring; hasn't been run.
3. `timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)`
   — real DINOv2, ungated. **Confirmed compatible** with the installed
   `timm==1.0.22` (import/API check only — actual weight download not
   performed in this session, no network access here).

Whichever succeeds is logged at `WARNING` level with its exact source string
(`"DINO backbone loaded: timm:vit_small_patch14_dinov2.lvd142m ..."` etc.)
so it's never ambiguous which weights were actually used, unlike
`dino_encoder.py`. If all three fail, `DinoSpatialBackbone()` raises — there
is no silent fallback to random weights here.

`forward(x)`: resizes to 224×224, converts 1→3 channel, ImageNet-normalizes,
and returns 3 intermediate feature maps (shallow/mid/deep, indices computed
as fractions of network depth) in `NCHW` spatial layout:
- `timm` path uses `model.forward_intermediates(x, indices=idx,
  output_fmt='NCHW', intermediates_only=True)` — **this API has not been
  runtime-tested against the installed timm version in this session**
  (no GPU/network access here); if `forward_intermediates` signature differs
  in practice, this will raise at first use, not silently misbehave.
- `transformers` path manually reads `output.hidden_states`, drops CLS (+
  register tokens if the model config exposes `num_register_tokens`), and
  reshapes to a spatial grid.

No `@torch.no_grad()` on `forward()` — parameters are frozen via
`requires_grad_(False)`, but gradients w.r.t. the *input* must still flow
through for `DinoPerceptualLoss` to train the generator. Callers that only
need a static weight map (`DinoSaliencyLoss`) wrap their own call site in
`torch.no_grad()`.

---

## 4. Losses — `losses.py`

All ten losses are real classes; `CompositeLoss` is the only thing the
trainer calls directly. Each sub-loss is instantiated **only if its
`use_*` flag is true** (`losses.py:496-572`) — disabled losses cost nothing
at init or per-step.

| # | Class | What it actually computes | Gate | Wired with real data? |
|---|---|---|---|---|
| 1 | `AdversarialLoss` | LSGAN (`mse_loss` vs 0/1) or BCE (label-smoothed 0.9/0 for real/fake) `disc_loss`/`gen_loss` | `use_adversarial` (default **on**) | Yes |
| 2 | `PerceptualLoss` | VGG16 `IMAGENET1K_V1` weights, L1 on 4 ReLU-layer feature maps (`relu1_2..relu4_3`), equal-weighted average | `use_perceptual` (default **on**), `perceptual_backbone='vgg'` (default) | Yes |
| 2b | `DinoPerceptualLoss` | Same idea, but 3 spatial feature maps from the shared `DinoSpatialBackbone` instead of VGG | `perceptual_backbone='dino'` | Yes (new) |
| 3 | `FeatureMatchingLoss` | L1 between discriminator's real vs fake intermediate block outputs, averaged over blocks | `use_feature_matching` (default **on**) | Yes, but only on steps where the discriminator actually runs (`global_step % disc_update_freq == 0`; at default `disc_update_freq=1` this is every step) |
| 4 | `SSIMLoss` | 1 − Gaussian-window SSIM (11×11, σ=1.5) | `use_ssim` (default off) | Yes |
| 5 | `GradientLoss` | L1 on Sobel gradient magnitude (2-D: x/y; 3-D: x/y/z approximation) | `use_gradient` (default off) | Yes |
| 6 | `FrequencyLoss` | L1 on `\|fft2(x, norm='ortho')\|` amplitude spectrum, no windowing/fftshift | `use_frequency` (default off) | Yes |
| 7 | `OrganWeightedLoss` | MSE weighted `organ_weight×` inside `mask`; **plain unweighted MSE if `mask` is `None`** | `use_organ` (default off) | Yes — mask now comes from `dataset.py`'s `_seg_reg.nii.gz` loading (see §1); patches with no on-disk mask silently get the unweighted fallback for that patch only |
| 8 | `PhaseSaliencyLoss` | MSE weighted `saliency_weight×` where `\|target-source\| > threshold` (raw-intensity heuristic) | `use_saliency` (default off), `saliency_mode='heuristic'` (default) | Yes |
| 8b | `DinoSaliencyLoss` | MSE weighted by a spatial map of `‖DINO(target) − DINO(source)‖` (normalized per-sample to its own max), using the deepest of the 3 shared-backbone feature maps, upsampled to patch resolution | `saliency_mode='dino'` | Yes (new) |
| 9 | `CyclicConsistencyLoss` | L1 between `G(G(source))` and `source` | `use_cycle` (default off) | Yes — see §6 for the single-generator caveat |
| 10 | `SegmentationConsistencyLoss` | L1 between Sobel edge maps of pred/target, optionally multiplied by `mask` (same organ mask as #7) before the L1; **identical to `GradientLoss` if `mask` is `None`** | `use_seg_consistency` (default off) | Yes, same mask source as `OrganWeightedLoss` |

### DINO backbone sharing
`CompositeLoss._get_dino_backbone()` (`losses.py:574-579`) lazily constructs
one `DinoSpatialBackbone` on first call and caches it on
`self._dino_backbone`; both `perceptual_backbone='dino'` and
`saliency_mode='dino'` call this same method, so at most one backbone is
ever loaded regardless of how many DINO-based losses are enabled — and none
is loaded if both are left at their `'vgg'`/`'heuristic'` defaults.

### Warmup schedules
- `_adv_w()`: adversarial weight ramps `0 → lambda_adv` linearly over
  `adv_warmup_epochs` (default 5), via `self._epoch` set each epoch by
  `Trainer.train()` calling `criterion.set_epoch(epoch)`.
- `_cycle_w()`: same mechanism for `lambda_cycle`, over
  `cycle_warmup_epochs` (default 5). Added because a single shared generator
  used for both directions (`G(source)=target` and `G(G(source))=source`)
  has two objectives that can conflict before `G` has learned anything —
  ramping avoids the cycle loss dominating early training.

### `CompositeLoss.forward` returns
`(total_loss: Tensor, loss_dict: Dict[str, float])` — every entry in
`loss_dict` (`l1`, `adversarial`, `perceptual`, `feature_matching`, `ssim`,
`gradient`, `frequency`, `organ`, `saliency`, `cycle`, `seg_consistency`,
`total`) is always present, `0.0` for disabled losses.

---

## 5. Trainer — `trainer.py`

One `Trainer` owns `G`, `D` (`None` if `use_adversarial=False`),
`opt_G`/`opt_D` (Adam, `betas=(0.5,0.999)`), an optional
`CosineAnnealingWarmRestarts` scheduler on `opt_G` only, `CompositeLoss`,
and two `GradScaler`s for AMP (`use_mixed_precision`, default on).

Per training step (`_train_step`, `trainer.py:201-262`):
1. Pull `source`, `target`, and `mask` (if present in the batch) to device.
2. **If** it's a discriminator-update step (`global_step % disc_update_freq
   == 0`, default every step): forward `G(source)` under `torch.no_grad()`
   to get `fake_for_d` (no autograd graph built — this is deliberately
   cheaper than the old version, which built a full backward-capable graph
   here only to discard it), then run `_disc_step(target, fake_for_d)`
   (forwards `D` on real+fake, LSGAN/BCE disc loss, backward, grad-clip to
   10.0, optimizer step). Returns the real-image discriminator features for
   feature matching.
3. **Generator step**: zero `opt_G`, forward `fake = G(source)` again (this
   second forward *does* need gradients — deliberate, not the same
   redundancy as before), forward `D(fake, return_features=True)` if `D`
   exists, optionally forward `cycle_pred = G(fake)` if `use_cycle`, call
   `CompositeLoss(pred=fake, target=target, source=source, mask=mask, ...)`,
   backward with grad scaling, grad-clip to 10.0, optimizer step.
4. Returns a flat dict of scalar loss values (all of `CompositeLoss`'s
   `loss_dict` entries plus `disc`) for logging/history.

Validation (`_validate`, `@torch.no_grad`): forward-only L1/PSNR/SSIM on
`val_loader`, PSNR/SSIM computed with a hand-rolled global (non-windowed)
formula in `trainer.py:44-58`, evaluated on the first item of each batch's
centre slice if 3-D.

Per-epoch (`train()`, `trainer.py:404-473`): one pass over `train_loader`
via `_train_step`, cosine scheduler step, `_validate`, JSON history dump
every epoch, matplotlib curve plot every 5 epochs, checkpoint save every
epoch (rolling keep of `keep_last_n_checkpoints`, plus a separate
`best_model.pth` on new best `val_loss`), sample grid PNG every
`save_samples_interval` epochs (rolling keep of
`keep_last_n_sample_epochs`), `EarlyStopping` on `val_loss` with
`early_stop_patience` (default 12), periodic `torch.cuda.empty_cache()` +
`gc.collect()` every 10 epochs.

`load_checkpoint`/`_save_checkpoint` persist `G`/`opt_G` state always,
`D`/`opt_D` state if `D` exists — `CompositeLoss` (and any DINO backbone it
holds) is **not** checkpointed, since it's either stateless or frozen.

---

## 6. Cycle consistency — architectural note

There is only **one** generator `G: NCCT → CECT`. `use_cycle=True` computes
`cycle_pred = G(fake) = G(G(source))` and pushes it toward `source` via L1.
Since the same weights must satisfy both `G(source) ≈ target` and
`G(G(source)) ≈ source`, `G` is effectively being trained to act as an
*involution* — its own inverse — conditioned implicitly on which phase its
input looks like (NCCT vs CECT are visually distinguishable, so this is
plausible, but it is a non-standard, harder objective than a classic
two-generator CycleGAN). This was a deliberate choice (see conversation
history) — not a bug — mitigated by the `cycle_warmup_epochs` ramp in §4.

---

## 7. What changed from the original version of this code (this session)

For future reference — these were real bugs/gaps in the code before this
session, now fixed:

1. **Generator output was `nn.Tanh()` (`[-1,1]`) while the dataset
   normalizes to `[0,1]`.** Fixed to `nn.Sigmoid()`. This also fixed a
   latent bug in `PerceptualLoss._to_vgg`, which used to conditionally
   rescale input by `(x+1)*0.5` only when `x.min() < 0` — an inconsistent,
   batch-dependent heuristic that's now removed entirely (input is always
   `[0,1]` by construction).
2. **`OrganWeightedLoss` and `SegmentationConsistencyLoss` never received a
   real mask** — `dataset.py` never loaded one and `trainer.py` never
   passed one to `CompositeLoss.forward`, so both losses silently degraded
   to their unweighted fallback whenever enabled. Fixed: `dataset.py` now
   discovers and loads the `*_seg_reg.nii.gz` masks that already exist on
   disk (gated behind `use_organ`/`use_seg_consistency` so it costs nothing
   when disabled), and `trainer.py` passes `mask=batch.get('mask')` through.
3. **Redundant full generator forward pass every training step** — the
   discriminator's fake input used to be computed with gradient tracking
   enabled, then immediately `.detach()`'d and thrown away. Now computed
   under `torch.no_grad()`.
4. **No DINO-based perceptual/saliency option** — added `DinoPerceptualLoss`,
   `DinoSaliencyLoss`, `dino_backbone.py`'s priority-ordered loader, sharing
   one backbone instance, gated by `perceptual_backbone`/`saliency_mode`.
5. **No cycle-loss warmup** — added `cycle_warmup_epochs`, mirroring the
   existing `adv_warmup_epochs` pattern.
6. **HU clip window (`[-200, 300]`) was a fixed literature guess, not
   data-driven.** `analyze_hu_range.py` (new) computes a percentile-based
   recommendation from the actual dataset.

---

## 8. `phase_detection/` — separate, unrelated to the training loop

This subdirectory (`dino_encoder.py`, `medViT_encoder.py`,
`phase_detector.py`, `*_feature_visualization.py`, `MedViT/`) is an
**offline contrast-phase classification experiment** (encoder → pooled
volume feature vector → LDA), not imported by `losses.py`, `trainer.py`,
`models.py`, or `train.py`. It's a separate research tool for validating
that phase-discriminative signal exists in various pretrained encoders
(results referenced in conversation: DINOv3-labeled encoder reached 97.3%
test accuracy vs MedViT 96.2%, TimmViT 75.2%).

**`dino_encoder.py`'s `DinoV3Encoder` does not reliably load real DINOv3
weights.** Its `__init__` (`phase_detection/dino_encoder.py:16-84`) tries,
in order: `torchvision.models.dinov3_small/base/large` (does not exist in
any torchvision release, confirmed `ImportError` on the checked
`torchvision==0.21.0`), then `torch.hub.load('facebookresearch/dino',
'dino_vits16', ...)` (this is **DINO v1**, 2021, not v3), then
`torchvision.models.vit_b_16(pretrained=True)` (ImageNet-*supervised* ViT,
not self-supervised DINO at all), then a fully randomly-initialized custom
ViT (`_create_custom_vit`) if all else fails — with **no logged distinction**
between these outcomes; all print `"✅ Loaded ... model"` and the class is
used downstream as `'Dino_v3'` regardless of which branch actually fired.
The reported 97.3% accuracy was very likely produced by the DINO v1 or
ImageNet-ViT fallback, not DINOv3. **Not fixed in this session** (it's a
standalone analysis script, out of scope for the training-loop loss work);
`dino_backbone.py` (§3) was written as a new, correctly-labeled loader for
the training loop specifically to avoid repeating this failure mode.

`MedViT/` is a vendored copy of the MedViT repository (architecture +
training utilities), used by `medViT_encoder.py` for the same offline
classification experiment.

---

## 9. Config flags reference — `config.py` / `train_config`

Every key in `train_config` maps 1:1 to a `TRAIN_CASE` constant defined
above it in the same file; see `config.py` for exact current values. Loss
`use_*` flags and `lambda_*` weights are listed in the `CompositeLoss`
table (§4). Notable non-loss keys:

- `hu_min=-200, hu_max=400` — clip bounds before `[0,1]` rescale. Data-driven
  via `analyze_hu_range.py`, but not its raw percentile output — see §10.
- `patch_size=128, patch_depth=1, overlap=0.5, dims=2` — 2-D by default;
  3-D requires setting `patch_depth>1` and `dims=3` together.
- `disc_update_freq=1` — if raised above 1, `FeatureMatchingLoss` silently
  contributes `0` on the skipped steps (real discriminator features are
  only produced on D-update steps).
- `perceptual_backbone='vgg'`, `saliency_mode='heuristic'` — the new DINO
  paths are opt-in, not default.

`train.py`'s CLI exposes `--use_X`/`--no_X` for every loss flag,
`--perceptual_backbone`, `--saliency_mode`, plus the usual
epochs/batch_size/lr/patch/dims/resume overrides. Every resolved config is
dumped to `{output_dir}/run_config.json` at the start of each run.

---

## 10. Session 2026-07-01 (part 2) — HU clip fix + trainer bug fixes

Prompted by a real run of `analyze_hu_range.py` (137 pairs, 30-case sample,
402.7M pooled voxels). Findings and fixes:

1. **`analyze_hu_range.py`'s raw percentile recommendation ([-900, 690] at
   p0.5-p99.5) is not safe to adopt as-is.** The logged percentiles
   (`p0.1=-899.0 .. p5=-881.0`) show a large voxel mass sitting right above
   the script's `AIR_FLOOR=-900` "crude air exclusion" — almost certainly
   lung bases / fat / partial-volume air-tissue edges, none of which carries
   NCCT<->CECT contrast-enhancement signal (contrast agent doesn't
   accumulate there). The *current* `hu_min=-200` already excludes most of
   this mass (only ~25% of pooled voxels fall below it, consistent with
   `p25=-113.4`). Blindly lowering `hu_min` to -900 would spend most of the
   normalised `[0,1]` range on that non-informative tail and compress the
   real soft-tissue/vessel enhancement band (`p25=-113.4 .. p99=524.7`) into
   a much smaller slice of it — a regression, not an improvement.
   **Fix applied:** kept `hu_min=-200` (already sound), raised `hu_max` from
   300 to 400 to keep more of the arterial/venous vessel-enhancement peak
   without pulling in the calcification/bone range that starts around
   `p99.5=691` / `p99.9=1049`. `analyze_hu_range.py` now also detects and
   warns about this specific low-end contamination pattern automatically
   (`contamination_margin` check after computing the recommended window),
   so future reruns don't require re-deriving this reasoning by hand.
2. **AMP hardcoded to `'cuda'` regardless of actual device.**
   `use_mixed_precision` defaulted to `True` independent of whether CUDA was
   actually available; `GradScaler('cuda', enabled=True)` /
   `autocast('cuda', enabled=True)` assume a CUDA context and error on a
   CPU-only machine even though `enabled` looks like a runtime toggle. Fixed
   in `trainer.py`: `self.use_amp = config.get('use_mixed_precision', True)
   and self.device == 'cuda'`.
3. **PSNR/SSIM validation metrics used a per-sample `data_range`** (derived
   from that sample's own `target.max() - target.min()`) instead of the
   fixed `1.0` that the `[0,1]`-normalised pipeline actually uses. This made
   `val_psnr`/`val_ssim` incomparable across patches/epochs — a near-uniform
   patch (small true dynamic range) got an arbitrary, unrelated
   `data_range`. Fixed in `trainer.py`'s `_psnr_ssim` to always pass `1.0`.
4. **`CTPairDataset._save_patch_grid` could crash (`IndexError` or
   `rows=0`) whenever a split preloaded fewer than 16 patches** — `rows =
   n // (cols // 2)` under-allocated subplot rows for any `n` not an exact
   multiple of 4 (e.g. `n=7` allocates 1 row / 8 axes but indexes up to
   `axes[13]`), and floored to 0 rows entirely for `n < 4`. Fixed with a
   ceiling-division row count in `dataset.py`.
5. **Unused `from sklearn.model_selection import train_test_split` import**
   in `dataset.py` (dead — the actual split logic is the hand-rolled
   `np.random.default_rng(seed).permutation` in `find_pairs_and_split`).
   Removed; `scikit-learn` is no longer a real dependency of this module.
6. Added `requirements.txt` for this directory, pinned to what was confirmed
   installed on the training server this session (`torch==2.6.0+cu124`,
   `torchvision==0.21.0+cu124`, `timm==1.0.22`, no `transformers`).

---

## 11. Session 2026-07-08 — history `KeyError` crash, preload fix, patch cache, tooling

1. **Training crashed with `KeyError` right after epoch 1, on every run.**
   `_train_step` returns a dict keyed `'gen_total'` (`trainer.py:258`);
   `_update_history` builds history keys as `f'train_{k}'` for `k` including
   `'gen_total'` (`trainer.py:366-368`), i.e. it appends to
   `self.history['train_gen_total']`. But `self.history` was initialized with
   `'train_total'` instead (`trainer.py:175`) — a typo, since `_plot_history`
   (`trainer.py:385`) already correctly expected `'train_gen_total'`. Fixed by
   renaming the init key to `'train_gen_total'`. This affected every
   configuration, not just a specific loss combination.
2. **Preload performance fix + on-disk patch cache** — see §1's "Preload
   performance fix" and "On-disk patch cache" subsections above. Turns a
   ~3-hour train-split preload into effectively one-time work, shared across
   scenario runs that only differ in loss flags.
3. **Added `smoke_test.py`** — a GPU-free structural smoke test that runs the
   `Trainer` for 2 epochs against tiny synthetic tensors (no real data) across
   every loss-flag combination (baseline, pix2pixHD combo, each extra flag
   individually, all-on), to catch bugs like the one above in seconds instead
   of after a multi-hour remote preload + training run. Verified locally:
   `l1_only` (the exact scenario that crashed) now passes; all non-perceptual
   scenarios pass; perceptual-loss scenarios need `torchvision` (present on
   the training server per `requirements.txt`, absent on the machine used to
   write this fix — expected, reported as a clear `FAIL` with the existing
   `losses.py` error message, not a silent failure).
4. **Added `run_scenarios.sh`** — sequential runner over `train.py`'s existing
   `--use_X`/`--no_X` CLI flags, one scenario at a time, each with its own
   `output_dir` and log file, stop-on-first-failure by default. The
   `SCENARIOS` array is left for the user to fill in.

---

## 12. Session 2026-07-09 — real-data mask coverage check, cache correctness test, 3-D smoke coverage, phase_detection feasibility

1. **Added `check_seg_masks.py`** — diagnostic for `use_organ`/
   `use_seg_consistency`'s real data dependency: `dataset.py` silently falls
   back to an all-zero mask whenever a case's `seg_path` is missing, fails to
   load, or shape-mismatches (no error raised — see §1's `CTPairDataset`
   notes) — so a data directory with poor mask coverage would train those two
   losses against mostly-empty masks without ever showing up as a bug. This
   script reuses `find_pairs_and_split` (same pairing logic as `train.py`)
   and reports, per split: missing/failed/shape-mismatched counts and
   non-zero-voxel coverage stats (min/median/max) across cases with a
   loadable mask. Must be run where the real data lives (not runnable
   locally in this dev environment — no `sample_data_reg/` checkout here).
2. **Added `test_patch_cache.py`** — correctness test for the on-disk patch
   cache added in §11 (`smoke_test.py` never exercises `dataset.py` at all,
   only `Trainer`/`CompositeLoss` against synthetic tensors, so the cache
   mechanism itself had no test coverage). Builds tiny synthetic NIfTI
   volumes (nibabel) and drives `CTPairDataset` directly through: (a) first
   construction — cache miss, preload from scratch, cache file written; (b)
   second construction, identical config — cache hit, and the reloaded
   src/tgt/mask patch arrays are asserted **bit-identical** (`np.array_equal`)
   to the original preload, not just "didn't crash"; (c) third construction
   with `patch_size` changed — resolves to a different cache file (no stale
   reuse across incompatible configs). All 9/9 checks passed locally (run via
   the `dicom_processing` conda env, which has `nibabel`).
3. **Extended `smoke_test.py` with 3-D coverage** — `_FakeDataset`/
   `_fake_loaders` now take a `dims` parameter (3-D yields `(1, 8, 64, 64)`
   volumes instead of `(1, 64, 64)` slices; depth=8 is arbitrary since
   `pool_stride=(1,2,2)` never pools the depth axis for `dims=3`, per
   `models.py`). Added `3d_l1_only` and `3d_all_losses` scenarios (not the
   full 2-D matrix — 3-D convs are heavier; `all_losses` alone exercises
   every loss's 3-D branch at once: `GradientLoss`'s 3-D Sobel, `SSIMLoss`'s
   per-slice loop, `FrequencyLoss`'s per-slice FFT, `PerceptualLoss`'s
   depth-flatten, `SegmentationConsistencyLoss`, the 3-D `PatchGANDiscriminator`,
   etc). Both pass locally with `use_perceptual` forced off (`torchvision`
   unavailable in this dev environment only — see §11 item 3); `3d_l1_only`
   also passes as part of the full default-config smoke run.
4. **Investigated `phase_detection/` for two proposed new losses** (a
   region-importance/saliency loss and a "does the generator's output
   classify as the target phase" consistency loss, both requested for
   `extra_organ`-style testing). **Not implemented — blocked, not just
   deferred:**
   - `phase_detector.py`'s `ContrastPhaseClassifier` has **no trained
     checkpoint anywhere** (searched for `.pth`/`.pt`/`.ckpt`/`.pkl` — none
     exist; `phase_detector.py`'s `main()` is a training script whose output
     dir, `contrast_phase_results/`, doesn't exist yet). Its `.predict()`
     also takes pre-extracted feature vectors, not raw image tensors — it
     is not an inference-ready module as-is.
   - No spatial attention/Grad-CAM/attribution output exists anywhere in
     `phase_detection/` to build a region-importance saliency map from —
     would need to be built from scratch (e.g. attention rollout on the DINO
     path, or Grad-CAM on MedViT).
   - `dino_encoder.py` (used by `medViT_encoder.py`'s DINO option) has a
     known silent-fallback bug: all three of its loading attempts (real
     DINOv3 → torch.hub DINOv1 → ImageNet-supervised ViT → random-weight
     `SimpleViT`) log `"✅ Loaded ... model"` regardless of which one
     actually fired — `dino_backbone.py` (the module the training pipeline's
     existing `DinoPerceptualLoss`/`DinoSaliencyLoss` already use) was
     written specifically to avoid this failure mode with clear per-attempt
     logging; do not reuse `phase_detection/dino_encoder.py` for a new loss
     without the same fix.
   - Conclusion: a classifier-consistency loss against this classifier today
     would optimize against a meaningless/random signal, since the
     classification head has never been trained on this project's CT data.
     Training that classifier is a separate, larger prerequisite task.

---

## 13. Session 2026-07-12 — stale sample/checkpoint pruning bug, parametric lambda_l1

1. **Fixed stale-sample/checkpoint pruning bug** (found while comparing real
   scenario results — see `scenario_results_overview.md`). `_save_samples` and
   `_save_checkpoint` (`trainer.py`) pruned `sorted(glob('ep*.png'))[:-keep]` /
   `sorted(glob('ckpt_ep*.pth'))[:-keep]` — sorted by the epoch number parsed
   from the filename, not by save recency. A fresh (non-`--resume`) run
   restarts its epoch counter at 1; if its output_dir already had
   higher-numbered files left over from an earlier, longer run of the same
   scenario, the new run's own low-numbered files got pruned *in favor of*
   the stale high-numbered leftovers, every single save — confirmed via file
   mtimes on `literature_baseline_adv_only` (re-run 5 times), where
   `samples/ep039-043.png` were still from the very first run days after two
   more full re-runs. Fixed: both now sort by `Path.stat().st_mtime` instead
   of filename/epoch number.
2. **`lambda_l1` is now parametric on which other losses are active**
   (`losses.py`'s `CompositeLoss.__init__`). Motivation, from real scenario
   results (`scenario_results_overview.md`): at the original flat
   `lambda_l1=100`, every adversarial/perceptual-inclusive scenario converged
   to near-identical val metrics *and* near-identical sample grids regardless
   of which extra losses were added, and `adv_only` was consistently the floor
   performer — L1 was drowning out everything else (100:1 L1:adv, vs
   `Yan22`/`Yan23`/`Yan24c`'s own cited 12.5:1). Fix: `lambda_l1` now resolves
   to `lambda_l1_reduced` (25, new `config.py` constant) whenever
   `use_adversarial`/`use_perceptual`/`use_feature_matching` is active — those
   three specifically trade pixel fidelity for realism, unlike
   `ssim`/`gradient`/`frequency`/`organ`/`saliency`/`cycle`/`seg_consistency`,
   which refine what fidelity means rather than compete with it, so they don't
   trigger the reduction. `lambda_adv` also reverted to `2.0` (matching
   `Yan22`/`Yan23`/`Yan24c`) after a since-superseded manual edit had pushed it
   to `20.0` untested. Verified: `CompositeLoss(...).lambda_l1` resolves to
   100 for `l1_only`/`gradient_only`, 25 for `adv_only`/`perceptual_only`/
   `feature_matching_only` — matches every real scenario already run this far.
3. **`early_stop_patience` raised from 12 to 30.** At 12, every adversarial-
   inclusive scenario plateaued and stopped ~12-14 epochs after best (epoch
   13-15), while `early_stop_patience` is keyed to `val_loss` — an L1 proxy,
   which is exactly what adversarial/perceptual are supposed to trade away for
   realism. A tight patience cuts the run right as that trade-off starts to
   develop, before it can show up in the samples. Not yet re-validated against
   a real training run (config-only change).
