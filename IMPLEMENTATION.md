# Implementation reference

What the code **actually does** right now, not the design intent. A snapshot —
re-verify against the code before trusting a claim here.

Scope: 2-D NCCT → CECT translation to a single target phase (venous). U-Net
generator + PatchGAN discriminator + a configurable composite loss with 11
optional terms. 3-D paths exist throughout but are untested (`patch_depth=1`,
`dims=2`).

For results, the noise floor and the forward roadmap, see `PROJECT_PLAN.md`.

---

## 0. Layout

```
train.py trainer.py dataset.py models.py losses.py    core training loop
metrics.py benchmark.py infer_volume.py               evaluation
config.py                                             all defaults + train_config
dino_backbone.py                                      optional, off by default
run_scenarios.sh                                      scenario driver
scripts/                                              analysis + diagnostic CLIs
tests/                                                run directly: python tests/<f>.py
splits/split.json                                     frozen split, for external models
orgFeatXGB_CTPhase/                                   REQUIRED dependency (see §7)
analysis/                                             results of record
```

Outputs go to `../out_synthesis_train/<run>/`, patch caches to
`../out_synthesis_train/patch_cache/`. Files in `scripts/` and `tests/` carry a
3-line `sys.path` shim so they run from the repo root as `python scripts/foo.py`.

---

## 1. Data pipeline — `dataset.py`

### Discovery and split (`find_pairs_and_split`)
Scans `cfg['data_dir']` per case, globs `*{file_tag}.nii.gz` (`file_tag='_deeds'`),
classifies each file's phase via `labels_csv` or filename keywords. A case becomes
a pair only if it has both a non-contrast and the `target_phase` volume.
`seg_path` is derived by string substitution (`..._deeds.nii.gz` →
`..._deeds{SEG_SUFFIX}.nii.gz`); missing → `None`, no error.

Case-level split (no patient leakage) via
`np.random.default_rng(data_seed).permutation`. **137 pairs → 97 train / 20 val /
20 test** at `val_split=test_split=0.15`, `data_seed=42`. Materialized to
`splits/split.json` by `scripts/dump_split.py` — the contract external benchmark
models read.

### `CTPairDataset`
Three phases in `__init__`:

1. **Index** — for every pair, load both volumes (`_load_vol`, LRU 256), slide a
   window at `stride = patch_size × (1 - overlap)`, keep a coordinate only if the
   **source** patch passes `min_patch_std / _mean / _max` (raw HU). Target and mask
   validity are not checked independently.
2. **Sub-sample** — if more coordinates than `max_patches`, uniform `rng.choice`
   without replacement. Then **sorted by `(src_path, tgt_path)`**: preloading in
   random order thrashed the volume LRU and once cost 3 hours for 20k patches.
3. **Preload** — crop, `clip(hu_min, hu_max)`, rescale to `[0,1]`, hold in RAM.

`load_mask = use_organ or use_seg_consistency` — segmentation volumes are only read
when one of those is on.

`__getitem__` is a pure array lookup returning `{'source', 'target'}` (+ `'mask'`
if masks were loaded). **No augmentation of any kind exists.**

⚠ **Silent failure modes.** If a mask's shape does not match its volume, that
patch's mask stays all-zero with no warning (`dataset.py:452`), and organ-focus
sampling silently returns no candidates (`dataset.py:491`). Both degrade the organ
losses to nothing. Verify coverage with `scripts/check_seg_masks.py` after any
geometry change.

### Organ-focused sampling
`ORGAN_FOCUS_FRAC` is the fraction of patches whose **centre** sits on an
organ/vessel voxel; the rest stay uniform-grid. Currently `0.0`. `config.py:82-84`
prescribes ~0.5 when organ losses are active, since vessels are ~0.7% of voxels.
Raising it changes the cache key.

### On-disk patch cache
`_cache_path` md5-hashes everything determining content: the sorted pair list,
patch geometry, validity thresholds, `hu_min/hu_max`, `max_patches`, `data_seed`,
`load_mask`, `mask_multilabel`, `seg_suffix`, `split_name`, and the organ-focus
knobs when `frac > 0`. Located under `cfg['cache_dir']`, **independent of
`output_dir`**, so scenarios differing only in loss flags share one cache. Organ
weight *values* are deliberately excluded — they apply in the loss, so re-tuning
must not force a re-preload.

⚠ Anything that changes patch **content** must be added to this key. Truncating
the pair list (as `max_train_cases` does) changes it automatically because `pairs`
is hashed; a transform applied *during* preload would not.

### No spatial resampling
`_load_vol` is `nib.load(path).get_fdata()` + transpose. Nothing reads voxel
spacing anywhere in the pipeline; `affine`/`header` are used only to write output
back out (`infer_volume.py:261-269`). A 128-px patch therefore covers a physically
different field of view per case. See `PROJECT_PLAN.md` §1.5.

---

## 2. Models — `models.py`

### `UNetGenerator` — 13,317,121 params at `base_ch=64`
4 encoder levels + bottleneck + 4 decoder levels, so patch size must be divisible
by 16. Channels `1→64→128→256→512`; bottleneck 2× conv at 512 with dropout;
decoder uses **`ConvTranspose`** with skip concatenation at every level.
`dims` selects 2-D/3-D layers via the `_conv`/`_convT`/`_norm`/`_pool`/`_drop`
factories.

Output is **`Sigmoid`**, matching the dataset's `[0,1]` range (was `Tanh`, which
wasted its negative half and saturated near the targets).

**Input channels are hardcoded to 1** (`models.py:154`) — a 2.5-D multi-slice input
needs an `in_channels` argument added.

Parameters are heavily back-loaded:

| module | share |
|---|---|
| bottleneck | 35.4% |
| enc4 | 26.6% |
| dec4 | 17.7% |
| **bottleneck + enc4** | **62.0%** (+dec4 → 79.7%) |

62% sits on the two coarsest grids (8×8 and 16×16 at patch 128 and 256), which is
exactly where a transformer bottleneck would go — so attention there *competes*
with existing capacity. Reproduce with `python scripts/capacity_profile.py`.

### `GEN_NORM` — a tiling decision, not a regularisation one
`'instance'` (current) | `'group'` | `'batch'`. Instance and group statistics are
computed over the patch's own extent, so two overlapping tiles apply different
content-dependent transforms to the same voxel — a DC offset that blending cannot
cancel. Measured two ways: `scripts/norm_attribution.py` (instance drift@shift32
13.98 HU vs group 6.83 HU) and `scripts/erf.py` (instance/group gradient support
spans the whole patch at *any* patch size; batch is bounded at 94 px). Left at
`'instance'` only so existing checkpoints load. See `PROJECT_PLAN.md` §1.7.

### `PatchGANDiscriminator` — 2.76 M params
70×70 receptive field, 4 strided blocks, `ndf=64`, `BatchNorm` except block 0.
**Unconditional: `in_ch=1`** (`models.py:249`) — it sees only the image, never
`cat([src, tgt])`, so this is *not* pix2pix's `D(x,y)`.
`forward(x, return_features=True)` also returns every intermediate block output for
`FeatureMatchingLoss`.

---

## 3. Losses — `losses.py`

`CompositeLoss` is the only thing the trainer calls. Each sub-loss is instantiated
**only if its `use_*` flag is true**, so disabled losses cost nothing.
**All 11 optional terms default to `False`** — the config default is L1 only.

| Class | Computes | Flag |
|---|---|---|
| `AdversarialLoss` | LSGAN (`mse` vs 0/1) or BCE (label-smoothed 0.9/0) | `use_adversarial` |
| `PerceptualLoss` | VGG16 ImageNet, L1 on 4 ReLU feature maps | `use_perceptual` |
| `DinoPerceptualLoss` | same, 3 maps from `DinoSpatialBackbone` | `perceptual_backbone='dino'` |
| `FeatureMatchingLoss` | L1 between D's real vs fake block outputs | `use_feature_matching` |
| `SSIMLoss` | 1 − Gaussian-window SSIM (11×11, σ=1.5) | `use_ssim` |
| `GradientLoss` | L1 on Sobel gradient magnitude | `use_gradient` |
| `FrequencyLoss` | L1 on `\|fft2\|` amplitude, no windowing | `use_frequency` |
| `OrganWeightedLoss` | MSE weighted `organ_weight×` inside `mask`; **plain MSE if `mask is None`** | `use_organ` |
| `OrganHUProfileLoss` | L1 between per-organ **mean** HU of pred vs target, same weight LUT; skips zero-weight organs and organs <16 voxels | `use_hu_profile` |
| `PhaseSaliencyLoss` | MSE weighted where `\|target−source\| > threshold` | `use_saliency` |
| `DinoSaliencyLoss` | MSE weighted by `‖DINO(target) − DINO(source)‖` | `saliency_mode='dino'` |
| `CyclicConsistencyLoss` | L1 between `G(G(source))` and `source` | `use_cycle` |
| `SegmentationConsistencyLoss` | L1 on masked Sobel edge maps; **identical to `GradientLoss` if `mask is None`** | `use_seg_consistency` |

`forward` returns `(total, loss_dict)`; every key is always present, `0.0` when
disabled.

### `OrganHUProfileLoss` — written, never run
The XGBoost phase model reads per-organ **median HU** and nothing else, so contrast
phase *is* each organ's absolute enhancement level. Per-voxel losses reach that only
indirectly. This targets it directly, constraining **level, not appearance** — a
patch can score 0 while looking nothing like the target, so it must never be the
only spatial term. `LAMBDA_HU_PROFILE = 50`, sized from the ~15 HU residual on a
600 HU window (≈0.025 normalised) so the term lands near 20% of the organ term; at
λ=10 it would have been <4% of the gradient, the mistake that made λ_organ=5
ineffective. Scenarios exist in `run_scenarios.sh`.

### Parametric `lambda_l1`
L1 stays at `LAMBDA_L1` (100) normally but drops to `LAMBDA_L1_REDUCED` (25)
whenever adversarial/perceptual/feature-matching is on — those three trade pixel
fidelity for realism, unlike ssim/gradient/frequency/organ, which refine what
fidelity means. At the original 100:1 ratio every adversarial scenario converged to
near-identical metrics regardless of what else was enabled.

⚠ **Historic bug worth knowing.** `use_l1_decay` used to start the decay from
`lambda_l1_reduced` (25), which equalled the floor, so `l1_adv_organ`'s curriculum
ran 25→25 for all 65 epochs — a no-op. Fixed: decay starts from the full
`LAMBDA_L1` and warns loudly when start == floor.
`tests/smoke_test_organ_weights.py` regression-tests the exact flat trace.

### Warmup and backbone sharing
`_adv_w()` ramps `0 → lambda_adv` over `adv_warmup_epochs`; `_cycle_w()` does the
same for `lambda_cycle`. Both driven by `criterion.set_epoch(epoch)`.
`_get_dino_backbone()` lazily builds **one** `DinoSpatialBackbone` shared by both
DINO losses, and none at all under the `'vgg'`/`'heuristic'` defaults.

### Single-generator cycle caveat
There is only one `G: NCCT → CECT`. `use_cycle` pushes `G(G(source)) → source`, so
the same weights must satisfy both directions — `G` is trained toward an
*involution*, conditioned implicitly on whether its input looks contrast-enhanced.
Deliberate, harder than a two-generator CycleGAN, mitigated by the warmup ramp.

---

## 4. Trainer — `trainer.py`

Owns `G`, `D` (`None` unless `use_adversarial`), two Adam optimizers
(`betas=(0.5,0.999)`), an optional `CosineAnnealingWarmRestarts` on `opt_G` only,
`CompositeLoss`, and AMP `GradScaler`s.

Per step (`_train_step`):
1. Move `source`/`target`/`mask` to device.
2. If a D-update step (`global_step % disc_update_freq == 0`): forward `G(source)`
   under `no_grad` for `fake_for_d`, then `_disc_step` (D on real+fake, loss,
   backward, grad-clip 10.0, step). Returns real features for feature matching.
3. Generator step: forward `fake = G(source)` **again**, this time with gradients;
   `D(fake, return_features=True)` if `D` exists; `G(fake)` if `use_cycle`;
   `CompositeLoss(...)`; scaled backward, grad-clip 10.0, step.

`_validate` (`no_grad`) computes L1/PSNR/SSIM globally and organ-masked, with a
hand-rolled non-windowed PSNR/SSIM.

Per epoch: one pass, scheduler step, validate, dump `history.json` (26 channels),
curves every 5 epochs, checkpoint every epoch (rolling
`keep_last_n_checkpoints=3`), plus `best_model.pth` chosen on
**`selection_metric`** (default `val_org_ssim`, not `val_loss`). `EarlyStopping` is
keyed to `val_loss` with `early_stop_patience=30` — raised from 12 because
`val_loss` is an L1 proxy, exactly what adversarial/perceptual trade away, so a
tight patience cut runs before the trade-off could develop.

Checkpoints persist `G`/`opt_G` always and `D`/`opt_D` when present.
`CompositeLoss` is not checkpointed (stateless or frozen).

⚠ Checkpoints are the only way to re-infer or re-benchmark a run. **Never delete
`best_model.pth`** — the 6 seed runs have none and cannot be re-scored.

---

## 5. Evaluation — `metrics.py`, `benchmark.py`, `infer_volume.py`

`benchmark.py:score_model` computes **three** mask sets per case: whole-volume,
organ (`org_*`, from the seg mask), and body interior (`body_*` + `body_frac`, via
`metrics.body_mask` — per-slice closing + largest connected component to drop the
scanner table). The body mask exists because global metrics are diluted by clipped
air, which every model gets free: measured global MAE 5.9 HU vs organ 19 HU. Always
build it from the **real** volume.

Key metrics and where they live:

| name | source |
|---|---|
| `org_mae` (`oMAE`) | `metrics.masked_metrics` |
| `featHU` (`feature_l1_hu`) | external `orgFeatXGB_CTPhase/phase_eval.py` — mean per-organ \|HU error\| |
| `raps_hf` | `metrics.raps_hf_ratio` — gen/real radial HF power ratio |
| `seam` | `metrics.seam_energy` — first-difference energy on tile boundaries vs interior |
| `grad_w1`, `zflicker`, `zaniso` | `metrics.grad_hist_distance`, `z_flicker`, `z_flicker_anisotropy` |

⚠ `raps_hf`, `seam` and `zflicker` are **ratio** metrics scored as distance from
**1.0**, not higher-is-better. `seam` is `NaN` when tiling geometry is unknown —
that is deliberate, better than a wrong number.

`infer_volume.py` tiles with `_starts` (always includes a flush-to-edge final tile),
blends overlaps, and saves on the **source grid with the source affine**. If
resampling is ever added, this needs the inverse transform or evaluation silently
mismatches training.

---

## 6. Config — `config.py`

Single source of defaults, assembled into `train_config`. `config.py` holds **no
scenario definitions** — those are the `SCENARIOS` array in `run_scenarios.sh`.

| key | value |
|---|---|
| `HU_MIN` / `HU_MAX` | `-200` / `400` → `[0,1]` (set from measured percentiles by `scripts/analyze_hu_range.py`) |
| `PATCH_SIZE` / `PATCH_DEPTH` / `DIMS` / `OVERLAP` | `128` / `1` / `2` / `0.5` |
| `BATCH_SIZE` / `EPOCHS` | `16` / `45` |
| `MAX_TRAIN_PATCHES` / `MAX_VAL_PATCHES` | `20_000` / `4_000` |
| `MAX_TRAIN_CASES` | `None` (capacity probe; caps train only, never val/test) |
| `GEN_BASE_CH` / `GEN_DROPOUT` / `GEN_NORM` | `64` / `0.20` / `'instance'` |
| `ORGAN_FOCUS_FRAC` | `0.0` |
| `COSINE_T0` | `45` — `≥ EPOCHS`, so a single anneal with no warm restarts. At `T0=15` the restarts cost org-SSIM in every run (worst −0.019) |
| `LAMBDA_ORGAN` | `20` (was 5, where the organ term peaked at 21% of L1) |
| `EARLY_STOP_PATIENCE` | `30` |
| `SEG_SUFFIX` | must stay `'_seg_full'` — `_seg_reg` uses different label ids |

Seeding is complete: `set_seed()` at `train.py:45` covers `random`/`numpy`/`torch`
and the cuDNN determinism flags, `dataset.py:690` passes a `torch.Generator` and
`worker_init_fn` to the DataLoader, and `--seed`/`--data_seed` are separate so
changing the model seed cannot silently move the split. `train.py` warns if
`data_seed != 42`, which would invalidate every cross-run benchmark. Not set:
`torch.use_deterministic_algorithms(True)` / `CUBLAS_WORKSPACE_CONFIG`, so bitwise
reproducibility is not guaranteed for every op.

CLI overrides live in `train.py` (`--help` is authoritative). Beyond the loss
`--use_X`/`--no_X` pairs: `--seed`, `--data_seed`, `--patch_size`, `--base_ch`,
`--dropout`, `--generator_norm`, `--max_train_cases`, `--no_early_stop`,
`--lambda_*`, `--selection_metric`, `--organ_weight_preset`.

---

## 7. `orgFeatXGB_CTPhase/` — required, do not remove

Two hard runtime dependencies:
- `benchmark.py:46` appends it to `sys.path`, then imports `phase_eval` and
  `organ_features`. It uses **`append`, not `insert(0, …)`** — the directory has its
  own `train.py`, and putting it first shadowed this repo's.
- `config.py:339` reads `retrain_out_full/ts_label_map_total.json`, consumed by
  `trainer.py:234`. Per-organ metrics and organ-weighted losses break without it.

It began as a rename of `phase-detection/CTPhase-XGBoost` but has diverged and
moved ahead; this fork is authoritative. `xgb_vindr_full.pkl` is force-added
against `.gitignore`'s `*.pkl` because `benchmark.py --weights` requires it.

**Archived:** `volEmbed_CTPhase/` (whole-volume phase classification, MedViT/DINOv3
encoders) and its design doc `phase_conditioning_plan.md` had zero code references
and were never implemented. Preserved on branch `archive/phase-conditioning`:

```bash
git checkout archive/phase-conditioning -- volEmbed_CTPhase/
```

Not the same thing as `orgFeatXGB_CTPhase/`. Do not confuse them.

`dino_backbone.py` **is** live — reached from `losses.py:757` behind
`PERCEPTUAL_BACKBONE='dino'` / `SALIENCY_MODE='dino'`, both off by default, so it
never loads in current runs.

---

## 8. Tooling

```bash
./run_scenarios.sh                      # all scenarios;  SEEDS="42 43 44" for a sweep
./run_scenarios.sh l1_organ_huprofile   # named subset

python scripts/analyze_runs.py --runs_dir ../out_synthesis_train --out analysis/
python scripts/seeds_stats.py           # mean ± std across _s<N> replicates
python scripts/audit_data_ceiling.py    # is the residual the model's or the data's?
python scripts/capacity_profile.py      # per-module params, VRAM, patch/s
python scripts/erf.py --compare_norms   # effective receptive field
python scripts/norm_attribution.py      # seam attribution per norm kind
python scripts/check_seg_masks.py       # mask coverage — run after geometry changes
python scripts/check_determinism.py same <runA> <runB>
```

Tests are `__main__` scripts printing PASS/FAIL, run by hand (no CI). Only
`tests/test_metrics.py` is pytest-style. `tests/smoke_test.py`'s perceptual
scenarios need `torchvision`.

```bash
python tests/test_metrics.py            # 15 checks
python tests/test_patch_cache.py        # 9 checks — cache correctness
python tests/test_max_train_cases.py    # 9 checks — probe caps train only, cache key differs
python tests/smoke_test_organ_weights.py
python tests/smoke_test_organ_focus.py
python tests/smoke_test.py              # 12 loss scenarios, GPU-free
python tests/smoke_test_infer.py
```

---

## 9. History

Per-session change logs were removed from this file; `git log` is the record.
Milestones: `Tanh`→`Sigmoid` output fix and the never-delivered organ mask
(2026-07-01); patch cache and preload ordering (07-08); parametric `lambda_l1`
(07-12); organ weights and the 25→25 decay no-op fix (07-20); metrics expansion
and seed control (07-22 → 07-26); repo restructure, `sys.path` shadowing fix, and
capacity/ERF tooling (07-26).
