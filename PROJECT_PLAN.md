# NCCT→CECT synthesis — master plan

State as of 2026-07-26. Companion: `IMPLEMENTATION.md` (what the code does).

Regenerate the numbers below with:
```bash
python scripts/analyze_runs.py --runs_dir ../out_synthesis_train --out analysis/
python scripts/seeds_stats.py  --csv analysis/benchmark_seeds/master_table.csv
```

Training runs on a remote GPU host. This checkout is a code + results mirror: no
data, no CUDA, no checkpoints. **Never delete `best_model.pth`** — the 6 seed runs
have no surviving checkpoint, so they cannot be re-inferred or re-benchmarked.

---

## 1. Where things stand

### 1.1 Runs on disk (`../out_synthesis_train/`)

| group | n | config | benchmarked |
|---|---|---|---|
| 5 loss-ablation runs | 1 seed | 80 ep, `T0=15` restarts, `λ_organ=5` | `analysis/benchmark/` |
| `l1_{adv_organ,organ_curriculum}_s{42,43,44}` | 3 seeds each | 45 ep, `T0=45`, `λ_organ=20` | `analysis/benchmark_seeds/` |
| `pix2pixhd_baseline` | 1 seed | 57 ep | **no — free literature row** |

The two groups differ in 6 config keys, so they are not comparable to each other.
Within the seed group, s42 vs s43 differ in exactly two lines (`seed`,
`output_dir`) — that sweep is clean.

### 1.2 The seed noise floor, and what it costs us

n=3, pooled sd, df=4:

| metric | adv_organ | organ_curriculum | 2σ gate | Δ | verdict |
|---|---|---|---|---|---|
| **feature_l1_hu** | 14.03 ± 0.49 | 14.50 ± 0.31 | **0.82** | −0.47 | **inside the gate** |
| psnr | 30.08 ± 0.04 | 30.37 ± 0.03 | 0.07 | −0.29 | real |
| org_mae | 0.0311 | 0.0298 | 0.0006 | +0.0013 | real |
| raps_hf | 0.917 ± 0.029 | 0.826 ± 0.005 | 0.042 | +0.092 | real |
| seam | 1.383 | 1.346 | 0.011 | +0.037 | real |

σ(featHU) ≈ 0.31–0.49 HU, so n=3 resolves ~1 HU effects. But **the headline claim
does not survive.** The seed-42 result was tiered organ weighting winning −1.58 HU
featHU (t=−4.22, p<.001); across seeds the gap is 0.47 HU — inside the 0.82 gate —
**and the sign flips** (adv_organ 14.03 now *beats* organ_curriculum 14.50, vs
16.04 / 15.74 at seed 42). λ_organ and epochs also changed, so this is
config-sensitivity plus noise. Either way it is not currently defensible.

Note σ(raps_hf) is 6× larger for the adversarial branch (0.029 vs 0.005): any
texture claim about adversarial runs needs more than 3 seeds.

### 1.3 There is no generalization gap

From the 45-epoch runs: raw train L1 **0.0139 (≈8.3 HU)** vs val MAE **0.0148
(≈8.9 HU)** — a 6% gap, both curves flat, val still marginally improving at the
last epoch. The model cannot fit its **own training data** below ~8 HU.

So regularization and augmentation are not the lever. Three candidate causes
remain — capacity, optimization, or the data — and Gates A/B separate them.

Adversarial runs have a different problem: `train_adv` 0.52→0.96 while
`train_disc` 0.245→0.178 (the discriminator wins) and `val_org_ssim` slips
0.9336→0.9312. That is instability, not overfitting.

### 1.4 ~94% of per-case error is set by the case, not the model

Per-case `feature_l1_hu` correlates **r = 0.957–0.983 (mean r² = 0.94)** across all
five differently-trained models. Between-case spread 6.2 HU vs between-model
spread 3.2 HU on a fixed case. Every model fails on the same cases.

`scripts/audit_data_ceiling.py` tests the leading explanation (registration error
via contrast-invariant bone HU, plus bowel-gas divergence). **It has never been
run.** It gates everything else and needs no training — Gate A1.

### 1.5 The context gap is in Z, not in-plane

Organ extents on a real `_seg_full` mask at 1.5 mm:

| structure | X | Y | **Z** |
|---|---|---|---|
| aorta | 36 mm | 81 mm | **258 mm** |
| portal vein | 144 mm | 63 mm | **90 mm** |
| liver | 228 mm | 171 mm | 126 mm |

At 1.5 mm a 128-px patch already covers **192 mm** in-plane — enough for the
vessels' cross-sections and most of the liver. But `patch_depth=1` gives the model
**one slice** of Z. The phase-critical vessels are exactly the Z-elongated
structures and exactly the ones with the worst HU error (portal vein 36 HU).
**A Z-running tube is being predicted from a single axial slice.** Larger
*in-plane* patches buy little; Z context is the gap.

⚠ That 1.5 mm figure comes from the segmentation masks. **The CT volumes' own
spacing distribution has never been measured**, and there is no resampling
anywhere in the pipeline ([dataset.py:48](dataset.py#L48) reads raw grids), so a
128-px patch covers a physically different field of view in every case. Gate A2
settles this; the in-plane-vs-Z conclusion above depends on it.

### 1.6 Two measurement ceilings already hit

- **`phase_acc = 1.000` for all 6 seed runs** (`gen_prob` 0.984–0.993). The
  downstream classifier is saturated and can no longer rank models. The thesis
  claim has to rest on `feature_l1_hu` + `org_mae` — and featHU is the noisiest
  metric in the table.
- **`seam` is 1.34–1.67 for every model**, never near 1.0, regardless of loss.

### 1.7 Seam and receptive field are one mechanism (measured 2026-07-26)

`scripts/erf.py`, identical init, patch 256, varying only the norm:

| norm | support radius | r50 | r95 |
|---|---|---|---|
| instance (current default) | **128 = patch/2** | 14 | 105 |
| group | **128 = patch/2** | 4 | 82 |
| batch | **94** | 2 | 4 |

Instance/group support equals `patch/2` at *every* patch size (r95 was 54/105/157
at patch 128/256/384). A receptive field does not scale with its input — this is
the normalization coupling the whole patch into every output pixel. Only batch
norm shows the real field: 94 px radius, **wider** than the ~140 px usually quoted.

Consequences: (a) the "conv already spans the patch, so attention adds nothing"
argument fails, because that reach transmits a per-channel *scalar*, not spatial
structure; (b) `GEN_NORM='group'` now addresses two symptoms of one cause for the
cost of one flag. Measured at random init — mass radii are a lower bound; re-run
on a trained checkpoint (Gate A3). Detail in
`analysis/texture_consistency_findings.md`.

### 1.8 Never exercised

| thing | state |
|---|---|
| `USE_HU_PROFILE` / `OrganHUProfileLoss` | written, scenarios defined in `run_scenarios.sh`, **never run** |
| `ORGAN_FOCUS_FRAC` | `0.0` — yet `config.py:82-84` prescribes ~0.5 for organ losses. Every organ run so far used 0.0 |
| augmentation | **none of any kind** |
| spatial resampling | **none** |
| discriminator conditioning | unconditional PatchGAN, `in_ch=1` — not pix2pix's `D(x,y)` |
| decoder upsampling | `ConvTranspose` (checkerboard risk) |
| `scripts/audit_data_ceiling.py` | never run |
| `phase_infer_hann/` | 15 of 20 volumes; seam never scored on it |

---

## 2. What to do next

### Gate A — free diagnostics, zero training. Do all of these first.

| # | what | decides |
|---|---|---|
| **A1** | `scripts/audit_data_ceiling.py` on `l1_organ_curriculum`'s existing manifest + report | **whether anything downstream has headroom.** If bone-HU misregistration predicts per-case error, the 8.3 HU floor is the data's. That is a thesis finding, not a failure |
| **A2** | print the dataset's voxel-spacing distribution | settles §1.5. Decides whether in-plane FOV is a problem at all, and sets `target_spacing` for Gate C |
| **A3** | `scripts/erf.py --scenario_dir <run>` on a real checkpoint | trained mass radii (§1.7 is random-init only) |
| **A4** | benchmark `pix2pixhd_baseline`; finish the 5 remaining `phase_infer_hann` volumes and score `seam`; `scripts/norm_attribution.py --scenario_dir` on a trained checkpoint | a free literature row, a direct answer on hann blending, and the instance-vs-group seam question on trained weights |
| **A5** | re-run `scripts/analyze_runs.py` + `scripts/seeds_stats.py` over all 13 runs | one consistent table. The two existing ones used different column sets *and different paired-test baselines* |

```bash
python scripts/audit_data_ceiling.py \
  --manifest ../out_synthesis_train/literature_baseline_l1_organ_curriculum/phase_infer/manifest.csv \
  --report   ../out_synthesis_train/literature_baseline_l1_organ_curriculum/phase_infer/phase_eval_report.json
```

**Exit gate:** if A1 shows a data ceiling, the thesis pivots to documenting it and
Gates B–E shrink to a short ablation. Decide before spending GPU time.

### Gate B — capacity, 3 runs

```bash
./run_scenarios.sh capacity_overfit    # 5 cases, dropout 0, L1 only, 200 ep
./run_scenarios.sh width32 width96     # base_ch 64 already exists
```

**`capacity_overfit` is decisive.** Train MAE → <2 HU means capacity suffices and
the ceiling is data/optimization, so attention's role would be inductive bias, not
parameters. A plateau >10 HU means capacity or optimization is the bottleneck and
width comes first.

`base_ch=96` (~30.0 M vs 13.3 M) doubles as the **parameter-matched control** any
attention claim must beat — a reviewer asks for it first. Run it even if the sweep
is flat.

### Gate C — the actual context gap

Ordered by expected value, informed by §1.5 and A2:

1. **2.5-D input** — feed `2k+1` adjacent axial slices as input channels, predict
   the centre slice. Stays a 2-D network, near-zero extra parameters. Sweep k=2
   (7.5 mm) vs k=5 (16.5 mm). Requires an `in_channels` argument on
   `UNetGenerator` — it is currently **hardcoded to 1**
   ([models.py:154](models.py#L154)) — plus asymmetric in/out depth in
   `infer_volume.py`. This targets the gap §1.5 identifies and is far cheaper
   than full 3-D.
2. **Spacing normalization**, if A2 shows wide spread. In-plane only (`order=1`
   for volumes, **`order=0` for masks**); leave Z alone. Two hazards: add
   `target_spacing` to the cache key at
   [dataset.py:559](dataset.py#L559) or stale caches load silently; and
   `infer_volume.py` needs the **inverse** resample before saving, or evaluation
   mismatches training.
3. **Larger in-plane patch — deprioritized.** §1.5 says in-plane FOV is not the
   bottleneck, and §1.7 removes the ERF argument for it. Revisit only if A2
   contradicts §1.5.

Lock patch geometry at the end of this gate and do not change it again.

### Gate D — architecture and loss, each isolated

| # | change | why |
|---|---|---|
| **D1** | `l1_organ_huprofile`, `l1_huprofile_only` | `OrganHUProfileLoss` optimizes featHU — the phase model's only input — directly. Highest-value unrun experiment |
| **D2** | `ORGAN_FOCUS_FRAC = 0.5` | vessels are ~0.7% of voxels; at 0.0 they are almost never patch-centred. Changes the cache key → one re-preload |
| **D3** | `l1_organ_groupnorm` | §1.7 + `scripts/norm_attribution.py`: one flag, two measured symptoms |
| **D4** | conditional D: `Conv(2, ndf, …)` + `cat([src, tgt])` | restores the real pix2pix baseline; may also fix the discriminator-wins instability |
| **D5** | `nearest + conv` instead of `ConvTranspose` | checkerboard artifacts |
| **D6** | horizontal flip in `__getitem__` | **low priority** — §1.3, no gap to regularize. Apply at `__getitem__`, *not* in the cache key |

**D1 and D2 must land together.** Both huprofile scenarios currently keep
`ORGAN_FOCUS_FRAC=0.0`, so the HU-profile loss would train on patches that mostly
lack the vessels it constrains.

Best combination → re-run at 3 seeds → that becomes the official attention baseline
with its own σ.

### Gate E — attention, conditional on A–D

New file `models_attn.py`; do not touch `models.py`. Build only after Gate D locks
a baseline, since window alignment depends on the locked patch size.

- **Zero-init gate** `gamma = nn.Parameter(torch.zeros(1))` per scale: bit-identical
  to baseline at step 0, allows fine-tuning from a Gate-D checkpoint, and `gamma`
  is itself a result — log it to `history.json`.
- **`window_size = 8`, not 7.** At patch 256 / overlap 0.5 the stride is 128, and
  128/64, 128/32, 128/16 and the 32-px shift all divide evenly. Nothing aligns
  with 7.
- **Separable SD + GroupNorm** (`9C²` → `~C²`; GroupNorm for the §1.7 reason).
  Flag as a deliberate deviation from the source paper and justify it.
- **At most one bottleneck block**, 512→256→attend→512. Four blocks at C=512 is
  ~37.7 M — nearly 3× the whole current generator.
- MAPformer input must be the **raw NCCT**, not encoder features; feeding encoder
  features just rebuilds TransUNet, which the source paper shows is worse.

Budget ~3.1 M (+23%). Runs: `dec3` only → 3 scales → +1 bottleneck block.

**Gate:** if the cheapest variant at 3 seeds does not clear Δ > 2σ on featHU
(currently 0.82 HU), or `gamma → 0`, **stop.** "Attention did not help, here is the
evidence, with a parameter-matched control" is a complete thesis section.

### Gate F — final ablation, only if E clears

Plain Swin without SD (isolates SD vs attention generally); MAPformer fed encoder
features (tests the "anatomical perception" claim); `base_ch=96` parameter-matched
control (already have it from Gate B).

---

## 3. Reading results without fooling yourself

- **Never judge on PSNR/SSIM alone.** They were flat across a range containing a
  t=−4.22 effect.
- **Check every difference against the noise floor.** §1.2 is the cautionary
  example: a p<.001 result vanished and reversed under 3 seeds.
- **Use paired per-case tests.** Between-case variance (6.2 HU) swamps
  between-model differences (3.2 HU).
- **Expect PSNR to fall** on a good model. Say so in advance.
- **Diff `run_config.json` against the previous run** — it should show only the
  keys you intended. That is how the s42/s43 pair was verified clean, and it is
  the cheapest guard against confounds.

---

## 4. Open risks

- **n=20 test set** is thin for the headline claim; expanding it is the cheapest
  way to harden the thesis.
- **`phase_acc` is saturated** (§1.6) — the downstream metric can no longer
  discriminate.
- **Zero-weighting bowel needs defending** — frame it as measured: 27% of voxels,
  ~1.3% of phase importance, gas not inferable from NCCT. Bowel is still
  *evaluated*, just not *trained*.
- **Masks derive from the target CECT**, so they cannot be fed as a generator
  input without NCCT-derived masks.
- **A mask whose shape does not match its volume silently becomes all-zero**
  ([dataset.py:452](dataset.py#L452)) and organ-focus silently finds no candidates
  ([dataset.py:491](dataset.py#L491)). Check mask coverage after any resampling
  change with `scripts/check_seg_masks.py`.
- **`SEG_SUFFIX` must stay `'_seg_full'`** — `_seg_reg` uses different label
  conventions than the weights assume.
- **Selection-metric change breaks comparability** with the two oldest runs.

---

## 5. Deferred, with the argument recorded

**CT-pretrained backbone as generator initialisation — argue against.** §1.4 says
~94% of per-case error is set by the case; a better initialisation cannot recover
error the input does not determine. nnU-Net is also a segmentation framework whose
value is the self-configuring pipeline, and its decoder emits class logits.

**As a perceptual loss — well-motivated, and the better use.** The current
perceptual loss is VGG16 (ImageNet) or DINOv2/v3 (web images); neither has seen a
CT. `DinoPerceptualLoss` already accepts a pre-built backbone.

**Full 3-D** — only if Gate C's 2.5-D probe moves vessel HU.

Revisit all three after Gate A1.
