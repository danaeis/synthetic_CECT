# NCCT→CECT synthesis — master plan

State as of 2026-07-27, after Gates A and B. Companion: `IMPLEMENTATION.md` (what the code does).

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

**Gates A and B are complete.** Both of the hypotheses they were built to test came
back **negative**, which narrows the explanation sharply — see §1.9.

### 1.1 Runs on disk (`../out_synthesis_train/`) — 15 runs

| group | n | config | benchmarked |
|---|---|---|---|
| 5 loss-ablation runs | 1 seed | 80 ep, `T0=15` restarts, `λ_organ=5` | ✅ |
| `l1_{adv_organ,organ_curriculum}_s{42,43,44}` | 3 seeds each | 45 ep, `T0=45`, `λ_organ=20` | ✅ |
| `pix2pixhd_baseline` | 1 seed | 57 ep | ✅ (new) |
| `capacity_overfit`, `width32`, `width96` | 1 seed | Gate B probes | n/a (train-curve only) |

All 15 are in one table at `analysis/benchmark_all/`. The first two groups differ
in 6 config keys and are not comparable to each other; within the seed group, s42
vs s43 differ in exactly two lines.

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
**and the sign flips**. λ_organ and epochs also changed, so this is
config-sensitivity plus noise. Either way it is not currently defensible.

σ(raps_hf) is 6× larger for the adversarial branch: texture claims about
adversarial runs need more than 3 seeds.

### 1.3 There is no generalization gap

From the 45-epoch runs: raw train L1 **0.0139 (≈8.3 HU)** vs val MAE **0.0148
(≈8.9 HU)** — a 6% gap, both curves flat, val still marginally improving at the
last epoch. The model cannot fit its **own training data** below ~8 HU.

Adversarial runs have a separate problem: `train_adv` 0.52→0.96 while `train_disc`
0.245→0.178 (the discriminator wins) and `val_org_ssim` slips 0.9336→0.9312. That
is instability, not overfitting.

### 1.4 Registration and anatomy change do NOT explain the error — REFUTED

`scripts/audit_data_ceiling.py`, 20 test cases:

| proxy | r | r² |
|---|---|---|
| bone NCC (registration quality) | **−0.090** | 0.008 |
| bone MAE (registration error) | **+0.249** | 0.062 |
| bowel-gas Dice (anatomy stability) | **+0.105** | 0.011 |

None reaches the |r| > 0.44 needed for p<0.05 at n=20, so this is *underpowered*,
not proof of absence — but the case-level detail is more damning than the
correlations:

- Case 15 is the **worst-registered** case by a wide margin (bone NCC 0.696 vs a
  0.949 median, bone MAE 111 vs 60) — and its featHU is **17.9**, near the median.
- Case 14 has **better-than-median registration** (0.955) **and** stable anatomy
  (gas Dice 0.832) — and the **worst featHU in the set, 32.0**.
- Case 1, second-worst featHU (26.9), also has above-median registration (0.935).

The two worst-synthesized cases are both better-registered than average. Whatever
sets per-case error, it is not registration quality and not bowel gas.

### 1.5 Voxel spacing is perfectly uniform — resampling is a dead work item

`scripts/spacing_stats.py` over all **274 volumes** (137 pairs):

**Every single volume is 1.5 × 1.5 × 1.5 mm.** max/min = 1.00× on all three axes;
`sx == sy` everywhere; no NCCT/CECT pair differs. The data was already resampled to
isotropic 1.5 mm upstream.

Consequences, all of them simplifying:

- **Spatial resampling is unnecessary.** Removed from the roadmap entirely. The
  earlier worry that "a 128-px patch is 102 mm in one case and 205 mm in another"
  was simply wrong for this dataset.
- **§1.5's original claim is confirmed by measurement**: a 128-px patch spans
  **192 mm** in-plane, against an aorta cross-section of 36 × 81 mm and a liver of
  228 × 171 mm. In-plane context is sufficient.
- **The gap is Z, and Z is isotropic too.** `patch_depth=1` gives the model 1.5 mm
  of a structure that runs 258 mm (aorta) or 90 mm (portal vein). A 2.5-D stack of
  `2k+1` slices buys 7.5 mm at k=2 and 16.5 mm at k=5, with no resampling and
  near-zero extra parameters.
- **`zaniso ≈ 1.09` is a model artifact, not geometry.** Spacing is isotropic, so
  anisotropic slice-to-slice behaviour in the output comes from the 2-D
  architecture itself.

### 1.6 Two measurement ceilings already hit

- **`phase_acc = 1.000` for all 6 seed runs** (`gen_prob` 0.984–0.993). The
  downstream classifier is saturated and can no longer rank models. The claim has
  to rest on `feature_l1_hu` + `org_mae` — and featHU is the noisiest metric.
- **`seam` is 1.34–1.67 for every model**, never near 1.0, regardless of loss.

### 1.7 Seam and receptive field are one mechanism

**Norm comparison at identical init** (`scripts/erf.py --compare_norms`, patch 256):

| norm | support radius | r50 | r95 |
|---|---|---|---|
| instance (current default) | **128 = patch/2** | 15 | 105 |
| group | **128 = patch/2** | 4 | 80 |
| batch | **94** | 2 | 4 |

Instance/group support equals `patch/2` at *every* patch size (r95 = 54/105/157 at
patch 128/256/384). A receptive field does not scale with its input — this is the
normalization coupling the whole patch into every output pixel. Only batch norm
shows the real field: 94 px radius, **wider** than the ~140 px usually quoted.

**Trained vs random init** (`analysis/erf_trained.json`, patch 128, epoch 39):

| | r50 | r95 |
|---|---|---|
| random init | 13 | 54 |
| **trained** | **28** | **59** |

Training more than **doubles** r50 (13 → 28 px). The model genuinely learns to use
wider spatial context than its initialisation does — so mass radii measured at
random init were, as suspected, a lower bound.

**Norm attribution on trained weights** (`scripts/norm_attribution.py`, epoch 39,
output std 125.97 HU):

| shift | total HU | DC HU | DC frac | drift/std |
|---|---|---|---|---|
| 4 | 2.65 | 1.19 | 44.8% | 2.1% |
| 32 | **4.99** | **3.22** | **64.5%** | 4.0% |

Compared to random init (13.98 HU total, 32.4% DC), training cut absolute drift
**4.3×** but **doubled the DC fraction**. What survives is almost entirely the
constant offset — precisely the component overlap-blending cannot cancel. That
makes `GEN_NORM='group'` a cleaner call than before, not a weaker one.

### 1.8 Never exercised

| thing | state |
|---|---|
| `USE_HU_PROFILE` / `OrganHUProfileLoss` | written, scenarios defined, **never run** |
| `ORGAN_FOCUS_FRAC` | `0.0` — yet `config.py:82-84` prescribes ~0.5 for organ losses |
| augmentation | **none of any kind** |
| 2.5-D / Z context | `patch_depth=1`; `UNetGenerator` in-channels hardcoded to 1 |
| discriminator conditioning | unconditional PatchGAN, `in_ch=1` |
| decoder upsampling | `ConvTranspose` |
| `phase_infer_hann/` | seam never scored on it |

### 1.9 What is left: the residual is most likely aleatoric

Gate B settles capacity. **Width sweep**, val MAE at best epoch:

| base_ch | params | train HU | val HU | val_org_ssim |
|---|---|---|---|---|
| 32 | 3.3 M | 8.81 | 9.14 | 0.93665 |
| 64 | 13.3 M | 8.32 | 8.86 | 0.93871 |
| 96 | 30.0 M | 8.02 | 8.81 | 0.93890 |

**9× the parameters buys 0.33 HU**, and 64→96 buys 0.05 HU. The curve is flat.

**Deliberate overfit** (5 cases, dropout 0, L1 only, 200 epochs):

| epoch | 1 | 25 | 75 | 125 | 200 |
|---|---|---|---|---|---|
| train HU | 57.0 | 6.0 | 4.7 | 3.5 | **3.45** |
| val HU | 32.0 | 11.3 | 11.3 | 11.8 | 11.5 |

Two things follow, and together they are decisive:

1. **Capacity is not binding.** On 5 cases the model reaches **3.45 HU** train
   error, while on 97 cases it plateaus at **8.32 HU**. Same architecture, same
   parameters. If capacity were the limit both would stop in the same place. The
   representational headroom exists; it simply cannot be applied across the full
   set. The 3.3× train/val gap (3.45 vs 11.5) confirms it can overfit when the
   data is small enough.
2. **Even 5 cases do not reach zero.** A 13.3 M-parameter network with no dropout,
   200 epochs and an L1-only objective plateaus at 3.45 HU on five volumes. If the
   NCCT→CECT mapping were deterministic it should very nearly memorise them.

Four independent measurements now agree, and they exclude everything except one
explanation:

| candidate | status |
|---|---|
| overfitting / regularisation | ruled out — §1.3, 6% train/val gap |
| registration error, anatomy change | ruled out — §1.4, and the two worst cases are well-registered |
| voxel-scale inconsistency | ruled out — §1.5, spacing is uniform |
| model capacity | ruled out — §1.9, flat width sweep + 5-case overfit |
| **irreducible input ambiguity** | **the remaining candidate** |

**Hypothesis: the residual is aleatoric.** Absolute contrast enhancement depends on
injection dose, bolus timing, cardiac output and patient physiology — *none of
which is observable in a non-contrast scan*. Two patients with identical NCCT
anatomy can have genuinely different venous-phase HU. A deterministic regressor
trained with L1 must then predict the conditional mean, and the residual is the
conditional spread. That explains, at once, why every model fails on the same
cases (r² = 0.94), why the error is a per-organ **level** error (which is exactly
what `featHU` measures), and why no loss or architecture change has moved it.

This is a **thesis finding, not a failure** — but it is still a hypothesis. §2
tests it directly rather than assuming it.

---

## 2. What to do next

Gates A and B are done. The roadmap below is reordered around §1.9: the next
question is not "how do we make the model bigger" but "is the target predictable
from the input at all, and if not, what should the model output instead."

### Gate C — test the aleatoric hypothesis (highest value, cheap)

> **Status 2026-07-27: implemented, ready to run.** `scripts/audit_enhancement.py`
> replaces the original C1. The metadata that design needed **does not exist** —
> `vindr_nifti_metadata.csv` carries only geometry and `labels.csv` only phase
> labels; there is no contrast dose, scan delay, kVp or patient weight anywhere.
> The replacement reads bolus position from the real CECT itself (aorta vs portal
> vein HU), so it needs no metadata at all, and it tests the hypothesis more
> directly than dose ever would.

| # | what | why |
|---|---|---|
| **C1** | Correlate per-case featHU against **CECT-side** scan metadata: contrast volume/dose, injection rate, scan delay, kVp, patient age/weight — whatever `labels.csv` and the DICOM headers carry. Same script shape as `audit_data_ceiling.py`. | The direct test. If bolus timing or dose predicts the residual, the hypothesis is confirmed and the case for **conditioning** on it is made |
| **C2** | Measure the residual's structure: per-case, per-organ, is the error a **constant offset** or a spatial pattern? Compute per-organ (pred_mean − target_mean) and compare its variance to the within-organ residual variance | If the error is DC-dominated per organ, it is a level problem, which no amount of spatial modelling fixes |
| **C3** | Run `l1_organ_huprofile` + `ORGAN_FOCUS_FRAC=0.5` | The discriminating experiment. `OrganHUProfileLoss` optimises organ level directly. If level is *predictable but unoptimised*, this moves featHU. If level is *unpredictable*, it cannot — and that null result is itself evidence for §1.9 |

**C2 is now a single command** — `scripts/audit_enhancement.py` reports, per organ:
`beta` (slope of generated median HU on real median HU across cases; 1 = tracks
the case, 0 = emits a constant), `var(gen)/var(real)` (under-dispersion), and the
signed-error slope. It prints a verdict keyed on the contrast organs. Run it on
`l1_organ_curriculum` and again on `l1_adv_organ` to confirm the finding is not
loss-specific.

### Gate C-bis — phase conditioning (implemented 2026-07-27)

136 arterial pairs exist alongside 137 venous **on the same patients**: identical
NCCT input, two different targets. That is a known, controlled instance of exactly
the ambiguity §1.9 describes, so it measures what an explicit conditioning
variable recovers.

| arm | scenario | purpose |
|---|---|---|
| **M1** | `l1_organ_curriculum_s{42,43,44}` — already run | venous-only baseline |
| **M2** | `multiphase_uncond` | 2× pairs, **no** phase input |
| **M3** | `multiphase_film` | 2× pairs + FiLM phase conditioning |

**M2 is not optional**: without it, an M3 gain cannot be attributed to
conditioning rather than to data volume. It carries its own prediction — if the
aleatoric story holds, M2 should be *worse* than M1 on venous featHU, because
pooling two phases without saying which one adds ambiguity.

```bash
./run_scenarios.sh multiphase_uncond multiphase_film     # seed 42 first
```
Expand to seeds 43/44 only if `featHU(M2) − featHU(M3)` clears the 0.82 HU 2σ
gate. Both change the patch-cache key, so the first pays a one-time re-preload.

⚠ **Blocking pre-flight**: arterial pairs need `_seg_full` masks beside the
arterial volumes, or every organ loss silently degrades to its unweighted
fallback ([dataset.py:452](dataset.py#L452) fills a zero mask with no warning).
```bash
python -c "
import sys; sys.path.insert(0,'.')
from config import train_config as c
from dataset import find_pairs_and_split
for ph in ['venous','arterial']:
    p = sum(find_pairs_and_split({**c,'target_phase':ph}), [])
    print(ph, len(p), 'pairs,', sum(x['seg_path'] is not None for x in p), 'with masks')
"
```

Read `gamma_*` in `history.json` as a result: converging to ~0 means the decoder
ignored the phase input, i.e. conditioning bought nothing.

C1 is the one to do first. If it lands, the thesis contribution becomes
"phase-conditioned synthesis with an explicit contrast-level input", which is a
stronger and more novel claim than another loss ablation — and the design doc for
it already exists on the `archive/phase-conditioning` branch.

### Gate D — Z context, the one architectural gap left standing

§1.5 rules out in-plane FOV and resampling. Z is the only geometric deficiency
that survived measurement.

**2.5-D input**: feed `2k+1` adjacent axial slices as input channels, predict the
centre slice. Stays a 2-D network; near-zero extra parameters.

Required first: an `in_channels` argument on `UNetGenerator` (hardcoded to 1 at
[models.py:154](models.py#L154)), a depth-crop in `dataset.py`, and asymmetric
in/out depth in `infer_volume.py`. Sweep k=2 (7.5 mm) vs k=5 (16.5 mm).

Full 3-D only if 2.5-D moves vessel HU. It is more tractable than previously
assumed, since §1.5 shows the data is already isotropic.

### Gate E — cheap architecture fixes, each isolated

| # | change | why |
|---|---|---|
| **E1** | `l1_organ_groupnorm` | §1.7: on trained weights the residual seam is **64.5% DC**, the un-blendable component. One flag |
| **E2** | conditional D: `Conv(2, ndf, …)` + `cat([src, tgt])` | restores the real pix2pix baseline; may fix the discriminator-wins instability |
| **E3** | `nearest + conv` instead of `ConvTranspose` | checkerboard artifacts |
| **E4** | horizontal flip | **low priority** — §1.3, no gap to regularize |

Best combination → 3 seeds → that becomes the official baseline with its own σ.

### Gate F — attention, now clearly last

§1.9 removes the strongest argument for it: capacity is not binding, so extra
parameters are not the lever, and attention would have to win purely on inductive
bias. §1.7 removes the argument against it (the ERF objection was a normalization
artifact), so it remains defensible — just no longer the priority.

Build `models_attn.py` only after Gate E locks a baseline. Design decisions
(zero-init `gamma` gate, `window_size=8` for stride alignment, separable SD +
GroupNorm, at most one bottleneck block, raw-NCCT input) are unchanged and
recorded in git history of this file.

**Gate:** if the cheapest variant at 3 seeds does not clear Δ > 2σ on featHU
(0.82 HU), or `gamma → 0`, **stop.** With §1.9 in hand, "attention did not help,
and here is why the ceiling is elsewhere" is a complete and well-evidenced thesis
section — arguably a better one than a marginal win.

### Where the literature baseline landed

`pix2pixhd_baseline` is now benchmarked: featHU **17.07**, oMAE 0.0319,
`gen_prob` 0.914 (lowest of all 12 models). `l1_organ_curriculum` beats it on
featHU (15.74) and organ MAE (0.0299). Its `gradW1` is the best in the table
(0.0007) — the adversarial+perceptual combination does buy texture realism, and
pays for it in HU accuracy. That trade-off is a clean result to report.

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

- **n=20 test set** is thin, and §1.4 is the concrete cost: at n=20 only |r| > 0.44
  is significant, so that audit could not distinguish "no effect" from
  "underpowered". Expanding the test set is the cheapest way to harden both the
  headline claim and every case-level correlation.
- **`phase_acc` is saturated** (§1.6) — the downstream metric can no longer
  discriminate.
- **Zero-weighting bowel needs defending** — frame it as measured: 27% of voxels,
  ~1.3% of phase importance, gas not inferable from NCCT. Bowel is still
  *evaluated*, just not *trained*.
- **Masks derive from the target CECT**, so they cannot be fed as a generator
  input without NCCT-derived masks.
- **A mask whose shape does not match its volume silently becomes all-zero**
  ([dataset.py:452](dataset.py#L452)) and organ-focus silently finds no candidates
  ([dataset.py:491](dataset.py#L491)). Check coverage with
  `scripts/check_seg_masks.py` before trusting any organ-loss run.
- **`SEG_SUFFIX` must stay `'_seg_full'`** — `_seg_reg` uses different label
  conventions than the weights assume.
- **Selection-metric change breaks comparability** with the two oldest runs.
- **The output rsync keeps re-nesting.** A relative destination has twice produced
  `out_synthesis_train/out_synthesis_train/`. Use an absolute destination path.

---

## 5. Deferred, with the argument recorded

**CT-pretrained backbone as generator initialisation — argue against, now firmly.**
§1.9 rules out capacity: a 9× parameter increase bought 0.33 HU. A better
initialisation of the same architecture cannot recover error the input does not
determine. nnU-Net is also a segmentation framework whose
value is the self-configuring pipeline, and its decoder emits class logits.

**As a perceptual loss — well-motivated, and the better use.** The current
perceptual loss is VGG16 (ImageNet) or DINOv2/v3 (web images); neither has seen a
CT. `DinoPerceptualLoss` already accepts a pre-built backbone.

**Full 3-D** — only if Gate D's 2.5-D probe moves vessel HU. More tractable than
assumed: §1.5 shows the data is already isotropic 1.5 mm, so there is no
anisotropy to design around.

**Spatial resampling — dropped entirely.** §1.5 measured every volume at exactly
1.5 mm isotropic. There is nothing to normalise.
