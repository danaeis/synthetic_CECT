# NCCT→CECT benchmark — master table

All metrics on the shared HU[-200,400]→[0,1] domain, same test cases. Metrics are split into categories (one table each); models are grouped by generator architecture. In every column **bold = best**, _italic = second best_, ranked in that metric's own direction (higher-better, lower-better, or closest-to-1.0 for the ratio metrics). Reference/floor rows are shown for scale but excluded from ranking.


### Image-level (global pixel)

| model | n | PSNR | SSIM | MAE | MSE | PCC |
|---|---|---|---|---|---|---|
| **UNet + PatchGAN (this repo)** |  |  |  |  |  |  |
| cytran | 20 | 28.87 ± 1.40 | _0.9296_ ± 0.0112 | 0.0127 ± 0.0019 | 0.00137 ± 0.00051 | 0.9835 ± 0.0062 |
| swinunetr_s5 | 20 | **29.34** ± 1.59 | **0.9301** ± 0.0117 | **0.0110** ± 0.0020 | **0.00125** ± 0.00052 | **0.9841** ± 0.0060 |
| transunet | 20 | 29.00 ± 1.59 | 0.9271 ± 0.0120 | 0.0115 ± 0.0020 | 0.00135 ± 0.00056 | 0.9829 ± 0.0063 |
| **External baseline** |  |  |  |  |  |  |
| resvit | 20 | _29.18_ ± 1.59 | 0.9293 ± 0.0116 | _0.0110_ ± 0.0019 | _0.00129_ ± 0.00053 | _0.9836_ ± 0.0060 |

### Organ-level (region-restricted)

| model | oPSNR | oSSIM | oMAE | bPSNR | bMAE | featHU |
|---|---|---|---|---|---|---|
| **UNet + PatchGAN (this repo)** |  |  |  |  |  |  |
| cytran | 22.70 ± 1.39 | 0.9547 ± 0.0177 | 0.0451 ± 0.0081 | 24.14 ± 1.68 | 0.0387 ± 0.0074 | 38.12 ± 12.81 |
| swinunetr_s5 | **23.59** ± 1.41 | **0.9611** ± 0.0153 | **0.0348** ± 0.0048 | **24.65** ± 1.76 | **0.0331** ± 0.0062 | _16.72_ ± 7.00 |
| transunet | 23.23 ± 1.44 | 0.9579 ± 0.0160 | 0.0368 ± 0.0046 | 24.28 ± 1.77 | 0.0349 ± 0.0064 | 17.15 ± 7.15 |
| **External baseline** |  |  |  |  |  |  |
| resvit | _23.44_ ± 1.42 | _0.9598_ ± 0.0157 | _0.0349_ ± 0.0046 | _24.51_ ± 1.75 | _0.0332_ ± 0.0062 | **15.17** ± 7.05 |

### Phase & level fidelity

| model | phase | prob | βlev | varR |
|---|---|---|---|---|
| **UNet + PatchGAN (this repo)** |  |  |  |  |
| cytran | 0.35 ± 0.11 | 0.2839 ± 0.2732 | 0.09 ± 0.28 | **0.55** ± 0.11 |
| swinunetr_s5 | _0.95_ ± 0.05 | 0.9224 ± 0.1747 | _0.26_ ± 0.15 | 0.29 ± 0.12 |
| transunet | _0.95_ ± 0.05 | _0.9432_ ± 0.1454 | 0.23 ± 0.06 | 0.32 ± 0.11 |
| **External baseline** |  |  |  |  |
| resvit | **1.00** ± 0.00 | **0.9674** ± 0.0849 | **0.31** ± 0.15 | _0.38_ ± 0.13 |

### Detail-focused (texture & consistency)

| model | RAPS | gradW1 | oGradW1 | seam | zflick | zaniso |
|---|---|---|---|---|---|---|
| **UNet + PatchGAN (this repo)** |  |  |  |  |  |  |
| cytran | **0.744** ± 0.071 | 0.0015 ± 0.0007 | 0.0031 ± 0.0016 | — | _1.074_ ± 0.087 | 1.212 ± 0.086 |
| swinunetr_s5 | 0.712 ± 0.074 | 0.0013 ± 0.0007 | 0.0027 ± 0.0012 | — | 0.900 ± 0.062 | **1.025** ± 0.059 |
| transunet | 0.713 ± 0.078 | **0.0013** ± 0.0007 | _0.0025_ ± 0.0011 | — | 1.085 ± 0.064 | 1.227 ± 0.071 |
| **External baseline** |  |  |  |  |  |  |
| resvit | _0.723_ ± 0.075 | _0.0013_ ± 0.0008 | **0.0025** ± 0.0010 | — | **1.038** ± 0.061 | _1.179_ ± 0.077 |

### Perceptual (literature comparability)

_Not computed. Re-run `benchmark.py --perceptual` (needs `torch` + `lpips` + `pytorch-fid`) to populate LPIPS and FID; they are comparability-only and secondary to the CT-native RAPS/gradW1 columns above._

**Spread.** Every cell is `mean ± sd` **across the n test cases**, sample sd (ddof=1), so it is directly comparable to the `x ± s` convention in the published tables. Bold/italic mark the best and second-best MEAN only. Two exceptions: **phase** and **agree_real** are rates over 0/1 outcomes, whose sd is fixed by the mean alone, so those carry the binomial standard error `sqrt(p(1-p)/n)` instead; and **βlev/varR** spread is across ORGANS, not cases, since each organ already used all n cases to fit its slope. FID has no per-case value at all (it is distributional) and so has no ±.

Between-case variance dominates between-model differences on this data — a large ± next to a small difference in means is the normal situation here, and it is the reason model comparisons go through the PAIRED per-case tests below rather than through these spreads.

**How to read these tables.** The *Image-level (global pixel)* category is SECONDARY and flat by construction: `to_unit` saturates air/lung/fat→0 and bone→1 identically in every model, so those columns average over a large error-free mass and an identity copy of the NCCT already scores most of the way to the best model on them (see metrics.py:body_mask). Read the PRIMARY categories instead — organ-level (oMAE/featHU), phase & level fidelity (phase/prob/βlev/varR) and detail-focused texture (RAPS/gradW1).

oPSNR/oSSIM/oMAE = organ-region. featHU = mean per-organ |HU error|. Higher PSNR/SSIM/PCC/prob/phase better; lower MAE/MSE/featHU better.

**Level recovery** (βlev, varR): generated per-organ median HU regressed on the real one across cases, averaged over aorta, portal_vein_and_splenic_vein, inferior_vena_cava, liver. **βlev** = mean slope and **varR** = mean var(gen)/var(real). Both target **1.0**: βlev/varR → 1 means the model tracks each case's true contrast level; βlev/varR → 0 means it emits the population average and is indistinguishable from a conditional-mean predictor — the textbook signature of an L1/L2 loss under irreducible enhancement uncertainty (dose/bolus timing are not visible in NCCT). featHU can look decent while varR is near 0, so these two columns are what separate a real generator from an averager (full breakdown: scripts/audit_enhancement.py). βlev/varR are NaN when generated volumes are not in HU (--gen_not_hu) or an organ map is unavailable.

**Texture and consistency** (`metrics.py`): RAPS = high-frequency spectral energy vs real; gradW1/oGradW1 = W1 distance between gradient-magnitude distributions, global and organ-region; seam = tile-boundary over interior gradient; zflick = inter-slice difference vs real. **RAPS, seam and zflick are ratios: 1.000 is the target and both directions are failures** — rank them by |value - 1|, never as "higher is better". RAPS < 1 is blur, > 1 is noise or hallucinated texture. seam > 1 is a visible tile boundary; zflick > 1 is slice-to-slice flicker. gradW1 is a distance: lower is better. seam is NaN for models with no known tiling geometry.

**Caveat:** external models retrained on this data at this scale do not reproduce their papers' reported numbers — this is a controlled same-data, same-split comparison, not a reproduction. PSNR/SSIM reward blur here (see PROJECT_PLAN.md); read organ-region + phase fidelity as primary.

## Paired per-case tests vs "cytran"  (negative = better)

model                     metric               delta       t  sig   better
resvit                    — no cases in common with the baseline; not comparable

swinunetr_s5              feature_l1_hu      -21.404    -11.70   ***   20/20
                          org_mae             -0.010     -8.38   ***   20/20
                          org_ssim            -0.006     -4.21   ***   18/20
                          psnr                -0.468     -3.57    **   15/20
                          body_mae            -0.006     -8.17   ***   20/20
                          |raps_hf-1|         +0.031     +4.04   ***    5/20
                          grad_w1             -0.000     -1.69    ns   12/20
                          org_grad_w1         -0.000     -1.71    ns   11/20
                          |seam-1|            +0.000     +0.00    ns    0/20
                          |zflicker-1|        +0.023     +0.88    ns    4/20
                          |zaniso-1|          -0.165    -13.99   ***   20/20

transunet                 feature_l1_hu      -20.971    -11.30   ***   20/20
                          org_mae             -0.008     -6.83   ***   20/20
                          org_ssim            -0.003     -1.96    ns   12/20
                          psnr                -0.124     -0.87    ns   11/20
                          body_mae            -0.004     -5.47   ***   18/20
                          |raps_hf-1|         +0.031     +3.40    **    4/20
                          grad_w1             -0.000     -2.20     *   16/20
                          org_grad_w1         -0.001     -2.15     *   14/20
                          |seam-1|            +0.000     +0.00    ns    0/20
                          |zflicker-1|        +0.007     +0.59    ns    6/20
                          |zaniso-1|          +0.014     +1.84    ns    6/20

  sign convention: negative delta = the model beats the baseline.
  sig: * p<.05  ** p<.01  *** p<.001 (paired, df~n-1)

_Table built from 4 model(s) accumulated in `analysis/bench_ncct2cect/store` (1 scored in this run). Each model is scored once and cached; every cross-model quantity — ranking, paired deltas, level-recovery regressions — is recomputed from the merged per-case rows on every run._