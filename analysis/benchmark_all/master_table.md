# NCCT→CECT benchmark — master table

All metrics on the shared HU[-200,400]→[0,1] domain, same test cases.

| model | n | PSNR | SSIM | MAE | MSE | PCC | oPSNR | oSSIM | oMAE | bPSNR | bMAE | body% | phase | prob | featHU | RAPS | gradW1 | oGradW1 | seam | zflick | zaniso |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| l1_adv | 20 | 30.06 | 0.9413 | 0.0099 | 0.00108 | 0.9863 | 24.26 | 0.9660 | 0.0319 | 25.35 | 0.0299 | 0.32 | 1.00 | 0.9751 | 17.85 | 0.896 | 0.0009 | 0.0025 | 1.632 | 0.998 | 1.076 |
| l1_adv_organ | 20 | 30.01 | 0.9425 | 0.0098 | 0.00109 | 0.9862 | 24.22 | 0.9659 | 0.0317 | 25.29 | 0.0300 | 0.32 | 1.00 | 0.9794 | 16.04 | 0.905 | 0.0009 | 0.0026 | 1.641 | 0.998 | 1.084 |
| l1_adv_organ_s42 | 20 | 30.07 | 0.9441 | 0.0096 | 0.00107 | 0.9865 | 24.30 | 0.9667 | 0.0311 | 25.42 | 0.0291 | 0.32 | 1.00 | 0.9892 | 14.57 | 0.897 | 0.0011 | 0.0027 | 1.389 | 0.996 | 1.098 |
| l1_adv_organ_s43 | 20 | 30.13 | 0.9440 | 0.0094 | 0.00106 | 0.9864 | 24.26 | 0.9663 | 0.0307 | 25.35 | 0.0289 | 0.32 | 1.00 | 0.9927 | 13.90 | 0.951 | 0.0009 | 0.0030 | 1.381 | 1.004 | 1.085 |
| l1_adv_organ_s44 | 20 | 30.05 | 0.9440 | 0.0096 | 0.00107 | 0.9864 | 24.20 | 0.9661 | 0.0314 | 25.32 | 0.0294 | 0.32 | 1.00 | 0.9897 | 13.62 | 0.903 | 0.0009 | 0.0029 | 1.379 | 1.007 | 1.093 |
| l1_bowel_zero | 20 | 30.46 | 0.9439 | 0.0094 | 0.00098 | 0.9874 | 24.58 | 0.9684 | 0.0299 | 25.72 | 0.0279 | 0.32 | 0.95 | 0.9517 | 17.39 | 0.836 | 0.0021 | 0.0065 | 1.655 | 0.901 | 1.095 |
| l1_only | 20 | 30.38 | 0.9413 | 0.0096 | 0.00100 | 0.9871 | 24.59 | 0.9684 | 0.0299 | 25.70 | 0.0280 | 0.32 | 0.95 | 0.9484 | 17.32 | 0.837 | 0.0020 | 0.0064 | 1.672 | 0.909 | 1.096 |
| l1_organ_curriculum | 20 | 30.41 | 0.9436 | 0.0095 | 0.00099 | 0.9873 | 24.58 | 0.9684 | 0.0299 | 25.70 | 0.0280 | 0.32 | 1.00 | 0.9792 | 15.74 | 0.840 | 0.0019 | 0.0062 | 1.663 | 0.911 | 1.098 |
| l1_organ_curriculum_s42 | 20 | 30.38 | 0.9422 | 0.0095 | 0.00100 | 0.9872 | 24.58 | 0.9684 | 0.0299 | 25.70 | 0.0280 | 0.32 | 1.00 | 0.9843 | 14.85 | 0.822 | 0.0021 | 0.0067 | 1.341 | 0.908 | 1.107 |
| l1_organ_curriculum_s43 | 20 | 30.40 | 0.9372 | 0.0096 | 0.00100 | 0.9872 | 24.57 | 0.9683 | 0.0297 | 25.68 | 0.0279 | 0.32 | 1.00 | 0.9869 | 14.39 | 0.831 | 0.0020 | 0.0065 | 1.346 | 0.908 | 1.103 |
| l1_organ_curriculum_s44 | 20 | 30.34 | 0.9409 | 0.0097 | 0.00101 | 0.9871 | 24.57 | 0.9682 | 0.0298 | 25.68 | 0.0280 | 0.32 | 1.00 | 0.9889 | 14.26 | 0.824 | 0.0021 | 0.0066 | 1.351 | 0.907 | 1.106 |
| pix2pixhd_baseline | 20 | 30.01 | 0.9409 | 0.0100 | 0.00109 | 0.9861 | 24.23 | 0.9658 | 0.0319 | 25.30 | 0.0302 | 0.32 | 0.95 | 0.9141 | 17.07 | 0.895 | 0.0007 | 0.0020 | 1.385 | 1.021 | 1.080 |

oPSNR/oSSIM/oMAE = organ-region. featHU = mean per-organ |HU error|. Higher PSNR/SSIM/PCC/prob/phase better; lower MAE/MSE/featHU better.

**Texture and consistency** (`metrics.py`): RAPS = high-frequency spectral energy vs real; gradW1/oGradW1 = W1 distance between gradient-magnitude distributions, global and organ-region; seam = tile-boundary over interior gradient; zflick = inter-slice difference vs real. **RAPS, seam and zflick are ratios: 1.000 is the target and both directions are failures** — rank them by |value - 1|, never as "higher is better". RAPS < 1 is blur, > 1 is noise or hallucinated texture. seam > 1 is a visible tile boundary; zflick > 1 is slice-to-slice flicker. gradW1 is a distance: lower is better. seam is NaN for models with no known tiling geometry.

**Caveat:** external models retrained on this data at this scale do not reproduce their papers' reported numbers — this is a controlled same-data, same-split comparison, not a reproduction. PSNR/SSIM reward blur here (see PROJECT_PLAN.md); read organ-region + phase fidelity as primary.