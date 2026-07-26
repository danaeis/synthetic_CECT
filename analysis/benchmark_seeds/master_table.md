# NCCT→CECT benchmark — master table

All metrics on the shared HU[-200,400]→[0,1] domain, same test cases.

| model | n | PSNR | SSIM | MAE | MSE | PCC | oPSNR | oSSIM | oMAE | bPSNR | bMAE | body% | phase | prob | featHU | RAPS | gradW1 | oGradW1 | seam | zflick | zaniso |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| l1_adv_organ_s42 | 20 | 30.07 | 0.9441 | 0.0096 | 0.00107 | 0.9865 | 24.30 | 0.9667 | 0.0311 | 25.42 | 0.0291 | 0.32 | 1.00 | 0.9892 | 14.57 | 0.897 | 0.0011 | 0.0027 | 1.389 | 0.996 | 1.098 |
| l1_adv_organ_s43 | 20 | 30.13 | 0.9440 | 0.0094 | 0.00106 | 0.9864 | 24.26 | 0.9663 | 0.0307 | 25.35 | 0.0289 | 0.32 | 1.00 | 0.9927 | 13.90 | 0.951 | 0.0009 | 0.0030 | 1.381 | 1.004 | 1.085 |
| l1_adv_organ_s44 | 20 | 30.05 | 0.9440 | 0.0096 | 0.00107 | 0.9864 | 24.20 | 0.9661 | 0.0314 | 25.32 | 0.0294 | 0.32 | 1.00 | 0.9897 | 13.62 | 0.903 | 0.0009 | 0.0029 | 1.379 | 1.007 | 1.093 |
| l1_organ_curriculum_s42 | 20 | 30.38 | 0.9422 | 0.0095 | 0.00100 | 0.9872 | 24.58 | 0.9684 | 0.0299 | 25.70 | 0.0280 | 0.32 | 1.00 | 0.9843 | 14.85 | 0.822 | 0.0021 | 0.0067 | 1.341 | 0.908 | 1.107 |
| l1_organ_curriculum_s43 | 20 | 30.40 | 0.9372 | 0.0096 | 0.00100 | 0.9872 | 24.57 | 0.9683 | 0.0297 | 25.68 | 0.0279 | 0.32 | 1.00 | 0.9869 | 14.39 | 0.831 | 0.0020 | 0.0065 | 1.346 | 0.908 | 1.103 |
| l1_organ_curriculum_s44 | 20 | 30.34 | 0.9409 | 0.0097 | 0.00101 | 0.9871 | 24.57 | 0.9682 | 0.0298 | 25.68 | 0.0280 | 0.32 | 1.00 | 0.9889 | 14.26 | 0.824 | 0.0021 | 0.0066 | 1.351 | 0.907 | 1.106 |

oPSNR/oSSIM/oMAE = organ-region. featHU = mean per-organ |HU error|. Higher PSNR/SSIM/PCC/prob/phase better; lower MAE/MSE/featHU better.

**Texture and consistency** (`metrics.py`): RAPS = high-frequency spectral energy vs real; gradW1/oGradW1 = W1 distance between gradient-magnitude distributions, global and organ-region; seam = tile-boundary over interior gradient; zflick = inter-slice difference vs real. **RAPS, seam and zflick are ratios: 1.000 is the target and both directions are failures** — rank them by |value - 1|, never as "higher is better". RAPS < 1 is blur, > 1 is noise or hallucinated texture. seam > 1 is a visible tile boundary; zflick > 1 is slice-to-slice flicker. gradW1 is a distance: lower is better. seam is NaN for models with no known tiling geometry.

**Caveat:** external models retrained on this data at this scale do not reproduce their papers' reported numbers — this is a controlled same-data, same-split comparison, not a reproduction. PSNR/SSIM reward blur here (see PROJECT_PLAN.md); read organ-region + phase fidelity as primary.