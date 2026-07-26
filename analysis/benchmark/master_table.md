# NCCT→CECT benchmark — master table

All metrics on the shared HU[-200,400]→[0,1] domain, same test cases.

| model | n | PSNR | SSIM | MAE | MSE | PCC | oPSNR | oSSIM | oMAE | phase | prob | featHU | RAPS | gradW1 | oGradW1 | seam | zflick | zaniso |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| l1_adv | 20 | 30.06 | 0.9413 | 0.0099 | 0.00108 | 0.9863 | 24.26 | 0.9660 | 0.0319 | 1.00 | 0.9751 | 17.85 | 0.896 | 0.0009 | 0.0025 | 1.632 | 0.998 | 1.076 |
| l1_adv_organ | 20 | 30.01 | 0.9425 | 0.0098 | 0.00109 | 0.9862 | 24.22 | 0.9659 | 0.0317 | 1.00 | 0.9794 | 16.04 | 0.905 | 0.0009 | 0.0026 | 1.641 | 0.998 | 1.084 |
| l1_bowel_zero | 20 | 30.46 | 0.9439 | 0.0094 | 0.00098 | 0.9874 | 24.58 | 0.9684 | 0.0299 | 0.95 | 0.9517 | 17.39 | 0.836 | 0.0021 | 0.0065 | 1.655 | 0.901 | 1.095 |
| l1_only | 20 | 30.38 | 0.9413 | 0.0096 | 0.00100 | 0.9871 | 24.59 | 0.9684 | 0.0299 | 0.95 | 0.9484 | 17.32 | 0.837 | 0.0020 | 0.0064 | 1.672 | 0.909 | 1.096 |
| l1_organ_curriculum | 20 | 30.41 | 0.9436 | 0.0095 | 0.00099 | 0.9873 | 24.58 | 0.9684 | 0.0299 | 1.00 | 0.9792 | 15.74 | 0.840 | 0.0019 | 0.0062 | 1.663 | 0.911 | 1.098 |

oPSNR/oSSIM/oMAE = organ-region. featHU = mean per-organ |HU error|. Higher PSNR/SSIM/PCC/prob/phase better; lower MAE/MSE/featHU better.

**Texture and consistency** (`metrics.py`): RAPS = high-frequency spectral energy vs real; gradW1/oGradW1 = W1 distance between gradient-magnitude distributions, global and organ-region; seam = tile-boundary over interior gradient; zflick = inter-slice difference vs real. **RAPS, seam and zflick are ratios: 1.000 is the target and both directions are failures** — rank them by |value - 1|, never as "higher is better". RAPS < 1 is blur, > 1 is noise or hallucinated texture. seam > 1 is a visible tile boundary; zflick > 1 is slice-to-slice flicker. gradW1 is a distance: lower is better. seam is NaN for models with no known tiling geometry.

**Caveat:** external models retrained on this data at this scale do not reproduce their papers' reported numbers — this is a controlled same-data, same-split comparison, not a reproduction. PSNR/SSIM reward blur here (see PROJECT_PLAN.md); read organ-region + phase fidelity as primary.

## Paired per-case tests vs "l1_organ_curriculum"  (negative = better)

model           metric               delta       t  sig   better
l1_adv          feature_l1_hu       +2.110     +5.82   ***    2/20
                org_mae             +0.002    +11.84   ***    0/20
                org_ssim            +0.002     +9.12   ***    1/20
                psnr                +0.356     +7.57   ***    1/20
                |raps_hf-1|         -0.049     -5.39   ***   18/20
                grad_w1             -0.001    -15.80   ***   20/20
                org_grad_w1         -0.004    -25.19   ***   20/20
                |seam-1|            -0.030     -2.32     *   15/20
                |zflicker-1|        -0.047     -3.53    **   16/20
                |zaniso-1|          -0.021     -2.37     *   16/20

l1_adv_organ    feature_l1_hu       +0.304     +1.10    ns   10/20
                org_mae             +0.002     +8.18   ***    0/20
                org_ssim            +0.003     +4.28   ***    0/20
                psnr                +0.402     +4.28   ***    0/20
                |raps_hf-1|         -0.057     -6.23   ***   18/20
                grad_w1             -0.001    -14.45   ***   20/20
                org_grad_w1         -0.004    -19.09   ***   20/20
                |seam-1|            -0.022     -2.20     *   17/20
                |zflicker-1|        -0.049     -3.53    **   17/20
                |zaniso-1|          -0.014     -1.84    ns   16/20

l1_bowel_zero   feature_l1_hu       +1.653     +3.35    **    4/20
                org_mae             -0.000     -0.04    ns   11/20
                org_ssim            +0.000     +0.43    ns    9/20
                psnr                -0.044     -1.87    ns   14/20
                |raps_hf-1|         +0.001     +1.04    ns    6/20
                grad_w1             +0.000     +6.82   ***    0/20
                org_grad_w1         +0.000     +9.56   ***    0/20
                |seam-1|            -0.007     -2.20     *   14/20
                |zflicker-1|        +0.008     +3.98   ***    2/20
                |zaniso-1|          -0.003     -2.66     *   15/20

l1_only         feature_l1_hu       +1.581     +4.22   ***    5/20
                org_mae             +0.000     +0.19    ns   10/20
                org_ssim            +0.000     +0.44    ns   11/20
                psnr                +0.030     +0.54    ns   14/20
                |raps_hf-1|         +0.001     +1.20    ns    7/20
                grad_w1             +0.000     +0.92    ns    8/20
                org_grad_w1         +0.000     +6.51   ***    1/20
                |seam-1|            +0.010     +1.04    ns   11/20
                |zflicker-1|        +0.001     +0.38    ns    7/20
                |zaniso-1|          -0.002     -1.32    ns   15/20

  sign convention: negative delta = the model beats the baseline.
  sig: * p<.05  ** p<.01  *** p<.001 (paired, df~n-1)