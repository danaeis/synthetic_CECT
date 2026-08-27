# NCCT→CECT benchmark — master table

All metrics on the shared HU[-200,400]→[0,1] domain, same test cases. Metrics are split into categories (one table each); models are grouped by generator architecture. In every column **bold = best**, _italic = second best_, ranked in that metric's own direction (higher-better, lower-better, or closest-to-1.0 for the ratio metrics). Reference/floor rows are shown for scale but excluded from ranking.


### Image-level (global pixel)

| model | n | PSNR | SSIM | MAE | MSE | PCC |
|---|---|---|---|---|---|---|
| **UNet + PatchGAN (this repo)** |  |  |  |  |  |  |
| b0_groupnorm_adv | 20 | 29.91 ± 1.58 | 0.9390 ± 0.0094 | 0.0107 ± 0.0016 | 0.00110 ± 0.00048 | 0.9860 ± 0.0063 |
| l1_adv | 20 | 30.06 ± 1.78 | 0.9413 ± 0.0123 | 0.0099 ± 0.0021 | 0.00108 ± 0.00050 | 0.9863 ± 0.0065 |
| l1_adv_organ | 20 | 30.01 ± 1.80 | 0.9425 ± 0.0113 | 0.0098 ± 0.0022 | 0.00109 ± 0.00050 | 0.9862 ± 0.0067 |
| l1_adv_organ_s42 | 20 | 30.07 ± 1.80 | 0.9441 ± 0.0113 | 0.0096 ± 0.0019 | 0.00107 ± 0.00051 | 0.9865 ± 0.0064 |
| l1_adv_organ_s43 | 20 | 30.13 ± 1.79 | 0.9440 ± 0.0107 | 0.0094 ± 0.0019 | 0.00106 ± 0.00051 | 0.9864 ± 0.0068 |
| l1_adv_organ_s44 | 20 | 30.05 ± 1.74 | 0.9440 ± 0.0114 | 0.0096 ± 0.0019 | 0.00107 ± 0.00050 | 0.9864 ± 0.0065 |
| l1_bowel_zero | 20 | 30.46 ± 1.79 | 0.9439 ± 0.0108 | 0.0094 ± 0.0018 | 0.00098 ± 0.00048 | 0.9874 ± 0.0063 |
| l1_huprofile_only | 20 | 30.31 ± 1.77 | 0.9390 ± 0.0144 | 0.0098 ± 0.0018 | 0.00102 ± 0.00049 | 0.9870 ± 0.0065 |
| l1_only | 20 | 30.38 ± 1.85 | 0.9413 ± 0.0147 | 0.0096 ± 0.0020 | 0.00100 ± 0.00049 | 0.9871 ± 0.0065 |
| l1_organ_curriculum | 20 | 30.41 ± 1.79 | 0.9436 ± 0.0122 | 0.0095 ± 0.0018 | 0.00099 ± 0.00048 | 0.9873 ± 0.0063 |
| l1_organ_curriculum_s42 | 20 | 30.38 ± 1.80 | 0.9422 ± 0.0130 | 0.0095 ± 0.0018 | 0.00100 ± 0.00048 | 0.9872 ± 0.0064 |
| l1_organ_curriculum_s43 | 20 | 30.40 ± 1.80 | 0.9372 ± 0.0172 | 0.0096 ± 0.0018 | 0.00100 ± 0.00048 | 0.9872 ± 0.0064 |
| l1_organ_curriculum_s44 | 20 | 30.34 ± 1.80 | 0.9409 ± 0.0137 | 0.0097 ± 0.0019 | 0.00101 ± 0.00048 | 0.9871 ± 0.0064 |
| l1_organ_groupnorm_s42 | 20 | 30.57 ± 1.92 | 0.9486 ± 0.0101 | 0.0088 ± 0.0017 | 0.00097 ± 0.00052 | 0.9876 ± 0.0064 |
| l1_organ_groupnorm_s43 | 20 | 30.66 ± 1.87 | 0.9492 ± 0.0097 | 0.0087 ± 0.0016 | 0.00095 ± 0.00050 | 0.9879 ± 0.0064 |
| l1_organ_groupnorm_s44 | 20 | 30.67 ± 1.86 | 0.9493 ± 0.0096 | 0.0087 ± 0.0016 | 0.00095 ± 0.00050 | 0.9879 ± 0.0063 |
| l1_organ_huprofile | 20 | 29.28 ± 1.78 | 0.8169 ± 0.0624 | 0.0146 ± 0.0031 | 0.00129 ± 0.00065 | 0.9842 ± 0.0077 |
| level_all8 | 20 | 29.66 ± 1.44 | 0.9397 ± 0.0089 | 0.0097 ± 0.0015 | 0.00115 ± 0.00048 | 0.9852 ± 0.0062 |
| level_aorta | 20 | 29.68 ± 1.54 | 0.9420 ± 0.0093 | 0.0107 ± 0.0017 | 0.00115 ± 0.00048 | 0.9856 ± 0.0063 |
| level_aorta_pv | 20 | 29.44 ± 1.87 | 0.9378 ± 0.0118 | 0.0099 ± 0.0019 | 0.00125 ± 0.00062 | 0.9842 ± 0.0070 |
| memorize97 | 20 | 30.28 ± 1.85 | 0.9411 ± 0.0144 | 0.0098 ± 0.0021 | 0.00103 ± 0.00049 | 0.9869 ± 0.0065 |
| multiphase_film/arterial | 20 | 29.89 ± 1.45 | 0.9427 ± 0.0126 | 0.0099 ± 0.0019 | 0.00108 ± 0.00035 | 0.9862 ± 0.0043 |
| multiphase_film/venous | 20 | 30.34 ± 1.82 | 0.9436 ± 0.0132 | 0.0095 ± 0.0020 | 0.00101 ± 0.00049 | 0.9870 ± 0.0065 |
| multiphase_film_adv/arterial | 20 | 28.18 ± 1.35 | 0.9093 ± 0.0309 | 0.0140 ± 0.0025 | 0.00159 ± 0.00052 | 0.9800 ± 0.0061 |
| multiphase_film_adv/venous | 20 | 28.51 ± 1.41 | 0.9105 ± 0.0302 | 0.0147 ± 0.0024 | 0.00149 ± 0.00054 | 0.9820 ± 0.0073 |
| multiphase_film_adv_slices11/arterial | 20 | 28.85 ± 1.12 | 0.9354 ± 0.0091 | 0.0108 ± 0.0015 | 0.00134 ± 0.00035 | 0.9829 ± 0.0037 |
| multiphase_film_adv_slices11/venous | 20 | 29.51 ± 1.50 | 0.9373 ± 0.0088 | 0.0103 ± 0.0015 | 0.00119 ± 0.00050 | 0.9848 ± 0.0064 |
| multiphase_uncond/arterial | 20 | 29.50 ± 1.46 | 0.9333 ± 0.0178 | 0.0111 ± 0.0021 | 0.00118 ± 0.00041 | 0.9850 ± 0.0049 |
| multiphase_uncond/venous | 20 | 30.05 ± 1.81 | 0.9344 ± 0.0185 | 0.0104 ± 0.0021 | 0.00108 ± 0.00051 | 0.9862 ± 0.0068 |
| ncase10 | 20 | 29.83 ± 1.79 | 0.9363 ± 0.0127 | 0.0108 ± 0.0025 | 0.00113 ± 0.00053 | 0.9859 ± 0.0067 |
| ncase25 | 20 | 30.07 ± 1.88 | 0.9357 ± 0.0202 | 0.0103 ± 0.0023 | 0.00108 ± 0.00052 | 0.9863 ± 0.0069 |
| ncase50 | 20 | 30.10 ± 1.91 | 0.9374 ± 0.0179 | 0.0101 ± 0.0023 | 0.00107 ± 0.00052 | 0.9864 ± 0.0068 |
| slices11_k5 | 20 | 30.73 ± 1.87 | _0.9501_ ± 0.0092 | _0.0086_ ± 0.0016 | _0.00093_ ± 0.00048 | _0.9881_ ± 0.0062 |
| slices5_k2 | 20 | _30.73_ ± 1.88 | **0.9502** ± 0.0093 | **0.0086** ± 0.0016 | 0.00093 ± 0.00049 | 0.9881 ± 0.0062 |
| width32 | 20 | 30.28 ± 1.83 | 0.9331 ± 0.0192 | 0.0100 ± 0.0020 | 0.00103 ± 0.00050 | 0.9868 ± 0.0068 |
| width96 | 20 | 30.42 ± 1.78 | 0.9455 ± 0.0111 | 0.0094 ± 0.0019 | 0.00099 ± 0.00048 | 0.9874 ± 0.0063 |
| **Diffusion** |  |  |  |  |  |  |
| diff_hetero_nll | 20 | **30.82** ± 1.79 | 0.9489 ± 0.0096 | 0.0088 ± 0.0015 | **0.00091** ± 0.00047 | **0.9883** ± 0.0062 |
| diff_l1_organ_groupnorm | 20 | 30.57 ± 1.92 | 0.9486 ± 0.0101 | 0.0088 ± 0.0017 | 0.00097 ± 0.00052 | 0.9876 ± 0.0064 |
| diff_v | 20 | 28.79 ± 1.36 | 0.9088 ± 0.0097 | 0.0124 ± 0.0017 | 0.00139 ± 0.00050 | 0.9822 ± 0.0063 |
| diff_v_nocfg | 20 | 28.88 ± 1.45 | 0.9198 ± 0.0096 | 0.0118 ± 0.0017 | 0.00137 ± 0.00052 | 0.9825 ± 0.0063 |
| diff_v_organ | 20 | 28.40 ± 1.27 | 0.9268 ± 0.0105 | 0.0125 ± 0.0018 | 0.00151 ± 0.00051 | 0.9808 ± 0.0063 |
| diff_x0 | 20 | 28.74 ± 1.35 | 0.9339 ± 0.0099 | 0.0113 ± 0.0017 | 0.00141 ± 0.00051 | 0.9820 ± 0.0062 |
| **External baseline** |  |  |  |  |  |  |
| pix2pixhd_baseline | 20 | 30.01 ± 1.82 | 0.9409 ± 0.0123 | 0.0100 ± 0.0023 | 0.00109 ± 0.00051 | 0.9861 ± 0.0068 |

### Organ-level (region-restricted)

| model | oPSNR | oSSIM | oMAE | bPSNR | bMAE | featHU |
|---|---|---|---|---|---|---|
| **UNet + PatchGAN (this repo)** |  |  |  |  |  |  |
| b0_groupnorm_adv | 23.83 ± 1.54 | 0.9633 ± 0.0162 | 0.0351 ± 0.0049 | 25.08 ± 1.89 | 0.0315 ± 0.0064 | 26.76 ± 9.17 |
| l1_adv | 24.26 ± 1.70 | 0.9660 ± 0.0162 | 0.0319 ± 0.0055 | 25.35 ± 2.00 | 0.0299 ± 0.0067 | 17.85 ± 6.21 |
| l1_adv_organ | 24.22 ± 1.71 | 0.9659 ± 0.0163 | 0.0317 ± 0.0054 | 25.29 ± 1.99 | 0.0300 ± 0.0066 | 16.04 ± 5.66 |
| l1_adv_organ_s42 | 24.30 ± 1.71 | 0.9667 ± 0.0161 | 0.0311 ± 0.0052 | 25.42 ± 1.99 | 0.0291 ± 0.0064 | 14.57 ± 5.05 |
| l1_adv_organ_s43 | 24.26 ± 1.74 | 0.9663 ± 0.0165 | 0.0307 ± 0.0053 | 25.35 ± 2.02 | 0.0289 ± 0.0065 | 13.90 ± 4.79 |
| l1_adv_organ_s44 | 24.20 ± 1.70 | 0.9661 ± 0.0162 | 0.0314 ± 0.0052 | 25.32 ± 1.99 | 0.0294 ± 0.0065 | 13.62 ± 4.56 |
| l1_bowel_zero | 24.58 ± 1.73 | 0.9684 ± 0.0161 | 0.0299 ± 0.0052 | 25.72 ± 2.04 | 0.0279 ± 0.0063 | 17.39 ± 7.03 |
| l1_huprofile_only | 24.46 ± 1.73 | 0.9680 ± 0.0161 | 0.0307 ± 0.0053 | 25.58 ± 2.01 | 0.0286 ± 0.0064 | 18.31 ± 5.35 |
| l1_only | 24.59 ± 1.75 | 0.9684 ± 0.0161 | 0.0299 ± 0.0053 | 25.70 ± 2.06 | 0.0280 ± 0.0064 | 17.32 ± 6.42 |
| l1_organ_curriculum | 24.58 ± 1.76 | 0.9684 ± 0.0161 | 0.0299 ± 0.0052 | 25.70 ± 2.05 | 0.0280 ± 0.0063 | 15.74 ± 5.86 |
| l1_organ_curriculum_s42 | 24.58 ± 1.73 | 0.9684 ± 0.0160 | 0.0299 ± 0.0050 | 25.70 ± 2.03 | 0.0280 ± 0.0062 | 14.85 ± 5.93 |
| l1_organ_curriculum_s43 | 24.57 ± 1.75 | 0.9683 ± 0.0161 | 0.0297 ± 0.0051 | 25.68 ± 2.03 | 0.0279 ± 0.0063 | 14.39 ± 5.37 |
| l1_organ_curriculum_s44 | 24.57 ± 1.75 | 0.9682 ± 0.0161 | 0.0298 ± 0.0052 | 25.68 ± 2.03 | 0.0280 ± 0.0063 | 14.26 ± 5.68 |
| l1_organ_groupnorm_s42 | 24.77 ± 1.79 | 0.9698 ± 0.0160 | 0.0288 ± 0.0051 | 25.93 ± 2.15 | 0.0268 ± 0.0065 | 13.88 ± 5.88 |
| l1_organ_groupnorm_s43 | 24.80 ± 1.81 | 0.9698 ± 0.0162 | 0.0286 ± 0.0052 | 25.96 ± 2.15 | 0.0267 ± 0.0065 | **12.90** ± 4.78 |
| l1_organ_groupnorm_s44 | 24.77 ± 1.79 | 0.9698 ± 0.0161 | 0.0288 ± 0.0052 | 25.95 ± 2.15 | 0.0268 ± 0.0066 | 13.60 ± 5.41 |
| l1_organ_huprofile | 23.77 ± 1.65 | 0.9629 ± 0.0170 | 0.0359 ± 0.0066 | 24.66 ± 1.91 | 0.0343 ± 0.0078 | 16.22 ± 6.22 |
| level_all8 | 23.73 ± 1.47 | 0.9625 ± 0.0162 | 0.0320 ± 0.0049 | 24.85 ± 1.77 | 0.0299 ± 0.0062 | 14.43 ± 3.69 |
| level_aorta | 23.79 ± 1.52 | 0.9632 ± 0.0160 | 0.0354 ± 0.0049 | 24.88 ± 1.82 | 0.0326 ± 0.0062 | 26.10 ± 8.07 |
| level_aorta_pv | 23.89 ± 1.59 | 0.9635 ± 0.0169 | 0.0316 ± 0.0051 | 24.94 ± 1.87 | 0.0299 ± 0.0064 | 14.81 ± 4.09 |
| memorize97 | 24.61 ± 1.74 | 0.9687 ± 0.0160 | 0.0298 ± 0.0051 | 25.68 ± 2.02 | 0.0282 ± 0.0062 | 16.32 ± 6.22 |
| multiphase_film/arterial | 23.76 ± 1.44 | 0.9646 ± 0.0109 | 0.0330 ± 0.0052 | 25.12 ± 1.52 | 0.0292 ± 0.0048 | 36.73 ± 18.21 |
| multiphase_film/venous | 24.52 ± 1.74 | 0.9680 ± 0.0162 | 0.0299 ± 0.0051 | 25.62 ± 2.04 | 0.0282 ± 0.0063 | 14.26 ± 5.10 |
| multiphase_film_adv/arterial | 21.90 ± 1.47 | 0.9463 ± 0.0191 | 0.0447 ± 0.0078 | 23.37 ± 1.40 | 0.0398 ± 0.0068 | 63.57 ± 23.03 |
| multiphase_film_adv/venous | 22.76 ± 1.38 | 0.9561 ± 0.0176 | 0.0448 ± 0.0070 | 23.75 ± 1.61 | 0.0422 ± 0.0081 | 35.01 ± 10.49 |
| multiphase_film_adv_slices11/arterial | 22.64 ± 1.15 | 0.9555 ± 0.0107 | 0.0381 ± 0.0049 | 24.01 ± 1.33 | 0.0334 ± 0.0053 | 42.47 ± 16.17 |
| multiphase_film_adv_slices11/venous | 23.71 ± 1.50 | 0.9629 ± 0.0159 | 0.0338 ± 0.0049 | 24.73 ± 1.83 | 0.0318 ± 0.0065 | 18.75 ± 7.94 |
| multiphase_uncond/arterial | 23.14 ± 1.42 | 0.9589 ± 0.0134 | 0.0374 ± 0.0059 | 24.70 ± 1.52 | 0.0318 ± 0.0052 | 50.34 ± 21.12 |
| multiphase_uncond/venous | 24.10 ± 1.67 | 0.9651 ± 0.0164 | 0.0332 ± 0.0058 | 25.31 ± 2.00 | 0.0300 ± 0.0068 | 19.39 ± 5.61 |
| ncase10 | 24.02 ± 1.66 | 0.9649 ± 0.0164 | 0.0335 ± 0.0064 | 25.06 ± 1.89 | 0.0314 ± 0.0068 | 20.98 ± 7.75 |
| ncase25 | 24.37 ± 1.67 | 0.9673 ± 0.0160 | 0.0313 ± 0.0052 | 25.49 ± 1.96 | 0.0292 ± 0.0063 | 16.78 ± 6.03 |
| ncase50 | 24.40 ± 1.73 | 0.9674 ± 0.0161 | 0.0308 ± 0.0055 | 25.51 ± 2.03 | 0.0288 ± 0.0065 | 15.08 ± 5.08 |
| slices11_k5 | **24.89** ± 1.81 | _0.9705_ ± 0.0159 | _0.0284_ ± 0.0051 | _26.06_ ± 2.16 | _0.0264_ ± 0.0064 | _13.15_ ± 5.17 |
| slices5_k2 | _24.89_ ± 1.82 | **0.9705** ± 0.0159 | **0.0283** ± 0.0051 | **26.06** ± 2.16 | **0.0263** ± 0.0064 | 13.29 ± 5.36 |
| width32 | 24.44 ± 1.73 | 0.9673 ± 0.0163 | 0.0306 ± 0.0053 | 25.52 ± 2.03 | 0.0288 ± 0.0065 | 15.58 ± 5.94 |
| width96 | 24.61 ± 1.75 | 0.9687 ± 0.0159 | 0.0297 ± 0.0052 | 25.69 ± 2.02 | 0.0280 ± 0.0063 | 13.95 ± 4.80 |
| **Diffusion** |  |  |  |  |  |  |
| diff_hetero_nll | 24.89 ± 1.77 | 0.9702 ± 0.0160 | 0.0298 ± 0.0052 | 26.02 ± 2.12 | 0.0274 ± 0.0066 | 14.46 ± 5.21 |
| diff_l1_organ_groupnorm | 24.77 ± 1.79 | 0.9698 ± 0.0160 | 0.0288 ± 0.0051 | 25.93 ± 2.15 | 0.0268 ± 0.0065 | 13.88 ± 5.88 |
| diff_v | 22.82 ± 1.30 | 0.9544 ± 0.0165 | 0.0390 ± 0.0052 | 24.03 ± 1.65 | 0.0350 ± 0.0067 | 22.25 ± 8.75 |
| diff_v_nocfg | 22.99 ± 1.35 | 0.9560 ± 0.0164 | 0.0371 ± 0.0050 | 24.15 ± 1.70 | 0.0338 ± 0.0067 | 18.36 ± 7.17 |
| diff_v_organ | 22.68 ± 1.26 | 0.9531 ± 0.0165 | 0.0410 ± 0.0053 | 23.64 ± 1.55 | 0.0378 ± 0.0068 | 14.60 ± 6.92 |
| diff_x0 | 22.71 ± 1.28 | 0.9535 ± 0.0165 | 0.0392 ± 0.0054 | 23.95 ± 1.64 | 0.0349 ± 0.0069 | 24.76 ± 9.53 |
| **External baseline** |  |  |  |  |  |  |
| pix2pixhd_baseline | 24.23 ± 1.73 | 0.9658 ± 0.0166 | 0.0319 ± 0.0055 | 25.30 ± 2.00 | 0.0302 ± 0.0066 | 17.07 ± 5.52 |

### Phase & level fidelity

| model | phase | prob | βlev | varR |
|---|---|---|---|---|
| **UNet + PatchGAN (this repo)** |  |  |  |  |
| b0_groupnorm_adv | **1.00** ± 0.00 | 0.9636 ± 0.0879 | 0.11 ± 0.12 | 0.18 ± 0.17 |
| l1_adv | **1.00** ± 0.00 | 0.9751 ± 0.0452 | 0.17 ± 0.07 | 0.13 ± 0.06 |
| l1_adv_organ | **1.00** ± 0.00 | 0.9794 ± 0.0341 | 0.21 ± 0.09 | 0.16 ± 0.04 |
| l1_adv_organ_s42 | **1.00** ± 0.00 | 0.9892 ± 0.0101 | 0.20 ± 0.11 | 0.16 ± 0.07 |
| l1_adv_organ_s43 | **1.00** ± 0.00 | 0.9927 ± 0.0085 | 0.19 ± 0.07 | 0.14 ± 0.03 |
| l1_adv_organ_s44 | **1.00** ± 0.00 | 0.9897 ± 0.0118 | 0.24 ± 0.11 | 0.21 ± 0.09 |
| l1_bowel_zero | _0.95_ ± 0.05 | 0.9517 ± 0.1226 | 0.21 ± 0.07 | 0.14 ± 0.04 |
| l1_huprofile_only | **1.00** ± 0.00 | 0.9768 ± 0.0381 | 0.23 ± 0.07 | 0.19 ± 0.08 |
| l1_only | _0.95_ ± 0.05 | 0.9484 ± 0.1265 | 0.21 ± 0.09 | 0.15 ± 0.06 |
| l1_organ_curriculum | **1.00** ± 0.00 | 0.9792 ± 0.0297 | 0.23 ± 0.07 | 0.15 ± 0.04 |
| l1_organ_curriculum_s42 | **1.00** ± 0.00 | 0.9843 ± 0.0209 | 0.20 ± 0.11 | 0.16 ± 0.08 |
| l1_organ_curriculum_s43 | **1.00** ± 0.00 | 0.9869 ± 0.0129 | 0.22 ± 0.09 | 0.15 ± 0.06 |
| l1_organ_curriculum_s44 | **1.00** ± 0.00 | 0.9889 ± 0.0130 | 0.20 ± 0.08 | 0.17 ± 0.08 |
| l1_organ_groupnorm_s42 | **1.00** ± 0.00 | 0.9879 ± 0.0331 | 0.26 ± 0.13 | 0.24 ± 0.13 |
| l1_organ_groupnorm_s43 | **1.00** ± 0.00 | 0.9894 ± 0.0294 | 0.25 ± 0.14 | 0.24 ± 0.15 |
| l1_organ_groupnorm_s44 | **1.00** ± 0.00 | 0.9705 ± 0.0841 | 0.25 ± 0.11 | 0.20 ± 0.11 |
| l1_organ_huprofile | **1.00** ± 0.00 | _0.9960_ ± 0.0043 | 0.06 ± 0.06 | 0.12 ± 0.08 |
| level_all8 | **1.00** ± 0.00 | 0.9840 ± 0.0188 | **0.68** ± 0.21 | **0.64** ± 0.26 |
| level_aorta | **1.00** ± 0.00 | 0.9781 ± 0.0425 | 0.12 ± 0.11 | 0.15 ± 0.10 |
| level_aorta_pv | **1.00** ± 0.00 | 0.9812 ± 0.0271 | _0.53_ ± 0.18 | _0.42_ ± 0.17 |
| memorize97 | **1.00** ± 0.00 | 0.9869 ± 0.0111 | 0.26 ± 0.06 | 0.20 ± 0.08 |
| multiphase_film/arterial | **1.00** ± 0.00 | **0.9962** ± 0.0037 | 0.18 ± 0.08 | 0.14 ± 0.06 |
| multiphase_film/venous | **1.00** ± 0.00 | 0.9861 ± 0.0206 | 0.22 ± 0.06 | 0.18 ± 0.06 |
| multiphase_film_adv/arterial | **1.00** ± 0.00 | 0.9075 ± 0.1982 | 0.05 ± 0.09 | 0.09 ± 0.02 |
| multiphase_film_adv/venous | 0.85 ± 0.08 | 0.6358 ± 0.2316 | 0.07 ± 0.19 | 0.23 ± 0.14 |
| multiphase_film_adv_slices11/arterial | **1.00** ± 0.00 | 0.9884 ± 0.0368 | 0.17 ± 0.09 | 0.12 ± 0.06 |
| multiphase_film_adv_slices11/venous | **1.00** ± 0.00 | 0.9899 ± 0.0188 | 0.16 ± 0.11 | 0.28 ± 0.26 |
| multiphase_uncond/arterial | 0.25 ± 0.10 | 0.3366 ± 0.3506 | 0.15 ± 0.12 | 0.16 ± 0.17 |
| multiphase_uncond/venous | 0.70 ± 0.10 | 0.6021 ± 0.3179 | 0.27 ± 0.13 | 0.35 ± 0.15 |
| ncase10 | **1.00** ± 0.00 | 0.9740 ± 0.0380 | 0.16 ± 0.14 | 0.17 ± 0.11 |
| ncase25 | **1.00** ± 0.00 | 0.9879 ± 0.0103 | 0.22 ± 0.04 | 0.18 ± 0.06 |
| ncase50 | **1.00** ± 0.00 | 0.9918 ± 0.0077 | 0.23 ± 0.04 | 0.20 ± 0.12 |
| slices11_k5 | **1.00** ± 0.00 | 0.9935 ± 0.0161 | 0.23 ± 0.08 | 0.21 ± 0.08 |
| slices5_k2 | **1.00** ± 0.00 | 0.9684 ± 0.0863 | 0.23 ± 0.09 | 0.22 ± 0.08 |
| width32 | **1.00** ± 0.00 | 0.9870 ± 0.0141 | 0.18 ± 0.08 | 0.16 ± 0.07 |
| width96 | **1.00** ± 0.00 | 0.9890 ± 0.0130 | 0.26 ± 0.07 | 0.20 ± 0.09 |
| **Diffusion** |  |  |  |  |
| diff_hetero_nll | **1.00** ± 0.00 | 0.9909 ± 0.0168 | 0.33 ± 0.15 | 0.27 ± 0.17 |
| diff_l1_organ_groupnorm | **1.00** ± 0.00 | 0.9879 ± 0.0331 | 0.26 ± 0.13 | 0.24 ± 0.13 |
| diff_v | **1.00** ± 0.00 | 0.9641 ± 0.0880 | 0.09 ± 0.13 | 0.14 ± 0.07 |
| diff_v_nocfg | **1.00** ± 0.00 | 0.9575 ± 0.0933 | 0.19 ± 0.13 | 0.18 ± 0.11 |
| diff_v_organ | **1.00** ± 0.00 | 0.9932 ± 0.0125 | 0.11 ± 0.03 | 0.14 ± 0.10 |
| diff_x0 | **1.00** ± 0.00 | 0.9500 ± 0.1018 | 0.08 ± 0.12 | 0.21 ± 0.12 |
| **External baseline** |  |  |  |  |
| pix2pixhd_baseline | _0.95_ ± 0.05 | 0.9141 ± 0.2110 | 0.15 ± 0.08 | 0.18 ± 0.05 |

### Detail-focused (texture & consistency)

| model | RAPS | gradW1 | oGradW1 | seam | zflick | zaniso |
|---|---|---|---|---|---|---|
| **UNet + PatchGAN (this repo)** |  |  |  |  |  |  |
| b0_groupnorm_adv | 0.801 ± 0.072 | 0.0017 ± 0.0005 | 0.0060 ± 0.0015 | 1.371 ± 0.226 | 0.927 ± 0.060 | 1.105 ± 0.062 |
| l1_adv | 0.896 ± 0.077 | _0.0009_ ± 0.0006 | 0.0025 ± 0.0012 | 1.632 ± 0.235 | _0.998_ ± 0.065 | 1.076 ± 0.067 |
| l1_adv_organ | 0.905 ± 0.077 | 0.0009 ± 0.0005 | 0.0026 ± 0.0012 | 1.641 ± 0.250 | **0.998** ± 0.064 | 1.084 ± 0.065 |
| l1_adv_organ_s42 | 0.897 ± 0.079 | 0.0011 ± 0.0006 | 0.0027 ± 0.0012 | 1.389 ± 0.205 | 0.996 ± 0.064 | 1.098 ± 0.069 |
| l1_adv_organ_s43 | **0.951** ± 0.081 | 0.0009 ± 0.0004 | 0.0030 ± 0.0011 | 1.381 ± 0.211 | 1.004 ± 0.066 | 1.085 ± 0.062 |
| l1_adv_organ_s44 | 0.903 ± 0.079 | 0.0009 ± 0.0005 | 0.0029 ± 0.0011 | 1.379 ± 0.205 | 1.007 ± 0.064 | 1.093 ± 0.065 |
| l1_bowel_zero | 0.836 ± 0.077 | 0.0021 ± 0.0007 | 0.0065 ± 0.0016 | 1.655 ± 0.259 | 0.901 ± 0.056 | 1.095 ± 0.059 |
| l1_huprofile_only | 0.826 ± 0.078 | 0.0019 ± 0.0007 | 0.0060 ± 0.0015 | 1.350 ± 0.219 | 0.923 ± 0.056 | 1.111 ± 0.057 |
| l1_only | 0.837 ± 0.078 | 0.0020 ± 0.0007 | 0.0064 ± 0.0016 | 1.672 ± 0.264 | 0.909 ± 0.056 | 1.096 ± 0.059 |
| l1_organ_curriculum | 0.840 ± 0.080 | 0.0019 ± 0.0007 | 0.0062 ± 0.0015 | 1.663 ± 0.257 | 0.911 ± 0.056 | 1.098 ± 0.058 |
| l1_organ_curriculum_s42 | 0.822 ± 0.079 | 0.0021 ± 0.0007 | 0.0067 ± 0.0016 | 1.341 ± 0.219 | 0.908 ± 0.057 | 1.107 ± 0.058 |
| l1_organ_curriculum_s43 | 0.831 ± 0.078 | 0.0020 ± 0.0008 | 0.0065 ± 0.0016 | 1.346 ± 0.216 | 0.908 ± 0.057 | 1.103 ± 0.060 |
| l1_organ_curriculum_s44 | 0.824 ± 0.078 | 0.0021 ± 0.0008 | 0.0066 ± 0.0016 | 1.351 ± 0.219 | 0.907 ± 0.056 | 1.106 ± 0.058 |
| l1_organ_groupnorm_s42 | 0.841 ± 0.078 | 0.0021 ± 0.0008 | 0.0065 ± 0.0016 | 1.356 ± 0.221 | 0.897 ± 0.058 | 1.099 ± 0.063 |
| l1_organ_groupnorm_s43 | 0.842 ± 0.078 | 0.0021 ± 0.0007 | 0.0066 ± 0.0015 | 1.353 ± 0.224 | 0.893 ± 0.058 | 1.093 ± 0.061 |
| l1_organ_groupnorm_s44 | 0.841 ± 0.078 | 0.0021 ± 0.0007 | 0.0067 ± 0.0015 | 1.354 ± 0.223 | 0.893 ± 0.057 | 1.095 ± 0.061 |
| l1_organ_huprofile | 0.811 ± 0.072 | 0.0014 ± 0.0006 | 0.0053 ± 0.0014 | **1.332** ± 0.205 | 0.983 ± 0.064 | 1.111 ± 0.060 |
| level_all8 | 0.935 ± 0.093 | 0.0015 ± 0.0004 | 0.0051 ± 0.0013 | 1.384 ± 0.225 | 0.976 ± 0.070 | 1.125 ± 0.062 |
| level_aorta | 0.820 ± 0.075 | 0.0017 ± 0.0005 | 0.0056 ± 0.0015 | 1.370 ± 0.231 | 0.933 ± 0.062 | 1.113 ± 0.064 |
| level_aorta_pv | 0.939 ± 0.090 | 0.0018 ± 0.0009 | 0.0048 ± 0.0010 | 1.402 ± 0.209 | 0.964 ± 0.066 | 1.127 ± 0.073 |
| memorize97 | 0.839 ± 0.077 | 0.0020 ± 0.0007 | 0.0066 ± 0.0015 | 1.349 ± 0.222 | 0.912 ± 0.056 | 1.102 ± 0.056 |
| multiphase_film/arterial | 0.816 ± 0.096 | 0.0022 ± 0.0010 | 0.0072 ± 0.0028 | 1.368 ± 0.220 | 0.899 ± 0.068 | 1.102 ± 0.036 |
| multiphase_film/venous | 0.841 ± 0.077 | 0.0020 ± 0.0007 | 0.0063 ± 0.0015 | 1.351 ± 0.222 | 0.914 ± 0.056 | 1.109 ± 0.058 |
| multiphase_film_adv/arterial | 0.746 ± 0.086 | 0.0025 ± 0.0011 | 0.0070 ± 0.0027 | 1.404 ± 0.209 | 0.930 ± 0.067 | 1.135 ± 0.037 |
| multiphase_film_adv/venous | 0.773 ± 0.072 | 0.0023 ± 0.0008 | 0.0056 ± 0.0014 | 1.395 ± 0.207 | 0.953 ± 0.061 | 1.150 ± 0.063 |
| multiphase_film_adv_slices11/arterial | 0.923 ± 0.116 | 0.0013 ± 0.0009 | 0.0042 ± 0.0025 | 1.431 ± 0.224 | 0.887 ± 0.088 | _0.992_ ± 0.052 |
| multiphase_film_adv_slices11/venous | _0.945_ ± 0.087 | 0.0012 ± 0.0005 | 0.0039 ± 0.0013 | 1.393 ± 0.224 | 0.902 ± 0.067 | **1.004** ± 0.066 |
| multiphase_uncond/arterial | 0.810 ± 0.099 | 0.0021 ± 0.0009 | 0.0076 ± 0.0029 | 1.346 ± 0.218 | 0.916 ± 0.069 | 1.112 ± 0.042 |
| multiphase_uncond/venous | 0.846 ± 0.078 | 0.0018 ± 0.0006 | 0.0059 ± 0.0015 | 1.346 ± 0.218 | 0.945 ± 0.059 | 1.123 ± 0.060 |
| ncase10 | 0.902 ± 0.083 | 0.0015 ± 0.0004 | 0.0054 ± 0.0014 | 1.352 ± 0.234 | 0.952 ± 0.061 | 1.091 ± 0.058 |
| ncase25 | 0.841 ± 0.076 | 0.0017 ± 0.0006 | 0.0060 ± 0.0015 | _1.339_ ± 0.221 | 0.931 ± 0.058 | 1.099 ± 0.057 |
| ncase50 | 0.853 ± 0.078 | 0.0018 ± 0.0006 | 0.0061 ± 0.0015 | 1.344 ± 0.219 | 0.932 ± 0.057 | 1.107 ± 0.058 |
| slices11_k5 | 0.840 ± 0.079 | 0.0021 ± 0.0008 | 0.0065 ± 0.0016 | 1.361 ± 0.225 | 0.790 ± 0.061 | 0.962 ± 0.064 |
| slices5_k2 | 0.844 ± 0.079 | 0.0021 ± 0.0008 | 0.0067 ± 0.0016 | 1.354 ± 0.222 | 0.792 ± 0.061 | 0.967 ± 0.065 |
| width32 | 0.831 ± 0.075 | 0.0019 ± 0.0007 | 0.0061 ± 0.0015 | 1.350 ± 0.216 | 0.927 ± 0.057 | 1.116 ± 0.059 |
| width96 | 0.826 ± 0.078 | 0.0020 ± 0.0007 | 0.0065 ± 0.0016 | 1.345 ± 0.219 | 0.906 ± 0.056 | 1.099 ± 0.057 |
| **Diffusion** |  |  |  |  |  |  |
| diff_hetero_nll | 0.800 ± 0.075 | 0.0022 ± 0.0007 | 0.0069 ± 0.0016 | 1.356 ± 0.229 | 0.893 ± 0.060 | 1.103 ± 0.061 |
| diff_l1_organ_groupnorm | 0.841 ± 0.078 | 0.0021 ± 0.0008 | 0.0065 ± 0.0016 | 1.356 ± 0.221 | 0.897 ± 0.058 | 1.099 ± 0.063 |
| diff_v | 0.886 ± 0.080 | 0.0012 ± 0.0006 | 0.0034 ± 0.0014 | 1.367 ± 0.208 | 1.857 ± 0.151 | 2.045 ± 0.170 |
| diff_v_nocfg | 0.890 ± 0.082 | 0.0013 ± 0.0006 | 0.0034 ± 0.0013 | 1.367 ± 0.205 | 1.751 ± 0.146 | 1.932 ± 0.165 |
| diff_v_organ | 0.867 ± 0.077 | 0.0009 ± 0.0005 | _0.0024_ ± 0.0013 | 1.396 ± 0.202 | 1.886 ± 0.153 | 2.031 ± 0.163 |
| diff_x0 | 0.848 ± 0.080 | 0.0016 ± 0.0006 | 0.0046 ± 0.0014 | 1.365 ± 0.222 | 1.586 ± 0.139 | 1.852 ± 0.166 |
| **External baseline** |  |  |  |  |  |  |
| pix2pixhd_baseline | 0.895 ± 0.080 | **0.0007** ± 0.0003 | **0.0020** ± 0.0011 | 1.385 ± 0.202 | 1.021 ± 0.071 | 1.080 ± 0.066 |

### Perceptual (literature comparability)

| model | LPIPS | FID | FID(Rad) |
|---|---|---|---|
| **UNet + PatchGAN (this repo)** |  |  |  |
| b0_groupnorm_adv | 0.0537 ± 0.0134 | 17.4 | 0.1 |
| l1_adv | 0.0388 ± 0.0126 | 8.2 | 0.0 |
| l1_adv_organ | 0.0384 ± 0.0095 | **7.2** | 0.1 |
| l1_adv_organ_s42 | _0.0378_ ± 0.0103 | 8.7 | 0.0 |
| l1_adv_organ_s43 | **0.0371** ± 0.0104 | 10.5 | **0.0** |
| l1_adv_organ_s44 | 0.0400 ± 0.0132 | 9.2 | 0.0 |
| l1_bowel_zero | 0.0463 ± 0.0082 | 21.9 | 0.1 |
| l1_huprofile_only | 0.0466 ± 0.0081 | 21.4 | 0.1 |
| l1_only | 0.0470 ± 0.0086 | 21.7 | 0.1 |
| l1_organ_curriculum | 0.0470 ± 0.0087 | 21.4 | 0.1 |
| l1_organ_curriculum_s42 | 0.0464 ± 0.0077 | 23.4 | 0.1 |
| l1_organ_curriculum_s43 | 0.0466 ± 0.0084 | 23.1 | 0.1 |
| l1_organ_curriculum_s44 | 0.0473 ± 0.0081 | 23.8 | 0.1 |
| l1_organ_groupnorm_s42 | 0.0478 ± 0.0086 | 28.2 | 0.1 |
| l1_organ_groupnorm_s43 | 0.0467 ± 0.0087 | 28.3 | 0.1 |
| l1_organ_groupnorm_s44 | 0.0464 ± 0.0079 | 28.3 | 0.1 |
| l1_organ_huprofile | 0.0531 ± 0.0134 | 23.3 | 0.1 |
| level_all8 | 0.0518 ± 0.0105 | 20.7 | 0.1 |
| level_aorta | 0.0551 ± 0.0149 | 17.4 | 0.1 |
| level_aorta_pv | 0.0528 ± 0.0096 | 21.4 | 0.1 |
| memorize97 | 0.0484 ± 0.0079 | 24.1 | 0.1 |
| multiphase_film/arterial | 0.0503 ± 0.0098 | 24.4 | 0.1 |
| multiphase_film/venous | 0.0462 ± 0.0083 | 23.5 | 0.1 |
| multiphase_film_adv/arterial | 0.0675 ± 0.0122 | 19.9 | 0.3 |
| multiphase_film_adv/venous | 0.0620 ± 0.0115 | 17.6 | 0.2 |
| multiphase_film_adv_slices11/arterial | 0.0538 ± 0.0118 | 15.6 | 0.1 |
| multiphase_film_adv_slices11/venous | 0.0500 ± 0.0131 | 15.2 | 0.1 |
| multiphase_uncond/arterial | 0.0518 ± 0.0104 | 24.9 | 0.2 |
| multiphase_uncond/venous | 0.0461 ± 0.0087 | 22.4 | 0.1 |
| ncase10 | 0.0452 ± 0.0090 | 17.6 | 0.1 |
| ncase25 | 0.0472 ± 0.0094 | 20.3 | 0.1 |
| ncase50 | 0.0463 ± 0.0087 | 22.1 | 0.1 |
| slices11_k5 | 0.0459 ± 0.0085 | 25.6 | 0.1 |
| slices5_k2 | 0.0457 ± 0.0077 | 26.8 | 0.1 |
| width32 | 0.0451 ± 0.0083 | 18.8 | 0.1 |
| width96 | 0.0465 ± 0.0078 | 25.2 | 0.1 |
| **Diffusion** |  |  |  |
| diff_hetero_nll | 0.0482 ± 0.0089 | 22.4 | 0.1 |
| diff_l1_organ_groupnorm | 0.0478 ± 0.0086 | 28.2 | 0.1 |
| diff_v | 0.0512 ± 0.0100 | 13.5 | 0.1 |
| diff_v_nocfg | 0.0502 ± 0.0096 | 13.3 | 0.1 |
| diff_v_organ | 0.0594 ± 0.0103 | 28.5 | 0.1 |
| diff_x0 | 0.0549 ± 0.0091 | 18.3 | 0.1 |
| **External baseline** |  |  |  |
| pix2pixhd_baseline | 0.0382 ± 0.0123 | _7.6_ | _0.0_ |

**Spread.** Every cell is `mean ± sd` **across the n test cases**, sample sd (ddof=1), so it is directly comparable to the `x ± s` convention in the published tables. Bold/italic mark the best and second-best MEAN only. Two exceptions: **phase** and **agree_real** are rates over 0/1 outcomes, whose sd is fixed by the mean alone, so those carry the binomial standard error `sqrt(p(1-p)/n)` instead; and **βlev/varR** spread is across ORGANS, not cases, since each organ already used all n cases to fit its slope. FID has no per-case value at all (it is distributional) and so has no ±.

Between-case variance dominates between-model differences on this data — a large ± next to a small difference in means is the normal situation here, and it is the reason model comparisons go through the PAIRED per-case tests below rather than through these spreads.

**How to read these tables.** The *Image-level (global pixel)* category is SECONDARY and flat by construction: `to_unit` saturates air/lung/fat→0 and bone→1 identically in every model, so those columns average over a large error-free mass and an identity copy of the NCCT already scores most of the way to the best model on them (see metrics.py:body_mask). Read the PRIMARY categories instead — organ-level (oMAE/featHU), phase & level fidelity (phase/prob/βlev/varR) and detail-focused texture (RAPS/gradW1).

oPSNR/oSSIM/oMAE = organ-region. featHU = mean per-organ |HU error|. Higher PSNR/SSIM/PCC/prob/phase better; lower MAE/MSE/featHU better.

**Level recovery** (βlev, varR): generated per-organ median HU regressed on the real one across cases, averaged over aorta, portal_vein_and_splenic_vein, inferior_vena_cava, liver. **βlev** = mean slope and **varR** = mean var(gen)/var(real). Both target **1.0**: βlev/varR → 1 means the model tracks each case's true contrast level; βlev/varR → 0 means it emits the population average and is indistinguishable from a conditional-mean predictor — the textbook signature of an L1/L2 loss under irreducible enhancement uncertainty (dose/bolus timing are not visible in NCCT). featHU can look decent while varR is near 0, so these two columns are what separate a real generator from an averager (full breakdown: scripts/audit_enhancement.py). βlev/varR are NaN when generated volumes are not in HU (--gen_not_hu) or an organ map is unavailable.

**Texture and consistency** (`metrics.py`): RAPS = high-frequency spectral energy vs real; gradW1/oGradW1 = W1 distance between gradient-magnitude distributions, global and organ-region; seam = tile-boundary over interior gradient; zflick = inter-slice difference vs real. **RAPS, seam and zflick are ratios: 1.000 is the target and both directions are failures** — rank them by |value - 1|, never as "higher is better". RAPS < 1 is blur, > 1 is noise or hallucinated texture. seam > 1 is a visible tile boundary; zflick > 1 is slice-to-slice flicker. gradW1 is a distance: lower is better. seam is NaN for models with no known tiling geometry.

**LPIPS / FID** (`perceptual.py`) — reported for comparability with the published NCCT→CECT literature, **not** as primary evidence. Both run ImageNet-pretrained networks on grayscale CT, so their absolute values have no physical meaning here and their ordering is not validated for this domain; the CT-native RAPS/gradW1 columns are what the texture claims rest on. LPIPS (alex) is paired and per-case, lower is better, computed at native in-plane resolution. FID is distributional: one value per model over ~5459 pooled body-containing axial slices from n=20 volumes, so it has no per-case value and no paired test. **FID is biased upward at small sample size** — these values are comparable between the rows of this table (identical slice counts, identical real set) and to nothing else. Backend: pytorch-fid (TF-ported InceptionV3).

**FID(Rad)** (`perceptual.py`) — the same distributional FID, from a ResNet50 pretrained on RadImageNet (radiology images) instead of ImageNet photographs, reported because the NCCT→CECT literature increasingly does. It is a DIFFERENT quantity from **FID** above — never rank or difference the two columns against each other. Reported evidence is mixed on whether this backbone is actually more valid for medical-image FID than ImageNet's: Woodland et al. 2024 ("Feature Extraction for Generative Medical Imaging Evaluation") found RadImageNet-based FID rankings were MORE volatile and LESS aligned with human judgment than ImageNet-based ones in their tests. Treat FID(Rad) as one more comparability column, not a more-trustworthy replacement for FID or for RAPS/gradW1. Backend: RadImageNet ResNet50 (PyTorch state_dict, community port).

**Caveat:** external models retrained on this data at this scale do not reproduce their papers' reported numbers — this is a controlled same-data, same-split comparison, not a reproduction. PSNR/SSIM reward blur here (see PROJECT_PLAN.md); read organ-region + phase fidelity as primary.

## Paired per-case tests vs "l1_only"  (negative = better)

model                                  metric               delta       t  sig   better
b0_groupnorm_adv                       feature_l1_hu       +9.437     +9.57   ***    0/20
                                       org_mae             +0.005    +14.91   ***    0/20
                                       org_ssim            +0.005     +5.76   ***    2/20
                                       psnr                +0.476     +3.06    **    3/20
                                       body_mae            +0.004     +7.11   ***    1/20
                                       |raps_hf-1|         +0.031     +7.02   ***    2/20
                                       grad_w1             -0.000     -2.63     *   16/20
                                       org_grad_w1         -0.000     -3.81    **   17/20
                                       |seam-1|            -0.294    -14.44   ***   20/20
                                       |zflicker-1|        -0.010     -1.56    ns   16/20
                                       |zaniso-1|          +0.009     +3.10    **    6/20
                                       lpips               +0.007     +3.32    **    5/20

diff_hetero_nll                        feature_l1_hu       -2.857     -3.52    **   15/20
                                       org_mae             -0.000     -0.42    ns    6/20
                                       org_ssim            -0.002     -3.95   ***   20/20
                                       psnr                -0.443     -3.02    **   20/20
                                       body_mae            -0.001     -1.12    ns    9/20
                                       |raps_hf-1|         +0.032     +7.56   ***    2/20
                                       grad_w1             +0.000     +3.16    **    2/20
                                       org_grad_w1         +0.000     +5.58   ***    0/20
                                       |seam-1|            -0.306    -13.97   ***   20/20
                                       |zflicker-1|        +0.017     +3.23    **    3/20
                                       |zaniso-1|          +0.006     +2.23     *    5/20
                                       lpips               +0.001     +1.17    ns    5/20

diff_l1_organ_groupnorm                feature_l1_hu       -3.437     -5.13   ***   18/20
                                       org_mae             -0.001     -3.49    **   18/20
                                       org_ssim            -0.001     -3.12    **   20/20
                                       psnr                -0.188     -1.15    ns   13/20
                                       body_mae            -0.001     -2.32     *   19/20
                                       |raps_hf-1|         -0.003     -1.06    ns   12/20
                                       grad_w1             +0.000     +3.14    **    5/20
                                       org_grad_w1         +0.000     +1.21    ns    8/20
                                       |seam-1|            -0.308    -13.84   ***   20/20
                                       |zflicker-1|        +0.012     +2.37     *    8/20
                                       |zaniso-1|          +0.003     +0.75    ns   10/20
                                       lpips               +0.001     +0.76    ns    5/20

diff_v                                 feature_l1_hu       +4.927     +6.99   ***    0/20
                                       org_mae             +0.009    +24.98   ***    0/20
                                       org_ssim            +0.014    +20.07   ***    0/20
                                       psnr                +1.594     +8.63   ***    1/20
                                       body_mae            +0.007    +11.39   ***    1/20
                                       |raps_hf-1|         -0.038     -4.92   ***   18/20
                                       grad_w1             -0.001    -10.03   ***   20/20
                                       org_grad_w1         -0.003    -19.82   ***   20/20
                                       |seam-1|            -0.304    -14.43   ***   20/20
                                       |zflicker-1|        +0.760    +19.57   ***    0/20
                                       |zaniso-1|          +0.948    +29.77   ***    0/20
                                       lpips               +0.004     +2.75     *    2/20

diff_v_nocfg                           feature_l1_hu       +1.041     +2.68     *    5/20
                                       org_mae             +0.007    +19.29   ***    0/20
                                       org_ssim            +0.012    +18.84   ***    0/20
                                       psnr                +1.501     +8.11   ***    1/20
                                       body_mae            +0.006     +9.50   ***    1/20
                                       |raps_hf-1|         -0.040     -4.68   ***   18/20
                                       grad_w1             -0.001    -11.29   ***   20/20
                                       org_grad_w1         -0.003    -20.56   ***   20/20
                                       |seam-1|            -0.305    -14.26   ***   20/20
                                       |zflicker-1|        +0.653    +17.38   ***    0/20
                                       |zaniso-1|          +0.836    +27.20   ***    0/20
                                       lpips               +0.003     +2.25     *    2/20

diff_v_organ                           feature_l1_hu       -2.715     -5.16   ***   18/20
                                       org_mae             +0.011    +29.26   ***    0/20
                                       org_ssim            +0.015    +19.10   ***    0/20
                                       psnr                +1.977    +10.27   ***    0/20
                                       body_mae            +0.010    +15.18   ***    1/20
                                       |raps_hf-1|         -0.025     -4.92   ***   18/20
                                       grad_w1             -0.001    -11.70   ***   20/20
                                       org_grad_w1         -0.004    -22.86   ***   20/20
                                       |seam-1|            -0.276    -12.89   ***   20/20
                                       |zflicker-1|        +0.789    +20.44   ***    0/20
                                       |zaniso-1|          +0.934    +31.71   ***    0/20
                                       lpips               +0.012     +7.11   ***    1/20

diff_x0                                feature_l1_hu       +7.442     +7.65   ***    0/20
                                       org_mae             +0.009    +20.89   ***    0/20
                                       org_ssim            +0.015    +18.52   ***    0/20
                                       psnr                +1.642     +8.51   ***    1/20
                                       body_mae            +0.007    +10.69   ***    1/20
                                       |raps_hf-1|         -0.010     -2.45     *   15/20
                                       grad_w1             -0.000     -4.43   ***   18/20
                                       org_grad_w1         -0.002    -12.96   ***   20/20
                                       |seam-1|            -0.302    -14.46   ***   20/20
                                       |zflicker-1|        +0.489    +13.96   ***    0/20
                                       |zaniso-1|          +0.755    +24.60   ***    0/20
                                       lpips               +0.008     +6.76   ***    1/20

l1_adv                                 feature_l1_hu       +0.529     +1.37    ns    9/20
                                       org_mae             +0.002    +10.28   ***    0/20
                                       org_ssim            +0.002    +13.24   ***    0/20
                                       psnr                +0.326     +6.02   ***    1/20
                                       body_mae            +0.002    +10.47   ***    0/20
                                       |raps_hf-1|         -0.050     -5.43   ***   18/20
                                       grad_w1             -0.001    -15.62   ***   20/20
                                       org_grad_w1         -0.004    -23.10   ***   20/20
                                       |seam-1|            -0.040     -4.39   ***   17/20
                                       |zflicker-1|        -0.048     -3.33    **   17/20
                                       |zaniso-1|          -0.019     -2.12     *   16/20
                                       lpips               -0.008     -3.99   ***   18/20

l1_adv_organ                           feature_l1_hu       -1.277     -3.20    **   14/20
                                       org_mae             +0.002     +7.71   ***    0/20
                                       org_ssim            +0.002     +5.57   ***    0/20
                                       psnr                +0.371     +4.47   ***    1/20
                                       body_mae            +0.002     +7.08   ***    0/20
                                       |raps_hf-1|         -0.058     -6.13   ***   18/20
                                       grad_w1             -0.001    -14.50   ***   20/20
                                       org_grad_w1         -0.004    -18.01   ***   20/20
                                       |seam-1|            -0.032     -3.23    **   18/20
                                       |zflicker-1|        -0.050     -3.31    **   17/20
                                       |zaniso-1|          -0.012     -1.57    ns   16/20
                                       lpips               -0.009     -7.10   ***   18/20

l1_adv_organ_s42                       feature_l1_hu       -2.748     -3.90   ***   15/20
                                       org_mae             +0.001     +5.78   ***    3/20
                                       org_ssim            +0.002     +8.40   ***    1/20
                                       psnr                +0.306     +3.41    **    3/20
                                       body_mae            +0.001     +7.21   ***    1/20
                                       |raps_hf-1|         -0.048     -5.40   ***   18/20
                                       grad_w1             -0.001    -16.00   ***   20/20
                                       org_grad_w1         -0.004    -20.34   ***   20/20
                                       |seam-1|            -0.283    -13.63   ***   20/20
                                       |zflicker-1|        -0.050     -3.33    **   17/20
                                       |zaniso-1|          +0.002     +0.19    ns   16/20
                                       lpips               -0.009     -5.60   ***   18/20

l1_adv_organ_s43                       feature_l1_hu       -3.419     -4.83   ***   17/20
                                       org_mae             +0.001     +4.82   ***    4/20
                                       org_ssim            +0.002     +4.28   ***    1/20
                                       psnr                +0.251     +2.71     *    3/20
                                       body_mae            +0.001     +5.47   ***    3/20
                                       |raps_hf-1|         -0.089     -5.63   ***   18/20
                                       grad_w1             -0.001    -10.00   ***   20/20
                                       org_grad_w1         -0.003    -17.96   ***   20/20
                                       |seam-1|            -0.290    -15.36   ***   20/20
                                       |zflicker-1|        -0.051     -3.16    **   16/20
                                       |zaniso-1|          -0.012     -1.93    ns   16/20
                                       lpips               -0.010     -5.91   ***   18/20

l1_adv_organ_s44                       feature_l1_hu       -3.696     -4.64   ***   15/20
                                       org_mae             +0.002     +7.08   ***    0/20
                                       org_ssim            +0.002    +10.74   ***    0/20
                                       psnr                +0.334     +5.13   ***    1/20
                                       body_mae            +0.001    +10.23   ***    0/20
                                       |raps_hf-1|         -0.054     -5.76   ***   18/20
                                       grad_w1             -0.001    -12.14   ***   20/20
                                       org_grad_w1         -0.004    -21.15   ***   20/20
                                       |seam-1|            -0.293    -14.73   ***   20/20
                                       |zflicker-1|        -0.054     -3.28    **   16/20
                                       |zaniso-1|          -0.004     -0.49    ns   16/20
                                       lpips               -0.007     -3.13    **   18/20

l1_bowel_zero                          feature_l1_hu       +0.072     +0.23    ns   11/20
                                       org_mae             -0.000     -0.27    ns   10/20
                                       org_ssim            -0.000     -0.11    ns   11/20
                                       psnr                -0.075     -1.23    ns   12/20
                                       body_mae            -0.000     -0.99    ns   10/20
                                       |raps_hf-1|         +0.000     +0.01    ns   11/20
                                       grad_w1             +0.000     +5.76   ***    0/20
                                       org_grad_w1         +0.000     +0.83    ns   10/20
                                       |seam-1|            -0.017     -2.44     *   19/20
                                       |zflicker-1|        +0.007     +3.59    **    2/20
                                       |zaniso-1|          -0.001     -0.65    ns   12/20
                                       lpips               -0.001     -1.08    ns    9/20

l1_huprofile_only                      feature_l1_hu       +0.993     +1.90    ns    8/20
                                       org_mae             +0.001     +4.24   ***    3/20
                                       org_ssim            +0.000     +1.44    ns    6/20
                                       psnr                +0.076     +1.43    ns    4/20
                                       body_mae            +0.001     +4.41   ***    4/20
                                       |raps_hf-1|         +0.009     +3.82    **    4/20
                                       grad_w1             -0.000     -1.90    ns   12/20
                                       org_grad_w1         -0.000     -8.28   ***   20/20
                                       |seam-1|            -0.312    -14.06   ***   20/20
                                       |zflicker-1|        -0.010     -3.81    **   18/20
                                       |zaniso-1|          +0.014    +11.92   ***    0/20
                                       lpips               -0.000     -0.57    ns   12/20

l1_organ_curriculum                    feature_l1_hu       -1.581     -4.22   ***   15/20
                                       org_mae             -0.000     -0.19    ns   10/20
                                       org_ssim            -0.000     -0.44    ns    9/20
                                       psnr                -0.030     -0.54    ns    6/20
                                       body_mae            +0.000     +0.04    ns   10/20
                                       |raps_hf-1|         -0.001     -1.20    ns   13/20
                                       grad_w1             -0.000     -0.92    ns   12/20
                                       org_grad_w1         -0.000     -6.51   ***   19/20
                                       |seam-1|            -0.010     -1.04    ns    9/20
                                       |zflicker-1|        -0.001     -0.38    ns   13/20
                                       |zaniso-1|          +0.002     +1.32    ns    5/20
                                       lpips               -0.000     -0.09    ns   14/20

l1_organ_curriculum_s42                feature_l1_hu       -2.469     -5.12   ***   18/20
                                       org_mae             -0.000     -0.17    ns    9/20
                                       org_ssim            -0.000     -0.62    ns   10/20
                                       psnr                -0.003     -0.10    ns    6/20
                                       body_mae            -0.000     -0.15    ns    7/20
                                       |raps_hf-1|         +0.012     +3.61    **    4/20
                                       grad_w1             +0.000     +3.55    **    2/20
                                       org_grad_w1         +0.000     +5.45   ***    1/20
                                       |seam-1|            -0.319    -13.81   ***   20/20
                                       |zflicker-1|        +0.001     +0.71    ns    8/20
                                       |zaniso-1|          +0.010     +4.26   ***    2/20
                                       lpips               -0.001     -1.01    ns   10/20

l1_organ_curriculum_s43                feature_l1_hu       -2.931     -5.34   ***   17/20
                                       org_mae             -0.000     -1.69    ns   12/20
                                       org_ssim            +0.000     +0.31    ns    9/20
                                       psnr                -0.016     -0.22    ns    6/20
                                       body_mae            -0.000     -1.10    ns   10/20
                                       |raps_hf-1|         +0.004     +2.46     *    5/20
                                       grad_w1             +0.000     +2.95    **    5/20
                                       org_grad_w1         +0.000     +1.57    ns    7/20
                                       |seam-1|            -0.317    -14.36   ***   20/20
                                       |zflicker-1|        +0.002     +0.91    ns   13/20
                                       |zaniso-1|          +0.006     +6.51   ***    1/20
                                       lpips               -0.000     -0.42    ns    8/20

l1_organ_curriculum_s44                feature_l1_hu       -3.055     -5.01   ***   17/20
                                       org_mae             -0.000     -1.02    ns   10/20
                                       org_ssim            +0.000     +1.00    ns    9/20
                                       psnr                +0.041     +1.61    ns    6/20
                                       body_mae            +0.000     +0.23    ns   10/20
                                       |raps_hf-1|         +0.010     +3.39    **    4/20
                                       grad_w1             +0.000     +5.15   ***    1/20
                                       org_grad_w1         +0.000     +3.04    **    6/20
                                       |seam-1|            -0.311    -13.86   ***   20/20
                                       |zflicker-1|        +0.001     +0.98    ns    8/20
                                       |zaniso-1|          +0.010     +8.69   ***    0/20
                                       lpips               +0.000     +0.54    ns    9/20

l1_organ_groupnorm_s42                 feature_l1_hu       -3.437     -5.13   ***   18/20
                                       org_mae             -0.001     -3.49    **   18/20
                                       org_ssim            -0.001     -3.12    **   20/20
                                       psnr                -0.188     -1.15    ns   13/20
                                       body_mae            -0.001     -2.32     *   19/20
                                       |raps_hf-1|         -0.003     -1.06    ns   12/20
                                       grad_w1             +0.000     +3.14    **    5/20
                                       org_grad_w1         +0.000     +1.21    ns    8/20
                                       |seam-1|            -0.308    -13.84   ***   20/20
                                       |zflicker-1|        +0.012     +2.37     *    8/20
                                       |zaniso-1|          +0.003     +0.75    ns   10/20
                                       lpips               +0.001     +0.76    ns    5/20

l1_organ_groupnorm_s43                 feature_l1_hu       -4.420     -5.32   ***   17/20
                                       org_mae             -0.001     -3.66    **   18/20
                                       org_ssim            -0.001     -2.87    **   20/20
                                       psnr                -0.280     -1.84    ns   13/20
                                       body_mae            -0.001     -2.52     *   18/20
                                       |raps_hf-1|         -0.005     -1.82    ns   15/20
                                       grad_w1             +0.000     +3.20    **    5/20
                                       org_grad_w1         +0.000     +2.48     *    4/20
                                       |seam-1|            -0.309    -13.73   ***   20/20
                                       |zflicker-1|        +0.016     +3.06    **    4/20
                                       |zaniso-1|          -0.003     -1.01    ns   11/20
                                       lpips               -0.000     -0.32    ns    8/20

l1_organ_groupnorm_s44                 feature_l1_hu       -3.719     -5.53   ***   17/20
                                       org_mae             -0.001     -3.21    **   19/20
                                       org_ssim            -0.001     -2.79     *   19/20
                                       psnr                -0.284     -1.86    ns   14/20
                                       body_mae            -0.001     -2.34     *   19/20
                                       |raps_hf-1|         -0.004     -1.43    ns   12/20
                                       grad_w1             +0.000     +3.10    **    6/20
                                       org_grad_w1         +0.000     +3.00    **    5/20
                                       |seam-1|            -0.310    -14.20   ***   20/20
                                       |zflicker-1|        +0.015     +3.01    **    2/20
                                       |zaniso-1|          -0.002     -0.74    ns   12/20
                                       lpips               -0.001     -0.59    ns    6/20

l1_organ_huprofile                     feature_l1_hu       -1.100     -1.42    ns   13/20
                                       org_mae             +0.006     +8.13   ***    0/20
                                       org_ssim            +0.005     +4.42   ***    0/20
                                       psnr                +1.104     +4.60   ***    1/20
                                       body_mae            +0.006     +5.99   ***    0/20
                                       |raps_hf-1|         +0.020     +4.03   ***    4/20
                                       grad_w1             -0.001     -6.27   ***   19/20
                                       org_grad_w1         -0.001    -12.86   ***   20/20
                                       |seam-1|            -0.328    -13.52   ***   20/20
                                       |zflicker-1|        -0.045     -3.77    **   17/20
                                       |zaniso-1|          +0.015     +3.26    **    5/20
                                       lpips               +0.006     +2.82     *    5/20

level_all8                             feature_l1_hu       -2.893     -2.64     *   14/20
                                       org_mae             +0.002     +4.12   ***    3/20
                                       org_ssim            +0.006     +7.66   ***    1/20
                                       psnr                +0.721     +4.14   ***    3/20
                                       body_mae            +0.002     +3.51    **    1/20
                                       |raps_hf-1|         -0.072     -4.37   ***   18/20
                                       grad_w1             -0.000     -3.55    **   18/20
                                       org_grad_w1         -0.001     -5.08   ***   19/20
                                       |seam-1|            -0.284    -14.17   ***   20/20
                                       |zflicker-1|        -0.038     -2.71     *   16/20
                                       |zaniso-1|          +0.028     +6.58   ***    2/20
                                       lpips               +0.005     +2.54     *    4/20

level_aorta                            feature_l1_hu       +8.780     +9.48   ***    0/20
                                       org_mae             +0.005    +12.78   ***    0/20
                                       org_ssim            +0.005     +7.47   ***    1/20
                                       psnr                +0.702     +4.27   ***    3/20
                                       body_mae            +0.005     +8.32   ***    1/20
                                       |raps_hf-1|         +0.012     +2.36     *    6/20
                                       grad_w1             -0.000     -2.96    **   14/20
                                       org_grad_w1         -0.001     -7.09   ***   18/20
                                       |seam-1|            -0.294    -14.20   ***   20/20
                                       |zflicker-1|        -0.014     -1.92    ns   16/20
                                       |zaniso-1|          +0.016     +3.94   ***    4/20
                                       lpips               +0.008     +3.47    **    4/20

level_aorta_pv                         feature_l1_hu       -2.513     -2.57     *   14/20
                                       org_mae             +0.002     +3.87    **    4/20
                                       org_ssim            +0.005     +7.18   ***    1/20
                                       psnr                +0.939     +4.12   ***    2/20
                                       body_mae            +0.002     +3.39    **    1/20
                                       |raps_hf-1|         -0.076     -4.50   ***   18/20
                                       grad_w1             -0.000     -1.46    ns   14/20
                                       org_grad_w1         -0.002     -6.12   ***   18/20
                                       |seam-1|            -0.270    -13.26   ***   20/20
                                       |zflicker-1|        -0.033     -2.74     *   16/20
                                       |zaniso-1|          +0.030     +3.22    **    5/20
                                       lpips               +0.006     +3.12    **    3/20

memorize97                             feature_l1_hu       -1.000     -2.60     *   15/20
                                       org_mae             -0.000     -0.97    ns   15/20
                                       org_ssim            -0.000     -2.82     *   14/20
                                       psnr                +0.101     +2.03    ns    7/20
                                       body_mae            +0.000     +1.36    ns    9/20
                                       |raps_hf-1|         -0.003     -1.67    ns   15/20
                                       grad_w1             +0.000     +1.01    ns    7/20
                                       org_grad_w1         +0.000     +4.88   ***    2/20
                                       |seam-1|            -0.313    -14.39   ***   20/20
                                       |zflicker-1|        -0.003     -0.91    ns   10/20
                                       |zaniso-1|          +0.005     +1.15    ns    9/20
                                       lpips               +0.001     +2.70     *    5/20

multiphase_film/arterial               — no cases in common with the baseline; not comparable

multiphase_film/venous                 feature_l1_hu       -3.061     -4.81   ***   17/20
                                       org_mae             -0.000     -0.12    ns   10/20
                                       org_ssim            +0.000     +1.79    ns    6/20
                                       psnr                +0.042     +0.69    ns    3/20
                                       body_mae            +0.000     +1.30    ns    7/20
                                       |raps_hf-1|         -0.004     -2.26     *   13/20
                                       grad_w1             +0.000     +1.07    ns    4/20
                                       org_grad_w1         -0.000     -2.18     *   13/20
                                       |seam-1|            -0.312    -14.49   ***   20/20
                                       |zflicker-1|        -0.003     -2.06    ns   16/20
                                       |zaniso-1|          +0.013     +8.76   ***    0/20
                                       lpips               -0.001     -1.22    ns   11/20

multiphase_film_adv/arterial           — no cases in common with the baseline; not comparable

multiphase_film_adv/venous             feature_l1_hu      +17.694    +12.90   ***    0/20
                                       org_mae             +0.015    +14.71   ***    0/20
                                       org_ssim            +0.012     +9.21   ***    0/20
                                       psnr                +1.869    +10.70   ***    0/20
                                       body_mae            +0.014    +14.98   ***    0/20
                                       |raps_hf-1|         +0.058     +8.97   ***    0/20
                                       grad_w1             +0.000     +5.50   ***    2/20
                                       org_grad_w1         -0.001     -5.18   ***   18/20
                                       |seam-1|            -0.278    -12.42   ***   20/20
                                       |zflicker-1|        -0.029     -3.66    **   17/20
                                       |zaniso-1|          +0.053     +9.91   ***    0/20
                                       lpips               +0.015    +10.50   ***    0/20

multiphase_film_adv_slices11/arterial  — no cases in common with the baseline; not comparable

multiphase_film_adv_slices11/venous    feature_l1_hu       +1.426     +1.35    ns    8/20
                                       org_mae             +0.004     +9.89   ***    1/20
                                       org_ssim            +0.005     +9.64   ***    1/20
                                       psnr                +0.868     +5.19   ***    2/20
                                       body_mae            +0.004     +6.47   ***    1/20
                                       |raps_hf-1|         -0.078     -4.54   ***   18/20
                                       grad_w1             -0.001     -9.44   ***   20/20
                                       org_grad_w1         -0.003    -20.40   ***   20/20
                                       |seam-1|            -0.278    -13.73   ***   20/20
                                       |zflicker-1|        +0.004     +0.38    ns   13/20
                                       |zaniso-1|          -0.049     -3.56    **   16/20
                                       lpips               +0.003     +1.51    ns    7/20

multiphase_uncond/arterial             — no cases in common with the baseline; not comparable

multiphase_uncond/venous               feature_l1_hu       +2.075     +2.16     *    7/20
                                       org_mae             +0.003     +7.73   ***    0/20
                                       org_ssim            +0.003     +6.85   ***    0/20
                                       psnr                +0.333     +3.22    **    2/20
                                       body_mae            +0.002     +8.25   ***    0/20
                                       |raps_hf-1|         -0.007     -3.70    **   16/20
                                       grad_w1             -0.000     -4.01   ***   18/20
                                       org_grad_w1         -0.001     -7.27   ***   19/20
                                       |seam-1|            -0.315    -13.68   ***   20/20
                                       |zflicker-1|        -0.026     -4.11   ***   18/20
                                       |zaniso-1|          +0.027     +7.85   ***    0/20
                                       lpips               -0.001     -1.08    ns   15/20

ncase10                                feature_l1_hu       +3.664     +4.41   ***    1/20
                                       org_mae             +0.004     +5.40   ***    0/20
                                       org_ssim            +0.003     +5.62   ***    0/20
                                       psnr                +0.548     +3.80    **    2/20
                                       body_mae            +0.003     +5.39   ***    0/20
                                       |raps_hf-1|         -0.052     -5.30   ***   18/20
                                       grad_w1             -0.000     -4.50   ***   19/20
                                       org_grad_w1         -0.001    -12.77   ***   20/20
                                       |seam-1|            -0.299    -12.86   ***   20/20
                                       |zflicker-1|        -0.028     -3.82    **   18/20
                                       |zaniso-1|          -0.004     -1.01    ns    6/20
                                       lpips               -0.002     -1.79    ns   12/20

ncase25                                feature_l1_hu       -0.538     -0.83    ns   11/20
                                       org_mae             +0.001     +6.51   ***    1/20
                                       org_ssim            +0.001     +5.11   ***    1/20
                                       psnr                +0.314     +2.55     *    3/20
                                       body_mae            +0.001     +6.43   ***    1/20
                                       |raps_hf-1|         -0.006     -2.61     *   12/20
                                       grad_w1             -0.000     -5.16   ***   19/20
                                       org_grad_w1         -0.000     -9.28   ***   19/20
                                       |seam-1|            -0.318    -13.56   ***   20/20
                                       |zflicker-1|        -0.018     -4.29   ***   18/20
                                       |zaniso-1|          +0.003     +0.80    ns    7/20
                                       lpips               +0.000     +0.21    ns   12/20

ncase50                                feature_l1_hu       -2.239     -3.06    **   14/20
                                       org_mae             +0.001     +3.18    **    2/20
                                       org_ssim            +0.001     +2.43     *    3/20
                                       psnr                +0.278     +3.05    **    2/20
                                       body_mae            +0.001     +3.22    **    3/20
                                       |raps_hf-1|         -0.013     -4.66   ***   17/20
                                       grad_w1             -0.000     -5.32   ***   19/20
                                       org_grad_w1         -0.000     -5.06   ***   18/20
                                       |seam-1|            -0.317    -14.07   ***   20/20
                                       |zflicker-1|        -0.018     -4.18   ***   18/20
                                       |zaniso-1|          +0.011     +2.82     *    2/20
                                       lpips               -0.001     -1.16    ns   14/20

pix2pixhd_baseline                     feature_l1_hu       -0.247     -0.38    ns    9/20
                                       org_mae             +0.002     +8.16   ***    0/20
                                       org_ssim            +0.003     +7.73   ***    0/20
                                       psnr                +0.376     +3.82    **    2/20
                                       body_mae            +0.002     +5.38   ***    1/20
                                       |raps_hf-1|         -0.046     -4.43   ***   17/20
                                       grad_w1             -0.001    -10.34   ***   20/20
                                       org_grad_w1         -0.004    -18.22   ***   20/20
                                       |seam-1|            -0.287    -13.96   ***   20/20
                                       |zflicker-1|        -0.051     -2.65     *   16/20
                                       |zaniso-1|          -0.015     -1.51    ns   16/20
                                       lpips               -0.009     -4.86   ***   17/20

slices11_k5                            feature_l1_hu       -4.171     -3.42    **   17/20
                                       org_mae             -0.002     -4.16   ***   18/20
                                       org_ssim            -0.002     -4.22   ***   20/20
                                       psnr                -0.348     -2.27     *   17/20
                                       body_mae            -0.002     -3.17    **   19/20
                                       |raps_hf-1|         -0.002     -0.64    ns   12/20
                                       grad_w1             +0.000     +1.84    ns    9/20
                                       org_grad_w1         +0.000     +1.13    ns   10/20
                                       |seam-1|            -0.303    -13.99   ***   20/20
                                       |zflicker-1|        +0.113    +12.65   ***    0/20
                                       |zaniso-1|          -0.045     -2.23     *   15/20
                                       lpips               -0.001     -1.11    ns   13/20

slices5_k2                             feature_l1_hu       -4.034     -5.39   ***   18/20
                                       org_mae             -0.002     -4.58   ***   19/20
                                       org_ssim            -0.002     -4.29   ***   20/20
                                       psnr                -0.351     -2.27     *   17/20
                                       body_mae            -0.002     -3.38    **   20/20
                                       |raps_hf-1|         -0.005     -1.77    ns   13/20
                                       grad_w1             +0.000     +2.27     *    7/20
                                       org_grad_w1         +0.000     +2.64     *    5/20
                                       |seam-1|            -0.310    -13.94   ***   20/20
                                       |zflicker-1|        +0.110    +12.34   ***    0/20
                                       |zaniso-1|          -0.047     -2.38     *   15/20
                                       lpips               -0.001     -1.32    ns   11/20

width32                                feature_l1_hu       -1.741     -3.55    **   17/20
                                       org_mae             +0.001     +4.96   ***    3/20
                                       org_ssim            +0.001     +3.21    **    2/20
                                       psnr                +0.096     +1.02    ns    4/20
                                       body_mae            +0.001     +3.76    **    1/20
                                       |raps_hf-1|         +0.003     +1.89    ns    7/20
                                       grad_w1             -0.000     -0.65    ns   14/20
                                       org_grad_w1         -0.000     -6.48   ***   18/20
                                       |seam-1|            -0.313    -14.41   ***   20/20
                                       |zflicker-1|        -0.014     -4.50   ***   17/20
                                       |zaniso-1|          +0.020    +14.44   ***    0/20
                                       lpips               -0.002     -2.45     *   15/20

width96                                feature_l1_hu       -3.373     -4.80   ***   18/20
                                       org_mae             -0.000     -1.10    ns   12/20
                                       org_ssim            -0.000     -1.46    ns   13/20
                                       psnr                -0.040     -0.68    ns    9/20
                                       body_mae            +0.000     +0.07    ns   10/20
                                       |raps_hf-1|         +0.009     +3.60    **    4/20
                                       grad_w1             +0.000     +0.10    ns   10/20
                                       org_grad_w1         +0.000     +2.06    ns    6/20
                                       |seam-1|            -0.316    -14.05   ***   20/20
                                       |zflicker-1|        +0.003     +2.00    ns    8/20
                                       |zaniso-1|          +0.003     +1.95    ns    7/20
                                       lpips               -0.001     -0.82    ns    8/20

  sign convention: negative delta = the model beats the baseline.
  sig: * p<.05  ** p<.01  *** p<.001 (paired, df~n-1)

_Table built from 43 model(s) accumulated in `analysis/benchmark_all/store` (43 scored in this run). Each model is scored once and cached; every cross-model quantity — ranking, paired deltas, level-recovery regressions — is recomputed from the merged per-case rows on every run._