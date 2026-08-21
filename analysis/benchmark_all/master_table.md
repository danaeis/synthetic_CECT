# NCCT→CECT benchmark — master table

All metrics on the shared HU[-200,400]→[0,1] domain, same test cases. Metrics are split into categories (one table each); models are grouped by generator architecture. In every column **bold = best**, _italic = second best_, ranked in that metric's own direction (higher-better, lower-better, or closest-to-1.0 for the ratio metrics). Reference/floor rows are shown for scale but excluded from ranking.


### Image-level (global pixel)

| model | n | PSNR | SSIM | MAE | MSE | PCC |
|---|---|---|---|---|---|---|
| **UNet + PatchGAN (this repo)** |  |  |  |  |  |  |
| b0_groupnorm_adv | 20 | 29.91 | 0.9390 | 0.0107 | 0.00110 | 0.9860 |
| l1_adv | 20 | 30.06 | 0.9413 | 0.0099 | 0.00108 | 0.9863 |
| l1_adv_organ | 20 | 30.01 | 0.9425 | 0.0098 | 0.00109 | 0.9862 |
| l1_adv_organ_s42 | 20 | 30.07 | 0.9441 | 0.0096 | 0.00107 | 0.9865 |
| l1_adv_organ_s43 | 20 | 30.13 | 0.9440 | 0.0094 | 0.00106 | 0.9864 |
| l1_adv_organ_s44 | 20 | 30.05 | 0.9440 | 0.0096 | 0.00107 | 0.9864 |
| l1_bowel_zero | 20 | 30.46 | 0.9439 | 0.0094 | 0.00098 | 0.9874 |
| l1_huprofile_only | 20 | 30.31 | 0.9390 | 0.0098 | 0.00102 | 0.9870 |
| l1_only | 20 | 30.38 | 0.9413 | 0.0096 | 0.00100 | 0.9871 |
| l1_organ_curriculum | 20 | 30.41 | 0.9436 | 0.0095 | 0.00099 | 0.9873 |
| l1_organ_curriculum_s42 | 20 | 30.38 | 0.9422 | 0.0095 | 0.00100 | 0.9872 |
| l1_organ_curriculum_s43 | 20 | 30.40 | 0.9372 | 0.0096 | 0.00100 | 0.9872 |
| l1_organ_curriculum_s44 | 20 | 30.34 | 0.9409 | 0.0097 | 0.00101 | 0.9871 |
| l1_organ_groupnorm_s42 | 20 | 30.57 | 0.9486 | 0.0088 | 0.00097 | 0.9876 |
| l1_organ_groupnorm_s43 | 20 | 30.66 | 0.9492 | 0.0087 | 0.00095 | 0.9879 |
| l1_organ_groupnorm_s44 | 20 | 30.67 | 0.9493 | 0.0087 | 0.00095 | 0.9879 |
| l1_organ_huprofile | 20 | 29.28 | 0.8169 | 0.0146 | 0.00129 | 0.9842 |
| level_all8 | 20 | 29.66 | 0.9397 | 0.0097 | 0.00115 | 0.9852 |
| level_aorta | 20 | 29.68 | 0.9420 | 0.0107 | 0.00115 | 0.9856 |
| level_aorta_pv | 20 | 29.44 | 0.9378 | 0.0099 | 0.00125 | 0.9842 |
| memorize97 | 20 | 30.28 | 0.9411 | 0.0098 | 0.00103 | 0.9869 |
| multiphase_film/arterial | 20 | 29.89 | 0.9427 | 0.0099 | 0.00108 | 0.9862 |
| multiphase_film/venous | 20 | 30.34 | 0.9436 | 0.0095 | 0.00101 | 0.9870 |
| multiphase_film_adv/arterial | 20 | 28.18 | 0.9093 | 0.0140 | 0.00159 | 0.9800 |
| multiphase_film_adv/venous | 20 | 28.51 | 0.9105 | 0.0147 | 0.00149 | 0.9820 |
| multiphase_film_adv_slices11/arterial | 20 | 28.85 | 0.9354 | 0.0108 | 0.00134 | 0.9829 |
| multiphase_film_adv_slices11/venous | 20 | 29.51 | 0.9373 | 0.0103 | 0.00119 | 0.9848 |
| multiphase_uncond/arterial | 20 | 29.50 | 0.9333 | 0.0111 | 0.00118 | 0.9850 |
| multiphase_uncond/venous | 20 | 30.05 | 0.9344 | 0.0104 | 0.00108 | 0.9862 |
| ncase10 | 20 | 29.83 | 0.9363 | 0.0108 | 0.00113 | 0.9859 |
| ncase25 | 20 | 30.07 | 0.9357 | 0.0103 | 0.00108 | 0.9863 |
| ncase50 | 20 | 30.10 | 0.9374 | 0.0101 | 0.00107 | 0.9864 |
| slices11_k5 | 20 | 30.73 | _0.9501_ | _0.0086_ | _0.00093_ | _0.9881_ |
| slices5_k2 | 20 | _30.73_ | **0.9502** | **0.0086** | 0.00093 | 0.9881 |
| width32 | 20 | 30.28 | 0.9331 | 0.0100 | 0.00103 | 0.9868 |
| width96 | 20 | 30.42 | 0.9455 | 0.0094 | 0.00099 | 0.9874 |
| **Diffusion** |  |  |  |  |  |  |
| diff_hetero_nll | 20 | **30.82** | 0.9489 | 0.0088 | **0.00091** | **0.9883** |
| diff_l1_organ_groupnorm | 20 | 30.57 | 0.9486 | 0.0088 | 0.00097 | 0.9876 |
| diff_v | 20 | 28.79 | 0.9088 | 0.0124 | 0.00139 | 0.9822 |
| diff_v_nocfg | 20 | 28.88 | 0.9198 | 0.0118 | 0.00137 | 0.9825 |
| diff_v_organ | 20 | 28.40 | 0.9268 | 0.0125 | 0.00151 | 0.9808 |
| diff_x0 | 20 | 28.74 | 0.9339 | 0.0113 | 0.00141 | 0.9820 |
| **External baseline** |  |  |  |  |  |  |
| pix2pixhd_baseline | 20 | 30.01 | 0.9409 | 0.0100 | 0.00109 | 0.9861 |

### Organ-level (region-restricted)

| model | oPSNR | oSSIM | oMAE | bPSNR | bMAE | featHU |
|---|---|---|---|---|---|---|
| **UNet + PatchGAN (this repo)** |  |  |  |  |  |  |
| b0_groupnorm_adv | 23.83 | 0.9633 | 0.0351 | 25.08 | 0.0315 | 26.76 |
| l1_adv | 24.26 | 0.9660 | 0.0319 | 25.35 | 0.0299 | 17.85 |
| l1_adv_organ | 24.22 | 0.9659 | 0.0317 | 25.29 | 0.0300 | 16.04 |
| l1_adv_organ_s42 | 24.30 | 0.9667 | 0.0311 | 25.42 | 0.0291 | 14.57 |
| l1_adv_organ_s43 | 24.26 | 0.9663 | 0.0307 | 25.35 | 0.0289 | 13.90 |
| l1_adv_organ_s44 | 24.20 | 0.9661 | 0.0314 | 25.32 | 0.0294 | 13.62 |
| l1_bowel_zero | 24.58 | 0.9684 | 0.0299 | 25.72 | 0.0279 | 17.39 |
| l1_huprofile_only | 24.46 | 0.9680 | 0.0307 | 25.58 | 0.0286 | 18.31 |
| l1_only | 24.59 | 0.9684 | 0.0299 | 25.70 | 0.0280 | 17.32 |
| l1_organ_curriculum | 24.58 | 0.9684 | 0.0299 | 25.70 | 0.0280 | 15.74 |
| l1_organ_curriculum_s42 | 24.58 | 0.9684 | 0.0299 | 25.70 | 0.0280 | 14.85 |
| l1_organ_curriculum_s43 | 24.57 | 0.9683 | 0.0297 | 25.68 | 0.0279 | 14.39 |
| l1_organ_curriculum_s44 | 24.57 | 0.9682 | 0.0298 | 25.68 | 0.0280 | 14.26 |
| l1_organ_groupnorm_s42 | 24.77 | 0.9698 | 0.0288 | 25.93 | 0.0268 | 13.88 |
| l1_organ_groupnorm_s43 | 24.80 | 0.9698 | 0.0286 | 25.96 | 0.0267 | **12.90** |
| l1_organ_groupnorm_s44 | 24.77 | 0.9698 | 0.0288 | 25.95 | 0.0268 | 13.60 |
| l1_organ_huprofile | 23.77 | 0.9629 | 0.0359 | 24.66 | 0.0343 | 16.22 |
| level_all8 | 23.73 | 0.9625 | 0.0320 | 24.85 | 0.0299 | 14.43 |
| level_aorta | 23.79 | 0.9632 | 0.0354 | 24.88 | 0.0326 | 26.10 |
| level_aorta_pv | 23.89 | 0.9635 | 0.0316 | 24.94 | 0.0299 | 14.81 |
| memorize97 | 24.61 | 0.9687 | 0.0298 | 25.68 | 0.0282 | 16.32 |
| multiphase_film/arterial | 23.76 | 0.9646 | 0.0330 | 25.12 | 0.0292 | 36.73 |
| multiphase_film/venous | 24.52 | 0.9680 | 0.0299 | 25.62 | 0.0282 | 14.26 |
| multiphase_film_adv/arterial | 21.90 | 0.9463 | 0.0447 | 23.37 | 0.0398 | 63.57 |
| multiphase_film_adv/venous | 22.76 | 0.9561 | 0.0448 | 23.75 | 0.0422 | 35.01 |
| multiphase_film_adv_slices11/arterial | 22.64 | 0.9555 | 0.0381 | 24.01 | 0.0334 | 42.47 |
| multiphase_film_adv_slices11/venous | 23.71 | 0.9629 | 0.0338 | 24.73 | 0.0318 | 18.75 |
| multiphase_uncond/arterial | 23.14 | 0.9589 | 0.0374 | 24.70 | 0.0318 | 50.34 |
| multiphase_uncond/venous | 24.10 | 0.9651 | 0.0332 | 25.31 | 0.0300 | 19.39 |
| ncase10 | 24.02 | 0.9649 | 0.0335 | 25.06 | 0.0314 | 20.98 |
| ncase25 | 24.37 | 0.9673 | 0.0313 | 25.49 | 0.0292 | 16.78 |
| ncase50 | 24.40 | 0.9674 | 0.0308 | 25.51 | 0.0288 | 15.08 |
| slices11_k5 | **24.89** | _0.9705_ | _0.0284_ | _26.06_ | _0.0264_ | _13.15_ |
| slices5_k2 | _24.89_ | **0.9705** | **0.0283** | **26.06** | **0.0263** | 13.29 |
| width32 | 24.44 | 0.9673 | 0.0306 | 25.52 | 0.0288 | 15.58 |
| width96 | 24.61 | 0.9687 | 0.0297 | 25.69 | 0.0280 | 13.95 |
| **Diffusion** |  |  |  |  |  |  |
| diff_hetero_nll | 24.89 | 0.9702 | 0.0298 | 26.02 | 0.0274 | 14.46 |
| diff_l1_organ_groupnorm | 24.77 | 0.9698 | 0.0288 | 25.93 | 0.0268 | 13.88 |
| diff_v | 22.82 | 0.9544 | 0.0390 | 24.03 | 0.0350 | 22.25 |
| diff_v_nocfg | 22.99 | 0.9560 | 0.0371 | 24.15 | 0.0338 | 18.36 |
| diff_v_organ | 22.68 | 0.9531 | 0.0410 | 23.64 | 0.0378 | 14.60 |
| diff_x0 | 22.71 | 0.9535 | 0.0392 | 23.95 | 0.0349 | 24.76 |
| **External baseline** |  |  |  |  |  |  |
| pix2pixhd_baseline | 24.23 | 0.9658 | 0.0319 | 25.30 | 0.0302 | 17.07 |

### Phase & level fidelity

| model | phase | prob | βlev | varR |
|---|---|---|---|---|
| **UNet + PatchGAN (this repo)** |  |  |  |  |
| b0_groupnorm_adv | **1.00** | 0.9636 | 0.11 | 0.18 |
| l1_adv | **1.00** | 0.9751 | 0.17 | 0.13 |
| l1_adv_organ | **1.00** | 0.9794 | 0.21 | 0.16 |
| l1_adv_organ_s42 | **1.00** | 0.9892 | 0.20 | 0.16 |
| l1_adv_organ_s43 | **1.00** | 0.9927 | 0.19 | 0.14 |
| l1_adv_organ_s44 | **1.00** | 0.9897 | 0.24 | 0.21 |
| l1_bowel_zero | _0.95_ | 0.9517 | 0.21 | 0.14 |
| l1_huprofile_only | **1.00** | 0.9768 | 0.23 | 0.19 |
| l1_only | _0.95_ | 0.9484 | 0.21 | 0.15 |
| l1_organ_curriculum | **1.00** | 0.9792 | 0.23 | 0.15 |
| l1_organ_curriculum_s42 | **1.00** | 0.9843 | 0.20 | 0.16 |
| l1_organ_curriculum_s43 | **1.00** | 0.9869 | 0.22 | 0.15 |
| l1_organ_curriculum_s44 | **1.00** | 0.9889 | 0.20 | 0.17 |
| l1_organ_groupnorm_s42 | **1.00** | 0.9879 | 0.26 | 0.24 |
| l1_organ_groupnorm_s43 | **1.00** | 0.9894 | 0.25 | 0.24 |
| l1_organ_groupnorm_s44 | **1.00** | 0.9705 | 0.25 | 0.20 |
| l1_organ_huprofile | **1.00** | _0.9960_ | 0.06 | 0.12 |
| level_all8 | **1.00** | 0.9840 | **0.68** | **0.64** |
| level_aorta | **1.00** | 0.9781 | 0.12 | 0.15 |
| level_aorta_pv | **1.00** | 0.9812 | _0.53_ | _0.42_ |
| memorize97 | **1.00** | 0.9869 | 0.26 | 0.20 |
| multiphase_film/arterial | **1.00** | **0.9962** | 0.18 | 0.14 |
| multiphase_film/venous | **1.00** | 0.9861 | 0.22 | 0.18 |
| multiphase_film_adv/arterial | **1.00** | 0.9075 | 0.05 | 0.09 |
| multiphase_film_adv/venous | 0.85 | 0.6358 | 0.07 | 0.23 |
| multiphase_film_adv_slices11/arterial | **1.00** | 0.9884 | 0.17 | 0.12 |
| multiphase_film_adv_slices11/venous | **1.00** | 0.9899 | 0.16 | 0.28 |
| multiphase_uncond/arterial | 0.25 | 0.3366 | 0.15 | 0.16 |
| multiphase_uncond/venous | 0.70 | 0.6021 | 0.27 | 0.35 |
| ncase10 | **1.00** | 0.9740 | 0.16 | 0.17 |
| ncase25 | **1.00** | 0.9879 | 0.22 | 0.18 |
| ncase50 | **1.00** | 0.9918 | 0.23 | 0.20 |
| slices11_k5 | **1.00** | 0.9935 | 0.23 | 0.21 |
| slices5_k2 | **1.00** | 0.9684 | 0.23 | 0.22 |
| width32 | **1.00** | 0.9870 | 0.18 | 0.16 |
| width96 | **1.00** | 0.9890 | 0.26 | 0.20 |
| **Diffusion** |  |  |  |  |
| diff_hetero_nll | **1.00** | 0.9909 | 0.33 | 0.27 |
| diff_l1_organ_groupnorm | **1.00** | 0.9879 | 0.26 | 0.24 |
| diff_v | **1.00** | 0.9641 | 0.09 | 0.14 |
| diff_v_nocfg | **1.00** | 0.9575 | 0.19 | 0.18 |
| diff_v_organ | **1.00** | 0.9932 | 0.11 | 0.14 |
| diff_x0 | **1.00** | 0.9500 | 0.08 | 0.21 |
| **External baseline** |  |  |  |  |
| pix2pixhd_baseline | _0.95_ | 0.9141 | 0.15 | 0.18 |

### Detail-focused (texture & consistency)

| model | RAPS | gradW1 | oGradW1 | seam | zflick | zaniso |
|---|---|---|---|---|---|---|
| **UNet + PatchGAN (this repo)** |  |  |  |  |  |  |
| b0_groupnorm_adv | 0.801 | 0.0017 | 0.0060 | 1.371 | 0.927 | 1.105 |
| l1_adv | 0.896 | _0.0009_ | 0.0025 | 1.632 | _0.998_ | 1.076 |
| l1_adv_organ | 0.905 | 0.0009 | 0.0026 | 1.641 | **0.998** | 1.084 |
| l1_adv_organ_s42 | 0.897 | 0.0011 | 0.0027 | 1.389 | 0.996 | 1.098 |
| l1_adv_organ_s43 | **0.951** | 0.0009 | 0.0030 | 1.381 | 1.004 | 1.085 |
| l1_adv_organ_s44 | 0.903 | 0.0009 | 0.0029 | 1.379 | 1.007 | 1.093 |
| l1_bowel_zero | 0.836 | 0.0021 | 0.0065 | 1.655 | 0.901 | 1.095 |
| l1_huprofile_only | 0.826 | 0.0019 | 0.0060 | 1.350 | 0.923 | 1.111 |
| l1_only | 0.837 | 0.0020 | 0.0064 | 1.672 | 0.909 | 1.096 |
| l1_organ_curriculum | 0.840 | 0.0019 | 0.0062 | 1.663 | 0.911 | 1.098 |
| l1_organ_curriculum_s42 | 0.822 | 0.0021 | 0.0067 | 1.341 | 0.908 | 1.107 |
| l1_organ_curriculum_s43 | 0.831 | 0.0020 | 0.0065 | 1.346 | 0.908 | 1.103 |
| l1_organ_curriculum_s44 | 0.824 | 0.0021 | 0.0066 | 1.351 | 0.907 | 1.106 |
| l1_organ_groupnorm_s42 | 0.841 | 0.0021 | 0.0065 | 1.356 | 0.897 | 1.099 |
| l1_organ_groupnorm_s43 | 0.842 | 0.0021 | 0.0066 | 1.353 | 0.893 | 1.093 |
| l1_organ_groupnorm_s44 | 0.841 | 0.0021 | 0.0067 | 1.354 | 0.893 | 1.095 |
| l1_organ_huprofile | 0.811 | 0.0014 | 0.0053 | **1.332** | 0.983 | 1.111 |
| level_all8 | 0.935 | 0.0015 | 0.0051 | 1.384 | 0.976 | 1.125 |
| level_aorta | 0.820 | 0.0017 | 0.0056 | 1.370 | 0.933 | 1.113 |
| level_aorta_pv | 0.939 | 0.0018 | 0.0048 | 1.402 | 0.964 | 1.127 |
| memorize97 | 0.839 | 0.0020 | 0.0066 | 1.349 | 0.912 | 1.102 |
| multiphase_film/arterial | 0.816 | 0.0022 | 0.0072 | 1.368 | 0.899 | 1.102 |
| multiphase_film/venous | 0.841 | 0.0020 | 0.0063 | 1.351 | 0.914 | 1.109 |
| multiphase_film_adv/arterial | 0.746 | 0.0025 | 0.0070 | 1.404 | 0.930 | 1.135 |
| multiphase_film_adv/venous | 0.773 | 0.0023 | 0.0056 | 1.395 | 0.953 | 1.150 |
| multiphase_film_adv_slices11/arterial | 0.923 | 0.0013 | 0.0042 | 1.431 | 0.887 | _0.992_ |
| multiphase_film_adv_slices11/venous | _0.945_ | 0.0012 | 0.0039 | 1.393 | 0.902 | **1.004** |
| multiphase_uncond/arterial | 0.810 | 0.0021 | 0.0076 | 1.346 | 0.916 | 1.112 |
| multiphase_uncond/venous | 0.846 | 0.0018 | 0.0059 | 1.346 | 0.945 | 1.123 |
| ncase10 | 0.902 | 0.0015 | 0.0054 | 1.352 | 0.952 | 1.091 |
| ncase25 | 0.841 | 0.0017 | 0.0060 | _1.339_ | 0.931 | 1.099 |
| ncase50 | 0.853 | 0.0018 | 0.0061 | 1.344 | 0.932 | 1.107 |
| slices11_k5 | 0.840 | 0.0021 | 0.0065 | 1.361 | 0.790 | 0.962 |
| slices5_k2 | 0.844 | 0.0021 | 0.0067 | 1.354 | 0.792 | 0.967 |
| width32 | 0.831 | 0.0019 | 0.0061 | 1.350 | 0.927 | 1.116 |
| width96 | 0.826 | 0.0020 | 0.0065 | 1.345 | 0.906 | 1.099 |
| **Diffusion** |  |  |  |  |  |  |
| diff_hetero_nll | 0.800 | 0.0022 | 0.0069 | 1.356 | 0.893 | 1.103 |
| diff_l1_organ_groupnorm | 0.841 | 0.0021 | 0.0065 | 1.356 | 0.897 | 1.099 |
| diff_v | 0.886 | 0.0012 | 0.0034 | 1.367 | 1.857 | 2.045 |
| diff_v_nocfg | 0.890 | 0.0013 | 0.0034 | 1.367 | 1.751 | 1.932 |
| diff_v_organ | 0.867 | 0.0009 | _0.0024_ | 1.396 | 1.886 | 2.031 |
| diff_x0 | 0.848 | 0.0016 | 0.0046 | 1.365 | 1.586 | 1.852 |
| **External baseline** |  |  |  |  |  |  |
| pix2pixhd_baseline | 0.895 | **0.0007** | **0.0020** | 1.385 | 1.021 | 1.080 |

### Perceptual (literature comparability)

| model | LPIPS | FID |
|---|---|---|
| **UNet + PatchGAN (this repo)** |  |  |
| b0_groupnorm_adv | 0.0537 | 16.4 |
| l1_adv | 0.0388 | 7.1 |
| l1_adv_organ | 0.0384 | **6.5** |
| l1_adv_organ_s42 | _0.0378_ | 7.1 |
| l1_adv_organ_s43 | **0.0371** | 8.2 |
| l1_adv_organ_s44 | 0.0400 | 7.3 |
| l1_bowel_zero | 0.0463 | 17.9 |
| l1_huprofile_only | 0.0466 | 16.9 |
| l1_only | 0.0470 | 17.9 |
| l1_organ_curriculum | 0.0470 | 17.2 |
| l1_organ_curriculum_s42 | 0.0464 | 18.9 |
| l1_organ_curriculum_s43 | 0.0466 | 18.3 |
| l1_organ_curriculum_s44 | 0.0473 | 18.8 |
| l1_organ_groupnorm_s42 | 0.0478 | 20.5 |
| l1_organ_groupnorm_s43 | 0.0467 | 20.5 |
| l1_organ_groupnorm_s44 | 0.0464 | 21.1 |
| l1_organ_huprofile | 0.0531 | 17.4 |
| level_all8 | 0.0518 | 19.7 |
| level_aorta | 0.0551 | 16.5 |
| level_aorta_pv | 0.0528 | 21.5 |
| memorize97 | 0.0484 | 19.2 |
| multiphase_film/arterial | 0.0503 | 18.7 |
| multiphase_film/venous | 0.0462 | 18.5 |
| multiphase_film_adv/arterial | 0.0675 | 19.3 |
| multiphase_film_adv/venous | 0.0620 | 17.7 |
| multiphase_film_adv_slices11/arterial | 0.0538 | 15.0 |
| multiphase_film_adv_slices11/venous | 0.0500 | 14.4 |
| multiphase_uncond/arterial | 0.0518 | 19.1 |
| multiphase_uncond/venous | 0.0461 | 17.5 |
| ncase10 | 0.0452 | 15.4 |
| ncase25 | 0.0472 | 16.7 |
| ncase50 | 0.0463 | 17.5 |
| slices11_k5 | 0.0459 | 19.8 |
| slices5_k2 | 0.0457 | 20.3 |
| width32 | 0.0451 | 16.0 |
| width96 | 0.0465 | 19.2 |
| **Diffusion** |  |  |
| diff_hetero_nll | 0.0482 | 18.3 |
| diff_l1_organ_groupnorm | 0.0478 | 20.5 |
| diff_v | 0.0512 | 9.7 |
| diff_v_nocfg | 0.0502 | 9.4 |
| diff_v_organ | 0.0594 | 19.2 |
| diff_x0 | 0.0549 | 14.4 |
| **External baseline** |  |  |
| pix2pixhd_baseline | 0.0382 | _6.5_ |

**How to read these tables.** The *Image-level (global pixel)* category is SECONDARY and flat by construction: `to_unit` saturates air/lung/fat→0 and bone→1 identically in every model, so those columns average over a large error-free mass and an identity copy of the NCCT already scores most of the way to the best model on them (see metrics.py:body_mask). Read the PRIMARY categories instead — organ-level (oMAE/featHU), phase & level fidelity (phase/prob/βlev/varR) and detail-focused texture (RAPS/gradW1).

oPSNR/oSSIM/oMAE = organ-region. featHU = mean per-organ |HU error|. Higher PSNR/SSIM/PCC/prob/phase better; lower MAE/MSE/featHU better.

**Level recovery** (βlev, varR): generated per-organ median HU regressed on the real one across cases, averaged over aorta, portal_vein_and_splenic_vein, inferior_vena_cava, liver. **βlev** = mean slope and **varR** = mean var(gen)/var(real). Both target **1.0**: βlev/varR → 1 means the model tracks each case's true contrast level; βlev/varR → 0 means it emits the population average and is indistinguishable from a conditional-mean predictor — the textbook signature of an L1/L2 loss under irreducible enhancement uncertainty (dose/bolus timing are not visible in NCCT). featHU can look decent while varR is near 0, so these two columns are what separate a real generator from an averager (full breakdown: scripts/audit_enhancement.py). βlev/varR are NaN when generated volumes are not in HU (--gen_not_hu) or an organ map is unavailable.

**Texture and consistency** (`metrics.py`): RAPS = high-frequency spectral energy vs real; gradW1/oGradW1 = W1 distance between gradient-magnitude distributions, global and organ-region; seam = tile-boundary over interior gradient; zflick = inter-slice difference vs real. **RAPS, seam and zflick are ratios: 1.000 is the target and both directions are failures** — rank them by |value - 1|, never as "higher is better". RAPS < 1 is blur, > 1 is noise or hallucinated texture. seam > 1 is a visible tile boundary; zflick > 1 is slice-to-slice flicker. gradW1 is a distance: lower is better. seam is NaN for models with no known tiling geometry.

**LPIPS / FID** (`perceptual.py`) — reported for comparability with the published NCCT→CECT literature, **not** as primary evidence. Both run ImageNet-pretrained networks on grayscale CT, so their absolute values have no physical meaning here and their ordering is not validated for this domain; the CT-native RAPS/gradW1 columns are what the texture claims rest on. LPIPS (alex) is paired and per-case, lower is better, computed at native in-plane resolution. FID is distributional: one value per model over ~5459 pooled body-containing axial slices from n=20 volumes, so it has no per-case value and no paired test. **FID is biased upward at small sample size** — these values are comparable between the rows of this table (identical slice counts, identical real set) and to nothing else. Backend: torchvision (ImageNet-1k InceptionV3) — NOT the canonical FID weights.

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