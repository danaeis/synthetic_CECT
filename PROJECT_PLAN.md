# NCCT→CECT synthesis — master plan

State as of 2026-07-20, after the five-scenario ablation.
Companion docs: `IMPLEMENTATION.md` (what the code does),
`phase_conditioning_plan.md` (literature + multi-phase design).

**Regenerate every number below with:**
```bash
python analyze_runs.py --runs_dir ../out_synthesis_train --out analysis/
```

---

## 1. Results

### 1.1 The headline

| run | PSNR | org-SSIM | phase acc | gen prob | **feature L1 (HU)** |
|---|---|---|---|---|---|
| l1_only | 29.75 | 0.9375 | 0.95 | 0.9484 | 17.32 |
| l1_adv | 29.33 | 0.9336 | 1.00 | 0.9751 | 17.85 |
| l1_bowel_zero | 29.75 | 0.9383 | 0.95 | 0.9517 | 17.39 |
| **l1_organ_curriculum** | 29.74 | 0.9385 | **1.00** | **0.9792** | **15.74** |
| l1_adv_organ | 29.24 | 0.9340 | 1.00 | 0.9794 | 16.04 |

Paired per-case tests vs `l1_only` (n=20, negative = better):

| | feature L1 | aorta | portal vein |
|---|---|---|---|
| bowel_zero (control) | +0.07 (ns) | +1.67 (ns) | −2.62 ** |
| **organ_curriculum** | **−1.58 \*\*\*** 15/20 | −2.52 (t=−2.05) 13/20 | **−3.84 \*\*\*** 16/20 |
| adv_organ | −1.28 ** 14/20 | −1.49 * 14/20 | −4.34 *** 19/20 |
| adv | +0.53 (ns) | +1.52 (ns) | +1.31 (ns) |

**Three findings, all thesis-relevant:**

1. **The tiered vessel up-weighting is the active ingredient.** The `gi_zero`
   control produced no overall gain (t=0.23). This refutes the pre-registered
   hypothesis that bowel exclusion alone would capture most of the effect — it
   captured none. The tiered vector earns its complexity. (GI-zeroing did help
   two organs locally: portal vein −2.62 ** and colon −1.43 **.)
2. **The gain is invisible to PSNR/SSIM.** Between-run differences sit in the 4th
   decimal, inside a single run's own epoch-to-epoch noise (org-SSIM std 0.0009–
   0.0011 for the L1 runs). Reporting pixel metrics alone would have concluded
   "no effect" from a result significant at t=−4.22.
3. **Adversarial and organ weighting improve different things.** Adversarial
   raises phase *classification* confidence (0.948→0.975) but not HU accuracy
   (t=+1.37, ns — directionally worse). Organ weighting improves both.
   `adv_organ` keeps both, so they compose.

Caveat kept attached: phase accuracy 0.95 vs 1.00 is **one case** of 20. The
continuous metrics (feature L1, per-organ HU) carry the real evidence.

### 1.2 The residual error is data-limited, not model-limited

Per-case `feature_l1_hu` correlates **r = 0.957–0.983** across all five models
(mean r = 0.970, r² = 0.94). Between-case spread 6.2 HU std; between-model spread
on a fixed case 3.2 HU.

~94% of per-case error variance is shared by every model — intrinsic to the case,
not the loss. All five fail on the same cases. **Loss engineering is near its
ceiling.** Real CECT scores 0.9937 target-prob vs 0.9792 for organ_curriculum, so
the phase-classification gap is nearly closed too.

`audit_data_ceiling.py` tests the leading explanation (see §3).

### 1.3 The context gap is in Z, not in-plane

Organ extents on a real `_seg_full` mask at 1.5 mm isotropic:

| structure | X | Y | **Z** |
|---|---|---|---|
| aorta | 36 mm | 81 mm | **258 mm** |
| portal vein | 144 mm | 63 mm | **90 mm** |
| liver | 228 mm | 171 mm | 126 mm |
| heart | 111 mm | 90 mm | 63 mm |

A 128 px patch already covers **192 mm** in-plane — enough for the heart, the
vessels' cross-sections, and most of the liver. But `patch_depth=1` gives the
model **1.5 mm** of Z. The phase-critical vessels are exactly the Z-elongated
structures, and exactly the ones with the worst HU errors (portal vein 36 HU,
pulmonary vein 32 HU). **A Z-running tube is being predicted from one axial
slice.** Bigger *in-plane* patches would buy little; Z context is the gap.

### 1.4 Two schedule problems

**`lambda_l1` was constant 25 in `l1_adv_organ` — the curriculum did nothing.**
`use_adversarial` pinned the start to `lambda_l1_reduced` (25), which equalled the
floor, so the decay ran 25→25 for all 65 epochs. Fixed (§2). In the
non-adversarial runs the decay worked correctly (100→25 over epochs 10–30, then
floor for ~59% of the run — which is also why it reads as flat if you look at the
tail). Its isolated effect remains unmeasured, and at λ_organ=5 the organ term
still only reached 21% of the L1 term.

**The LR warm restarts hurt, and hurt the adversarial runs most.** With
T0=15/tmult=2 the restarts land at epochs 15 and 45/46, and every one cost
org-SSIM:

| run | ep15 restart | ep45 restart |
|---|---|---|
| l1_only | −0.0031 | −0.0005 |
| l1_organ_curriculum | −0.0026 | −0.0024 |
| **l1_adv** | −0.0056 | **−0.0194** |
| **l1_adv_organ** | −0.0047 | **−0.0136** |

Best values always occurred at the *low-LR end* of a cycle, never after a
restart. Adversarial runs are also ~10× noisier epoch-to-epoch (org-SSIM std
0.011 vs 0.0011) — the instability is quantified, not just visible.

Every run reached within 0.001 of its peak org-SSIM by epoch 24–28 and then added
nothing for 31–45 further epochs.

---

## 2. What changed in the code

| file | change |
|---|---|
| `losses.py` | Decay now starts from full `lambda_l1`, ignoring `lambda_l1_reduced` — applying both was double-counting and produced the 25→25 no-op. Hard warning when start == floor. |
| `losses.py` | New `OrganHUProfileLoss` (§3). |
| `config.py` | `COSINE_T0` 15→45 (≥ EPOCHS ⇒ single anneal, no restarts); `EPOCHS` 80→45 (nothing improves past ~epoch 28). |
| `config.py` | `LAMBDA_ORGAN` 5→20. At λ=5 the organ term peaked at 21% of L1 — a minority of the gradient throughout, yet still significant; give it room to matter. Main knob to sweep. |
| `analyze_runs.py` | **New.** Reproduces every number in §1, with significance tests and an explicit noise floor. |
| `audit_data_ceiling.py` | **New.** Tests whether the r=0.97 pattern is registration quality. |
| `smoke_test_organ_weights.py` | Regression test for the 25→25 no-op (it reproduces the exact flat `[25.0×5]` trace and fails without the fix), plus 6 HU-profile tests. 26 checks total. |

### `OrganHUProfileLoss` — the new lever

The XGBoost phase model reads per-organ **median HU** and nothing else. Contrast
phase *is* the absolute enhancement level of each organ. Per-voxel losses optimise
that only indirectly, and the intervention that most improved HU error did so as a
side effect. This targets it directly: per-organ mean of pred vs target, L1
between the two means, weighted by the same `ORGAN_WEIGHTS` LUT, zero-weight
organs skipped, organs under 16 voxels skipped.

It constrains **level, not appearance** — a patch can score 0 while looking
nothing like the target, so it must never be the only spatial term. Verified: a
texture-scrambled patch with matching means scores 3e-08; a 0.10 offset scores
exactly 0.10; a 0.50 offset in zero-weighted bowel scores 0.

`LAMBDA_HU_PROFILE = 50`, sized from the measured residual (~15 HU on a 600 HU
window ≈ 0.025 normalised) so the term lands at ~20% of the organ term. Deliberate:
at λ=10 it would have been under 4% of the gradient, which is exactly the mistake
that made λ_organ=5 ineffective.

---

## 3. Next steps, in order

**1. Re-run the ladder on the fixed schedule** (`l1_organ_curriculum` is the
reference to beat: feature L1 15.74). The LR and λ_organ changes alone warrant it,
and `l1_adv_organ` must be re-run — its curriculum never ran.

**2. `audit_data_ceiling.py` — do this early; it gates everything below.**
```bash
python audit_data_ceiling.py \
  --manifest ../out_synthesis_train/literature_baseline_l1_organ_curriculum/phase_infer/manifest.csv \
  --report   ../out_synthesis_train/literature_baseline_l1_organ_curriculum/phase_infer/phase_eval_report.json
```
Correlates per-case error against (a) NCCT-vs-CECT agreement **inside bone** —
bone is contrast-invariant, so disagreement there is misregistration — and (b)
bowel-gas Dice between the two scans. If either explains the r=0.97 pattern, the
honest conclusion is a **data** ceiling, which is a thesis finding: improving
registration would move the metric more than any loss. CPU-only, no re-inference.

**3. HU-profile scenarios** (already in `run_scenarios.sh`):
- `l1_organ_huprofile` — tiered weighting + HU-profile
- `l1_huprofile_only` — HU-profile without the per-voxel organ term, isolating
  how much is the *level* constraint vs the texture weighting

**4. 2.5-D input — the cheap test of §1.3.** Feed `2k+1` adjacent axial slices as
input channels, predict the centre slice. `UNetGenerator(in_channels=2k+1)`; stays
a 2D network, ~no extra params; `dataset.py`'s depth-crop path already exists;
`infer_volume.py` already handles `pd > 1` and needs only asymmetric in/out depth.
Sweep k=2 (7.5 mm) vs k=5 (16.5 mm). Full `dims=3` only if this moves vessel HU.

**5. Deferred:** full 3D; CT-pretrained backbone as a **perceptual loss** (not as
initialisation — see §5).

Order rationale: 3 before 4 because it is cheaper, targets the metric that
demonstrably moved, and needs no cache rebuild; 4 costs a full re-preload and
would confound with 3 if run together.

---

## 4. Reading results without fooling yourself

- **Never judge on PSNR/SSIM alone.** They were flat across a range that
  contained a t=−4.22 effect. Lead with phase fidelity and per-organ HU.
- **Check every difference against the noise floor** that `analyze_runs.py`
  prints. Between-run org-SSIM spread was 0.0049 while `l1_adv`'s own within-run
  std was 0.011 — that run's ranking is not resolvable at all.
- **Use the paired per-case tests.** Between-case variance (6.2 HU) swamps
  between-model differences (3.2 HU); an unpaired comparison would find nothing.
- **Expect PSNR to fall** on a good model. Say so in advance.

---

## 5. On nnU-Net / open-weight models — deferred, with the argument recorded

**As generator initialisation — argue against, for now.** The r=0.97 cross-model
correlation says ~94% of per-case error is set by the case, not the model. A
better initialisation cannot recover error the input does not determine. nnU-Net
is also a *segmentation framework*: its value is the self-configuring pipeline,
and its decoder emits class logits, so you would keep only the encoder and discard
what makes it nnU-Net. Segmentation→translation transfer has prior art in medical
imaging (Models Genesis, MedicalNet, Swin UNETR are better-matched CT-pretraining
routes), but it is not established for NCCT→CECT, and nothing in the current
evidence points at initialisation as the bottleneck.

**As a perceptual loss — well-motivated, and the better use.** The perceptual loss
is VGG16 (ImageNet) or DINOv2/v3 (web images); neither has seen a CT. A
CT-pretrained encoder as feature extractor is domain-correct, and the plumbing
exists: `DinoPerceptualLoss` (`losses.py`) takes a pre-built backbone, and
`volEmbed_CTPhase/encoders.py` shows this project's pattern for loading a frozen
pretrained medical encoder with a guard against silent random-init.

Revisit both after step 2.

---

## 6. Open risks

- **n=20 test set** is thin for the headline claim. Expanding it is the cheapest
  way to harden the thesis's central result.
- **Zero-weighting bowel needs defending** — frame it as measured: 27% of voxels,
  ~1.3% of phase importance, gas/content not inferable from NCCT. Bowel is still
  *evaluated*, just not *trained*; keep it that way.
- **`_seg_reg` and `_seg_full` use different label conventions.** Weights assume
  `_seg_full`; `SEG_SUFFIX` should stay `'_seg_full'`.
- **Masks derive from the target CECT** (`dataset.py`). Fine as a loss
  (training-time only), but they **cannot** be fed as a generator input without
  NCCT-derived masks — relevant if mask conditioning is ever proposed.
- **Selection-metric change breaks comparability** with the two oldest runs,
  which were selected on global MAE.
