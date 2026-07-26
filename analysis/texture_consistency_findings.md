# Texture and reconstruction consistency — first measurements

Addresses two problems visible in reconstructed volumes but absent from every
results table in `PROJECT_PLAN.md`: **patch seams** and **lost vein/tissue detail**.

## Why they were invisible

Every metric in the suite — PSNR, SSIM, MAE, PCC, the organ-region variants, and
`feature_l1_hu` with its per-organ breakdown — scores either a per-voxel difference
or a median HU *level* inside a mask. None measures texture, and none is aware that
the volume was assembled from tiles.

This also bounds the reach of `PROJECT_PLAN.md` §1.2. The r = 0.957–0.983 cross-model
correlation shows ~94% of per-case error is set by the case rather than the loss —
but that error is `feature_l1_hu`, a level metric. **The ceiling is on the level axis;
the texture axis was never measured.** That distinction is what keeps a diffusion arm
worth running: a data ceiling on HU level does not bound texture fidelity.

## New metrics (`metrics.py`, tested in `test_metrics.py`)

| metric | reads | target |
|---|---|---|
| `raps_hf_ratio` | high-frequency spectral energy, gen/real (amplitude) | 1.0 |
| `grad_hist_distance` | W1 between gradient-magnitude distributions, global + masked | 0.0 |
| `seam_energy` | tile-boundary over interior first-difference | 1.0 |
| `z_flicker` | inter-slice difference, gen/real | 1.0 |
| `z_flicker_anisotropy` | the same, each volume normalised by its own in-plane gradient | 1.0 |

`raps_hf_ratio`, `seam_energy` and both flicker terms are **ratios where 1.0 is the
target and both directions are failures** — rank by |value − 1|, never "higher is
better". A model can beat 1.0 on RAPS by hallucinating texture.

Wired into `benchmark.py`: master-table columns plus paired per-case tests (ratio
metrics enter the paired tests as |x − 1|). `seam` is NaN when a model has no known
tiling geometry, so external/whole-slice baselines are not scored against a tiling
that never happened.

## Measurements

### Seams are real, and it is not the metric's imagination

One test case, `l1_bowel_zero`, scored against its real venous CECT, with the case's
own **NCCT as a control** — a real scan the model never touched, so it establishes
the floor these metrics can reach on this data:

| | seam | raps_hf | zflicker | zaniso |
|---|---|---|---|---|
| generated | **1.273** | **0.822** | 0.876 | 1.031 |
| NCCT (control) | 1.035 | 1.333 | 1.390 | 1.087 |

1. **seam 1.273 vs a 1.035 control.** Tile boundaries carry 27% more gradient energy
   than interior lines, while an untouched real scan measured with the identical
   tile indices sits at 1.04. The seam is a reconstruction artifact, confirmed.
2. **raps_hf 0.822, from an input at 1.333.** The model retains 82% of the ground
   truth's high-frequency amplitude — while the NCCT it was given carries 33% *more*
   than the target. **The detail is being destroyed by the model, not missing from
   its input.**
3. **Through-plane flicker is not the problem.** `zaniso` 1.031 vs a 1.087 control:
   through-plane anisotropy is already better than the control's. This contradicts
   the expectation that independent per-slice generation would show as flicker, and
   it **deprioritises Z-consistent inference** relative to the stitching and norm work.

Note the raw `z_flicker` of 0.876 reads as "too smooth in Z" even though every slice
was generated independently — in-plane blur lowers it faster than slice-independence
raises it. `z_flicker_anisotropy` exists because of this measurement.

### Seam cause: stitching or weights? (`norm_attribution.py`)

Two crops of the same voxels differing only in surrounding context; the disagreement
on the shared region splits into a DC offset (which survives overlap-averaging as a
seam) and a residual (which blending attenuates).

Random-init architecture comparison, one volume, 12 sites — **relative ordering only**:

| norm | output std (HU) | drift @ shift 32 | DC fraction |
|---|---|---|---|
| instance | 46.3 | 13.98 HU (30.2% of output std) | 32.4% |
| group | 26.7 | 6.83 HU (25.6%) | 70.7% |
| batch | **0.29 — degenerate** | not interpretable | — |

**The BatchNorm row is a trap that the script now refuses to report as a win.**
Eval-mode BN applies `running_var=1`, which does not match an untrained network's
activations, so the output collapses to a near-constant 0.29 HU std — trivially
context-invariant, and it scored a spurious 0.02 HU drift, ~700× "better" than
InstanceNorm. `context_shift_drift` now returns an `out_std_hu` control and flags
any model below 5 HU as degenerate. The structural argument for eval-mode BN being
tile-invariant still holds; the *number* has to come from a trained checkpoint.

## Status and what is still needed

Done and verified locally: the metrics (14 test groups pass), the benchmark wiring,
weighted blending in `infer_volume.py` (`--blend hann|gaussian|uniform`,
`--edge_margin`, `--overlap`; 20/20 smoke checks including an identity-generator
partition-of-unity round trip), a selectable `GEN_NORM`, and `norm_attribution.py`.

Blocked on the GPU box, where the trained checkpoints and the other 19 test volumes
live:

1. `python benchmark.py --runs_dir ../out_synthesis_train --weights ...` — the five
   existing runs on the new columns. The expected result, which validates the whole
   exercise, is that `l1_only` is **worst** on `raps_hf`/`grad_w1` while leading PSNR.
2. `python scripts/norm_attribution.py --scenario_dir <run>` on a trained checkpoint — the
   number that decides whether `GEN_NORM` must change. High DC fraction ⇒ retrain;
   low ⇒ `--blend hann` suffices.
3. Re-infer one run with `--blend hann` and re-measure `seam`. If it does not
   approach 1.0, the residual is the norm, and (2) says so independently.

Local caveat: `smoke_test.py` fails every scenario that builds `PerceptualLoss`
(`No module named 'torchvision'`) — a local environment gap, unrelated to these
changes. The `l1_only` and `3d_l1_only` scenarios, which exercise the modified
generator construction on both paths, pass.

---

## Update 2026-07-26 — the seam and the receptive field are one mechanism

`scripts/erf.py` backpropagates from a single centre output pixel and measures
where the input gradient lives. Run at identical init, patch 256, base_ch 64,
varying only the norm layer:

| norm | support radius | r50 | r95 | nonzero pixels |
|---|---|---|---|---|
| instance (current default) | **128 = patch/2** | 14 | 105 | 100.0% |
| group | **128 = patch/2** | 4 | 82 | 100.0% |
| batch (eval, fixed stats) | **94** | 2 | 4 | 53.9% |

Under instance and group norm the support radius equals `patch/2` at **every**
patch size tried — r95 was 54 / 105 / 157 px at patch 128 / 256 / 384, a constant
82% of `patch/2`. A receptive field does not scale with the input. This is the
normalisation: instance and group statistics are computed over the patch's own
spatial extent, so every input pixel reaches every output pixel through them, and
the reach grows with the patch without bound.

Only batch norm exposes the architecture's real field: support radius 94 px, a
189-px box — **wider** than the ~140 px usually quoted for a 4-level U-Net.

**This is the same mechanism as `seam_energy`, observed from the gradient side
instead of the output side.** The norm-attribution numbers above (instance
drift@shift32 13.98 HU / 30.2% vs group 6.83 HU / 25.6%) and this ERF measurement
are two symptoms of one cause: the tile's own statistics enter every output voxel.
That is why `seam` is 1.34–1.67 for every model trained so far and never
approaches 1.0, regardless of loss configuration.

Two consequences for the roadmap.

1. **The ERF argument cannot be used against attention here.** "Convolution
   already spans the patch, so a long-range module adds nothing" fails, because
   the apparent global reach is a per-channel *scalar* (the patch mean/std)
   leaking everywhere — it carries no spatial structure. Attention carries
   structured spatial relations; the two are not substitutes, and the ERF number
   does not adjudicate between them. Patch geometry has to be decided on
   physical field-of-view grounds instead.
2. **`GEN_NORM='group'` is now a two-symptom hypothesis, not a one-off seam
   tweak.** It should reduce `seam` *and* narrow the spurious coupling (r50 drops
   14 → 4 px). Cheapest high-value run available: one flag, no code change.

Caveat: measured at random init. Under batch norm r95 = 4 px while support = 94
px, meaning gradient mass sits almost entirely on the shallow enc1→dec1 skip — an
untrained U-Net is dominated by its shortest path, and training is what pushes
signal deeper. The mass radii are therefore a **lower bound** on a trained
model's. Re-run `scripts/erf.py --scenario_dir <run>` on the training host, where
the checkpoints live.
