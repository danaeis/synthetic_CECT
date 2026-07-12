# Scenario ablation — results overview (as of the 6 completed runs in `simlified_train/`)

Source: `train/simlified_train/literature_baseline_*/{history.json,run_config.json,samples/}`.
`l1_only` has now been synced in and is included below.

## Results table

| scenario | losses | best/final epoch | val_loss | PSNR | SSIM |
|---|---|---|---|---|---|
| **l1_only** | **L1 only** | **41/53** | **0.0148** | **29.93** | **0.9840** |
| adv_only | L1+adv | 15/27 | 0.0154 | 29.51 | 0.9828 |
| pix2pixhd_nofeat_baseline | L1+adv+perc | 15/27 | 0.0153 | 29.70 | 0.9832 |
| pix2pixhd_baseline | L1+adv+perc+FM | 13/25 | 0.0153 | 29.57 | 0.9829 |
| extra_nofeat_ssim | L1+adv+perc+ssim | 35/47 | 0.0152 | 29.65 | 0.9832 |
| extra_ssim | L1+adv+perc+FM+ssim | 13/25 | 0.0153 | 29.69 | 0.9831 |
| extra_gradient | L1+adv+perc+gradient | 40/42 | 0.0153 | 29.64 | 0.9831 |

**`l1_only` has the best PSNR/SSIM/val_loss of all 7 scenarios — but this is not a
win, it's the classic L1-regression-to-the-mean signature.** Visual check
(`ep053.png` vs e.g. `pix2pixhd_baseline/ep025.png`, same fixed val batch): `l1_only`'s
liver/kidney patch (row 3) is visibly flatter/smoother, missing the fine
parenchymal texture present in the real CECT — L1 alone optimizes toward a "safe,"
blurry pixel-wise average, which pixel-distance metrics (PSNR/SSIM) systematically
reward even though it's the wrong target clinically. It also trained far longer
before stopping (53 epochs, best at 41) than any adversarial-inclusive scenario
(25-27 epochs, best at 13-15) — the adversarial term's warmup (epoch 5) introduces a
competing objective that the current L1-dominated weights aren't letting it win,
and the model plateaus early instead of the adversarial loss driving further,
different (sharper) improvement.

## Findings

1. **All 6 scenarios converge to nearly identical val metrics** (val_loss 0.0152–0.0154,
   PSNR 29.5–29.7, SSIM 0.9828–0.9832 — a ~0.2 PSNR / 0.0004 SSIM spread across the
   *entire* ablation). Sample grids (which all draw the same fixed first val batch,
   since `val_loader` isn't shuffled) look visually near-indistinguishable across
   `pix2pixhd_baseline`, `pix2pixhd_nofeat_baseline`, and `extra_gradient` too —
   consistent with the metrics, not just a metric-blindspot.
2. **Root cause, most likely: `lambda_l1=100` dominates everything else.**
   `lambda_adv=1` (and adversarial's own weight is further ramped via
   `adv_warmup_epochs=5`, capping its *effective* magnitude at 1.0 vs L1's 100 —
   a 100:1 ratio), `lambda_perceptual=10`, `lambda_gradient=5`, `lambda_ssim=10`.
   Every extra loss is numerically present but too small relative to L1 to visibly
   steer the output. This matches the LR sheet: `Cho21` used the same 1:100 GAN:L1
   ratio (consistent with adv_only's floor-level result), but `Yan22`/`Yan23`/
   `Yan24c` — the very papers this composite loss cites for the "adversarial"
   component — use `lambda_adv=2, lambda_l1=25` (12.5:1, not 100:1). Current config
   is more L1-dominant than most of its own cited baselines.
3. **adv_only is the consistent floor** — lowest PSNR/SSIM of the six, though only
   marginally (adding perceptual recovers ~0.2 PSNR). Adversarial alone isn't
   pulling its weight yet, consistent with finding 2.
4. **`extra_gradient` and `extra_nofeat_ssim` trained the longest before early-stopping**
   (best epoch 40/35 vs 13–15 for the rest) — i.e. they kept finding real
   improvement for longer, rather than plateauing right after the adversarial
   warmup (epoch 5) like the others. Structural losses (gradient/SSIM) may be more
   complementary to L1 under the current weight regime than adversarial/perceptual,
   which seem to plateau fast and then get overridden by L1's dominance.
5. **Data-hygiene bug found while reading these results — not yet fixed**:
   `adv_only`'s `train.log` shows 3 separate `Training 1 → 80` starts (2026-07-09,
   07-11, 07-12 — the last interrupted after ~30s). `history.json` reflects only the
   *second* run (27 epochs), but `samples/` still has `ep039-043.png` left over from
   the *first*, longer run. This isn't just stale — `_save_samples`'s pruning
   (`sorted(glob('ep*.png'))[:-keep]`, `trainer.py`) sorts by epoch-number-in-filename,
   not by save recency, so when a fresh (non-`--resume`) run restarts at epoch 1 in
   an output_dir that already has higher-numbered images from an earlier longer run,
   the new run's own low-numbered samples get pruned *in favor of* the stale
   high-numbered ones every time — the current samples for a re-run scenario can be
   the wrong run entirely. Same pruning pattern exists for `ckpt_ep*.pth`
   checkpoints. `pix2pixhd_nofeat_baseline` was also re-run once but happened not to
   hit this (its 2nd run's final epoch, 27, exceeded whatever the 1st run left
   behind). `best_model.pth` is unaffected (fixed filename, unconditionally
   overwritten on new best within a run).

## Recommended λ rebalance

Change exactly two weights, leave everything else untouched — keeps the ablation
interpretable (one hypothesis, one change) and both new values are taken directly
from this codebase's own cited baselines rather than guessed:

| weight | current | recommended | source |
|---|---|---|---|
| `lambda_l1` | 100 | **25** | matches `Yan22`/`Yan23`/`Yan24c` (`lambda2`/`lambda4`=25) |
| `lambda_adv` | 1 | **2** | matches `Yan22`/`Yan23`/`Yan24c` (`lambda1`=2) |

Everything else (`lambda_perceptual=10`, `lambda_feature_match=10`, `lambda_ssim=10`,
`lambda_gradient=5`, ...) stays as-is — shrinking `lambda_l1` alone already
quadruples their *relative* influence (e.g. perceptual:L1 goes from 10:100 to
10:25), without touching more than one thing at a time. Resulting ratios move from
the current 100:1 (L1:adv, matching `Cho21` — which is consistent with `adv_only`
being the floor performer here) to 12.5:1, matching the papers this codebase's own
`config.py` docstring already cites for the adversarial component.

**Caveat going in**: expect PSNR/SSIM/val_loss to likely get *slightly worse*, not
better, if this works as intended — that would mean the adversarial/perceptual
terms are finally winning enough influence to trade some pixel-fidelity for
sharpness/texture, the classic and expected signature (same reasoning as `l1_only`'s
result above, just in the other direction). Judge this rebalance primarily by the
sample grids (does row 3's liver texture look closer to real CECT than `l1_only`'s
flat blur?), not by whether it beats `l1_only`'s PSNR — beating that metric was
never really the goal.

## Other next steps

1. **Fix the stale-sample/checkpoint pruning bug** (finding 5) before more scenarios
   accumulate re-runs — otherwise "look at the latest sample" silently becomes
   unreliable. Fix: prune by file mtime, not by parsed/sorted epoch number.
2. Once the λ rebalance is tested, re-run the two "most promising" extras
   (`extra_gradient`, `extra_nofeat_ssim` — finding 4) under the new weights before
   running the remaining untested extras (`extra_frequency`, `extra_saliency`,
   `extra_cycle`, `extra_organ`, `extra_seg_consistency`), rather than running
   everything blind under weights already shown to suppress this whole class of
   loss.
3. `train.py` currently has no CLI override for loss weights (only the `use_*`
   on/off flags, `--epochs`/`--batch_size`/`--lr`/patch settings) — testing the
   rebalance above requires either editing `config.py`'s shared defaults directly,
   or adding `--lambda_l1`/`--lambda_adv` CLI overrides so it can be expressed as
   one more `run_scenarios.sh` line without changing the default every other
   scenario would also pick up. Recommend the latter — ask before doing either.
