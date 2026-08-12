# Fix plan — "models barely differ" in the NCCT→CECT benchmark

Diagnosis (see the analysis behind each item):
- Global PSNR/SSIM/MAE are structurally flat — an identity copy scores 27.5/0.91 vs
  30/0.94 for the best model (`metrics.py:505-523`). Not a bug; wrong columns.
- The real problem is **variance collapse**: the generator recovers only ~10–20% of
  the real per-case enhancement variance (`analysis/enhancement.json`, slope β≈0.2,
  var_ratio≈0.12 for aorta). Absolute enhancement (dose/bolus timing/cardiac output)
  is not identifiable from NCCT, so every L1-family loss regresses to the same fixed
  per-organ average. All models converge to that same solution → they look alike.
- **Do NOT switch to L2.** L2 regresses to the conditional *mean* (L1 to the median)
  and collapses variance *harder* — it makes the sameness worse. Confirmed by the
  code's own note in `losses.py:341-344`.

Work top-down. Tier 0/1 are cheap and make the rest measurable; Tier 2 is the
actual fix; Tier 3 is how you prove it worked.

---

## Tier 0 — Correctness / consistency (do first, ~30 min)

- [ ] **Fix the HU-window default mismatch.**
  `dataset.py:307` defaults `hu_max=300`; `config.py:131` uses `400`. Config wins in
  the normal path, but any dataset built without config trains on a different domain
  than the benchmark scores on (`metrics.py:77` = [-200,400]).
  → Make `hu_min`/`hu_max` **required** in the dataset (no silent default), or set
  the defaults to `-200/400` to match config. Add an assert that dataset and
  `metrics.to_unit` share the same window.

- [ ] **Pin one HU window as a single source of truth.** Import `HU_MIN/HU_MAX` from
  config into `metrics.to_unit` defaults (or vice-versa) so training and scoring can
  never drift apart again.

## Tier 1 — Make the differences visible (evaluation/reporting, ~half day)

- [ ] **Promote the discriminating metrics; demote the flat ones.**
  In `benchmark.py` master table, mark global PSNR/SSIM/MAE as *secondary* and lead
  with `oMAE`, `featHU`, `phase`, `prob`. Keep the identity + (add) a
  conditional-mean-predictor row as permanent floors so every real model is read
  against "do nothing" and "predict the average".

- [ ] **Add variance-recovery columns to the master table.** Fold the three numbers
  `audit_enhancement.py` already computes — level-tracking **slope β**, **var_ratio**,
  and **bias** — into `benchmark.py` as first-class per-model columns (at least for
  aorta / portal vein / IVC / liver). These are the only columns that actually move
  when the core problem is fixed; without them you're flying blind.

- [ ] **Report the identifiability ceiling once.** Add a short note / oracle row: the
  best achievable per-organ HU error given NCCT-invisible enhancement, so nobody
  chases a floor that is data-limited, not model-limited.

## Tier 2 — The actual fix: stop averaging over the unknown enhancement level

Pick **2a first** (cheapest, machinery already exists); escalate to 2b if needed.
Keep L1 as the pixel term throughout — the fix is *conditioning/распределение*, not
the norm.

- [ ] **2a. Make per-organ level conditioning the default.**
  The pieces exist: `n_levels`/`level_proj`/FiLM in `models.py` and
  `OrganHUProfileLoss` in `losses.py`. Feed each case's target per-organ level as
  conditioning so the network is no longer forced to average over the unknown dose.
  Evidence it works: `level_all8` and `diff_hetero_nll` already reach the best
  featHU (~14.4) in the table.
  - [ ] Turn on `use_hu_profile` with real `organ_weights` (vessels weighted up).
  - [ ] Confirm FiLM `|gamma|` is non-zero after training (`film_stats()`); if it
        stays ~0 the conditioning bought nothing and needs a stronger signal.

- [ ] **2b. Model the distribution, not the mean.**
  If conditioning alone doesn't restore variance, move the *level* to a stochastic
  head: heteroscedastic NLL (already the strongest run, `diff_hetero_nll`) or a
  properly-tuned diffusion that *samples* an enhancement level rather than averaging
  all of them. Reserve this for when 2a is measured and insufficient.

- [ ] **2c. (Explicitly NOT doing) switch to L2.** Leave the note in the plan so it
  doesn't get re-proposed: L2 worsens variance collapse and blur.

## Tier 3 — Validate the fix (definition of done)

- [ ] Re-run `scripts/audit_enhancement.py` after 2a/2b. **Success = var_ratio rises
  from ~0.12 toward 1 and slope β rises from ~0.2 toward 1** on aorta/PV/IVC/liver.
- [ ] Re-run `benchmark.py`; success = models now **separate** on featHU / var_ratio,
  and RAPS moves toward 1.0 (less blur). Global PSNR/SSIM are allowed to stay flat —
  that's expected and not the target.
- [ ] Gut check: the trained model must beat the new conditional-mean-predictor row
  on var_ratio, not just on featHU. If it only beats it on featHU, it's still just
  predicting the average.

---

### One-line summary
It's not a benchmark bug and it's not the output gate — the pixel metrics are just
flat by construction, and every L1/L2 model collapses to the same per-case average
because NCCT doesn't contain the enhancement level. Fix reporting (Tier 1) so you can
see it, then restore the missing variance by **conditioning on / sampling the level**
(Tier 2) — keep L1, don't move to L2.
