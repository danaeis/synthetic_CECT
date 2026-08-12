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

## Tier 0 — Correctness / consistency (do first, ~30 min)  ✅ DONE

- [x] **Fix the HU-window default mismatch.**
  `dataset.py` default was `hu_max=300` while `config.py` uses `400`. Fixed the
  fallback to `-200/400` and added a runtime warning when the dataset window differs
  from the canonical `metrics.HU_MIN/HU_MAX`.

- [x] **Pin one HU window as a single source of truth.** `metrics.py` now owns the
  canonical window as module constants `HU_MIN/HU_MAX` (used as `to_unit` defaults);
  `config.py` asserts its `HU_MIN/HU_MAX` equal them at import time, so any future
  drift fails loudly instead of silently scoring in the wrong domain.

## Tier 1 — Make the differences visible (evaluation/reporting, ~half day)  ✅ DONE

- [x] **Promote the discriminating metrics; demote the flat ones.**
  `benchmark.master_table` now carries a "How to read this table" note flagging the
  global PSNR/SSIM/MAE/MSE/PCC columns as SECONDARY and flat by construction, and
  points readers at the organ-region / phase / level-recovery columns as primary.
  The `identity` floor rows already exist in the shipped tables.

- [x] **Add variance-recovery columns to the master table.** New **βlev** (mean
  level-tracking slope) and **varR** (mean var(gen)/var(real)) columns, computed
  across cases over aorta / portal vein / IVC / liver — the same quantities
  `audit_enhancement.py` produces, folded in as first-class per-model columns.
  Covered by `tests/test_benchmark_discovery.py::test_level_recovery`.

- [x] **Report the identifiability ceiling once.** The level-recovery legend states
  that βlev/varR → 0 is indistinguishable from a conditional-mean predictor (the
  L1/L2 fixed point under NCCT-invisible enhancement), so a low varR is read as the
  data ceiling, not model failure. (A dedicated conditional-mean *oracle row* was not
  added — varR itself is the collapse detector — but could be added later if wanted.)

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
