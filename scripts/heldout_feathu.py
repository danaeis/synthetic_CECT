#!/usr/bin/env python
"""
featHU recomputed over the organs the model was NOT told about.

Why this is necessary. `featHU` is the mean per-organ |median-HU error| over 16
organs — which is exactly what a level-conditioned model is handed. Conditioning
on aorta median HU and then scoring aorta median-HU error is partly circular, and
the more organs are conditioned on, the worse the contamination gets.

So report two numbers side by side:

    featHU (all)       comparable to every existing run, but contaminated
    featHU (held-out)  organs never used as conditioning — non-circular

The held-out number is the one that answers the scientific question: does knowing
ONE organ's level let the model fix the OTHERS? If conditioning on the aorta alone
pulls down error across the remaining 15, contrast level is a single global latent
factor. If it only fixes the aorta, each organ carries independent variation.

Reads the `per_organ_abs_err_hu` that phase_eval.py already emits, so no
re-inference is needed.

Usage:
    python scripts/heldout_feathu.py \
      --report ../out_synthesis_train/literature_baseline_level_aorta/phase_infer/phase_eval_report.json \
      --cond_organs aorta
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--report', type=Path, required=True,
                    help='phase_eval_report.json')
    ap.add_argument('--cond_organs', nargs='*', default=None,
                    help='conditioned organs to exclude; default reads '
                         'cond_organs from the run_config.json beside the report')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    rep = json.loads(a.report.read_text())
    cases = rep.get('per_case') or rep.get('cases') or []
    if not cases:
        raise SystemExit(f"no per-case entries in {a.report}")

    cond = a.cond_organs
    if cond is None:
        # run_config.json sits two levels up from phase_infer/<phase>/report.json
        # and one level up from phase_infer/report.json — try both.
        for up in (2, 3):
            rc = a.report.parents[up - 1] / 'run_config.json'
            if rc.exists():
                cond = json.loads(rc.read_text()).get('cond_organs') or []
                print(f"cond_organs from {rc}: {cond}")
                break
        else:
            cond = []
    cond = set(cond or [])
    if not cond:
        print("no conditioned organs — held-out featHU equals featHU by definition")

    all_err, held_err, cond_err = [], [], []
    organs_seen = set()
    for c in cases:
        po = c.get('per_organ_abs_err_hu') or {}
        organs_seen |= set(po)
        vals = [(o, e) for o, e in po.items() if e is not None]
        if not vals:
            continue
        all_err.append(np.mean([e for _, e in vals]))
        h = [e for o, e in vals if o not in cond]
        k = [e for o, e in vals if o in cond]
        if h:
            held_err.append(np.mean(h))
        if k:
            cond_err.append(np.mean(k))

    n_held = len(organs_seen - cond)
    print(f"\n{len(all_err)} cases | {len(organs_seen)} organs scored | "
          f"{len(cond & organs_seen)} conditioned, {n_held} held out")
    print(f"\n  featHU (all organs)       {np.mean(all_err):7.3f} HU")
    if held_err:
        print(f"  featHU (HELD-OUT organs)  {np.mean(held_err):7.3f} HU   <-- non-circular")
    if cond_err:
        print(f"  featHU (conditioned only) {np.mean(cond_err):7.3f} HU   "
              f"(expected low — the model was told these)")

    out = {'n_cases': len(all_err),
           'cond_organs': sorted(cond),
           'n_heldout_organs': n_held,
           'feathu_all': float(np.mean(all_err)) if all_err else None,
           'feathu_heldout': float(np.mean(held_err)) if held_err else None,
           'feathu_conditioned': float(np.mean(cond_err)) if cond_err else None}

    if held_err and cond_err:
        print(f"\n  Compare the held-out number against the same arm run with\n"
              f"  --level_mode population. THAT gap is the result; the oracle\n"
              f"  number alone is not reportable.")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=2))
        print(f"\n[written] {a.out}")


if __name__ == '__main__':
    main()
