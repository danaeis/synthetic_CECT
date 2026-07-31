#!/usr/bin/env python
"""
Can the CECT contrast level be predicted from the NCCT at all?

This is the most direct test of the aleatoric claim in the whole project. Every
other measurement is an elimination — capacity ruled out by a flat width sweep,
data volume by a saturating learning curve, registration by a null correlation,
voxel scale by uniform spacing. Those together leave "the input does not determine
the target" as the only survivor, but they never measure it.

This does. Regress each case's CECT organ level on features of its OWN NCCT, under
a case-level split:

    R^2 ~ 0   the NCCT carries no information about how enhanced the CECT will be.
              That is aleatoric uncertainty, measured directly rather than inferred,
              and no architecture or loss can recover it.
    R^2 high  the level IS predictable and the model simply is not learning it —
              which would make this a modelling failure, not a data ceiling, and
              would redirect the whole roadmap.

Features are the NCCT's own per-organ median HU (already dumped by
scripts/dump_levels.py, so this costs nothing extra). They describe tissue density
before contrast — exactly the information a model has at inference.

A ridge fit in closed form, no sklearn: with ~97 training cases and <10 features,
anything fancier would overfit and obscure the answer.

Usage:
    python scripts/dump_levels.py            # produces splits/levels.json first
    python scripts/probe_level_predictability.py
    python scripts/probe_level_predictability.py --target liver
"""
import argparse
import json
from pathlib import Path

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np


def _ridge_fit(X, y, lam=1.0):
    """Closed-form ridge with an unpenalised intercept."""
    Xc = np.hstack([np.ones((len(X), 1)), X])
    n_f = Xc.shape[1]
    P = np.eye(n_f) * lam
    P[0, 0] = 0.0
    return np.linalg.solve(Xc.T @ Xc + P, Xc.T @ y)


def _r2(y, pred):
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--levels', type=Path, default=Path('splits/levels.json'))
    ap.add_argument('--target', default='aorta',
                    help='organ whose CECT median HU is predicted (default aorta)')
    ap.add_argument('--phase', default=None, help='default: first phase in the dump')
    ap.add_argument('--lam', type=float, default=1.0, help='ridge penalty')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    if not a.levels.exists():
        raise SystemExit(f"{a.levels} not found — run scripts/dump_levels.py first")
    db = json.loads(a.levels.read_text())
    organs = db['organs']
    phase = a.phase or db['phases'][0]
    if a.target not in organs:
        raise SystemExit(f"--target {a.target} not in {organs}")

    rows = {'train': [], 'test': []}
    for cid, rec in db['cases'].items():
        if phase not in rec:
            continue
        x = [rec[phase]['ncct'][o] for o in organs]
        y = rec[phase]['cect'][a.target]
        if not np.isfinite(y) or not all(np.isfinite(v) for v in x):
            continue
        # val+test pooled: this is a 2-parameter fit, not a tuned model, so a
        # held-out set is all that is needed.
        rows['train' if rec.get('split') == 'train' else 'test'].append((x, y))

    if len(rows['train']) < 10 or len(rows['test']) < 5:
        raise SystemExit(f"too few cases: {len(rows['train'])} train, "
                         f"{len(rows['test'])} held-out")

    Xtr = np.array([r[0] for r in rows['train']], float)
    ytr = np.array([r[1] for r in rows['train']], float)
    Xte = np.array([r[0] for r in rows['test']], float)
    yte = np.array([r[1] for r in rows['test']], float)

    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    w = _ridge_fit((Xtr - mu) / sd, ytr, a.lam)

    def pred(X):
        return np.hstack([np.ones((len(X), 1)), (X - mu) / sd]) @ w

    r2_tr, r2_te = _r2(ytr, pred(Xtr)), _r2(yte, pred(Xte))
    # The honest reference: predicting the training mean for every case. A model
    # that cannot beat this has learned nothing.
    r2_mean = _r2(yte, np.full_like(yte, ytr.mean()))

    print(f"target: {a.target} CECT median HU, phase '{phase}'")
    print(f"features: {len(organs)} NCCT organ medians")
    print(f"cases: {len(ytr)} train, {len(yte)} held-out\n")
    print(f"  target spread (held-out)   {yte.min():.0f}-{yte.max():.0f} HU "
          f"(sd {yte.std():.1f})")
    print(f"  R^2 train                  {r2_tr:+.3f}")
    print(f"  R^2 held-out               {r2_te:+.3f}")
    print(f"  R^2 of predicting the mean {r2_mean:+.3f}   (0 by construction)")
    print(f"  held-out MAE               {np.abs(yte - pred(Xte)).mean():.1f} HU")
    print(f"  MAE of predicting the mean {np.abs(yte - ytr.mean()).mean():.1f} HU")

    print(f"\n{'=' * 72}")
    if r2_te < 0.15:
        print(f"NOT PREDICTABLE (R^2 = {r2_te:+.3f}). The NCCT carries essentially no\n"
              f"information about how enhanced the CECT will be. This is aleatoric\n"
              f"uncertainty measured DIRECTLY, not inferred by elimination — the\n"
              f"strongest single piece of evidence that the ~8 HU floor is a property\n"
              f"of the task. No architecture or loss can recover it; only conditioning\n"
              f"on the level, or predicting a distribution instead of a point.")
    elif r2_te > 0.5:
        print(f"PREDICTABLE (R^2 = {r2_te:+.3f}). The level IS recoverable from the NCCT,\n"
              f"so the model's failure to track it is a MODELLING failure, not a data\n"
              f"ceiling. This contradicts the aleatoric hypothesis and should redirect\n"
              f"the roadmap toward why the model is not learning an available signal.")
    else:
        print(f"PARTIALLY PREDICTABLE (R^2 = {r2_te:+.3f}). Some of the level is in the\n"
              f"NCCT and some is not. Expect conditioning to recover roughly the\n"
              f"complement, and size the ceiling accordingly.")

    print(f"\nCaveat: a linear fit on {len(organs)} organ medians. A null here bounds\n"
          f"LINEAR predictability from these features, not every possible predictor —\n"
          f"but the features are exactly the tissue-density information the generator\n"
          f"itself has access to.")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({
            'target': a.target, 'phase': phase, 'organs': organs,
            'n_train': len(ytr), 'n_test': len(yte),
            'r2_train': r2_tr, 'r2_heldout': r2_te,
            'mae_heldout': float(np.abs(yte - pred(Xte)).mean()),
            'mae_mean_baseline': float(np.abs(yte - ytr.mean()).mean()),
        }, indent=2))
        print(f"\n[written] {a.out}")


if __name__ == '__main__':
    main()
