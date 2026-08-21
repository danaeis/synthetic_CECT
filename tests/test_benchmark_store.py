"""
benchmark.py's accumulating result store.

WHY THIS EXISTS. Models finish training days apart, and re-scoring a finished
model costs ~an hour of volume I/O — but the table is only meaningful as a joint
comparison, because best/second-best marks, the paired t-tests and the
level-recovery regressions all need every model present at once. So each model's
PER-CASE rows are cached under `<out>/store/` and merged back in on later runs.

The failure modes that matter here are all silent ones:

  1. A later run scores only the new model and the earlier models VANISH from
     the table (the pre-store behaviour).
  2. A cached model is scored again and appears TWICE, or its stale rows win
     over the fresh ones.
  3. Entries scored under a different HU window / phase classifier are pooled
     into one table, where nothing in the numbers reveals the mismatch.
  4. FID/LPIPS come from the run that scored each model, not the run that prints
     the table, so the perceptual block disappears whenever the final invocation
     happens not to pass --perceptual.
  5. A stored entry predates a metric column, and csv.DictWriter raises on the
     unexpected key — losing the whole report rather than one cell.

Run directly:
    python tests/test_benchmark_store.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import csv
import json
import tempfile
import types

import benchmark as B
from organ_features import ORGANS

FAILS = []


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ''))
    if not ok:
        FAILS.append(name)


# --- stubs -----------------------------------------------------------------
# score_model is the expensive part (it loads three NIfTI volumes per case and
# runs the XGBoost phase classifier); the store is orthogonal to what it computes,
# so it is replaced by deterministic synthetic rows. That also makes "was this
# model re-scored?" directly observable via SCORED.

SCORED = []


def _phase_case(seed):
    return {
        'target_phase': 2,
        'pred_gen': 2, 'pred_real': 2,
        'gen_matches_target': True,
        'gen_matches_real': True,
        'real_matches_target': True,
        'gen_target_prob': 0.7 + 0.01 * seed,
        'real_target_prob': 0.9,
        'feature_l1_hu': 20.0 + seed,
        'per_organ_abs_err_hu': {o: 10.0 + seed for o in ORGANS},
        'gen_probs': [0.1, 0.1, 0.7, 0.1], 'real_probs': [0.0, 0.0, 1.0, 0.0],
    }


def _rows(model, n=4, lpips=None, extra_key=None):
    out = []
    for i in range(n):
        r = {
            'model': model, '_key': f'/data/case{i}/real.nii.gz',
            'case': 'real.nii.gz',
            'psnr': 20.0 + i, 'ssim': 0.8, 'mae': 0.05, 'mse': 0.004, 'pcc': 0.9,
            'org_psnr': 18.0 + i, 'org_ssim': 0.7, 'org_mae': 0.06,
            'org_mse': 0.005, 'org_pcc': 0.85,
            'body_psnr': 19.0, 'body_ssim': 0.75, 'body_mae': 0.055,
            'body_mse': 0.004, 'body_pcc': 0.88, 'body_frac': 0.4,
            'raps_hf': 0.9, 'grad_w1': 0.02, 'org_grad_w1': 0.03,
            'seam': 1.0, 'zflicker': 1.1, 'zaniso': 1.0,
            'lpips': float('nan') if lpips is None else lpips + 0.001 * i,
            'phase_match': 1, 'agree_real': 1,
            'gen_prob': 0.7, 'feature_l1_hu': 20.0 + i,
            '_phase_case': _phase_case(i),
            '_lvl': {o: (100.0 + 10 * i, 90.0 + 8 * i) for o in B.LEVEL_ORGANS},
        }
        if extra_key:
            r[extra_key] = 1.23
        out.append(r)
    return out


_ORIG = {'score_model': B.score_model, 'PhaseEvaluator': B.PhaseEvaluator}


def _install_stubs(spec):
    """spec: {model_name: kwargs for _rows}."""
    B.PhaseEvaluator = lambda *a, **k: types.SimpleNamespace()

    def fake_score_model(name, manifest, ev, hu_min, hu_max, gen_in_hu, **kw):
        SCORED.append(name)
        return _rows(name, **spec.get(name, {}))

    B.score_model = fake_score_model


def isolated(fn):
    """Undo the module-level stubbing after each test.

    `pytest tests/` imports every test module into ONE process, so a stubbed
    B.score_model left behind here would silently replace the real one for any
    other test file collected afterwards. Running this file directly never
    exposed that; the decorator makes both entry points equivalent.
    """
    def wrapper(*a, **kw):
        argv = sys.argv[:]
        try:
            return fn(*a, **kw)
        finally:
            B.score_model = _ORIG['score_model']
            B.PhaseEvaluator = _ORIG['PhaseEvaluator']
            sys.argv = argv
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _run(*argv):
    """Invoke benchmark.main() with argv; return the written master_table.md."""
    out = pathlib.Path(argv[argv.index('--out') + 1])
    sys.argv = ['benchmark.py'] + list(argv)
    B.main()
    md = out / 'master_table.md'
    return md.read_text() if md.exists() else ''


def _env():
    """tmpdir with a fake weights file, an --out dir and two dummy manifests."""
    root = pathlib.Path(tempfile.mkdtemp())
    (root / 'xgb.pkl').write_text('not-a-real-model')
    for m in ('a', 'b', 'c'):
        (root / f'{m}.csv').write_text('gen_path,real_path,mask_path,target_phase\n')
    return root


def _base(root, *extra):
    return ['--weights', str(root / 'xgb.pkl'),
            '--organ_map', str(root / 'no_such_map.json'),
            '--out', str(root / 'analysis'), *extra]


# --- tests -----------------------------------------------------------------

@isolated
def test_accumulates_across_runs():
    print('\naccumulation across separate invocations')
    root = _env()
    SCORED.clear()
    _install_stubs({'resvit': {}, 'syndiff': {}})

    t1 = _run(*_base(root, '--manifest', f'resvit={root}/a.csv'))
    check('run 1 scores and tables its own model',
          'resvit' in t1 and SCORED == ['resvit'], f'SCORED={SCORED}')
    check('run 1 wrote a store entry',
          len(list((root / 'analysis' / 'store').glob('*.json'))) == 1)

    SCORED.clear()
    t2 = _run(*_base(root, '--manifest', f'syndiff={root}/b.csv'))
    check('run 2 re-scores ONLY the new model', SCORED == ['syndiff'], f'{SCORED}')
    check('run 2 tables BOTH models', 'resvit' in t2 and 'syndiff' in t2)
    check('run 2 ran the paired block over both',
          'Paired per-case tests' in t2)
    check('report states what was merged',
          'accumulated in' in t2 and '1 scored in this run' in t2)

    # per_case.csv must hold every model's cases, not just this run's
    with (root / 'analysis' / 'per_case.csv').open() as f:
        models = {r['model'] for r in csv.DictReader(f)}
    check('per_case.csv carries both models', models == {'resvit', 'syndiff'}, models)


@isolated
def test_rescore_replaces_not_duplicates():
    print('\nre-scoring a cached model replaces its entry')
    root = _env()
    SCORED.clear()
    _install_stubs({'resvit': {}})
    _run(*_base(root, '--manifest', f'resvit={root}/a.csv'))
    t = _run(*_base(root, '--manifest', f'resvit={root}/a.csv'))
    check('model appears exactly once in the table',
          sum(1 for ln in t.splitlines() if ln.startswith('| resvit |')) ==
          sum(1 for _ in B.CATEGORY_SPECS[:4]),      # one row per rendered category
          'duplicate rows would double it')
    check('store still holds one entry for it',
          len(list((root / 'analysis' / 'store').glob('*.json'))) == 1)


@isolated
def test_incompatible_settings_are_refused():
    print('\nentries scored under different settings are not pooled')
    root = _env()
    _install_stubs({'resvit': {}, 'syndiff': {}})
    _run(*_base(root, '--manifest', f'resvit={root}/a.csv'))
    # different HU window -> the columns are not the same quantity
    t = _run(*_base(root, '--manifest', f'syndiff={root}/b.csv',
                    '--hu_min', '-1000'))
    check('mismatched entry excluded from the table', 'resvit' not in t)
    check('new model still reported', 'syndiff' in t)


@isolated
def test_fid_and_lpips_survive_the_store():
    print('\nFID/LPIPS cached from the run that computed them')
    root = _env()
    _install_stubs({'syndiff': {}})
    # Hand-write the entry the --perceptual path would have written, so the test
    # does not need torch + lpips + pytorch-fid installed.
    args = types.SimpleNamespace(hu_min=-200.0, hu_max=400.0, gen_not_hu=False)
    fp = B._fingerprint(args, root / 'xgb.pkl')
    B.store_write(root / 'analysis' / 'store', 'resvit',
                  _rows('resvit', lpips=0.31), fp,
                  {'fid': 42.5, 'fid_n_slices': 1800,
                   'fid_backend': 'pytorch-fid (TF-ported InceptionV3)',
                   'lpips_net': 'alex'})
    # ...and this run does NOT pass --perceptual.
    t = _run(*_base(root, '--manifest', f'syndiff={root}/b.csv'))
    check('perceptual table rendered from cached values',
          '| LPIPS | FID |' in t and '42.5' in t and 'Not computed' not in t)
    check('LPIPS entered the paired block', 'lpips' in t.split('Paired')[-1])
    check('a model without perceptual numbers shows —, not 0',
          '| syndiff | — | — |' in t, [ln for ln in t.splitlines()
                                       if ln.startswith('| syndiff |')])


@isolated
def test_store_survives_a_new_metric_column():
    print('\nstored rows predating a metric column')
    root = _env()
    _install_stubs({'resvit': {}, 'syndiff': {'extra_key': 'newmetric'}})
    _run(*_base(root, '--manifest', f'resvit={root}/a.csv'))
    _run(*_base(root, '--manifest', f'syndiff={root}/b.csv'))
    with (root / 'analysis' / 'per_case.csv').open() as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        cols = rd.fieldnames
    check('union of columns is written', 'newmetric' in cols, cols)
    check('the older model gets an empty cell, not a crash',
          all(r['newmetric'] == '' for r in rows if r['model'] == 'resvit'))


@isolated
def test_list_and_drop():
    print('\n--list_store / --drop')
    root = _env()
    _install_stubs({'resvit': {}, 'syndiff': {}})
    _run(*_base(root, '--manifest', f'resvit={root}/a.csv'))
    _run(*_base(root, '--manifest', f'syndiff={root}/b.csv'))
    sys.argv = ['benchmark.py'] + _base(root, '--drop', 'resvit')
    B.main()
    left = [json.loads(p.read_text())['model']
            for p in (root / 'analysis' / 'store').glob('*.json')]
    check('--drop removes exactly that model', left == ['syndiff'], left)
    t = _run(*_base(root, '--manifest', f'syndiff={root}/b.csv'))
    check('dropped model no longer in the table', 'resvit' not in t)


@isolated
def test_fresh_and_no_store():
    print('\n--fresh / --no_store')
    root = _env()
    _install_stubs({'resvit': {}, 'syndiff': {}})
    _run(*_base(root, '--manifest', f'resvit={root}/a.csv'))
    t = _run(*_base(root, '--manifest', f'syndiff={root}/b.csv', '--fresh'))
    check('--fresh tables only this run', 'resvit' not in t and 'syndiff' in t)
    n_before = len(list((root / 'analysis' / 'store').glob('*.json')))
    _run(*_base(root, '--manifest', f'syndiff={root}/c.csv', '--no_store'))
    check('--no_store writes nothing to the store',
          len(list((root / 'analysis' / 'store').glob('*.json'))) == n_before)


@isolated
def test_slug_collisions():
    print('\nstore filenames')
    a, b = B._slug('run/venous'), B._slug('run_venous')
    check('names that sanitise alike still get distinct files', a != b, (a, b))
    check('slug is filesystem-safe', '/' not in a and '/' not in b)


if __name__ == '__main__':
    print('=' * 70)
    print('BENCHMARK RESULT STORE (cross-run accumulation)')
    print('=' * 70)
    test_accumulates_across_runs()
    test_rescore_replaces_not_duplicates()
    test_incompatible_settings_are_refused()
    test_fid_and_lpips_survive_the_store()
    test_store_survives_a_new_metric_column()
    test_list_and_drop()
    test_fresh_and_no_store()
    test_slug_collisions()
    print('\n' + '=' * 70)
    if FAILS:
        print(f'FAILED ({len(FAILS)}): {", ".join(FAILS)}')
        sys.exit(1)
    print('ALL PASS')
