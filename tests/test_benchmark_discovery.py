"""
benchmark.py's run discovery, tiling lookup and paired-test guard.

WHY THIS EXISTS. A multi-phase run writes one manifest per phase SUBDIRECTORY
(`<run>/phase_infer/venous/manifest.csv`), because a multi-phase model emits
several volumes per case and they would collide on `{case_id}_syn.nii.gz` in one
directory. `discover()` looked only for the flat `<run>/phase_infer/manifest.csv`,
so `multiphase_film` and `multiphase_uncond` were absent from every benchmark
table — not reported as skipped, just gone.

Three failures, all silent, all of the kind this project cares most about:

  1. discover()    multi-phase runs missing from the table entirely.
  2. read_tiling() `manifest.parent.parent` is `<run>` for a flat manifest but
                   `<run>/phase_infer` for a per-phase one, so the config was not
                   found and `seam` came out NaN — reading as "external model, no
                   tiling geometry" rather than as a bug.
  3. paired_block() `paired_t([])` returns (0.0, 0.0, 0), so an arterial arm
                   compared against a venous baseline printed a full block of
                   '+0.000 ns' rows that read as "identical to the baseline" when
                   in fact no cases were comparable at all.

Run directly:
    python tests/test_benchmark_discovery.py
"""

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json
import shutil
import tempfile

import benchmark as B

FAILS = []


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ''))
    if not ok:
        FAILS.append(name)


def _make_tree(root: pathlib.Path):
    """A runs_dir with one of each layout the benchmark has to handle."""
    cfg = json.dumps({'patch_size': 128, 'overlap': 0.5})

    # flat, single-phase
    d = root / 'literature_baseline_l1_only'
    (d / 'phase_infer').mkdir(parents=True)
    (d / 'run_config.json').write_text(cfg)
    (d / 'phase_infer' / 'manifest.csv').write_text('gen_path\n')

    # multi-phase: one manifest per phase subdirectory
    d = root / 'literature_baseline_multiphase_film'
    for ph in ('venous', 'arterial'):
        (d / 'phase_infer' / ph).mkdir(parents=True)
        (d / 'phase_infer' / ph / 'manifest.csv').write_text('gen_path\n')
    (d / 'run_config.json').write_text(cfg)

    # trained but never inferred → must be REPORTED, not silently dropped
    d = root / 'literature_baseline_notinferred'
    d.mkdir(parents=True)
    (d / 'best_model.pth').write_text('x')
    (d / 'run_config.json').write_text(cfg)

    # external model: a manifest but no run_config anywhere → seam must be NaN
    d = root / 'external_baseline'
    (d / 'phase_infer').mkdir(parents=True)
    (d / 'phase_infer' / 'manifest.csv').write_text('gen_path\n')

    # a stray file in runs_dir must not crash the scan
    (root / 'notes.txt').write_text('x')


def test_discover():
    print('\ndiscover()')
    root = pathlib.Path(tempfile.mkdtemp())
    try:
        _make_tree(root)
        found = B.discover(root)

        check('flat single-phase run found', 'l1_only' in found)
        check('multi-phase run found as one model PER PHASE',
              {'multiphase_film/venous', 'multiphase_film/arterial'} <= set(found),
              f'{sorted(k for k in found if "multiphase" in k)}')
        # Pooling the phases would average an arterial model's featHU into the
        # venous comparison and make the multi-phase arms incomparable to every
        # single-phase run — which is the entire point of the M1/M2/M3 design.
        check('phases are NOT pooled into one row', 'multiphase_film' not in found)
        check('external model (no run_config) still found', 'external_baseline' in found)
        check('un-inferred run is not scored', 'notinferred' not in found)
        check('stray file does not break the scan', len(found) == 4, f'{len(found)}')

        for name, m in found.items():
            check(f'{name}: manifest path exists', m.exists())
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_read_tiling():
    print('\nread_tiling()')
    root = pathlib.Path(tempfile.mkdtemp())
    try:
        _make_tree(root)
        found = B.discover(root)

        flat = B.read_tiling(found['l1_only'])
        check('flat layout resolves the config',
              flat == {'patch_size': 128, 'overlap': 0.5}, f'{flat}')

        for ph in ('venous', 'arterial'):
            t = B.read_tiling(found[f'multiphase_film/{ph}'])
            check(f'per-phase layout resolves the config ({ph})',
                  t == {'patch_size': 128, 'overlap': 0.5}, f'{t}')

        # An external model genuinely has no tiling geometry, and NaN seam is the
        # right answer there. The bug was that multi-phase runs looked like this.
        check('no run_config anywhere → {} (seam NaN is correct)',
              B.read_tiling(found['external_baseline']) == {})

        # Bounded walk: a config far above must not be picked up, or seam would be
        # scored against a tiling geometry from an unrelated run.
        (root / 'run_config.json').write_text(json.dumps({'patch_size': 999,
                                                          'overlap': 0.9}))
        check('does not reach a config outside the run dir',
              B.read_tiling(found['external_baseline']) == {})
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _rows(keys):
    return [{'_key': k, 'psnr': 30.0, 'ssim': 0.94, 'org_ssim': 0.96, 'org_mae': 0.03,
             'feature_l1_hu': 14.0, 'raps_hf': 0.84, 'grad_w1': 0.002,
             'org_grad_w1': 0.006, 'seam': 1.35, 'zflicker': 0.9} for k in keys]


def test_paired_block_disjoint_cases():
    print('\npaired_block() with no shared cases')
    out = []
    B.paired_block({'base': _rows(['/venous/a', '/venous/b', '/venous/c']),
                    'mp/venous': _rows(['/venous/a', '/venous/b', '/venous/c']),
                    'mp/arterial': _rows(['/arterial/a', '/arterial/b'])},
                   'base', out)
    txt = '\n'.join(out)

    check('arterial arm is flagged not comparable',
          'no cases in common' in txt)
    # The failure mode: a full block of zero-delta rows that reads as "identical
    # to the baseline" when nothing was compared.
    arterial_lines = [l for l in out if l.startswith('mp/arterial')]
    check('arterial arm emits no fake zero-delta rows',
          not any('+0.000' in l for l in arterial_lines),
          f'{len(arterial_lines)} line(s)')
    check('venous arm still gets a real comparison block',
          any(l.startswith('mp/venous') and 'feature_l1_hu' in l for l in out))
    check('model column is wide enough for a <run>/<phase> name',
          all('mp/arterialfeature' not in l and 'mp/venousfeature' not in l
              for l in out))

    # Real run names are far longer than the old fixed 26-char column, which
    # printed 'multiphase_film_adv/venousfeature_l1_hu' in the shipped report.
    long = 'multiphase_film_adv_slices11/venous'
    out2 = []
    B.paired_block({'l1_only': _rows(['/a', '/b']), long: _rows(['/a', '/b'])},
                   'l1_only', out2)
    check('a 35-char model name does not collide with the metric column',
          not any(f'{long}feature' in l for l in out2),
          next((l for l in out2 if long in l), ''))


def test_resolve_baseline():
    print('\nresolve_baseline()')
    scored = ['l1_only', 'multiphase_film/venous', 'resvit']

    check('exact name resolves', B.resolve_baseline(scored, 'resvit') == 'resvit')
    # discover() strips the prefix, so the run DIRECTORY name must still match.
    check('literature_baseline_ prefix is stripped',
          B.resolve_baseline(scored, 'literature_baseline_l1_only') == 'l1_only')
    check('bare name matches a prefixed key',
          B.resolve_baseline(['literature_baseline_l1_only'], 'l1_only')
          == 'literature_baseline_l1_only')
    check('unset falls back to the *only* reference arm',
          B.resolve_baseline(scored, None) == 'l1_only')
    # `l1_huprofile_only` sorts first and used to win the fallback, silently
    # baselining a whole paired report against an ablation.
    check('unset prefers l1_only over another *only* name',
          B.resolve_baseline(['l1_huprofile_only', 'l1_only'], None) == 'l1_only')
    check('unset with no *only* arm takes the first model',
          B.resolve_baseline(['resvit', 'ours'], None) == 'resvit')
    # The real failure: a baseline whose manifest was never passed. It must be
    # reported as unresolvable so main() can exit BEFORE scoring, not silently
    # resolve to some other model's rows.
    check('a name that was never scored resolves to None',
          B.resolve_baseline(scored, 'ours') is None)
    check('empty model list does not crash',
          B.resolve_baseline([], None) is None)


def test_level_recovery():
    """The two columns that separate a real generator from a conditional-mean
    averager (variance collapse). β/varR → 1 track the case, → 0 emit the mean."""
    print('\n_level_recovery() / _organ_medians()')
    import math
    import numpy as np

    real = np.array([100., 120., 140., 90., 160., 110.])

    perfect = B._level_recovery(real, real.copy())
    check('perfect tracker → beta≈1, var_ratio≈1',
          abs(perfect['beta'] - 1) < 1e-9 and abs(perfect['var_ratio'] - 1) < 1e-9,
          f"beta={perfect['beta']:.3f} varR={perfect['var_ratio']:.3f}")

    mean_pred = B._level_recovery(real, np.full_like(real, real.mean()))
    check('constant (mean) output → beta≈0, var_ratio≈0',
          abs(mean_pred['beta']) < 1e-9 and abs(mean_pred['var_ratio']) < 1e-9,
          f"beta={mean_pred['beta']:.3f} varR={mean_pred['var_ratio']:.3f}")

    # Aorta-like partial recovery matching analysis/enhancement.json (beta≈0.23).
    partial = B._level_recovery(real, real.mean() + 0.23 * (real - real.mean()))
    check('partial tracker → beta matches the injected slope',
          abs(partial['beta'] - 0.23) < 1e-6, f"beta={partial['beta']:.4f}")

    check('fewer than 3 valid cases → NaN, not a spurious slope',
          math.isnan(B._level_recovery(np.array([1., 2.]), np.array([1., 2.]))['beta']))
    check('NaNs dropped pairwise (n counts only complete pairs)',
          B._level_recovery(np.array([np.nan, 1, 2, 3, 4]),
                            np.array([1, np.nan, 2, 3, 4]))['n'] == 3)

    # _organ_medians: HU medians per label, voxel floor, missing-organ NaN.
    mask = np.zeros((8, 8, 8)); mask[0:2] = 7; mask[2:4] = 52
    realv = np.ones((8, 8, 8)) * 100.; realv[2:4] = 130.
    genv = np.ones((8, 8, 8)) * 98.;  genv[2:4] = 120.
    om = B._organ_medians(realv, genv, mask,
                          {'aorta': 7, 'inferior_vena_cava': 52, 'liver': 99})
    check('organ medians read the right label in HU',
          om['aorta'] == (100., 98.) and om['inferior_vena_cava'] == (130., 120.),
          f'{om}')
    check('organ absent from the mask → NaN pair', math.isnan(om['liver'][0]))
    check('no organ map → empty dict (columns become NaN)',
          B._organ_medians(realv, genv, mask, None) == {})
    small = np.zeros((8, 8, 8)); small[0, 0, :] = 7          # 8 voxels < floor
    check('below the voxel floor → NaN (median too noisy)',
          math.isnan(B._organ_medians(realv, genv, small, {'aorta': 7})['aorta'][0]))


def _summ(name, **kw):
    base = dict(model=name, n=20, psnr=30.0, ssim=0.94, mae=0.01, mse=0.001, pcc=0.98,
                org_psnr=24.5, org_ssim=0.96, org_mae=0.03, body_psnr=25.5,
                body_mae=0.028, body_frac=0.32, phase_acc=1.0, gen_prob=0.98,
                feature_l1_hu=14.0, lev_beta=0.3, lev_varr=0.2, raps_hf=0.84,
                grad_w1=0.002, org_grad_w1=0.006, seam=1.36, zflicker=0.9,
                zaniso=1.05, lpips=float('nan'), fid=float('nan'))
    base.update(kw)
    return base


def test_model_family():
    print('\nmodel_family()')
    check('this-repo GAN default', B.model_family('l1_organ_groupnorm_s43')
          == 'UNet + PatchGAN (this repo)')
    check('diffusion by prefix', B.model_family('diff_hetero_nll') == 'Diffusion')
    check('external baseline by name', B.model_family('pix2pixhd_baseline')
          == 'External baseline')
    check('identity is reference/floor', B.model_family('identity_venous')
          == 'Reference / floor')
    # A multi-phase arm keeps its family after the '/phase' suffix is stripped.
    check('phase arm classified by its base run', B.model_family('diff_v/arterial')
          == 'Diffusion')


def test_rank_marks():
    print('\n_rank_marks() best/second, direction-aware, floors excluded')
    summ = [_summ('a', psnr=30.0, org_mae=0.030, lev_beta=0.30, raps_hf=0.84),
            _summ('b', psnr=30.8, org_mae=0.028, lev_beta=0.95, raps_hf=1.20),
            _summ('c', psnr=30.5, org_mae=0.029, lev_beta=0.50, raps_hf=0.90),
            _summ('identity_x', psnr=27.0, org_mae=0.055, lev_beta=0.0, raps_hf=1.36)]

    hi = B._rank_marks(summ, 'psnr', 'high')
    check('higher-better: best=max, second=next', hi.get('b') == 'best' and hi.get('c') == 'second')
    lo = B._rank_marks(summ, 'org_mae', 'low')
    check('lower-better: best=min, second=next', lo.get('b') == 'best' and lo.get('c') == 'second')
    one = B._rank_marks(summ, 'lev_beta', 'one')
    check('target-1: best=closest to 1 (0.95), second=0.50',
          one.get('b') == 'best' and one.get('c') == 'second')
    r1 = B._rank_marks(summ, 'raps_hf', 'one')
    check('target-1 both sides: 0.90 beats 0.84 and 1.20', r1.get('c') == 'best',
          f'{r1}')
    check('reference/floor never marked',
          all('identity_x' not in m for m in (hi, lo, one, r1)))
    check("unranked column (dir None) → no marks", B._rank_marks(summ, 'n', None) == {})


def test_master_table_categorised():
    print('\nmaster_table() categorised + highlighted')
    summ = [_summ('l1_only', psnr=30.4, feature_l1_hu=17.0),
            _summ('diff_hetero_nll', psnr=30.8, feature_l1_hu=13.0),
            _summ('identity_venous', psnr=27.5, feature_l1_hu=56.0)]
    out = []
    B.master_table(summ, out, perceptual=False)
    txt = '\n'.join(out)
    for cat in ('Image-level', 'Organ-level', 'Phase & level fidelity',
                'Detail-focused', 'Perceptual'):
        check(f'category present: {cat}', cat in txt)
    check('architecture family headers present',
          'UNet + PatchGAN (this repo)' in txt and 'Diffusion' in txt
          and 'Reference / floor' in txt)
    check('best featHU (diff, 13.0) is bolded', '**13.00**' in txt)
    check('floor row not bolded on PSNR', '**27.5' not in txt and '_27.5' not in txt)
    check('perceptual OFF → note, no LPIPS table row', 'Re-run `benchmark.py --perceptual`' in txt)
    # With perceptual on, the LPIPS/FID table renders instead of the note.
    out2 = []
    B.master_table([_summ('l1_only', lpips=0.24, fid=55.0),
                    _summ('diff_hetero_nll', lpips=0.19, fid=38.0)],
                   out2, perceptual=True, fid_backend='pytorch-fid')
    t2 = '\n'.join(out2)
    check('perceptual ON → LPIPS/FID column + best bolded',
          '| LPIPS | FID |' in t2 and '**0.1900**' in t2)


if __name__ == '__main__':
    print('=' * 70)
    print('BENCHMARK DISCOVERY / TILING / PAIRED TESTS')
    print('=' * 70)
    test_discover()
    test_read_tiling()
    test_paired_block_disjoint_cases()
    test_resolve_baseline()
    test_level_recovery()
    test_model_family()
    test_rank_marks()
    test_master_table_categorised()
    print('\n' + '=' * 70)
    if FAILS:
        print(f'FAILED ({len(FAILS)}): {", ".join(FAILS)}')
        sys.exit(1)
    print('ALL PASS')
