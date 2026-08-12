#!/usr/bin/env python
"""
Is the DECODER + OUTPUT HEAD what stops the model from reaching the target levels?

The question. `UNetGenerator` ends `... -> _DecBlock(conv, norm, LeakyReLU) ->
Conv1x1 -> Sigmoid`. Three properties of that tail are each, on their face, a
candidate explanation for the ~8-15 HU per-organ floor and for the beta~0.22
level-tracking slope `scripts/audit_enhancement.py` measured:

  H1  SIGMOID SATURATION. dy/dz = y(1-y) collapses near 0 and 1, so under L1 the
      gradient on a bright voxel is a fraction of the gradient on a mid-grey one
      and the model should systematically undershoot enhancement.
  H2  WINDOW TRUNCATION. HU_MAX=400 clips the target itself, and a sigmoid can
      never emit exactly 1.0 anyway, so the top of the range is doubly
      unreachable.
  H3  NORMALISATION KILLS THE LEVEL. InstanceNorm/GroupNorm subtract a per-sample
      spatial mean, which is EXACTLY DC-invariant. The absolute HU level of the
      input is therefore destroyed at the first norm in enc1 and can never reach
      the head, and the final feature map handed to the 1x1 conv has its own
      per-channel mean pinned. `_norm('instance')` is `nn.InstanceNorm2d(ch)`,
      whose default is affine=FALSE — so the instance arms do not even have a
      learnable per-channel scale/shift at the last layer.

These are not equivalent and they imply different fixes, so this script measures
all three rather than arguing about them. PART A needs no data or checkpoint
(the properties are architectural); PART B needs only the generated + real
volumes a run already wrote, no checkpoint and no GPU.

WHAT PART B DECIDES. H1/H2 are claims about the TOP of the intensity range, so
they are testable by asking where the quantities actually being scored live:

  * If the per-organ medians sit near the middle of [0,1], the sigmoid is at its
    steepest there and H1 cannot be the mechanism, whatever else is wrong.
  * If the generated volume's upper percentiles fall well short of the real
    volume's, the head IS truncating and H1/H2 are live.
  * If a single free scalar offset per case removes most of the per-organ error,
    the decoder produces the right PATTERN at the wrong LEVEL — which is H3, and
    is fixed by conditioning or a norm change, not by touching the activation.
  * If that offset removes little, the error is per-organ and structural, and no
    change to the output head addresses it.

Usage:
    # architecture only — runs anywhere, no data
    python scripts/output_head_capacity.py --arch_only

    # full: architecture + the volumes a run already produced
    python scripts/output_head_capacity.py \
      --manifest ../out_synthesis_train/literature_baseline_l1_organ_groupnorm_s42/phase_infer/manifest.csv \
      --organ_map orgFeatXGB_CTPhase/retrain_out_full/ts_label_map_total.json \
      --out analysis/output_head.json
"""
import argparse
import csv
import json
import sys
import pathlib

# Repo modules live one level up (this file sits in scripts/ or tests/).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent /
                    'orgFeatXGB_CTPhase'))

import numpy as np

import metrics as M

RULE = '=' * 78


# ---------------------------------------------------------------------------
# PART A — architecture (no data, no checkpoint)
# ---------------------------------------------------------------------------

def part_a(base_ch: int = 16, patch: int = 64) -> dict:
    import torch
    import torch.nn as nn
    from models import UNetGenerator, _norm

    out = {}
    print(f'{RULE}\nPART A — what the architecture can represent\n{RULE}')

    # A1. Does the norm layer itself destroy the DC level? Exact, weight-free.
    print('\nA1. Is the normalisation layer DC-invariant?  max |f(x+c) - f(x)|, c=0.5')
    print('    (0 = the absolute level is destroyed and cannot reach any later layer)')
    torch.manual_seed(0)
    x = torch.randn(1, 8, 32, 32)
    a1 = {}
    for kind in ('instance', 'group', 'batch'):
        n = _norm(2, 8, kind).eval()
        with torch.no_grad():
            d = float((n(x + 0.5) - n(x)).abs().max())
        a1[kind] = d
        verdict = 'DC DESTROYED' if d < 1e-4 else 'DC preserved'
        print(f'      {kind:9s} {d:.3e}   {verdict}')
    out['norm_dc_shift'] = a1

    # A2. Same question for the whole generator. Conv weights are forced positive
    # so a DC offset actually propagates: with the default zero-mean init the DC
    # gain of a random conv is ~0 BY CHANCE, which would fake this result.
    print('\nA2. Whole-generator DC gain  d(mean output) / d(input offset)')
    print('    (conv weights forced positive so DC can propagate; ~0 = structural)')
    a2 = {}
    for kind in ('instance', 'group', 'batch'):
        torch.manual_seed(0)
        g = UNetGenerator(dims=2, base_channels=base_ch, norm=kind, dropout=0.0).eval()
        with torch.no_grad():
            for m in g.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    m.weight.abs_()
                    if m.bias is not None:
                        m.bias.zero_()
            xin = torch.rand(1, 1, patch, patch) * 0.4 + 0.2
            base = float(g(xin).mean())
            gains = [(float(g(xin + c).mean()) - base) / c for c in (0.05, 0.10, 0.20)]
        a2[kind] = {'base_mean': base, 'gains': gains}
        # A saturated base (0 or 1) makes the gain meaningless — flag it rather
        # than reporting "gain 0, therefore invariant".
        note = ('  [base saturated — gain uninformative]'
                if base > 0.999 or base < 0.001 else '')
        print(f'      {kind:9s} base={base:.4f}  gain={np.mean(gains):+.4f}{note}')
    out['generator_dc_gain'] = a2

    # A3. The head itself.
    torch.manual_seed(0)
    a3 = {}
    for kind in ('instance', 'group', 'batch'):
        g = UNetGenerator(dims=2, base_channels=base_ch, norm=kind, dropout=0.0)
        last = [m for m in g.dec1.modules()
                if isinstance(m, (nn.InstanceNorm2d, nn.GroupNorm, nn.BatchNorm2d))][-1]
        a3[kind] = {'last_norm': type(last).__name__,
                    'affine': bool(getattr(last, 'affine', False)),
                    'out_conv_params': sum(p.numel() for p in g.out_conv.parameters())}
    print('\nA3. The last norm before out_conv, and the head\'s size')
    for k, v in a3.items():
        print(f"      {k:9s} {v['last_norm']:16s} affine={str(v['affine']):5s} "
              f"out_conv params={v['out_conv_params']}")
    print('    affine=False means no learnable per-channel scale/shift at the very')
    print('    last layer — the level must come from out_conv\'s single bias.')
    out['head'] = a3
    return out


def sigmoid_profile(hu_min: float, hu_max: float) -> dict:
    """Where the sigmoid is steep, in HU. dy/dz = y(1-y), max 0.25 at y=0.5."""
    print(f'\nA4. Sigmoid gradient across the HU[{hu_min:.0f},{hu_max:.0f}] window')
    print(f"      {'unit y':>7}{'HU':>8}{'dy/dz':>9}{'vs best':>9}")
    rows = {}
    for y in (0.10, 0.30, 0.50, 0.60, 0.70, 0.85, 0.92, 0.95, 0.98):
        g = y * (1 - y)
        hu = y * (hu_max - hu_min) + hu_min
        rows[f'{y:.2f}'] = {'hu': hu, 'grad': g, 'x_weaker': 0.25 / g}
        print(f'      {y:7.2f}{hu:8.0f}{g:9.4f}{0.25 / g:8.1f}x')
    return rows


# ---------------------------------------------------------------------------
# PART B — what the trained model actually emits
# ---------------------------------------------------------------------------

def _load(p):
    import nibabel as nib
    return np.asanyarray(nib.load(p).dataobj).astype(np.float32)


def part_b(manifest: pathlib.Path, organ_map_json: pathlib.Path,
           hu_min: float, hu_max: float, max_cases=None) -> dict:
    from organ_features import ORGANS, features_from_mask

    full = json.loads(organ_map_json.read_text())
    organ_map = {o: int(full[o]) for o in ORGANS if o in full}

    with manifest.open() as f:
        cases = list(csv.DictReader(f))
    if max_cases:
        cases = cases[:max_cases]

    print(f'\n{RULE}\nPART B — what the trained model actually emits ({len(cases)} cases)'
          f'\n{RULE}')

    per_case = []
    for c in cases:
        try:
            gen, real, mask = (_load(c['gen_path']), _load(c['real_path']),
                               _load(c['mask_path']))
        except Exception as e:
            print(f"  skip {pathlib.Path(c.get('gen_path','?')).name}: {e}")
            continue
        if not (gen.shape == real.shape == mask.shape):
            continue

        g01, r01 = M.to_unit(gen, hu_min, hu_max), M.to_unit(real, hu_min, hu_max)
        body = M.body_mask(real).astype(bool)
        gb, rb = g01[body], r01[body]

        # B1/B2. Range usage and saturation, inside the body only — outside it
        # every voxel is clipped air and would dominate any percentile.
        pct = [50, 90, 99, 99.9]
        rec = {
            'case': pathlib.Path(c['real_path']).name,
            'gen_pct': dict(zip(map(str, pct), np.percentile(gb, pct).tolist())),
            'real_pct': dict(zip(map(str, pct), np.percentile(rb, pct).tolist())),
            'gen_max': float(gb.max()), 'real_max': float(rb.max()),
            'gen_frac_hi': float((gb > 0.98).mean()),
            'real_frac_hi': float((rb > 0.98).mean()),
            'gen_frac_lo': float((gb < 0.02).mean()),
            'real_frac_lo': float((rb < 0.02).mean()),
            # H2: how much of the real volume the HU window truncates before the
            # model ever sees it.
            'real_frac_above_window': float((real[body] > hu_max).mean()),
        }

        # B3. Per-organ level, and what one free scalar offset would buy. The
        # offset that minimises mean |error| is the MEDIAN of the per-organ
        # differences, not the mean — matching the L1 the table is scored in.
        f_gen = features_from_mask(gen, mask, organ_map)
        f_real = features_from_mask(np.clip(real, hu_min, hu_max), mask, organ_map)
        ok = np.isfinite(f_gen) & np.isfinite(f_real)
        d = f_real[ok] - f_gen[ok]
        rec['feat_l1_hu'] = float(np.mean(np.abs(d)))
        rec['best_global_offset_hu'] = float(np.median(d))
        rec['feat_l1_hu_after_offset'] = float(np.mean(np.abs(d - np.median(d))))
        per_case.append(rec)

    if not per_case:
        raise SystemExit('no cases scored')

    agg = lambda k: float(np.mean([r[k] for r in per_case]))
    print('\nB1. Range usage inside the body mask (unit domain, 1.0 = HU '
          f'{hu_max:.0f})')
    print(f"      {'percentile':>12}{'gen':>10}{'real':>10}{'gap':>10}")
    for p in ('50', '90', '99', '99.9'):
        gp = float(np.mean([r['gen_pct'][p] for r in per_case]))
        rp = float(np.mean([r['real_pct'][p] for r in per_case]))
        print(f'      {p:>12}{gp:10.4f}{rp:10.4f}{gp - rp:+10.4f}')
    print(f"      {'max':>12}{agg('gen_max'):10.4f}{agg('real_max'):10.4f}"
          f"{agg('gen_max') - agg('real_max'):+10.4f}")
    print('    A large negative gap at p99/p99.9 = the head is truncating the top '
          'of the range (H1/H2).')

    print('\nB2. Saturation and window truncation')
    print(f"      voxels > 0.98 :  gen {agg('gen_frac_hi'):.4%}   "
          f"real {agg('real_frac_hi'):.4%}")
    print(f"      voxels < 0.02 :  gen {agg('gen_frac_lo'):.4%}   "
          f"real {agg('real_frac_lo'):.4%}")
    print(f"      real HU above the window ceiling: "
          f"{agg('real_frac_above_window'):.4%}  (H2 bites only if this is large)")

    print('\nB3. Level vs pattern — what ONE free scalar per case would buy')
    b, a = agg('feat_l1_hu'), agg('feat_l1_hu_after_offset')
    off = float(np.mean([abs(r['best_global_offset_hu']) for r in per_case]))
    print(f'      featHU as scored                        : {b:6.2f} HU')
    print(f'      featHU after an oracle global offset    : {a:6.2f} HU')
    print(f'      removed by one scalar per case          : {b - a:6.2f} HU '
          f'({(b - a) / b:.0%})')
    print(f'      mean |offset| the oracle applied        : {off:6.2f} HU')
    print('    A large removed fraction = the decoder gets the PATTERN right and')
    print('    the LEVEL wrong (H3: a conditioning/normalisation problem).')
    print('    A small one = the error is per-organ, and the output head is not it.')

    return {'per_case': per_case,
            'summary': {'feat_l1_hu': b, 'feat_l1_hu_after_offset': a,
                        'frac_removed_by_global_offset': (b - a) / b if b else float('nan'),
                        'mean_abs_global_offset_hu': off,
                        'gen_p999': float(np.mean([r['gen_pct']['99.9'] for r in per_case])),
                        'real_p999': float(np.mean([r['real_pct']['99.9'] for r in per_case])),
                        'real_frac_above_window': agg('real_frac_above_window'),
                        'n': len(per_case)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', type=pathlib.Path, default=None)
    ap.add_argument('--organ_map', type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent.parent /
                            'orgFeatXGB_CTPhase' / 'retrain_out_full' /
                            'ts_label_map_total.json')
    ap.add_argument('--hu_min', type=float, default=-200.0)
    ap.add_argument('--hu_max', type=float, default=400.0)
    ap.add_argument('--base_ch', type=int, default=16,
                    help='width for the PART A probes; the DC properties are '
                         'width-independent, so a small net keeps it fast')
    ap.add_argument('--max_cases', type=int, default=None)
    ap.add_argument('--arch_only', action='store_true')
    ap.add_argument('--out', type=pathlib.Path, default=None)
    a = ap.parse_args()

    report = {'part_a': part_a(base_ch=a.base_ch)}
    report['part_a']['sigmoid_profile'] = sigmoid_profile(a.hu_min, a.hu_max)

    if not a.arch_only:
        if a.manifest is None:
            raise SystemExit('PART B needs --manifest (or pass --arch_only)')
        report['part_b'] = part_b(a.manifest, a.organ_map, a.hu_min, a.hu_max,
                                  a.max_cases)

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2))
        print(f'\n[written] {a.out}')


if __name__ == '__main__':
    main()
