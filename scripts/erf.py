#!/usr/bin/env python
"""
How far can this generator actually see? (Effective receptive field)

Motivation. The theoretical receptive field of the 4-level U-Net is often quoted
at ~140 px, which is *larger* than the 128-px patch — read naively, that says
convolution already spans the whole patch and a long-range attention module has
nothing to add. But theoretical RF counts every path that could carry signal, not
the ones that do. The decision "is a 128 patch big enough for attention to
matter, or must we go to 256" was supposed to turn on the measured number.

It turns out the measurement does not settle that question, for a reason worth
knowing — see the MEASURED FINDING block below. Run it anyway: what it does
settle is that the ERF argument cannot be used against attention here, and it
gives an independent handle on the seam problem.

    ⚠ MEASURED FINDING (2026-07-26, random init, patch 256, base_ch 64,
      n_probe 2-3; ±1 px run-to-run from the random probes)

Method. Backpropagate from a SINGLE centre output pixel to the input and read
|d out_centre / d in|. Report two different things, because they disagree
sharply here:

  support radius — furthest input pixel with ANY nonzero gradient. This is the
      true architectural receptive field.
  mass radii (r50/r95/...) — square radius containing that fraction of total
      |grad| (Chebyshev distance, i.e. a (2r+1)x(2r+1) box; conv fields are
      square, so a box is the honest shape). This is where the influence
      actually concentrates.

Averaged over several random inputs, because instance/group norm make the
response input-dependent.

    norm       support r    r50     r95    border mass
    instance      128       14     105       0.130%     ← == patch/2: UNBOUNDED
    group         128        4      82       0.075%     ← == patch/2: UNBOUNDED
    batch          94        2       4       0.000%     ← bounded

    With instance/group norm the support radius equals patch/2 at EVERY patch
    size tried (r95 was 54/105/157 px at patch 128/256/384 — a constant 82% of
    patch/2). That is not a receptive field. Instance and group norm compute
    statistics over the patch's own spatial extent, so every input pixel reaches
    every output pixel through the normalisation, and the reach grows with the
    patch without limit. Only batch norm (eval, fixed running stats) shows the
    architecture's real field: support radius 94 px, i.e. a 189-px box — notably
    WIDER than the ~140 px usually quoted for a 4-level U-Net.

    Two consequences.

    1. Do not use ERF to argue "convolution already spans the patch, so
       attention has nothing to add." The apparent global reach under the
       default instance norm is a per-channel SCALAR (the patch's mean/std)
       leaking everywhere — it carries no spatial structure. Attention carries
       structured spatial relations, which this path cannot. The two are not
       substitutes and the ERF number does not adjudicate between them.
    2. This is the same mechanism as the seam problem, seen from the gradient
       side rather than the output side (config.py:130-141, and measured
       directly by scripts/norm_attribution.py: instance drift@shift32 13.98 HU
       vs group 6.83 HU). One cause, two symptoms.

    Caveat on random init: r95 = 4 px under batch norm while support = 94 px
    means gradient mass sits almost entirely on the shallowest enc1→dec1 skip.
    An untrained U-Net is dominated by its shortest path; training is what
    pushes signal down the deep path. So random-init mass radii are a LOWER
    BOUND on the trained ones — run --scenario_dir on a real checkpoint (on the
    training host; the results mirror keeps no .pth) before concluding anything
    about where a trained model looks.

Gradients are taken from the final conv PRE-sigmoid by default: a saturated
sigmoid scales the whole map by a near-zero scalar, which is numerically fragile
even though it leaves the normalised shape intact. `--post_sigmoid` to override.

Usage:
    # trained vs random init, side by side (the gap is itself informative)
    python scripts/erf.py --scenario_dir ../out_synthesis_train/literature_baseline_l1_organ_curriculum

    python scripts/erf.py --random_init                  # no checkpoint needed
    python scripts/erf.py --random_init --compare_norms  # reproduce the table above
    python scripts/erf.py --scenario_dir <run> --patch 256
"""
import argparse
import json
from pathlib import Path

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from config import train_config
from models import NORM_KINDS, UNetGenerator


def _grad_map(G, patch, device, pre_sigmoid=True, n_probe=8, seed=0):
    """Mean |d out[centre] / d in| over n_probe random inputs. Returns (H, W)."""
    # Tap the conv inside out_conv = Sequential(Conv, Sigmoid) so we can choose
    # to bypass the sigmoid's scalar saturation factor.
    tapped = {}
    handle = None
    if pre_sigmoid:
        conv = G.out_conv[0]

        def hook(_m, _inp, out):
            tapped['y'] = out
        handle = conv.register_forward_hook(hook)

    c = patch // 2
    acc = np.zeros((patch, patch), dtype=np.float64)
    g = torch.Generator(device='cpu').manual_seed(seed)
    try:
        for i in range(n_probe):
            x = torch.randn(1, 1, patch, patch, generator=g).to(device)
            x.requires_grad_(True)
            out = G(x)
            y = tapped['y'] if pre_sigmoid else out
            G.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad = None
            y[0, 0, c, c].backward()
            acc += x.grad.abs()[0, 0].double().cpu().numpy()
    finally:
        if handle is not None:
            handle.remove()
    return acc / n_probe


def _radii(erf, fracs=(0.5, 0.9, 0.95, 0.99)):
    """Mass radii + support radius, all as square (Chebyshev) box radii."""
    h, w = erf.shape
    cy, cx = h // 2, w // 2
    total = erf.sum()
    yy, xx = np.ogrid[:h, :w]
    cheb = np.maximum(np.abs(yy - cy), np.abs(xx - cx))
    rmax = int(cheb.max())

    if not np.isfinite(total) or total <= 0:
        return {**{str(f): float('nan') for f in fracs},
                'support': float('nan'), 'frac_nonzero': float('nan')}

    cum = np.array([erf[cheb <= r].sum() for r in range(rmax + 1)]) / total
    out = {}
    for f in fracs:
        idx = np.searchsorted(cum, f)
        out[str(f)] = float(idx) if idx <= rmax else float('inf')

    nz = cheb[erf > 0]
    out['support'] = float(nz.max()) if nz.size else 0.0
    out['frac_nonzero'] = float(100 * (erf > 0).mean())
    return out


def _report(label, erf, patch):
    rad = _radii(erf)
    half = patch // 2
    sup = rad['support']
    print(f"\n── {label} ──")
    print(f"   support   = {sup:6.1f} px   (box {2 * sup + 1:.0f} px, "
          f"{rad['frac_nonzero']:.1f}% of pixels nonzero)")
    for f in (0.5, 0.9, 0.95, 0.99):
        r = rad[str(f)]
        print(f"   r{int(f * 100):<3}      = {r:6.1f} px   "
              f"(box {2 * r + 1:.0f} px, {100 * r / half:5.1f}% of patch/2={half})")

    # The distinction that matters: is the support bounded by the ARCHITECTURE or
    # by the patch? Instance/group norm make it the latter, at any patch size.
    if sup >= half - 1:
        print(f"   → support == patch/2: reach is UNBOUNDED, set by the patch, not the\n"
              f"     architecture. Expected for instance/group norm, whose statistics\n"
              f"     span the patch. Mass radii below are contaminated by that scalar\n"
              f"     coupling and do NOT measure spatial feature mixing.")
    else:
        r95 = rad['0.95']
        verdict = ('LOCAL — attention has room at this patch size' if r95 < 0.55 * half
                   else 'BORDERLINE — conv reach ~= patch' if r95 < 0.95 * half
                   else 'GLOBAL — conv already spans the patch; enlarge patch first')
        print(f"   → architectural field is bounded at {sup:.0f} px. {verdict}")
    return rad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario_dir', default=None,
                    help='trained run dir (reads run_config.json + --ckpt_name)')
    ap.add_argument('--ckpt_name', default='best_model.pth')
    ap.add_argument('--random_init', action='store_true',
                    help='skip the checkpoint; profile an untrained model only')
    ap.add_argument('--patch', type=int, default=None, help='default: config/run patch_size')
    ap.add_argument('--base_ch', type=int, default=None)
    ap.add_argument('--norm', default=None)
    ap.add_argument('--n_probe', type=int, default=8,
                    help='random inputs to average over (InstanceNorm is input-dependent)')
    ap.add_argument('--post_sigmoid', action='store_true',
                    help='differentiate the sigmoid output instead of the final conv')
    ap.add_argument('--compare_norms', action='store_true',
                    help='profile instance/group/batch at identical init — isolates '
                         'how much of the reach is normalisation coupling vs conv')
    ap.add_argument('--device', default=None)
    ap.add_argument('--out_json', default=None)
    a = ap.parse_args()

    if not a.scenario_dir and not a.random_init and not a.compare_norms:
        ap.error('give --scenario_dir, --random_init, or --compare_norms')

    cfg = dict(train_config)
    if a.scenario_dir:
        rc = Path(a.scenario_dir) / 'run_config.json'
        if rc.exists():
            cfg.update(json.load(open(rc)))
            print(f"config from {rc}")
        else:
            print(f"WARNING: no run_config.json in {a.scenario_dir} — using config.py "
                  f"defaults, which may not match the checkpoint")

    patch = a.patch or (cfg['patch_size'] if isinstance(cfg['patch_size'], int)
                        else cfg['patch_size'][0])
    base_ch = a.base_ch or cfg['generator_base_channels']
    norm = a.norm or cfg.get('generator_norm', 'instance')
    device = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if cfg.get('dims', 2) != 2:
        ap.error(f"dims={cfg.get('dims')}: this script only handles the 2-D generator")

    print(f"patch={patch}  base_ch={base_ch}  norm={norm}  device={device}  "
          f"n_probe={a.n_probe}  grad from {'sigmoid' if a.post_sigmoid else 'final conv'}")
    print(f"theoretical RF of this 4-level U-Net: ~140 px (radius ~70)")

    results = {'patch': patch, 'base_ch': base_ch, 'norm': norm,
               'n_probe': a.n_probe, 'pre_sigmoid': not a.post_sigmoid}

    def build():
        return UNetGenerator(dims=2, base_channels=base_ch,
                             dropout=cfg.get('generator_dropout', 0.2),
                             norm=norm).to(device).eval()

    if a.compare_norms:
        # Same seed for every variant, so the ONLY difference is the norm layer.
        rows = {}
        for kind in NORM_KINDS:
            torch.manual_seed(0)
            Gn = UNetGenerator(dims=2, base_channels=base_ch,
                               dropout=cfg.get('generator_dropout', 0.2),
                               norm=kind).to(device).eval()
            e = _grad_map(Gn, patch, device, not a.post_sigmoid, a.n_probe)
            rows[kind] = _report(f'random init, norm={kind}', e, patch)
            del Gn
        results['compare_norms'] = rows
        print(f"\n   {'norm':<10}{'support':>9}{'r50':>7}{'r95':>7}{'nonzero':>10}")
        for kind, r in rows.items():
            flag = '  ← UNBOUNDED (== patch/2)' if r['support'] >= patch // 2 - 1 else ''
            print(f"   {kind:<10}{r['support']:>9.0f}{r['0.5']:>7.0f}"
                  f"{r['0.95']:>7.0f}{r['frac_nonzero']:>9.1f}%{flag}")
        print("\n   Reach that scales with the patch is normalisation statistics, not a\n"
              "   receptive field: it transmits a per-channel scalar, carrying no spatial\n"
              "   structure. See this file's header for why that voids the ERF-based\n"
              "   argument against attention.")
        if a.out_json:
            Path(a.out_json).write_text(json.dumps(results, indent=2))
            print(f"\nwrote {a.out_json}")
        if not a.scenario_dir:
            return

    # Random init: the architecture's reach with no training. The trained-vs-random
    # gap says whether training pushed signal down the deep path.
    torch.manual_seed(0)
    G = build()
    erf = _grad_map(G, patch, device, not a.post_sigmoid, a.n_probe)
    results['random_init'] = _report('random init', erf, patch)
    del G

    if a.scenario_dir:
        ckpt = Path(a.scenario_dir) / a.ckpt_name
        if not ckpt.exists():
            cands = sorted(Path(a.scenario_dir).glob('ckpt_ep*.pth'))
            if not cands:
                raise SystemExit(
                    f"no checkpoint at {ckpt} and no ckpt_ep*.pth in {a.scenario_dir}.\n"
                    f"Checkpoints are not kept in the results mirror — run this on the "
                    f"training host.")
            ckpt = cands[-1]
            print(f"\n{a.ckpt_name} missing; using {ckpt.name}")
        G = build()
        state = torch.load(ckpt, map_location=device, weights_only=False)
        G.load_state_dict(state['G_state'])
        G.eval()
        print(f"\nloaded {ckpt.name} (epoch {state.get('epoch', '?')})")
        erf_t = _grad_map(G, patch, device, not a.post_sigmoid, a.n_probe)
        results['trained'] = _report(f'trained ({ckpt.name})', erf_t, patch)
        results['ckpt'] = str(ckpt)
        results['epoch'] = state.get('epoch')

        r_rand = float(results['random_init']['0.95'])
        r_tr = float(results['trained']['0.95'])
        if np.isfinite(r_rand) and np.isfinite(r_tr):
            print(f"\ntrained r95 {r_tr:.1f} px vs random-init {r_rand:.1f} px "
                  f"({r_tr - r_rand:+.1f}) — a large positive gap means training "
                  f"pushed signal down the deep path")
        if norm in ('instance', 'group'):
            print(f"⚠ norm={norm}: the support radius is patch-bound, so these mass\n"
                  f"  radii mix spatial mixing with normalisation coupling. Re-run with\n"
                  f"  --compare_norms to see the split.")

    if a.out_json:
        Path(a.out_json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {a.out_json}")


if __name__ == '__main__':
    main()
