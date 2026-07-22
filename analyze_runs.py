#!/usr/bin/env python
"""
Compare trained scenarios across every metric family, with significance testing.

Why this exists: the pixel metrics (PSNR/SSIM) are blind to the effect that
actually distinguishes these models. Between-run PSNR/SSIM differences sit in the
4th decimal — inside a single run's own epoch-to-epoch noise — while the paired
per-case phase-fidelity test shows the organ-weighted run beating the baseline at
t = -4.22. Reading only the headline table would have concluded "no effect".

So this script always reports, together:
  * the headline numbers,
  * the NOISE FLOOR they must be judged against,
  * paired per-case tests on the phase metrics,
  * and a cross-model per-case correlation, which tells you whether the residual
    error is attributable to the model at all or is intrinsic to the cases.

Reads only files already on disk — no GPU, no re-inference:
    history.json, organ_metrics.json, run_config.json,
    phase_infer/phase_eval_report.json

Usage:
    python analyze_runs.py --runs_dir ../out_synthesis_train
    python analyze_runs.py --runs_dir ../out_synthesis_train \
        --baseline literature_baseline_l1_only --out analysis/
"""

import argparse
import json
import math
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional

# Organs worth calling out: the phase classifier's top features, plus the
# vessels that carry the largest HU errors, plus the zero-weighted GI tract.
KEY_ORGANS = [
    'aorta', 'inferior_vena_cava', 'portal_vein_and_splenic_vein',
    'heart', 'liver', 'pancreas', 'pulmonary_vein',
    'small_bowel', 'colon', 'stomach',
]
# Fallback ids so runs written before the label map was wired (organs named
# `label_<id>`) still resolve.
ORGAN_IDS = {
    'aorta': 52, 'inferior_vena_cava': 63, 'portal_vein_and_splenic_vein': 64,
    'heart': 51, 'liver': 5, 'pancreas': 7, 'pulmonary_vein': 53,
    'small_bowel': 18, 'colon': 20, 'stomach': 6, 'duodenum': 19, 'esophagus': 15,
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class Run:
    def __init__(self, d: Path):
        self.dir = d
        self.name = d.name.replace('literature_baseline_', '')
        self.history = _load(d / 'history.json')
        self.config = _load(d / 'run_config.json')
        self.organs = _load(d / 'organ_metrics.json')
        self.phase = _load(d / 'phase_infer' / 'phase_eval_report.json')

    @property
    def ok(self) -> bool:
        return bool(self.history and self.history.get('epoch'))

    def best_idx(self) -> int:
        """Index of this run's own best epoch, by the metric it selected on."""
        h = self.history
        metric = (self.config or {}).get('selection_metric', 'val_loss')
        if metric == 'val_loss' or metric not in h:
            return h['val_loss'].index(min(h['val_loss']))
        return h[metric].index(max(h[metric]))

    def organs_at(self, epoch: int) -> Dict:
        if not self.organs:
            return {}
        rec = next((x for x in self.organs if x['epoch'] == epoch), None)
        return (rec or self.organs[-1])['organs']

    def organ_metric(self, organs: Dict, name: str, metric: str) -> Optional[float]:
        for k in (name, f"label_{ORGAN_IDS.get(name, -1)}"):
            if k in organs:
                return organs[k][metric]
        return None


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def paired_t(deltas: List[float]) -> tuple:
    """(mean, t-statistic, n_improved). t on the paired differences; these are
    the same 20 test cases scored by two models, so a paired test is the right
    one — it removes the between-case variance that dominates here."""
    n = len(deltas)
    if n < 2:
        return (0.0, 0.0, 0)
    m = st.mean(deltas)
    s = st.stdev(deltas)
    t = m / (s / math.sqrt(n)) if s > 0 else 0.0
    return (m, t, sum(1 for d in deltas if d < 0))


def pearson(a: List[float], b: List[float]) -> float:
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else 0.0


def sig(t: float) -> str:
    """Rough two-sided flag at n~20 (df~19): |t|>2.09 ~ p<.05, >2.86 ~ p<.01,
    >3.88 ~ p<.001."""
    a = abs(t)
    return '***' if a > 3.88 else '**' if a > 2.86 else '*' if a > 2.09 else 'ns'


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def headline(runs: List[Run], out: List[str]):
    out.append('\n' + '=' * 100)
    out.append('1. HEADLINE — each run at its OWN best-selection epoch')
    out.append('=' * 100)
    out.append(f"{'run':24s}{'ep':>4}{'PSNR':>8}{'SSIM':>9}{'oPSNR':>8}{'oSSIM':>9}"
               f"{'phase':>7}{'prob':>8}{'featHU':>8}")
    for r in runs:
        h = r.history
        i = r.best_idx()
        ph = (r.phase or {}).get('summary', {})
        row = (f"{r.name:24s}{int(h['epoch'][i]):4d}"
               f"{h['val_psnr'][i]:8.2f}{h['val_ssim'][i]:9.4f}"
               f"{h['val_org_psnr'][i]:8.2f}{h['val_org_ssim'][i]:9.4f}")
        row += (f"{ph['gen_phase_accuracy_vs_target']:7.2f}"
                f"{ph['mean_gen_target_prob']:8.4f}"
                f"{ph['mean_feature_l1_hu']:8.2f}") if ph else f"{'--':>7}{'--':>8}{'--':>8}"
        out.append(row)
    out.append('\n  oPSNR/oSSIM = organ-region. featHU = mean per-organ |HU error| vs real CECT.')
    out.append('  Judge every difference here against the noise floor in section 2.')


def noise_floor(runs: List[Run], out: List[str], last_n: int):
    out.append('\n' + '=' * 100)
    out.append(f'2. NOISE FLOOR — epoch-to-epoch std within each run (last {last_n} epochs)')
    out.append('=' * 100)
    out.append('  A between-run difference smaller than these is NOT a result.')
    out.append(f"\n{'run':24s}{'oSSIM std':>12}{'oSSIM range':>22}{'oPSNR std':>12}")
    for r in runs:
        h = r.history
        v = h['val_org_ssim'][-last_n:]
        p = h['val_org_psnr'][-last_n:]
        if len(v) < 3:
            continue
        out.append(f"{r.name:24s}{st.stdev(v):12.5f}"
                   f"   [{min(v):.5f},{max(v):.5f}]{st.stdev(p):12.4f}")
    spread = []
    for r in runs:
        i = r.best_idx()
        spread.append(r.history['val_org_ssim'][i])
    if len(spread) > 1:
        out.append(f"\n  Between-run spread of best oSSIM: {max(spread) - min(spread):.5f}")
        out.append('  Compare to the within-run std above before claiming any run is better.')


def phase_tests(runs: List[Run], base: Run, out: List[str]):
    out.append('\n' + '=' * 100)
    out.append(f'3. PAIRED PER-CASE TESTS vs "{base.name}"  (negative = improvement)')
    out.append('=' * 100)
    if not base.phase:
        out.append('  baseline has no phase_eval_report.json — skipping')
        return
    bc = base.phase['cases']
    out.append(f"\n{'run':24s}{'metric':30s}{'delta HU':>10}{'t':>8}{'sig':>5}{'better':>9}")
    for r in runs:
        if r is base or not r.phase:
            continue
        rc = r.phase['cases']
        if len(rc) != len(bc):
            out.append(f"  {r.name}: case count mismatch ({len(rc)} vs {len(bc)}) — skipped")
            continue
        rows = [('feature_l1_hu', [y['feature_l1_hu'] - x['feature_l1_hu']
                                   for x, y in zip(bc, rc)])]
        for o in KEY_ORGANS:
            d = [y['per_organ_abs_err_hu'].get(o, None) for y in rc]
            e = [x['per_organ_abs_err_hu'].get(o, None) for x in bc]
            pairs = [(a - b) for a, b in zip(d, e) if a is not None and b is not None]
            if len(pairs) >= 3:
                rows.append((o, pairs))
        first = True
        for nm, d in rows:
            m, t, w = paired_t(d)
            label = r.name if first else ''
            first = False
            out.append(f"{label:24s}{nm:30s}{m:+10.2f}{t:+8.2f}{sig(t):>5}{w:>5}/{len(d)}")
        out.append('')
    out.append('  sig: * p<.05  ** p<.01  *** p<.001 (two-sided, df~n-1)')


def correlation(runs: List[Run], out: List[str]):
    out.append('\n' + '=' * 100)
    out.append('4. CROSS-MODEL PER-CASE CORRELATION — is the error the model, or the data?')
    out.append('=' * 100)
    have = [r for r in runs if r.phase]
    if len(have) < 2:
        out.append('  need >=2 runs with phase_eval — skipping')
        return
    V = {r.name: [c['feature_l1_hu'] for c in r.phase['cases']] for r in have}
    names = list(V)
    n = len(V[names[0]])
    if any(len(v) != n for v in V.values()):
        out.append('  case counts differ across runs — skipping')
        return
    out.append(f"\n{'':22s}" + ''.join(f'{x[:10]:>12s}' for x in names))
    for a in names:
        out.append(f'{a[:22]:22s}' + ''.join(
            f'{pearson(V[a], V[b]):12.3f}' if a != b else f'{"-":>12}' for b in names))

    rs = [pearson(V[a], V[b]) for i, a in enumerate(names) for b in names[i + 1:]]
    between_case = st.stdev(V[names[0]])
    between_model = st.mean([max(V[k][i] for k in names) - min(V[k][i] for k in names)
                             for i in range(n)])
    out.append(f"\n  mean r = {st.mean(rs):.3f}  ->  r^2 = {st.mean(rs) ** 2:.3f}")
    out.append(f"  between-CASE  spread (std of per-case error): {between_case:6.2f} HU")
    out.append(f"  between-MODEL spread on the same case (mean):  {between_model:6.2f} HU")
    out.append('')
    if st.mean(rs) > 0.9:
        out.append(f"  => r^2={st.mean(rs)**2:.2f}: ~{st.mean(rs)**2*100:.0f}% of per-case error variance is shared by")
        out.append('     ALL models, i.e. intrinsic to the case, not the loss. The models fail on')
        out.append('     the same cases. Loss engineering is near its ceiling; look at the DATA')
        out.append('     (registration quality, contrast timing) for the remaining headroom.')


def schedules(runs: List[Run], out: List[str]):
    out.append('\n' + '=' * 100)
    out.append('5. SCHEDULES — lambda_l1 curriculum and LR restarts')
    out.append('=' * 100)
    for r in runs:
        h, c = r.history, (r.config or {})
        out.append(f"\n  {r.name}")
        lam = h.get('lambda_l1')
        if not lam:
            out.append('    lambda_l1: not recorded (run predates the curriculum)')
        else:
            u = sorted(set(lam))
            if len(u) == 1:
                out.append(f"    lambda_l1: CONSTANT {u[0]}  <-- NO-OP" +
                           ('  (use_l1_decay is ON but start==floor: the curriculum '
                            'did nothing)' if c.get('use_l1_decay') else ''))
            else:
                out.append(f"    lambda_l1: {max(lam)} -> {min(lam)}  "
                           f"(reaches floor at epoch {int(h['epoch'][lam.index(min(lam))])}, "
                           f"constant for {sum(1 for x in lam if x == min(lam))}/{len(lam)} epochs)")
        lr = h.get('lr_gen') or []
        rst = [int(h['epoch'][i]) for i in range(1, len(lr)) if lr[i] > lr[i - 1] * 2]
        if rst:
            out.append(f"    LR warm restarts at epochs {rst} "
                       f"(T0={c.get('cosine_t0')}, tmult={c.get('cosine_tmult')})")
            for e in rst:
                i = [j for j, x in enumerate(h['epoch']) if int(x) == e]
                if i and i[0] >= 2 and i[0] + 3 < len(h['val_org_ssim']):
                    j = i[0]
                    before, after = h['val_org_ssim'][j - 1], min(h['val_org_ssim'][j:j + 4])
                    out.append(f"      ep{e}: oSSIM {before:.5f} -> {after:.5f} "
                               f"({'COST' if after < before else 'ok'} {before - after:+.5f})")
        # plateau detection
        v = h['val_org_ssim']
        if len(v) > 10:
            peak = max(v)
            first = next(i for i, x in enumerate(v) if x >= peak - 0.001)
            out.append(f"    reaches within 0.001 of its peak oSSIM at epoch "
                       f"{int(h['epoch'][first])} of {int(h['epoch'][-1])} "
                       f"({len(v) - first - 1} later epochs added nothing)")


def per_organ(runs: List[Run], base: Run, out: List[str]):
    out.append('\n' + '=' * 100)
    out.append(f'6. PER-ORGAN SSIM vs "{base.name}" at each run\'s best epoch')
    out.append('=' * 100)
    bo = base.organs_at(int(base.history['epoch'][base.best_idx()]))
    if not bo:
        out.append('  no organ_metrics.json for the baseline — skipping')
        return
    others = [r for r in runs if r is not base and r.organs]
    # Weights come from whichever run actually defines them (the baseline usually
    # has none), so the tier each organ was trained under is visible here.
    wsrc = next((r for r in runs
                 if (r.config or {}).get('organ_weight_preset') == 'tiered'
                 and (r.config or {}).get('organ_weights')), None)
    w = ((wsrc.config or {}).get('organ_weights') or {}) if wsrc else {}
    if wsrc:
        out.append(f"\n  weight column = the tiered scheme from '{wsrc.name}'")
    out.append(f"\n{'organ':30s}{'wt':>4}{base.name[:9]:>10}" +
               ''.join(f'{r.name[:10]:>12s}' for r in others))
    for o in KEY_ORGANS:
        b = base.organ_metric(bo, o, 'ssim')
        if b is None:
            continue
        oid = ORGAN_IDS.get(o)
        raw = w.get(str(oid), w.get(oid))
        wt = f'{raw:g}' if raw is not None else '-'
        row = f"{o:30s}{str(wt):>4}{b:10.4f}"
        for r in others:
            ro = r.organs_at(int(r.history['epoch'][r.best_idx()]))
            v = r.organ_metric(ro, o, 'ssim')
            row += f'{v - b:+12.4f}' if v is not None else f'{"--":>12}'
        out.append(row)
    out.append('\n  Values are DELTAS vs baseline. Check them against per-organ jitter:')
    late = [x for x in (base.organs or []) if x['epoch'] >= int(base.history['epoch'][-1]) - 25]
    for o in ('aorta', 'heart', 'portal_vein_and_splenic_vein'):
        vals = [base.organ_metric(x['organs'], o, 'ssim') for x in late]
        vals = [v for v in vals if v is not None]
        if len(vals) > 3:
            out.append(f"    {o:32s} within-run std {st.stdev(vals):.4f}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs_dir', type=Path, default=Path('../out_synthesis_train'))
    ap.add_argument('--baseline', type=str, default=None,
                    help='run to compare against (default: the one with l1_only in its name)')
    ap.add_argument('--out', type=Path, default=None, help='also write the report here')
    ap.add_argument('--last_n', type=int, default=25,
                    help='epochs used for the noise-floor estimate')
    args = ap.parse_args()

    runs = [Run(d) for d in sorted(args.runs_dir.iterdir()) if d.is_dir()]
    runs = [r for r in runs if r.ok]
    if not runs:
        raise SystemExit(f'no runs with a history.json under {args.runs_dir}')

    if args.baseline:
        base = next((r for r in runs if args.baseline in (r.name, r.dir.name)), None)
        if base is None:
            raise SystemExit(f'baseline {args.baseline!r} not found among '
                             f'{[r.name for r in runs]}')
    else:
        base = next((r for r in runs if 'l1_only' in r.name), runs[0])

    out: List[str] = [f'Run comparison — {len(runs)} scenarios under {args.runs_dir}',
                      f'Baseline: {base.name}']
    headline(runs, out)
    noise_floor(runs, out, args.last_n)
    phase_tests(runs, base, out)
    correlation(runs, out)
    schedules(runs, out)
    per_organ(runs, base, out)

    text = '\n'.join(out)
    print(text)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'run_comparison.txt').write_text(text)
        print(f'\n[written] {args.out / "run_comparison.txt"}')


if __name__ == '__main__':
    main()
