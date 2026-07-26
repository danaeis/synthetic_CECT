#!/usr/bin/env python
"""Verify that set_seed() actually pinned every RNG.

Two levels, run both:
  history.json  — fast, catches most misses
  checkpoint    — authoritative: if the weights match bit-for-bit, nothing
                  stochastic was left unseeded

Usage:
  python scripts/check_determinism.py same /tmp/s42a /tmp/s42b   # must MATCH
  python scripts/check_determinism.py diff /tmp/s42a /tmp/s43    # must DIFFER
"""
import json, sys
from pathlib import Path

def load_hist(d):
    p = Path(d) / 'history.json'
    if not p.exists():
        sys.exit(f"FAIL: {p} does not exist — did the run finish an epoch?")
    return json.loads(p.read_text())

def cmp_hist(a, b):
    """Returns list of differing keys ([] if identical)."""
    bad = []
    if set(a) != set(b):
        bad.append(f"key sets differ: {set(a) ^ set(b)}")
    for k in sorted(set(a) & set(b)):
        va, vb = a[k], b[k]
        if len(va) != len(vb):
            bad.append(f"{k}: length {len(va)} vs {len(vb)}")
        elif va != vb:
            i = next(j for j, (x, y) in enumerate(zip(va, vb)) if x != y)
            bad.append(f"{k}: first diff at epoch idx {i}: {va[i]!r} vs {vb[i]!r}")
    return bad

#def cmp_ckpt(da, db):
#    try:
#        import torch
#    except ImportError:
#        return None
#    ca = sorted(Path(da).glob('ckpt_ep*.pth'))
#    cb = sorted(Path(db).glob('ckpt_ep*.pth'))
#    if not ca or not cb:
#        return None
#    sa = torch.load(ca[-1], map_location='cpu')['gen_state']
#    sb = torch.load(cb[-1], map_location='cpu')['gen_state']
#    bad = []
#    for k in sa:
#        if k not in sb:
#            bad.append(f"{k}: missing"); continue
#        if not torch.equal(sa[k].float(), sb[k].float()):
#            d = (sa[k].float() - sb[k].float()).abs().max().item()
#            bad.append(f"{k}: max|delta| = {d:.3e}")
#    return bad


def cmp_ckpt(da, db):
    try:
        import torch
    except ImportError:
        return None
    ca = sorted(Path(da).glob('ckpt_ep*.pth'))
    cb = sorted(Path(db).glob('ckpt_ep*.pth'))
    if not ca or not cb:
        return None
    # weights_only=False: the checkpoint holds optimiser/scheduler/history
    # objects, not just tensors, so the future default would refuse to load it.
    A = torch.load(ca[-1], map_location='cpu', weights_only=False)
    B = torch.load(cb[-1], map_location='cpu', weights_only=False)
    bad = []
    for part in ('G_state', 'D_state'):          # D_state only exists for adversarial runs
        if part not in A or part not in B:
            continue
        sa, sb = A[part], B[part]
        for k in sa:
            if k not in sb:
                bad.append(f"{part}.{k}: missing"); continue
            if not torch.equal(sa[k].float(), sb[k].float()):
                d = (sa[k].float() - sb[k].float()).abs().max().item()
                bad.append(f"{part}.{k}: max|delta| = {d:.3e}")
    if A.get('epoch') != B.get('epoch'):
        bad.append(f"epoch: {A.get('epoch')} vs {B.get('epoch')}")
    return bad


mode, da, db = sys.argv[1], sys.argv[2], sys.argv[3]
hb = cmp_hist(load_hist(da), load_hist(db))
cb = cmp_ckpt(da, db)

print(f"history.json : {'IDENTICAL' if not hb else str(len(hb)) + ' differing keys'}")
for line in hb[:8]:
    print(f"   {line}")
if cb is None:
    print("checkpoint   : skipped (no ckpt_ep*.pth found, or torch unavailable)")
else:
    print(f"checkpoint   : {'IDENTICAL' if not cb else str(len(cb)) + ' differing tensors'}")
    for line in cb[:8]:
        print(f"   {line}")

differs = bool(hb) or bool(cb)
if mode == 'same':
    print("\nPASS: seeding is complete." if not differs else
          "\nFAIL: same seed produced different runs — an RNG is still unseeded.")
    sys.exit(1 if differs else 0)
else:
    print("\nPASS: the seed actually changes the run." if differs else
          "\nFAIL: different seeds produced identical runs — set_seed() is being "
          "called in the wrong place, or --seed never reached the config.")
    sys.exit(0 if differs else 1)
