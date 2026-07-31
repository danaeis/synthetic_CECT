#!/usr/bin/env python
"""
Per-case contrast level, dumped once so training can condition on it.

Why this exists. `scripts/audit_enhancement.py` measured that the model recovers
only ~18% of case-to-case variation in organ enhancement (beta = 0.18) and that its
output is 5.7x under-dispersed — it emits roughly the training-set average level
because injection dose and bolus timing are not visible in a non-contrast scan.
Conditioning on the level tests how much of the residual that accounts for.

The level is read from the REAL CECT, so this file is ORACLE INFORMATION. That is
deliberate and it is why it lives in a separate, inspectable artefact instead of
being computed inside the training loop: at evaluation time the same trained model
is run three ways —

    oracle      feed the case's true value        -> the ceiling
    population  feed the training-set mean        -> must reproduce the baseline
    fixed       feed any clinician-chosen value   -> the deployable mode

and the REPORTABLE RESULT IS THE GAP between oracle and population, never the
oracle number on its own.

Also dumps per-organ medians of the NCCT (free — the volume is already open).
`scripts/probe_level_predictability.py` uses those to ask the sharper question:
can the CECT level be predicted from the NCCT at all? If it cannot, that is a
direct positive measurement of the aleatoric claim.

Standardisation statistics are computed on the TRAIN SPLIT ONLY. Using all cases
would leak test-set statistics into the conditioning scale.

Usage:
    python scripts/dump_levels.py                      # -> splits/levels.json
    python scripts/dump_levels.py --phases venous arterial
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

try:
    import nibabel as nib
except ImportError:                                          # pragma: no cover
    raise SystemExit("dump_levels.py needs nibabel (pip install nibabel)")

# Same organ set audit_enhancement.py reports on, so the two are directly
# comparable. Order is fixed: it defines the conditioning vector's layout and is
# written into the JSON so training and inference cannot disagree.
LEVEL_ORGANS: List[str] = [
    'aorta', 'portal_vein_and_splenic_vein', 'inferior_vena_cava', 'heart',
    'liver', 'pancreas', 'gallbladder', 'colon',
]
MIN_VOXELS = 64          # below this an organ median is too noisy to condition on


def _load(p: str) -> np.ndarray:
    return np.asanyarray(nib.load(p).dataobj).astype(np.float32)


def _organ_medians(vol: np.ndarray, mask: np.ndarray,
                   name_to_id: Dict[str, int], organs: List[str]) -> Dict[str, float]:
    """Median HU per organ. Median, not mean, because that is what the XGBoost
    phase model reads and therefore what featHU scores."""
    out = {}
    for o in organs:
        sel = mask == name_to_id[o]
        out[o] = float(np.median(vol[sel])) if sel.sum() >= MIN_VOXELS else float('nan')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--phases', nargs='+', default=None,
                    help='default: config target_phases')
    ap.add_argument('--organs', nargs='+', default=LEVEL_ORGANS)
    ap.add_argument('--label_map', type=Path,
                    default=Path(__file__).resolve().parent.parent /
                            'orgFeatXGB_CTPhase' / 'retrain_out_full' /
                            'ts_label_map_total.json')
    ap.add_argument('--out', type=Path, default=Path('splits/levels.json'))
    a = ap.parse_args()

    if not a.label_map.exists():
        raise SystemExit(f"label map not found: {a.label_map}")
    name_to_id = json.loads(a.label_map.read_text())
    organs = [o for o in a.organs if o in name_to_id]
    if missing := [o for o in a.organs if o not in name_to_id]:
        print(f"note: not in label map, skipped: {missing}")

    from config import train_config as cfg
    from dataset import find_pairs_and_split
    phases = a.phases or cfg.get('target_phases') or [cfg.get('target_phase', 'venous')]
    tr, va, te = find_pairs_and_split({**cfg, 'target_phases': phases})
    split_of = {}
    for name, ps in (('train', tr), ('val', va), ('test', te)):
        for p in ps:
            split_of[p['case_id']] = name
    pairs = tr + va + te
    print(f"phases {phases}: {len(pairs)} pairs over {len(split_of)} cases\n")

    cases: Dict[str, Dict] = {}
    n_ok = n_skip = 0
    for p in pairs:
        cid, ph = p['case_id'], p.get('phase', phases[0])
        seg = p.get('seg_path')
        if not seg:
            n_skip += 1
            print(f"  {cid} [{ph}]: no seg mask — skipped")
            continue
        try:
            tgt, mask = _load(p['target_path']), _load(seg)
            src = _load(p['source_path'])
        except Exception as e:                               # noqa: BLE001
            n_skip += 1
            print(f"  {cid} [{ph}]: {e}")
            continue
        if not (tgt.shape == mask.shape == src.shape):
            n_skip += 1
            print(f"  {cid} [{ph}]: shape mismatch — skipped")
            continue
        cases.setdefault(cid, {'split': split_of.get(cid)})[ph] = {
            'cect': _organ_medians(tgt, mask, name_to_id, organs),
            'ncct': _organ_medians(src, mask, name_to_id, organs),
        }
        n_ok += 1

    if not n_ok:
        raise SystemExit("no readable pairs — is this the training host?")

    # Standardisation from the TRAIN split only. Using every case would leak
    # test-set statistics into the conditioning scale.
    per_organ: Dict[str, List[float]] = {o: [] for o in organs}
    for cid, rec in cases.items():
        if rec.get('split') != 'train':
            continue
        for ph in phases:
            if ph in rec:
                for o in organs:
                    v = rec[ph]['cect'][o]
                    if np.isfinite(v):
                        per_organ[o].append(v)

    mean = [float(np.mean(per_organ[o])) if per_organ[o] else 0.0 for o in organs]
    std = [float(np.std(per_organ[o])) or 1.0 for o in organs]

    out = {
        'organs': organs,
        'phases': phases,
        # The population fallback IS the standardisation mean, so a missing organ
        # or 'population' mode both resolve to a standardised vector of zeros.
        'standardize': {'mean': mean, 'std': std, 'source_split': 'train'},
        'cases': cases,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))

    print(f"\n{n_ok} pairs written, {n_skip} skipped -> {a.out}")
    print(f"\n{'organ':<30}{'train mean':>12}{'train sd':>10}{'n':>6}")
    for o, m, s in zip(organs, mean, std):
        print(f"  {o:<28}{m:>12.1f}{s:>10.1f}{len(per_organ[o]):>6}")
    print("\nThis file is ORACLE information (read from the real CECT). Use it for "
          "oracle-vs-population\ngap measurement, never as a headline metric.")


if __name__ == '__main__':
    main()
