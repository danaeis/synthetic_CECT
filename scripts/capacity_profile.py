#!/usr/bin/env python
"""
Where does this generator actually spend its parameters, memory and time?

Motivation. Before adding an attention branch, it matters *where* the existing
capacity sits. At base_ch=64 the answer is lopsided: bottleneck (35.4%) + enc4
(26.6%) = 62% of all parameters live at the two coarsest scales, on 8x8 and
16x16 grids. Add dec4 and it is 79.7%. That is exactly where a transformer
bottleneck block would also go — so attention there *competes* with existing
capacity rather than complementing it, and any "attention helped" claim needs a
parameter-matched control (a wider CNN) to be believable.

The three cost numbers (VRAM, s/epoch, patch/s) exist so a proposed variant can
be priced before it is trained, not after.

Usage:
    python scripts/capacity_profile.py                     # config defaults
    python scripts/capacity_profile.py --base_ch 32 96     # compare widths
    python scripts/capacity_profile.py --no_bench          # params only, no GPU
    python scripts/capacity_profile.py --patch 256 --batch 4
"""
import argparse
import time

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from config import train_config
from models import UNetGenerator


def module_table(g: torch.nn.Module):
    """Per-top-level-child parameter counts, largest first."""
    total = sum(p.numel() for p in g.parameters())
    rows = [(name, sum(p.numel() for p in mod.parameters()))
            for name, mod in g.named_children()]
    rows = [r for r in rows if r[1] > 0]
    rows.sort(key=lambda r: -r[1])
    return rows, total


def benchmark(g, dims, patch, batch, device, n_iter=12):
    """Peak memory + throughput for a fwd/bwd step at this batch and patch size."""
    shape = ((batch, 1, patch, patch) if dims == 2
             else (batch, 1, train_config['patch_depth'], patch, patch))
    x = torch.randn(*shape, device=device)
    opt = torch.optim.Adam(g.parameters(), lr=1e-4)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # warmup, excluded from timing (first iterations pay allocator + autotune cost)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        g(x).mean().backward()
        opt.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        opt.zero_grad(set_to_none=True)
        g(x).mean().backward()
        opt.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_s = (time.perf_counter() - t0) / n_iter

    g.eval()
    with torch.no_grad():
        for _ in range(3):
            g(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            g(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        infer_s = (time.perf_counter() - t0) / n_iter
    g.train()

    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3
            if device.type == 'cuda' else float('nan'))
    return dict(peak_gb=peak,
                train_s_per_step=train_s,
                train_patch_s=batch / train_s,
                infer_patch_s=batch / infer_s)


def main():
    cfg = train_config
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base_ch', type=int, nargs='+',
                    default=[cfg['generator_base_channels']],
                    help='one or more widths to profile (default: config value)')
    ap.add_argument('--patch', type=int, default=None, help='default: config patch_size')
    ap.add_argument('--batch', type=int, default=None, help='default: config batch_size')
    ap.add_argument('--dims', type=int, default=None, choices=[2, 3])
    ap.add_argument('--norm', default=None, help='default: config generator_norm')
    ap.add_argument('--device', default=None)
    ap.add_argument('--no_bench', action='store_true',
                    help='parameter table only — no forward/backward, runs anywhere')
    a = ap.parse_args()

    patch = a.patch or (cfg['patch_size'] if isinstance(cfg['patch_size'], int)
                        else cfg['patch_size'][0])
    batch = a.batch or cfg['batch_size']
    dims = a.dims or cfg['dims']
    norm = a.norm or cfg.get('generator_norm', 'instance')
    device = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    print(f"dims={dims}  patch={patch}  batch={batch}  norm={norm}  device={device}")
    if not a.no_bench and device.type != 'cuda':
        print("NOTE: no CUDA — peak memory will be nan and timings are CPU-bound "
              "(not comparable to the training host).")

    # steps of 2 per level, 4 levels + bottleneck; the coarsest grid the
    # bottleneck actually sees. Worth printing because it is the token count an
    # attention block at that depth would operate on.
    print(f"bottleneck grid at patch {patch}: {patch // 16}x{patch // 16} "
          f"= {(patch // 16) ** 2} tokens\n")

    for bc in a.base_ch:
        g = UNetGenerator(dims=dims, base_channels=bc,
                          dropout=cfg['generator_dropout'], norm=norm).to(device)
        rows, total = module_table(g)

        print(f"── base_ch={bc} ──  {total:,} params ({total / 1e6:.2f} M)")
        print(f"   {'module':<14}{'params':>13}  {'share':>7}   cumulative")
        cum = 0
        for name, n in rows:
            cum += n
            print(f"   {name:<14}{n:>13,}  {100 * n / total:6.2f}%  "
                  f"{100 * cum / total:6.2f}%")

        coarse = sum(n for name, n in rows if name in ('bottleneck', 'enc4'))
        coarse3 = coarse + sum(n for name, n in rows if name == 'dec4')
        print(f"   bottleneck+enc4      : {100 * coarse / total:.2f}%  "
              f"(+dec4: {100 * coarse3 / total:.2f}%)")

        if not a.no_bench:
            b = benchmark(g, dims, patch, batch, device)
            print(f"   peak VRAM            : {b['peak_gb']:.2f} GB")
            print(f"   train step           : {b['train_s_per_step'] * 1e3:.1f} ms "
                  f"({b['train_patch_s']:.1f} patch/s)")
            print(f"   inference            : {b['infer_patch_s']:.1f} patch/s")
            n_steps = cfg['max_train_patches'] / batch
            print(f"   → est. s/epoch       : {n_steps * b['train_s_per_step']:.0f} s "
                  f"at max_train_patches={cfg['max_train_patches']}")
        print()

        del g
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
