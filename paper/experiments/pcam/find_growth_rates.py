#!/usr/bin/env python3
"""Print parameter counts for all (model, growth_rate) combinations.

Helps pick per-model growth_rate values that match a target param budget.
Also sweeps bandlimit values for the so2_disk model.
"""
from __future__ import annotations

import sys

from model import build_model

TARGET = 100_000
MODELS = ['standard', 'norm', 'gate', 'fourier_elu', 'bispectrum']
GROWTH_RATES = range(2, 16)
GROUP = 'c8'
BLOCK_CONFIG = (4, 4, 4)

SO2_DISK_BANDLIMITS = [10, 15, 20, 25, 30, 40, 50, 60]


def main() -> None:
    best: dict[str, tuple[int, int, int]] = {}

    print(f'Target: {TARGET:,} params | Group: {GROUP} | Blocks: {BLOCK_CONFIG}')
    print(f'{"Model":>15}  {"gr":>3}  {"Params":>10}  {"Delta":>8}')
    print('-' * 45)

    for m in MODELS:
        for gr in GROWTH_RATES:
            model = build_model(m, GROUP, gr, BLOCK_CONFIG)
            n = sum(p.numel() for p in model.parameters() if p.requires_grad)
            delta = n - TARGET
            if m not in best or abs(delta) < best[m][2]:
                best[m] = (gr, n, abs(delta))
                marker = ' <-- closest'
            else:
                marker = ''
            print(f'{m:>15}  {gr:>3}  {n:>10,}  {delta:>+8,}{marker}')
        print()

    print('=' * 45)
    print('RECOMMENDED (closest to target):')
    print(f'{"Model":>15}  {"gr":>3}  {"Params":>10}')
    print('-' * 35)
    for m in MODELS:
        gr, n, _ = best[m]
        print(f'{m:>15}  {gr:>3}  {n:>10,}')

    print()
    print('=' * 60)
    print('SO2onDisk (so2_disk) — bandlimit sweep')
    print(f'Target: {TARGET:,} params (MLP auto-sized)')
    print(f'{"bandlimit":>10}  {"bispec_out":>10}  {"features":>10}  {"Params":>10}')
    print('-' * 50)

    best_bl: tuple[float, int, int] | None = None
    for bl in SO2_DISK_BANDLIMITS:
        model = build_model('so2_disk', bandlimit=float(bl), target_params=TARGET)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_feat = model._n_features
        bispec_out = model.bispec.output_size
        delta = abs(n - TARGET)
        if best_bl is None or delta < best_bl[2]:
            best_bl = (bl, n, delta)
            marker = ' <-- closest'
        else:
            marker = ''
        print(f'{bl:>10}  {bispec_out:>10}  {n_feat:>10}  {n:>10,}{marker}')

    if best_bl is not None:
        print(f'\nRecommended bandlimit: {best_bl[0]} ({best_bl[1]:,} params)')


if __name__ == '__main__':
    main()
