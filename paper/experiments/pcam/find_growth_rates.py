#!/usr/bin/env python3
"""Print parameter counts for all (model, growth_rate) combinations.

Helps pick per-model growth_rate values that match a target param budget.
"""
from __future__ import annotations

import sys

from model import build_model

TARGET = 100_000
MODELS = ['standard', 'norm', 'gate', 'fourier_elu', 'bispectrum']
GROWTH_RATES = range(2, 16)
GROUP = 'c8'
BLOCK_CONFIG = (4, 4, 4)


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
            marker = ' <-- closest' if m not in best or abs(delta) < best[m][2] else ''
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


if __name__ == '__main__':
    main()
