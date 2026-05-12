#!/usr/bin/env python3
"""Print Spherical MNIST parameter counts for normalized rerun budgets."""

from __future__ import annotations

from model import build_model

TARGET = 232_000
MODELS = ['standard', 'power_spectrum', 'bispectrum']
HIDDEN_VALUES = list(range(64, 1025, 32))
LMAX = 15
NLAT = 64
NLON = 128


def count_params(model_name: str, hidden: int) -> int:
    model = build_model(
        model_name,
        lmax=LMAX,
        nlat=NLAT,
        nlon=NLON,
        selective=True,
        hidden=hidden,
    )
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    print(f'Target: {TARGET:,} params | lmax={LMAX} | grid={NLAT}x{NLON}')
    print(f'{"Model":>18}  {"Hidden":>6}  {"Params":>10}  {"Delta":>9}')
    print('-' * 52)

    best: dict[str, tuple[int, int, int]] = {}
    for model_name in MODELS:
        hidden_values = [256] if model_name == 'standard' else HIDDEN_VALUES
        for hidden in hidden_values:
            n_params = count_params(model_name, hidden)
            delta = n_params - TARGET
            if model_name not in best or abs(delta) < best[model_name][2]:
                best[model_name] = (hidden, n_params, abs(delta))
                marker = ' <-- closest'
            else:
                marker = ''
            print(f'{model_name:>18}  {hidden:>6}  {n_params:>10,}  {delta:>+9,}{marker}')
        print()

    print('Recommended closest configs:')
    print(f'{"Model":>18}  {"Hidden":>6}  {"Params":>10}')
    print('-' * 40)
    for model_name in MODELS:
        hidden, n_params, _ = best[model_name]
        print(f'{model_name:>18}  {hidden:>6}  {n_params:>10,}')


if __name__ == '__main__':
    main()
