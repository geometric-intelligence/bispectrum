#!/usr/bin/env python3
"""Print OrganMNIST3D parameter counts for normalized rerun budgets."""

from __future__ import annotations

from itertools import product

from model import build_model

TARGET = 400_000
MODELS = ['standard', 'max_pool', 'norm_pool', 'bispectrum']
CHANNELS = [(2, 4), (3, 6), (4, 8), (6, 12), (8, 16), (12, 24), (16, 32)]
HEAD_DIMS = [32, 48, 64, 96, 128]


def count_params(model_name: str, channels: tuple[int, int], head_dim: int) -> int:
    model = build_model(model_name, channels=channels, head_dim=head_dim)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    print(f'Target: {TARGET:,} params')
    print(f'{"Model":>12}  {"Channels":>10}  {"Head":>5}  {"Params":>10}  {"Delta":>9}')
    print('-' * 58)

    best: dict[str, tuple[tuple[int, int], int, int, int]] = {}
    for model_name in MODELS:
        head_dims = HEAD_DIMS if model_name == 'bispectrum' else [64]
        for channels, head_dim in product(CHANNELS, head_dims):
            n_params = count_params(model_name, channels, head_dim)
            delta = n_params - TARGET
            if model_name not in best or abs(delta) < best[model_name][3]:
                best[model_name] = (channels, head_dim, n_params, abs(delta))
                marker = ' <-- closest'
            else:
                marker = ''
            print(
                f'{model_name:>12}  {str(channels):>10}  {head_dim:>5}  '
                f'{n_params:>10,}  {delta:>+9,}{marker}'
            )
        print()

    print('Recommended closest configs:')
    print(f'{"Model":>12}  {"Channels":>10}  {"Head":>5}  {"Params":>10}')
    print('-' * 45)
    for model_name in MODELS:
        channels, head_dim, n_params, _ = best[model_name]
        print(f'{model_name:>12}  {str(channels):>10}  {head_dim:>5}  {n_params:>10,}')


if __name__ == '__main__':
    main()
