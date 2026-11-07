#!/usr/bin/env python3
"""Spherical MNIST capacity sweep: rotated-test accuracy against MLP size.

Reads the runs written by run_capacity_sweep.sh and prints the power spectrum and
bispectrum columns as a Markdown table and as LaTeX rows (NR/R accuracy in %, mean
+- sample std over seeds). Stale runs are dropped by the same filter as the paper figures.

Usage:
    python analyze_capacity.py [--results_dir DIR] [--include_stale]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Final

EXPERIMENTS_DIR: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_DIR))

from make_grid_figure import _canonical_mode, _load_runs, _mean_std  # noqa: E402

MODELS: Final = ('power_spectrum', 'bispectrum')

type Row = tuple[int, float, float, int]


def capacity_rows(runs: list[dict]) -> dict[str, list[Row]]:
    """Group canonical-trained runs by (model, parameter count), sorted by size."""
    accuracies: dict[tuple[str, int], list[float]] = defaultdict(list)
    for run in runs:
        model = run.get('base_model')
        if model not in MODELS or _canonical_mode(run.get('train_mode', 'C')) != 'C':
            continue
        accuracies[(model, run['n_params'])].append(100.0 * run['test_r']['accuracy'])
    rows: dict[str, list[Row]] = {model: [] for model in MODELS}
    for (model, n_params), values in sorted(accuracies.items(), key=lambda item: item[0][1]):
        mean, std = _mean_std(values)
        rows[model].append((n_params, mean, std, len(values)))
    return rows


def format_params(n_params: int) -> str:
    if n_params < 10_000:
        return f'{n_params / 1000:.1f}K'
    return f'{round(n_params / 1000)}K'


def markdown_table(rows: dict[str, list[Row]]) -> str:
    lines = ['| Model | Params | NR/R (%) | Seeds |', '|---|---:|---:|---:|']
    for model in MODELS:
        for n_params, mean, std, n_seeds in rows[model]:
            lines.append(f'| {model} | {format_params(n_params)} | {mean:.2f} ± {std:.2f} | {n_seeds} |')
    return '\n'.join(lines)


def latex_rows(rows: dict[str, list[Row]]) -> str:
    """Power spectrum and bispectrum side by side, one width per line."""
    power, bispectrum = rows['power_spectrum'], rows['bispectrum']
    lines: list[str] = []
    for i in range(max(len(power), len(bispectrum))):
        cells: list[str] = []
        for column in (power, bispectrum):
            if i < len(column):
                n_params, mean, std, _ = column[i]
                cells += [format_params(n_params), f'${mean:.2f} \\pm {std:.2f}$']
            else:
                cells += ['', '']
        lines.append('    ' + ' & '.join(cells) + ' \\\\')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument(
        '--results_dir', type=Path, default=EXPERIMENTS_DIR / 'spherical_mnist' / 'smnist_results_capacity'
    )
    parser.add_argument('--include_stale', action='store_true')
    args = parser.parse_args()

    rows = capacity_rows(_load_runs(args.results_dir, args.include_stale))
    if not any(rows.values()):
        raise SystemExit(f'no capacity runs found under {args.results_dir}')
    print(markdown_table(rows))
    print('\n% LaTeX rows: power spectrum & bispectrum')
    print(latex_rows(rows))


if __name__ == '__main__':
    main()
