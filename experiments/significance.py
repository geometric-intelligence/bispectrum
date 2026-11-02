#!/usr/bin/env python3
"""Seed-level significance of the grid-figure comparisons.

For every cell of the 3x3 grid where the bispectrum shares an x-value with
another method (training-set size on the data-efficiency curves, channel width
on the OrganMNIST3D parameter curve, the full-data bars), report

* mean +/- sample std and the seed count for both arms,
* the difference of means (bispectrum minus the other method),
* the Welch t-test p-value (unequal variances, uncorrected for multiplicity),
* the seeds per arm needed for 80% power at alpha = 0.05 to detect the
  observed difference with the observed pooled spread
  (``2 (z_0.975 + z_0.8)^2 s_p^2 / delta^2``, normal approximation).

The last column is the practical one: it says whether a non-significant gap
is a "needs 2 more seeds" gap or a "would need 500 seeds" gap. Runs are
loaded through ``make_grid_figure`` so the same stale-run filter and method
protocol apply.

Usage:
    python significance.py                       # rotated-test metric (figure curves)
    python significance.py --canonical           # canonical-test metric (bar panel)
    python significance.py --output stats.md     # write Markdown instead of printing
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from make_grid_figure import (
    EXPERIMENTS_DIR,
    ORGAN3D_METHODS,
    PCAM_METHODS,
    SMNIST_METHODS,
    _canonical_mode,
    _expected_mode,
    _group_runs,
    _is_full_train,
    _load_runs,
    _metric,
)
from scipy import stats

REFERENCE: Final = 'bispectrum'
ALPHA: Final = 0.05
POWER: Final = 0.80
# 2 * (z_{1-alpha/2} + z_{power})^2, the per-arm sample-size constant for two
# equal arms under the normal approximation (~15.7 for alpha=0.05, power=0.8).
POWER_CONSTANT: Final = 2.0 * (stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER)) ** 2


@dataclass(frozen=True)
class Arm:
    values: tuple[float, ...]

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return float(np.mean(self.values))

    @property
    def std(self) -> float:
        return float(np.std(self.values, ddof=1)) if self.n > 1 else 0.0


@dataclass(frozen=True)
class Comparison:
    row: str
    cell: str
    method: str
    other: Arm
    reference: Arm

    @property
    def delta(self) -> float:
        return self.reference.mean - self.other.mean

    @property
    def p_value(self) -> float:
        if self.reference.n < 2 or self.other.n < 2:
            return math.nan
        result = stats.ttest_ind(self.reference.values, self.other.values, equal_var=False)
        return float(result.pvalue)

    @property
    def seeds_for_power(self) -> float:
        """Seeds per arm for 80% power at the observed effect and pooled spread."""
        if self.reference.n < 2 or self.other.n < 2:
            return math.nan
        pooled_var = (self.reference.std**2 + self.other.std**2) / 2.0
        if self.delta == 0.0:
            return math.inf
        if pooled_var == 0.0:
            return 2.0
        return math.ceil(POWER_CONSTANT * pooled_var / self.delta**2)


# --------------------------------------------------------------------------
# Cell extraction
# --------------------------------------------------------------------------

Cells = dict[str, dict[str, list[dict]]]  # cell label -> method -> runs


def _size_label(n: object) -> str:
    return f'n={int(n):,}' if isinstance(n, int | float) else f'n={n}'


def _pcam_cells(data_pareto_dir: Path, include_stale: bool) -> Cells:
    grouped = _group_runs(_load_runs(data_pareto_dir, include_stale), PCAM_METHODS)
    cells: Cells = defaultdict(lambda: defaultdict(list))
    for method, runs in grouped.items():
        for r in runs:
            cells[_size_label(r.get('train_examples'))][method].append(r)
    return cells


def _organ3d_cells(results_dir: Path, include_stale: bool) -> Cells:
    grouped = _group_runs(_load_runs(results_dir, include_stale), ORGAN3D_METHODS)
    cells: Cells = defaultdict(lambda: defaultdict(list))
    for method, runs in grouped.items():
        for r in runs:
            channels = tuple(r.get('channels', ()))
            if _is_full_train(r):
                cells[f'full, ch={channels}'][method].append(r)
            elif channels == (4, 8):
                cells[_size_label(r.get('train_examples'))][method].append(r)
    return cells


def _smnist_cells(results_dir: Path, include_stale: bool) -> Cells:
    runs = _load_runs(results_dir, include_stale)
    grouped = _group_runs(runs, SMNIST_METHODS, model_key='base_model')
    for r in runs:  # older results may lack base_model
        if 'base_model' not in r and r.get('model') in SMNIST_METHODS:
            method = r['model']
            if _canonical_mode(r.get('train_mode', 'C')) == _expected_mode(method):
                grouped[method].append(r)
    cells: Cells = defaultdict(lambda: defaultdict(list))
    for method, method_runs in grouped.items():
        for r in method_runs:
            cells[_size_label(r.get('train_examples'))][method].append(r)
    return cells


def _arm(runs: Iterable[dict], key: str, metric: str) -> Arm:
    return Arm(tuple(v for r in runs if (v := _metric(r, key, metric)) is not None))


def _compare_cells(
    row: str,
    cells: Cells,
    methods: Sequence[str],
    key: str,
    metric: str,
) -> list[Comparison]:
    out: list[Comparison] = []
    for cell in sorted(cells, key=_cell_sort_key):
        by_method = cells[cell]
        if REFERENCE not in by_method:
            continue
        reference = _arm(by_method[REFERENCE], key, metric)
        if reference.n == 0:
            continue
        for method in methods:
            if method == REFERENCE or method not in by_method:
                continue
            other = _arm(by_method[method], key, metric)
            if other.n == 0:
                continue
            out.append(Comparison(row, cell, method, other, reference))
    return out


def _cell_sort_key(label: str) -> tuple[int, float, str]:
    if label.startswith('n='):
        return (0, float(label[2:].replace(',', '')), label)
    return (1, 0.0, label)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _fmt_arm(arm: Arm, scale: float, digits: int) -> str:
    return f'{arm.mean * scale:.{digits}f} ± {arm.std * scale:.{digits}f} (n={arm.n})'


def _fmt_power(value: float) -> str:
    if math.isnan(value):
        return '—'
    if math.isinf(value):
        return '∞'
    return f'{int(value)}'


def _fmt_p(value: float) -> str:
    if math.isnan(value):
        return '—'
    marker = ' *' if value < ALPHA else ''
    return f'{value:.3f}{marker}'


def render_markdown(title: str, comparisons: Sequence[Comparison], scale: float, digits: int) -> str:
    lines = [
        f'### {title}',
        '',
        f'| cell | vs. | {REFERENCE} | other | Δ | Welch p | seeds/arm for {POWER:.0%} power |',
        '|---|---|---|---|---|---|---|',
    ]
    for c in comparisons:
        lines.append(
            f'| {c.cell} | {c.method} | {_fmt_arm(c.reference, scale, digits)} | '
            f'{_fmt_arm(c.other, scale, digits)} | {c.delta * scale:+.{digits}f} | '
            f'{_fmt_p(c.p_value)} | {_fmt_power(c.seeds_for_power)} |'
        )
    if len(lines) == 4:
        lines.append('| (no cells with a bispectrum arm) | | | | | | |')
    lines.append('')
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RowSpec:
    title: str
    methods: tuple[str, ...]
    canonical_key: str
    metric: str
    scale: float
    digits: int


ROW_SPECS: Final = {
    'pcam': RowSpec('PCam — test AUC', PCAM_METHODS, 'test_c', 'auc', 1.0, 3),
    'organ3d': RowSpec('OrganMNIST3D — test accuracy (%)', ORGAN3D_METHODS, 'test_c', 'accuracy', 100.0, 2),
    'spherical_mnist': RowSpec('Spherical MNIST — test accuracy (%)', SMNIST_METHODS, 'test_nr', 'accuracy', 100.0, 2),
}


def main() -> None:
    parser = argparse.ArgumentParser(description='Welch tests and seed-power for the grid comparisons')
    parser.add_argument(
        '--pcam_data_pareto_dir',
        type=Path,
        default=EXPERIMENTS_DIR / 'pcam' / 'pcam_results_data_pareto',
    )
    parser.add_argument(
        '--organ3d_dir',
        type=Path,
        default=EXPERIMENTS_DIR / 'organ3d' / 'organ3d_results',
    )
    parser.add_argument(
        '--smnist_dir',
        type=Path,
        default=EXPERIMENTS_DIR / 'spherical_mnist' / 'smnist_results',
    )
    parser.add_argument(
        '--canonical',
        action='store_true',
        help='Compare the canonical-test metric (bar panel) instead of the rotated-test metric.',
    )
    parser.add_argument(
        '--include_stale',
        action='store_true',
        help='Keep results.json files that predate the current protocol (see make_grid_figure).',
    )
    parser.add_argument('--output', type=Path, default=None, help='Write Markdown here instead of stdout.')
    args = parser.parse_args()

    rows: dict[str, Cells] = {
        'pcam': _pcam_cells(args.pcam_data_pareto_dir, args.include_stale),
        'organ3d': _organ3d_cells(args.organ3d_dir, args.include_stale),
        'spherical_mnist': _smnist_cells(args.smnist_dir, args.include_stale),
    }

    which = 'canonical test' if args.canonical else 'rotated (OOD) test'
    sections = [
        f'## Seed-level comparisons against `{REFERENCE}` — {which} metric',
        '',
        'Welch t-test, two-sided, uncorrected for multiple comparisons; `*` marks p < 0.05. '
        'Spread is the sample std over seeds (ddof=1). The last column is the per-arm seed count '
        'that would give 80% power for the observed effect and spread.',
        '',
    ]
    for name, spec in ROW_SPECS.items():
        key = spec.canonical_key if args.canonical else 'test_r'
        comparisons = _compare_cells(name, rows[name], spec.methods, key, spec.metric)
        sections.append(render_markdown(spec.title, comparisons, spec.scale, spec.digits))
    text = '\n'.join(sections)

    if args.output is not None:
        args.output.write_text(text)
        print(f'Wrote {args.output}')
    else:
        print(text)


if __name__ == '__main__':
    main()
