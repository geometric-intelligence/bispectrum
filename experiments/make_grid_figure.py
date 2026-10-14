#!/usr/bin/env python3
"""Build the 3x3 experiment grid figure (bars + param/data efficiency curves).

Rows: PCam (C8), OrganMNIST3D (octahedral O), Spherical MNIST (SO(3)).
Columns:
    1. OOD rotation robustness — canonical vs rotated test, bar pairs.
    2. Parameter efficiency    — rotated-test metric vs trainable params.
    3. Data efficiency         — rotated-test metric vs training examples.

Protocol (consistent across rows):
    - "Aug. CNN" baselines are non-equivariant CNNs trained with
      G-augmentation (train_mode R); invariant models are C-trained.
    - All line plots report the rotated (OOD) test metric, mean +/- std
      over seeds.
    - Cohen et al. (2018) S2CNN is shown as a published-reference dashed
      line in the Spherical MNIST row (cited, not re-run).

Usage:
    # Real data (after syncing results dirs from the GPU machines):
    python make_grid_figure.py

    # Layout preview with deterministic mock data:
    python make_grid_figure.py --mock
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
LegendHandle = Line2D | Patch

EXPERIMENTS_DIR: Final = Path(__file__).parent
PANEL_SIZE: Final = (4.0, 3.0)

COHEN_S2CNN_NRR: Final = 0.94  # Cohen et al. (2018), NR/R accuracy, published.


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodStyle:
    label: str
    color: str
    marker: str


@dataclass(frozen=True)
class ReferenceLine:
    value: float
    label: str
    linestyle: str = '--'


@dataclass(frozen=True)
class BarPanel:
    methods: tuple[str, ...]
    original: FloatArray
    ood: FloatArray
    original_error: FloatArray
    ood_error: FloatArray
    ylabel: str
    ylim: tuple[float, float]
    reference: ReferenceLine | None = None


@dataclass(frozen=True)
class CurveSeries:
    method: str
    x: FloatArray
    y: FloatArray
    error: FloatArray


@dataclass(frozen=True)
class CurvePanel:
    series: tuple[CurveSeries, ...]
    ylabel: str
    ylim: tuple[float, float]
    xticks: tuple[float, ...] | None = None
    xticklabels: tuple[str, ...] | None = None
    reference: ReferenceLine | None = None


@dataclass(frozen=True)
class RowData:
    bar: BarPanel
    params: CurvePanel
    data: CurvePanel


GridData = tuple[RowData, RowData, RowData]


METHOD_STYLE: Final[dict[str, MethodStyle]] = {
    # Non-equivariant CNN baselines, trained with G-augmentation.
    'standard': MethodStyle('Aug. CNN', '#888888', 's'),
    'standard_3d': MethodStyle('Aug. 3D CNN', '#888888', 's'),
    'standard_s2': MethodStyle('Aug. CNN', '#888888', 's'),
    # Equivariant-pooling baselines.
    'norm': MethodStyle('NormReLU', '#2196F3', 'o'),
    'gate': MethodStyle('Gated', '#FF9800', '^'),
    'fourier_elu': MethodStyle('Fourier-ELU', '#9C27B0', 'D'),
    'max_pool': MethodStyle('Max Pool', '#00897B', 'P'),
    # Incomplete second-order invariants (shared color family across rows).
    'norm_pool': MethodStyle('Norm pool', '#5C6BC0', 'v'),
    'power_spectrum': MethodStyle('Power spectrum', '#5C6BC0', 'v'),
    # Complete invariant (ours).
    'bispectrum': MethodStyle('Bispectrum', '#E53935', '*'),
}

ROW_METHODS: Final[tuple[tuple[str, ...], ...]] = (
    ('standard', 'norm', 'gate', 'fourier_elu', 'norm_pool', 'bispectrum'),
    ('standard_3d', 'norm_pool', 'max_pool', 'bispectrum'),
    ('standard_s2', 'power_spectrum', 'bispectrum'),
)

ROW_LABELS: Final = (
    r'PCam — $C_8$ rotations',
    r'OrganMNIST3D — octahedral $O$',
    r'Spherical MNIST — $\mathrm{SO}(3)$',
)

COLUMN_LABELS: Final = (
    'OOD rotation robustness',
    'Parameter efficiency',
    'Data efficiency',
)

PANEL_FILENAMES: Final = (
    ('grid_r1c1_pcam_ood.pdf', 'grid_r1c2_pcam_params.pdf', 'grid_r1c3_pcam_data.pdf'),
    ('grid_r2c1_organ3d_ood.pdf', 'grid_r2c2_organ3d_params.pdf', 'grid_r2c3_organ3d_data.pdf'),
    ('grid_r3c1_smnist_ood.pdf', 'grid_r3c2_smnist_params.pdf', 'grid_r3c3_smnist_data.pdf'),
)

COHEN_REFERENCE: Final = ReferenceLine(COHEN_S2CNN_NRR, r'Cohen S$^2$CNN (published)')

CAPTION_TEXT: Final = """\
Consistent comparison of invariant pooling strategies across three
group-structured benchmarks: PatchCamelyon (planar C8 rotations, test AUC),
OrganMNIST3D (octahedral rotations, test accuracy), and Spherical MNIST
(SO(3) rotations, test accuracy). Columns: (left) canonical vs rotated
(OOD) test performance as paired bars; (middle) parameter efficiency;
(right) data efficiency. Line plots report the rotated-test metric,
mean +/- std over 3 seeds. "Aug. CNN" denotes the non-equivariant CNN
baseline trained with G-augmentation; all invariant models are trained on
canonical (non-augmented) data. "Norm pool" (finite groups) and "Power
spectrum" (SO(3)) are incomplete second-order invariant baselines with
matched backbones; the bispectrum is the complete invariant. The dashed
line marks the published Spherical CNN result of Cohen et al. (2018)
(NR/R = 0.94), cited rather than re-run. We include equivariant
architectures in every row: the NormReLU, Gated, Fourier-ELU, Max-Pool,
and Norm-Pool variants are G-equivariant networks differing only in
their invariant map. The S2CNN reference appears only in the Spherical
MNIST row because it is the canonical published architecture designed
specifically for spherical images; rows 1-2 have no comparably canonical
external baseline, and their equivariant variants already fill that role.
"""


# --------------------------------------------------------------------------
# Results loading helpers
# --------------------------------------------------------------------------


def _load_runs(root: Path) -> list[dict]:
    """Recursively load every results.json under *root*."""
    runs: list[dict] = []
    if not root.exists():
        print(f'WARNING: results dir not found: {root}')
        return runs
    for p in sorted(root.rglob('results.json')):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _metric(run: dict, key: str, metric: str) -> float | None:
    """Extract run[key][metric], returning None when absent or empty."""
    block = run.get(key)
    if not block:
        return None
    value = block.get(metric)
    return float(value) if value is not None else None


def _expected_mode(method: str) -> str:
    """Training protocol per method: Aug. CNN baselines R, invariant models C."""
    return 'R' if method.startswith('standard') else 'C'


def _canonical_mode(mode: str) -> str:
    return 'C' if mode in ('C', 'NR') else 'R'


@dataclass(frozen=True)
class Aggregate:
    canonical_mean: float
    canonical_std: float
    ood_mean: float
    ood_std: float
    n_params: float
    n_seeds: int


def _aggregate(
    runs: list[dict],
    canonical_key: str,
    metric: str,
) -> Aggregate | None:
    """Aggregate canonical and rotated metrics over a list of seed runs."""
    canon = [v for r in runs if (v := _metric(r, canonical_key, metric)) is not None]
    ood = [v for r in runs if (v := _metric(r, 'test_r', metric)) is not None]
    if not canon or not ood:
        return None
    c_mean, c_std = _mean_std(canon)
    o_mean, o_std = _mean_std(ood)
    return Aggregate(
        canonical_mean=c_mean,
        canonical_std=c_std,
        ood_mean=o_mean,
        ood_std=o_std,
        n_params=float(np.mean([r['n_params'] for r in runs])),
        n_seeds=len(runs),
    )


def _build_bar_panel(
    methods: tuple[str, ...],
    aggregates: dict[str, Aggregate],
    ylabel: str,
    ylim: tuple[float, float],
    reference: ReferenceLine | None = None,
) -> BarPanel:
    present = tuple(m for m in methods if m in aggregates)
    missing = [m for m in methods if m not in aggregates]
    if missing:
        print(f'WARNING: bar panel missing methods {missing}')
    return BarPanel(
        methods=present,
        original=np.asarray([aggregates[m].canonical_mean for m in present]),
        ood=np.asarray([aggregates[m].ood_mean for m in present]),
        original_error=np.asarray([aggregates[m].canonical_std for m in present]),
        ood_error=np.asarray([aggregates[m].ood_std for m in present]),
        ylabel=ylabel,
        ylim=ylim,
        reference=reference,
    )


def _build_curve_panel(
    methods: tuple[str, ...],
    points: dict[str, list[tuple[float, float, float]]],
    ylabel: str,
    ylim: tuple[float, float],
    xticks: tuple[float, ...] | None = None,
    xticklabels: tuple[str, ...] | None = None,
    reference: ReferenceLine | None = None,
) -> CurvePanel:
    """Build a curve panel from per-method (x, y_mean, y_std) points."""
    series: list[CurveSeries] = []
    for method in methods:
        pts = sorted(points.get(method, []))
        if not pts:
            print(f'WARNING: curve panel missing method {method}')
            continue
        xs, ys, es = zip(*pts, strict=True)
        series.append(
            CurveSeries(
                method,
                np.asarray(xs, dtype=np.float64),
                np.asarray(ys, dtype=np.float64),
                np.asarray(es, dtype=np.float64),
            )
        )
    return CurvePanel(
        series=tuple(series),
        ylabel=ylabel,
        ylim=ylim,
        xticks=xticks,
        xticklabels=xticklabels,
        reference=reference,
    )


def _group_runs(
    runs: list[dict],
    methods: tuple[str, ...],
    model_key: str = 'model',
) -> dict[str, list[dict]]:
    """Group runs by method, keeping only protocol-conforming train modes."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        method = r.get(model_key)
        if method not in methods:
            continue
        if _canonical_mode(r.get('train_mode', 'C')) != _expected_mode(method):
            continue
        grouped[method].append(r)
    return grouped


# --------------------------------------------------------------------------
# Experiment loaders
# --------------------------------------------------------------------------

PCAM_METHODS: Final = ('standard', 'norm', 'gate', 'fourier_elu', 'norm_pool', 'bispectrum')
ORGAN3D_METHODS: Final = ('standard', 'max_pool', 'norm_pool', 'bispectrum')
SMNIST_METHODS: Final = ('standard', 'power_spectrum', 'bispectrum')


def _is_full_train(run: dict) -> bool:
    size = run.get('train_size')
    return size is None or size <= 0


def load_pcam(pareto_dir: Path, data_pareto_dir: Path) -> RowData:
    """Row 1: PCam. Bars from matched-budget full-data runs, curves from sweeps."""
    pareto_runs = _group_runs(_load_runs(pareto_dir), PCAM_METHODS)
    data_runs = _group_runs(_load_runs(data_pareto_dir), PCAM_METHODS)

    # Bars: matched ~100K-param configs trained on the full training set.
    aggregates: dict[str, Aggregate] = {}
    for method, runs in data_runs.items():
        full = [r for r in runs if _is_full_train(r)]
        by_seed = defaultdict(list)
        for r in full:
            by_seed[r['seed']].append(r)
        agg = _aggregate([rs[0] for rs in by_seed.values()], 'test_c', 'auc')
        if agg is not None:
            aggregates[method] = agg

    bar = _build_bar_panel(
        PCAM_METHODS,
        aggregates,
        ylabel='Test AUC',
        ylim=(0.82, 0.97),
    )

    # Param curves: pareto sweep grouped by growth rate.
    param_points: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for method, runs in pareto_runs.items():
        by_gr = defaultdict(list)
        for r in runs:
            by_gr[r.get('growth_rate')].append(r)
        for gr_runs in by_gr.values():
            agg = _aggregate(gr_runs, 'test_c', 'auc')
            if agg is not None:
                param_points[method].append((agg.n_params, agg.ood_mean, agg.ood_std))

    params = _build_curve_panel(
        PCAM_METHODS,
        param_points,
        ylabel='Rotated test AUC',
        ylim=(0.85, 0.97),
    )

    # Data curves: data-pareto sweep grouped by training-set size.
    data_points: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for method, runs in data_runs.items():
        by_size = defaultdict(list)
        for r in runs:
            by_size[r.get('train_examples')].append(r)
        for size, size_runs in by_size.items():
            agg = _aggregate(size_runs, 'test_c', 'auc')
            if agg is not None:
                data_points[method].append((float(size), agg.ood_mean, agg.ood_std))

    data = _build_curve_panel(
        PCAM_METHODS,
        data_points,
        ylabel='Rotated test AUC',
        ylim=(0.6, 0.98),
    )
    return RowData(bar=bar, params=params, data=data)


def load_organ3d(results_dir: Path) -> RowData:
    """Row 2: OrganMNIST3D. Bars from ch(4,8) full-data runs, curves from sweeps."""
    grouped = _group_runs(_load_runs(results_dir), ORGAN3D_METHODS)

    aggregates: dict[str, Aggregate] = {}
    param_points: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    data_points: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

    for method, runs in grouped.items():
        full = [r for r in runs if _is_full_train(r)]
        subset = [r for r in runs if not _is_full_train(r)]

        # Bars: default (4, 8) channel config, full training set.
        base = [r for r in full if tuple(r.get('channels', ())) == (4, 8)]
        agg = _aggregate(base, 'test_c', 'accuracy')
        if agg is not None:
            aggregates[method] = agg

        # Param curve: one point per channel config (full training set).
        by_channels = defaultdict(list)
        for r in full:
            by_channels[tuple(r.get('channels', ()))].append(r)
        for ch_runs in by_channels.values():
            ch_agg = _aggregate(ch_runs, 'test_c', 'accuracy')
            if ch_agg is not None:
                param_points[method].append((ch_agg.n_params, ch_agg.ood_mean, ch_agg.ood_std))

        # Data curve: (4, 8) channels across training-set sizes + full point.
        by_size = defaultdict(list)
        for r in subset:
            if tuple(r.get('channels', ())) == (4, 8):
                by_size[r.get('train_examples')].append(r)
        if base:
            by_size[base[0].get('train_examples')] = base
        for size, size_runs in by_size.items():
            size_agg = _aggregate(size_runs, 'test_c', 'accuracy')
            if size_agg is not None:
                data_points[method].append((float(size), size_agg.ood_mean, size_agg.ood_std))

    remap = {'standard': 'standard_3d'}
    aggregates = {remap.get(k, k): v for k, v in aggregates.items()}
    param_points = {remap.get(k, k): v for k, v in param_points.items()}
    data_points = {remap.get(k, k): v for k, v in data_points.items()}

    bar = _build_bar_panel(
        ROW_METHODS[1],
        aggregates,
        ylabel='Test accuracy',
        ylim=(0.0, 0.88),
    )
    params = _build_curve_panel(
        ROW_METHODS[1],
        param_points,
        ylabel='Rotated test accuracy',
        ylim=(0.3, 0.86),
    )
    data = _build_curve_panel(
        ROW_METHODS[1],
        data_points,
        ylabel='Rotated test accuracy',
        ylim=(0.05, 0.82),
    )
    return RowData(bar=bar, params=params, data=data)


def load_smnist(results_dir: Path, capacity_dir: Path) -> RowData:
    """Row 3: Spherical MNIST. Bars + data from main sweep, params from capacity."""
    main_runs = _load_runs(results_dir)
    capacity_runs = _load_runs(capacity_dir)

    # Capacity runs use run_label 'model_h{width}'; recover the base model.
    grouped_main = _group_runs(main_runs, SMNIST_METHODS, model_key='base_model')
    for r in main_runs:  # older results may lack base_model
        if 'base_model' not in r and r.get('model') in SMNIST_METHODS:
            method = r['model']
            if _canonical_mode(r.get('train_mode', 'C')) == _expected_mode(method):
                grouped_main[method].append(r)

    aggregates: dict[str, Aggregate] = {}
    data_points: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

    for method, runs in grouped_main.items():
        full = [r for r in runs if _is_full_train(r)]
        subset = [r for r in runs if not _is_full_train(r)]

        agg = _aggregate(full, 'test_nr', 'accuracy')
        if agg is not None:
            aggregates[method] = agg

        by_size = defaultdict(list)
        for r in subset:
            by_size[r.get('train_examples')].append(r)
        if full:
            by_size[full[0].get('train_examples')] = full
        for size, size_runs in by_size.items():
            size_agg = _aggregate(size_runs, 'test_nr', 'accuracy')
            if size_agg is not None:
                data_points[method].append((float(size), size_agg.ood_mean, size_agg.ood_std))

    # Param curve: capacity sweep grouped by run label (one label per width).
    param_points: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    by_label = defaultdict(list)
    for r in capacity_runs:
        base = r.get('base_model')
        if base in SMNIST_METHODS and _canonical_mode(r.get('train_mode', 'C')) == 'C':
            by_label[(base, r.get('model'))].append(r)
    for (base, _label), label_runs in by_label.items():
        agg = _aggregate(label_runs, 'test_nr', 'accuracy')
        if agg is not None:
            param_points[base].append((agg.n_params, agg.ood_mean, agg.ood_std))

    # The Aug. CNN has a fixed architecture: single point from the main sweep.
    if 'standard' in aggregates:
        std = aggregates['standard']
        param_points['standard'].append((std.n_params, std.ood_mean, std.ood_std))

    remap = {'standard': 'standard_s2'}
    aggregates = {remap.get(k, k): v for k, v in aggregates.items()}
    param_points = {remap.get(k, k): v for k, v in param_points.items()}
    data_points = {remap.get(k, k): v for k, v in data_points.items()}

    bar = _build_bar_panel(
        ROW_METHODS[2],
        aggregates,
        ylabel='Test accuracy',
        ylim=(0.0, 1.08),
        reference=COHEN_REFERENCE,
    )
    params = _build_curve_panel(
        ROW_METHODS[2],
        param_points,
        ylabel='Rotated test accuracy',
        ylim=(0.1, 1.0),
        reference=COHEN_REFERENCE,
    )
    data = _build_curve_panel(
        ROW_METHODS[2],
        data_points,
        ylabel='Rotated test accuracy',
        ylim=(0.05, 1.0),
        reference=COHEN_REFERENCE,
    )
    return RowData(bar=bar, params=params, data=data)


# --------------------------------------------------------------------------
# Mock data (layout preview only)
# --------------------------------------------------------------------------


def _array(values: list[float]) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def mock_grid_data() -> GridData:
    """Deterministic mock numbers exercising the exact real-data layout."""
    pcam = RowData(
        bar=BarPanel(
            methods=ROW_METHODS[0],
            original=_array([0.896, 0.942, 0.941, 0.945, 0.930, 0.941]),
            ood=_array([0.861, 0.941, 0.940, 0.944, 0.929, 0.941]),
            original_error=_array([0.010, 0.004, 0.009, 0.004, 0.007, 0.004]),
            ood_error=_array([0.013, 0.004, 0.008, 0.004, 0.007, 0.004]),
            ylabel='Test AUC',
            ylim=(0.82, 0.97),
        ),
        params=CurvePanel(
            series=(
                CurveSeries(
                    'standard',
                    _array([30_000, 102_000, 267_000, 582_000, 786_000]),
                    _array([0.880, 0.861, 0.870, 0.858, 0.862]),
                    _array([0.012, 0.013, 0.011, 0.013, 0.012]),
                ),
                CurveSeries(
                    'norm',
                    _array([69_000, 110_000, 222_000, 372_000, 791_000]),
                    _array([0.927, 0.941, 0.924, 0.911, 0.901]),
                    _array([0.007, 0.004, 0.008, 0.011, 0.012]),
                ),
                CurveSeries(
                    'gate',
                    _array([136_000, 218_000, 440_000, 741_000, 1_580_000]),
                    _array([0.940, 0.938, 0.935, 0.930, 0.924]),
                    _array([0.009, 0.008, 0.007, 0.009, 0.010]),
                ),
                CurveSeries(
                    'fourier_elu',
                    _array([69_000, 110_000, 222_000, 372_000, 790_000]),
                    _array([0.935, 0.944, 0.943, 0.937, 0.926]),
                    _array([0.005, 0.004, 0.004, 0.006, 0.008]),
                ),
                CurveSeries(
                    'norm_pool',
                    _array([69_000, 110_000, 222_000, 372_000, 790_000]),
                    _array([0.921, 0.929, 0.928, 0.922, 0.915]),
                    _array([0.008, 0.007, 0.006, 0.008, 0.009]),
                ),
                CurveSeries(
                    'bispectrum',
                    _array([80_000, 128_000, 258_000, 433_000, 920_000]),
                    _array([0.934, 0.941, 0.943, 0.941, 0.943]),
                    _array([0.005, 0.004, 0.004, 0.004, 0.004]),
                ),
            ),
            ylabel='Rotated test AUC',
            ylim=(0.85, 0.96),
        ),
        data=CurvePanel(
            series=(
                CurveSeries(
                    'standard',
                    _array([100, 500, 2_500, 12_500, 262_144]),
                    _array([0.660, 0.735, 0.805, 0.861, 0.930]),
                    _array([0.032, 0.024, 0.019, 0.013, 0.007]),
                ),
                CurveSeries(
                    'norm',
                    _array([100, 500, 2_500, 12_500, 262_144]),
                    _array([0.720, 0.805, 0.873, 0.941, 0.953]),
                    _array([0.028, 0.020, 0.018, 0.004, 0.004]),
                ),
                CurveSeries(
                    'gate',
                    _array([100, 500, 2_500, 12_500, 262_144]),
                    _array([0.735, 0.820, 0.884, 0.940, 0.952]),
                    _array([0.027, 0.018, 0.015, 0.009, 0.005]),
                ),
                CurveSeries(
                    'fourier_elu',
                    _array([100, 500, 2_500, 12_500, 262_144]),
                    _array([0.715, 0.810, 0.876, 0.944, 0.954]),
                    _array([0.031, 0.019, 0.016, 0.004, 0.004]),
                ),
                CurveSeries(
                    'norm_pool',
                    _array([100, 500, 2_500, 12_500, 262_144]),
                    _array([0.700, 0.790, 0.860, 0.928, 0.940]),
                    _array([0.030, 0.021, 0.017, 0.006, 0.005]),
                ),
                CurveSeries(
                    'bispectrum',
                    _array([100, 500, 2_500, 12_500, 262_144]),
                    _array([0.770, 0.850, 0.912, 0.941, 0.956]),
                    _array([0.025, 0.017, 0.025, 0.004, 0.004]),
                ),
            ),
            ylabel='Rotated test AUC',
            ylim=(0.6, 0.98),
            xticks=(100, 500, 2_500, 12_500, 262_144),
            xticklabels=('100', '500', '2.5K', '12.5K', 'full'),
        ),
    )

    organ3d = RowData(
        bar=BarPanel(
            methods=ROW_METHODS[1],
            original=_array([0.601, 0.568, 0.730, 0.726]),
            ood=_array([0.576, 0.568, 0.730, 0.726]),
            original_error=_array([0.017, 0.099, 0.033, 0.027]),
            ood_error=_array([0.021, 0.099, 0.033, 0.027]),
            ylabel='Test accuracy',
            ylim=(0.0, 0.88),
        ),
        params=CurvePanel(
            series=(
                CurveSeries(
                    'standard_3d',
                    _array([16_000, 60_000, 230_000]),
                    _array([0.576, 0.610, 0.640]),
                    _array([0.021, 0.020, 0.022]),
                ),
                CurveSeries(
                    'norm_pool',
                    _array([375_000, 1_520_000, 6_100_000]),
                    _array([0.568, 0.590, 0.600]),
                    _array([0.099, 0.080, 0.075]),
                ),
                CurveSeries(
                    'max_pool',
                    _array([374_000, 1_500_000, 6_000_000]),
                    _array([0.730, 0.743, 0.785]),
                    _array([0.033, 0.015, 0.032]),
                ),
                CurveSeries(
                    'bispectrum',
                    _array([463_000, 1_700_000, 6_300_000]),
                    _array([0.726, 0.745, 0.685]),
                    _array([0.027, 0.006, 0.039]),
                ),
            ),
            ylabel='Rotated test accuracy',
            ylim=(0.3, 0.86),
        ),
        data=CurvePanel(
            series=(
                CurveSeries(
                    'standard_3d',
                    _array([50, 100, 250, 500, 971]),
                    _array([0.115, 0.155, 0.320, 0.465, 0.576]),
                    _array([0.026, 0.031, 0.036, 0.031, 0.021]),
                ),
                CurveSeries(
                    'norm_pool',
                    _array([50, 100, 250, 500, 971]),
                    _array([0.150, 0.230, 0.390, 0.500, 0.568]),
                    _array([0.060, 0.070, 0.080, 0.090, 0.099]),
                ),
                CurveSeries(
                    'max_pool',
                    _array([50, 100, 250, 500, 971]),
                    _array([0.189, 0.280, 0.500, 0.640, 0.730]),
                    _array([0.035, 0.040, 0.045, 0.038, 0.033]),
                ),
                CurveSeries(
                    'bispectrum',
                    _array([50, 100, 250, 500, 971]),
                    _array([0.332, 0.430, 0.580, 0.680, 0.726]),
                    _array([0.040, 0.038, 0.035, 0.030, 0.027]),
                ),
            ),
            ylabel='Rotated test accuracy',
            ylim=(0.05, 0.82),
            xticks=(50, 100, 250, 500, 971),
            xticklabels=('50', '100', '250', '500', 'full'),
        ),
    )

    smnist = RowData(
        bar=BarPanel(
            methods=ROW_METHODS[2],
            original=_array([0.460, 0.792, 0.950]),
            ood=_array([0.230, 0.790, 0.951]),
            original_error=_array([0.012, 0.010, 0.001]),
            ood_error=_array([0.015, 0.010, 0.001]),
            ylabel='Test accuracy',
            ylim=(0.0, 1.08),
            reference=COHEN_REFERENCE,
        ),
        params=CurvePanel(
            series=(
                CurveSeries('standard_s2', _array([185_000]), _array([0.230]), _array([0.015])),
                CurveSeries(
                    'power_spectrum',
                    _array([3_700, 11_500, 39_000, 144_000, 550_000]),
                    _array([0.740, 0.765, 0.777, 0.786, 0.792]),
                    _array([0.012, 0.008, 0.001, 0.007, 0.010]),
                ),
                CurveSeries(
                    'bispectrum',
                    _array([25_000, 52_000, 108_000, 232_000, 529_000]),
                    _array([0.915, 0.932, 0.944, 0.951, 0.951]),
                    _array([0.007, 0.005, 0.003, 0.001, 0.002]),
                ),
            ),
            ylabel='Rotated test accuracy',
            ylim=(0.1, 1.0),
            reference=COHEN_REFERENCE,
        ),
        data=CurvePanel(
            series=(
                CurveSeries(
                    'standard_s2',
                    _array([100, 500, 2_500, 12_500, 60_000]),
                    _array([0.130, 0.155, 0.185, 0.212, 0.230]),
                    _array([0.008, 0.008, 0.009, 0.010, 0.015]),
                ),
                CurveSeries(
                    'power_spectrum',
                    _array([100, 500, 2_500, 12_500, 60_000]),
                    _array([0.420, 0.580, 0.690, 0.750, 0.777]),
                    _array([0.018, 0.015, 0.012, 0.006, 0.001]),
                ),
                CurveSeries(
                    'bispectrum',
                    _array([100, 500, 2_500, 12_500, 60_000]),
                    _array([0.670, 0.840, 0.920, 0.946, 0.951]),
                    _array([0.016, 0.012, 0.006, 0.002, 0.001]),
                ),
            ),
            ylabel='Rotated test accuracy',
            ylim=(0.05, 1.0),
            xticks=(100, 500, 2_500, 12_500, 60_000),
            xticklabels=('100', '500', '2.5K', '12.5K', 'full'),
            reference=COHEN_REFERENCE,
        ),
    )

    return (pcam, organ3d, smnist)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def configure_style() -> None:
    """Apply the shared Illustrator-friendly matplotlib style."""
    plt.rcParams.update(
        {
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
            'font.size': 9,
            'axes.labelsize': 10,
            'axes.linewidth': 0.65,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'xtick.major.width': 0.6,
            'ytick.major.width': 0.6,
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            'lines.linewidth': 1.6,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.04,
        }
    )


def style_axes(ax: Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#D9D9D9', linewidth=0.5, alpha=0.8)
    ax.tick_params(axis='x', pad=2)
    ax.tick_params(axis='y', pad=2)


def add_reference(ax: Axes, reference: ReferenceLine | None) -> None:
    if reference is None:
        return
    ax.axhline(
        reference.value,
        color='#666666',
        linestyle=reference.linestyle,
        linewidth=1.0,
        zorder=1,
    )


def draw_bar_panel(ax: Axes, panel: BarPanel) -> None:
    """Draw one canonical-vs-rotated grouped bar panel."""
    x = np.arange(len(panel.methods), dtype=np.float64)
    width = 0.35

    for index, method in enumerate(panel.methods):
        style = METHOD_STYLE[method]
        ax.bar(
            x[index] - width / 2,
            panel.original[index],
            width,
            yerr=panel.original_error[index],
            color=style.color,
            edgecolor=style.color,
            linewidth=0.8,
            capsize=2,
            error_kw={'elinewidth': 0.8, 'capthick': 0.8},
            zorder=3,
        )
        ax.bar(
            x[index] + width / 2,
            panel.ood[index],
            width,
            yerr=panel.ood_error[index],
            color=style.color,
            edgecolor=style.color,
            linewidth=0.8,
            hatch='////',
            alpha=0.42,
            capsize=2,
            error_kw={'elinewidth': 0.8, 'capthick': 0.8},
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_STYLE[method].label for method in panel.methods], rotation=24, ha='right')
    ax.set_ylabel(panel.ylabel)
    ax.set_ylim(panel.ylim)
    if 'accuracy' in panel.ylabel.lower():
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f'{value:.0%}'))
    add_reference(ax, panel.reference)
    style_axes(ax)


def draw_curve_panel(ax: Axes, panel: CurvePanel, *, xlabel: str) -> None:
    """Draw one parameter- or data-efficiency curve panel."""
    for series in panel.series:
        style = METHOD_STYLE[series.method]
        line_style = '-' if len(series.x) > 1 else 'none'
        ax.errorbar(
            series.x,
            series.y,
            yerr=series.error,
            color=style.color,
            marker=style.marker,
            linestyle=line_style,
            markersize=6.5 if style.marker == '*' else 4.5,
            markeredgewidth=0.7,
            markeredgecolor='white',
            capsize=2,
            elinewidth=0.7,
            zorder=3,
        )

    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(panel.ylabel)
    ax.set_ylim(panel.ylim)
    if panel.xticks is not None:
        ax.set_xticks(panel.xticks)
    if panel.xticklabels is not None:
        ax.set_xticklabels(panel.xticklabels)
    if 'accuracy' in panel.ylabel.lower():
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f'{value:.0%}'))
    add_reference(ax, panel.reference)
    style_axes(ax)


def draw_panel(ax: Axes, row: RowData, column: int) -> None:
    if column == 0:
        draw_bar_panel(ax, row.bar)
    elif column == 1:
        draw_curve_panel(ax, row.params, xlabel='Trainable parameters')
    else:
        draw_curve_panel(ax, row.data, xlabel='Training examples')


def method_legend_handle(method: str) -> Line2D:
    style = METHOD_STYLE[method]
    return Line2D(
        [0],
        [0],
        color=style.color,
        marker=style.marker,
        linewidth=1.6,
        markersize=7 if style.marker == '*' else 5,
        markeredgecolor='white',
        markeredgewidth=0.7,
        label=style.label,
    )


def row_legend_handles(row_index: int) -> list[LegendHandle]:
    handles: list[LegendHandle] = [method_legend_handle(method) for method in ROW_METHODS[row_index]]
    handles.extend(
        [
            Patch(facecolor='#777777', edgecolor='#777777', label='Canonical test'),
            Patch(
                facecolor='#D8D8D8',
                edgecolor='#777777',
                hatch='////',
                linewidth=0.8,
                label='Rotated test (OOD)',
            ),
        ]
    )
    if row_index == 2:
        handles.append(
            Line2D(
                [0],
                [0],
                color='#666666',
                linestyle='--',
                linewidth=1.0,
                label=COHEN_REFERENCE.label,
            )
        )
    return handles


def save_panels(grid: GridData, output_dir: Path) -> None:
    """Save title-free, legend-free vector panels plus per-row legends."""
    for row_index, row in enumerate(grid):
        for column in range(3):
            fig, ax = plt.subplots(figsize=PANEL_SIZE, constrained_layout=True)
            draw_panel(ax, row, column)
            fig.savefig(output_dir / PANEL_FILENAMES[row_index][column])
            plt.close(fig)

        handles = row_legend_handles(row_index)
        height = 0.30 * len(handles) + 0.25
        fig = plt.figure(figsize=(2.45, height))
        fig.legend(
            handles=handles,
            labels=[handle.get_label() for handle in handles],
            loc='center left',
            bbox_to_anchor=(0.02, 0.5),
            frameon=False,
            handlelength=2.1,
            handletextpad=0.8,
            borderaxespad=0.0,
            labelspacing=0.7,
            fontsize=9,
        )
        fig.savefig(output_dir / f'legend_row{row_index + 1}.pdf')
        plt.close(fig)


def save_contact_sheet(grid: GridData, output_dir: Path, *, mock: bool) -> None:
    """Save an assembled 3x3 sheet with column headers and side legends."""
    fig = plt.figure(figsize=(15.2, 9.6), constrained_layout=False)
    gridspec = fig.add_gridspec(
        3,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.58),
        left=0.075,
        right=0.985,
        bottom=0.07,
        top=0.91,
        wspace=0.34,
        hspace=0.42,
    )

    axes: list[list[Axes]] = []
    for row_index, row in enumerate(grid):
        row_axes: list[Axes] = []
        for column in range(3):
            ax = fig.add_subplot(gridspec[row_index, column])
            draw_panel(ax, row, column)
            row_axes.append(ax)
        axes.append(row_axes)

        legend_ax = fig.add_subplot(gridspec[row_index, 3])
        legend_ax.axis('off')
        handles = row_legend_handles(row_index)
        legend_ax.legend(
            handles=handles,
            labels=[handle.get_label() for handle in handles],
            loc='center left',
            frameon=False,
            handlelength=2.0,
            handletextpad=0.7,
            labelspacing=0.55,
            fontsize=8.5,
        )

    for column, label in enumerate(COLUMN_LABELS):
        position = axes[0][column].get_position()
        fig.text(
            (position.x0 + position.x1) / 2,
            0.945,
            label,
            ha='center',
            va='center',
            fontsize=13,
            fontweight='medium',
        )

    for row_index, label in enumerate(ROW_LABELS):
        position = axes[row_index][0].get_position()
        fig.text(
            0.018,
            (position.y0 + position.y1) / 2,
            label,
            ha='center',
            va='center',
            rotation=90,
            fontsize=11,
            fontweight='medium',
        )

    if mock:
        fig.text(
            0.985,
            0.015,
            'MOCK DATA — layout preview only',
            ha='right',
            va='bottom',
            fontsize=8,
            color='#777777',
        )
    fig.savefig(output_dir / 'grid_contact_sheet.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'grid_contact_sheet.pdf', bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description='Build the 3x3 experiment grid figure')
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Render deterministic mock data instead of loading results.',
    )
    parser.add_argument(
        '--pcam_pareto_dir',
        type=Path,
        default=EXPERIMENTS_DIR / 'pcam' / 'pcam_results_pareto',
    )
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
        '--smnist_capacity_dir',
        type=Path,
        default=EXPERIMENTS_DIR / 'spherical_mnist' / 'smnist_results_capacity',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=None,
        help='Defaults to ./grid_mockup for --mock, ./grid_figure otherwise.',
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir or (EXPERIMENTS_DIR / ('grid_mockup' if args.mock else 'grid_figure'))
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()

    if args.mock:
        grid = mock_grid_data()
    else:
        grid = (
            load_pcam(args.pcam_pareto_dir, args.pcam_data_pareto_dir),
            load_organ3d(args.organ3d_dir),
            load_smnist(args.smnist_dir, args.smnist_capacity_dir),
        )

    save_panels(grid, output_dir)
    save_contact_sheet(grid, output_dir, mock=args.mock)
    (output_dir / 'caption.txt').write_text(CAPTION_TEXT)

    print(f'Wrote grid assets to {output_dir}')


if __name__ == '__main__':
    main()
