#!/usr/bin/env python3
"""Build the 2x3 G-bispectral pooling grid figure (bars + param/data efficiency curves).

Rows (ROW_ORDER): OrganMNIST3D (octahedral O), PCam (C8). Spherical MNIST is not a
pooling experiment (fixed invariant + MLP head) and has its own figure and protocol
table in make_smnist_figure.py, which reuses the loaders and drawing helpers below.
Columns:
    1. OOD rotation robustness — canonical vs rotated test, bar pairs.
    2. Parameter efficiency    — rotated-test metric vs trainable params.
    3. Data efficiency         — rotated-test metric vs training examples.

Protocol (consistent across rows):
    - "Aug. CNN" baselines are non-equivariant CNNs trained with
      G-augmentation (train_mode R); invariant models are C-trained.
    - All line plots report the rotated (OOD) test metric, mean +/- std
      over seeds.

Usage:
    # Real data (after syncing results dirs from the GPU machines):
    python make_grid_figure.py

    # Layout preview with deterministic mock data:
    python make_grid_figure.py --mock
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
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
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
LegendHandle = Line2D | Patch

EXPERIMENTS_DIR: Final = Path(__file__).parent
# Final print size: NeurIPS text width is 5.5 in.
PANEL_SIZE: Final = (1.75, 1.35)
SHEET_WIDTH: Final = 5.5
SHEET_ROW_HEIGHT: Final = 0.851  # inches per axes row; 2 rows -> a 5.5 x 3.3 in sheet
SHEET_HSPACE: Final = 0.62
SHEET_BOTTOM: Final = 0.36
LEGEND_ROW_HEIGHT: Final = 0.13
BAR_YLIM: Final = (0.5, 1.0)

# Protocol markers used to reject stale results.json files (see _is_current).
SAMPLE_FLOOR: Final = 10_000  # default --min_epoch_samples in experiments/*/train.py
FULL_TRAIN_SIZE: Final = {'pcam': 262_144, 'organ3d': 971, 'spherical_mnist': 60_000}
MIN_LIBRARY_VERSION: Final = {'spherical_mnist': (0, 4, 0)}  # SO3onS2 selective map redesign
HEAD_FIX_DATASETS: Final = frozenset({'pcam', 'organ3d'})  # bispectral pooling head options


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodStyle:
    label: str
    color: str
    marker: str


@dataclass(frozen=True)
class BarPanel:
    methods: tuple[str, ...]
    original: FloatArray
    ood: FloatArray
    original_error: FloatArray
    ood_error: FloatArray
    ylabel: str


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
    ylim: tuple[float, float] | None = None
    xticks: tuple[float, ...] | None = None
    xticklabels: tuple[str, ...] | None = None


@dataclass(frozen=True)
class RowData:
    dataset: str
    bar: BarPanel
    params: CurvePanel
    data: CurvePanel


GridData = tuple[RowData, ...]


METHOD_STYLE: Final[dict[str, MethodStyle]] = {
    # Non-equivariant CNN baselines (3D for OrganMNIST3D), trained with G-augmentation.
    'standard': MethodStyle('Aug. CNN', '#888888', 's'),
    'standard_3d': MethodStyle('Aug. CNN', '#888888', 's'),
    'standard_s2': MethodStyle('Aug. CNN', '#888888', 's'),
    # Equivariant-pooling baselines.
    'norm': MethodStyle('NormReLU', '#2196F3', 'o'),
    'gate': MethodStyle('Gated', '#FF9800', '^'),
    'fourier_elu': MethodStyle('Fourier-ELU', '#9C27B0', 'D'),
    'max_pool': MethodStyle('Max pool', '#00897B', 'P'),
    # Incomplete second-order invariants (shared style across rows).
    'norm_pool': MethodStyle('Norm pool', '#6D4C41', 'v'),
    'power_spectrum': MethodStyle('Power spectrum', '#6D4C41', 'v'),
    # Complete invariant (ours).
    'bispectrum': MethodStyle('Bispectrum', '#E53935', '*'),
}

SHARED_LEGEND: Final = (
    ('bispectrum', 'Bispectrum (ours)'),
    ('max_pool', 'Max pool'),
    ('norm_pool', 'Norm pool'),
    ('standard', 'Aug. CNN'),
    ('norm', 'NormReLU'),
    ('gate', 'Gated'),
    ('fourier_elu', 'Fourier-ELU'),
)

ROW_ORDER: Final = ('organ3d', 'pcam')

ROW_METHODS: Final[dict[str, tuple[str, ...]]] = {
    'pcam': ('standard', 'norm', 'gate', 'fourier_elu', 'norm_pool', 'bispectrum'),
    'organ3d': ('standard_3d', 'norm_pool', 'max_pool', 'bispectrum'),
    'smnist': ('standard_s2', 'power_spectrum', 'bispectrum'),
}

ROW_LABELS: Final[dict[str, str]] = {
    'pcam': 'PCam\n' + r'$C_8$, test AUC',
    'organ3d': 'OrganMNIST3D\n' + r'octahedral $O$, accuracy',
    'smnist': 'Spherical MNIST\n' + r'$\mathrm{SO}(3)$, accuracy',
}

COLUMN_LABELS: Final = (
    'OOD rotation robustness',
    'Parameter efficiency',
    'Data efficiency',
)

COLUMN_KINDS: Final = ('ood', 'params', 'data')

CAPTION_TEXT: Final = """\
G-bispectral pooling in G-CNNs on OrganMNIST3D (octahedral rotations, test
accuracy, 10 seeds) and PatchCamelyon (planar C8 rotations, test AUC, 5 seeds
at n <= 2.5K, 3 above). Left: canonical (solid) vs rotated (hatched, OOD) test
metric on the full training set, at the smallest width for OrganMNIST3D and
matched ~100K-parameter configs for PCam. Middle and right: rotated test
metric, mean +/- sample std over seeds, against trainable parameters (full
training set for OrganMNIST3D, n = 12.5K for PCam) and training-set size
(smallest width for OrganMNIST3D, ~100K-parameter configs for PCam). "Aug. CNN" is a
non-equivariant CNN (3D for OrganMNIST3D) trained with group augmentation; the
invariant models are trained on canonical data and share a G-equivariant
backbone, differing only in the invariant map. Norm pool is an incomplete
second-order invariant; the G-bispectrum is complete.
"""


# --------------------------------------------------------------------------
# Results loading helpers
# --------------------------------------------------------------------------


def _parse_version(text: object) -> tuple[int, ...] | None:
    """Leading numeric components of a version string ('0.4.1.dev3+g1a2b' -> (0, 4, 1))."""
    if not isinstance(text, str):
        return None
    parts: list[int] = []
    for token in text.split('.'):
        digits = ''
        for ch in token:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or None


def _stale_reason(run: dict) -> str | None:
    """Why a results.json predates the current protocol, or None if it is current.

    Three classes of stale runs are rejected:

    * ``pre-floor``: low-data runs (``train_size`` below the per-epoch sample
      floor) written before ``min_epoch_samples`` existed. They were trained
      with a collapsed optimizer-step budget (one step per epoch at n=100) and
      are not comparable to current runs at the same ``train_size``.
    * ``pre-<version>``: runs of a dataset whose model code changed layout in
      the bispectrum library (Spherical MNIST: the SO3onS2 selective map was
      redesigned in 0.4.0), written by an older or unrecorded library version.
    * ``early-stopped``: runs trained with patience-based early stopping (or
      written before ``early_stopping_patience`` was recorded). Validation dips
      after every SGDR warm restart, so patience fired at restart-dependent
      epochs and the training budget differed between seeds and models.
    * ``pre-head-fix``: PCam / OrganMNIST3D bispectrum runs written before the
      pooling head recorded its options (``bispectrum_head``). That head applied
      a ReLU to the signed bispectrum before the projection, discarding the
      phase information that makes the bispectrum complete.
    """
    dataset = run.get('dataset')
    size = run.get('train_size')
    full = FULL_TRAIN_SIZE.get(dataset, SAMPLE_FLOOR)
    if size is not None and 0 < size < min(SAMPLE_FLOOR, full) and 'min_epoch_samples' not in run:
        return 'pre-floor'
    minimum = MIN_LIBRARY_VERSION.get(dataset)
    if minimum is not None:
        version = _parse_version(run.get('bispectrum_version'))
        if version is None or version < minimum:
            return 'pre-' + '.'.join(str(v) for v in minimum)
    if run.get('early_stopping_patience') != 0:
        return 'early-stopped'
    if dataset in HEAD_FIX_DATASETS and run.get('model') == 'bispectrum':
        if 'bispectrum_head' not in run:
            return 'pre-head-fix'
    return None


def _is_current(run: dict) -> bool:
    return _stale_reason(run) is None


def _load_runs(root: Path, include_stale: bool = False) -> list[dict]:
    """Recursively load every results.json under *root*, dropping stale runs.

    Stale runs (see ``_stale_reason``) are skipped unless *include_stale* is
    set; the number skipped per reason is reported so a silently reused
    pre-fix results directory is visible in the log.
    """
    runs: list[dict] = []
    skipped: dict[str, int] = defaultdict(int)
    if not root.exists():
        print(f'WARNING: results dir not found: {root}')
        return runs
    for p in sorted(root.rglob('results.json')):
        with open(p) as f:
            run = json.load(f)
        reason = _stale_reason(run)
        if reason is not None and not include_stale:
            skipped[reason] += 1
            continue
        runs.append(run)
    if skipped:
        detail = ', '.join(f'{n} {reason}' for reason, n in sorted(skipped.items()))
        print(f'{root}: loaded {len(runs)} runs, skipped {sum(skipped.values())} stale ({detail})')
    return runs


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof=1; 0.0 for a single value)."""
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std


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
    )


def _size_tick(n: float) -> str:
    return f'{n / 1000:g}K' if n >= 1000 else f'{n:g}'


def _data_ticks(sizes: Sequence[float], full_size: float) -> tuple[tuple[float, ...], tuple[str, ...]]:
    ticks = tuple(sorted(set(sizes)))
    return ticks, tuple('full' if t == full_size else _size_tick(t) for t in ticks)


def _build_curve_panel(
    methods: tuple[str, ...],
    points: dict[str, list[tuple[float, float, float]]],
    ylabel: str,
    ylim: tuple[float, float] | None = None,
    xticks: tuple[float, ...] | None = None,
    xticklabels: tuple[str, ...] | None = None,
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


def load_pcam(pareto_dir: Path, data_pareto_dir: Path, include_stale: bool = False) -> RowData:
    """PCam. Bars from matched-budget full-data runs, curves from sweeps."""
    pareto_runs = _group_runs(_load_runs(pareto_dir, include_stale), PCAM_METHODS)
    data_runs = _group_runs(_load_runs(data_pareto_dir, include_stale), PCAM_METHODS)

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

    bar = _build_bar_panel(PCAM_METHODS, aggregates, ylabel='Test AUC')

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

    params = _build_curve_panel(PCAM_METHODS, param_points, ylabel='Rotated test AUC')

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

    sizes = [x for pts in data_points.values() for x, _, _ in pts]
    xticks, xticklabels = _data_ticks(sizes, FULL_TRAIN_SIZE['pcam'])
    data = _build_curve_panel(
        PCAM_METHODS,
        data_points,
        ylabel='Rotated test AUC',
        xticks=xticks,
        xticklabels=xticklabels,
    )
    return RowData('pcam', bar=bar, params=params, data=data)


def load_organ3d(results_dir: Path, include_stale: bool = False) -> RowData:
    """OrganMNIST3D. Bars from ch(4,8) full-data runs, curves from sweeps."""
    grouped = _group_runs(_load_runs(results_dir, include_stale), ORGAN3D_METHODS)

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

    bar = _build_bar_panel(ROW_METHODS['organ3d'], aggregates, ylabel='Test accuracy')
    params = _build_curve_panel(ROW_METHODS['organ3d'], param_points, ylabel='Rotated test accuracy')
    sizes = [x for pts in data_points.values() for x, _, _ in pts]
    xticks, xticklabels = _data_ticks(sizes, FULL_TRAIN_SIZE['organ3d'])
    data = _build_curve_panel(
        ROW_METHODS['organ3d'],
        data_points,
        ylabel='Rotated test accuracy',
        xticks=xticks,
        xticklabels=xticklabels,
    )
    return RowData('organ3d', bar=bar, params=params, data=data)


def load_smnist(results_dir: Path, capacity_dir: Path, include_stale: bool = False) -> RowData:
    """Spherical MNIST. Bars + data from main sweep, params from capacity."""
    main_runs = _load_runs(results_dir, include_stale)
    capacity_runs = _load_runs(capacity_dir, include_stale)

    # Capacity runs use run_label 'model_h{width}'; recover the base model.
    grouped_main = _group_runs(main_runs, SMNIST_METHODS, model_key='base_model')
    for r in main_runs:  # older results may lack base_model
        if 'base_model' not in r and r.get('model') in SMNIST_METHODS:
            method = r['model']
            if _canonical_mode(r.get('train_mode', 'C')) == _expected_mode(method):
                grouped_main[method].append(r)

    aggregates: dict[str, Aggregate] = {}
    data_points: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    full_examples = math.nan

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
            full_examples = float(full[0].get('train_examples'))
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

    bar = _build_bar_panel(ROW_METHODS['smnist'], aggregates, ylabel='Test accuracy')
    params = _build_curve_panel(ROW_METHODS['smnist'], param_points, ylabel='Rotated test accuracy')
    sizes = [x for pts in data_points.values() for x, _, _ in pts]
    xticks, xticklabels = _data_ticks(sizes, full_examples)
    data = _build_curve_panel(
        ROW_METHODS['smnist'],
        data_points,
        ylabel='Rotated test accuracy',
        xticks=xticks,
        xticklabels=xticklabels,
    )
    return RowData('smnist', bar=bar, params=params, data=data)


# --------------------------------------------------------------------------
# Mock data (layout preview only)
# --------------------------------------------------------------------------


def _array(values: list[float]) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def mock_grid_data() -> GridData:
    """Deterministic mock numbers exercising the exact real-data layout."""
    pcam = RowData(
        'pcam',
        bar=BarPanel(
            methods=ROW_METHODS['pcam'],
            original=_array([0.896, 0.942, 0.941, 0.945, 0.930, 0.941]),
            ood=_array([0.861, 0.941, 0.940, 0.944, 0.929, 0.941]),
            original_error=_array([0.010, 0.004, 0.009, 0.004, 0.007, 0.004]),
            ood_error=_array([0.013, 0.004, 0.008, 0.004, 0.007, 0.004]),
            ylabel='Test AUC',
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
        'organ3d',
        bar=BarPanel(
            methods=ROW_METHODS['organ3d'],
            original=_array([0.601, 0.568, 0.730, 0.726]),
            ood=_array([0.576, 0.568, 0.730, 0.726]),
            original_error=_array([0.017, 0.099, 0.033, 0.027]),
            ood_error=_array([0.021, 0.099, 0.033, 0.027]),
            ylabel='Test accuracy',
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
        'smnist',
        bar=BarPanel(
            methods=ROW_METHODS['smnist'],
            original=_array([0.460, 0.792, 0.950]),
            ood=_array([0.230, 0.790, 0.951]),
            original_error=_array([0.012, 0.010, 0.001]),
            ood_error=_array([0.015, 0.010, 0.001]),
            ylabel='Test accuracy',
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
        ),
    )

    rows = {'pcam': pcam, 'organ3d': organ3d, 'smnist': smnist}
    return tuple(rows[dataset] for dataset in ROW_ORDER)


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
            'font.size': 6.5,
            'axes.labelsize': 6.5,
            'axes.linewidth': 0.5,
            'xtick.labelsize': 5.5,
            'ytick.labelsize': 5.5,
            'xtick.major.width': 0.5,
            'ytick.major.width': 0.5,
            'xtick.major.size': 2,
            'ytick.major.size': 2,
            'xtick.minor.size': 1.2,
            'xtick.minor.width': 0.4,
            'lines.linewidth': 1.0,
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
    ax.yaxis.grid(True, color='#D9D9D9', linewidth=0.4, alpha=0.8)
    ax.tick_params(axis='x', pad=1.5)
    ax.tick_params(axis='y', pad=1.5)


@dataclass(frozen=True)
class LineWeight:
    linewidth: float
    markersize: float
    elinewidth: float
    capsize: float
    alpha: float
    zorder: int


# The bispectrum is drawn on top in full weight; baselines stay legible but recede.
OURS: Final = 'bispectrum'
OURS_WEIGHT: Final = LineWeight(linewidth=1.5, markersize=5.0, elinewidth=0.7, capsize=1.2, alpha=1.0, zorder=4)
BASELINE_WEIGHT: Final = LineWeight(linewidth=0.8, markersize=2.4, elinewidth=0.5, capsize=0.0, alpha=0.55, zorder=2)
# Horizontal spread (log10 units) between methods that share x-values on the data panels.
DODGE_DECADES: Final = 0.035
# A single unconnected point has no line to carry it, so it needs a larger marker.
LONE_MARKER_SIZE: Final = 4.0


def _line_weight(method: str) -> LineWeight:
    return OURS_WEIGHT if method == OURS else BASELINE_WEIGHT


def _percent_axis(ax: Axes, label: str) -> None:
    if 'accuracy' in label.lower():
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f'{value:.0%}'))


def draw_bar_panel(ax: Axes, panel: BarPanel) -> None:
    """Draw one canonical-vs-rotated grouped bar panel."""
    x = np.arange(len(panel.methods), dtype=np.float64)
    width = 0.35
    error_kw = {'elinewidth': 0.6, 'capthick': 0.6}

    for index, method in enumerate(panel.methods):
        style = METHOD_STYLE[method]
        ax.bar(
            x[index] - width / 2,
            panel.original[index],
            width,
            yerr=panel.original_error[index],
            color=style.color,
            edgecolor=style.color,
            linewidth=0.5,
            capsize=1.2,
            error_kw=error_kw,
            zorder=3,
        )
        ax.bar(
            x[index] + width / 2,
            panel.ood[index],
            width,
            yerr=panel.ood_error[index],
            color=style.color,
            edgecolor=style.color,
            linewidth=0.5,
            hatch='////',
            alpha=0.42,
            capsize=1.2,
            error_kw=error_kw,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_STYLE[method].label for method in panel.methods],
        rotation=35,
        ha='right',
        rotation_mode='anchor',
    )
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel(panel.ylabel)
    ax.set_ylim(BAR_YLIM)
    _percent_axis(ax, panel.ylabel)
    style_axes(ax)


def draw_curve_panel(ax: Axes, panel: CurvePanel, *, xlabel: str, dodge: bool = False) -> None:
    """Draw one parameter- or data-efficiency curve panel.

    With *dodge*, methods sharing x-values are spread horizontally so their
    error bars do not stack.
    """
    count = len(panel.series)
    for index, series in enumerate(panel.series):
        style = METHOD_STYLE[series.method]
        weight = _line_weight(series.method)
        shift = 10 ** ((index - (count - 1) / 2) * DODGE_DECADES) if dodge else 1.0
        connected = len(series.x) > 1
        ax.errorbar(
            series.x * shift,
            series.y,
            yerr=series.error,
            color=style.color,
            marker=style.marker,
            linestyle='-' if connected else 'none',
            linewidth=weight.linewidth,
            markersize=weight.markersize if connected else max(weight.markersize, LONE_MARKER_SIZE),
            markeredgewidth=0.4,
            markeredgecolor='white',
            capsize=weight.capsize,
            elinewidth=weight.elinewidth,
            alpha=weight.alpha,
            zorder=weight.zorder,
        )

    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(panel.ylabel)
    if panel.ylim is not None:
        ax.set_ylim(panel.ylim)
    if panel.xticks is not None:
        ax.set_xticks(panel.xticks)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
    if panel.xticklabels is not None:
        ax.set_xticklabels(panel.xticklabels, rotation=35, ha='right', rotation_mode='anchor')
    _percent_axis(ax, panel.ylabel)
    style_axes(ax)


def draw_panel(ax: Axes, row: RowData, column: int) -> None:
    if column == 0:
        draw_bar_panel(ax, row.bar)
    elif column == 1:
        draw_curve_panel(ax, row.params, xlabel='Trainable parameters')
    else:
        draw_curve_panel(ax, row.data, xlabel='Training examples', dodge=True)


def _legend_key(method: str) -> str:
    return 'standard' if method.startswith('standard') else method


def legend_handles(datasets: Sequence[str] = ROW_ORDER) -> list[LegendHandle]:
    present = {_legend_key(method) for dataset in datasets for method in ROW_METHODS[dataset]}
    handles: list[LegendHandle] = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[method].color,
            marker=METHOD_STYLE[method].marker,
            linewidth=_line_weight(method).linewidth,
            markersize=_line_weight(method).markersize + 0.6,
            markeredgecolor='white',
            markeredgewidth=0.4,
            label=label,
        )
        for method, label in SHARED_LEGEND
        if method in present
    ]
    handles.extend(
        [
            Patch(facecolor='#777777', edgecolor='#777777', label='Canonical test'),
            Patch(facecolor='#D8D8D8', edgecolor='#777777', hatch='////', linewidth=0.5, label='Rotated test (OOD)'),
        ]
    )
    return handles


def _panel_filename(row_index: int, column: int, dataset: str) -> str:
    return f'grid_r{row_index + 1}c{column + 1}_{dataset}_{COLUMN_KINDS[column]}.pdf'


def save_panels(grid: GridData, output_dir: Path) -> None:
    """Save title-free, legend-free vector panels plus the shared legend."""
    for row_index, row in enumerate(grid):
        for column in range(3):
            fig, ax = plt.subplots(figsize=PANEL_SIZE, constrained_layout=True)
            draw_panel(ax, row, column)
            fig.savefig(output_dir / _panel_filename(row_index, column, row.dataset))
            plt.close(fig)

    handles = legend_handles()
    fig = plt.figure(figsize=(SHEET_WIDTH, 0.35))
    fig.legend(handles=handles, loc='center', ncol=5, frameon=False, fontsize=6, handlelength=1.8)
    fig.savefig(output_dir / 'legend.pdf')
    plt.close(fig)


def save_contact_sheet(grid: GridData, output_dir: Path, *, mock: bool, stem: str = 'grid_contact_sheet') -> None:
    """Save the assembled grid at print size, with a legend on top listing only the methods in `grid`."""
    handles = legend_handles([row.dataset for row in grid])
    ncol = len(handles) if len(handles) <= 6 else 5
    top = 0.45 + LEGEND_ROW_HEIGHT * math.ceil(len(handles) / ncol)
    height = SHEET_BOTTOM + top + SHEET_ROW_HEIGHT * (len(grid) + SHEET_HSPACE * (len(grid) - 1))
    fig = plt.figure(figsize=(SHEET_WIDTH, height), constrained_layout=False)
    gridspec = fig.add_gridspec(
        len(grid),
        3,
        left=0.13,
        right=0.99,
        bottom=SHEET_BOTTOM / height,
        top=1.0 - top / height,
        wspace=0.42,
        hspace=SHEET_HSPACE,
    )

    axes: list[list[Axes]] = []
    for row_index, row in enumerate(grid):
        row_axes: list[Axes] = []
        for column in range(3):
            ax = fig.add_subplot(gridspec[row_index, column])
            draw_panel(ax, row, column)
            row_axes.append(ax)
        axes.append(row_axes)

    fig.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.55, 1.0),
        ncol=ncol,
        frameon=False,
        fontsize=6,
        handlelength=1.8,
        columnspacing=1.2,
    )

    for column, label in enumerate(COLUMN_LABELS):
        position = axes[0][column].get_position()
        fig.text(
            (position.x0 + position.x1) / 2,
            position.y1 + 0.066 / height,
            label,
            ha='center',
            va='bottom',
            fontsize=7.5,
            fontweight='medium',
        )

    for row_index, row in enumerate(grid):
        position = axes[row_index][0].get_position()
        fig.text(
            0.012,
            (position.y0 + position.y1) / 2,
            ROW_LABELS[row.dataset],
            ha='center',
            va='center',
            rotation=90,
            fontsize=7,
            fontweight='medium',
            linespacing=1.3,
        )

    if mock:
        fig.text(
            0.99,
            0.005,
            'MOCK DATA — layout preview only',
            ha='right',
            va='bottom',
            fontsize=6,
            color='#777777',
        )
    fig.savefig(output_dir / f'{stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description='Build the 2x3 pooling grid figure')
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
        '--output_dir',
        type=Path,
        default=None,
        help='Defaults to ./grid_mockup for --mock, ./grid_figure otherwise.',
    )
    parser.add_argument(
        '--include_stale',
        action='store_true',
        help='Keep results.json files that predate the current protocol (pre-floor low-data runs, '
        'pre-0.4.0 Spherical MNIST runs). Default: skip them and report the count.',
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir or (EXPERIMENTS_DIR / ('grid_mockup' if args.mock else 'grid_figure'))
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()

    if args.mock:
        grid = mock_grid_data()
    else:
        rows = {
            'pcam': load_pcam(args.pcam_pareto_dir, args.pcam_data_pareto_dir, args.include_stale),
            'organ3d': load_organ3d(args.organ3d_dir, args.include_stale),
        }
        grid = tuple(rows[dataset] for dataset in ROW_ORDER)

    save_panels(grid, output_dir)
    save_contact_sheet(grid, output_dir, mock=args.mock)
    for row in grid:
        save_contact_sheet((row,), output_dir, mock=args.mock, stem=f'grid_row_{row.dataset}')
    (output_dir / 'caption.txt').write_text(CAPTION_TEXT)

    print(f'Wrote grid assets to {output_dir}')


if __name__ == '__main__':
    main()
