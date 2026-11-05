#!/usr/bin/env python3
"""Build the 2x3 G-bispectral pooling grid figure (split dots + param/data efficiency curves).

Rows (ROW_ORDER): PCam (C8), OrganMNIST3D (octahedral O). Spherical MNIST is not a
pooling experiment (fixed invariant + MLP head) and has its own figure and protocol
table in make_smnist_figure.py, which reuses the loaders and drawing helpers below.
Columns:
    1. Full training set       — canonical (solid) vs rotated (light) test, bar pairs.
    2. Parameter efficiency    — rotated-test metric vs trainable params.
    3. Data efficiency         — rotated-test metric vs training examples.

Protocol (consistent across rows):
    - "Aug. CNN" baselines are non-equivariant CNNs trained with
      G-augmentation (train_mode R); invariant models are C-trained.
    - "Rotated test" is the metric averaged over the test set transformed by every
      element of G (identity included).
    - Points are mean +/- sample std over seeds; the panels of a row share one y-axis.
    - The contact sheet is exactly the NeurIPS text width (5.5 in), with a one-row legend on top.

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
import sys
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
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator
from numpy.typing import NDArray
from scipy.stats import ttest_ind_from_stats

EXPERIMENTS_DIR: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS_DIR.parent))

from assets import brand  # noqa: E402

FloatArray = NDArray[np.float64]

# Print sizes in inches and points. The sheet is exactly the NeurIPS text width, so it is
# included at 100% and these font sizes are the printed ones (caption text is 9 pt).
PANEL_SIZE: Final = (1.75, 1.35)
SHEET_WIDTH: Final = 5.5
SHEET_LEFT: Final = 0.64  # row label, y label and y tick labels
SHEET_RIGHT: Final = 0.03
SHEET_WSPACE: Final = 0.2  # rows share a y-axis, so no tick labels sit between panels
SHEET_ROW_HEIGHT: Final = 1.0
SHEET_ROW_GAP: Final = 0.44  # method names and x labels of the row above
SHEET_BOTTOM: Final = 0.3
SHEET_TITLES: Final = 0.15  # column title above the first row
LEGEND_HEIGHT: Final = 0.12  # one entry row
LEGEND_GAP: Final = 0.06
TITLE_FONT_SIZE: Final = 6.5
SUBTITLE_FONT_SIZE: Final = 5.5
ROW_LABEL_FONT_SIZE: Final = 6.5
LEGEND_FONT_SIZE: Final = 5.5

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
class SplitPanel:
    """Full-training-set metric per method on the canonical and the rotated test set."""

    methods: tuple[str, ...]
    canonical: FloatArray
    rotated: FloatArray
    canonical_error: FloatArray
    rotated_error: FloatArray
    ylabel: str
    n_seeds: tuple[int, ...] = ()


@dataclass(frozen=True)
class CurveSeries:
    method: str
    x: FloatArray
    y: FloatArray
    error: FloatArray
    n_seeds: tuple[int, ...] = ()


@dataclass(frozen=True)
class CurvePanel:
    series: tuple[CurveSeries, ...]
    ylabel: str
    xticks: tuple[float, ...] | None = None
    xticklabels: tuple[str, ...] | None = None


@dataclass(frozen=True)
class RowData:
    dataset: str
    split: SplitPanel
    params: CurvePanel
    data: CurvePanel


# (x, mean, sample std, number of seeds) for one point of a curve.
type Point = tuple[float, float, float, int]


GridData = tuple[RowData, ...]


METHOD_STYLE: Final[dict[str, MethodStyle]] = {
    # Non-equivariant CNN baselines (3D for OrganMNIST3D), trained with G-augmentation.
    'standard': MethodStyle('Aug. CNN', brand.METHOD_COLORS['standard'], 's'),
    'standard_3d': MethodStyle('Aug. CNN', brand.METHOD_COLORS['standard_3d'], 's'),
    'standard_s2': MethodStyle('Aug. CNN', brand.METHOD_COLORS['standard_s2'], 's'),
    # Equivariant-pooling baselines.
    'norm': MethodStyle('NormReLU', brand.METHOD_COLORS['norm'], 'o'),
    'gate': MethodStyle('Gated', brand.METHOD_COLORS['gate'], '^'),
    'fourier_elu': MethodStyle('Fourier-ELU', brand.METHOD_COLORS['fourier_elu'], 'D'),
    'max_pool': MethodStyle('Max pool', brand.METHOD_COLORS['max_pool'], 'P'),
    # Incomplete second-order invariants (shared style across rows).
    'norm_pool': MethodStyle('Norm pool', brand.METHOD_COLORS['norm_pool'], 'v'),
    'power_spectrum': MethodStyle('Power spectrum', brand.METHOD_COLORS['power_spectrum'], 'v'),
    # Complete invariant (ours).
    'bispectrum': MethodStyle('Bispectrum', brand.METHOD_COLORS['bispectrum'], '*'),
}

# C8 first as the baseline case, then the octahedral group where completeness pays off.
ROW_ORDER: Final = ('pcam', 'organ3d')

ROW_METHODS: Final[dict[str, tuple[str, ...]]] = {
    'pcam': ('standard', 'norm', 'gate', 'fourier_elu', 'norm_pool', 'bispectrum'),
    'organ3d': ('standard_3d', 'norm_pool', 'max_pool', 'bispectrum'),
    'smnist': ('standard_s2', 'power_spectrum', 'bispectrum'),
}

# The metric lives on the y-axis and the test split in the column subtitle, so each is said once.
ROW_LABELS: Final[dict[str, str]] = {
    'pcam': 'PCam\n' + r'$C_8$',
    'organ3d': 'OrganMNIST3D\n' + r'octahedral $O$',
    'smnist': 'Spherical MNIST\n' + r'$\mathrm{SO}(3)$',
}

COLUMN_LABELS: Final = (
    'Full training set',
    'Parameter efficiency',
    'Data efficiency',
)

COLUMN_KINDS: Final = ('split', 'params', 'data')

# \input{caption.tex} in the paper's figure environment and \input{caption_details.tex} in the
# appendix; preview.tex typesets both. @name@ fields are filled by tell_story() from the plotted data.
CAPTION_TEX: Final = r"""\caption{\textbf{A complete invariant pays off on the octahedral group and is on par on $C_8$.}
  On PCam ($C_8$), no model is significantly better than bispectral pooling at any training-set size.
  On OrganMNIST3D ($O$), bispectral pooling beats max pooling on the same backbone by @gain@\,pp;
  its smallest model beats max pooling's largest, which has $@ratio@\times$ the parameters; and with
  @data_n@ training examples it beats max pooling trained on all @full_n@.
  Rotated test: the metric averaged over all $|G|$ rotations of the test set. Equivariant models
  are trained on canonical data only; Aug.\ CNN is trained with augmentation over $G$.
  Mean $\pm$ std over 3--10 seeds; details in Appendix~\ref{app:pooling-grid}.}
\label{fig:pooling-grid}
"""

DETAILS_TEX: Final = r"""\subsection{Invariant maps in $G$-CNNs (Figure~\ref{fig:pooling-grid})}
\label{app:pooling-grid}

\paragraph{Protocol.}
The rotated test averages the metric over the test set transformed by every element of $G$,
identity included. On OrganMNIST3D the 24 octahedral rotations permute voxels exactly, so
$G$-invariant models score identically on the canonical and rotated test sets; on PCam the
$45^\circ$ rotations are bilinearly interpolated, so invariance is exact only at multiples of
$90^\circ$. Aug.\ CNN is a non-equivariant CNN (3D for OrganMNIST3D) trained with augmentation
over $G$, so the rotated test is in-distribution for it. All other models are $G$-equivariant,
trained on canonical data only, and end in an invariant map over the group axis: a maximum (Max
pool; NormReLU, Gated and Fourier-ELU name the backbone nonlinearity), the $\ell_2$ norm (Norm
pool, an incomplete invariant), or the selective $G$-bispectrum (complete). Max pool and
Bispectrum on OrganMNIST3D, and Norm pool and Bispectrum on PCam, share a ReLU backbone and differ
only in the invariant map; OrganMNIST3D's Norm pool uses NormReLU.

\paragraph{Held fixed.}
Left: full training set, at the smallest width (OrganMNIST3D) or ${\sim}100$K-parameter
configurations (PCam). Middle: full training set (OrganMNIST3D) or $n = 12.5$K (PCam). Right:
smallest width (OrganMNIST3D) or ${\sim}100$K parameters (PCam); ``all'' is the full training set,
and the PCam axis is broken before it (262K examples). The panels of a row share the $y$-axis.

\paragraph{Statistics.}
Mean $\pm$ sample standard deviation over 10 seeds (OrganMNIST3D) and 3 seeds (PCam; 5 at
$n \le 2.5$K). $p$-values are two-sided Welch $t$-tests, uncorrected for multiplicity. On
OrganMNIST3D, bispectral pooling beats max pooling by @gain@\,pp ($p @gain_p@$); its smallest
model (@ours_params@ parameters) beats max pooling's largest (@rival_params@) by @param_gain@\,pp
($p @param_p@$); and at $n = @data_n@$ it beats max pooling at $n = @full_n@$ by
@data_gain@\,pp ($p @data_p@$). On PCam, no model is significantly better than bispectral
pooling at any training-set size; the closest is @rival@ @rival_at@ (${+}@rival_gap@$ AUC,
$p @rival_p@$). With 3 seeds per arm and PCam's seed spread, these tests have 80\% power only for
gaps of about @mde@ AUC, so ``on par'' means no difference at that resolution.
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


def _build_split_panel(
    methods: tuple[str, ...],
    aggregates: dict[str, Aggregate],
    ylabel: str,
) -> SplitPanel:
    present = tuple(m for m in methods if m in aggregates)
    missing = [m for m in methods if m not in aggregates]
    if missing:
        print(f'WARNING: split panel missing methods {missing}')
    return SplitPanel(
        methods=present,
        canonical=np.asarray([aggregates[m].canonical_mean for m in present]),
        rotated=np.asarray([aggregates[m].ood_mean for m in present]),
        canonical_error=np.asarray([aggregates[m].canonical_std for m in present]),
        rotated_error=np.asarray([aggregates[m].ood_std for m in present]),
        ylabel=ylabel,
        n_seeds=tuple(aggregates[m].n_seeds for m in present),
    )


def _size_tick(n: float) -> str:
    if n >= 1_000_000:
        return f'{n / 1_000_000:g}M'
    return f'{n / 1000:g}K' if n >= 1000 else f'{n:g}'


def _data_ticks(sizes: Sequence[float], full_size: float) -> tuple[tuple[float, ...], tuple[str, ...]]:
    ticks = tuple(sorted(set(sizes)))
    return ticks, tuple('all' if t == full_size else _size_tick(t) for t in ticks)


def _compress_tail(ticks: Sequence[float]) -> tuple[float, float] | None:
    """(true, drawn) x of a last tick that sits far past an otherwise geometric run of ticks.

    PCam's full set is 21x its largest subset; on a true log axis that one gap takes ~40% of
    the width and crowds the subset labels. It is drawn one typical step out instead, and the
    caller marks the jump with an axis break.
    """
    if len(ticks) < 3:
        return None
    *head, last = sorted(ticks)
    step = (head[-1] / head[0]) ** (1 / (len(head) - 1))
    if last / head[-1] <= 2 * step:
        return None
    return last, head[-1] * step


def _draw_axis_break(ax: Axes, x: float) -> None:
    ax.text(
        x,
        0.0,
        '//',
        transform=ax.get_xaxis_transform(),
        ha='center',
        va='center',
        fontsize=5.5,
        color=brand.MUTED,
        bbox={'facecolor': brand.surface_color(), 'edgecolor': 'none', 'pad': 0.6},
        clip_on=False,
        zorder=5,
    )


def _drawn_sizes(x: FloatArray, ticks: Sequence[float]) -> FloatArray:
    tail = _compress_tail(ticks)
    return x if tail is None else np.where(x == tail[0], tail[1], x)


def _set_size_axis(ax: Axes, ticks: Sequence[float], labels: Sequence[str], *, dodge_decades: float) -> None:
    """Training-size ticks on a log axis trimmed to the (dodged) data, with an axis break
    before a compressed last tick. Plot x through `_drawn_sizes` with the same ticks."""
    drawn = sorted(_drawn_sizes(np.asarray(ticks, dtype=np.float64), ticks))
    ax.set_xscale('log')
    ax.set_xticks(drawn, labels=list(labels))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    pad = 10 ** (dodge_decades + SIZE_AXIS_PAD_DECADES)
    ax.set_xlim(drawn[0] / pad, drawn[-1] * pad)
    if _compress_tail(ticks) is not None:
        _draw_axis_break(ax, math.sqrt(drawn[-2] * drawn[-1]))


def _build_curve_panel(
    methods: tuple[str, ...],
    points: dict[str, list[Point]],
    ylabel: str,
    xticks: tuple[float, ...] | None = None,
    xticklabels: tuple[str, ...] | None = None,
) -> CurvePanel:
    """Build a curve panel from per-method (x, y_mean, y_std, n_seeds) points."""
    series: list[CurveSeries] = []
    for method in methods:
        pts = sorted(points.get(method, []))
        if not pts:
            print(f'WARNING: curve panel missing method {method}')
            continue
        xs, ys, es, ns = zip(*pts, strict=True)
        series.append(
            CurveSeries(
                method,
                np.asarray(xs, dtype=np.float64),
                np.asarray(ys, dtype=np.float64),
                np.asarray(es, dtype=np.float64),
                tuple(ns),
            )
        )
    return CurvePanel(
        series=tuple(series),
        ylabel=ylabel,
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

    split = _build_split_panel(PCAM_METHODS, aggregates, ylabel='Test AUC')

    # Param curves: pareto sweep grouped by growth rate.
    param_points: dict[str, list[Point]] = defaultdict(list)
    for method, runs in pareto_runs.items():
        by_gr = defaultdict(list)
        for r in runs:
            by_gr[r.get('growth_rate')].append(r)
        for gr_runs in by_gr.values():
            agg = _aggregate(gr_runs, 'test_c', 'auc')
            if agg is not None:
                param_points[method].append((agg.n_params, agg.ood_mean, agg.ood_std, agg.n_seeds))

    params = _build_curve_panel(PCAM_METHODS, param_points, ylabel='Test AUC')

    # Data curves: data-pareto sweep grouped by training-set size.
    data_points: dict[str, list[Point]] = defaultdict(list)
    for method, runs in data_runs.items():
        by_size = defaultdict(list)
        for r in runs:
            by_size[r.get('train_examples')].append(r)
        for size, size_runs in by_size.items():
            agg = _aggregate(size_runs, 'test_c', 'auc')
            if agg is not None:
                data_points[method].append((float(size), agg.ood_mean, agg.ood_std, agg.n_seeds))

    sizes = [x for pts in data_points.values() for x, _, _, _ in pts]
    xticks, xticklabels = _data_ticks(sizes, FULL_TRAIN_SIZE['pcam'])
    data = _build_curve_panel(
        PCAM_METHODS,
        data_points,
        ylabel='Test AUC',
        xticks=xticks,
        xticklabels=xticklabels,
    )
    return RowData('pcam', split=split, params=params, data=data)


def load_organ3d(results_dir: Path, include_stale: bool = False) -> RowData:
    """OrganMNIST3D. Bars from ch(4,8) full-data runs, curves from sweeps."""
    grouped = _group_runs(_load_runs(results_dir, include_stale), ORGAN3D_METHODS)

    aggregates: dict[str, Aggregate] = {}
    param_points: dict[str, list[Point]] = defaultdict(list)
    data_points: dict[str, list[Point]] = defaultdict(list)

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
                param_points[method].append((ch_agg.n_params, ch_agg.ood_mean, ch_agg.ood_std, ch_agg.n_seeds))

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
                data_points[method].append((float(size), size_agg.ood_mean, size_agg.ood_std, size_agg.n_seeds))

    remap = {'standard': 'standard_3d'}
    aggregates = {remap.get(k, k): v for k, v in aggregates.items()}
    param_points = {remap.get(k, k): v for k, v in param_points.items()}
    data_points = {remap.get(k, k): v for k, v in data_points.items()}

    split = _build_split_panel(ROW_METHODS['organ3d'], aggregates, ylabel='Test accuracy')
    params = _build_curve_panel(ROW_METHODS['organ3d'], param_points, ylabel='Test accuracy')
    sizes = [x for pts in data_points.values() for x, _, _, _ in pts]
    xticks, xticklabels = _data_ticks(sizes, FULL_TRAIN_SIZE['organ3d'])
    data = _build_curve_panel(
        ROW_METHODS['organ3d'],
        data_points,
        ylabel='Test accuracy',
        xticks=xticks,
        xticklabels=xticklabels,
    )
    return RowData('organ3d', split=split, params=params, data=data)


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
    data_points: dict[str, list[Point]] = defaultdict(list)
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
                data_points[method].append((float(size), size_agg.ood_mean, size_agg.ood_std, size_agg.n_seeds))

    # Param curve: capacity sweep grouped by run label (one label per width).
    param_points: dict[str, list[Point]] = defaultdict(list)
    by_label = defaultdict(list)
    for r in capacity_runs:
        base = r.get('base_model')
        if base in SMNIST_METHODS and _canonical_mode(r.get('train_mode', 'C')) == 'C':
            by_label[(base, r.get('model'))].append(r)
    for (base, _label), label_runs in by_label.items():
        agg = _aggregate(label_runs, 'test_nr', 'accuracy')
        if agg is not None:
            param_points[base].append((agg.n_params, agg.ood_mean, agg.ood_std, agg.n_seeds))

    # The Aug. CNN has a fixed architecture: single point from the main sweep.
    if 'standard' in aggregates:
        std = aggregates['standard']
        param_points['standard'].append((std.n_params, std.ood_mean, std.ood_std, std.n_seeds))

    remap = {'standard': 'standard_s2'}
    aggregates = {remap.get(k, k): v for k, v in aggregates.items()}
    param_points = {remap.get(k, k): v for k, v in param_points.items()}
    data_points = {remap.get(k, k): v for k, v in data_points.items()}

    split = _build_split_panel(ROW_METHODS['smnist'], aggregates, ylabel='Test accuracy')
    params = _build_curve_panel(ROW_METHODS['smnist'], param_points, ylabel='Test accuracy')
    sizes = [x for pts in data_points.values() for x, _, _, _ in pts]
    xticks, xticklabels = _data_ticks(sizes, full_examples)
    data = _build_curve_panel(
        ROW_METHODS['smnist'],
        data_points,
        ylabel='Test accuracy',
        xticks=xticks,
        xticklabels=xticklabels,
    )
    return RowData('smnist', split=split, params=params, data=data)


# --------------------------------------------------------------------------
# Mock data (layout preview only)
# --------------------------------------------------------------------------


def _array(values: list[float]) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def mock_grid_data() -> GridData:
    """Deterministic mock numbers exercising the exact real-data layout."""
    pcam = RowData(
        'pcam',
        split=SplitPanel(
            methods=ROW_METHODS['pcam'],
            canonical=_array([0.896, 0.942, 0.941, 0.945, 0.930, 0.941]),
            rotated=_array([0.861, 0.941, 0.940, 0.944, 0.929, 0.941]),
            canonical_error=_array([0.010, 0.004, 0.009, 0.004, 0.007, 0.004]),
            rotated_error=_array([0.013, 0.004, 0.008, 0.004, 0.007, 0.004]),
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
            ylabel='Test AUC',
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
            ylabel='Test AUC',
            xticks=(100, 500, 2_500, 12_500, 262_144),
            xticklabels=('100', '500', '2.5K', '12.5K', 'all'),
        ),
    )

    organ3d = RowData(
        'organ3d',
        split=SplitPanel(
            methods=ROW_METHODS['organ3d'],
            canonical=_array([0.601, 0.568, 0.730, 0.726]),
            rotated=_array([0.576, 0.568, 0.730, 0.726]),
            canonical_error=_array([0.017, 0.099, 0.033, 0.027]),
            rotated_error=_array([0.021, 0.099, 0.033, 0.027]),
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
            ylabel='Test accuracy',
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
            ylabel='Test accuracy',
            xticks=(50, 100, 250, 500, 971),
            xticklabels=('50', '100', '250', '500', 'all'),
        ),
    )

    smnist = RowData(
        'smnist',
        split=SplitPanel(
            methods=ROW_METHODS['smnist'],
            canonical=_array([0.460, 0.792, 0.950]),
            rotated=_array([0.230, 0.790, 0.951]),
            canonical_error=_array([0.012, 0.010, 0.001]),
            rotated_error=_array([0.015, 0.010, 0.001]),
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
            ylabel='Test accuracy',
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
            ylabel='Test accuracy',
            xticks=(100, 500, 2_500, 12_500, 60_000),
            xticklabels=('100', '500', '2.5K', '12.5K', 'all'),
        ),
    )

    rows = {'pcam': pcam, 'organ3d': organ3d, 'smnist': smnist}
    return tuple(rows[dataset] for dataset in ROW_ORDER)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def configure_style() -> None:
    """Apply the brand style at print sizes (fonts stay editable in Illustrator)."""
    brand.use_style()
    plt.rcParams.update(
        {
            'font.size': 6,
            'axes.labelsize': 6,
            'axes.linewidth': 0.5,
            'xtick.labelsize': 5,
            'ytick.labelsize': 5,
            'legend.fontsize': LEGEND_FONT_SIZE,
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
    brand.style_axes(ax, grid='y')
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


# The bispectrum is drawn on top in full weight; baselines stay legible but recede. Baselines
# are told apart by marker shape alone (they share the gray ramp), so their markers stay readable.
OURS: Final = 'bispectrum'
OURS_WEIGHT: Final = LineWeight(linewidth=1.5, markersize=5.0, elinewidth=0.7, capsize=1.2, alpha=1.0, zorder=4)
BASELINE_WEIGHT: Final = LineWeight(linewidth=0.8, markersize=3.2, elinewidth=0.5, capsize=0.0, alpha=1.0, zorder=2)
# Horizontal spread (log10 units) between methods that share x-values on the data panels.
DODGE_DECADES: Final = 0.035
# Room (log10 units) between the outermost dodged marker and the axis edge on size axes.
SIZE_AXIS_PAD_DECADES: Final = 0.1
# A single unconnected point has no line to carry it, so it needs a larger marker.
LONE_MARKER_SIZE: Final = 4.0
SPLIT_BAR_WIDTH: Final = 0.36
# Rotated-test bars are an opaque tint of the method color with a full-color edge, which
# stays legible for the lightest grays and prints the same in every PDF viewer.
ROTATED_TINT: Final = 0.35
SPLIT_EDGE_WIDTH: Final = 0.5
# Fraction of the row's data range added above and below it.
ROW_YLIM_PAD: Final = 0.04


def _line_weight(method: str) -> LineWeight:
    return OURS_WEIGHT if method == OURS else BASELINE_WEIGHT


def _metric_axis(ax: Axes, label: str) -> None:
    """Accuracy in whole percent, AUC to two decimals, on every panel alike."""
    if 'accuracy' in label.lower():
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f'{value:.0%}'))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f'{value:.2f}'))


def _row_ylim(row: RowData) -> tuple[float, float]:
    """One y-range for all three panels of a row, covering every mean +/- std."""
    split = row.split
    bounds = [
        (split.canonical - split.canonical_error, split.canonical + split.canonical_error),
        (split.rotated - split.rotated_error, split.rotated + split.rotated_error),
        *((s.y - s.error, s.y + s.error) for panel in (row.params, row.data) for s in panel.series),
    ]
    low = min(float(np.min(lo)) for lo, _ in bounds)
    high = max(float(np.max(hi)) for _, hi in bounds)
    pad = ROW_YLIM_PAD * (high - low)
    return low - pad, high + pad


def draw_split_panel(ax: Axes, panel: SplitPanel) -> None:
    """Canonical (solid) vs rotated (light tint) test metric per method, full training set."""
    x = np.arange(len(panel.methods), dtype=np.float64)
    error_kw = {'elinewidth': 0.5, 'capthick': 0.5, 'ecolor': brand.INK}
    for index, method in enumerate(panel.methods):
        color = METHOD_STYLE[method].color
        for offset, mean, error, face in (
            (-SPLIT_BAR_WIDTH / 2, panel.canonical[index], panel.canonical_error[index], color),
            (SPLIT_BAR_WIDTH / 2, panel.rotated[index], panel.rotated_error[index], brand.tint(color, ROTATED_TINT)),
        ):
            ax.bar(
                x[index] + offset,
                mean,
                SPLIT_BAR_WIDTH,
                yerr=error,
                color=face,
                edgecolor=color,
                linewidth=SPLIT_EDGE_WIDTH,
                capsize=1.0,
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
    ax.set_xlim(-0.5, len(panel.methods) - 0.5)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel(panel.ylabel)
    _metric_axis(ax, panel.ylabel)
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
        x = series.x if panel.xticks is None else _drawn_sizes(series.x, panel.xticks)
        ax.errorbar(
            x * shift,
            series.y,
            yerr=series.error,
            color=style.color,
            marker=style.marker,
            linestyle='-' if connected else 'none',
            linewidth=weight.linewidth,
            markersize=weight.markersize if connected else max(weight.markersize, LONE_MARKER_SIZE),
            markeredgewidth=0.4,
            markeredgecolor=brand.surface_color(),
            capsize=weight.capsize,
            elinewidth=weight.elinewidth,
            alpha=weight.alpha,
            zorder=weight.zorder,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(panel.ylabel)
    if panel.xticks is None:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: _size_tick(value)))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        labels = panel.xticklabels or tuple(_size_tick(t) for t in panel.xticks)
        dodge_decades = (count - 1) / 2 * DODGE_DECADES if dodge else 0.0
        _set_size_axis(ax, panel.xticks, labels, dodge_decades=dodge_decades)
    _metric_axis(ax, panel.ylabel)
    style_axes(ax)


def draw_panel(ax: Axes, row: RowData, column: int) -> None:
    if column == 0:
        draw_split_panel(ax, row.split)
    elif column == 1:
        draw_curve_panel(ax, row.params, xlabel='Trainable parameters')
    else:
        draw_curve_panel(ax, row.data, xlabel='Training examples', dodge=True)
    ax.set_ylim(_row_ylim(row))


# --------------------------------------------------------------------------
# Legend: one row of methods, then the test-set key
# --------------------------------------------------------------------------

LEGEND_ORDER: Final = ('bispectrum', 'norm_pool', 'max_pool', 'norm', 'gate', 'fourier_elu', 'standard')
LEGEND_LABELS: Final = {'bispectrum': 'Bispectrum (ours)'}


def _legend_key(method: str) -> str:
    return 'standard' if method.startswith('standard') else method


def _method_handle(method: str) -> Line2D:
    style = METHOD_STYLE[method]
    weight = _line_weight(method)
    return Line2D(
        [0],
        [0],
        color=style.color,
        marker=style.marker,
        linewidth=weight.linewidth,
        markersize=weight.markersize + 0.6,
        markeredgecolor=brand.surface_color(),
        markeredgewidth=0.4,
        label=LEGEND_LABELS.get(method, style.label),
    )


def _split_handles() -> list[Patch]:
    return [
        Patch(
            facecolor=brand.tint(brand.MUTED, strength),
            edgecolor=brand.MUTED,
            linewidth=SPLIT_EDGE_WIDTH,
            label=label,
        )
        for label, strength in (('Canonical', 1.0), ('Rotated', ROTATED_TINT))
    ]


def legend_handles(datasets: Sequence[str] = ROW_ORDER) -> list[Line2D | Patch]:
    present = {_legend_key(method) for dataset in datasets for method in ROW_METHODS[dataset]}
    return [*(_method_handle(method) for method in LEGEND_ORDER if method in present), *_split_handles()]


def draw_legend(fig: Figure, datasets: Sequence[str], *, left: float, right: float, top: float) -> None:
    """One legend row centered between *left* and *right*, hanging from *top* (all in inches)."""
    width, height = fig.get_size_inches()
    handles = legend_handles(datasets)
    fig.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=((left + right) / 2 / width, top / height),
        ncol=len(handles),
        frameon=False,
        numpoints=1,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.4,
        handletextpad=0.4,
        columnspacing=0.7,
        borderpad=0.0,
        borderaxespad=0.0,
    )


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def _panel_filename(row_index: int, column: int, dataset: str) -> str:
    return f'grid_r{row_index + 1}c{column + 1}_{dataset}_{COLUMN_KINDS[column]}.pdf'


def _warn_if_clipped(fig: Figure, name: str) -> None:
    width, height = fig.get_size_inches()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    tolerance = 1e-3
    if min(bbox.x0, bbox.y0) < -tolerance or bbox.x1 > width + tolerance or bbox.y1 > height + tolerance:
        extents = ', '.join(f'{value:.3f}' for value in bbox.extents)
        print(f'WARNING: {name} draws outside its {width:.2f} x {height:.2f} in canvas: ({extents})')


def save_panels(grid: GridData, output_dir: Path) -> None:
    """Save title-free, legend-free vector panels plus the shared legend."""
    for row_index, row in enumerate(grid):
        for column in range(3):
            fig, ax = plt.subplots(figsize=PANEL_SIZE, constrained_layout=True)
            draw_panel(ax, row, column)
            brand.finish(fig)
            fig.savefig(output_dir / _panel_filename(row_index, column, row.dataset))
            plt.close(fig)

    fig = plt.figure(figsize=(SHEET_WIDTH, LEGEND_HEIGHT + 0.1))
    draw_legend(fig, ROW_ORDER, left=0.0, right=SHEET_WIDTH, top=LEGEND_HEIGHT + 0.05)
    fig.savefig(output_dir / 'legend.pdf')
    plt.close(fig)


def save_contact_sheet(grid: GridData, output_dir: Path, *, mock: bool, stem: str = 'grid_contact_sheet') -> None:
    """Save the assembled grid at exactly the NeurIPS text width, so it is included at 100%.

    Layout is in inches. Each row shares one y-axis, labelled once on the left.
    """
    rows = len(grid)
    panel_width = (SHEET_WIDTH - SHEET_LEFT - SHEET_RIGHT - 2 * SHEET_WSPACE) / 3
    axes_top = SHEET_BOTTOM + rows * SHEET_ROW_HEIGHT + (rows - 1) * SHEET_ROW_GAP
    height = axes_top + SHEET_TITLES + LEGEND_GAP + LEGEND_HEIGHT
    fig = plt.figure(figsize=(SHEET_WIDTH, height))

    axes: list[list[Axes]] = []
    for row_index, row in enumerate(grid):
        bottom = axes_top - SHEET_ROW_HEIGHT - row_index * (SHEET_ROW_HEIGHT + SHEET_ROW_GAP)
        row_axes: list[Axes] = []
        for column in range(3):
            left = SHEET_LEFT + column * (panel_width + SHEET_WSPACE)
            ax = fig.add_axes(
                (left / SHEET_WIDTH, bottom / height, panel_width / SHEET_WIDTH, SHEET_ROW_HEIGHT / height),
                sharey=row_axes[0] if row_axes else None,
            )
            draw_panel(ax, row, column)
            if column > 0:
                ax.tick_params(labelleft=False)
                ax.set_ylabel('')
            row_axes.append(ax)
        axes.append(row_axes)

    for column, label in enumerate(COLUMN_LABELS):
        position = axes[0][column].get_position()
        fig.text(
            (position.x0 + position.x1) / 2,
            position.y1 + 0.04 / height,
            label,
            ha='center',
            va='bottom',
            fontsize=TITLE_FONT_SIZE,
            fontweight='medium',
        )

    for row_index, row in enumerate(grid):
        position = axes[row_index][0].get_position()
        fig.text(
            0.02 / SHEET_WIDTH,
            (position.y0 + position.y1) / 2,
            ROW_LABELS[row.dataset],
            ha='left',
            va='center',
            multialignment='center',
            rotation=90,
            fontsize=ROW_LABEL_FONT_SIZE,
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
            color=brand.MUTED,
        )
    brand.finish(fig)
    draw_legend(
        fig,
        [row.dataset for row in grid],
        left=SHEET_LEFT,
        right=SHEET_WIDTH - SHEET_RIGHT,
        top=height - 0.02,
    )
    _warn_if_clipped(fig, stem)
    with plt.rc_context({'savefig.bbox': 'standard'}):
        fig.savefig(output_dir / f'{stem}.png', dpi=300)
        fig.savefig(output_dir / f'{stem}.pdf')
    plt.close(fig)


ALPHA: Final = 0.05
# (mean, sample std, number of seeds) of one plotted point.
type Arm = tuple[float, float, int]


def _series(panel: CurvePanel, method: str) -> CurveSeries:
    return next(series for series in panel.series if series.method == method)


def _arm(series: CurveSeries, index: int) -> Arm:
    return float(series.y[index]), float(series.error[index]), series.n_seeds[index]


def _welch_p(a: Arm, b: Arm) -> float:
    return float(ttest_ind_from_stats(*a, *b, equal_var=False).pvalue)


def _fmt_p(p: float) -> str:
    return r'< 10^{-3}' if p < 1e-3 else f'= {p:.2g}'


def _fmt_count(n: float) -> str:
    return f'{n / 1e6:.1f}M' if n >= 1e6 else f'{n / 1e3:.0f}K'


def tell_story(grid: GridData) -> dict[str, str]:
    """Recompute every number the caption quotes, and warn if a claim no longer holds."""
    rows = {row.dataset: row for row in grid}
    organ, pcam = rows['organ3d'], rows['pcam']
    failed: list[str] = []

    ours, rival = _series(organ.data, OURS), _series(organ.data, 'max_pool')
    full = _arm(rival, -1)
    gain_p = _welch_p(_arm(ours, -1), full)
    if ours.y[-1] <= full[0] or gain_p >= ALPHA:
        failed.append('bispectrum beats max pool on the full OrganMNIST3D training set')
    data_index = next(
        (i for i in range(len(ours.x)) if ours.y[i] > full[0] and _welch_p(_arm(ours, i), full) < ALPHA),
        len(ours.x) - 1,
    )
    if data_index == len(ours.x) - 1:
        failed.append('bispectrum on a smaller OrganMNIST3D training set beats max pool on all of it')

    ours_params, rival_params = _series(organ.params, OURS), _series(organ.params, 'max_pool')
    param_p = _welch_p(_arm(ours_params, 0), _arm(rival_params, -1))
    if ours_params.y[0] <= rival_params.y[-1] or param_p >= ALPHA:
        failed.append("bispectrum's smallest OrganMNIST3D model beats max pool's largest")

    pcam_ours = _series(pcam.data, OURS)
    rivals = [
        (_welch_p(_arm(series, i), _arm(pcam_ours, j)), series, i, j)
        for series in pcam.data.series
        if series.method != OURS
        for i, x in enumerate(series.x)
        for j in np.flatnonzero(pcam_ours.x == x)
        if series.y[i] > pcam_ours.y[j]
    ]
    rival_p, rival_series, i, j = min(rivals, key=lambda item: item[0])
    if rival_p < ALPHA:
        failed.append('no PCam model is significantly better than the bispectrum')
    spread = float(np.sqrt(np.mean(pcam.split.rotated_error**2)))
    mde = 2.8 * spread * math.sqrt(2 / 3)  # (z_0.975 + z_0.8) * s * sqrt(2 / n), n = 3
    for claim in failed:
        print(f'WARNING: caption claim no longer holds: {claim}')

    full_n = int(ours.x[-1])
    return {
        'gain': f'{(ours.y[-1] - full[0]) * 100:.1f}',
        'gain_p': _fmt_p(gain_p),
        'ratio': f'{rival_params.x[-1] / ours_params.x[0]:.0f}',
        'ours_params': _fmt_count(ours_params.x[0]),
        'rival_params': _fmt_count(rival_params.x[-1]),
        'param_gain': f'{(ours_params.y[0] - rival_params.y[-1]) * 100:.1f}',
        'param_p': _fmt_p(param_p),
        'data_n': f'{ours.x[data_index]:.0f}',
        'full_n': f'{full_n}',
        'data_gain': f'{(ours.y[data_index] - full[0]) * 100:.1f}',
        'data_p': _fmt_p(_welch_p(_arm(ours, data_index), full)),
        'rival': METHOD_STYLE[rival_series.method].label,
        'rival_at': (
            'on the full training set'
            if rival_series.x[i] == pcam_ours.x[-1]
            else f'at $n = {_size_tick(rival_series.x[i])}$'
        ),
        'rival_gap': f'{rival_series.y[i] - pcam_ours.y[j]:.3f}',
        'rival_p': _fmt_p(rival_p),
        'mde': f'{mde:.3f}',
    }


def fill(template: str, values: dict[str, str]) -> str:
    for name, value in values.items():
        template = template.replace(f'@{name}@', value)
    if '@' in template:
        raise ValueError(f'unfilled caption field in: {template}')
    return template


def report_seed_counts(grid: GridData) -> None:
    """Print the seeds behind every plotted point (the caption states them)."""
    for row in grid:
        print(f'{row.dataset} full-data split: seeds {sorted(set(row.split.n_seeds))}')
        for name, panel in (('params', row.params), ('data', row.data)):
            by_x: dict[str, set[int]] = defaultdict(set)
            for series in panel.series:
                for x, n in zip(series.x, series.n_seeds, strict=True):
                    by_x[_size_tick(x) if name == 'data' else 'all'].add(n)
            detail = ', '.join(f'{x}: {sorted(ns)}' for x, ns in by_x.items())
            print(f'{row.dataset} {name}: seeds {detail}')


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

    if not args.mock:
        report_seed_counts(grid)
    save_panels(grid, output_dir)
    save_contact_sheet(grid, output_dir, mock=args.mock)
    for row in grid:
        save_contact_sheet((row,), output_dir, mock=args.mock, stem=f'grid_row_{row.dataset}')
    if not args.mock:
        story = tell_story(grid)
        (output_dir / 'caption.tex').write_text(fill(CAPTION_TEX, story))
        (output_dir / 'caption_details.tex').write_text(fill(DETAILS_TEX, story))

    print(f'Wrote grid assets to {output_dir}')


if __name__ == '__main__':
    main()
