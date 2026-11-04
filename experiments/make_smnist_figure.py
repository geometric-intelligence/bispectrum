#!/usr/bin/env python3
"""Spherical MNIST figure and protocol table: the SO(3) extension against matched baselines.

Panels:
    a. Protocols           — NR/NR, R/R and NR/R accuracy per model (Cohen et al., 2018).
    b. Accuracy vs latency — rotated-test accuracy against H100 forward latency.
    c. Data efficiency     — rotated-test accuracy against training examples.

Also writes the protocol table (LaTeX + Markdown), including the published S2CNN row.

Model configurations:
    - Bispectrum, S2CNN + BN and the tensor-product net were each selected on
      validation accuracy (seed 42, NR, full data) from a batch-size x learning-rate grid
      (the tensor-product grid also covers architecture); every panel and table uses the
      selected configs, reused unchanged at every training-set size. The CNN and power
      spectrum use the train.py defaults.
    - Panel c and the data-efficiency table: rotated-test accuracy in each model's headline
      training mode; Welch tests of ours vs. S2CNN + BN and vs. the tensor product at every
      size, Holm-corrected over all of them. R-trained S2CNN + BN is a sensitivity row.

Usage:
    python make_smnist_figure.py [--output_dir DIR]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from make_grid_figure import (
    BASELINE_WEIGHT,
    DODGE_DECADES,
    EXPERIMENTS_DIR,
    OURS_WEIGHT,
    LineWeight,
    _canonical_mode,
    _is_full_train,
    _load_runs,
    _mean_std,
    _percent_axis,
    _size_tick,
    configure_style,
    style_axes,
)
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter, NullLocator
from scipy import stats

REPO_ROOT: Final = EXPERIMENTS_DIR.parent
SMNIST_DIR: Final = EXPERIMENTS_DIR / 'spherical_mnist'
FIGURE_SIZE: Final = (5.5, 2.05)
LATENCY_BATCH: Final = 16
DEFAULT_SUBSET_DIR: Final = SMNIST_DIR / 'smnist_results'
MATCHED_SUBSET_DIR: Final = SMNIST_DIR / 'smnist_results_baselines_data'
MATCHED_WEIGHT: Final = LineWeight(linewidth=1.0, markersize=3.4, elinewidth=0.6, capsize=0.8, alpha=0.9, zorder=3)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    table_label: str
    results_dir: Path
    run_label: str
    timing_case: str
    color: str
    marker: str
    headline_mode: str  # training protocol plotted against latency: 'C' -> NR/R, 'R' -> R/R
    subset_dir: Path = DEFAULT_SUBSET_DIR  # train_size > 0 runs with the same run_label and config


MODELS: Final = (
    ModelSpec(
        'standard',
        'CNN',
        'CNN (equirectangular)',
        SMNIST_DIR / 'smnist_results',
        'standard',
        'standard',
        '#888888',
        's',
        'R',
    ),
    ModelSpec(
        'power_spectrum',
        'Power spectrum',
        'Power spectrum + MLP',
        SMNIST_DIR / 'smnist_results',
        'power_spectrum',
        'power_spectrum',
        '#6D4C41',
        'v',
        'C',
    ),
    ModelSpec(
        's2cnn_bn',
        'S2CNN + BN',
        'S2CNN + BatchNorm (matched)',
        SMNIST_DIR / 'smnist_results_baselines',
        's2cnn_bn',
        's2cnn_bn',
        '#1565C0',
        'h',
        'C',
        MATCHED_SUBSET_DIR,
    ),
    ModelSpec(
        'tensor_product',
        'Tensor product',
        'Tensor-product net (matched)',
        SMNIST_DIR / 'smnist_results_baselines',
        'tensor_product',
        'tp_c24_l6',
        '#2E7D32',
        'X',
        'C',
        MATCHED_SUBSET_DIR,
    ),
    ModelSpec(
        'bispectrum',
        'Bispectrum (ours)',
        'Bispectrum + MLP (ours)',
        SMNIST_DIR / 'smnist_results_bispectrum_tuned',
        'bispectrum_bs32_lr1e-3',
        'bispectrum',
        '#E53935',
        '*',
        'C',
        MATCHED_SUBSET_DIR,
    ),
)
MATCHED_KEYS: Final = ('s2cnn_bn', 'tensor_product')
# Sensitivity row for the data-efficiency table: S2CNN + BN is the only model that is not exactly
# invariant, so it may gain from rotation augmentation where the others cannot.
S2CNN_AUGMENTED: Final = replace(MODELS[2], key='s2cnn_bn_r', label='S2CNN + BN, R-trained', headline_mode='R')
# The bispectrum at the train.py defaults (batch 256, lr 1e-3), for the data-efficiency table.
BISPECTRUM_DEFAULT: Final = replace(
    MODELS[4],
    key='bispectrum_default',
    label='Bispectrum, default settings',
    results_dir=DEFAULT_SUBSET_DIR,
    run_label='bispectrum',
    subset_dir=DEFAULT_SUBSET_DIR,
)

# Cohen et al. (2018), Table 2: transcribed, not re-run.
PUBLISHED_S2CNN: Final = {'NR/NR': 96.0, 'R/R': 95.0, 'NR/R': 94.0}
# Cohen et al.'s exact architecture and training settings (no BatchNorm), re-run by us.
AS_PUBLISHED: Final = ModelSpec(
    's2cnn',
    'S2CNN',
    'S2CNN, published architecture',
    SMNIST_DIR / 'smnist_results_baselines',
    's2cnn',
    's2cnn',
    '#90A4AE',
    'h',
    'C',
)

PROTOCOLS: Final = ('NR/NR', 'R/R', 'NR/R')
# Protocol -> (training mode, test split).
PROTOCOL_SPLIT: Final = {'NR/NR': ('C', 'test_nr'), 'R/R': ('R', 'test_r'), 'NR/R': ('C', 'test_r')}
PROTOCOL_STYLE: Final = {
    'NR/NR': {'alpha': 1.0, 'hatch': None},
    'R/R': {'alpha': 0.7, 'hatch': '....'},
    'NR/R': {'alpha': 0.42, 'hatch': '////'},
}
PROTOCOL_LABEL: Final = {
    'NR/NR': 'NR/NR (train, test unrotated)',
    'R/R': 'R/R (train, test rotated)',
    'NR/R': 'NR/R (train unrotated, test rotated)',
}

# Label placement in the latency panel: (dx, dy) in points, horizontal and vertical alignment.
LATENCY_LABEL_OFFSET: Final = {
    'standard': ((0, 5), 'center', 'bottom'),
    'power_spectrum': ((0, -5), 'center', 'top'),
    's2cnn_bn': ((0, -5), 'center', 'top'),
    'tensor_product': ((0, -5), 'center', 'top'),
    'bispectrum': ((0, -5), 'center', 'top'),
}

CAPTION_TEXT: Final = """\
Spherical MNIST: the augmented selective SO(3) invariant against matched
learned-equivariant baselines (5 seeds, mean +/- sample std). (a) Accuracy
under the three protocols of Cohen et al. (2018): NR/NR (train and test
unrotated), R/R (train and test rotated) and NR/R (train unrotated, test
rotated). (b) Rotated-test accuracy against end-to-end forward latency
(spherical grid in, logits out; float32, batch 16, H100, median); invariant
and equivariant models trained unrotated (NR/R), the CNN trained with rotation
augmentation (R/R); labels give trainable parameters. (c) Rotated-test
accuracy against training-set size, each model in its panel (b) training mode.
Bispectrum, S2CNN + BatchNorm and the tensor-product net were each selected on
validation accuracy from a batch-size x learning-rate grid at full data, and the
selected configs are reused unchanged at every training-set size; the CNN and
power spectrum use the default settings (batch 256, lr 1e-3).
"""

TABLE_CAPTION: Final = """\
Spherical MNIST, test accuracy (%) under the protocols of Cohen et al. (2018),
mean +/- sample std over 5 seeds, and end-to-end forward latency (spherical grid
in, logits out; float32, batch 16, H100, median). Matched baselines are our runs
in the same training pipeline; the published S2CNN row is transcribed from Cohen
et al. (2018). Bispectrum, S2CNN + BatchNorm and the tensor-product net were each
selected on validation accuracy from a batch-size x learning-rate grid (the
tensor-product grid also covers channels and internal band-limit); the CNN and
power spectrum use the default settings (batch 256, lr 1e-3). S2CNN with the
published architecture and training settings (no BatchNorm) reaches
{as_published} in our pipeline.
"""


@dataclass(frozen=True)
class ProtocolStats:
    values: dict[str, list[float]]  # protocol -> per-seed accuracy (%)
    n_params: int
    latency_ms: float | None

    def mean(self, protocol: str) -> float:
        return _mean_std(self.values[protocol])[0]

    def std(self, protocol: str) -> float:
        return _mean_std(self.values[protocol])[1]

    def headline(self, mode: str) -> list[float]:
        return self.values['NR/R' if mode == 'C' else 'R/R']


def _load_timing(path: Path) -> dict[str, tuple[float, int]]:
    """Timing case -> (median ms, params) at LATENCY_BATCH."""
    data = json.loads(path.read_text())
    return {
        r['model']: (float(r['median_ms']), int(r['n_params']))
        for r in data['results']
        if r['batch_size'] == LATENCY_BATCH and r.get('error') is None
    }


def load_protocol_stats(spec: ModelSpec, timing: dict[str, tuple[float, int]]) -> ProtocolStats:
    runs = [r for r in _load_runs(spec.results_dir) if r.get('model') == spec.run_label and _is_full_train(r)]
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_mode[r['train_mode']].append(r)
    values: dict[str, list[float]] = {}
    for protocol, (mode, split) in PROTOCOL_SPLIT.items():
        seeds = sorted(by_mode[mode], key=lambda r: r['seed'])
        if not seeds:
            raise SystemExit(f'{spec.key}: no {mode}-trained runs in {spec.results_dir}')
        values[protocol] = [100.0 * r[split]['accuracy'] for r in seeds]
    n_params = int(runs[0]['n_params'])
    latency = timing.get(spec.timing_case)
    if latency is not None and latency[1] != n_params:
        raise SystemExit(f'{spec.key}: timing case {spec.timing_case} has {latency[1]} params, runs have {n_params}')
    return ProtocolStats(values=values, n_params=n_params, latency_ms=latency[0] if latency else None)


@dataclass(frozen=True)
class DataCurve:
    values: dict[int, list[float]]  # training examples -> per-seed rotated-test accuracy (%)
    full_examples: int | None

    @property
    def sizes(self) -> list[int]:
        return sorted(self.values)


def load_data_curve(spec: ModelSpec) -> DataCurve:
    """Rotated-test accuracy per training-set size, in the spec's headline training mode."""

    def matching(root: Path, *, full: bool) -> list[dict]:
        return [
            r
            for r in _load_runs(root)
            if r.get('model') == spec.run_label
            and _canonical_mode(r.get('train_mode', 'C')) == spec.headline_mode
            and _is_full_train(r) == full
        ]

    full_runs = matching(spec.results_dir, full=True)
    by_size: dict[int, dict[int, float]] = defaultdict(dict)
    for r in (*matching(spec.subset_dir, full=False), *full_runs):
        size, seed = int(r['train_examples']), int(r['seed'])
        if seed in by_size[size]:
            raise SystemExit(f'{spec.key}: two runs for seed {seed} at n={size}')
        by_size[size][seed] = 100.0 * r['test_r']['accuracy']
    values = {size: [acc for _, acc in sorted(seeds.items())] for size, seeds in sorted(by_size.items())}
    return DataCurve(values=values, full_examples=int(full_runs[0]['train_examples']) if full_runs else None)


def _params_label(n: int) -> str:
    return f'{n / 1000:.0f}K' if n < 1_000_000 else f'{n / 1e6:.1f}M'


def _welch_p(a: list[float], b: list[float]) -> float:
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------


def draw_protocol_panel(ax: Axes, results: dict[str, ProtocolStats]) -> None:
    width = 0.27
    error_kw = {'elinewidth': 0.5, 'capthick': 0.5}
    for index, spec in enumerate(MODELS):
        stat = results[spec.key]
        for offset, protocol in zip((-width, 0.0, width), PROTOCOLS, strict=True):
            style = PROTOCOL_STYLE[protocol]
            ax.bar(
                index + offset,
                stat.mean(protocol) / 100,
                width,
                yerr=stat.std(protocol) / 100,
                color=spec.color,
                edgecolor=spec.color,
                linewidth=0.5,
                alpha=style['alpha'],
                hatch=style['hatch'],
                capsize=0.8,
                error_kw=error_kw,
                zorder=3,
            )
        collapse = stat.mean('NR/R')
        if collapse < 50:
            ax.annotate(
                f'{collapse:.0f}%',
                (index + width, collapse / 100),
                xytext=(0, 2),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=5,
                color='#444444',
            )
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels([spec.label for spec in MODELS], rotation=30, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel('Test accuracy')
    _percent_axis(ax, 'accuracy')
    style_axes(ax)


def draw_latency_panel(ax: Axes, results: dict[str, ProtocolStats]) -> None:
    for spec in MODELS:
        stat = results[spec.key]
        if stat.latency_ms is None:
            continue
        mean, std = _mean_std(stat.headline(spec.headline_mode))
        ours = spec.key == 'bispectrum'
        ax.errorbar(
            stat.latency_ms,
            mean / 100,
            yerr=std / 100,
            color=spec.color,
            marker=spec.marker,
            markersize=6.5 if ours else 4.2,
            markeredgecolor='white',
            markeredgewidth=0.4,
            elinewidth=0.6,
            capsize=1.0,
            linestyle='none',
            zorder=4 if ours else 3,
        )
        (dx, dy), ha, va = LATENCY_LABEL_OFFSET[spec.key]
        ax.annotate(
            _params_label(stat.n_params),
            (stat.latency_ms, mean / 100),
            xytext=(dx, dy),
            textcoords='offset points',
            ha=ha,
            va=va,
            fontsize=5,
            color=spec.color if ours else '#333333',
            linespacing=1.1,
        )
    ax.set_xscale('log')
    ax.set_xlim(0.1, 300)
    ax.set_ylim(0.72, 1.0)
    ax.set_xlabel(f'H100 latency, batch {LATENCY_BATCH} (ms)')
    ax.set_ylabel('Rotated test accuracy')
    _percent_axis(ax, 'accuracy')
    style_axes(ax)


def _size_label(n: int, full_examples: set[int]) -> str:
    return 'full' if n in full_examples else _size_tick(n)


def draw_data_panel(ax: Axes, curves: dict[str, DataCurve]) -> None:
    shown = [spec for spec in MODELS if curves[spec.key].values]
    for index, spec in enumerate(shown):
        curve = curves[spec.key]
        moments = np.asarray([_mean_std(curve.values[n]) for n in curve.sizes]) / 100
        if spec.key == 'bispectrum':
            weight = OURS_WEIGHT
        elif spec.key in MATCHED_KEYS:
            weight = MATCHED_WEIGHT
        else:
            weight = BASELINE_WEIGHT
        shift = 10 ** ((index - (len(shown) - 1) / 2) * DODGE_DECADES)
        ax.errorbar(
            np.asarray(curve.sizes, dtype=np.float64) * shift,
            moments[:, 0],
            yerr=moments[:, 1],
            color=spec.color,
            marker=spec.marker,
            linewidth=weight.linewidth,
            markersize=weight.markersize,
            markeredgewidth=0.4,
            markeredgecolor='white',
            capsize=weight.capsize,
            elinewidth=weight.elinewidth,
            alpha=weight.alpha,
            zorder=weight.zorder,
        )
    full_examples = {curves[spec.key].full_examples for spec in shown} - {None}
    ticks = sorted({n for spec in shown for n in curves[spec.key].sizes})
    ax.set_xscale('log')
    ax.set_xticks(ticks)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([_size_label(n, full_examples) for n in ticks], rotation=35, ha='right', rotation_mode='anchor')
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Rotated test accuracy')
    _percent_axis(ax, 'accuracy')
    style_axes(ax)


def model_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=spec.color,
            marker=spec.marker,
            linestyle='none',
            markersize=5.5 if spec.key == 'bispectrum' else 4.0,
            markeredgecolor='white',
            markeredgewidth=0.4,
            label=spec.label,
        )
        for spec in MODELS
    ]


def protocol_handles() -> list[Patch]:
    return [
        Patch(
            facecolor='#777777',
            edgecolor='#777777',
            alpha=PROTOCOL_STYLE[protocol]['alpha'],
            hatch=PROTOCOL_STYLE[protocol]['hatch'],
            linewidth=0.5,
            label=PROTOCOL_LABEL[protocol],
        )
        for protocol in PROTOCOLS
    ]


def save_figure(results: dict[str, ProtocolStats], curves: dict[str, DataCurve], output_dir: Path) -> None:
    fig = plt.figure(figsize=FIGURE_SIZE)
    height = FIGURE_SIZE[1]
    gridspec = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.35, 1.0, 1.0),
        left=0.075,
        right=0.99,
        bottom=0.5 / height,
        top=1.0 - 0.62 / height,
        wspace=0.42,
    )
    axes = [fig.add_subplot(gridspec[0, column]) for column in range(3)]
    draw_protocol_panel(axes[0], results)
    draw_latency_panel(axes[1], results)
    draw_data_panel(axes[2], curves)

    for ax, title in zip(axes, ('(a) Protocols', '(b) Accuracy vs. latency', '(c) Data efficiency'), strict=True):
        ax.set_title(title, fontsize=7, pad=4)
    for handles, anchor in ((model_handles(), 1.0), (protocol_handles(), 1.0 - 0.16 / height)):
        fig.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.53, anchor),
            ncol=len(handles),
            frameon=False,
            fontsize=5.8,
            handlelength=1.6,
            columnspacing=1.4,
        )
    fig.savefig(output_dir / 'smnist_figure.pdf', bbox_inches='tight')
    fig.savefig(output_dir / 'smnist_figure.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    for name, draw, data in (
        ('protocols', draw_protocol_panel, results),
        ('latency', draw_latency_panel, results),
        ('data_efficiency', draw_data_panel, curves),
    ):
        fig, ax = plt.subplots(figsize=(2.2, 1.6), constrained_layout=True)
        draw(ax, data)
        fig.savefig(output_dir / f'smnist_{name}.pdf')
        plt.close(fig)


# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------


def _cell(values: list[float]) -> str:
    mean, std = _mean_std(values)
    return f'{mean:.1f} ± {std:.1f}'


def _tex_cell(values: list[float]) -> str:
    mean, std = _mean_std(values)
    return f'${mean:.1f} {{\\pm}} {std:.1f}$'


def _latency(stat: ProtocolStats) -> str:
    if stat.latency_ms is None:
        return '---'
    return f'{stat.latency_ms:.2f}' if stat.latency_ms < 10 else f'{stat.latency_ms:.1f}'


def write_tables(results: dict[str, ProtocolStats], as_published: ProtocolStats, output_dir: Path) -> None:
    ours = results['bispectrum']
    published = ' / '.join(f'{as_published.mean(p):.1f}' for p in PROTOCOLS)

    md = [
        '| Model | Params | NR/NR | R/R | NR/R | Latency (ms) | p vs. ours (NR/NR, R/R, NR/R) |',
        '|---|---:|---|---|---|---:|---|',
    ]
    tex = [
        r'\begin{tabular}{lrcccr}',
        r'\toprule',
        r'Model & Params & NR/NR & R/R & NR/R & Latency (ms) \\',
        r'\midrule',
    ]
    rows: list[tuple[str, ProtocolStats | None]] = [
        (MODELS[0].table_label, results['standard']),
        (MODELS[1].table_label, results['power_spectrum']),
        ('S2CNN, published', None),
        (MODELS[2].table_label, results['s2cnn_bn']),
        (MODELS[3].table_label, results['tensor_product']),
        (MODELS[4].table_label, ours),
    ]
    for label, stat in rows:
        if stat is None:
            cells = [f'{PUBLISHED_S2CNN[p]:.0f}' for p in PROTOCOLS]
            md.append(f'| {label} (Cohen et al., 2018) | --- | {" | ".join(cells)} | --- | --- |')
            tex.append(f'S2CNN~\\citep{{cohen2018spherical}}, published & --- & {" & ".join(cells)} & --- \\\\')
            continue
        p_values = (
            '---' if stat is ours else ', '.join(f'{_welch_p(ours.values[p], stat.values[p]):.2g}' for p in PROTOCOLS)
        )
        md.append(
            f'| {label} | {_params_label(stat.n_params)} | '
            + ' | '.join(_cell(stat.values[p]) for p in PROTOCOLS)
            + f' | {_latency(stat)} | {p_values} |'
        )
        name = f'\\textbf{{{label}}}' if stat is ours else label
        tex.append(
            f'{name} & {_params_label(stat.n_params)} & '
            + ' & '.join(_tex_cell(stat.values[p]) for p in PROTOCOLS)
            + f' & {_latency(stat)} \\\\'
        )
        if label in (MODELS[1].table_label, MODELS[3].table_label):
            tex.append(r'\midrule')
    tex += [r'\bottomrule', r'\end{tabular}']

    caption = TABLE_CAPTION.format(as_published=published + ' (NR/NR / R/R / NR/R)')
    (output_dir / 'smnist_protocol_table.md').write_text('\n'.join(md) + '\n\n' + caption)
    (output_dir / 'smnist_protocol_table.tex').write_text('\n'.join(tex) + '\n')
    (output_dir / 'smnist_table_caption.txt').write_text(caption)
    print('\n'.join(md))


def _holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def write_data_table(curves: dict[str, DataCurve], output_dir: Path) -> None:
    ours = curves['bispectrum']
    sizes = ours.sizes
    full_examples = {ours.full_examples} - {None}
    expected_seeds = max(len(v) for v in ours.values.values())
    lines = [
        '| Model (rotated test, %) | ' + ' | '.join(_size_label(n, full_examples) for n in sizes) + ' |',
        '|---|' + '---|' * len(sizes),
    ]
    for spec in (*MODELS, S2CNN_AUGMENTED, BISPECTRUM_DEFAULT):
        curve = curves[spec.key]
        cells = []
        for n in sizes:
            seeds = curve.values.get(n, [])
            cell = _cell(seeds) if len(seeds) > 1 else '---'
            cells.append(cell if len(seeds) in (0, expected_seeds) else f'{cell} ({len(seeds)} seeds)')
        lines.append(f'| {spec.label} | ' + ' | '.join(cells) + ' |')

    tests: list[tuple[str, int, float, float]] = []
    for spec in MODELS:
        if spec.key not in MATCHED_KEYS:
            continue
        other = curves[spec.key]
        for n in sizes:
            if len(other.values.get(n, [])) > 1 and len(ours.values[n]) > 1:
                diff = _mean_std(ours.values[n])[0] - _mean_std(other.values[n])[0]
                tests.append((spec.label, n, diff, _welch_p(ours.values[n], other.values[n])))
    lines += ['', '| Ours minus | n | Δ (pp) | Welch p | Holm p |', '|---|---|---:|---:|---:|']
    for (label, n, diff, p), p_holm in zip(tests, _holm([t[3] for t in tests]), strict=True):
        lines.append(f'| {label} | {_size_label(n, full_examples)} | {diff:+.2f} | {p:.2g} | {p_holm:.2g} |')
    lines += [
        '',
        f'Holm over the {len(tests)} tests above. Configs selected at full data and reused at every n; '
        'CNN R-trained, all other rows NR-trained unless marked.',
    ]
    text = '\n'.join(lines) + '\n'
    (output_dir / 'smnist_data_efficiency.md').write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description='Spherical MNIST figure and protocol table')
    parser.add_argument('--output_dir', type=Path, default=EXPERIMENTS_DIR / 'smnist_figure')
    parser.add_argument(
        '--timing', type=Path, default=REPO_ROOT / 'rebuttal' / 'results' / 'smnist_baselines' / 'timing_h100.json'
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    timing = _load_timing(args.timing)
    results = {spec.key: load_protocol_stats(spec, timing) for spec in MODELS}
    as_published = load_protocol_stats(AS_PUBLISHED, timing)
    curves = {spec.key: load_data_curve(spec) for spec in (*MODELS, S2CNN_AUGMENTED, BISPECTRUM_DEFAULT)}

    save_figure(results, curves, args.output_dir)
    write_tables(results, as_published, args.output_dir)
    write_data_table(curves, args.output_dir)
    (args.output_dir / 'caption.txt').write_text(CAPTION_TEXT)
    print(f'Wrote Spherical MNIST assets to {args.output_dir}')


if __name__ == '__main__':
    main()
