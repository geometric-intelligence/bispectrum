#!/usr/bin/env python3
"""Analyze Spherical MNIST experiment results.

Produces:
  1. A NeurIPS-quality 2-panel figure (PDF + PNG):
     (a) Cohen et al. (2018) protocol bar chart (NR/NR, R/R, NR/R)
     (b) Rotation robustness strip plot under random SO(3)
  2. LaTeX-formatted table rows for direct insertion into the paper.
  3. Console summary table.

Usage:
    python analyze_results.py [--results_dir ./smnist_results]
        [--matched_dir ./smnist_results_matched]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

MODEL_ORDER = ['standard', 'power_spectrum', 'power_spectrum_matched', 'bispectrum']
MODEL_LABELS = {
    'standard': 'Std. CNN',
    'power_spectrum': 'PowSpec 11K',
    'power_spectrum_matched': 'PowSpec 166K',
    'bispectrum': 'Bispectrum',
}

MODEL_LABELS_FULL = {
    'standard': 'Standard CNN',
    'power_spectrum': 'Power Spectrum (11K)',
    'power_spectrum_matched': 'Power Spectrum (166K)',
    'bispectrum': 'Bispectrum (ours)',
}

COHEN_RESULTS = {
    'Cohen planar CNN': {'NR/NR': 0.98, 'R/R': 0.23, 'NR/R': 0.11},
    'Cohen spherical CNN': {'NR/NR': 0.96, 'R/R': 0.95, 'NR/R': 0.94},
}

NEURIPS_RCPARAMS = {
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'text.usetex': False,
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.4,
    'ytick.major.width': 0.4,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'lines.linewidth': 0.8,
    'patch.linewidth': 0.4,
    'grid.linewidth': 0.3,
    'grid.alpha': 0.4,
}

COLORS = {
    'standard': '#7f7f7f',
    'power_spectrum': '#aec7e8',
    'power_spectrum_matched': '#1f77b4',
    'bispectrum': '#d62728',
}

PROTOCOL_COLORS = {
    'NR/NR': '#2D6A9F',
    'R/R': '#4CAF50',
    'NR/R': '#D4722A',
}


def load_results(
    results_dir: str,
    matched_dir: str | None = None,
) -> dict[tuple[str, str], list[dict]]:
    """Load all results.json files, keyed by (model, train_mode).

    If *matched_dir* is provided, loads matched-param power spectrum runs
    from that directory under the key ``power_spectrum_matched``.
    """
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    results_path = Path(results_dir)
    if not results_path.exists():
        return grouped

    for p in sorted(results_path.glob('*/results.json')):
        with open(p) as f:
            r = json.load(f)
        grouped[(r['model'], r['train_mode'])].append(r)

    if matched_dir:
        matched_path = Path(matched_dir)
        if matched_path.exists():
            for p in sorted(matched_path.glob('*/results.json')):
                with open(p) as f:
                    r = json.load(f)
                grouped[('power_spectrum_matched', r['train_mode'])].append(r)

    return grouped


def _fmt(vals: list[float]) -> str:
    m = np.mean(vals)
    if len(vals) > 1:
        s = np.std(vals)
        return f'{m:.3f}\u00b1{s:.3f}'
    return f'{m:.3f}'


def _latex_fmt(vals: list[float], bold: bool = False) -> str:
    m = np.mean(vals)
    if len(vals) > 1:
        s = np.std(vals)
        core = f'{m:.3f} \\pm {s:.3f}'
    else:
        core = f'{m:.3f}'
    if bold:
        return f'$\\mathbf{{{core}}}$'
    return f'${core}$'


def print_cohen_table(grouped: dict[tuple[str, str], list[dict]]):
    """Print the NR/NR, R/R, NR/R accuracy table to console."""
    print('\n' + '=' * 90)
    print('ACCURACY TABLE (Cohen et al. 2018 protocol)')
    print('Train/Test: X/Y = trained on X, evaluated on Y')
    print('=' * 90)
    hdr = f'{"Method":<28} {"Params":>8}  {"NR/NR":>12}  {"R/R":>12}  {"NR/R":>12}  {"Rot \u03c3":>10}'
    print(hdr)
    print('-' * 90)

    for method, accs in COHEN_RESULTS.items():
        print(
            f'{method:<28} {"\u2014":>8}  '
            f'{accs["NR/NR"]:>12.3f}  {accs["R/R"]:>12.3f}  {accs["NR/R"]:>12.3f}  '
            f'{"\u2014":>10}'
        )

    print('-' * 90)

    for model in MODEL_ORDER:
        nr_runs = grouped.get((model, 'NR'), [])
        r_runs = grouped.get((model, 'R'), [])
        if not nr_runs and not r_runs:
            continue

        n_params = (nr_runs or r_runs)[0]['n_params']
        nr_nr = [r['test_nr']['accuracy'] for r in nr_runs]
        nr_r = [r['test_r']['accuracy'] for r in nr_runs]
        r_r = [r['test_r']['accuracy'] for r in r_runs]

        rot_stds = [
            r['rotation_robustness']['std_accuracy']
            for r in nr_runs
            if r.get('rotation_robustness', {}).get('std_accuracy') is not None
        ]
        rot_str = f'{np.mean(rot_stds):.5f}' if rot_stds else '\u2014'

        label = MODEL_LABELS_FULL.get(model, model)
        print(
            f'{label:<28} {n_params:>8,}  '
            f'{_fmt(nr_nr):>12}  {_fmt(r_r):>12}  {_fmt(nr_r):>12}  '
            f'{rot_str:>10}'
        )


def print_latex_table(grouped: dict[tuple[str, str], list[dict]]):
    """Print LaTeX-formatted table rows for direct insertion into paper."""
    print('\n% LaTeX table rows for Spherical MNIST results')
    print('% Paste into \\begin{tabular}{lrcccc}')
    print(r'%   Model & Params & NR/NR & R/R & NR/R & Rot.\ $\sigma$ \\')

    for method, accs in COHEN_RESULTS.items():
        print(
            f'    {method} & --- '
            f'& ${accs["NR/NR"]:.2f}$ & ${accs["R/R"]:.2f}$ '
            f'& ${accs["NR/R"]:.2f}$ & --- \\\\'
        )
    print(r'    \midrule')

    for model in MODEL_ORDER:
        nr_runs = grouped.get((model, 'NR'), [])
        r_runs = grouped.get((model, 'R'), [])
        if not nr_runs and not r_runs:
            continue

        n_params = (nr_runs or r_runs)[0]['n_params']
        nr_nr = [r['test_nr']['accuracy'] for r in nr_runs]
        nr_r = [r['test_r']['accuracy'] for r in nr_runs]
        r_r = [r['test_r']['accuracy'] for r in r_runs]

        rot_stds = [
            r['rotation_robustness']['std_accuracy']
            for r in nr_runs
            if r.get('rotation_robustness', {}).get('std_accuracy') is not None
        ]

        is_bisp = model == 'bispectrum'
        label = MODEL_LABELS_FULL.get(model, model)
        if is_bisp:
            label = f'\\textbf{{{label}}}'

        params_str = f'{n_params // 1000}K'
        rot_val = f'{np.mean(rot_stds):.4f}' if rot_stds else '---'

        print(
            f'    {label} & {params_str} '
            f'& {_latex_fmt(nr_nr, bold=is_bisp)} '
            f'& {_latex_fmt(r_r, bold=is_bisp)} '
            f'& {_latex_fmt(nr_r, bold=is_bisp)} '
            f'& ${rot_val}$ \\\\'
        )


def _clean_axes(ax: plt.Axes):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)


def plot_combined_figure(
    grouped: dict[tuple[str, str], list[dict]],
    output_path: str,
):
    """NeurIPS-quality 2-panel figure: (a) Cohen protocol, (b) rotation robustness."""
    plt.rcParams.update(NEURIPS_RCPARAMS)

    models_with_data = [
        m for m in MODEL_ORDER
        if grouped.get((m, 'NR')) or grouped.get((m, 'R'))
    ]
    if not models_with_data:
        print('No data to plot.')
        return

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(6.5, 3.0),
        gridspec_kw={'width_ratios': [1.4, 1], 'wspace': 0.40},
    )
    fig.subplots_adjust(bottom=0.27)

    _plot_cohen_panel(ax_a, grouped, models_with_data)
    _plot_rotation_panel(ax_b, grouped, models_with_data)

    ax_a.text(
        -0.08, 1.05, '(a)', transform=ax_a.transAxes,
        fontsize=10, fontweight='bold', va='bottom',
    )
    ax_b.text(
        -0.10, 1.05, '(b)', transform=ax_b.transAxes,
        fontsize=10, fontweight='bold', va='bottom',
    )

    fig.savefig(output_path, dpi=300)
    pdf_path = output_path.replace('.png', '.pdf')
    fig.savefig(pdf_path)
    print(f'\nFigure saved to {output_path} and {pdf_path}')
    plt.close(fig)


def _plot_cohen_panel(
    ax: plt.Axes,
    grouped: dict[tuple[str, str], list[dict]],
    models: list[str],
):
    """Panel (a): Grouped bar chart of NR/NR, R/R, NR/R accuracy."""
    x = np.arange(len(models))
    width = 0.22
    protocols = ['NR/NR', 'R/R', 'NR/R']

    vals_prev: list[float] = []
    for i, proto in enumerate(protocols):
        vals = []
        errs = []
        for m in models:
            if proto == 'NR/NR':
                runs = grouped.get((m, 'NR'), [])
                accs = [r['test_nr']['accuracy'] for r in runs]
            elif proto == 'R/R':
                runs = grouped.get((m, 'R'), [])
                accs = [r['test_r']['accuracy'] for r in runs]
            else:
                runs = grouped.get((m, 'NR'), [])
                accs = [r['test_r']['accuracy'] for r in runs]
            vals.append(np.mean(accs) if accs else 0)
            errs.append(np.std(accs) if len(accs) > 1 else 0)

        offset = (i - 1) * width
        bars = ax.bar(
            x + offset, vals, width, yerr=errs,
            label=proto, color=PROTOCOL_COLORS[proto],
            edgecolor='white', linewidth=0.4,
            capsize=2, error_kw={'linewidth': 0.6, 'capthick': 0.6},
            zorder=3,
        )
        for j, (bar, v) in enumerate(zip(bars, vals)):
            if v <= 0:
                continue
            show_label = True
            if i > 0 and v > 0.2:
                prev_v = vals_prev[j] if j < len(vals_prev) else 0
                if abs(v - prev_v) < 0.03:
                    show_label = False
            if show_label:
                y_pos = max(bar.get_height(), 0) + 0.02
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_pos,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=5.5,
                )
        vals_prev = vals

    cohen_s2 = COHEN_RESULTS['Cohen spherical CNN']
    ax.axhline(
        cohen_s2['NR/NR'], color='#888888', linewidth=0.6,
        linestyle='--', zorder=2, alpha=0.6,
        label=f'Cohen S\u00b2CNN NR/NR ({cohen_s2["NR/NR"]:.2f})',
    )
    ax.axhline(
        cohen_s2['NR/R'], color='#888888', linewidth=0.6,
        linestyle=':', zorder=2, alpha=0.6,
        label=f'Cohen S\u00b2CNN NR/R ({cohen_s2["NR/R"]:.2f})',
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_LABELS.get(m, m) for m in models],
        rotation=20, ha='right',
    )
    ax.set_ylabel('Test accuracy')
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0%}'))
    _clean_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        frameon=False, ncol=3, handlelength=1.5,
        fontsize=6, loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        columnspacing=1.2,
    )


def _plot_rotation_panel(
    ax: plt.Axes,
    grouped: dict[tuple[str, str], list[dict]],
    models: list[str],
):
    """Panel (b): Strip plot of accuracy under random SO(3) rotations (NR-trained)."""
    rng = np.random.default_rng(42)

    for i, m in enumerate(models):
        nr_runs = grouped.get((m, 'NR'), [])
        all_rot_accs: list[float] = []
        for run in nr_runs:
            rob = run.get('rotation_robustness', {})
            for k, v in rob.items():
                if k.startswith('rot_'):
                    all_rot_accs.append(v)

        if not all_rot_accs:
            continue

        pts = np.array(all_rot_accs)
        jitter = rng.uniform(-0.15, 0.15, size=len(pts))
        color = COLORS[m]

        ax.scatter(
            np.full_like(pts, i) + jitter, pts,
            s=12, alpha=0.55, color=color, edgecolors='white',
            linewidths=0.3, zorder=3,
        )

        mean_val = np.mean(pts)
        ax.hlines(
            mean_val, i - 0.25, i + 0.25,
            colors=color, linewidths=1.2, zorder=4,
        )

        y_offset = 0.03 if m != 'standard' else 0.04
        ax.text(
            i + 0.30, mean_val + y_offset,
            f'{mean_val:.1%}',
            fontsize=6.5, color=color, va='bottom', fontweight='medium',
        )

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(
        [MODEL_LABELS.get(m, m) for m in models],
        rotation=20, ha='right',
    )
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylabel('Accuracy under random SO(3)')
    ax.set_ylim(-0.02, 1.08)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0%}'))
    _clean_axes(ax)

    ax.axhline(0.10, color='#aaaaaa', linewidth=0.5, linestyle=':', zorder=1)
    ax.text(
        len(models) - 0.5, 0.115, 'chance (10%)',
        fontsize=6, color='#aaaaaa', ha='right',
    )


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Spherical MNIST results',
    )
    parser.add_argument('--results_dir', type=str, default='./smnist_results')
    parser.add_argument('--matched_dir', type=str, default='./smnist_results_matched')
    parser.add_argument(
        '--output', type=str, default='./smnist_analysis.png',
    )
    args = parser.parse_args()

    grouped = load_results(args.results_dir, args.matched_dir)

    if not grouped:
        print('No results found. Run the sweep first.')
        return

    print_cohen_table(grouped)
    print_latex_table(grouped)
    plot_combined_figure(grouped, args.output)


if __name__ == '__main__':
    main()
