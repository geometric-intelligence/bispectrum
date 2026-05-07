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
    'power_spectrum': 'PowSpec',
    'power_spectrum_matched': 'PowSpec (matched)',
    'bispectrum': 'Bispectrum',
}

MODEL_LABELS_FULL = {
    'standard': 'Standard CNN',
    'power_spectrum': 'Power Spectrum',
    'power_spectrum_matched': 'Power Spectrum (matched)',
    'bispectrum': 'Bispectrum (ours)',
}

COHEN_RESULTS = {
    'Cohen planar CNN (NR/X)': {'C/C': 0.98, 'R/R': 0.23, 'C/R': 0.11},
    'Cohen spherical CNN (NR/X)': {'C/C': 0.96, 'R/R': 0.95, 'C/R': 0.94},
}


def _canonical_train_mode(mode: str) -> str:
    """Map legacy `NR` to the canonical `C` mode."""
    return 'C' if mode in {'C', 'NR'} else 'R'


def _params_label(n_params: int) -> str:
    """Format a parameter count as ``11K`` / ``1.6M`` for table labels."""
    if n_params >= 1_000_000:
        return f'{n_params / 1_000_000:.1f}M'
    if n_params >= 1000:
        return f'{n_params / 1000:.0f}K'
    return f'{n_params}'


def _full_label(model: str, n_params: int | None) -> str:
    base = MODEL_LABELS_FULL.get(model, model)
    if n_params is None:
        return base
    return f'{base} ({_params_label(n_params)})'


def _short_label(model: str, n_params: int | None) -> str:
    base = MODEL_LABELS.get(model, model)
    if n_params is None:
        return base
    return f'{base} {_params_label(n_params)}'


def _test_c(run: dict) -> dict:
    return run.get('test_c') or run.get('test_nr') or run.get('test', {})


def _test_r(run: dict) -> dict:
    return run.get('test_r') or {}


def _train_size_value(record: dict) -> int | None:
    """Resolved train_size, or None for full-set runs."""
    raw = record.get('train_size')
    if raw is None:
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _is_full_train(record: dict) -> bool:
    return _train_size_value(record) is None

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
    'C/C': '#2D6A9F',
    'R/R': '#4CAF50',
    'C/R': '#D4722A',
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
        if not _is_full_train(r):
            continue
        mode = _canonical_train_mode(r['train_mode'])
        grouped[(r['model'], mode)].append(r)

    if matched_dir:
        matched_path = Path(matched_dir)
        if matched_path.exists():
            for p in sorted(matched_path.glob('*/results.json')):
                with open(p) as f:
                    r = json.load(f)
                mode = _canonical_train_mode(r['train_mode'])
                grouped[('power_spectrum_matched', mode)].append(r)

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
    """Print the C/C, R/R, C/R accuracy table to console."""
    print('\n' + '=' * 100)
    print('ACCURACY TABLE (Cohen et al. 2018 protocol, canonical/random)')
    print('Train/Test: X/Y = trained on X, evaluated on Y. Cohen reference uses NR for canonical.')
    print('=' * 100)
    hdr = f'{"Method":<32} {"Params":>10}  {"C/C":>12}  {"R/R":>12}  {"C/R":>12}  {"Rot \u03c3":>10}'
    print(hdr)
    print('-' * 100)

    for method, accs in COHEN_RESULTS.items():
        print(
            f'{method:<32} {"\u2014":>10}  '
            f'{accs["C/C"]:>12.3f}  {accs["R/R"]:>12.3f}  {accs["C/R"]:>12.3f}  '
            f'{"\u2014":>10}'
        )

    print('-' * 100)

    for model in MODEL_ORDER:
        c_runs = grouped.get((model, 'C'), [])
        r_runs = grouped.get((model, 'R'), [])
        if not c_runs and not r_runs:
            continue

        n_params = (c_runs or r_runs)[0]['n_params']
        c_c = [_test_c(r)['accuracy'] for r in c_runs]
        c_r = [_test_r(r)['accuracy'] for r in c_runs if _test_r(r)]
        r_r = [_test_r(r)['accuracy'] for r in r_runs if _test_r(r)]

        rot_stds = [
            r['rotation_robustness']['std_accuracy']
            for r in c_runs
            if r.get('rotation_robustness', {}).get('std_accuracy') is not None
        ]
        rot_str = f'{np.mean(rot_stds):.5f}' if rot_stds else '\u2014'

        label = _full_label(model, n_params)
        print(
            f'{label:<32} {n_params:>10,}  '
            f'{_fmt(c_c):>12}  {_fmt(r_r):>12}  {_fmt(c_r):>12}  '
            f'{rot_str:>10}'
        )


def print_latex_table(grouped: dict[tuple[str, str], list[dict]]):
    """Print LaTeX-formatted table rows for direct insertion into paper."""
    print('\n% LaTeX table rows for Spherical MNIST results')
    print(r'% Paste into \begin{tabular}{lrcccc}')
    print(r'%   Model & Params & C/C & R/R & C/R & Rot.\ $\sigma$ \\')

    for method, accs in COHEN_RESULTS.items():
        print(
            f'    {method} & --- '
            f'& ${accs["C/C"]:.2f}$ & ${accs["R/R"]:.2f}$ '
            f'& ${accs["C/R"]:.2f}$ & --- \\\\'
        )
    print(r'    \midrule')

    for model in MODEL_ORDER:
        c_runs = grouped.get((model, 'C'), [])
        r_runs = grouped.get((model, 'R'), [])
        if not c_runs and not r_runs:
            continue

        n_params = (c_runs or r_runs)[0]['n_params']
        c_c = [_test_c(r)['accuracy'] for r in c_runs]
        c_r = [_test_r(r)['accuracy'] for r in c_runs if _test_r(r)]
        r_r = [_test_r(r)['accuracy'] for r in r_runs if _test_r(r)]

        rot_stds = [
            r['rotation_robustness']['std_accuracy']
            for r in c_runs
            if r.get('rotation_robustness', {}).get('std_accuracy') is not None
        ]

        is_bisp = model == 'bispectrum'
        label = MODEL_LABELS_FULL.get(model, model)
        if is_bisp:
            label = f'\\textbf{{{label}}}'

        params_str = _params_label(n_params)
        rot_val = f'{np.mean(rot_stds):.4f}' if rot_stds else '---'

        print(
            f'    {label} & {params_str} '
            f'& {_latex_fmt(c_c, bold=is_bisp)} '
            f'& {_latex_fmt(r_r, bold=is_bisp)} '
            f'& {_latex_fmt(c_r, bold=is_bisp)} '
            f'& ${rot_val}$ \\\\'
        )


SMNIST_FULL_TRAIN = 60_000


def load_data_efficiency_results(
    results_dir: str,
) -> dict[tuple[str, str, int], list[dict]]:
    """Load all results, grouped by (model, canonical train_mode, train_size).

    ``train_size`` is the absolute number of training examples actually used.
    Full-set runs are bucketed at ``SMNIST_FULL_TRAIN``.
    """
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    results_path = Path(results_dir)
    if not results_path.exists():
        return grouped

    for p in sorted(results_path.glob('*/results.json')):
        with open(p) as f:
            r = json.load(f)
        size = _train_size_value(r)
        size_bucket = SMNIST_FULL_TRAIN if size is None else size
        mode = _canonical_train_mode(r['train_mode'])
        grouped[(r['model'], mode, size_bucket)].append(r)

    return grouped


def plot_data_efficiency(
    grouped: dict[tuple[str, str, int], list[dict]],
    output_path: str,
):
    """Two-panel figure: C/C and C/R accuracy vs absolute training size."""
    plt.rcParams.update(NEURIPS_RCPARAMS)

    sizes_to_plot = [100, 500, 2500, 12500, SMNIST_FULL_TRAIN]
    models_to_plot = ['standard', 'power_spectrum', 'bispectrum']
    colors = {
        'standard': '#7f7f7f',
        'power_spectrum': '#1f77b4',
        'bispectrum': '#d62728',
    }
    markers = {'standard': 's', 'power_spectrum': 'D', 'bispectrum': 'o'}
    labels = {
        'standard': 'Standard CNN',
        'power_spectrum': 'Power Spectrum',
        'bispectrum': 'Bispectrum',
    }

    panels = [
        ('C', _test_c, '(a) C/C accuracy vs. training examples'),
        ('C', _test_r, '(b) C/R accuracy vs. training examples'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)

    for ax, (train_mode, test_fn, title) in zip(axes, panels):
        for model in models_to_plot:
            means = []
            stds = []
            valid_sizes = []
            for size in sizes_to_plot:
                runs = grouped.get((model, train_mode, size), [])
                if not runs:
                    continue
                vals = [test_fn(r)['accuracy'] for r in runs if test_fn(r)]
                if not vals:
                    continue
                means.append(np.mean(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0.0)
                valid_sizes.append(size)

            if not valid_sizes:
                continue

            means_arr = np.array(means)
            stds_arr = np.array(stds)

            ax.errorbar(
                valid_sizes, means_arr, yerr=stds_arr,
                marker=markers[model], color=colors[model],
                label=labels[model], linewidth=1.2, markersize=5,
                capsize=3, zorder=3,
            )

        ax.set_xscale('log')
        ax.set_xlabel('Training examples (N)')
        ax.set_xticks(sizes_to_plot)
        ax.set_xticklabels([f'{s:,}' for s in sizes_to_plot])
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0%}'))
        _clean_axes(ax)

    axes[0].set_ylabel('Test accuracy')
    axes[0].legend(
        frameon=False, fontsize=7, loc='lower right',
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    pdf_path = output_path.replace('.png', '.pdf')
    fig.savefig(pdf_path)
    print(f'\nData efficiency figure saved to {output_path} and {pdf_path}')
    plt.close(fig)


def print_data_efficiency_table(
    grouped: dict[tuple[str, str, int], list[dict]],
):
    """Print data efficiency summary table."""
    sizes = [100, 500, 2500, 12500, SMNIST_FULL_TRAIN]
    models = ['standard', 'power_spectrum', 'bispectrum']

    print('\n' + '=' * 90)
    print('DATA EFFICIENCY (Spherical MNIST)')
    print('=' * 90)
    print(f'{"Model":<20} {"Mode":<5} {"N":>7} {"Test C ACC":>15} {"Test R ACC":>15} {"Seeds":>6}')
    print('-' * 78)

    for model in models:
        for mode in ['C', 'R']:
            for size in sizes:
                runs = grouped.get((model, mode, size), [])
                if not runs:
                    continue
                c_accs = [_test_c(r)['accuracy'] for r in runs]
                r_accs = [_test_r(r)['accuracy'] for r in runs if _test_r(r)]
                n = len(runs)
                c_str = f'{np.mean(c_accs):.4f}\u00b1{np.std(c_accs):.4f}' if n > 1 else f'{c_accs[0]:.4f}'
                if r_accs:
                    r_str = f'{np.mean(r_accs):.4f}\u00b1{np.std(r_accs):.4f}' if len(r_accs) > 1 else f'{r_accs[0]:.4f}'
                else:
                    r_str = '\u2014'
                print(f'{model:<20} {mode:<5} {size:>7d} {c_str:>15} {r_str:>15} {n:>6}')


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
        if grouped.get((m, 'C')) or grouped.get((m, 'R'))
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
    """Panel (a): Grouped bar chart of C/C, R/R, C/R accuracy."""
    x = np.arange(len(models))
    width = 0.22
    protocols = ['C/C', 'R/R', 'C/R']

    vals_prev: list[float] = []
    for i, proto in enumerate(protocols):
        vals = []
        errs = []
        for m in models:
            if proto == 'C/C':
                runs = grouped.get((m, 'C'), [])
                accs = [_test_c(r)['accuracy'] for r in runs]
            elif proto == 'R/R':
                runs = grouped.get((m, 'R'), [])
                accs = [_test_r(r)['accuracy'] for r in runs if _test_r(r)]
            else:
                runs = grouped.get((m, 'C'), [])
                accs = [_test_r(r)['accuracy'] for r in runs if _test_r(r)]
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

    cohen_s2 = COHEN_RESULTS['Cohen spherical CNN (NR/X)']
    ax.axhline(
        cohen_s2['C/C'], color='#888888', linewidth=0.6,
        linestyle='--', zorder=2, alpha=0.6,
        label=f'Cohen S\u00b2CNN C/C ({cohen_s2["C/C"]:.2f})',
    )
    ax.axhline(
        cohen_s2['C/R'], color='#888888', linewidth=0.6,
        linestyle=':', zorder=2, alpha=0.6,
        label=f'Cohen S\u00b2CNN C/R ({cohen_s2["C/R"]:.2f})',
    )

    def _label_for(m: str) -> str:
        runs = grouped.get((m, 'C')) or grouped.get((m, 'R'))
        n_params = runs[0]['n_params'] if runs else None
        return _short_label(m, n_params)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_label_for(m) for m in models],
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
    """Panel (b): Strip plot of accuracy under random SO(3) rotations (C-trained)."""
    rng = np.random.default_rng(42)

    for i, m in enumerate(models):
        c_runs = grouped.get((m, 'C'), [])
        all_rot_accs: list[float] = []
        for run in c_runs:
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

    def _label_for(m: str) -> str:
        runs = grouped.get((m, 'C')) or grouped.get((m, 'R'))
        n_params = runs[0]['n_params'] if runs else None
        return _short_label(m, n_params)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(
        [_label_for(m) for m in models],
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

    de_grouped = load_data_efficiency_results(args.results_dir)
    if de_grouped:
        print_data_efficiency_table(de_grouped)
        de_output = args.output.replace('smnist_analysis', 'smnist_data_efficiency')
        if de_output == args.output:
            de_output = './smnist_data_efficiency.png'
        plot_data_efficiency(de_grouped, de_output)


if __name__ == '__main__':
    main()
