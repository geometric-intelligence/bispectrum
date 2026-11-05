#!/usr/bin/env python3
"""Analyze OrganMNIST3D experiment results.

Loads results JSONs from organ3d_results/, prints comparison tables, and
generates a bar chart comparing original vs rotated accuracy.

Usage:
    python analyze_results.py [--results_dir ./organ3d_results]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import brand  # noqa: E402

MODEL_ORDER = ['standard', 'norm_pool', 'max_pool', 'bispectrum']
MODEL_LABELS = {
    'standard': 'Aug. 3D CNN',
    'max_pool': 'O-Equiv + Max Pool',
    'norm_pool': 'O-Equiv + Norm Pool',
    'bispectrum': 'O-Equiv + Bispectrum',
}

DEFAULT_TRAIN_MODE = 'C'
ORGAN_FULL_TRAIN = 971


def _canonical_train_mode(mode: str) -> str:
    if mode in {'C', 'NR'}:
        return 'C'
    if mode == 'R':
        return 'R'
    return DEFAULT_TRAIN_MODE


def _result_train_mode(record: dict) -> str:
    return _canonical_train_mode(record.get('train_mode', DEFAULT_TRAIN_MODE))


def _expected_train_mode(model: str) -> str:
    """Protocol mode per model: Aug. CNN baseline is R-trained, invariants C."""
    return 'R' if model == 'standard' else 'C'


def _test_c(record: dict) -> dict:
    return record.get('test_c') or record.get('test') or {}


def _test_r(record: dict) -> dict:
    return record.get('test_r') or {}


PUBLISHED_BASELINES = [
    {
        'method': 'ResNet-18 + 3D',
        'venue': 'Yang et al. 2023',
        'params': '33M',
        'acc': 0.907,
        'auc': 0.996,
    },
    {
        'method': 'ResNet-18 + ACS',
        'venue': 'Yang et al. 2023',
        'params': '11M',
        'acc': 0.900,
        'auc': 0.994,
    },
    {
        'method': 'ILPOResNet-50',
        'venue': 'Zhemchuzhnikov 2024',
        'params': '38K',
        'acc': 0.879,
        'auc': 0.992,
    },
    {
        'method': 'EquiLoPO (local train.)',
        'venue': 'ICLR 2025',
        'params': '418K',
        'acc': 0.866,
        'auc': 0.991,
    },
    {
        'method': 'SE3MovFrNet',
        'venue': 'Sangalli et al. 2023',
        'params': '—',
        'acc': 0.745,
        'auc': None,
    },
    {
        'method': 'Regular SE(3) conv',
        'venue': 'Kuipers & Bekkers 2023',
        'params': '172K',
        'acc': 0.698,
        'auc': None,
    },
]


def load_results(results_dir: str) -> dict[str, list[dict]]:
    """Load all results.json files, grouped by model name."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f'Results directory not found: {results_dir}')
        return grouped

    for p in sorted(results_path.glob('*/results.json')):
        with open(p) as f:
            r = json.load(f)
        grouped[r['model']].append(r)

    return grouped


def _is_full_train(record: dict) -> bool:
    """A run trained on the entire dataset has no positive train_size."""
    raw = record.get('train_size')
    if raw is None:
        return True
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return True
    return n <= 0


def _train_size_value(record: dict) -> int | None:
    raw = record.get('train_size')
    if raw is None:
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _data_sizes(grouped: dict[str, list[dict]]) -> list[int]:
    """Subset sizes present among the [4, 8] runs, then the full training set."""
    sizes = {
        n
        for runs in grouped.values()
        for r in runs
        if r.get('channels', [4, 8]) == [4, 8] and (n := _train_size_value(r)) is not None
    }
    return [*sorted(sizes), ORGAN_FULL_TRAIN]


def _baseline_runs(grouped: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Filter to channels=[4,8], full-set runs, each model in its protocol train mode."""
    filtered: dict[str, list[dict]] = {}
    for model, runs in grouped.items():
        f = [
            r
            for r in runs
            if r.get('channels', [4, 8]) == [4, 8]
            and _is_full_train(r)
            and _result_train_mode(r) == _expected_train_mode(model)
        ]
        if f:
            filtered[model] = f
    return filtered


def print_our_results(grouped: dict[str, list[dict]]):
    """Print table of our experimental results (baseline channels, protocol train modes)."""
    baseline = _baseline_runs(grouped)
    print('\n' + '=' * 90)
    print('OUR RESULTS (controlled ablation — same backbone, different pooling; Aug. CNN R-trained)')
    print('=' * 90)
    header = f'{"Model":<25} {"Params":>8} {"Test ACC":>12} {"Test AUC":>12} {"Rot ACC":>12} {"Rot \u03c3_ACC":>10}'
    print(header)
    print('-' * 90)

    for model_name in MODEL_ORDER:
        runs = baseline.get(model_name, [])
        if not runs:
            continue

        accs = [_test_c(r).get('accuracy', 0.0) for r in runs]
        aucs = [_test_c(r).get('auc', 0.0) for r in runs]
        n_params = runs[0]['n_params']

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        rot_accs = [
            r['rotation_robustness']['mean_accuracy']
            for r in runs
            if r.get('rotation_robustness', {}).get('mean_accuracy', 0) > 0
        ]
        rot_stds = [
            r['rotation_robustness']['std_accuracy']
            for r in runs
            if r.get('rotation_robustness', {}).get('mean_accuracy', 0) > 0
        ]

        label = MODEL_LABELS.get(model_name, model_name)
        rot_str = f'{np.mean(rot_accs):.4f}' if rot_accs else '\u2014'
        rot_std_str = f'{np.mean(rot_stds):.4f}' if rot_stds else '\u2014'

        print(
            f'{label:<25} {n_params:>8,} '
            f'{mean_acc:.4f}\u00b1{std_acc:.4f} '
            f'{mean_auc:.4f}\u00b1{std_auc:.4f} '
            f'{rot_str:>12} {rot_std_str:>10}'
        )


def print_published_baselines():
    """Print table of published baseline results for context."""
    print('\n' + '=' * 80)
    print('PUBLISHED BASELINES (from respective papers, same MedMNIST3D protocol)')
    print('=' * 80)
    print(f'{"Method":<30} {"Venue":<25} {"Params":>8} {"ACC":>8} {"AUC":>8}')
    print('-' * 80)
    for b in PUBLISHED_BASELINES:
        auc_str = f'{b["auc"]:.3f}' if b['auc'] is not None else '—'
        print(f'{b["method"]:<30} {b["venue"]:<25} {b["params"]:>8} {b["acc"]:.3f}    {auc_str:>8}')


def plot_rotation_comparison(grouped: dict[str, list[dict]], output_path: str):
    """Generate bar chart: accuracy on canonical vs octahedrally rotated test."""
    grouped = _baseline_runs(grouped)
    models_with_data = [m for m in MODEL_ORDER if m in grouped]
    if not models_with_data:
        print('No data to plot.')
        return

    brand.use_style()
    plt.rcParams.update(
        {
            'font.size': 11,
            'axes.labelsize': 13,
            'axes.titlesize': 14,
            'xtick.labelsize': 11,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
        }
    )

    PLOT_LABELS = {
        'standard': 'Aug. 3D CNN',
        'max_pool': 'Max Pool',
        'norm_pool': 'Norm Pool',
        'bispectrum': 'Bispectrum',
    }
    labels = [PLOT_LABELS.get(m, m) for m in models_with_data]

    orig_accs = []
    orig_stds = []
    rot_accs = []
    rot_stds = []

    for m in models_with_data:
        runs = grouped[m]
        accs = [_test_c(r).get('accuracy', 0.0) for r in runs]
        orig_accs.append(np.mean(accs))
        orig_stds.append(np.std(accs))
        rot_vals = [
            r['rotation_robustness']['mean_accuracy']
            for r in runs
            if r.get('rotation_robustness', {}).get('mean_accuracy', 0) > 0
        ]
        rot_accs.append(np.mean(rot_vals) if rot_vals else 0)
        rot_stds.append(np.std(rot_vals) if rot_vals else 0)

    orig_accs = np.array(orig_accs)
    orig_stds = np.array(orig_stds)
    rot_accs = np.array(rot_accs)
    rot_stds = np.array(rot_stds)

    x = np.arange(len(labels))
    width = 0.32

    c_orig = brand.MUTED
    c_rot = brand.NEUTRAL_LIGHT

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    bars1 = ax.bar(
        x - width / 2,
        orig_accs,
        width,
        yerr=orig_stds,
        capsize=3,
        color=c_orig,
        edgecolor=brand.surface_color(),
        linewidth=0.6,
        error_kw={'linewidth': 0.9, 'capthick': 0.9},
        label='Original test',
        zorder=3,
    )
    rot_display = np.where(rot_accs > 0, rot_accs, 0)
    rot_err = np.where(rot_accs > 0, rot_stds, 0)
    bars2 = ax.bar(
        x + width / 2,
        rot_display,
        width,
        yerr=rot_err,
        capsize=3,
        color=c_rot,
        edgecolor=brand.surface_color(),
        linewidth=0.6,
        error_kw={'linewidth': 0.9, 'capthick': 0.9},
        label='Rotated test (24 oct. rotations)',
        zorder=3,
    )

    def annotate_bar(bar, val: float, std: float):
        h = bar.get_height()
        offset = std + 0.015 if std > 0 else 0.015
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            f'{val:.1%}',
            ha='center',
            va='bottom',
            fontsize=6.5,
            fontfamily=brand.MONO,
        )

    for bar, v, s in zip(bars1, orig_accs, orig_stds, strict=False):
        annotate_bar(bar, v, s)
    for bar, v, s in zip(bars2, rot_display, rot_err, strict=False):
        if v > 0:
            annotate_bar(bar, v, s)

    ax.set_ylabel('Test accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, len(labels) - 0.5)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))

    brand.style_axes(ax, grid='y')
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.tick_params(width=0.6, length=3)

    ax.legend(
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        loc='upper left',
    )

    fig.tight_layout()
    brand.finish(fig)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.replace('.png', '.pdf'))
    print(f'\nPlot saved to {output_path} (+ PDF)')
    plt.close(fig)


def plot_data_efficiency(grouped: dict[str, list[dict]], output_path: str):
    """Plot data efficiency curves: test accuracy vs train fraction for each model."""
    brand.use_style()
    plt.rcParams.update(
        {
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 9,
            'figure.dpi': 300,
        }
    )

    organ_full = ORGAN_FULL_TRAIN
    sizes = _data_sizes(grouped)
    models_to_plot = ['standard', 'norm_pool', 'max_pool', 'bispectrum']
    colors = {model: brand.METHOD_COLORS[model] for model in models_to_plot}
    markers = {'standard': 's', 'norm_pool': 'v', 'max_pool': 'D', 'bispectrum': 'o'}
    labels = {
        'standard': 'Aug. 3D CNN',
        'norm_pool': 'Norm Pool',
        'max_pool': 'Max Pool',
        'bispectrum': 'Bispectrum',
    }

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for model in models_to_plot:
        means = []
        stds = []
        for size in sizes:
            target = None if size >= organ_full else size
            runs = [
                r
                for r in grouped.get(model, [])
                if _train_size_value(r) == target
                and r.get('channels', [4, 8]) == [4, 8]
                and _result_train_mode(r) == _expected_train_mode(model)
            ]
            if runs:
                vals = [_test_c(r).get('accuracy', 0.0) for r in runs]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(np.nan)
                stds.append(0)

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(
            sizes,
            means,
            color=colors[model],
            marker=markers[model],
            markersize=7,
            linewidth=2,
            label=labels[model],
            zorder=3,
        )
        ax.fill_between(
            sizes,
            means - stds,
            means + stds,
            color=colors[model],
            alpha=0.15,
            zorder=2,
        )

    ax.set_xlabel('Training examples (N)')
    ax.set_ylabel('Test accuracy')
    ax.set_xscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_ylim(0, 0.9)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))

    brand.style_axes(ax, grid='y')
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.tick_params(width=0.6, length=3)

    ax.legend(
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        loc='upper left',
    )

    fig.tight_layout()
    brand.finish(fig)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.replace('.png', '.pdf'))
    print(f'\nData efficiency plot saved to {output_path} (+ PDF)')
    plt.close(fig)


def print_tier1_summary(grouped: dict[str, list[dict]]):
    """Print summary tables for tier 1 experiments."""
    organ_full = ORGAN_FULL_TRAIN
    sizes = _data_sizes(grouped)
    models = MODEL_ORDER

    print('\n' + '=' * 90)
    print('DATA EFFICIENCY (channels [4,8]; Aug. CNN R-trained)')
    print('=' * 90)
    print(f'{"Model":<15} {"N":>6} {"Params":>10} {"Test ACC":>18} {"Test AUC":>18} {"Rot ACC":>10}')
    print('-' * 85)

    for model in models:
        for size in sizes:
            target = None if size >= organ_full else size
            runs = [
                r
                for r in grouped.get(model, [])
                if _train_size_value(r) == target
                and r.get('channels', [4, 8]) == [4, 8]
                and _result_train_mode(r) == _expected_train_mode(model)
            ]
            if runs:
                accs = [_test_c(r).get('accuracy', 0.0) for r in runs]
                aucs = [_test_c(r).get('auc', 0.0) for r in runs]
                rots = [r.get('rotation_robustness', {}).get('mean_accuracy', 0) for r in runs]
                n_seeds = len(runs)
                print(
                    f'{model:<15} {size:>6d} {runs[0]["n_params"]:>10,} '
                    f'{np.mean(accs):>7.4f}±{np.std(accs):.4f} ({n_seeds}s) '
                    f'{np.mean(aucs):>7.4f}±{np.std(aucs):.4f} '
                    f'{np.mean(rots):>10.4f}'
                )

    print('\n' + '=' * 90)
    print('WIDER CHANNELS (full training set)')
    print('=' * 90)
    print(f'{"Model":<15} {"Channels":>10} {"Params":>10} {"Test ACC":>18} {"Test AUC":>18} {"Rot ACC":>10}')
    print('-' * 85)

    for ch in [[4, 8], [8, 16], [16, 32]]:
        for model in MODEL_ORDER:
            runs = [
                r
                for r in grouped.get(model, [])
                if r.get('channels', [4, 8]) == ch
                and _is_full_train(r)
                and _result_train_mode(r) == _expected_train_mode(model)
            ]
            if runs:
                accs = [_test_c(r).get('accuracy', 0.0) for r in runs]
                aucs = [_test_c(r).get('auc', 0.0) for r in runs]
                rots = [r.get('rotation_robustness', {}).get('mean_accuracy', 0) for r in runs]
                n_seeds = len(runs)
                ch_str = '\u2192'.join(map(str, ch))
                print(
                    f'{model:<15} {ch_str:>10} {runs[0]["n_params"]:>10,} '
                    f'{np.mean(accs):>7.4f}\u00b1{np.std(accs):.4f} ({n_seeds}s) '
                    f'{np.mean(aucs):>7.4f}\u00b1{np.std(aucs):.4f} '
                    f'{np.mean(rots):>10.4f}'
                )


def main():
    parser = argparse.ArgumentParser(
        description='Analyze OrganMNIST3D experiment results',
    )
    parser.add_argument('--results_dir', type=str, default='./organ3d_results')
    parser.add_argument('--output_plot', type=str, default='./organ3d_analysis.png')
    args = parser.parse_args()

    grouped = load_results(args.results_dir)

    if grouped:
        print_our_results(grouped)
    else:
        print('No experimental results found. Run the sweep first.')

    print_published_baselines()

    if grouped:
        plot_rotation_comparison(grouped, args.output_plot)
        plot_data_efficiency(grouped, './data_efficiency.png')
        print_tier1_summary(grouped)


if __name__ == '__main__':
    main()
