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
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = ['standard', 'norm_pool', 'max_pool', 'bispectrum']
MODEL_LABELS = {
    'standard': 'Standard 3D CNN',
    'max_pool': 'O-Equiv + Max Pool',
    'norm_pool': 'O-Equiv + Norm Pool',
    'bispectrum': 'O-Equiv + Bispectrum',
}

PUBLISHED_BASELINES = [
    {'method': 'ResNet-18 + 3D', 'venue': 'Yang et al. 2023', 'params': '33M', 'acc': 0.907, 'auc': 0.996},
    {'method': 'ResNet-18 + ACS', 'venue': 'Yang et al. 2023', 'params': '11M', 'acc': 0.900, 'auc': 0.994},
    {'method': 'ILPOResNet-50', 'venue': 'Zhemchuzhnikov 2024', 'params': '38K', 'acc': 0.879, 'auc': 0.992},
    {'method': 'EquiLoPO (local train.)', 'venue': 'ICLR 2025', 'params': '418K', 'acc': 0.866, 'auc': 0.991},
    {'method': 'SE3MovFrNet', 'venue': 'Sangalli et al. 2023', 'params': '—', 'acc': 0.745, 'auc': None},
    {'method': 'Regular SE(3) conv', 'venue': 'Kuipers & Bekkers 2023', 'params': '172K', 'acc': 0.698, 'auc': None},
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


def _baseline_runs(grouped: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Filter to channels=[4,8] and train_fraction=1.0 runs only."""
    filtered: dict[str, list[dict]] = {}
    for model, runs in grouped.items():
        f = [r for r in runs
             if r.get('channels', [4, 8]) == [4, 8]
             and abs(r.get('train_fraction', 1.0) - 1.0) < 0.01]
        if f:
            filtered[model] = f
    return filtered


def print_our_results(grouped: dict[str, list[dict]]):
    """Print table of our experimental results (baseline channels only)."""
    baseline = _baseline_runs(grouped)
    print('\n' + '=' * 90)
    print('OUR RESULTS (controlled ablation — same backbone, different pooling)')
    print('=' * 90)
    header = (
        f'{"Model":<25} {"Params":>8} {"Test ACC":>12} {"Test AUC":>12} '
        f'{"Rot ACC":>12} {"Rot \u03c3_ACC":>10}'
    )
    print(header)
    print('-' * 90)

    for model_name in MODEL_ORDER:
        runs = baseline.get(model_name, [])
        if not runs:
            continue

        accs = [r['test']['accuracy'] for r in runs]
        aucs = [r['test']['auc'] for r in runs]
        n_params = runs[0]['n_params']

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        rot_accs = [r['rotation_robustness']['mean_accuracy'] for r in runs
                    if r['rotation_robustness'].get('mean_accuracy', 0) > 0]
        rot_stds = [r['rotation_robustness']['std_accuracy'] for r in runs
                    if r['rotation_robustness'].get('mean_accuracy', 0) > 0]

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
        print(
            f'{b["method"]:<30} {b["venue"]:<25} {b["params"]:>8} '
            f'{b["acc"]:.3f}    {auc_str:>8}'
        )


def plot_rotation_comparison(grouped: dict[str, list[dict]], output_path: str):
    """Generate bar chart: original vs rotated accuracy per model (baseline channels only)."""
    grouped = _baseline_runs(grouped)
    models_with_data = [m for m in MODEL_ORDER if m in grouped]
    if not models_with_data:
        print('No data to plot.')
        return

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

    PLOT_LABELS = {
        'standard': 'Standard',
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
        accs = [r['test']['accuracy'] for r in runs]
        orig_accs.append(np.mean(accs))
        orig_stds.append(np.std(accs))
        rot_vals = [r['rotation_robustness']['mean_accuracy'] for r in runs
                    if r['rotation_robustness'].get('mean_accuracy', 0) > 0]
        rot_accs.append(np.mean(rot_vals) if rot_vals else 0)
        rot_stds.append(np.std(rot_vals) if rot_vals else 0)

    orig_accs = np.array(orig_accs)
    orig_stds = np.array(orig_stds)
    rot_accs = np.array(rot_accs)
    rot_stds = np.array(rot_stds)

    x = np.arange(len(labels))
    width = 0.32

    c_orig = '#2D6A9F'
    c_rot = '#D4722A'

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    bars1 = ax.bar(
        x - width / 2, orig_accs, width, yerr=orig_stds,
        capsize=3, color=c_orig, edgecolor='white', linewidth=0.6,
        error_kw={'linewidth': 0.9, 'capthick': 0.9},
        label='Original test', zorder=3,
    )
    rot_display = np.where(rot_accs > 0, rot_accs, 0)
    rot_err = np.where(rot_accs > 0, rot_stds, 0)
    bars2 = ax.bar(
        x + width / 2, rot_display, width, yerr=rot_err,
        capsize=3, color=c_rot, edgecolor='white', linewidth=0.6,
        error_kw={'linewidth': 0.9, 'capthick': 0.9},
        label='Rotated test (24 oct. rotations)', zorder=3,
    )

    def annotate_bar(bar, val: float, std: float):
        h = bar.get_height()
        offset = std + 0.015 if std > 0 else 0.015
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + offset,
            f'{val:.1%}',
            ha='center', va='bottom', fontsize=8, fontweight='medium',
        )

    for bar, v, s in zip(bars1, orig_accs, orig_stds):
        annotate_bar(bar, v, s)
    for bar, v, s in zip(bars2, rot_display, rot_err):
        if v > 0:
            annotate_bar(bar, v, s)

    ax.set_ylabel('Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, len(labels) - 0.5)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.tick_params(width=0.6, length=3)

    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        frameon=True, fancybox=False, edgecolor='#cccccc',
        framealpha=0.95, loc='upper left',
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.replace('.png', '.pdf'))
    print(f'\nPlot saved to {output_path} (+ PDF)')
    plt.close(fig)


def plot_data_efficiency(grouped: dict[str, list[dict]], output_path: str):
    """Plot data efficiency curves: test accuracy vs train fraction for each model."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 300,
    })

    fracs = [0.05, 0.10, 0.25, 0.50, 1.0]
    models_to_plot = ['standard', 'max_pool', 'bispectrum']
    colors = {'standard': '#888888', 'max_pool': '#2D6A9F', 'bispectrum': '#C44E52'}
    markers = {'standard': 's', 'max_pool': 'D', 'bispectrum': 'o'}
    labels = {'standard': 'Standard CNN', 'max_pool': 'Max Pool', 'bispectrum': 'Bispectrum'}

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for model in models_to_plot:
        means = []
        stds = []
        for frac in fracs:
            runs = [r for r in grouped.get(model, [])
                    if abs(r.get('train_fraction', 1.0) - frac) < 0.01
                    and r.get('channels', [4, 8]) == [4, 8]]
            if runs:
                vals = [r['test']['accuracy'] for r in runs]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(np.nan)
                stds.append(0)

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(
            fracs, means,
            color=colors[model], marker=markers[model],
            markersize=7, linewidth=2, label=labels[model],
            zorder=3,
        )
        ax.fill_between(
            fracs, means - stds, means + stds,
            color=colors[model], alpha=0.15, zorder=2,
        )

    ax.set_xlabel('Training data fraction')
    ax.set_ylabel('Test accuracy')
    ax.set_xscale('log')
    ax.set_xticks(fracs)
    ax.set_xticklabels(['5%', '10%', '25%', '50%', '100%'])
    ax.set_ylim(0, 0.9)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.tick_params(width=0.6, length=3)

    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        frameon=True, fancybox=False, edgecolor='#cccccc',
        framealpha=0.95, loc='upper left',
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.replace('.png', '.pdf'))
    print(f'\nData efficiency plot saved to {output_path} (+ PDF)')
    plt.close(fig)


def print_tier1_summary(grouped: dict[str, list[dict]]):
    """Print summary tables for tier 1 experiments."""
    fracs = [0.05, 0.10, 0.25, 0.50, 1.0]
    models = ['standard', 'max_pool', 'bispectrum']

    print('\n' + '=' * 90)
    print('DATA EFFICIENCY (channels [4,8], 3 seeds)')
    print('=' * 90)
    print(f'{"Model":<15} {"Frac":>6} {"Params":>10} {"Test ACC":>18} {"Test AUC":>18} {"Rot ACC":>10}')
    print('-' * 85)

    for model in models:
        for frac in fracs:
            runs = [r for r in grouped.get(model, [])
                    if abs(r.get('train_fraction', 1.0) - frac) < 0.01
                    and r.get('channels', [4, 8]) == [4, 8]]
            if runs:
                accs = [r['test']['accuracy'] for r in runs]
                aucs = [r['test']['auc'] for r in runs]
                rots = [r['rotation_robustness'].get('mean_accuracy', 0) for r in runs]
                n_seeds = len(runs)
                print(
                    f'{model:<15} {frac:>6.0%} {runs[0]["n_params"]:>10,} '
                    f'{np.mean(accs):>7.4f}±{np.std(accs):.4f} ({n_seeds}s) '
                    f'{np.mean(aucs):>7.4f}±{np.std(aucs):.4f} '
                    f'{np.mean(rots):>10.4f}'
                )

    print('\n' + '=' * 90)
    print('WIDER CHANNELS (frac=1.0)')
    print('=' * 90)
    print(f'{"Model":<15} {"Channels":>10} {"Params":>10} {"Test ACC":>18} {"Test AUC":>18} {"Rot ACC":>10}')
    print('-' * 85)

    for ch in [[4, 8], [8, 16], [16, 32]]:
        for model in ['max_pool', 'bispectrum']:
            runs = [r for r in grouped.get(model, [])
                    if r.get('channels', [4, 8]) == ch
                    and abs(r.get('train_fraction', 1.0) - 1.0) < 0.01]
            if runs:
                accs = [r['test']['accuracy'] for r in runs]
                aucs = [r['test']['auc'] for r in runs]
                rots = [r['rotation_robustness'].get('mean_accuracy', 0) for r in runs]
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
