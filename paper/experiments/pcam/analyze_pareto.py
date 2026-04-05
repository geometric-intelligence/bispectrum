#!/usr/bin/env python3
"""Analyze Pareto sweep results and produce a multi-panel figure."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PARETO_DIR = Path(__file__).parent / 'pcam_results_pareto'
ONEPCT_DIR = Path(__file__).parent / 'pcam_results_1pct'
DATA_PARETO_DIR = Path(__file__).parent / 'pcam_results_data_pareto'

MODEL_STYLE = {
    'standard':    {'color': '#888888', 'marker': 's', 'label': 'Standard (augmented)'},
    'norm':        {'color': '#2196F3', 'marker': 'o', 'label': 'NormReLU'},
    'gate':        {'color': '#FF9800', 'marker': '^', 'label': 'Gated'},
    'fourier_elu': {'color': '#9C27B0', 'marker': 'D', 'label': 'Fourier-ELU'},
    'bispectrum':  {'color': '#E53935', 'marker': '*', 'label': 'Bispectrum'},
}
MODEL_ORDER = ['standard', 'norm', 'gate', 'fourier_elu', 'bispectrum']


def load_results(directory: Path) -> list[dict]:
    results = []
    for d in sorted(os.listdir(directory)):
        rpath = directory / d / 'results.json'
        if rpath.is_file():
            with open(rpath) as f:
                results.append(json.load(f))
    return results


def aggregate_1pct(results: list[dict]) -> dict[str, dict]:
    """Group 1% results by model, compute mean ± std."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[r['model']].append(r)
    agg = {}
    for model, runs in grouped.items():
        aucs = [r['test']['auc'] for r in runs]
        rot_stds = [r['rotation_robustness']['std_auc'] for r in runs]
        agg[model] = {
            'n_params': runs[0]['n_params'],
            'mean_auc': np.mean(aucs),
            'std_auc': np.std(aucs, ddof=1),
            'mean_rot_std': np.mean(rot_stds),
        }
    return agg


def load_data_pareto() -> dict[str, dict[float, float]]:
    """Load data-efficiency results: model -> {fraction: auc}."""
    result: dict[str, dict[float, float]] = {}
    if not DATA_PARETO_DIR.is_dir():
        return result
    for frac_dir in sorted(DATA_PARETO_DIR.iterdir()):
        if not frac_dir.is_dir() or not frac_dir.name.startswith('frac_'):
            continue
        frac = float(frac_dir.name.replace('frac_', ''))
        for run_dir in sorted(frac_dir.iterdir()):
            rpath = run_dir / 'results.json'
            if not rpath.is_file():
                continue
            with open(rpath) as f:
                r = json.load(f)
            model = r['model']
            result.setdefault(model, {})[frac] = r['test']['auc']
    return result


def _style_ax(ax: plt.Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.tick_params(width=0.6, length=3)
    ax.set_axisbelow(True)


def main() -> None:
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

    pareto = load_results(PARETO_DIR)
    data_eff = load_data_pareto()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3.8))

    for model in MODEL_ORDER:
        s = MODEL_STYLE[model]
        pts = sorted(
            [(r['n_params'], r['test']['auc']) for r in pareto if r['model'] == model],
            key=lambda x: x[0],
        )
        if not pts:
            continue
        params, aucs = zip(*pts)
        ax1.plot(params, aucs, marker=s['marker'], color=s['color'],
                 label=s['label'], linewidth=2, markersize=7, zorder=3)

    ax1.set_xscale('log')
    ax1.set_xlabel('Parameters')
    ax1.set_ylabel('Test AUC')
    ax1.set_ylim(0.875, 0.96)
    ax1.set_title('(a) AUC vs. parameter count (10% data)')
    _style_ax(ax1)
    ax1.yaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
    ax1.legend(frameon=True, fancybox=False, edgecolor='#cccccc',
               framealpha=0.95, loc='lower right')

    matched_data = []
    for model in MODEL_ORDER:
        pts = [(r['n_params'], r['test']['auc'], r.get('growth_rate', 0))
               for r in pareto if r['model'] == model]
        if not pts:
            continue
        best_at_100k = min(pts, key=lambda x: abs(x[0] - 100_000))
        matched_data.append((model, best_at_100k[0], best_at_100k[1], best_at_100k[2]))

    matched_data.sort(key=lambda x: x[2])
    y_pos = np.arange(len(matched_data))
    colors = [MODEL_STYLE[m[0]]['color'] for m in matched_data]
    aucs_matched = [m[2] for m in matched_data]
    labels_matched = [
        f'{MODEL_STYLE[m[0]]["label"]} ({m[1]//1000}K)'
        for m in matched_data
    ]

    bars = ax2.barh(y_pos, aucs_matched, color=colors, edgecolor='white',
                    linewidth=0.6, height=0.6, zorder=3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels_matched)
    ax2.set_xlabel('Test AUC')
    ax2.set_xlim(0.915, 0.96)
    ax2.set_title('(b) Ranking at ~100K parameters')
    _style_ax(ax2)
    ax2.xaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)

    for bar, auc in zip(bars, aucs_matched):
        ax2.text(auc + 0.0005, bar.get_y() + bar.get_height() / 2,
                 f'{auc:.3f}', va='center', fontsize=9, fontweight='medium')

    for model in MODEL_ORDER:
        s = MODEL_STYLE[model]
        if model not in data_eff:
            continue
        fracs_aucs = sorted(data_eff[model].items())
        fracs, aucs = zip(*fracs_aucs)
        ax3.plot(fracs, aucs, marker=s['marker'], color=s['color'],
                 label=s['label'], linewidth=2, markersize=7, zorder=3)

    ax3.set_xscale('log')
    ax3.set_xlabel('Training data fraction')
    ax3.set_ylabel('Test AUC')
    ax3.set_title('(c) Data efficiency (~100K params)')
    ax3.set_xticks([0.01, 0.05, 0.1, 0.25])
    ax3.set_xticklabels(['1%', '5%', '10%', '25%'])
    ax3.set_ylim(0.855, 0.96)
    _style_ax(ax3)
    ax3.yaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
    ax3.legend(frameon=True, fancybox=False, edgecolor='#cccccc',
               framealpha=0.95, loc='lower right', fontsize=7)

    fig.tight_layout()
    out_path = Path(__file__).parent / 'pareto_analysis.png'
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix('.pdf'))
    print(f'Saved to {out_path} (+ PDF)')

    # Print text summary
    print('\n' + '='*70)
    print('KEY FINDINGS')
    print('='*70)

    print('\n1. PARETO RANKING (full sweep, 10% data):')
    for model in MODEL_ORDER:
        pts = sorted(
            [(r['n_params'], r['test']['auc']) for r in pareto if r['model'] == model],
            key=lambda x: x[0],
        )
        best = max(pts, key=lambda x: x[1])
        print(f'   {MODEL_STYLE[model]["label"]:25s}  best AUC={best[1]:.4f} at {best[0]:>10,} params')

    print('\n2. HEAD-TO-HEAD at ~100K params:')
    for m, params, auc, gr in sorted(matched_data, key=lambda x: x[2], reverse=True):
        print(f'   {MODEL_STYLE[m]["label"]:25s}  AUC={auc:.4f}  ({params:,} params, gr={gr})')

    print('\n3. SCALING BEHAVIOR:')
    for model in MODEL_ORDER:
        pts = sorted(
            [(r['n_params'], r['test']['auc']) for r in pareto if r['model'] == model],
            key=lambda x: x[0],
        )
        if len(pts) >= 2:
            slope = 'degrades' if pts[-1][1] < pts[0][1] else 'improves'
            delta = pts[-1][1] - pts[0][1]
            print(f'   {MODEL_STYLE[model]["label"]:25s}  {pts[0][1]:.4f} → {pts[-1][1]:.4f}  ({slope}, {delta:+.4f})')

    print('\n4. DATA EFFICIENCY (matched ~100K params):')
    for m in MODEL_ORDER:
        if m not in data_eff:
            continue
        fracs_aucs = sorted(data_eff[m].items())
        vals = '  '.join(f'{f:.0%}={a:.4f}' for f, a in fracs_aucs)
        print(f'   {MODEL_STYLE[m]["label"]:25s}  {vals}')


if __name__ == '__main__':
    main()
