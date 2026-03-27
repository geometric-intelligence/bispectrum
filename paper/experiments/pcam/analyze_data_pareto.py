#!/usr/bin/env python3
"""Analyze data-efficiency Pareto sweep: AUC vs. training set fraction at matched ~100K params."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent / 'pcam_results_data_pareto'

MODEL_STYLE: dict[str, dict] = {
    'standard':    {'color': '#888888', 'marker': 's', 'label': 'Standard (augmented)'},
    'norm':        {'color': '#2196F3', 'marker': 'o', 'label': 'NormReLU'},
    'gate':        {'color': '#FF9800', 'marker': '^', 'label': 'Gated'},
    'fourier_elu': {'color': '#9C27B0', 'marker': 'D', 'label': 'Fourier-ELU'},
    'bispectrum':  {'color': '#E53935', 'marker': '*', 'label': 'Bispectrum'},
}
MODEL_ORDER = ['standard', 'norm', 'gate', 'fourier_elu', 'bispectrum']


def load_all_results(base_dir: Path) -> list[dict]:
    results: list[dict] = []
    for frac_dir in sorted(base_dir.iterdir()):
        if not frac_dir.is_dir() or not frac_dir.name.startswith('frac_'):
            continue
        for run_dir in sorted(frac_dir.iterdir()):
            rpath = run_dir / 'results.json'
            if rpath.is_file():
                with open(rpath) as f:
                    results.append(json.load(f))
    return results


def aggregate(
    results: list[dict],
) -> dict[str, dict[float, dict[str, float]]]:
    """Group by (model, train_fraction), compute mean/std of test AUC across seeds."""
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in results:
        key = (r['model'], r['train_fraction'])
        grouped[key].append(r['test']['auc'])

    agg: dict[str, dict[float, dict[str, float]]] = defaultdict(dict)
    for (model, frac), aucs in grouped.items():
        agg[model][frac] = {
            'mean': float(np.mean(aucs)),
            'std': float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            'n_seeds': len(aucs),
        }
    return dict(agg)


def main() -> None:
    if not BASE_DIR.exists():
        print(f'No results directory found at {BASE_DIR}')
        return

    results = load_all_results(BASE_DIR)
    if not results:
        print(f'No results.json files found under {BASE_DIR}')
        return

    agg = aggregate(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        'PCam Data Efficiency: AUC vs. Training Set Fraction (matched ~100K params)',
        fontsize=13, y=0.98,
    )

    # Panel 1: AUC vs train fraction (log-x)
    ax = axes[0]
    for model in MODEL_ORDER:
        if model not in agg:
            continue
        s = MODEL_STYLE[model]
        fracs_data = agg[model]
        fracs = sorted(fracs_data.keys())
        means = [fracs_data[f]['mean'] for f in fracs]
        stds = [fracs_data[f]['std'] for f in fracs]

        ax.errorbar(
            fracs, means, yerr=stds,
            marker=s['marker'], color=s['color'], label=s['label'],
            linewidth=2, markersize=8, capsize=4, zorder=3,
        )

    ax.set_xscale('log')
    ax.set_xlabel('Training Set Fraction')
    ax.set_ylabel('Test AUC')
    ax.set_title('(a) Test AUC vs. Data Fraction')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(['1%', '5%', '10%', '25%', '50%', '100%'])

    # Panel 2: relative AUC gain over standard baseline at each fraction
    ax = axes[1]
    standard_data = agg.get('standard', {})
    if standard_data:
        for model in MODEL_ORDER:
            if model == 'standard' or model not in agg:
                continue
            s = MODEL_STYLE[model]
            fracs = sorted(set(agg[model].keys()) & set(standard_data.keys()))
            deltas = [
                agg[model][f]['mean'] - standard_data[f]['mean'] for f in fracs
            ]
            ax.plot(
                fracs, deltas,
                marker=s['marker'], color=s['color'], label=s['label'],
                linewidth=2, markersize=8, zorder=3,
            )

        ax.axhline(0, color='#888888', linestyle='--', alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel('Training Set Fraction')
        ax.set_ylabel('AUC Difference vs. Standard')
        ax.set_title('(b) AUC Gain over Standard Baseline')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
        ax.set_xticklabels(['1%', '5%', '10%', '25%', '50%', '100%'])
    else:
        ax.text(0.5, 0.5, 'No standard baseline data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = Path(__file__).parent / 'data_pareto_analysis.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved figure to {out_path}')

    print('\n' + '=' * 70)
    print('DATA EFFICIENCY SUMMARY')
    print('=' * 70)

    for model in MODEL_ORDER:
        if model not in agg:
            continue
        fracs_data = agg[model]
        fracs = sorted(fracs_data.keys())
        print(f'\n  {MODEL_STYLE[model]["label"]}:')
        for f in fracs:
            d = fracs_data[f]
            std_str = f' ± {d["std"]:.4f}' if d['n_seeds'] > 1 else ''
            print(f'    frac={f:<5}  AUC={d["mean"]:.4f}{std_str}  (n={d["n_seeds"]})')

    if standard_data:
        print('\n  Relative to Standard:')
        for model in MODEL_ORDER:
            if model == 'standard' or model not in agg:
                continue
            fracs = sorted(set(agg[model].keys()) & set(standard_data.keys()))
            deltas = [agg[model][f]['mean'] - standard_data[f]['mean'] for f in fracs]
            if deltas:
                best_frac = fracs[int(np.argmax(deltas))]
                print(
                    f'    {MODEL_STYLE[model]["label"]:25s}  '
                    f'best advantage at frac={best_frac}: {max(deltas):+.4f}'
                )


if __name__ == '__main__':
    main()
