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


def main() -> None:
    pareto = load_results(PARETO_DIR)
    onepct = load_results(ONEPCT_DIR)
    onepct_agg = aggregate_1pct(onepct)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('PCam Invariant Pooling: Pareto Analysis (10% data, seed=42)', fontsize=14, y=0.98)

    # Panel 1: AUC vs Params (Pareto curves)
    ax = axes[0, 0]
    for model in MODEL_ORDER:
        s = MODEL_STYLE[model]
        pts = sorted(
            [(r['n_params'], r['test']['auc']) for r in pareto if r['model'] == model],
            key=lambda x: x[0],
        )
        if not pts:
            continue
        params, aucs = zip(*pts)
        ax.plot(params, aucs, marker=s['marker'], color=s['color'],
                label=s['label'], linewidth=2, markersize=8, zorder=3)

    ax.set_xscale('log')
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Test AUC')
    ax.set_title('(a) AUC vs. Parameter Count')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.88, 0.96)
    ax.axhline(y=0.9277, color='#888888', linestyle=':', alpha=0.5, label='_standard@102K')

    # Panel 2: AUC vs Params zoomed to 60K-200K overlap region
    ax = axes[0, 1]
    for model in MODEL_ORDER:
        s = MODEL_STYLE[model]
        pts = sorted(
            [(r['n_params'], r['test']['auc']) for r in pareto
             if r['model'] == model and 50_000 <= r['n_params'] <= 250_000],
            key=lambda x: x[0],
        )
        if not pts:
            continue
        params, aucs = zip(*pts)
        ax.plot(params, aucs, marker=s['marker'], color=s['color'],
                label=s['label'], linewidth=2, markersize=10, zorder=3)
        for p, a in zip(params, aucs):
            ax.annotate(f'{a:.3f}', (p, a), textcoords='offset points',
                        xytext=(0, 10), fontsize=7, ha='center', color=s['color'])

    ax.set_xlabel('Parameters')
    ax.set_ylabel('Test AUC')
    ax.set_title('(b) Head-to-Head at ~100K Params')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.90, 0.96)

    # Panel 3: 1% vs 10% comparison (at gr=12 / default params)
    ax = axes[1, 0]
    pareto_gr12 = {r['model']: r['test']['auc'] for r in pareto
                   if r.get('growth_rate') == 12 or (r['model'] == 'standard' and r['n_params'] == 101959)}

    x_pos = np.arange(len(MODEL_ORDER))
    width = 0.35

    auc_1pct = [onepct_agg.get(m, {}).get('mean_auc', 0) for m in MODEL_ORDER]
    err_1pct = [onepct_agg.get(m, {}).get('std_auc', 0) for m in MODEL_ORDER]
    auc_10pct = [pareto_gr12.get(m, 0) for m in MODEL_ORDER]

    bars1 = ax.bar(x_pos - width/2, auc_1pct, width, yerr=err_1pct,
                   label='1% data (3 seeds)', color=[MODEL_STYLE[m]['color'] for m in MODEL_ORDER],
                   alpha=0.5, capsize=3, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x_pos + width/2, auc_10pct, width,
                   label='10% data (1 seed)', color=[MODEL_STYLE[m]['color'] for m in MODEL_ORDER],
                   alpha=0.9, capsize=3, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([MODEL_STYLE[m]['label'] for m in MODEL_ORDER], fontsize=8, rotation=15)
    ax.set_ylabel('Test AUC')
    ax.set_title('(c) 1% vs 10% Data (gr=12, default arch)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.82, 0.97)
    for i, (v1, v10) in enumerate(zip(auc_1pct, auc_10pct)):
        ax.text(i - width/2, v1 + 0.005, f'{v1:.3f}', ha='center', fontsize=7)
        ax.text(i + width/2, v10 + 0.005, f'{v10:.3f}', ha='center', fontsize=7)

    # Panel 4: Parameter efficiency (AUC at matched ~100K budget)
    ax = axes[1, 1]
    matched_data = []
    for model in MODEL_ORDER:
        pts = [(r['n_params'], r['test']['auc'], r.get('growth_rate', 0))
               for r in pareto if r['model'] == model]
        if not pts:
            continue
        best_at_100k = min(pts, key=lambda x: abs(x[0] - 100_000))
        matched_data.append((model, best_at_100k[0], best_at_100k[1], best_at_100k[2]))

    matched_data.sort(key=lambda x: x[2], reverse=True)
    y_pos = np.arange(len(matched_data))
    colors = [MODEL_STYLE[m[0]]['color'] for m in matched_data]
    aucs = [m[2] for m in matched_data]
    labels = [f'{MODEL_STYLE[m[0]]["label"]}\n({m[1]:,} params, gr={m[3]})' for m in matched_data]

    bars = ax.barh(y_pos, aucs, color=colors, edgecolor='black', linewidth=0.5, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Test AUC')
    ax.set_title('(d) Ranking at Closest to ~100K Params')
    ax.set_xlim(0.90, 0.96)
    ax.grid(True, alpha=0.3, axis='x')
    for bar, auc in zip(bars, aucs):
        ax.text(auc + 0.001, bar.get_y() + bar.get_height()/2,
                f'{auc:.4f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = Path(__file__).parent / 'pareto_analysis.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {out_path}')

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

    print('\n4. DATA EFFICIENCY (1% → 10% improvement, gr=12):')
    for m in MODEL_ORDER:
        a1 = onepct_agg.get(m, {}).get('mean_auc', 0)
        a10 = pareto_gr12.get(m, 0)
        if a1 and a10:
            print(f'   {MODEL_STYLE[m]["label"]:25s}  {a1:.4f} → {a10:.4f}  ({a10-a1:+.4f})')


if __name__ == '__main__':
    main()
