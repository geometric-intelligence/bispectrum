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
    'standard':    {'color': '#888888', 'marker': 's', 'label': 'Standard'},
    'norm':        {'color': '#2196F3', 'marker': 'o', 'label': 'NormReLU'},
    'gate':        {'color': '#FF9800', 'marker': '^', 'label': 'Gated'},
    'fourier_elu': {'color': '#9C27B0', 'marker': 'D', 'label': 'Fourier-ELU'},
    'bispectrum':  {'color': '#E53935', 'marker': '*', 'label': 'Bispectrum'},
}
MODEL_ORDER = ['standard', 'norm', 'gate', 'fourier_elu', 'bispectrum']

DEFAULT_TRAIN_MODE = 'R'


def _canonical_train_mode(mode: str) -> str:
    """Map legacy `NR` to canonical `C`. Pre-normalized runs default to `R`."""
    if mode in {'C', 'NR'}:
        return 'C'
    if mode == 'R':
        return 'R'
    return DEFAULT_TRAIN_MODE


def _result_train_mode(record: dict) -> str:
    return _canonical_train_mode(record.get('train_mode', DEFAULT_TRAIN_MODE))


def _result_test_metrics(record: dict) -> dict:
    return record.get('test_c') or record.get('test') or {}


def load_results(directory: Path) -> list[dict]:
    results = []
    if not directory.is_dir():
        return results
    for d in sorted(os.listdir(directory)):
        rpath = directory / d / 'results.json'
        if rpath.is_file():
            with open(rpath) as f:
                results.append(json.load(f))
    return results


def aggregate_1pct(results: list[dict], train_mode: str = DEFAULT_TRAIN_MODE) -> dict[str, dict]:
    """Group 1% results by model (filtered to *train_mode*), compute mean +/- std."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        if _result_train_mode(r) != train_mode:
            continue
        grouped[r['model']].append(r)
    agg = {}
    for model, runs in grouped.items():
        aucs = [_result_test_metrics(r).get('auc', 0.0) for r in runs]
        rot_stds = [r.get('rotation_robustness', {}).get('std_auc', 0.0) for r in runs]
        agg[model] = {
            'n_params': runs[0]['n_params'],
            'mean_auc': np.mean(aucs),
            'std_auc': np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0,
            'mean_rot_std': np.mean(rot_stds) if rot_stds else 0.0,
        }
    return agg


def aggregate_pareto_multiseed(
    results: list[dict],
    train_mode: str = DEFAULT_TRAIN_MODE,
) -> dict[str, list[dict]]:
    """Group pareto results by (model, n_params) for *train_mode*, mean +/- std AUC.

    Returns model -> sorted list of {n_params, mean_auc, std_auc, n_seeds}.
    """
    from collections import defaultdict
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in results:
        if _result_train_mode(r) != train_mode:
            continue
        grouped[(r['model'], r['n_params'])].append(_result_test_metrics(r).get('auc', 0.0))

    agg: dict[str, list[dict]] = defaultdict(list)
    for (model, n_params), aucs in grouped.items():
        agg[model].append({
            'n_params': n_params,
            'mean_auc': float(np.mean(aucs)),
            'std_auc': float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            'n_seeds': len(aucs),
        })
    for model in agg:
        agg[model].sort(key=lambda x: x['n_params'])
    return dict(agg)


PCAM_FULL_TRAIN: int = 262_144


def _train_size_value(record: dict) -> int:
    """Resolve a record's training-subset size. Treat None / <=0 / missing as full."""
    raw = record.get('train_size')
    if raw is None:
        return PCAM_FULL_TRAIN
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return PCAM_FULL_TRAIN
    return n if n > 0 else PCAM_FULL_TRAIN


def load_data_pareto(
    train_mode: str = DEFAULT_TRAIN_MODE,
) -> dict[str, dict[int, dict[str, float]]]:
    """Load data-efficiency results with multi-seed aggregation.

    Filters by *train_mode* (default `R`). Returns model -> {train_size: stats}.
    """
    from collections import defaultdict
    raw: dict[tuple[str, int], list[float]] = defaultdict(list)
    if not DATA_PARETO_DIR.is_dir():
        return {}
    for run_dir in sorted(DATA_PARETO_DIR.rglob('results.json')):
        with open(run_dir) as f:
            r = json.load(f)
        if _result_train_mode(r) != train_mode:
            continue
        size = _train_size_value(r)
        raw[(r['model'], size)].append(_result_test_metrics(r).get('auc', 0.0))

    result: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for (model, size), aucs in raw.items():
        result[model][size] = {
            'mean': float(np.mean(aucs)),
            'std': float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            'n_seeds': len(aucs),
        }
    return dict(result)


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
    pareto_agg = aggregate_pareto_multiseed(pareto)
    data_eff = load_data_pareto()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3.8))

    for model in MODEL_ORDER:
        s = MODEL_STYLE[model]
        pts = pareto_agg.get(model, [])
        if not pts:
            continue
        params = [p['n_params'] for p in pts]
        means = [p['mean_auc'] for p in pts]
        stds = [p['std_auc'] for p in pts]

        has_multiseed = any(p['n_seeds'] > 1 for p in pts)
        if has_multiseed:
            ax1.errorbar(params, means, yerr=stds,
                         marker=s['marker'], color=s['color'],
                         label=s['label'], linewidth=2, markersize=7,
                         capsize=3, zorder=3)
        else:
            ax1.plot(params, means, marker=s['marker'], color=s['color'],
                     label=s['label'], linewidth=2, markersize=7, zorder=3)

    ax1.set_xscale('log')
    ax1.set_xlabel('Parameters')
    ax1.set_ylabel('Test AUC')
    ax1.set_ylim(0.75, 0.96)
    ax1.set_title('(a) AUC vs. parameter count (10% data)')
    _style_ax(ax1)
    ax1.yaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
    ax1.legend(frameon=True, fancybox=False, edgecolor='#cccccc',
               framealpha=0.95, loc='lower right')

    matched_data = []
    for model in MODEL_ORDER:
        pts = pareto_agg.get(model, [])
        if not pts:
            continue
        best_at_100k = min(pts, key=lambda x: abs(x['n_params'] - 100_000))
        matched_data.append((model, best_at_100k['n_params'],
                             best_at_100k['mean_auc'], best_at_100k['std_auc']))

    matched_data.sort(key=lambda x: x[2])
    y_pos = np.arange(len(matched_data))
    colors = [MODEL_STYLE[m[0]]['color'] for m in matched_data]
    aucs_matched = [m[2] for m in matched_data]
    errs_matched = [m[3] for m in matched_data]
    labels_matched = [
        f'{MODEL_STYLE[m[0]]["label"]} ({m[1]//1000}K)'
        for m in matched_data
    ]

    bars = ax2.barh(y_pos, aucs_matched, xerr=errs_matched, color=colors,
                    edgecolor='white', linewidth=0.6, height=0.6, zorder=3,
                    capsize=3, error_kw={'linewidth': 0.8, 'capthick': 0.8})
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels_matched)
    ax2.set_xlabel('Test AUC')
    ax2.set_xlim(0.75, 0.96)
    ax2.set_title('(b) Ranking at ~100K parameters')
    _style_ax(ax2)
    ax2.xaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)

    for bar, auc, err in zip(bars, aucs_matched, errs_matched):
        label = f'{auc:.3f}'
        if err > 0:
            label += f'\u00b1{err:.3f}'
        ax2.text(auc + err + 0.001, bar.get_y() + bar.get_height() / 2,
                 label, va='center', fontsize=8, fontweight='medium')

    for model in MODEL_ORDER:
        s = MODEL_STYLE[model]
        if model not in data_eff:
            continue
        sizes = sorted(data_eff[model].keys())
        means = [data_eff[model][n]['mean'] for n in sizes]
        stds = [data_eff[model][n]['std'] for n in sizes]
        has_multiseed = any(data_eff[model][n]['n_seeds'] > 1 for n in sizes)
        if has_multiseed:
            ax3.errorbar(sizes, means, yerr=stds,
                         marker=s['marker'], color=s['color'],
                         label=s['label'], linewidth=2, markersize=7,
                         capsize=3, zorder=3)
        else:
            ax3.plot(sizes, means, marker=s['marker'], color=s['color'],
                     label=s['label'], linewidth=2, markersize=7, zorder=3)

    ax3.set_xscale('log')
    ax3.set_xlabel('Training examples (N)')
    ax3.set_ylabel('Test AUC')
    ax3.set_title('(c) Data efficiency (~100K params)')
    ticks = [100, 500, 2500, 12500, PCAM_FULL_TRAIN]
    ax3.set_xticks(ticks)
    ax3.set_xticklabels([f'{t:,}' if t < PCAM_FULL_TRAIN else 'full' for t in ticks])
    ax3.set_ylim(0.75, 0.96)
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
        pts = pareto_agg.get(model, [])
        if not pts:
            continue
        best = max(pts, key=lambda x: x['mean_auc'])
        std_str = f' +/- {best["std_auc"]:.4f}' if best['n_seeds'] > 1 else ''
        print(f'   {MODEL_STYLE[model]["label"]:25s}  best AUC={best["mean_auc"]:.4f}{std_str} at {best["n_params"]:>10,} params ({best["n_seeds"]} seeds)')

    print('\n2. HEAD-TO-HEAD at ~100K params:')
    for m, params, auc, std in sorted(matched_data, key=lambda x: x[2], reverse=True):
        std_str = f' +/- {std:.4f}' if std > 0 else ''
        print(f'   {MODEL_STYLE[m]["label"]:25s}  AUC={auc:.4f}{std_str}  ({params:,} params)')

    print('\n3. SCALING BEHAVIOR:')
    for model in MODEL_ORDER:
        pts = pareto_agg.get(model, [])
        if len(pts) >= 2:
            slope = 'degrades' if pts[-1]['mean_auc'] < pts[0]['mean_auc'] else 'improves'
            delta = pts[-1]['mean_auc'] - pts[0]['mean_auc']
            print(f'   {MODEL_STYLE[model]["label"]:25s}  {pts[0]["mean_auc"]:.4f} -> {pts[-1]["mean_auc"]:.4f}  ({slope}, {delta:+.4f})')

    print('\n4. DATA EFFICIENCY (matched ~100K params):')
    for m in MODEL_ORDER:
        if m not in data_eff:
            continue
        sizes = sorted(data_eff[m].keys())
        vals_parts = []
        for n in sizes:
            d = data_eff[m][n]
            std_str = f'+/-{d["std"]:.4f}' if d['n_seeds'] > 1 else ''
            label = 'full' if n >= PCAM_FULL_TRAIN else f'n={n:,}'
            vals_parts.append(f'{label}={d["mean"]:.4f}{std_str}')
        print(f'   {MODEL_STYLE[m]["label"]:25s}  {"  ".join(vals_parts)}')


if __name__ == '__main__':
    main()
