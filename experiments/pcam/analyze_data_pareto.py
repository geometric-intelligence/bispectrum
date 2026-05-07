#!/usr/bin/env python3
"""Analyze data-efficiency sweep: AUC vs. absolute training-set size at matched params."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent / 'pcam_results_data_pareto'
PCAM_FULL_TRAIN = 262_144

MODEL_STYLE: dict[str, dict] = {
    'standard': {'color': '#888888', 'marker': 's', 'label': 'Standard'},
    'norm': {'color': '#2196F3', 'marker': 'o', 'label': 'NormReLU'},
    'gate': {'color': '#FF9800', 'marker': '^', 'label': 'Gated'},
    'fourier_elu': {'color': '#9C27B0', 'marker': 'D', 'label': 'Fourier-ELU'},
    'bispectrum': {'color': '#E53935', 'marker': '*', 'label': 'Bispectrum'},
}
MODEL_ORDER = ['standard', 'norm', 'gate', 'fourier_elu', 'bispectrum']

DEFAULT_TRAIN_MODE = 'R'


def _canonical_train_mode(mode: str) -> str:
    if mode in {'C', 'NR'}:
        return 'C'
    if mode == 'R':
        return 'R'
    return DEFAULT_TRAIN_MODE


def _result_train_mode(record: dict) -> str:
    return _canonical_train_mode(record.get('train_mode', DEFAULT_TRAIN_MODE))


def _result_test_metrics(record: dict) -> dict:
    return record.get('test_c') or record.get('test') or {}


def _train_size_value(record: dict) -> int:
    """Resolved absolute training-set size for one run.

    Subset runs report a positive ``train_size``; full-set runs leave it
    unset/None and we bucket them at the dataset's full training count.
    """
    raw = record.get('train_size')
    if raw is not None:
        try:
            n = int(raw)
        except (TypeError, ValueError):
            n = 0
        if n > 0:
            return n
    examples = record.get('train_examples')
    if isinstance(examples, int) and examples > 0:
        return examples
    return PCAM_FULL_TRAIN


def load_all_results(base_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(base_dir.glob('**/results.json'))]


def aggregate(
    results: list[dict],
    train_mode: str = DEFAULT_TRAIN_MODE,
) -> dict[str, dict[int, dict[str, float]]]:
    """Group by (model, train_size) for *train_mode*; report mean/std test AUC."""
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in results:
        if _result_train_mode(r) != train_mode:
            continue
        key = (r['model'], _train_size_value(r))
        grouped[key].append(_result_test_metrics(r).get('auc', 0.0))

    agg: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for (model, size), aucs in grouped.items():
        agg[model][size] = {
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

    all_sizes = sorted({size for sizes in agg.values() for size in sizes})
    if not all_sizes:
        print('No data-efficiency runs to plot.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        'PCam Data Efficiency: AUC vs. Training Examples (matched params)',
        fontsize=13,
        y=0.98,
    )

    ax = axes[0]
    for model in MODEL_ORDER:
        if model not in agg:
            continue
        s = MODEL_STYLE[model]
        sizes = sorted(agg[model].keys())
        means = [agg[model][n]['mean'] for n in sizes]
        stds = [agg[model][n]['std'] for n in sizes]

        ax.errorbar(
            sizes,
            means,
            yerr=stds,
            marker=s['marker'],
            color=s['color'],
            label=s['label'],
            linewidth=2,
            markersize=8,
            capsize=4,
            zorder=3,
        )

    ax.set_xscale('log')
    ax.set_xlabel('Training examples (N)')
    ax.set_ylabel('Test AUC')
    ax.set_title('(a) Test AUC vs. training examples')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_sizes)
    ax.set_xticklabels([f'{n:,}' for n in all_sizes])

    ax = axes[1]
    standard_data = agg.get('standard', {})
    if standard_data:
        for model in MODEL_ORDER:
            if model == 'standard' or model not in agg:
                continue
            s = MODEL_STYLE[model]
            sizes = sorted(set(agg[model].keys()) & set(standard_data.keys()))
            deltas = [agg[model][n]['mean'] - standard_data[n]['mean'] for n in sizes]
            ax.plot(
                sizes,
                deltas,
                marker=s['marker'],
                color=s['color'],
                label=s['label'],
                linewidth=2,
                markersize=8,
                zorder=3,
            )

        ax.axhline(0, color='#888888', linestyle='--', alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel('Training examples (N)')
        ax.set_ylabel('AUC difference vs. Standard')
        ax.set_title('(b) AUC gain over Standard baseline')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(all_sizes)
        ax.set_xticklabels([f'{n:,}' for n in all_sizes])
    else:
        ax.text(
            0.5,
            0.5,
            'No standard baseline data',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=12,
        )

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
        sizes = sorted(agg[model].keys())
        print(f'\n  {MODEL_STYLE[model]["label"]}:')
        for n in sizes:
            d = agg[model][n]
            std_str = f' ± {d["std"]:.4f}' if d['n_seeds'] > 1 else ''
            print(f'    N={n:>7}  AUC={d["mean"]:.4f}{std_str}  (seeds={d["n_seeds"]})')

    if standard_data:
        print('\n  Relative to Standard:')
        for model in MODEL_ORDER:
            if model == 'standard' or model not in agg:
                continue
            sizes = sorted(set(agg[model].keys()) & set(standard_data.keys()))
            deltas = [agg[model][n]['mean'] - standard_data[n]['mean'] for n in sizes]
            if deltas:
                best_n = sizes[int(np.argmax(deltas))]
                print(
                    f'    {MODEL_STYLE[model]["label"]:25s}  '
                    f'best advantage at N={best_n}: {max(deltas):+.4f}'
                )


if __name__ == '__main__':
    main()
