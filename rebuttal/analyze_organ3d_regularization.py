#!/usr/bin/env python3
"""Summarize and plot the OrganMNIST3D high-capacity regularization sweep."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def scalar_stats(values: list[float]) -> dict[str, float]:
    return {
        'mean': statistics.fmean(values),
        'std': statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def best_epoch_record(run: dict[str, Any]) -> dict[str, float | int]:
    history = run.get('history', [])
    if not history:
        raise RuntimeError(f'Run {run.get("run_label")} has no epoch history')
    return max(history, key=lambda row: float(row['val_auc']))


def aggregate(label: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    best = [best_epoch_record(run) for run in runs]
    train = [float(row['train_accuracy']) for row in best]
    val = [float(row['val_accuracy']) for row in best]
    test = [float(run['test_c']['accuracy']) for run in runs]
    return {
        'run_label': label,
        'model': runs[0]['model'],
        'weight_decay': float(runs[0]['weight_decay']),
        'dropout': float(runs[0]['dropout']),
        'seeds': [int(run['seed']) for run in runs],
        'best_epoch': scalar_stats([float(row['epoch']) for row in best]),
        'train_accuracy': scalar_stats(train),
        'val_accuracy': scalar_stats(val),
        'test_accuracy': scalar_stats(test),
        'train_test_gap': scalar_stats([a - b for a, b in zip(train, test, strict=True)]),
    }


def aligned_curves(
    runs: list[dict[str, Any]],
    key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_common = min(len(run['history']) for run in runs)
    values = np.array(
        [[float(row[key]) for row in run['history'][:max_common]] for run in runs],
        dtype=np.float64,
    )
    epochs = np.arange(1, max_common + 1)
    return epochs, values.mean(axis=0), values.std(axis=0)


def plot_analysis(
    grouped: dict[str, list[dict[str, Any]]],
    summaries: list[dict[str, Any]],
    output: Path,
) -> None:
    fig, (ax_curve, ax_bar) = plt.subplots(1, 2, figsize=(8.5, 3.3))
    colors = {'max_base': '#2D6A9F', 'bsp_base': '#C44E52'}
    for label in ('max_base', 'bsp_base'):
        runs = grouped.get(label, [])
        if not runs:
            continue
        for key, linestyle, suffix in (
            ('train_accuracy', '-', 'train'),
            ('val_accuracy', '--', 'validation'),
        ):
            epochs, mean, std = aligned_curves(runs, key)
            ax_curve.plot(
                epochs,
                mean,
                color=colors[label],
                linestyle=linestyle,
                label=f'{label} {suffix}',
            )
            ax_curve.fill_between(epochs, mean - std, mean + std, color=colors[label], alpha=0.12)
    ax_curve.set_xlabel('Epoch')
    ax_curve.set_ylabel('Accuracy')
    ax_curve.set_title('(a) Baseline learning curves')
    ax_curve.legend(fontsize=7)
    ax_curve.grid(alpha=0.2)

    labels = [str(row['run_label']) for row in summaries]
    means = np.array([float(row['test_accuracy']['mean']) for row in summaries])
    stds = np.array([float(row['test_accuracy']['std']) for row in summaries])
    x = np.arange(len(labels))
    ax_bar.bar(x, means, yerr=stds, capsize=3, color='#C44E52')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=35, ha='right', fontsize=7)
    ax_bar.set_ylabel('Test accuracy')
    ax_bar.set_title('(b) Regularization sweep')
    ax_bar.grid(axis='y', alpha=0.2)

    fig.tight_layout()
    fig.savefig(output, bbox_inches='tight')
    fig.savefig(output.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main() -> int:
    args = parse_args()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(args.results_dir.glob('*/results.json')):
        run = json.loads(path.read_text())
        grouped[str(run['run_label'])].append(run)
    if not grouped:
        raise RuntimeError(f'No results found under {args.results_dir}')

    order = [
        'max_base',
        'bsp_base',
        'bsp_wd1e-3',
        'bsp_wd1e-2',
        'bsp_drop0.2',
        'bsp_drop0.5',
        'bsp_wd1e-3_drop0.2',
    ]
    summaries = [aggregate(label, grouped[label]) for label in order if grouped.get(label)]

    print('| Run | WD | Dropout | Seeds | Train ACC | Val ACC | Test ACC | Train-test gap |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|')
    for row in summaries:
        train = row['train_accuracy']
        val = row['val_accuracy']
        test = row['test_accuracy']
        gap = row['train_test_gap']
        print(
            f'| {row["run_label"]} | {row["weight_decay"]:.0e} | {row["dropout"]:.1f} | '
            f'{len(row["seeds"])} | {train["mean"]:.3f}±{train["std"]:.3f} | '
            f'{val["mean"]:.3f}±{val["std"]:.3f} | '
            f'{test["mean"]:.3f}±{test["std"]:.3f} | '
            f'{gap["mean"]:.3f}±{gap["std"]:.3f} |'
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'summary.json').write_text(json.dumps({'rows': summaries}, indent=2) + '\n')
    plot_analysis(grouped, summaries, args.output_dir / 'curves_and_sweep.pdf')
    print(f'Wrote {args.output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
