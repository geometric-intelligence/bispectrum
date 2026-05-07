#!/usr/bin/env python3
"""Aggregate normalized experiment results and guard against incomplete cells."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

MetricKey = tuple[str, str, int | None, str]


def load_results(paths: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and path.name == 'results.json':
            candidates = [path]
        else:
            candidates = sorted(path.glob('**/results.json'))
        for candidate in candidates:
            with candidate.open() as f:
                record = json.load(f)
            record['_path'] = str(candidate)
            results.append(record)
    return results


def canonical_train_mode(record: dict[str, Any]) -> str:
    mode = str(record.get('train_mode', 'C'))
    return 'C' if mode in {'C', 'NR'} else 'R'


def protocol_values(record: dict[str, Any]) -> dict[str, float]:
    train_mode = canonical_train_mode(record)
    test_c = record.get('test_c') or record.get('test_nr') or record.get('test') or {}
    test_r = record.get('test_r') or {}
    metric_name = 'auc' if 'auc' in test_c else 'accuracy'
    values: dict[str, float] = {}
    if metric_name in test_c:
        values[f'{train_mode}/C'] = float(test_c[metric_name])
    if metric_name in test_r:
        values[f'{train_mode}/R'] = float(test_r[metric_name])
    return values


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {'n': 0, 'mean': math.nan, 'std': math.nan, 'ci95': math.nan}
    if len(values) == 1:
        return {'n': 1, 'mean': values[0], 'std': 0.0, 'ci95': math.nan}
    sd = stdev(values)
    return {
        'n': len(values),
        'mean': mean(values),
        'std': sd,
        'ci95': 1.96 * sd / math.sqrt(len(values)),
    }


def _train_size(record: dict[str, Any]) -> int | None:
    """Return the canonical train_size (int) or None for full-set runs."""
    raw = record.get('train_size')
    if raw is None:
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def aggregate(results: list[dict[str, Any]]) -> dict[MetricKey, list[float]]:
    grouped: dict[MetricKey, list[float]] = defaultdict(list)
    for record in results:
        dataset = str(record.get('dataset') or infer_dataset(record.get('_path', 'unknown')))
        model = str(record.get('model', 'unknown'))
        size = _train_size(record)
        for protocol, value in protocol_values(record).items():
            grouped[(dataset, model, size, protocol)].append(value)
    return grouped


def infer_dataset(path: str) -> str:
    if 'pcam' in path:
        return 'pcam'
    if 'organ3d' in path:
        return 'organ3d'
    if 'spherical_mnist' in path or 'smnist' in path:
        return 'spherical_mnist'
    return 'unknown'


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Aggregate normalized results and validate seed/protocol coverage.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('paths', nargs='+', help='Result directories or results.json files.')
    parser.add_argument('--min_seeds', type=int, default=5)
    parser.add_argument('--required_protocols', nargs='+', default=['C/C', 'C/R', 'R/R'])
    parser.add_argument('--output', type=str, default=None, help='Optional JSON summary path.')
    parser.add_argument('--fail_on_warn', action='store_true')
    args = parser.parse_args()

    results = load_results(args.paths)
    grouped = aggregate(results)
    warnings: list[str] = []
    summary: list[dict[str, Any]] = []
    if not results:
        warnings.append('no results.json files found')

    def _size_sort_key(cell: tuple[str, str, int | None]) -> tuple[str, str, int]:
        dataset, model, size = cell
        return (dataset, model, size if size is not None else 10**12)

    seen_cells = sorted({key[:3] for key in grouped}, key=_size_sort_key)
    for dataset, model, size in seen_cells:
        size_label = 'full' if size is None else str(size)
        for protocol in args.required_protocols:
            key = (dataset, model, size, protocol)
            stats = summarize(grouped.get(key, []))
            row = {
                'dataset': dataset,
                'model': model,
                'train_size': size,
                'train_size_label': size_label,
                'protocol': protocol,
                **stats,
            }
            summary.append(row)
            if stats['n'] == 0:
                warnings.append(f'missing cell: {dataset} {model} n={size_label} {protocol}')
            elif int(stats['n']) < args.min_seeds:
                warnings.append(
                    f'low seed count: {dataset} {model} n={size_label} '
                    f'{protocol} n_seeds={stats["n"]} < {args.min_seeds}'
                )

    print(f'Loaded {len(results)} result files.')
    for row in summary:
        n = int(row['n'])
        mean_val = row['mean']
        std_val = row['std']
        ci_val = row['ci95']
        print(
            f'{row["dataset"]:<16} {row["model"]:<24} n={row["train_size_label"]:<6} '
            f'{row["protocol"]:<3} seeds={n:<2} mean={mean_val:.4f} '
            f'std={std_val:.4f} ci95={ci_val:.4f}'
        )

    if warnings:
        print('\nWARNINGS')
        for warning in warnings:
            print(f'- {warning}')

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({'summary': summary, 'warnings': warnings}, indent=2))

    if warnings and args.fail_on_warn:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
