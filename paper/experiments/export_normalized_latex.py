#!/usr/bin/env python3
"""Export normalized summary JSON as LaTeX table rows for the paper."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def fmt_metric(row: dict[str, Any]) -> str:
    n = int(row['n'])
    if n == 0:
        return '--'
    mean_val = float(row['mean'])
    ci95 = row.get('ci95')
    if ci95 is None or ci95 != ci95:
        return f'{mean_val:.3f}'
    return f'{mean_val:.3f} $\\pm$ {float(ci95):.3f}'


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Export normalized experiment summary to LaTeX.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--summary', required=True, help='normalized_summary.json path.')
    parser.add_argument('--output', required=True, help='Output .tex path.')
    args = parser.parse_args()

    with Path(args.summary).open() as f:
        payload = json.load(f)

    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in payload.get('summary', []):
        size_raw = row.get('train_size')
        size_key = int(size_raw) if isinstance(size_raw, int) else 10**12
        key = (row['dataset'], row['model'], size_key)
        grouped[key][row['protocol']] = row

    lines = [
        '% Auto-generated from normalized_results/normalized_summary.json.',
        '% Re-run paper/experiments/run_normalized_experiments.sh analyze after new runs.',
        '\\begin{tabular}{lllrrrr}',
        '\\toprule',
        'Dataset & Model & N & Seeds & C/C & C/R & R/R \\\\',
        '\\midrule',
    ]
    for (dataset, model, size_key), protocols in sorted(grouped.items()):
        any_row = next(iter(protocols.values()))
        size_label = str(any_row.get('train_size_label', 'full'))
        seed_counts = [int(row['n']) for row in protocols.values() if int(row['n']) > 0]
        seed_str = str(min(seed_counts)) if seed_counts else '0'
        lines.append(
            f'{dataset} & {model} & {size_label} & {seed_str} & '
            f'{fmt_metric(protocols.get("C/C", {"n": 0}))} & '
            f'{fmt_metric(protocols.get("C/R", {"n": 0}))} & '
            f'{fmt_metric(protocols.get("R/R", {"n": 0}))} \\\\'
        )
    lines.extend(['\\bottomrule', '\\end{tabular}', ''])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines))
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
