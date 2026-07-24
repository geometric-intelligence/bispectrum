#!/usr/bin/env python3
"""Summarize the cumulative Spherical MNIST feature ablation."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float]:
    return {
        'mean': statistics.fmean(values),
        'std': statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def main() -> int:
    args = parse_args()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(args.results_dir.glob('*/results.json')):
        record = json.loads(path.read_text())
        feature_set = str(record['bispectrum_features'])
        mode = str(record['train_mode'])
        grouped[(feature_set, mode)].append(record)

    rows: list[dict[str, Any]] = []
    print('| Features | Active real channels | C seeds | C/C | C/R | R/R (optional) |')
    print('|---|---:|---:|---:|---:|---:|')
    for feature_set in ('bootstrap', 'bootstrap_self', 'bootstrap_self_cg', 'full'):
        canonical = grouped[(feature_set, 'C')]
        rotated = grouped[(feature_set, 'R')]
        if not canonical:
            raise RuntimeError(f'Missing canonical-training runs for {feature_set}')
        cc = stats([float(run['test_c']['accuracy']) for run in canonical])
        cr = stats([float(run['test_r']['accuracy']) for run in canonical])
        rr = stats([float(run['test_r']['accuracy']) for run in rotated]) if rotated else None
        active_dim = int(canonical[0]['active_invariant_dim'])
        row = {
            'feature_set': feature_set,
            'active_real_channels': active_dim,
            'canonical_seeds': len(canonical),
            'rotated_seeds': len(rotated),
            'c_over_c': cc,
            'r_over_r': rr,
            'c_over_r': cr,
        }
        rows.append(row)
        rr_text = '—' if rr is None else f'{rr["mean"]:.3f}±{rr["std"]:.3f}'
        print(
            f'| {feature_set} | {active_dim} | {len(canonical)} | '
            f'{cc["mean"]:.3f}±{cc["std"]:.3f} | '
            f'{cr["mean"]:.3f}±{cr["std"]:.3f} | '
            f'{rr_text} |'
        )

    payload = {'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(f'Wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
