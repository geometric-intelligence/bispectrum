#!/usr/bin/env python3
"""Aggregate L=15 reconstruction evidence for the rebuttal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feature-threshold', type=float, default=1e-2)
    parser.add_argument('--aligned-threshold', type=float, default=0.3)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        'median': float(np.median(array)),
        'p25': float(np.percentile(array, 25)),
        'p75': float(np.percentile(array, 75)),
        'max': float(np.max(array)),
    }


def summarize(
    payload: dict[str, Any],
    feature_threshold: float,
    aligned_threshold: float,
) -> dict[str, Any]:
    records = payload['records']
    feature = [float(row['final_recon_rel']) for row in records]
    aligned = [float(row['aligned_image_space_rel']) for row in records]
    invariance = [
        float(row['invariance_rel']) for row in records if int(row.get('rot_idx', 0)) > 0
    ]
    joint = [
        feature_value <= feature_threshold and aligned_value <= aligned_threshold
        for feature_value, aligned_value in zip(feature, aligned, strict=True)
    ]
    return {
        'signal_source': payload['meta'].get('signal_source', 'smnist'),
        'meta': payload['meta'],
        'n_records': len(records),
        'feature_residual': stats(feature),
        'aligned_image_residual': stats(aligned),
        'invariance_residual': stats(invariance) if invariance else None,
        'feature_threshold': feature_threshold,
        'feature_success_rate': sum(value <= feature_threshold for value in feature) / len(feature),
        'aligned_threshold': aligned_threshold,
        'aligned_success_rate': sum(value <= aligned_threshold for value in aligned) / len(aligned),
        'joint_success_rate': sum(joint) / len(joint),
    }


def main() -> int:
    args = parse_args()
    summaries = [
        summarize(json.loads(path.read_text()), args.feature_threshold, args.aligned_threshold)
        for path in args.results
    ]

    print('| Source | N | Feature residual median [IQR] | Aligned residual median [IQR] | Joint success |')
    print('|---|---:|---:|---:|---:|')
    for row in summaries:
        feature = row['feature_residual']
        aligned = row['aligned_image_residual']
        print(
            f'| {row["signal_source"]} | {row["n_records"]} | '
            f'{feature["median"]:.3e} [{feature["p25"]:.3e}, {feature["p75"]:.3e}] | '
            f'{aligned["median"]:.3f} [{aligned["p25"]:.3f}, {aligned["p75"]:.3f}] | '
            f'{row["joint_success_rate"]:.1%} |'
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'rows': summaries}, indent=2) + '\n')
    print(f'Wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
