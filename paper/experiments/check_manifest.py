#!/usr/bin/env python3
"""Validate a planned training run against memory_manifest.jsonl.

Exit codes:
    0  matching record found with status=ok
    2  no matching record (run check first)
    3  matching record(s) exist but none have status=ok
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f'WARN: skipping malformed manifest line: {line[:80]}', file=sys.stderr)
    return records


def _match(record: dict[str, Any], query: dict[str, Any]) -> bool:
    for key, value in query.items():
        if key not in record:
            return False
        ref = record[key]
        if isinstance(value, list):
            if not isinstance(ref, list) or list(ref) != list(value):
                return False
        elif isinstance(value, float):
            if not isinstance(ref, (int, float)) or abs(float(ref) - value) > 1e-9:
                return False
        else:
            if ref != value:
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--dataset', required=True, choices=['pcam', 'organ3d', 'spherical_mnist'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--train_mode', required=True, choices=['C', 'R'])
    parser.add_argument(
        '--train_size', default=None, type=int,
        help='Absolute training subset size. Omit for full training set.',
    )
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--group', default=None)
    parser.add_argument('--growth_rate', default=None, type=int)
    parser.add_argument('--channels', default=None, type=int, nargs='+')
    parser.add_argument('--head_dim', default=None, type=int)
    parser.add_argument('--hidden', default=None, type=int)
    parser.add_argument('--lmax', default=None, type=int)
    args = parser.parse_args()

    records = _load_manifest(args.manifest)
    if not records:
        print(f'NO_MANIFEST: {args.manifest} is missing or empty', file=sys.stderr)
        sys.exit(2)

    train_size = args.train_size if args.train_size and args.train_size > 0 else None
    query: dict[str, Any] = {
        'dataset': args.dataset,
        'model': args.model,
        'train_mode': args.train_mode,
        'train_size': train_size,
        'batch_size': args.batch_size,
    }
    if args.dataset == 'pcam':
        if args.group is None or args.growth_rate is None:
            parser.error('pcam requires --group and --growth_rate')
        query['group'] = args.group
        query['growth_rate'] = args.growth_rate
    elif args.dataset == 'organ3d':
        if args.channels is None or args.head_dim is None:
            parser.error('organ3d requires --channels and --head_dim')
        query['channels'] = list(args.channels)
        query['head_dim'] = args.head_dim
    else:
        if args.hidden is None or args.lmax is None:
            parser.error('spherical_mnist requires --hidden and --lmax')
        query['hidden'] = args.hidden
        query['lmax'] = args.lmax

    matches = [r for r in records if _match(r, query)]
    if not matches:
        print(f'NO_MATCH: nothing in manifest matches query={query}', file=sys.stderr)
        sys.exit(2)

    ok = [r for r in matches if r.get('status') == 'ok']
    if not ok:
        statuses = sorted({r.get('status', '?') for r in matches})
        errors = sorted({r.get('error', '') for r in matches if r.get('error')})
        print(
            f'MANIFEST_FAIL: {len(matches)} match(es) but none ok '
            f'(statuses={statuses}, errors={errors})',
            file=sys.stderr,
        )
        sys.exit(3)


if __name__ == '__main__':
    main()
