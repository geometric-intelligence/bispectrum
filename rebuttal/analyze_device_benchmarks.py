#!/usr/bin/env python3
"""Combine portable device benchmark JSON files into one table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payloads: list[dict[str, Any]] = [json.loads(path.read_text()) for path in args.results]
    combined = {'devices': payloads}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(combined, indent=2) + '\n')

    print('| Device | Backend | PyTorch | Module | Median ms | Status |')
    print('|---|---|---|---|---:|---|')
    for payload in payloads:
        environment = payload['environment']
        device = environment['device_name']
        backend = environment['device']
        torch_version = environment['torch']
        for result in payload['results']:
            median = '—' if result['median_ms'] is None else f'{result["median_ms"]:.3f}'
            status = result['error'] or 'ok'
            print(
                f'| {device} | {backend} | {torch_version} | '
                f'{result["module"]} | {median} | {status} |'
            )
    print(f'Wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
