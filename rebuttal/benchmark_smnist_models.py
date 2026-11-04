#!/usr/bin/env python3
"""Full-model Spherical MNIST inference timing under the ``benchmark_devices.py`` protocol.

Times every Table 4 classifier end to end (grid in, logits out) in float32 inference mode, so the
bispectrum classifier, S2CNN and the tensor-product network are measured identically.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from benchmark_devices import environment, resolve_device, synchronize

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'experiments' / 'spherical_mnist'))
from model import build_model  # noqa: E402

CASES: dict[str, dict[str, Any]] = {
    'bispectrum': {'model_type': 'bispectrum'},
    'power_spectrum': {'model_type': 'power_spectrum'},
    'standard': {'model_type': 'standard'},
    's2cnn': {'model_type': 's2cnn'},
    's2cnn_bn': {'model_type': 's2cnn', 's2cnn_batch_norm': True},
    'tensor_product': {'model_type': 'tensor_product'},
}


@dataclass(frozen=True)
class Result:
    model: str
    batch_size: int
    n_params: int | None
    median_ms: float | None
    p25_ms: float | None
    p75_ms: float | None
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[16, 256])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--inner-loops', type=int, default=20)
    parser.add_argument('--models', nargs='+', choices=sorted(CASES), default=list(CASES))
    parser.add_argument(
        '--case',
        action='append',
        default=[],
        metavar='NAME=JSON',
        help='Extra case, e.g. tp_c32=\'{"model_type": "tensor_product", "tp_channels": 32}\'.',
    )
    parser.add_argument(
        '--tp_grid',
        action='store_true',
        help='Also time every tensor-product architecture in the baseline tuning grid.',
    )
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def tp_grid_cases() -> dict[str, dict[str, Any]]:
    from smnist_baselines_manifest import FAMILIES

    cases: dict[str, dict[str, Any]] = {}
    for flags in FAMILIES['tensor_product'].values():
        channels = int(flags[flags.index('--tp_channels') + 1])
        lmax = int(flags[flags.index('--tp_lmax') + 1])
        cases[f'tp_c{channels}_l{lmax}'] = {
            'model_type': 'tensor_product',
            'tp_channels': channels,
            'tp_lmax': lmax,
        }
    return cases


def time_model(
    name: str,
    kwargs: dict[str, Any],
    batch_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Result:
    n_params: int | None = None
    try:
        model = build_model(**kwargs).to(device).eval()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        x = torch.rand(batch_size, 64, 128, device=device)
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(x)
            synchronize(device)
            samples_ms: list[float] = []
            for _ in range(args.repeats):
                synchronize(device)
                start = time.perf_counter()
                for _ in range(args.inner_loops):
                    model(x)
                synchronize(device)
                samples_ms.append((time.perf_counter() - start) * 1_000 / args.inner_loops)
        q = statistics.quantiles(samples_ms, n=4, method='inclusive')
        return Result(name, batch_size, n_params, statistics.median(samples_ms), q[0], q[2], None)
    except Exception as exc:  # noqa: BLE001 - keep unsupported-backend failures in the artifact
        return Result(name, batch_size, n_params, None, None, None, f'{type(exc).__name__}: {exc}')


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    cases = {name: CASES[name] for name in args.models}
    if args.tp_grid:
        cases.update(tp_grid_cases())
    for spec in args.case:
        name, _, payload = spec.partition('=')
        cases[name] = json.loads(payload)
    results = [
        time_model(n, kw, bs, args, device) for bs in args.batch_sizes for n, kw in cases.items()
    ]

    env = environment(device, argparse.Namespace(batch_size=args.batch_sizes, **vars(args)))
    payload = {'environment': env, 'cases': cases, 'results': [asdict(r) for r in results]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')

    print(f'Device: {env["device_name"]} ({device})')
    print('| Model | Batch | Params | Median ms | IQR ms |')
    print('|---|---:|---:|---:|---:|')
    for r in results:
        if r.error:
            print(f'| {r.model} | {r.batch_size} | {r.n_params or "—"} | ERROR | {r.error} |')
        else:
            print(
                f'| {r.model} | {r.batch_size} | {r.n_params:,} | {r.median_ms:.3f} | [{r.p25_ms:.3f}, {r.p75_ms:.3f}] |'
            )
    print(f'Wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
