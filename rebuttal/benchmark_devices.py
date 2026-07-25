#!/usr/bin/env python3
"""Portable forward-pass benchmark used for the NeurIPS rebuttal."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from bispectrum import CnonCn, DnonDn, OctaonOcta, SO2onDisk, SO3onS2, TorusOnTorus


@dataclass(frozen=True)
class Result:
    module: str
    settings: str
    output_size: int | None
    median_ms: float | None
    p25_ms: float | None
    p75_ms: float | None
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--inner-loops', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable')
    return device


def synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def build_case(
    name: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.Tensor, str]:
    if name == 'CnonCn':
        return (
            CnonCn(n=128, selective=True).to(device=device, dtype=torch.float32),
            torch.randn(batch_size, 128, device=device),
            'n=128',
        )
    if name == 'TorusOnTorus':
        return (
            TorusOnTorus(ns=(32, 32), selective=True).to(device=device, dtype=torch.float32),
            torch.randn(batch_size, 32, 32, device=device),
            'ns=(32,32)',
        )
    if name == 'DnonDn':
        return (
            DnonDn(n=32, selective=True).to(device=device, dtype=torch.float32),
            torch.randn(batch_size, 64, device=device),
            'n=32',
        )
    if name == 'SO2onDisk':
        return (
            SO2onDisk(L=16, selective=True).to(device=device, dtype=torch.float32),
            torch.randn(batch_size, 16, 16, device=device),
            'L=16',
        )
    if name == 'SO3onS2':
        return (
            SO3onS2(lmax=16, nlat=64, nlon=128, selective=True).to(
                device=device, dtype=torch.float32
            ),
            torch.randn(batch_size, 64, 128, device=device),
            'lmax=16, grid=64x128',
        )
    if name == 'OctaonOcta':
        return (
            OctaonOcta(selective=True).to(device=device, dtype=torch.float32),
            torch.randn(batch_size, 24, device=device),
            '|O|=24',
        )
    raise ValueError(f'Unknown benchmark case: {name}')


def benchmark_case(
    name: str,
    batch_size: int,
    warmup: int,
    repeats: int,
    inner_loops: int,
    device: torch.device,
) -> Result:
    settings = ''
    try:
        module, inputs, settings = build_case(name, batch_size, device)
        module.eval()
        with torch.inference_mode():
            for _ in range(warmup):
                module(inputs)
            synchronize(device)

            samples_ms: list[float] = []
            for _ in range(repeats):
                synchronize(device)
                start = time.perf_counter()
                for _ in range(inner_loops):
                    module(inputs)
                synchronize(device)
                samples_ms.append((time.perf_counter() - start) * 1_000 / inner_loops)

        quartiles = statistics.quantiles(samples_ms, n=4, method='inclusive')
        return Result(
            module=name,
            settings=settings,
            output_size=int(getattr(module, 'output_size')),  # noqa: B009 - heterogeneous modules
            median_ms=statistics.median(samples_ms),
            p25_ms=quartiles[0],
            p75_ms=quartiles[2],
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 - preserve unsupported-backend failures in artifact
        return Result(
            module=name,
            settings=settings,
            output_size=None,
            median_ms=None,
            p25_ms=None,
            p75_ms=None,
            error=f'{type(exc).__name__}: {exc}',
        )


def environment(device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'torch': torch.__version__,
        'device': str(device),
        'cuda_runtime': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        'mps_available': torch.backends.mps.is_available(),
        'batch_size': args.batch_size,
        'warmup': args.warmup,
        'repeats': args.repeats,
        'inner_loops': args.inner_loops,
        'dtype': 'float32',
    }
    if device.type == 'cuda':
        metadata['device_name'] = torch.cuda.get_device_name(device)
        metadata['compute_capability'] = list(torch.cuda.get_device_capability(device))
    elif device.type == 'mps':
        metadata['device_name'] = 'Apple Metal Performance Shaders'
    else:
        metadata['device_name'] = platform.processor() or platform.machine()
        metadata['torch_num_threads'] = torch.get_num_threads()
    return metadata


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    names = ['CnonCn', 'TorusOnTorus', 'DnonDn', 'SO2onDisk', 'SO3onS2', 'OctaonOcta']
    results = [
        benchmark_case(
            name,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
            inner_loops=args.inner_loops,
            device=device,
        )
        for name in names
    ]

    payload = {'environment': environment(device, args), 'results': [asdict(row) for row in results]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')

    print(f'Device: {payload["environment"]["device_name"]} ({device})')
    print('| Module | Settings | Output | Median ms | IQR ms |')
    print('|---|---|---:|---:|---:|')
    for row in results:
        if row.error:
            print(f'| {row.module} | {row.settings} | — | ERROR | {row.error} |')
        else:
            assert row.median_ms is not None
            assert row.p25_ms is not None
            assert row.p75_ms is not None
            print(
                f'| {row.module} | {row.settings} | {row.output_size} | '
                f'{row.median_ms:.3f} | [{row.p25_ms:.3f}, {row.p75_ms:.3f}] |'
            )
    print(f'Wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
