"""Benchmark: compare dense Python, CUDA Graph, and Sparse forward paths.

Usage:
    python benchmarks/bench_sparse.py
"""

from __future__ import annotations

import time

import torch
import torch.utils.benchmark as torchbench

from bispectrum import SO3onS2


def _so3_params(lmax: int) -> dict[str, int]:
    return {'lmax': lmax, 'nlat': max(2 * (lmax + 1), 16), 'nlon': max(4 * (lmax + 1), 32)}


def _time_fn(fn, *, min_run_time: float = 0.3, warmup: int = 3) -> float:
    """Time a callable, return median ms."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    timer = torchbench.Timer(stmt='fn()', globals={'fn': fn}, num_threads=1)
    m = timer.blocked_autorange(min_run_time=min_run_time)
    return m.median * 1e3


def _print(*args: object, **kwargs: object) -> None:
    print(*args, **kwargs, flush=True)


def bench_forward_paths() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _print(f'Device: {device}')
    if device.type == 'cuda':
        _print(f'GPU: {torch.cuda.get_device_name(0)}')
    _print()

    header = f'{"lmax":>5s}  {"entries":>7s}  {"sparse":>10s}  {"cuda_graph":>10s}  {"init_s":>10s}  {"sp_MB":>8s}'
    _print(header)
    _print('-' * len(header))

    batch = 4

    for lmax in [5, 10, 15, 20, 25, 30, 40, 50, 64, 128]:
        p = _so3_params(lmax)
        f_cpu = torch.randn(batch, p['nlat'], p['nlon'])

        t0 = time.time()
        bsp = SO3onS2(**p, selective=True).to(device)
        t_init = time.time() - t0

        f = f_cpu.to(device)
        entries = bsp.output_size
        sp_mb = bsp._sparse_cg_vals.numel() * 8 / 1e6

        t_sparse = _time_fn(lambda bsp=bsp, fv=f: bsp(fv))

        t_graph = '--'
        if device.type == 'cuda':
            bsp.reset_cuda_graph_cache()
            with torch.no_grad():
                gr = bsp._forward_cuda_graph(f, batch, entries)
            if gr is not None:
                t_graph = f'{_time_fn(lambda bsp=bsp, fv=f, b=batch, e=entries: bsp._forward_cuda_graph(fv, b, e)):.3f}'

        _print(
            f'{lmax:>5d}  {entries:>7d}  {t_sparse:>10.3f}  {t_graph:>10s}  '
            f'{t_init:>10.2f}  {sp_mb:>8.1f}'
        )


if __name__ == '__main__':
    bench_forward_paths()
