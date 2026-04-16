"""Benchmark: compare dense Python, Triton, CUDA Graph, and Sparse forward paths.

Usage:
    python benchmarks/bench_sparse.py
"""

from __future__ import annotations

import time

import torch
import torch.utils.benchmark as torchbench

from bispectrum import SO3onS2
from bispectrum.so3_on_s2 import _get_full_sh_coefficients


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

    try:
        from bispectrum._triton_so3 import triton_bispectrum_forward

        has_triton = True
    except ImportError:
        has_triton = False

    header = f'{"lmax":>5s}  {"entries":>7s}  {"dense_py":>10s}  {"triton":>10s}  {"cuda_graph":>10s}  {"sparse":>10s}  {"sp_init":>10s}  {"sp_MB":>8s}'
    _print(header)
    _print('-' * len(header))

    lmax_values = [5, 10, 15, 20, 25, 30]
    batch = 4

    for lmax in lmax_values:
        p = _so3_params(lmax)
        f_cpu = torch.randn(batch, p['nlat'], p['nlon'])

        # Dense module (lmax <= 32)
        orig_thresh = SO3onS2._SPARSE_LMAX_THRESHOLD
        SO3onS2._SPARSE_LMAX_THRESHOLD = 999
        bsp_dense = SO3onS2(**p, selective=True).to(device)
        SO3onS2._SPARSE_LMAX_THRESHOLD = orig_thresh

        f = f_cpu.to(device)
        entries = bsp_dense.output_size

        coeffs = bsp_dense._sht(f)
        fc = _get_full_sh_coefficients(coeffs)

        # Dense Python
        t_dense = _time_fn(
            lambda fc=fc, bsp=bsp_dense, c=coeffs, fv=f, b=batch, e=entries: bsp._forward_python(
                fc, c.dtype, fv.device, b, e
            )
        )

        # Triton
        t_triton = '--'
        if has_triton and hasattr(bsp_dense, '_fused_entry_desc') and device.type == 'cuda':
            t_triton = f'{_time_fn(lambda fc=fc, bsp=bsp_dense, e=entries: triton_bispectrum_forward(fc, bsp, e)):.3f}'

        # CUDA Graph (with dense)
        t_graph = '--'
        if device.type == 'cuda':
            bsp_dense.reset_cuda_graph_cache()
            with torch.no_grad():
                gr = bsp_dense._forward_cuda_graph(f, batch, entries)
            if gr is not None:
                t_graph = f'{_time_fn(lambda bsp=bsp_dense, fv=f, b=batch, e=entries: bsp._forward_cuda_graph(fv, b, e)):.3f}'

        # Sparse
        SO3onS2._SPARSE_LMAX_THRESHOLD = 0
        t0 = time.time()
        bsp_sparse = SO3onS2(**p, selective=True).to(device)
        t_init = time.time() - t0
        SO3onS2._SPARSE_LMAX_THRESHOLD = orig_thresh

        sp_mb = bsp_sparse._sparse_cg_vals.numel() * 8 / 1e6

        coeffs_s = bsp_sparse._sht(f)
        fc_s = _get_full_sh_coefficients(coeffs_s)
        t_sparse = _time_fn(
            lambda fc=fc_s,
            bsp=bsp_sparse,
            c=coeffs_s,
            fv=f,
            b=batch,
            e=entries: bsp._forward_sparse(fc, c.dtype, fv.device, b, e)
        )

        _print(
            f'{lmax:>5d}  {entries:>7d}  {t_dense:>10.3f}  {t_triton:>10s}  '
            f'{t_graph:>10s}  {t_sparse:>10.3f}  {t_init:>10.2f}  {sp_mb:>8.1f}'
        )

    _print()
    _print('Dense + Sparse high-lmax:')
    header2 = f'{"lmax":>5s}  {"entries":>7s}  {"sparse":>10s}  {"sp_init":>10s}  {"sp_MB":>8s}'
    _print(header2)
    _print('-' * len(header2))

    for lmax in [40, 50, 64]:
        p = _so3_params(lmax)
        f_cpu = torch.randn(batch, p['nlat'], p['nlon'])

        SO3onS2._SPARSE_LMAX_THRESHOLD = 0
        t0 = time.time()
        bsp = SO3onS2(**p, selective=True).to(device)
        t_init = time.time() - t0
        SO3onS2._SPARSE_LMAX_THRESHOLD = orig_thresh

        entries = bsp.output_size
        sp_mb = bsp._sparse_cg_vals.numel() * 8 / 1e6
        f = f_cpu.to(device)

        coeffs_s = bsp._sht(f)
        fc_s = _get_full_sh_coefficients(coeffs_s)
        t_sparse = _time_fn(
            lambda fc=fc_s, bsp=bsp, c=coeffs_s, fv=f, b=batch, e=entries: bsp._forward_sparse(
                fc, c.dtype, fv.device, b, e
            )
        )

        _print(f'{lmax:>5d}  {entries:>7d}  {t_sparse:>10.3f}  {t_init:>10.2f}  {sp_mb:>8.1f}')


if __name__ == '__main__':
    bench_forward_paths()
