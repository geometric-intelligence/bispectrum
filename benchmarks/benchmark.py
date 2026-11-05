"""Bispectrum benchmark suite.

Produces 4 benchmarks comparing selective vs full bispectrum across all modules:
  1. Coefficient count scaling (output_size vs |G|)
  2. Wall-clock forward pass timing
  3. Wall-clock inversion timing
  4. GPU batch scaling (throughput vs batch_size)

Usage:
    python benchmarks/benchmark.py              # all benchmarks, GPU included
    python benchmarks/benchmark.py --cpu-only   # skip GPU benchmarks
    python benchmarks/benchmark.py --bench 1 3  # run only benchmarks 1 and 3
    python benchmarks/benchmark.py --paper      # paper figures + table; caches figures/paper_timings.json
    python benchmarks/benchmark.py --replot     # redraw them from the cache, no GPU
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess  # nosec B404
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Final, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.benchmark as torchbench
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator

from bispectrum import CnonCn, DnonDn, OctaonOcta, SO2onDisk, SO3onS2, TorusOnTorus

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assets import brand  # noqa: E402
from experiments.make_grid_figure import (  # noqa: E402
    LEGEND_FONT_SIZE,
    SUBTITLE_FONT_SIZE,
    TITLE_FONT_SIZE,
    _warn_if_clipped,
    configure_style,
)

FIGURES_DIR = Path(__file__).parent / 'figures'

# SO2onDisk L=64 takes ~minutes for Bessel root computation; cap at L=32.
_SO2ONDISK_LS = [8, 16, 32]


def _so3_params(lmax: int) -> dict[str, int]:
    return {'lmax': lmax, 'nlat': max(2 * (lmax + 1), 16), 'nlon': max(4 * (lmax + 1), 32)}


def _make_input(
    module_name: str,
    params: dict[str, Any],
    batch: int,
    device: torch.device,
) -> torch.Tensor:
    """Create a random input tensor appropriate for the given module."""
    if module_name == 'CnonCn':
        return torch.randn(batch, params['n'], device=device)
    if module_name == 'TorusOnTorus':
        ns = params['ns']
        return torch.randn(batch, *ns, device=device)
    if module_name == 'DnonDn':
        return torch.randn(batch, 2 * params['n'], device=device)
    if module_name == 'SO2onDisk':
        L = params['L']
        return torch.randn(batch, L, L, device=device)
    if module_name == 'SO3onS2':
        return torch.randn(batch, params['nlat'], params['nlon'], device=device)
    if module_name == 'OctaonOcta':
        return torch.randn(batch, 24, device=device)
    raise ValueError(f'Unknown module: {module_name}')


def _group_order(module_name: str, params: dict[str, Any]) -> int:
    """Compute |G| for display purposes."""
    if module_name == 'CnonCn':
        return params['n']
    if module_name == 'TorusOnTorus':
        return math.prod(params['ns'])
    if module_name == 'DnonDn':
        return 2 * params['n']
    if module_name == 'SO2onDisk':
        return params['L'] ** 2
    if module_name == 'SO3onS2':
        return (params.get('lmax', 5) + 1) ** 2
    if module_name == 'OctaonOcta':
        return 24
    return 0


def _build_module(
    module_name: str,
    params: dict[str, Any],
    selective: bool,
) -> torch.nn.Module:
    if module_name == 'CnonCn':
        return CnonCn(n=params['n'], selective=selective)
    if module_name == 'TorusOnTorus':
        return TorusOnTorus(ns=params['ns'], selective=selective)
    if module_name == 'DnonDn':
        return DnonDn(n=params['n'], selective=selective)
    if module_name == 'SO2onDisk':
        return SO2onDisk(L=params['L'], selective=selective)
    if module_name == 'SO3onS2':
        return SO3onS2(
            lmax=params.get('lmax', 5),
            nlat=params.get('nlat', 64),
            nlon=params.get('nlon', 128),
            selective=selective,
        )
    if module_name == 'OctaonOcta':
        return OctaonOcta(selective=selective)
    raise ValueError(f'Unknown module: {module_name}')


def bench_coefficient_count() -> plt.Figure:
    """Benchmark 1: coefficient count scaling."""
    print('\n' + '=' * 70)
    print('BENCHMARK 1: Coefficient Count Scaling')
    print('=' * 70)

    selective_configs: list[tuple[str, list[dict[str, Any]]]] = [
        ('CnonCn', [{'n': n} for n in [4, 8, 16, 32, 64, 128, 256, 512, 1024]]),
        ('TorusOnTorus', [{'ns': (n, n)} for n in [4, 8, 16, 32, 64, 128, 256]]),
        ('DnonDn', [{'n': n} for n in [4, 8, 16, 32, 64, 128, 256]]),
        ('SO2onDisk', [{'L': L} for L in [8, 16, 32]]),
        ('SO3onS2', [_so3_params(l) for l in [2, 4, 8, 16, 32]]),
    ]

    # Full-mode configs: only instantiate for small enough sizes (to avoid OOM),
    # use analytical formula for the rest.
    full_configs: list[tuple[str, list[dict[str, Any]]]] = [
        ('CnonCn', [{'n': n} for n in [4, 8, 16, 32, 64, 128, 256, 512, 1024]]),
        ('TorusOnTorus', [{'ns': (n, n)} for n in [4, 8, 16, 32, 64, 128, 256]]),
        ('SO3onS2', [_so3_params(l) for l in [2, 4, 8]]),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    line_idx = 0
    for module_name, param_list in selective_configs:
        group_orders = []
        output_sizes = []
        for params in param_list:
            bsp = _build_module(module_name, params, selective=True)
            g = _group_order(module_name, params)
            group_orders.append(g)
            output_sizes.append(bsp.output_size)

        label = f'{module_name} selective'
        ax.plot(
            group_orders,
            output_sizes,
            marker=markers[line_idx % len(markers)],
            color=brand.GROUP_COLORS[module_name],
            label=label,
            linewidth=2,
            markersize=6,
        )
        print(f'\n{label}:')
        print(f'  {"|G|":>10s}  {"output_size":>12s}')
        for g, o in zip(group_orders, output_sizes, strict=False):
            print(f'  {g:>10d}  {o:>12d}')
        line_idx += 1

    for module_name, param_list in full_configs:
        group_orders = []
        output_sizes = []
        for params in param_list:
            g = _group_order(module_name, params)
            # Analytical full count: |G|*(|G|+1)/2  (upper triangular)
            output_sizes.append(g * (g + 1) // 2)
            group_orders.append(g)

        label = f'{module_name} full'
        ax.plot(
            group_orders,
            output_sizes,
            marker=markers[line_idx % len(markers)],
            color=brand.GROUP_COLORS[module_name],
            alpha=0.5,
            label=label,
            linewidth=2,
            markersize=6,
            linestyle='--',
        )
        print(f'\n{label}:')
        print(f'  {"|G|":>10s}  {"output_size":>12s}')
        for g, o in zip(group_orders, output_sizes, strict=False):
            print(f'  {g:>10d}  {o:>12d}')
        line_idx += 1

    # OctaonOcta: single point
    bsp_octa = OctaonOcta(selective=True)
    ax.scatter(
        [24],
        [bsp_octa.output_size],
        marker='*',
        s=200,
        color=brand.GROUP_COLORS['OctaonOcta'],
        label=f'OctaonOcta selective ({bsp_octa.output_size})',
        zorder=5,
    )
    print(f'\nOctaonOcta selective: |G|=24, output_size={bsp_octa.output_size}')
    line_idx += 1

    # Reference slopes
    x_ref = [4, 1024]
    ax.plot(x_ref, x_ref, color=brand.MUTED, linestyle=':', label='O(|G|)')
    ax.plot(x_ref, [x**2 for x in x_ref], color=brand.MUTED, linestyle='--', label='O(|G|²)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('|G| (group order)', fontsize=13)
    ax.set_ylabel('output_size (# coefficients)', fontsize=13)
    ax.set_title('Coefficient Count: Selective vs Full Bispectrum', fontsize=14)
    ax.legend(fontsize=9, loc='upper left')
    brand.style_axes(ax, grid='both')
    fig.tight_layout()
    return fig


def _time_fn(
    stmt: str,
    globs: dict[str, Any],
    label: str,
    device: torch.device,
    min_run_time: float = 0.5,
    inference: bool = False,
) -> float:
    """Time a statement, returning median time in milliseconds.

    ``inference`` runs it under ``torch.inference_mode()``; ``SO3onS2`` replays a CUDA graph
    only when autograd is off, so its training-mode forward is ~3x slower on GPU.
    """
    if inference:
        stmt = f'with torch.inference_mode():\n    {stmt}'
        globs = {**globs, 'torch': torch}
    timer = torchbench.Timer(
        stmt=stmt,
        globals=globs,
        label=label,
        num_threads=1,
    )
    measurement = timer.blocked_autorange(min_run_time=min_run_time)
    return measurement.median * 1e3  # seconds -> ms


def bench_forward_pass(device: torch.device) -> plt.Figure:
    """Benchmark 2: wall-clock forward pass timing."""
    dev_name = 'GPU' if device.type == 'cuda' else 'CPU'
    print(f'\n{"=" * 70}')
    print(f'BENCHMARK 2: Forward Pass Wall-Clock ({dev_name})')
    print('=' * 70)

    batch = 16
    # has_full: whether to also benchmark selective=False for this module.
    # Full-mode TorusOnTorus and CnonCn are capped to avoid O(|G|^2) output OOM.
    selective_ns = [8, 16, 32, 64, 128, 256, 512, 1024]
    full_ns_cn = [8, 16, 32, 64, 128, 256]
    selective_torus = [8, 16, 32, 64, 128, 256]
    full_torus = [8, 16, 32]

    configs: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]] | None]] = [
        ('CnonCn', [{'n': n} for n in selective_ns], [{'n': n} for n in full_ns_cn]),
        (
            'TorusOnTorus',
            [{'ns': (n, n)} for n in selective_torus],
            [{'ns': (n, n)} for n in full_torus],
        ),
        ('DnonDn', [{'n': n} for n in [4, 8, 16, 32, 64, 128]], None),
        ('SO2onDisk', [{'L': L} for L in _SO2ONDISK_LS], None),
        (
            'SO3onS2',
            [_so3_params(l) for l in [2, 4, 8, 16, 32]],
            [_so3_params(l) for l in [2, 4, 8]],
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    ax_idx = 0

    for module_name, sel_param_list, full_param_list in configs:
        ax = axes_flat[ax_idx]
        ax_idx += 1

        for sel_mode, sel_label, plist, mk, ls, color in [
            (True, 'selective', sel_param_list, 'o', '-', brand.ORANGE_TEXT),
            (False, 'full', full_param_list, 's', '--', brand.MUTED),
        ]:
            if plist is None:
                continue
            group_orders = []
            times_ms = []
            for params in plist:
                try:
                    bsp = _build_module(module_name, params, selective=sel_mode).to(device)
                except NotImplementedError:
                    continue
                f = _make_input(module_name, params, batch, device)
                g = _group_order(module_name, params)

                t = _time_fn(
                    'bsp(f)', {'bsp': bsp, 'f': f}, f'{module_name}_{g}_{sel_label}', device
                )
                group_orders.append(g)
                times_ms.append(t)

            if group_orders:
                ax.plot(
                    group_orders,
                    times_ms,
                    marker=mk,
                    linestyle=ls,
                    color=color,
                    label=sel_label,
                    linewidth=2,
                )
                print(f'\n{module_name} {sel_label} ({dev_name}):')
                print(f'  {"|G|":>10s}  {"time_ms":>10s}')
                for g, t in zip(group_orders, times_ms, strict=False):
                    print(f'  {g:>10d}  {t:>10.3f}')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('|G|')
        ax.set_ylabel('time (ms)')
        ax.set_title(module_name)
        ax.legend()
        brand.style_axes(ax, grid='both')

    # OctaonOcta: single bar
    ax = axes_flat[ax_idx]
    bsp = OctaonOcta(selective=True).to(device)
    f = _make_input('OctaonOcta', {}, batch, device)
    t = _time_fn('bsp(f)', {'bsp': bsp, 'f': f}, 'OctaonOcta', device)
    ax.bar(['selective'], [t], color=brand.ORANGE)
    ax.set_ylabel('time (ms)')
    ax.set_title('OctaonOcta (|G|=24)')
    ax.text(0, t * 1.05, f'{t:.3f} ms', ha='center', fontsize=10, fontfamily=brand.MONO)
    brand.style_axes(ax)
    print(f'\nOctaonOcta selective ({dev_name}): {t:.3f} ms')

    fig.suptitle(f'Forward Pass Timing — batch={batch}, {dev_name}', fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def bench_inversion(device: torch.device) -> plt.Figure:
    """Benchmark 3: wall-clock inversion timing."""
    dev_name = 'GPU' if device.type == 'cuda' else 'CPU'
    print(f'\n{"=" * 70}')
    print(f'BENCHMARK 3: Inversion Wall-Clock ({dev_name})')
    print('=' * 70)

    batch = 4

    configs: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = [
        ('CnonCn', [{'n': n} for n in [8, 16, 32, 64, 128, 256, 512, 1024]], {}),
        ('TorusOnTorus', [{'ns': (n, n)} for n in [8, 16, 32, 64, 128, 256]], {}),
        ('DnonDn', [{'n': n} for n in [4, 8, 16, 32, 64, 128]], {}),
        ('SO2onDisk', [{'L': L} for L in _SO2ONDISK_LS], {}),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    ax_idx = 0
    for module_name, param_list, _invert_kwargs in configs:
        ax = axes_flat[ax_idx]
        ax_idx += 1

        group_orders: list[int] = []
        fwd_times: list[float] = []
        inv_times: list[float] = []

        for params in param_list:
            bsp = _build_module(module_name, params, selective=True).to(device)
            f = _make_input(module_name, params, batch, device)
            g = _group_order(module_name, params)

            t_fwd = _time_fn('bsp(f)', {'bsp': bsp, 'f': f}, f'{module_name}_fwd_{g}', device)

            with torch.no_grad():
                beta = bsp(f)
            t_inv = _time_fn(
                'bsp.invert(beta)',
                {'bsp': bsp, 'beta': beta},
                f'{module_name}_inv_{g}',
                device,
            )

            group_orders.append(g)
            fwd_times.append(t_fwd)
            inv_times.append(t_inv)

        ax.plot(group_orders, fwd_times, marker='o', label='forward', linewidth=2)
        ax.plot(group_orders, inv_times, marker='s', label='invert', linewidth=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('|G|')
        ax.set_ylabel('time (ms)')
        ax.set_title(module_name)
        ax.legend()
        brand.style_axes(ax, grid='both')

        print(f'\n{module_name} inversion ({dev_name}):')
        print(f'  {"|G|":>10s}  {"fwd_ms":>10s}  {"inv_ms":>10s}')
        for g, tf, ti in zip(group_orders, fwd_times, inv_times, strict=False):
            print(f'  {g:>10d}  {tf:>10.3f}  {ti:>10.3f}')

    # OctaonOcta (LM-based, slower)
    ax = axes_flat[ax_idx]
    ax_idx += 1
    bsp = OctaonOcta(selective=True).to(device)
    f = _make_input('OctaonOcta', {}, batch, device)
    t_fwd = _time_fn('bsp(f)', {'bsp': bsp, 'f': f}, 'OctaonOcta_fwd', device)
    with torch.no_grad():
        beta = bsp(f)
    t_inv = _time_fn(
        'bsp.invert(beta)',
        {'bsp': bsp, 'beta': beta},
        'OctaonOcta_inv',
        device,
    )
    ax.bar(['forward', 'invert'], [t_fwd, t_inv], color=[brand.GRAPHITE, brand.ASH])
    ax.set_ylabel('time (ms)')
    ax.set_title('OctaonOcta (|G|=24, LM inversion)')
    for i, (_lbl, val) in enumerate([('fwd', t_fwd), ('inv', t_inv)]):
        ax.text(i, val * 1.05, f'{val:.2f}', ha='center', fontsize=9, fontfamily=brand.MONO)
    brand.style_axes(ax)
    print(f'\nOctaonOcta ({dev_name}): fwd={t_fwd:.3f} ms, inv={t_inv:.3f} ms')

    # SO3onS2 note
    ax = axes_flat[ax_idx]
    ax.text(
        0.5,
        0.5,
        'SO3onS2\n\nInversion N/A\n(open problem)',
        ha='center',
        va='center',
        fontsize=14,
        transform=ax.transAxes,
        style='italic',
        color=brand.MUTED,
    )
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    print('\nSO3onS2: inversion N/A (open problem)')

    fig.suptitle(f'Inversion Timing — batch={batch}, {dev_name}', fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def bench_gpu_scaling() -> plt.Figure:
    """Benchmark 4: GPU batch scaling (throughput vs batch_size)."""
    print(f'\n{"=" * 70}')
    print('BENCHMARK 4: GPU Batch Scaling')
    print('=' * 70)

    device = torch.device('cuda')

    module_configs: list[tuple[str, dict[str, Any]]] = [
        ('CnonCn', {'n': 128}),
        ('TorusOnTorus', {'ns': (32, 32)}),
        ('DnonDn', {'n': 32}),
        ('SO2onDisk', {'L': 32}),
        ('SO3onS2', _so3_params(16)),
        ('OctaonOcta', {}),
    ]

    batch_sizes = [1, 4, 16, 64, 256, 1024, 4096]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    markers = ['o', 's', '^', 'D', 'v', 'P']

    for midx, (module_name, params) in enumerate(module_configs):
        bsp = _build_module(module_name, params, selective=True).to(device)
        g = _group_order(module_name, params)

        throughputs: list[float] = []
        valid_batches: list[int] = []

        print(f'\n{module_name} (|G|={g}):')
        print(f'  {"batch":>8s}  {"time_ms":>10s}  {"samples/s":>12s}')

        for bs in batch_sizes:
            try:
                f = _make_input(module_name, params, bs, device)
            except RuntimeError:
                # OOM for large batches
                break

            try:
                t = _time_fn('bsp(f)', {'bsp': bsp, 'f': f}, f'{module_name}_b{bs}', device)
            except RuntimeError:
                break

            throughput = bs / (t / 1e3)  # samples per second
            throughputs.append(throughput)
            valid_batches.append(bs)
            print(f'  {bs:>8d}  {t:>10.3f}  {throughput:>12.0f}')

        ax.plot(
            valid_batches,
            throughputs,
            marker=markers[midx % len(markers)],
            color=brand.GROUP_COLORS[module_name],
            label=f'{module_name} (|G|={g})',
            linewidth=2,
            markersize=7,
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('batch size', fontsize=13)
    ax.set_ylabel('throughput (samples/sec)', fontsize=13)
    ax.set_title('GPU Batch Scaling — Forward Pass Throughput', fontsize=14)
    ax.legend(fontsize=10)
    brand.style_axes(ax, grid='both')
    fig.tight_layout()
    return fig


def _paper_rc() -> None:
    """Brand style at print sizes for the paper figures."""
    brand.use_style()
    plt.rcParams.update(
        {
            'font.size': 9,
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 7.5,
            'legend.framealpha': 0.9,
            'figure.dpi': 300,
            'lines.linewidth': 1.2,
            'lines.markersize': 4,
            'axes.linewidth': 0.6,
            'xtick.major.width': 0.5,
            'ytick.major.width': 0.5,
            'xtick.minor.width': 0.3,
            'ytick.minor.width': 0.3,
            'grid.linewidth': 0.4,
        }
    )


def _panel_label(ax: plt.Axes, label: str) -> None:  # type: ignore[name-defined]
    ax.text(
        0.03,
        0.93,
        f'({label})',
        transform=ax.transAxes,
        fontsize=10,
        fontweight='semibold',
        va='top',
        ha='left',
    )


def _grid(ax: plt.Axes) -> None:  # type: ignore[name-defined]
    brand.style_axes(ax, grid='both')


def _save_figure(fig: plt.Figure, stem: str, dpi: int = 300) -> None:  # type: ignore[name-defined]
    brand.finish(fig)
    fig.savefig(FIGURES_DIR / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / f'{stem}.png', bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f'\nSaved: {FIGURES_DIR / f"{stem}.pdf"}')


def paper_fig_coefficients() -> None:
    """Paper Figure 1: selective vs full coefficient counts (no timing, CPU-only)."""
    _paper_rc()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    c = brand.GROUP_COLORS

    fig1, ax1 = plt.subplots(figsize=(5.5, 2.8))

    sel_modules: list[tuple[str, str, list[dict[str, Any]], str, str]] = [
        (
            r'$C_n$',
            'CnonCn',
            [{'n': n} for n in [4, 8, 16, 32, 64, 128, 256, 512, 1024]],
            c['CnonCn'],
            'o',
        ),
        (
            r'$\mathbb{T}^2$',
            'TorusOnTorus',
            [{'ns': (n, n)} for n in [4, 8, 16, 32, 64, 128, 256]],
            c['TorusOnTorus'],
            's',
        ),
        (r'$D_n$', 'DnonDn', [{'n': n} for n in [4, 8, 16, 32, 64, 128, 256]], c['DnonDn'], '^'),
        (
            r'$\mathrm{SO}(2)/D^2$',
            'SO2onDisk',
            [{'L': L} for L in [8, 16, 32]],
            c['SO2onDisk'],
            'v',
        ),
        (
            r'$\mathrm{SO}(3)/S^2$',
            'SO3onS2',
            [_so3_params(l) for l in [2, 4, 8, 16, 32]],
            c['SO3onS2'],
            'D',
        ),
    ]

    all_gs_flat: list[int] = []
    for label, mod_name, params_list, color, marker in sel_modules:
        gs = [_group_order(mod_name, p) for p in params_list]
        sel_sizes = [_build_module(mod_name, p, selective=True).output_size for p in params_list]
        full_sizes = [g * (g + 1) // 2 for g in gs]
        ax1.plot(
            gs,
            sel_sizes,
            marker=marker,
            color=color,
            label=label + ' sel.',
            markersize=5,
            linewidth=1.3,
        )
        if mod_name in ('CnonCn', 'TorusOnTorus', 'SO3onS2'):
            ax1.plot(
                gs,
                full_sizes,
                marker=marker,
                color=color,
                label=label + ' full',
                linestyle='--',
                alpha=0.45,
                markersize=3,
                linewidth=0.9,
            )
        all_gs_flat.extend(gs)

    bsp_octa = OctaonOcta(selective=True)
    ax1.scatter(
        [24],
        [bsp_octa.output_size],
        marker='*',
        s=80,
        color=c['OctaonOcta'],
        edgecolor=brand.surface_color(),
        linewidth=0.4,
        zorder=5,
        label=r'$O$ sel.',
    )

    x_lo, x_hi = 4, max(all_gs_flat) * 2
    xs = [x_lo, x_hi]
    ax1.plot(xs, xs, color=brand.MUTED, linestyle=':', linewidth=0.8, label=r'$O(|G|)$')
    ax1.fill_between(
        xs,
        [x**2 for x in xs],
        [x_hi**2] * 2,
        color=brand.SAND,
        alpha=0.6,
        linewidth=0,
        label=r'full $O(|G|^2)$ region',
    )
    ax1.plot(
        xs,
        [x**2 for x in xs],
        color=brand.MUTED,
        linestyle='-.',
        linewidth=0.8,
        label=r'$O(|G|^2)$',
    )

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$|G|$')
    ax1.set_ylabel('# coefficients')
    ax1.set_title('Bispectral coefficient count: selective vs. full')
    ax1.legend(
        fontsize=7, ncol=3, columnspacing=1.0, handlelength=1.5, loc='upper left', framealpha=0.9
    )
    ax1.set_ylim(bottom=5)
    _grid(ax1)

    fig1.tight_layout()
    _save_figure(fig1, 'paper_fig1')


FORWARD_BATCH: Final = 16
INVERSION_BATCH: Final = 4
TIMINGS_PATH: Final = FIGURES_DIR / 'paper_timings.json'

CN_NS: Final = (8, 16, 32, 64, 128, 256, 512, 1024)
TORUS_NS: Final = (8, 16, 32, 64, 128, 256)
# The full torus map indexes every frequency pair in a Python list: 8.4M at |G|=4096, 134M at 16384.
TORUS_FULL_NS: Final = (8, 16, 32, 64)
THROUGHPUT_BATCHES: Final = (1, 4, 16, 64, 256, 1024, 4096)
THROUGHPUT_CONFIGS: Final[tuple[tuple[str, dict[str, Any]], ...]] = (
    ('CnonCn', {'n': 128}),
    ('TorusOnTorus', {'ns': (32, 32)}),
    ('DnonDn', {'n': 32}),
    ('SO3onS2', _so3_params(16)),
    ('OctaonOcta', {}),
)
INVERSION_CONFIGS: Final[tuple[tuple[str, tuple[dict[str, Any], ...]], ...]] = (
    ('CnonCn', tuple({'n': n} for n in CN_NS)),
    ('TorusOnTorus', tuple({'ns': (n, n)} for n in TORUS_NS)),
    ('DnonDn', tuple({'n': n} for n in (4, 8, 16, 32, 64, 128))),
    ('SO2onDisk', tuple({'L': L} for L in _SO2ONDISK_LS)),
    ('OctaonOcta', ({},)),
)
TABLE_CONFIGS: Final[tuple[tuple[str, str, str, dict[str, Any], bool, bool], ...]] = (
    (r'\texttt{CnonCn}', r'$C_{128}$', 'CnonCn', {'n': 128}, True, True),
    (
        r'\texttt{TorusOnTorus}',
        r'$C_{32}{\times}C_{32}$',
        'TorusOnTorus',
        {'ns': (32, 32)},
        True,
        True,
    ),
    (r'\texttt{DnonDn}', r'$D_{32}$', 'DnonDn', {'n': 32}, True, False),
    (r'\texttt{SO2onDisk}', r'$\mathrm{SO}(2)$', 'SO2onDisk', {'L': 16}, True, False),
    (r'\texttt{SO3onS2}', r'$\mathrm{SO}(3)$', 'SO3onS2', _so3_params(16), False, False),
    (r'\texttt{OctaonOcta}', r'$O$', 'OctaonOcta', {}, True, False),
)

# Legend order and marks for the library modules; colors come from brand.GROUP_COLORS. Adjacent grays
# alternate solid/filled and dashed/hollow so coinciding curves (C_n and T^2 inversion) stay separable.
GROUP_STYLE: Final[dict[str, tuple[str, str, bool]]] = {
    'CnonCn': (r'$C_n$', 'o', False),
    'TorusOnTorus': (r'$\mathbb{T}^2$', 's', True),
    'DnonDn': (r'$D_n$', '^', False),
    'SO2onDisk': (r'SO(2) on disk', 'v', True),
    'SO3onS2': (r'SO(3) on $S^2$', 'D', False),
    'OctaonOcta': (r'$O$', '*', True),
}
CAPTION_NAMES: Final[dict[str, str]] = {
    'CnonCn': r'$C_n$',
    'TorusOnTorus': r'$\mathbb{T}^2$',
    'DnonDn': r'$D_n$',
    'SO2onDisk': 'the disk',
    'SO3onS2': r'$\mathrm{SO}(3)$ on $S^2$',
    'OctaonOcta': r'$O$',
}

TIMING_FIGURE_SIZE: Final = (5.5, 1.8)
TIMING_MARGINS: Final = {'left': 0.42, 'right': 0.03, 'bottom': 0.33, 'top': 0.47}


type Curve = list[list[float]]


class ThroughputCurve(TypedDict):
    group_order: int
    batch: list[int]
    ms: list[float]


class TableRow(TypedDict):
    tex_name: str
    group_tex: str
    module: str
    group_order: int
    coefs_sel: int
    coefs_full: int | None
    fwd_sel: float
    fwd_inf: float
    fwd_full: float | None
    inv: float | None


class PaperTimings(TypedDict):
    meta: dict[str, str | int | float]
    forward: dict[str, dict[str, Curve]]
    throughput: dict[str, ThroughputCurve]
    inversion: dict[str, Curve]
    table: list[TableRow]


def _forward_curve(
    module: str, params_list: Sequence[dict[str, Any]], *, selective: bool, device: torch.device
) -> Curve:
    mode = 'selective' if selective else 'full'
    points: Curve = []
    for params in params_list:
        try:
            bsp = _build_module(module, params, selective=selective).to(device)
            f = _make_input(module, params, FORWARD_BATCH, device)
            ms = _time_fn('bsp(f)', {'bsp': bsp, 'f': f}, f'{module}_{mode}', device)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f'  {module} {mode} {params}: out of memory, curve stops here')
            break
        g = _group_order(module, params)
        points.append([g, ms])
        print(f'  forward {module} {mode} |G|={g}: {ms:.3f} ms')
    return points


def _throughput_curve(
    module: str, params: dict[str, Any], device: torch.device
) -> ThroughputCurve:
    bsp = _build_module(module, params, selective=True).to(device)
    batches: list[int] = []
    times: list[float] = []
    for batch in THROUGHPUT_BATCHES:
        try:
            f = _make_input(module, params, batch, device)
            ms = _time_fn('bsp(f)', {'bsp': bsp, 'f': f}, f'{module}_b{batch}', device)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        batches.append(batch)
        times.append(ms)
    best = max(b / (t / 1e3) for b, t in zip(batches, times, strict=True))
    print(f'  throughput {module}: max {best:.3g} samples/s')
    return {'group_order': _group_order(module, params), 'batch': batches, 'ms': times}


def _inversion_curve(
    module: str, params_list: Sequence[dict[str, Any]], device: torch.device
) -> Curve:
    points: Curve = []
    for params in params_list:
        bsp = _build_module(module, params, selective=True).to(device)
        f = _make_input(module, params, INVERSION_BATCH, device)
        with torch.no_grad():
            beta = bsp(f)
        ms = _time_fn('bsp.invert(beta)', {'bsp': bsp, 'beta': beta}, f'{module}_inv', device)
        g = _group_order(module, params)
        points.append([g, ms])
        print(f'  inversion {module} |G|={g}: {ms:.2f} ms')
    return points


def _table_row(
    tex_name: str,
    group_tex: str,
    module: str,
    params: dict[str, Any],
    has_inv: bool,
    has_full: bool,
    device: torch.device,
) -> TableRow:
    g = _group_order(module, params)
    bsp = _build_module(module, params, selective=True).to(device)
    f = _make_input(module, params, FORWARD_BATCH, device)
    fwd_sel = _time_fn('bsp(f)', {'bsp': bsp, 'f': f}, f'tab_{module}_sel', device)
    fwd_inf = _time_fn('bsp(f)', {'bsp': bsp, 'f': f}, f'tab_{module}_inf', device, inference=True)
    fwd_full = None
    if has_full:
        bsp_full = _build_module(module, params, selective=False).to(device)
        fwd_full = _time_fn('bsp(f)', {'bsp': bsp_full, 'f': f}, f'tab_{module}_full', device)
    inv = None
    if has_inv:
        with torch.no_grad():
            beta = bsp(f)
        inv = _time_fn('bsp.invert(beta)', {'bsp': bsp, 'beta': beta}, f'tab_{module}_inv', device)
    return {
        'tex_name': tex_name,
        'group_tex': group_tex,
        'module': module,
        'group_order': g,
        'coefs_sel': bsp.output_size,
        'coefs_full': g * (g + 1) // 2 if has_full else None,
        'fwd_sel': fwd_sel,
        'fwd_inf': fwd_inf,
        'fwd_full': fwd_full,
        'inv': inv,
    }


def _git_revision() -> str:
    def git(*args: str) -> str:
        return subprocess.run(  # nosec B603 B607
            ['git', *args], cwd=FIGURES_DIR.parent, capture_output=True, text=True, check=False
        ).stdout.strip()

    dirty = git('status', '--porcelain', '--', '../src')
    return git('rev-parse', '--short', 'HEAD') + ('+dirty-src' if dirty else '')


def measure_paper_timings(device: torch.device) -> PaperTimings:
    """Time everything behind paper Figures 5-6 and the benchmark table (GPU, ~10 min)."""
    print('\nForward pass')
    forward = {
        'CnonCn': {
            'selective': _forward_curve(
                'CnonCn', [{'n': n} for n in CN_NS], selective=True, device=device
            ),
            'full': _forward_curve(
                'CnonCn', [{'n': n} for n in CN_NS], selective=False, device=device
            ),
        },
        'TorusOnTorus': {
            'selective': _forward_curve(
                'TorusOnTorus', [{'ns': (n, n)} for n in TORUS_NS], selective=True, device=device
            ),
            'full': _forward_curve(
                'TorusOnTorus',
                [{'ns': (n, n)} for n in TORUS_FULL_NS],
                selective=False,
                device=device,
            ),
        },
    }
    print('\nThroughput')
    throughput = {
        module: _throughput_curve(module, params, device) for module, params in THROUGHPUT_CONFIGS
    }
    print('\nInversion')
    inversion = {
        module: _inversion_curve(module, params_list, device)
        for module, params_list in INVERSION_CONFIGS
    }
    print('\nTable')
    table = [_table_row(*config, device=device) for config in TABLE_CONFIGS]
    meta: dict[str, str | int | float] = {
        'device': torch.cuda.get_device_name(device),
        'torch': torch.__version__,
        'cuda': str(torch.version.cuda),
        'revision': _git_revision(),
        'date': datetime.now().astimezone().isoformat(timespec='seconds'),
        'forward_batch': FORWARD_BATCH,
        'inversion_batch': INVERSION_BATCH,
        'statistic': 'median of torch.utils.benchmark blocked_autorange(min_run_time=0.5), one thread',
    }
    return {
        'meta': meta,
        'forward': forward,
        'throughput': throughput,
        'inversion': inversion,
        'table': table,
    }


def _timing_figure(*, sharey: bool, wspace: float) -> tuple[Figure, tuple[Axes, Axes]]:
    width, height = TIMING_FIGURE_SIZE
    fig = plt.figure(figsize=TIMING_FIGURE_SIZE)
    grid = fig.add_gridspec(
        1,
        2,
        left=TIMING_MARGINS['left'] / width,
        right=1.0 - TIMING_MARGINS['right'] / width,
        bottom=TIMING_MARGINS['bottom'] / height,
        top=1.0 - TIMING_MARGINS['top'] / height,
        wspace=wspace,
    )
    left = fig.add_subplot(grid[0, 0])
    right = fig.add_subplot(grid[0, 1], sharey=left if sharey else None)
    return fig, (left, right)


def _title(ax: Axes, title: str, subtitle: str) -> None:
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='medium', pad=10)
    ax.annotate(
        subtitle,
        xy=(0.5, 1.0),
        xycoords='axes fraction',
        xytext=(0, 3.5),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=SUBTITLE_FONT_SIZE,
        color=brand.MUTED,
    )


def _log_axes(ax: Axes, xs: Sequence[float], pad: float = 1.35) -> None:
    """Log-log axes whose x-range hugs the data, so no tick label overhangs the canvas."""
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(min(xs) / pad, max(xs) * pad)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    brand.style_axes(ax, grid='both')


def _top_legend(fig: Figure, handles: Sequence[Line2D]) -> None:
    fig.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(
            0.5 + (TIMING_MARGINS['left'] - TIMING_MARGINS['right']) / 2 / fig.get_figwidth(),
            1.0,
        ),
        ncol=len(handles),
        frameon=False,
        borderaxespad=0.0,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.8,
        columnspacing=1.4,
    )


def _save_timing_figure(fig: Figure, stem: str) -> None:
    brand.finish(fig)
    _warn_if_clipped(fig, stem)
    with plt.rc_context({'savefig.bbox': 'standard'}):
        fig.savefig(FIGURES_DIR / f'{stem}.pdf')
        fig.savefig(FIGURES_DIR / f'{stem}.png', dpi=300)
    plt.close(fig)
    print(f'Saved: {FIGURES_DIR / f"{stem}.pdf"}')


def _mode_style(mode: str) -> dict[str, Any]:
    if mode == 'selective':
        return {
            'color': brand.ORANGE_TEXT,
            'marker': 'o',
            'markersize': 3.2,
            'markeredgecolor': brand.surface_color(),
            'markeredgewidth': 0.4,
            'linewidth': 1.3,
        }
    return {
        'color': brand.INK,
        'marker': 's',
        'markersize': 2.8,
        'markerfacecolor': brand.surface_color(),
        'markeredgewidth': 0.8,
        'linewidth': 0.9,
        'linestyle': (0, (3, 1.5)),
    }


def plot_forward_figure(data: PaperTimings) -> None:
    """Paper Figure 5: selective vs. full forward time on C_n and the torus."""
    fig, axes = _timing_figure(sharey=True, wspace=0.12)
    panels = (
        ('CnonCn', '(a) Cyclic group $C_n$', '$n$'),
        ('TorusOnTorus', r'(b) Torus $\mathbb{T}^2 = C_n \times C_n$', r'$|G| = n^2$'),
    )
    for ax, (module, title, xlabel) in zip(axes, panels, strict=True):
        for mode in ('full', 'selective'):
            xs, ys = zip(*data['forward'][module][mode], strict=True)
            ax.plot(xs, ys, **_mode_style(mode), zorder=3 if mode == 'selective' else 2)
        _log_axes(ax, [x for x, _ in data['forward'][module]['selective']])
        ax.set_xlabel(xlabel)
        _title(ax, title, f'forward pass, batch {data["meta"]["forward_batch"]}')
    axes[0].set_ylabel('Forward time (ms)')
    axes[1].tick_params(labelleft=False)
    handles = [
        Line2D([], [], **_mode_style('selective'), label='Selective (ours)'),
        Line2D([], [], **_mode_style('full'), label='Full bispectrum'),
    ]
    _top_legend(fig, handles)
    _save_timing_figure(fig, 'paper_fig2')


def _group_line(module: str) -> dict[str, Any]:
    label, marker, dashed = GROUP_STYLE[module]
    color = brand.GROUP_COLORS[module]
    style: dict[str, Any] = {
        'color': color,
        'marker': marker,
        'markersize': 5.0 if marker == '*' else 3.2,
        'linewidth': 1.1,
        'label': label,
    }
    if dashed:
        hollow = {
            'markerfacecolor': brand.surface_color(),
            'markeredgecolor': color,
            'markeredgewidth': 0.8,
        }
        return style | hollow | {'linestyle': (0, (3, 1.5))}
    return style | {'markeredgecolor': brand.surface_color(), 'markeredgewidth': 0.4}


def plot_scaling_figure(data: PaperTimings) -> None:
    """Paper Figure 6: selective throughput against batch size, and inversion time against |G|."""
    fig, (ax_throughput, ax_inversion) = _timing_figure(sharey=False, wspace=0.32)
    for module, curve in data['throughput'].items():
        rates = [b / (ms / 1e3) for b, ms in zip(curve['batch'], curve['ms'], strict=True)]
        ax_throughput.plot(curve['batch'], rates, **_group_line(module))
    _log_axes(ax_throughput, [b for c in data['throughput'].values() for b in c['batch']])
    ax_throughput.set_xlabel('Batch size')
    ax_throughput.set_ylabel('Throughput (samples/s)')
    _title(ax_throughput, '(a) Throughput', 'selective forward pass')

    for module, points in data['inversion'].items():
        xs, ys = zip(*points, strict=True)
        style = _group_line(module)
        if len(xs) == 1:
            style['linestyle'] = 'none'
        ax_inversion.plot(xs, ys, **style)
    _log_axes(ax_inversion, [x for points in data['inversion'].values() for x, _ in points])
    ax_inversion.set_xlabel('$|G|$')
    ax_inversion.set_ylabel('Inversion time (ms)')
    _title(
        ax_inversion,
        '(b) Inversion',
        f'signal from selective invariant, batch {data["meta"]["inversion_batch"]}',
    )

    handles = [Line2D([], [], **_group_line(module)) for module in GROUP_STYLE]
    _top_legend(fig, handles)
    _save_timing_figure(fig, 'paper_fig3')


def write_benchmark_table(data: PaperTimings) -> None:
    def cell(value: float | None) -> str:
        return '--' if value is None else f'{value:.2f}'

    rows = [
        f'        {r["tex_name"]} & {r["group_tex"]} & {r["group_order"]} & {r["coefs_sel"]} '
        f'& {"--" if r["coefs_full"] is None else r["coefs_full"]} & {cell(r["fwd_sel"])} '
        f'& {cell(r["fwd_inf"])} & {cell(r["fwd_full"])} & {cell(r["inv"])} \\\\'
        for r in data['table']
    ]
    table_tex = (
        r"""\begin{table}[t]
    \centering
    \caption{Benchmark summary for the \texttt{bispectrum} library.
    Coefficient counts compare the selective ($O(|G|)$) and full ($O(|G|^2)$) bispectra.
    Timings are median wall-clock on a single """
        + str(data['meta']['device'])
        + r""" (batch\,=\,16 for forward, 4 for inversion),
    with autograd enabled (Sel.) and under \texttt{torch.inference\_mode()} (Inf.).
    ``--'' indicates the mode is not implemented.}
    \label{tab:benchmarks}
    \small
    \begin{tabular}{llrrrrrrr}
        \toprule
        Module & $G$ & $|G|$ & \multicolumn{2}{c}{Coefs} & \multicolumn{3}{c}{Fwd (ms)} & Inv (ms) \\
        \cmidrule(lr){4-5} \cmidrule(lr){6-8}
         & & & Sel. & Full & Sel. & Inf. & Full & Sel. \\
        \midrule
"""
        + '\n'.join(rows)
        + r"""
        \bottomrule
    \end{tabular}
\end{table}
"""
    )
    table_path = FIGURES_DIR / 'table_benchmarks.tex'
    table_path.write_text(table_tex)
    print(f'Saved: {table_path}')


def _tex_int(value: float) -> str:
    return f'{round(value):,}'.replace(',', '{,}')


def _tex_sci(value: float) -> str:
    mantissa, exponent = f'{value:.1e}'.split('e')
    return rf'{mantissa}{{\times}}10^{{{int(exponent)}}}'


def _ms(value: float) -> str:
    if value < 1:
        return f'{value:.2g}'
    return f'{value:.1f}' if value < 10 else f'{value:.0f}'


def _ms_range(values: Sequence[float]) -> str:
    lo, hi = _ms(min(values)), _ms(max(values))
    return lo if lo == hi else f'{lo}--{hi}'


def _and(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1]


def _slope(points: Curve) -> float:
    xs, ys = np.log(np.asarray(points, dtype=float).T)
    return float(np.polyfit(xs, ys, 1)[0])


def _check(claim: bool, message: str) -> None:
    if not claim:
        print(f'WARNING: caption claim no longer holds: {message}')


def write_timing_captions(data: PaperTimings) -> None:
    """Captions for Figures 5-6 with every number read from the timings."""
    meta = data['meta']
    device = str(meta['device']).replace('NVIDIA ', '').replace(' 80GB HBM3', '')

    cn_sel = [ms for _, ms in data['forward']['CnonCn']['selective']]
    cn_full = data['forward']['CnonCn']['full']
    t_sel = data['forward']['TorusOnTorus']['selective']
    t_full = data['forward']['TorusOnTorus']['full']
    cn_full_n, cn_full_ms = cn_full[-1]
    cn_sel_at = dict(map(tuple, data['forward']['CnonCn']['selective']))[cn_full_n]
    t_full_g, t_full_ms = t_full[-1]
    t_sel_at = dict(map(tuple, t_sel))[t_full_g]
    t_sel_ms = [ms for _, ms in t_sel]
    _check(max(cn_sel) / min(cn_sel) < 1.5, 'C_n selective is flat')
    _check(max(t_sel_ms) / min(t_sel_ms) < 1.5, 'torus selective is flat')
    forward = rf"""\caption{{\textbf{{The selective forward pass costs the same at every group size; the full
    bispectrum grows with it.}} Median forward time, {device}, batch {meta['forward_batch']}, autograd enabled.
    (a)~$C_n$: the selective pass takes {_ms_range(cn_sel)}\,ms at every $n$ from 8 to
    {_tex_int(data['forward']['CnonCn']['selective'][-1][0])}; at $n{{=}}{_tex_int(cn_full_n)}$ the full bispectrum
    takes {_ms(cn_full_ms)}\,ms, ${round(cn_full_ms / cn_sel_at):,}\times$ longer.
    (b)~$\mathbb{{T}}^2$: {_ms_range(t_sel_ms)}\,ms up to $|G|{{=}}{_tex_int(t_sel[-1][0])}$; the full
    bispectrum, run up to $|G|{{=}}{_tex_int(t_full_g)}$ ({_tex_int(t_full_g * (t_full_g + 1) // 2)} coefficients),
    takes {_ms(t_full_ms)}\,ms there, ${round(t_full_ms / t_sel_at):,}\times$ longer.}}
  \label{{fig:forward-timing}}
"""

    rates = {
        module: max(b / (ms / 1e3) for b, ms in zip(c['batch'], c['ms'], strict=True))
        for module, c in data['throughput'].items()
    }
    fast = [CAPTION_NAMES[m] for m, r in rates.items() if r >= 1e7]
    _check(len(fast) >= 2, '>= 2 modules above 1e7 samples/s')
    s2 = data['throughput']['SO3onS2']
    s2_rates = [b / (ms / 1e3) for b, ms in zip(s2['batch'], s2['ms'], strict=True)]
    _check(s2_rates[-1] / s2_rates[-3] < 1.5, 'SO(3) throughput saturates')
    inv = data['inversion']
    slopes = {module: _slope(points) for module, points in inv.items() if len(points) > 2}
    linear = [m for m, k in slopes.items() if 0.8 <= k <= 1.2]
    o_ms = inv['OctaonOcta'][0][1]
    slope_text = _and([rf'{k:.1f} for {CAPTION_NAMES[m]}' for m, k in slopes.items()])
    scaling = rf"""\caption{{\textbf{{Selective $G$-bispectra have high throughput of millions of samples/s
    on one {device} and are invertible.}}
    (a)~Selective forward throughput against batch size, {device}, autograd enabled ($C_{{128}}$,
    $C_{{32}}{{\times}}C_{{32}}$, $D_{{32}}$, $S^2$ at $L{{=}}16$, $O$): {_and(fast)} exceed $10^7$ samples/s,
    $O$ reaches ${_tex_sci(rates['OctaonOcta'])}$, and $\mathrm{{SO}}(3)$ on $S^2$ saturates near
    ${_tex_sci(rates['SO3onS2'])}$. (b)~Time to recover a signal from its selective invariant (batch
    {meta['inversion_batch']}); log--log slopes against $|G|$ are {slope_text}. $O$ ({_ms(o_ms)}\,ms) uses
    iterative Levenberg--Marquardt corrections; $\mathrm{{SO}}(3)$ on $S^2$ has no closed-form inversion.}}
  \label{{fig:gpu-scaling}}
"""
    _check(len(linear) >= 2, 'at least two inversion curves are roughly linear')
    for name, text in (('caption_forward.tex', forward), ('caption_scaling.tex', scaling)):
        (FIGURES_DIR / name).write_text(text)
        print(f'Saved: {FIGURES_DIR / name}')


def replot_paper_figures(data: PaperTimings) -> None:
    paper_fig_coefficients()
    configure_style()
    plot_forward_figure(data)
    plot_scaling_figure(data)
    write_benchmark_table(data)
    write_timing_captions(data)


def paper_figures(device: torch.device) -> None:
    """Measure on the GPU, cache the raw timings, then draw from the cache."""
    print('\n' + '=' * 70)
    print('PAPER FIGURES')
    print('=' * 70)
    data = measure_paper_timings(device)
    TIMINGS_PATH.write_text(json.dumps(data, indent=1) + '\n')
    print(f'Saved: {TIMINGS_PATH}')
    replot_paper_figures(data)
    print('\nPaper figures complete.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Bispectrum benchmark suite')
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Skip GPU benchmarks',
    )
    parser.add_argument(
        '--bench',
        nargs='+',
        type=int,
        default=[1, 2, 3, 4],
        help='Which benchmarks to run (1-4, default: all)',
    )
    parser.add_argument(
        '--paper',
        action='store_true',
        help='Generate NeurIPS publication figures and LaTeX table only',
    )
    parser.add_argument(
        '--replot',
        action='store_true',
        help=f'Redraw the paper figures, table and captions from {TIMINGS_PATH.name} (no GPU)',
    )
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    has_cuda = torch.cuda.is_available() and not args.cpu_only
    cpu = torch.device('cpu')
    gpu = torch.device('cuda') if has_cuda else None

    print(f'PyTorch {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if has_cuda:
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    if args.replot:
        replot_paper_figures(json.loads(TIMINGS_PATH.read_text()))
        return

    if args.paper:
        if not has_cuda:
            print('ERROR: --paper requires CUDA. Remove --cpu-only or use a GPU machine.')
            return
        paper_figures(gpu)  # type: ignore[arg-type]
        print('\nDone.')
        return

    brand.use_style()
    print(f'Running benchmarks: {args.bench}')

    if 1 in args.bench:
        _save_figure(bench_coefficient_count(), 'bench1_coefficient_count', dpi=150)

    if 2 in args.bench:
        _save_figure(bench_forward_pass(cpu), 'bench2_forward_cpu', dpi=150)
        if has_cuda:
            _save_figure(bench_forward_pass(gpu), 'bench2_forward_gpu', dpi=150)  # type: ignore[arg-type]

    if 3 in args.bench:
        _save_figure(bench_inversion(cpu), 'bench3_inversion_cpu', dpi=150)
        if has_cuda:
            _save_figure(bench_inversion(gpu), 'bench3_inversion_gpu', dpi=150)  # type: ignore[arg-type]

    if 4 in args.bench:
        if has_cuda:
            _save_figure(bench_gpu_scaling(), 'bench4_gpu_scaling', dpi=150)
        else:
            print('\nBenchmark 4 (GPU scaling) skipped: no CUDA device available.')

    print('\nDone.')


if __name__ == '__main__':
    main()
