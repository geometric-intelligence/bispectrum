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
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.utils.benchmark as torchbench

from bispectrum import CnonCn, DnonDn, OctaonOcta, SO2onD2, SO3onS2, TorusOnTorus

FIGURES_DIR = Path(__file__).parent / 'figures'

# SO2onD2 L=64 takes ~minutes for Bessel root computation; cap at L=32.
_SO2OND2_LS = [8, 16, 32]


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
    if module_name == 'SO2onS1':
        return torch.randn(batch, params['n'], device=device)
    if module_name == 'TorusOnTorus':
        ns = params['ns']
        return torch.randn(batch, *ns, device=device)
    if module_name == 'DnonDn':
        return torch.randn(batch, 2 * params['n'], device=device)
    if module_name == 'SO2onD2':
        L = params['L']
        return torch.randn(batch, L, L, device=device)
    if module_name == 'SO3onS2':
        return torch.randn(batch, params['nlat'], params['nlon'], device=device)
    if module_name == 'OctaonOcta':
        return torch.randn(batch, 24, device=device)
    raise ValueError(f'Unknown module: {module_name}')


def _group_order(module_name: str, params: dict[str, Any]) -> int:
    """Compute |G| for display purposes."""
    if module_name in ('CnonCn', 'SO2onS1'):
        return params['n']
    if module_name == 'TorusOnTorus':
        return math.prod(params['ns'])
    if module_name == 'DnonDn':
        return 2 * params['n']
    if module_name == 'SO2onD2':
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
    if module_name == 'SO2onD2':
        return SO2onD2(L=params['L'], selective=selective)
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
        ('SO2onD2', [{'L': L} for L in [8, 16, 32]]),
        ('SO3onS2', [_so3_params(l) for l in [2, 3, 4, 5]]),
    ]

    # Full-mode configs: only instantiate for small enough sizes (to avoid OOM),
    # use analytical formula for the rest.
    full_configs: list[tuple[str, list[dict[str, Any]]]] = [
        ('CnonCn', [{'n': n} for n in [4, 8, 16, 32, 64, 128, 256, 512, 1024]]),
        ('TorusOnTorus', [{'ns': (n, n)} for n in [4, 8, 16, 32, 64, 128, 256]]),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

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
            color=colors[line_idx % len(colors)],
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
            color=colors[line_idx % len(colors)],
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
        color=colors[line_idx % len(colors)],
        label=f'OctaonOcta selective ({bsp_octa.output_size})',
        zorder=5,
    )
    print(f'\nOctaonOcta selective: |G|=24, output_size={bsp_octa.output_size}')
    line_idx += 1

    # Reference slopes
    x_ref = [4, 1024]
    ax.plot(x_ref, x_ref, 'k:', alpha=0.4, label='O(|G|)')
    ax.plot(x_ref, [x**2 for x in x_ref], 'k--', alpha=0.4, label='O(|G|²)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('|G| (group order)', fontsize=13)
    ax.set_ylabel('output_size (# coefficients)', fontsize=13)
    ax.set_title('Coefficient Count: Selective vs Full Bispectrum', fontsize=14)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    return fig


def _time_fn(
    stmt: str,
    globs: dict[str, Any],
    label: str,
    device: torch.device,
    min_run_time: float = 0.5,
) -> float:
    """Time a statement, returning median time in milliseconds."""
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
        ('SO2onD2', [{'L': L} for L in _SO2OND2_LS], None),
        ('SO3onS2', [_so3_params(l) for l in [2, 3, 4, 5]], None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    ax_idx = 0

    for module_name, sel_param_list, full_param_list in configs:
        ax = axes_flat[ax_idx]
        ax_idx += 1

        for sel_mode, sel_label, plist, mk, ls in [
            (True, 'selective', sel_param_list, 'o', '-'),
            (False, 'full', full_param_list, 's', '--'),
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
                    group_orders, times_ms, marker=mk, linestyle=ls, label=sel_label, linewidth=2
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
        ax.grid(True, alpha=0.3, which='both')

    # OctaonOcta: single bar
    ax = axes_flat[ax_idx]
    bsp = OctaonOcta(selective=True).to(device)
    f = _make_input('OctaonOcta', {}, batch, device)
    t = _time_fn('bsp(f)', {'bsp': bsp, 'f': f}, 'OctaonOcta', device)
    ax.bar(['selective'], [t], color='steelblue')
    ax.set_ylabel('time (ms)')
    ax.set_title('OctaonOcta (|G|=24)')
    ax.text(0, t * 1.05, f'{t:.3f} ms', ha='center', fontsize=10)
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
        ('SO2onD2', [{'L': L} for L in _SO2OND2_LS], {}),
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
        ax.grid(True, alpha=0.3, which='both')

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
    ax.bar(['forward', 'invert'], [t_fwd, t_inv], color=['steelblue', 'darkorange'])
    ax.set_ylabel('time (ms)')
    ax.set_title('OctaonOcta (|G|=24, LM inversion)')
    for i, (_lbl, val) in enumerate([('fwd', t_fwd), ('inv', t_inv)]):
        ax.text(i, val * 1.05, f'{val:.2f}', ha='center', fontsize=9)
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
        color='gray',
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
        ('SO2onD2', {'L': 32}),
        ('SO3onS2', _so3_params(5)),
        ('OctaonOcta', {}),
    ]

    batch_sizes = [1, 4, 16, 64, 256, 1024, 4096]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
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
            color=colors[midx % len(colors)],
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
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    return fig


OKABE_ITO = [
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#F0E442',  # yellow
    '#0072B2',  # blue
    '#D55E00',  # vermilion
    '#CC79A7',  # reddish purple
    '#000000',  # black
]


def _paper_rc() -> None:
    """Set matplotlib rcParams for NeurIPS publication figures."""
    # Try usetex first; fall back to mathtext if LaTeX fonts are missing.
    use_tex = True
    try:
        import subprocess  # nosec B404

        subprocess.run(['kpsewhich', 'type1ec.sty'], check=True, capture_output=True)  # nosec B603 B607
    except (FileNotFoundError, subprocess.CalledProcessError):
        use_tex = False

    plt.rcParams.update(
        {
            'font.family': 'serif',
            'font.serif': ['DejaVu Serif', 'Computer Modern Roman', 'Times New Roman'],
            'mathtext.fontset': 'cm',
            'text.usetex': use_tex,
            'font.size': 9,
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 7.5,
            'legend.framealpha': 0.8,
            'legend.edgecolor': '0.8',
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
    if plt.rcParams.get('text.usetex'):
        txt = rf'\textbf{{({label})}}'
    else:
        txt = f'({label})'
    ax.text(
        0.03,
        0.93,
        txt,
        transform=ax.transAxes,
        fontsize=10,
        fontweight='bold',
        va='top',
        ha='left',
    )


def _grid(ax: plt.Axes) -> None:  # type: ignore[name-defined]
    ax.grid(True, which='major', alpha=0.15, linewidth=0.4)
    ax.grid(False, which='minor')


def paper_figures(device: torch.device) -> None:
    """Generate NeurIPS-quality composite figures and LaTeX table."""
    _paper_rc()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    batch_fwd = 16
    batch_inv = 4
    c = OKABE_ITO

    print('\n' + '=' * 70)
    print('PAPER FIGURES')
    print('=' * 70)

    # === Figure 1: Coefficient count (standalone) =========================
    fig1, ax1 = plt.subplots(figsize=(5.5, 2.8))

    sel_modules: list[tuple[str, str, list[dict[str, Any]], str, str]] = [
        (
            r'$C_n$',
            'CnonCn',
            [{'n': n} for n in [4, 8, 16, 32, 64, 128, 256, 512, 1024]],
            c[4],
            'o',
        ),
        (
            r'$\mathbb{T}^2$',
            'TorusOnTorus',
            [{'ns': (n, n)} for n in [4, 8, 16, 32, 64, 128, 256]],
            c[0],
            's',
        ),
        (r'$D_n$', 'DnonDn', [{'n': n} for n in [4, 8, 16, 32, 64, 128, 256]], c[2], '^'),
        (r'$\mathrm{SO}(2)/D^2$', 'SO2onD2', [{'L': L} for L in [8, 16, 32]], c[6], 'v'),
        (r'$\mathrm{SO}(3)/S^2$', 'SO3onS2', [_so3_params(l) for l in [2, 3, 4, 5]], c[3], 'D'),
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
        if mod_name in ('CnonCn', 'TorusOnTorus'):
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
        [24], [bsp_octa.output_size], marker='*', s=80, color=c[5], zorder=5, label=r'$O$ sel.'
    )

    x_lo, x_hi = 4, max(all_gs_flat) * 2
    xs = [x_lo, x_hi]
    ax1.plot(xs, xs, color='0.55', linestyle=':', linewidth=0.8, label=r'$O(|G|)$')
    ax1.fill_between(
        xs,
        [x**2 for x in xs],
        [x_hi**2] * 2,
        color='0.88',
        alpha=0.3,
        linewidth=0,
        label=r'full $O(|G|^2)$ region',
    )
    ax1.plot(
        xs, [x**2 for x in xs], color='0.55', linestyle='-.', linewidth=0.8, label=r'$O(|G|^2)$'
    )

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$|G|$')
    ylab = r'$\#$ coefficients' if plt.rcParams.get('text.usetex') else '# coefficients'
    ax1.set_ylabel(ylab)
    ax1.set_title('Bispectral coefficient count: selective vs. full')
    ax1.legend(
        fontsize=7, ncol=3, columnspacing=1.0, handlelength=1.5, loc='upper left', framealpha=0.9
    )
    ax1.set_ylim(bottom=5)
    _grid(ax1)

    fig1.tight_layout()
    fig1.savefig(FIGURES_DIR / 'paper_fig1.pdf', bbox_inches='tight')
    fig1.savefig(FIGURES_DIR / 'paper_fig1.png', bbox_inches='tight', dpi=300)
    plt.close(fig1)
    print(f'\nSaved: {FIGURES_DIR / "paper_fig1.pdf"}')

    # === Figure 2: Forward timing (2 panels) ============================
    fig2, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(5.5, 2.4))

    # Panel (a): CnonCn forward selective vs full (GPU)
    _panel_label(ax_b, 'a')
    cn_sel_ns = [8, 16, 32, 64, 128, 256, 512, 1024]
    cn_full_ns = [8, 16, 32, 64, 128, 256]

    cn_sel_times: list[float] = []
    for n in cn_sel_ns:
        bsp = CnonCn(n=n, selective=True).to(device)
        f = torch.randn(batch_fwd, n, device=device)
        cn_sel_times.append(_time_fn('bsp(f)', {'bsp': bsp, 'f': f}, f'cn_sel_{n}', device))
        print(f'  CnonCn sel n={n}: {cn_sel_times[-1]:.3f} ms')

    cn_full_times: list[float] = []
    for n in cn_full_ns:
        bsp = CnonCn(n=n, selective=False).to(device)
        f = torch.randn(batch_fwd, n, device=device)
        cn_full_times.append(_time_fn('bsp(f)', {'bsp': bsp, 'f': f}, f'cn_full_{n}', device))
        print(f'  CnonCn full n={n}: {cn_full_times[-1]:.3f} ms')

    ax_b.plot(cn_sel_ns, cn_sel_times, marker='o', color=c[4], label='selective')
    ax_b.plot(cn_full_ns, cn_full_times, marker='s', color=c[5], linestyle='--', label='full')
    ax_b.set_xscale('log')
    ax_b.set_yscale('log')
    ax_b.set_xlabel(r'$n$')
    ax_b.set_ylabel('time (ms)')
    ax_b.set_title(r'Forward pass: $C_n$ on $C_n$')
    ax_b.legend(fontsize=7.5)
    _grid(ax_b)

    # Panel (b): TorusOnTorus forward selective vs full (GPU)
    _panel_label(ax_c, 'b')
    tor_sel_ns = [8, 16, 32, 64, 128, 256]
    tor_full_ns = [8, 16, 32]

    tor_sel_gs: list[int] = []
    tor_sel_times: list[float] = []
    for n in tor_sel_ns:
        bsp = TorusOnTorus(ns=(n, n), selective=True).to(device)
        f = torch.randn(batch_fwd, n, n, device=device)
        g = n * n
        tor_sel_gs.append(g)
        tor_sel_times.append(_time_fn('bsp(f)', {'bsp': bsp, 'f': f}, f'tor_sel_{n}', device))
        print(f'  Torus sel n={n} |G|={g}: {tor_sel_times[-1]:.3f} ms')

    tor_full_gs: list[int] = []
    tor_full_times: list[float] = []
    for n in tor_full_ns:
        bsp = TorusOnTorus(ns=(n, n), selective=False).to(device)
        f = torch.randn(batch_fwd, n, n, device=device)
        g = n * n
        tor_full_gs.append(g)
        tor_full_times.append(_time_fn('bsp(f)', {'bsp': bsp, 'f': f}, f'tor_full_{n}', device))
        print(f'  Torus full n={n} |G|={g}: {tor_full_times[-1]:.3f} ms')

    ax_c.plot(tor_sel_gs, tor_sel_times, marker='o', color=c[4], label='selective')
    ax_c.plot(tor_full_gs, tor_full_times, marker='s', color=c[5], linestyle='--', label='full')
    ax_c.set_xscale('log')
    ax_c.set_xlabel(r'$|G| = n^2$')
    ax_c.set_ylabel('time (ms)')
    ax_c.set_title(r'Forward pass: $\mathbb{T}^2$')
    ax_c.legend(fontsize=7.5)
    _grid(ax_c)

    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / 'paper_fig2.pdf', bbox_inches='tight')
    fig2.savefig(FIGURES_DIR / 'paper_fig2.png', bbox_inches='tight', dpi=300)
    plt.close(fig2)
    print(f'\nSaved: {FIGURES_DIR / "paper_fig2.pdf"}')

    # === Figure 3: 2 panels ============================================
    fig3, (ax_d, ax_e) = plt.subplots(1, 2, figsize=(5.5, 2.4))

    # Panel (a): GPU batch scaling (finite groups)
    _panel_label(ax_d, 'a')
    batch_sizes = [1, 4, 16, 64, 256, 1024, 4096]
    gpu_configs: list[tuple[str, str, dict[str, Any], str, str]] = [
        (r'$C_n$', 'CnonCn', {'n': 128}, c[4], 'o'),
        (r'$\mathbb{T}^2$', 'TorusOnTorus', {'ns': (32, 32)}, c[0], 's'),
        (r'$D_n$', 'DnonDn', {'n': 32}, c[2], '^'),
        (r'$O$', 'OctaonOcta', {}, c[5], 'D'),
    ]

    for label, mod_name, params, color, marker in gpu_configs:
        bsp = _build_module(mod_name, params, selective=True).to(device)
        g = _group_order(mod_name, params)
        throughputs: list[float] = []
        valid_batches: list[int] = []
        for bs in batch_sizes:
            try:
                f = _make_input(mod_name, params, bs, device)
                t = _time_fn('bsp(f)', {'bsp': bsp, 'f': f}, f'{mod_name}_b{bs}', device)
            except RuntimeError:
                break
            throughputs.append(bs / (t / 1e3))
            valid_batches.append(bs)
        ax_d.plot(
            valid_batches, throughputs, marker=marker, color=color, label=rf'{label} ($|G|$={g})'
        )
        print(f'  GPU scaling {mod_name}: max throughput = {max(throughputs):.0f} samples/s')

    ax_d.set_xscale('log')
    ax_d.set_yscale('log')
    ax_d.set_xlabel('batch size')
    ax_d.set_ylabel('throughput (samples/s)')
    ax_d.set_title('GPU batch scaling')
    ax_d.legend(fontsize=6.5, loc='upper left')
    _grid(ax_d)

    # Panel (b): Inversion timing overlay (all modules with inversion)
    _panel_label(ax_e, 'b')
    inv_configs: list[tuple[str, str, list[dict[str, Any]], str, str]] = [
        (r'$C_n$', 'CnonCn', [{'n': n} for n in [8, 16, 32, 64, 128, 256, 512, 1024]], c[4], 'o'),
        (
            r'$\mathbb{T}^2$',
            'TorusOnTorus',
            [{'ns': (n, n)} for n in [8, 16, 32, 64, 128, 256]],
            c[0],
            's',
        ),
        (r'$D_n$', 'DnonDn', [{'n': n} for n in [4, 8, 16, 32, 64, 128]], c[2], '^'),
        (r'$\mathrm{SO}(2)/D^2$', 'SO2onD2', [{'L': L} for L in _SO2OND2_LS], c[6], 'v'),
    ]

    for label, mod_name, params_list, color, marker in inv_configs:
        gs: list[int] = []
        inv_times: list[float] = []
        for params in params_list:
            bsp = _build_module(mod_name, params, selective=True).to(device)
            f = _make_input(mod_name, params, batch_inv, device)
            g = _group_order(mod_name, params)
            with torch.no_grad():
                beta = bsp(f)
            t = _time_fn(
                'bsp.invert(beta)', {'bsp': bsp, 'beta': beta}, f'{mod_name}_inv_{g}', device
            )
            gs.append(g)
            inv_times.append(t)
        ax_e.plot(gs, inv_times, marker=marker, color=color, label=label)
        print(f'  Inversion {mod_name}: {gs[-1]} -> {inv_times[-1]:.2f} ms')

    # OctaonOcta single point
    bsp_o = OctaonOcta(selective=True).to(device)
    f_o = torch.randn(batch_inv, 24, device=device)
    with torch.no_grad():
        beta_o = bsp_o(f_o)
    t_o = _time_fn('bsp.invert(beta)', {'bsp': bsp_o, 'beta': beta_o}, 'octa_inv', device)
    ax_e.scatter([24], [t_o], marker='*', s=50, color=c[5], zorder=5, label=rf'$O$ ({t_o:.0f} ms)')
    print(f'  Inversion OctaonOcta: 24 -> {t_o:.2f} ms')

    ax_e.set_xscale('log')
    ax_e.set_yscale('log')
    ax_e.set_xlabel(r'$|G|$')
    ax_e.set_ylabel('inversion time (ms)')
    ax_e.set_title('Selective inversion')
    ax_e.legend(fontsize=6, loc='lower right')
    _grid(ax_e)

    fig3.tight_layout()
    fig3.savefig(FIGURES_DIR / 'paper_fig3.pdf', bbox_inches='tight')
    fig3.savefig(FIGURES_DIR / 'paper_fig3.png', bbox_inches='tight', dpi=300)
    plt.close(fig3)
    print(f'\nSaved: {FIGURES_DIR / "paper_fig3.pdf"}')

    # === LaTeX table ===================================================
    print('\nGenerating LaTeX table...')
    table_configs: list[tuple[str, str, str, dict[str, Any], bool, bool]] = [
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
        (r'\texttt{SO2onD2}', r'$\mathrm{SO}(2)$', 'SO2onD2', {'L': 32}, True, False),
        (r'\texttt{SO3onS2}', r'$\mathrm{SO}(3)$', 'SO3onS2', _so3_params(5), False, False),
        (r'\texttt{OctaonOcta}', r'$O$', 'OctaonOcta', {}, True, False),
    ]

    rows: list[str] = []
    for tex_name, group_tex, mod_name, params, has_inv, has_full in table_configs:
        g = _group_order(mod_name, params)
        bsp_sel = _build_module(mod_name, params, selective=True).to(device)
        coefs_sel = bsp_sel.output_size

        if has_full:
            coefs_full = str(g * (g + 1) // 2)
        else:
            coefs_full = '--'

        f = _make_input(mod_name, params, batch_fwd, device)
        t_fwd_sel = _time_fn('bsp(f)', {'bsp': bsp_sel, 'f': f}, f'tab_{mod_name}_sel', device)

        if has_full:
            try:
                bsp_full = _build_module(mod_name, params, selective=False).to(device)
                f2 = _make_input(mod_name, params, batch_fwd, device)
                t_fwd_full = f'{_time_fn("bsp(f)", {"bsp": bsp_full, "f": f2}, f"tab_{mod_name}_full", device):.2f}'
            except NotImplementedError:
                t_fwd_full = '--'
        else:
            t_fwd_full = '--'

        if has_inv:
            with torch.no_grad():
                beta = bsp_sel(f)
            t_inv = f'{_time_fn("bsp.invert(beta)", {"bsp": bsp_sel, "beta": beta}, f"tab_{mod_name}_inv", device):.2f}'
        else:
            t_inv = '--'

        rows.append(
            f'        {tex_name} & {group_tex} & {g} & {coefs_sel} & {coefs_full} '
            f'& {t_fwd_sel:.2f} & {t_fwd_full} & {t_inv} \\\\'
        )

    table_tex = r"""\begin{table}[t]
    \centering
    \caption{Benchmark summary for the \texttt{bispectrum} library.
    Coefficient counts compare the selective ($O(|G|)$) and full ($O(|G|^2)$) bispectra.
    Timings are median wall-clock on a single NVIDIA A100 GPU (batch\,=\,16 for forward, 4 for inversion).
    ``--'' indicates the mode is not implemented.}
    \label{tab:benchmarks}
    \small
    \begin{tabular}{llrrrrrr}
        \toprule
        Module & $G$ & $|G|$ & \multicolumn{2}{c}{Coefs} & \multicolumn{2}{c}{Fwd (ms)} & Inv (ms) \\
        \cmidrule(lr){4-5} \cmidrule(lr){6-7}
         & & & Sel. & Full & Sel. & Full & Sel. \\
        \midrule
"""
    table_tex += '\n'.join(rows)
    table_tex += r"""
        \bottomrule
    \end{tabular}
\end{table}
"""

    table_path = FIGURES_DIR / 'table_benchmarks.tex'
    table_path.write_text(table_tex)
    print(f'Saved: {table_path}')
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
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    has_cuda = torch.cuda.is_available() and not args.cpu_only
    cpu = torch.device('cpu')
    gpu = torch.device('cuda') if has_cuda else None

    print(f'PyTorch {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if has_cuda:
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    if args.paper:
        if not has_cuda:
            print('ERROR: --paper requires CUDA. Remove --cpu-only or use a GPU machine.')
            return
        paper_figures(gpu)  # type: ignore[arg-type]
        print('\nDone.')
        return

    print(f'Running benchmarks: {args.bench}')

    if 1 in args.bench:
        fig = bench_coefficient_count()
        fig.savefig(FIGURES_DIR / 'bench1_coefficient_count.pdf', bbox_inches='tight')
        fig.savefig(FIGURES_DIR / 'bench1_coefficient_count.png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f'\nSaved: {FIGURES_DIR / "bench1_coefficient_count.pdf"}')

    if 2 in args.bench:
        fig_cpu = bench_forward_pass(cpu)
        fig_cpu.savefig(FIGURES_DIR / 'bench2_forward_cpu.pdf', bbox_inches='tight')
        fig_cpu.savefig(FIGURES_DIR / 'bench2_forward_cpu.png', bbox_inches='tight', dpi=150)
        plt.close(fig_cpu)
        print(f'\nSaved: {FIGURES_DIR / "bench2_forward_cpu.pdf"}')

        if has_cuda:
            fig_gpu = bench_forward_pass(gpu)  # type: ignore[arg-type]
            fig_gpu.savefig(FIGURES_DIR / 'bench2_forward_gpu.pdf', bbox_inches='tight')
            fig_gpu.savefig(FIGURES_DIR / 'bench2_forward_gpu.png', bbox_inches='tight', dpi=150)
            plt.close(fig_gpu)
            print(f'\nSaved: {FIGURES_DIR / "bench2_forward_gpu.pdf"}')

    if 3 in args.bench:
        fig_cpu = bench_inversion(cpu)
        fig_cpu.savefig(FIGURES_DIR / 'bench3_inversion_cpu.pdf', bbox_inches='tight')
        fig_cpu.savefig(FIGURES_DIR / 'bench3_inversion_cpu.png', bbox_inches='tight', dpi=150)
        plt.close(fig_cpu)
        print(f'\nSaved: {FIGURES_DIR / "bench3_inversion_cpu.pdf"}')

        if has_cuda:
            fig_gpu = bench_inversion(gpu)  # type: ignore[arg-type]
            fig_gpu.savefig(FIGURES_DIR / 'bench3_inversion_gpu.pdf', bbox_inches='tight')
            fig_gpu.savefig(FIGURES_DIR / 'bench3_inversion_gpu.png', bbox_inches='tight', dpi=150)
            plt.close(fig_gpu)
            print(f'\nSaved: {FIGURES_DIR / "bench3_inversion_gpu.pdf"}')

    if 4 in args.bench:
        if has_cuda:
            fig = bench_gpu_scaling()
            fig.savefig(FIGURES_DIR / 'bench4_gpu_scaling.pdf', bbox_inches='tight')
            fig.savefig(FIGURES_DIR / 'bench4_gpu_scaling.png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f'\nSaved: {FIGURES_DIR / "bench4_gpu_scaling.pdf"}')
        else:
            print('\nBenchmark 4 (GPU scaling) skipped: no CUDA device available.')

    print('\nDone.')


if __name__ == '__main__':
    main()
