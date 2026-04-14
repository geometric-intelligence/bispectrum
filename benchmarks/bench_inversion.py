#!/usr/bin/env python3
"""Octahedral inversion benchmarks: success rate, convergence, timing.

Generates:
  - LaTeX table with success rates, residual stats, and timing
  - Convergence figure showing residual vs. LM step
  - Console summary

Usage:
    python bench_inversion.py [--n_signals 1000] [--device cuda]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from bispectrum import OctaonOcta

NEURIPS_RCPARAMS = {
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'text.usetex': False,
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth': 0.5,
}


def recovery_error_up_to_group_action(
    f_orig: torch.Tensor,
    f_recovered: torch.Tensor,
) -> torch.Tensor:
    """Min L2 error over 24 cyclic shifts (group actions on the signal vector).

    For each sample, tries all 24 cyclic permutations and returns the minimum relative error.
    """
    batch = f_orig.shape[0]
    best_err = torch.full((batch,), float('inf'), device=f_orig.device)

    for shift in range(24):
        f_shifted = torch.roll(f_recovered, shifts=shift, dims=-1)
        err = (f_orig - f_shifted).norm(dim=-1) / f_orig.norm(dim=-1).clamp(min=1e-12)
        best_err = torch.minimum(best_err, err)

    return best_err


def invert_with_convergence(
    bsp: OctaonOcta,
    beta: torch.Tensor,
    n_corrections: int = 10,
    n_restarts: int = 4,
) -> tuple[torch.Tensor, list[list[float]]]:
    """Like bsp.invert() but also logs per-step residuals for convergence plotting.

    Returns (recovered_signal, residual_traces) where residual_traces[trial] is a list of median
    residuals after each LM step.
    """
    target_real = beta.real
    f_init = bsp._bootstrap_init(beta)
    fhat_init = bsp._group_dft(f_init.to(torch.float64))

    best_f = f_init.to(beta.real.dtype)
    best_loss = torch.full(
        (beta.shape[0],),
        float('inf'),
        dtype=beta.real.dtype,
        device=beta.device,
    )

    all_traces: list[list[float]] = []

    for trial in range(n_restarts):
        if trial == 0:
            f = f_init.to(beta.real.dtype)
        else:
            fhat_pert = [fk.clone() for fk in fhat_init]
            for k in range(bsp.N_IRREPS):
                dk = bsp._irrep_dims[k]
                if dk > 1:
                    Q = torch.linalg.qr(
                        torch.randn(
                            beta.shape[0],
                            dk,
                            dk,
                            dtype=torch.float64,
                            device=beta.device,
                        ),
                    ).Q
                    fhat_pert[k] = Q @ fhat_pert[k]
            f = bsp._inverse_dft(fhat_pert).to(beta.real.dtype)

        trace: list[float] = []
        with torch.no_grad():
            r0 = (bsp.forward(f).real - target_real).norm(dim=-1)
            trace.append(float(r0.median()))

        for _ in range(n_corrections):
            f = bsp._lm_step(f, beta)
            with torch.no_grad():
                res = (bsp.forward(f).real - target_real).norm(dim=-1)
                trace.append(float(res.median()))

        all_traces.append(trace)

        with torch.no_grad():
            residuals = (bsp.forward(f).real - target_real).norm(dim=-1)
            improved = residuals < best_loss
            if improved.any():
                best_f[improved] = f[improved]
                best_loss[improved] = residuals[improved]

    return best_f, all_traces


def run_benchmark(
    n_signals: int,
    device: str,
    n_corrections: int = 10,
    n_restarts: int = 4,
    batch_size: int = 64,
    output_dir: str = '.',
) -> dict:
    """Run the full inversion benchmark."""
    bsp = OctaonOcta(selective=True).to(device)

    torch.manual_seed(42)
    all_signals = torch.randn(n_signals, 24, device=device)

    all_recovery_errors: list[float] = []
    all_residual_norms: list[float] = []
    all_times: list[float] = []
    convergence_traces: list[list[float]] = []

    n_batches = (n_signals + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_signals)
        f_batch = all_signals[start:end]

        with torch.no_grad():
            beta = bsp(f_batch)

        t0 = time.perf_counter()
        f_rec, traces = invert_with_convergence(
            bsp,
            beta,
            n_corrections=n_corrections,
            n_restarts=n_restarts,
        )
        t1 = time.perf_counter()

        with torch.no_grad():
            residuals = (bsp(f_rec).real - beta.real).norm(dim=-1)
            rec_errors = recovery_error_up_to_group_action(f_batch, f_rec)

        all_residual_norms.extend(residuals.cpu().tolist())
        all_recovery_errors.extend(rec_errors.cpu().tolist())
        all_times.append((t1 - t0) / (end - start))

        if i == 0:
            convergence_traces = traces

    residuals_arr = np.array(all_residual_norms)
    errors_arr = np.array(all_recovery_errors)
    times_arr = np.array(all_times)

    thresholds = [1e-1, 1e-2, 1e-3]
    success_rates = {f'<{t}': float((errors_arr < t).mean()) for t in thresholds}

    stats = {
        'n_signals': n_signals,
        'n_corrections': n_corrections,
        'n_restarts': n_restarts,
        'success_rates': success_rates,
        'residual_median': float(np.median(residuals_arr)),
        'residual_p95': float(np.percentile(residuals_arr, 95)),
        'residual_max': float(np.max(residuals_arr)),
        'recovery_error_median': float(np.median(errors_arr)),
        'recovery_error_p95': float(np.percentile(errors_arr, 95)),
        'recovery_error_max': float(np.max(errors_arr)),
        'time_per_signal_ms': float(np.median(times_arr) * 1000),
        'time_per_signal_p95_ms': float(np.percentile(times_arr, 95) * 1000),
    }

    print('\n' + '=' * 70)
    print(f'OCTAHEDRAL INVERSION BENCHMARK (N={n_signals}, device={device})')
    print(f'  n_corrections={n_corrections}, n_restarts={n_restarts}')
    print('=' * 70)
    print('\n  Success rates (recovery error up to group action):')
    for k, v in success_rates.items():
        print(f'    {k}: {v:.1%}')
    print('\n  Bispectrum residual ||beta(f_rec) - beta_target||:')
    print(f'    median: {stats["residual_median"]:.2e}')
    print(f'    p95:    {stats["residual_p95"]:.2e}')
    print(f'    max:    {stats["residual_max"]:.2e}')
    print('\n  Signal recovery error (min over group action):')
    print(f'    median: {stats["recovery_error_median"]:.2e}')
    print(f'    p95:    {stats["recovery_error_p95"]:.2e}')
    print(f'    max:    {stats["recovery_error_max"]:.2e}')
    print('\n  Timing (per signal):')
    print(f'    median: {stats["time_per_signal_ms"]:.1f} ms')
    print(f'    p95:    {stats["time_per_signal_p95_ms"]:.1f} ms')

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_latex_table(stats, out / 'inversion_stats.tex')
    _plot_convergence(convergence_traces, out / 'inversion_convergence.pdf')
    _plot_error_histogram(errors_arr, out / 'inversion_errors.pdf')

    return stats


def _write_latex_table(stats: dict, path: Path) -> None:
    """Write a compact LaTeX table summarizing inversion statistics."""
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Octahedral bispectrum inversion statistics '
        rf'(N={stats["n_signals"]}, '
        rf'{stats["n_restarts"]} restarts $\times$ {stats["n_corrections"]} LM steps).}}',
        r'\label{tab:inversion-stats}',
        r'\begin{tabular}{lcc}',
        r'\toprule',
        r'Metric & Median & P95 \\',
        r'\midrule',
    ]

    lines.append(
        rf'Bisp.\ residual $\|\beta(\hat{{f}}) - \beta^*\|$ '
        rf'& {stats["residual_median"]:.1e} & {stats["residual_p95"]:.1e} \\'
    )
    lines.append(
        rf'Recovery error (mod $O$) '
        rf'& {stats["recovery_error_median"]:.1e} & {stats["recovery_error_p95"]:.1e} \\'
    )
    lines.append(
        rf'Wall-clock time (ms) '
        rf'& {stats["time_per_signal_ms"]:.1f} & {stats["time_per_signal_p95_ms"]:.1f} \\'
    )
    lines.append(r'\midrule')

    for thresh_str, rate in stats['success_rates'].items():
        lines.append(
            rf'Success rate (error {thresh_str}) & \multicolumn{{2}}{{c}}{{{rate:.1%}}} \\'
        )

    lines.extend(
        [
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}',
        ]
    )

    path.write_text('\n'.join(lines) + '\n')
    print(f'\nLaTeX table written to {path}')


def _plot_convergence(traces: list[list[float]], path: Path) -> None:
    """Plot median residual vs.

    LM step for each restart.
    """
    plt.rcParams.update(NEURIPS_RCPARAMS)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    colors = ['#2D6A9F', '#D4722A', '#4CAF50', '#9C27B0', '#E53935']

    for i, trace in enumerate(traces):
        steps = list(range(len(trace)))
        color = colors[i % len(colors)]
        label = 'Bootstrap' if i == 0 else f'Restart {i}'
        ax.plot(
            steps,
            trace,
            marker='o',
            markersize=3,
            linewidth=1.0,
            color=color,
            label=label,
            zorder=3,
        )

    ax.set_xlabel('LM step')
    ax.set_ylabel('Median residual')
    ax.set_yscale('log')
    ax.set_xticks(range(len(traces[0])))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6, loc='upper right')

    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(str(path).replace('.pdf', '.png'), dpi=300)
    print(f'Convergence plot saved to {path}')
    plt.close(fig)


def _plot_error_histogram(errors: np.ndarray, path: Path) -> None:
    """Histogram of recovery errors (log scale)."""
    plt.rcParams.update(NEURIPS_RCPARAMS)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    log_errors = np.log10(errors.clip(min=1e-15))
    ax.hist(log_errors, bins=50, color='#2D6A9F', edgecolor='white', linewidth=0.3, zorder=3)
    ax.set_xlabel('log10(recovery error)')
    ax.set_ylabel('Count')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    for thresh in [-1, -2, -3]:
        pct = (log_errors < thresh).mean()
        ax.axvline(thresh, color='#D4722A', linewidth=0.6, linestyle='--', zorder=4)
        ax.text(
            thresh + 0.1,
            ax.get_ylim()[1] * 0.9,
            f'{pct:.0%}',
            fontsize=6,
            color='#D4722A',
            va='top',
        )

    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(str(path).replace('.pdf', '.png'), dpi=300)
    print(f'Error histogram saved to {path}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Octahedral inversion benchmark')
    parser.add_argument('--n_signals', type=int, default=1000)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--n_corrections', type=int, default=10)
    parser.add_argument('--n_restarts', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='./inversion_results')
    args = parser.parse_args()

    run_benchmark(
        n_signals=args.n_signals,
        device=args.device,
        n_corrections=args.n_corrections,
        n_restarts=args.n_restarts,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
