"""Per-degree real-Jacobian rank check for the SO3onS2 selective invariant.

Generic local completeness of the augmented selective bispectrum follows from
a block-triangular argument: the entries of degree ``ell`` (bispectral triples
whose largest index is ``ell`` and CG power entries ``(l1, l2, l_out)`` with
``max(l1, l2) == ell``) depend only on ``F_0 .. F_ell``.  If, at a generic
point, the real Jacobian of the degree-``ell`` entries with respect to the
``2 ell + 1`` real degrees of freedom of ``F_ell`` has full rank for every
``ell >= 4``, and the degree ``<= 3`` seed block has rank ``16 - 3 = 13``
(dimension minus ``dim SO(3)``), then the whole invariant has real Jacobian
rank ``(L+1)^2 - 3`` for every band-limit ``L``.

This script evaluates each degree block at a seeded random witness using the
same sparse CG coefficients as the production forward pass, and reports the
numerical rank and the condition number of the block.  A full-rank witness
proves that the generic rank is full (the rank-deficiency locus is a proper
algebraic subvariety); a deficient witness is reported as FAIL.

Real signals are used throughout (``a_l^{-m} = (-1)^m conj(a_l^m)``), so every
bispectral entry is a single real number: its real part for even
``l1 + l2 + l``, its imaginary part for odd parity.  An earlier version of this
script stacked chain rows (linear in ``conj(a_ell)``) and cross rows (linear
in ``a_ell``) into one complex matrix, which does not measure the real rank and
missed that the two forms of a triple are the same real equation.

Usage:
    python benchmarks/verify_linear_bootstrap.py [--lmax 100] [--ell-min 4] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable

import torch

from bispectrum._cg import compute_sparse_cg_parallel
from bispectrum.so3_on_s2 import _build_cg_power_index_map, _build_selective_index_map

SparseCG = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _real_block_to_complex(blk: torch.Tensor, l_val: int) -> torch.Tensor:
    """Real dof ``[a^0, Re a^1, Im a^1, ..., Re a^l, Im a^l]`` -> ``a^m`` for m = -l..l."""
    coeffs: list[torch.Tensor] = [torch.zeros((), dtype=torch.complex128)] * (2 * l_val + 1)
    coeffs[l_val] = torch.complex(blk[0], torch.zeros_like(blk[0]))
    for m in range(1, l_val + 1):
        am = torch.complex(blk[2 * m - 1], blk[2 * m])
        coeffs[l_val + m] = am
        coeffs[l_val - m] = ((-1.0) ** m) * torch.conj(am)
    return torch.stack(coeffs)


def _degree_entries(ell: int) -> list[tuple[int, int, int, bool]]:
    """Selective entries of degree ``ell`` (built with lmax = ell, the smallest set this degree
    ever receives).

    Returns ``(l1, l2, l3, is_power)``.
    """
    entries: list[tuple[int, int, int, bool]] = []
    for l1, l2, l_val in _build_selective_index_map(ell):
        if max(l1, l2, l_val) == ell:
            entries.append((l1, l2, l_val, False))
    for l1, l2, l_out in _build_cg_power_index_map(ell):
        if max(l1, l2) == ell:
            entries.append((l1, l2, l_out, True))
    return entries


def _sparse_cg_tables(entries: list[tuple[int, int, int, bool]]) -> list[SparseCG]:
    """(m1_idx, m2_idx, m_idx, cg) per entry, with m2 resolved from the selection rule."""
    tasks = [(i, l1, l2, l3, is_power) for i, (l1, l2, l3, is_power) in enumerate(entries)]
    tables: list[SparseCG] = []
    for (l1, l2, l3, _), (m1_idx, m_idx, cg) in zip(
        entries, compute_sparse_cg_parallel(tasks), strict=True
    ):
        m1_t = torch.as_tensor(m1_idx, dtype=torch.long)
        m_t = torch.as_tensor(m_idx, dtype=torch.long)
        m2_t = m_t - l3 - m1_t + l1 + l2
        tables.append((m1_t, m2_t, m_t, torch.as_tensor(cg, dtype=torch.float64)))
    return tables


def _make_block_fn(
    ell: int,
    entries: list[tuple[int, int, int, bool]],
    tables: list[SparseCG],
    lower: dict[int, torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Real-valued degree-``ell`` entries as a function of the ``2 ell + 1`` real dof of F_ell."""

    def block(y: torch.Tensor) -> torch.Tensor:
        coeffs = dict(lower)
        coeffs[ell] = _real_block_to_complex(y, ell)
        out: list[torch.Tensor] = []
        for (l1, l2, l3, is_power), (m1_t, m2_t, m_t, cg) in zip(entries, tables, strict=True):
            prods = coeffs[l1][m1_t] * coeffs[l2][m2_t] * cg
            if is_power:
                coupled = torch.zeros(2 * l3 + 1, dtype=torch.complex128)
                coupled = coupled.index_add(0, m_t, prods)
                out.append((coupled.real**2 + coupled.imag**2).sum())
            else:
                val = (prods * torch.conj(coeffs[l3][m_t])).sum()
                out.append(val.real if (l1 + l2 + l3) % 2 == 0 else val.imag)
        return torch.stack(out)

    return block


def _rank_and_cond(J: torch.Tensor, target: int, rtol: float = 1e-9) -> tuple[int, float, float]:
    sv = torch.linalg.svdvals(J)
    rank = int((sv > rtol * sv[0]).sum())
    sigma_target = sv[target - 1].item() if sv.numel() >= target else 0.0
    cond = sv[0].item() / sigma_target if sigma_target > 0 else float('inf')
    return rank, sigma_target, cond


def _witness(lmax: int, seed: int) -> dict[int, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    coeffs: dict[int, torch.Tensor] = {}
    for l_val in range(lmax + 1):
        blk = torch.randn(2 * l_val + 1, generator=gen, dtype=torch.float64)
        coeffs[l_val] = _real_block_to_complex(blk, l_val)
    return coeffs


def verify_seed_block(seed: int) -> bool:
    """Degrees 0..3 jointly: rank must be 16 - 3 (the SO(3) gauge directions)."""
    entries: list[tuple[int, int, int, bool]] = []
    for ell in range(4):
        entries.extend(_degree_entries(ell))
    tables = _sparse_cg_tables(entries)
    gen = torch.Generator().manual_seed(seed)
    x0 = torch.randn(16, generator=gen, dtype=torch.float64)

    def seed_fn(x: torch.Tensor) -> torch.Tensor:
        coeffs: dict[int, torch.Tensor] = {}
        pos = 0
        for l_val in range(4):
            coeffs[l_val] = _real_block_to_complex(x[pos : pos + 2 * l_val + 1], l_val)
            pos += 2 * l_val + 1
        out: list[torch.Tensor] = []
        for (l1, l2, l3, is_power), (m1_t, m2_t, m_t, cg) in zip(entries, tables, strict=True):
            prods = coeffs[l1][m1_t] * coeffs[l2][m2_t] * cg
            if is_power:
                coupled = torch.zeros(2 * l3 + 1, dtype=torch.complex128).index_add(0, m_t, prods)
                out.append((coupled.real**2 + coupled.imag**2).sum())
            else:
                val = (prods * torch.conj(coeffs[l3][m_t])).sum()
                out.append(val.real if (l1 + l2 + l3) % 2 == 0 else val.imag)
        return torch.stack(out)

    J = torch.autograd.functional.jacobian(seed_fn, x0)
    rank, sigma, cond = _rank_and_cond(J, 13)
    status = 'OK' if rank == 13 else 'FAIL'
    print(
        f'  seed (ell<=3): {len(entries)} entries, rank={rank:3d}/ 13  '
        f'cond={cond:.1e}  sigma_13={sigma:.2e}  {status}',
        flush=True,
    )
    return rank == 13


def verify_degree(ell: int, witness: dict[int, torch.Tensor], seed: int) -> bool:
    t0 = time.time()
    target = 2 * ell + 1
    entries = _degree_entries(ell)
    tables = _sparse_cg_tables(entries)
    lower = {l_val: witness[l_val] for l_val in range(ell)}
    block = _make_block_fn(ell, entries, tables, lower)
    gen = torch.Generator().manual_seed(seed + ell)
    y0 = torch.randn(target, generator=gen, dtype=torch.float64)

    J = torch.autograd.functional.jacobian(block, y0)
    rank, sigma, cond = _rank_and_cond(J, target)
    status = 'OK' if rank == target else 'FAIL'
    n_lin = sum(1 for l1, l2, l3, p in entries if not p and (l1, l2, l3).count(ell) == 1)
    print(
        f'  ell={ell:3d}: {len(entries):3d} entries ({n_lin} linear)  '
        f'rank={rank:3d}/{target:3d}  cond={cond:.1e}  sigma_min={sigma:.2e}  '
        f'{time.time() - t0:.2f}s  {status}',
        flush=True,
    )
    return rank == target


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Per-degree real-Jacobian rank check of the SO3onS2 selective invariant'
    )
    parser.add_argument('--lmax', type=int, default=100, help='Largest degree to verify')
    parser.add_argument('--ell-min', type=int, default=4, help='First degree to verify')
    parser.add_argument('--seed', type=int, default=42, help='Witness RNG seed')
    args = parser.parse_args()

    print(f'Per-degree real Jacobian rank, ell = {args.ell_min} .. {args.lmax} (seed {args.seed})')
    t_start = time.time()
    ok = verify_seed_block(args.seed)
    witness = _witness(args.lmax, args.seed)
    for ell in range(max(args.ell_min, 4), args.lmax + 1):
        ok &= verify_degree(ell, witness, args.seed)
    elapsed = time.time() - t_start

    print()
    if ok:
        print(f'ALL PASSED (seed block and ell = {args.ell_min} .. {args.lmax}) in {elapsed:.1f}s')
    else:
        print('SOME FAILURES DETECTED')
        sys.exit(1)


if __name__ == '__main__':
    main()
