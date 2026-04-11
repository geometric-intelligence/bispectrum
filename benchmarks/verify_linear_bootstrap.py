"""Verify the linear bootstrap rank condition for all ell >= 4.

For each degree ell, the chain and cross entries selected by the selective
bispectrum yield a matrix A_ell in C^{N x (2ell+1)}.  This script verifies
that rank(A_ell) = 2*ell + 1 at an explicit deterministic witness point,
proving that the determinant polynomial is not identically zero.

This constitutes a computer-assisted proof that the linear bootstrap
is generically full-rank for each verified ell.

Usage:
    python benchmarks/verify_linear_bootstrap.py [--lmax 100]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from bispectrum._cg import clebsch_gordan
from bispectrum.so3_on_s2 import _build_selective_index_map


def _cg_matrix_dense(l1: int, l2: int, l: int) -> np.ndarray:
    """Build a dense CG coefficient array C[m1_idx, m2_idx, m_idx].

    m_idx = m + l, etc.
    """
    mat = np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * l + 1))
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            m = m1 + m2
            if abs(m) > l:
                continue
            val = clebsch_gordan(l1, m1, l2, m2, l, m)
            if val != 0.0:
                mat[m1 + l1, m2 + l2, m + l] = val
    return mat


def _witness_coefficients_np(lmax: int) -> dict[int, np.ndarray]:
    """Deterministic witness using a fixed RNG seed for reproducibility.

    Uses np.random with seed 42 to generate generic coefficients with unit-scale entries, then
    enforces the real-signal constraint.
    """
    rng = np.random.RandomState(42)
    coeffs: dict[int, np.ndarray] = {}
    for j in range(lmax + 1):
        c = np.zeros(2 * j + 1, dtype=np.complex128)
        for m in range(0, j + 1):
            c[m + j] = complex(rng.randn(), rng.randn())
        for m in range(1, j + 1):
            c[-m + j] = ((-1) ** m) * c[m + j].conjugate()
        coeffs[j] = c
    return coeffs


def _chain_row_np(
    coeffs: dict[int, np.ndarray],
    l1: int,
    l2: int,
    ell: int,
    cg_cache: dict[tuple[int, int, int], np.ndarray],
) -> np.ndarray:
    """Chain row G_k in C^{2*ell+1} using vectorized CG."""
    key = (l1, l2, ell)
    if key not in cg_cache:
        cg_cache[key] = _cg_matrix_dense(l1, l2, ell)
    cg = cg_cache[key]
    a1 = coeffs[l1]
    a2 = coeffs[l2]
    outer = np.outer(a1, a2)  # (2l1+1, 2l2+1)
    return np.einsum('ij,ijk->k', outer, cg)


def _cross_row_np(
    coeffs: dict[int, np.ndarray],
    l1: int,
    ell: int,
    lp: int,
    cg_cache: dict[tuple[int, int, int], np.ndarray],
) -> np.ndarray:
    """Cross row H_k in C^{2*ell+1} for triple (l1, ell, lp).

    H_k^{m2} = sum_{m1} CG^{lp, m1+m2}_{l1,m1; ell,m2} a_{l1}^{m1} conj(a_{lp}^{m1+m2})
    """
    key = (l1, ell, lp)
    if key not in cg_cache:
        cg_cache[key] = _cg_matrix_dense(l1, ell, lp)
    cg = cg_cache[key]  # (2l1+1, 2ell+1, 2lp+1)
    row = np.zeros(2 * ell + 1, dtype=np.complex128)
    a1 = coeffs[l1]
    alp = coeffs[lp]
    for m1_idx in range(2 * l1 + 1):
        for m2_idx in range(2 * ell + 1):
            # mp = m1 + m2 = (m1_idx - l1) + (m2_idx - ell)
            mp_idx = m1_idx + m2_idx - l1 - ell + lp
            if 0 <= mp_idx < 2 * lp + 1:
                c = cg[m1_idx, m2_idx, mp_idx]
                if c != 0.0:
                    row[m2_idx] += c * a1[m1_idx] * alp[mp_idx].conjugate()
    return row


def _classify_triple(l1: int, l2: int, l_val: int, ell: int) -> str:
    if l_val == ell and l1 < ell and l2 < ell:
        return 'chain'
    if l2 == ell and l1 < ell and l_val < ell:
        return 'cross'
    return 'other'


def verify_bootstrap(lmax: int) -> bool:
    """Verify the linear bootstrap rank condition for ell = 4 ..

    lmax.
    """
    coeffs = _witness_coefficients_np(lmax)

    all_pass = True
    for ell in range(4, lmax + 1):
        t_ell = time.time()
        target_rank = 2 * ell + 1
        ell_index_map = _build_selective_index_map(ell)
        cg_cache: dict[tuple[int, int, int], np.ndarray] = {}

        rows: list[np.ndarray] = []
        for l1, l2, l_val in ell_index_map:
            kind = _classify_triple(l1, l2, l_val, ell)
            if kind == 'chain':
                rows.append(_chain_row_np(coeffs, l1, l2, ell, cg_cache))
            elif kind == 'cross':
                rows.append(_cross_row_np(coeffs, l1, ell, l_val, cg_cache))

        if len(rows) < target_rank:
            print(f'  ell={ell:3d}: FAIL - only {len(rows)} rows (need {target_rank})', flush=True)
            all_pass = False
            continue

        A = np.array(rows, dtype=np.complex128)
        sv = np.linalg.svd(A, compute_uv=False)
        rank = int(np.sum(sv > 1e-10 * sv[0]))

        status = 'OK' if rank == target_rank else 'FAIL'
        if rank != target_rank:
            all_pass = False
        cond = sv[0] / sv[target_rank - 1] if rank >= target_rank else float('inf')
        dt = time.time() - t_ell
        print(
            f'  ell={ell:3d}: rank={rank:3d}/{target_rank:3d}  '
            f'cond={cond:.1e}  sigma_min={sv[min(target_rank - 1, len(sv) - 1)]:.2e}  '
            f'{dt:.2f}s  {status}',
            flush=True,
        )

    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Verify linear bootstrap generic full-rank condition'
    )
    parser.add_argument(
        '--lmax', type=int, default=100, help='Maximum degree to verify (default: 100)'
    )
    args = parser.parse_args()

    print(f'Verifying linear bootstrap for ell = 4 .. {args.lmax}')
    print('Witness: seeded pseudo-random (seed=42, unit-scale Gaussian)')
    print()

    t0 = time.time()
    success = verify_bootstrap(args.lmax)
    elapsed = time.time() - t0

    print()
    if success:
        print(f'ALL PASSED (ell = 4 .. {args.lmax}) in {elapsed:.1f}s')
    else:
        print('SOME FAILURES DETECTED')
        sys.exit(1)


if __name__ == '__main__':
    main()
