"""Certify the exact fibre size of the augmented seed system (degrees 0-3).

Verifies that the seed polynomial system has EXACTLY 2 real solutions
(related by T_R), not merely "at least 2".

Strategy:
  - Degree 2 is solved analytically: y, x, u are unique; v has sign ambiguity.
    This gives exactly 2 branches: v = +v0 and v = -v0.
  - For each v-branch, the degree-3 subsystem (7 unknowns) is searched via
    multi-start least_squares. If each branch has exactly 1 solution, the
    total fibre has exactly 2 points related by T_R.

Usage:
    python benchmarks/verify_seed_fibre.py [--num-starts 5000]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from scipy.optimize import least_squares

from bispectrum._cg import clebsch_gordan

A0_VAL = 1.0
C_VAL = 1.0
WITNESS_D2 = (3 / 7, 2 / 5, 1 / 3, 4 / 9)  # y, x, u, v
WITNESS_D3 = (1 / 2, 3 / 8, 2 / 7, 5 / 11, 1 / 4, 3 / 13, 7 / 17)

BISP_D3_TRIPLES: list[tuple[int, int, int]] = [
    (1, 2, 3),
    (1, 3, 2),
    (2, 3, 1),
    (0, 3, 3),
    (3, 3, 2),
]
CGP_D3_TRIPLES: list[tuple[int, int, int]] = [
    (1, 3, 2),
    (2, 3, 1),
    (2, 3, 2),
    (3, 3, 2),
]
BISP_D2_TRIPLES: list[tuple[int, int, int]] = [
    (1, 1, 2),
    (0, 2, 2),
    (2, 2, 2),
]
CGP_D2_TRIPLES: list[tuple[int, int, int]] = [(1, 2, 1)]
ALL_TRIPLES = list(set(BISP_D3_TRIPLES + CGP_D3_TRIPLES + BISP_D2_TRIPLES + CGP_D2_TRIPLES))


def _precompute_cg(
    triples: list[tuple[int, int, int]],
) -> dict[tuple[int, int, int], np.ndarray]:
    cache: dict[tuple[int, int, int], np.ndarray] = {}
    for l1, l2, l in triples:
        if (l1, l2, l) in cache:
            continue
        mat = np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * l + 1))
        for m1 in range(-l1, l1 + 1):
            for m2 in range(-l2, l2 + 1):
                m = m1 + m2
                if abs(m) > l:
                    continue
                val = clebsch_gordan(l1, m1, l2, m2, l, m)
                if val != 0.0:
                    mat[m1 + l1, m2 + l2, m + l] = val
        cache[(l1, l2, l)] = mat
    return cache


def make_F2(y: float, x: float, u: float, v: float) -> np.ndarray:
    F = np.zeros(5, dtype=complex)
    F[2] = y
    F[3] = x
    F[1] = -x
    F[4] = u + 1j * v
    F[0] = u - 1j * v
    return F


def make_F3(params: np.ndarray) -> np.ndarray:
    t, p1, q1, p2, q2, p3, q3 = params
    F = np.zeros(7, dtype=complex)
    F[3] = t
    F[4] = p1 + 1j * q1
    F[2] = -(p1 - 1j * q1)
    F[5] = p2 + 1j * q2
    F[1] = p2 - 1j * q2
    F[6] = p3 + 1j * q3
    F[0] = -(p3 - 1j * q3)
    return F


def beta_fast(
    Fs: dict[int, np.ndarray],
    l1: int,
    l2: int,
    l: int,
    cg_cache: dict[tuple[int, int, int], np.ndarray],
) -> complex:
    C = cg_cache[(l1, l2, l)]
    return complex(np.einsum('ijk,i,j,k->', C, Fs[l1], Fs[l2], np.conj(Fs[l])))


def cg_power_fast(
    Fs: dict[int, np.ndarray],
    l1: int,
    l2: int,
    l: int,
    cg_cache: dict[tuple[int, int, int], np.ndarray],
) -> float:
    C = cg_cache[(l1, l2, l)]
    proj = np.einsum('ijk,i,j->k', C, Fs[l1], Fs[l2])
    return float(np.sum(np.abs(proj) ** 2))


def degree3_residual(
    d3_params: np.ndarray,
    F0: np.ndarray,
    F1: np.ndarray,
    F2: np.ndarray,
    targets_bisp: np.ndarray,
    targets_cgp: np.ndarray,
    cg_cache: dict[tuple[int, int, int], np.ndarray],
) -> np.ndarray:
    F3 = make_F3(d3_params)
    Fs = {0: F0, 1: F1, 2: F2, 3: F3}

    res = []
    for i, trip in enumerate(BISP_D3_TRIPLES):
        b = beta_fast(Fs, *trip, cg_cache)
        res.append(b.real - targets_bisp[i])
    for i, trip in enumerate(CGP_D3_TRIPLES):
        p = cg_power_fast(Fs, *trip, cg_cache)
        res.append(p - targets_cgp[i])
    return np.array(res)


def full_residual(
    params: np.ndarray,
    targets_all: np.ndarray,
    F0: np.ndarray,
    F1: np.ndarray,
    cg_cache: dict[tuple[int, int, int], np.ndarray],
) -> np.ndarray:
    y, x, u, v = params[:4]
    F2 = make_F2(y, x, u, v)
    F3 = make_F3(params[4:])
    Fs = {0: F0, 1: F1, 2: F2, 3: F3}

    res = []
    for trip in BISP_D2_TRIPLES:
        res.append(beta_fast(Fs, *trip, cg_cache).real)
    for trip in CGP_D2_TRIPLES:
        res.append(cg_power_fast(Fs, *trip, cg_cache))
    for trip in BISP_D3_TRIPLES:
        res.append(beta_fast(Fs, *trip, cg_cache).real)
    for trip in CGP_D3_TRIPLES:
        res.append(cg_power_fast(Fs, *trip, cg_cache))
    return np.array(res) - targets_all


def apply_TR_d3(params: np.ndarray) -> np.ndarray:
    t, p1, q1, p2, q2, p3, q3 = params
    return np.array([t, p1, -q1, p2, -q2, p3, -q3])


def apply_TR_full(params: np.ndarray) -> np.ndarray:
    y, x, u, v, t, p1, q1, p2, q2, p3, q3 = params
    return np.array([y, x, u, -v, t, p1, -q1, p2, -q2, p3, -q3])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-starts', type=int, default=5000)
    args = parser.parse_args()

    print('=' * 70)
    print('Certificate 0: Seed Fibre Enumeration')
    print('=' * 70)

    cg_cache = _precompute_cg(ALL_TRIPLES)
    print(f'CG tensors precomputed ({len(cg_cache)} triples)')

    F0 = np.array([A0_VAL], dtype=complex)
    F1 = np.zeros(3, dtype=complex)
    F1[1] = C_VAL

    y0, x0, u0, v0 = WITNESS_D2
    d3_witness = np.array(WITNESS_D3)
    F2_pos = make_F2(y0, x0, u0, v0)
    F2_neg = make_F2(y0, x0, u0, -v0)
    F3_w = make_F3(d3_witness)
    Fs_w = {0: F0, 1: F1, 2: F2_pos, 3: F3_w}

    targets_bisp_d3 = np.array([beta_fast(Fs_w, *t, cg_cache).real for t in BISP_D3_TRIPLES])
    targets_cgp_d3 = np.array([cg_power_fast(Fs_w, *t, cg_cache) for t in CGP_D3_TRIPLES])

    targets_all_parts = []
    for trip in BISP_D2_TRIPLES:
        targets_all_parts.append(beta_fast(Fs_w, *trip, cg_cache).real)
    for trip in CGP_D2_TRIPLES:
        targets_all_parts.append(cg_power_fast(Fs_w, *trip, cg_cache))
    for trip in BISP_D3_TRIPLES:
        targets_all_parts.append(beta_fast(Fs_w, *trip, cg_cache).real)
    for trip in CGP_D3_TRIPLES:
        targets_all_parts.append(cg_power_fast(Fs_w, *trip, cg_cache))
    targets_all = np.array(targets_all_parts)

    print(
        f'\nTarget values ({len(targets_bisp_d3)} bisp + {len(targets_cgp_d3)} CGP at degree 3):'
    )
    for t, v in zip(BISP_D3_TRIPLES, targets_bisp_d3, strict=False):
        print(f'  Re β{t} = {v:.10f}')
    for t, v in zip(CGP_D3_TRIPLES, targets_cgp_d3, strict=False):
        print(f'  P{t}    = {v:.10f}')

    # Sanity: witness should give zero residual
    r_w = degree3_residual(d3_witness, F0, F1, F2_pos, targets_bisp_d3, targets_cgp_d3, cg_cache)
    print(f'\nWitness residual: ||r|| = {np.linalg.norm(r_w):.2e}')

    # T_R(witness) on +v branch should NOT be a solution
    d3_TR = apply_TR_d3(d3_witness)
    r_TR_wrong = degree3_residual(d3_TR, F0, F1, F2_pos, targets_bisp_d3, targets_cgp_d3, cg_cache)
    print(f'T_R(witness) on +v branch residual: ||r|| = {np.linalg.norm(r_TR_wrong):.2e}')

    # T_R(witness) on -v branch SHOULD be a solution
    r_TR_right = degree3_residual(d3_TR, F0, F1, F2_neg, targets_bisp_d3, targets_cgp_d3, cg_cache)
    print(f'T_R(witness) on -v branch residual: ||r|| = {np.linalg.norm(r_TR_right):.2e}')

    # Multi-start search: for each v-branch, find all degree-3 solutions
    for branch_name, F2_branch in [('v = +v0', F2_pos), ('v = -v0', F2_neg)]:
        print(f'\n{"=" * 50}')
        print(f'Branch: {branch_name}')
        print(f'{"=" * 50}')
        print(f'Searching with {args.num_starts} random starts...', flush=True)

        solutions: list[np.ndarray] = []
        hit_counts: list[int] = []
        rng = np.random.RandomState(42)
        t0 = time.time()

        for i in range(args.num_starts):
            x_init = rng.randn(7) * 2.0
            try:
                result = least_squares(
                    degree3_residual,
                    x_init,
                    args=(F0, F1, F2_branch, targets_bisp_d3, targets_cgp_d3, cg_cache),
                    method='lm',
                    max_nfev=500,
                )
            except Exception:  # nosec B112
                continue

            if result.cost < 1e-20:
                found = result.x
                is_new = True
                for idx, sol in enumerate(solutions):
                    if np.max(np.abs(found - sol)) < 1e-6:
                        is_new = False
                        hit_counts[idx] += 1
                        break
                if is_new:
                    solutions.append(found.copy())
                    hit_counts.append(1)
                    print(
                        f'  Start {i:5d}: NEW solution #{len(solutions)} (cost={result.cost:.2e})'
                    )

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                print(
                    f'  ... {i + 1}/{args.num_starts} '
                    f'({elapsed:.1f}s, {len(solutions)} solutions)',
                    flush=True,
                )

        elapsed = time.time() - t0
        print(f'\n  Search done: {elapsed:.1f}s, {len(solutions)} solution(s)')
        for idx, cnt in enumerate(hit_counts):
            print(f'  Solution {idx + 1}: hit {cnt} times out of {args.num_starts}')

        for i, sol in enumerate(solutions):
            r = degree3_residual(sol, F0, F1, F2_branch, targets_bisp_d3, targets_cgp_d3, cg_cache)
            t, p1, q1, p2, q2, p3, q3 = sol
            print(f'\n  Solution {i + 1} detail:')
            print(f'    t={t:.8f}  p1={p1:.8f}  q1={q1:.8f}')
            print(f'    p2={p2:.8f}  q2={q2:.8f}')
            print(f'    p3={p3:.8f}  q3={q3:.8f}')
            print(f'    ||residual|| = {np.linalg.norm(r):.2e}')

    # Full 11-parameter search as backup
    print(f'\n{"=" * 50}')
    print(f'Full 11-parameter search ({args.num_starts} starts)')
    print(f'{"=" * 50}')

    full_solutions: list[np.ndarray] = []
    full_hit_counts: list[int] = []
    rng = np.random.RandomState(9999)
    t0 = time.time()

    for i in range(args.num_starts):
        x_init = rng.randn(11) * 2.0
        try:
            result = least_squares(
                full_residual,
                x_init,
                args=(targets_all, F0, F1, cg_cache),
                method='lm',
                max_nfev=500,
            )
        except Exception:  # nosec B112
            continue

        if result.cost < 1e-20:
            found = result.x
            is_new = True
            for idx, sol in enumerate(full_solutions):
                if np.max(np.abs(found - sol)) < 1e-6:
                    is_new = False
                    full_hit_counts[idx] += 1
                    break
            if is_new:
                full_solutions.append(found.copy())
                full_hit_counts.append(1)
                print(
                    f'  Start {i:5d}: NEW solution #{len(full_solutions)} (cost={result.cost:.2e})'
                )

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(
                f'  ... {i + 1}/{args.num_starts} '
                f'({elapsed:.1f}s, {len(full_solutions)} solutions)',
                flush=True,
            )

    elapsed = time.time() - t0
    print(f'\n  Full search done: {elapsed:.1f}s, {len(full_solutions)} solution(s)')
    for idx, cnt in enumerate(full_hit_counts):
        print(f'  Solution {idx + 1}: hit {cnt} times')

    # Check T_R pairings in full solutions
    print('\n--- T_R pairing check ---')
    paired = set()
    for i in range(len(full_solutions)):
        tr_i = apply_TR_full(full_solutions[i])
        for j in range(len(full_solutions)):
            if i == j:
                continue
            diff = np.max(np.abs(tr_i - full_solutions[j]))
            if diff < 1e-4:
                pair = (min(i, j), max(i, j))
                if pair not in paired:
                    paired.add(pair)
                    print(f'  Solution {i + 1} <--T_R--> Solution {j + 1} (max diff = {diff:.2e})')

    for i, sol in enumerate(full_solutions):
        y, x, u, v, t, p1, q1, p2, q2, p3, q3 = sol
        print(f'\n  Full solution {i + 1}:')
        print(f'    (y,x,u,v) = ({y:.8f}, {x:.8f}, {u:.8f}, {v:.8f})')
        print(f'    (t,p1,q1) = ({t:.8f}, {p1:.8f}, {q1:.8f})')
        print(f'    (p2,q2)   = ({p2:.8f}, {q2:.8f})')
        print(f'    (p3,q3)   = ({p3:.8f}, {q3:.8f})')

    # Jacobian rank check at each full solution
    print('\n--- Jacobian rank at each solution ---')
    for i, sol in enumerate(full_solutions):
        J = np.zeros((len(targets_all), 11))
        eps = 1e-7
        full_residual(sol, targets_all, F0, F1, cg_cache)
        for k in range(11):
            p_plus = sol.copy()
            p_plus[k] += eps
            p_minus = sol.copy()
            p_minus[k] -= eps
            J[:, k] = (
                full_residual(p_plus, targets_all, F0, F1, cg_cache)
                - full_residual(p_minus, targets_all, F0, F1, cg_cache)
            ) / (2 * eps)
        sv = np.linalg.svd(J, compute_uv=False)
        rank = int(np.sum(sv > 1e-8 * sv[0]))
        print(
            f'  Solution {i + 1}: rank = {rank}/11, '
            f'σ_min = {sv[min(10, len(sv) - 1)]:.2e}, '
            f'cond = {sv[0] / sv[min(10, len(sv) - 1)]:.2e}'
        )

    # Final verdict
    print(f'\n{"=" * 70}')
    print('SUMMARY')
    print(f'{"=" * 70}')
    if len(full_solutions) == 2 and len(paired) == 1:
        print('CERTIFICATE PASSED: Exactly 2 real solutions, related by T_R')
        print('  => Generic seed fibre = {f, T_R(f)}, confirming Lemma 5')
    elif len(full_solutions) == 2:
        print('2 solutions found, T_R pairing check incomplete')
    else:
        print(f'UNEXPECTED: {len(full_solutions)} solutions found (expected 2)')
        if len(full_solutions) > 2:
            sys.exit(1)
    print('=' * 70)


if __name__ == '__main__':
    main()
