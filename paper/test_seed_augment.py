"""Test: does adding more CG power entries shrink the seed fibre?

Baseline: 4 CG power entries at degree 3 gives 10 solutions per v-branch.
Candidate extra entries: P(1,3,3), P(2,3,3).
Goal: see if augmenting reduces the fibre count to 1 per branch.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.optimize import least_squares

from bispectrum._cg import clebsch_gordan


A0_VAL = 1.0
C_VAL = 1.0
WITNESS_D2 = (3 / 7, 2 / 5, 1 / 3, 4 / 9)
WITNESS_D3 = (1 / 2, 3 / 8, 2 / 7, 5 / 11, 1 / 4, 3 / 13, 7 / 17)

BISP_D2: list[tuple[int, int, int]] = [(1, 1, 2), (0, 2, 2), (2, 2, 2)]
CGP_D2: list[tuple[int, int, int]] = [(1, 2, 1)]
BISP_D3: list[tuple[int, int, int]] = [(1, 2, 3), (1, 3, 2), (2, 3, 1), (0, 3, 3), (3, 3, 2)]

BASELINE_CGP_D3: list[tuple[int, int, int]] = [
    (1, 3, 2), (2, 3, 1), (2, 3, 2), (3, 3, 2),
]
EXTRA_CANDIDATES: list[tuple[int, int, int]] = [
    (1, 3, 3), (2, 3, 3), (2, 2, 3), (1, 1, 3),
]


def precompute_cg(triples: list[tuple[int, int, int]]) -> dict:
    cache = {}
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


def make_F2(y, x, u, v):
    F = np.zeros(5, dtype=complex)
    F[2] = y
    F[3] = x
    F[1] = -x
    F[4] = u + 1j * v
    F[0] = u - 1j * v
    return F


def make_F3(params):
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


def beta(Fs, l1, l2, l, cg):
    C = cg[(l1, l2, l)]
    return complex(np.einsum('ijk,i,j,k->', C, Fs[l1], Fs[l2], np.conj(Fs[l])))


def cgp(Fs, l1, l2, l, cg):
    C = cg[(l1, l2, l)]
    proj = np.einsum('ijk,i,j->k', C, Fs[l1], Fs[l2])
    return float(np.sum(np.abs(proj) ** 2))


def build_residual(bisp_d3, cgp_d3, targets_b, targets_p, F0, F1, F2, cg):
    def resid(p):
        F3 = make_F3(p)
        Fs = {0: F0, 1: F1, 2: F2, 3: F3}
        r = []
        for i, t in enumerate(bisp_d3):
            r.append(beta(Fs, *t, cg).real - targets_b[i])
        for i, t in enumerate(cgp_d3):
            r.append(cgp(Fs, *t, cg) - targets_p[i])
        return np.array(r)
    return resid


def count_solutions(cgp_d3, n_starts=2000, seed=42):
    all_triples = list(set(BISP_D2 + CGP_D2 + BISP_D3 + cgp_d3))
    cg = precompute_cg(all_triples)

    F0 = np.array([A0_VAL], dtype=complex)
    F1 = np.zeros(3, dtype=complex)
    F1[1] = C_VAL

    y0, x0, u0, v0 = WITNESS_D2
    F2_pos = make_F2(y0, x0, u0, v0)
    F2_neg = make_F2(y0, x0, u0, -v0)
    F3_w = make_F3(np.array(WITNESS_D3))
    Fs_w = {0: F0, 1: F1, 2: F2_pos, 3: F3_w}

    targets_b = np.array([beta(Fs_w, *t, cg).real for t in BISP_D3])
    targets_p = np.array([cgp(Fs_w, *t, cg) for t in cgp_d3])

    total_counts = {}
    for name, F2 in [('+v', F2_pos), ('-v', F2_neg)]:
        resid = build_residual(BISP_D3, cgp_d3, targets_b, targets_p, F0, F1, F2, cg)
        sols = []
        rng = np.random.RandomState(seed)
        for i in range(n_starts):
            x0_ = rng.randn(7) * 2.0
            try:
                r = least_squares(resid, x0_, method='lm', max_nfev=500)
            except Exception:
                continue
            if r.cost < 1e-20:
                is_new = all(np.max(np.abs(r.x - s)) > 1e-6 for s in sols)
                if is_new:
                    sols.append(r.x.copy())
        total_counts[name] = len(sols)
    return total_counts


def main():
    N = 800
    print('=' * 60)
    print('Baseline CGP augmentation:', BASELINE_CGP_D3, flush=True)
    counts = count_solutions(BASELINE_CGP_D3, n_starts=N)
    print(f'  Solutions per branch: {counts}', flush=True)

    for extra in [
        [(1, 3, 3)],
        [(2, 3, 3)],
        [(1, 3, 3), (2, 3, 3)],
        [(2, 2, 3)],
        [(1, 1, 3)],
        [(1, 3, 3), (2, 3, 3), (2, 2, 3), (1, 1, 3)],
    ]:
        aug = BASELINE_CGP_D3 + extra
        print('=' * 60)
        print(f'Augmented with extra: {extra}', flush=True)
        counts = count_solutions(aug, n_starts=N)
        print(f'  Solutions per branch: {counts}', flush=True)


if __name__ == '__main__':
    main()
