"""Test: uniform structural witness for bootstrap rank uniform in ℓ.

Hypothesis: for the closed-form selective family T_ℓ (ℓ ≥ 8), the
bootstrap matrix A_ℓ has full complex rank 2ℓ+1 at a specific signal
F_a^m = something simple (ζ^{a+m}, or similar).

If this works at one explicit ζ for ALL ℓ tested, then by
density the polynomial det A_ℓ(F_0,...,F_{ℓ-1}) is generically nonzero.

We test several candidates:
  W1: F_a^m = 1 for all a, m (constant)
  W2: F_a^m = (m+1) (varying with m)
  W3: F_a^m = ζ^{a+m} for ζ = 2, 3
  W4: F_a^m = (a+1)(m+ℓ+1)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from bispectrum._cg import clebsch_gordan


def cg_array(l1: int, l2: int, l: int) -> np.ndarray:
    mat = np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * l + 1), dtype=float)
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            m = m1 + m2
            if abs(m) > l:
                continue
            v = clebsch_gordan(l1, m1, l2, m2, l, m)
            if v != 0.0:
                mat[m1 + l1, m2 + l2, m + l] = v
    return mat


def closed_form_block(ell: int) -> list[tuple[int, int, int]]:
    block: list[tuple[int, int, int]] = []
    for a in range(1, ell):
        block.append((a, ell, ell - a))
    for a in range(2, ell):
        block.append((a, ell, ell - a + 1))
    for a in range(1, 5):
        block.append((a, ell - a, ell))
    return block


def witness_F(name: str, lmax: int, zeta: complex = 2.0 + 0.3j) -> dict[int, np.ndarray]:
    Fs: dict[int, np.ndarray] = {}
    for a in range(lmax + 1):
        size = 2 * a + 1
        F = np.zeros(size, dtype=complex)
        for m_off in range(size):
            m = m_off - a
            if name == 'W1':
                F[m_off] = 1.0
            elif name == 'W2':
                F[m_off] = m + 1
            elif name == 'W3':
                F[m_off] = zeta ** (a + m)
            elif name == 'W4':
                F[m_off] = (a + 1) * (m + a + 1)
            elif name == 'W5':
                F[m_off] = (1 + 0.1j) ** (a + m + 1)
        Fs[a] = F
    return Fs


def build_bootstrap_matrix(
    ell: int, Fs: dict[int, np.ndarray], block: list[tuple[int, int, int]]
) -> np.ndarray:
    n_unk = 2 * ell + 1
    rows = []
    for trip in block:
        l1, l2, l_out = trip
        C = cg_array(l1, l2, l_out)
        if l1 == ell:
            row = np.zeros(n_unk, dtype=complex)
            for m1 in range(-l1, l1 + 1):
                for m2 in range(-l2, l2 + 1):
                    m = m1 + m2
                    if abs(m) > l_out:
                        continue
                    coef = C[m1 + l1, m2 + l2, m + l_out] * Fs[l2][m2 + l2] * np.conj(Fs[l_out][m + l_out])
                    row[m1 + l1] += coef
        elif l2 == ell:
            row = np.zeros(n_unk, dtype=complex)
            for m1 in range(-l1, l1 + 1):
                for m2 in range(-l2, l2 + 1):
                    m = m1 + m2
                    if abs(m) > l_out:
                        continue
                    coef = C[m1 + l1, m2 + l2, m + l_out] * Fs[l1][m1 + l1] * np.conj(Fs[l_out][m + l_out])
                    row[m2 + l2] += coef
        elif l_out == ell:
            row = np.zeros(n_unk, dtype=complex)
            for m1 in range(-l1, l1 + 1):
                for m2 in range(-l2, l2 + 1):
                    m = m1 + m2
                    if abs(m) > l_out:
                        continue
                    coef = C[m1 + l1, m2 + l2, m + l_out] * Fs[l1][m1 + l1] * Fs[l2][m2 + l2]
                    row[m + l_out] += coef
        else:
            raise RuntimeError(f'ell {ell} not in triple {trip}')
        rows.append(row)
    return np.array(rows)


def test_witness(name: str, l_range: list[int]) -> None:
    print(f'\n=== Witness {name} ===')
    for ell in l_range:
        block = closed_form_block(ell)
        Fs = witness_F(name, ell)
        A = build_bootstrap_matrix(ell, Fs, block)
        sv = np.linalg.svd(A, compute_uv=False)
        rank = int(np.sum(sv > 1e-9 * sv[0]))
        cond = sv[0] / max(sv[-1], 1e-300)
        marker = 'OK ' if rank == 2 * ell + 1 else 'BAD'
        print(f'  {marker} ell={ell:3d}  rank={rank:3d}/{2 * ell + 1:3d}  cond={cond:.2e}')


def witness_random(lmax: int, seed: int = 42) -> dict[int, np.ndarray]:
    rng = np.random.RandomState(seed)
    Fs: dict[int, np.ndarray] = {}
    for a in range(lmax + 1):
        size = 2 * a + 1
        F = rng.randn(size) + 1j * rng.randn(size)
        Fs[a] = F
    return Fs


def witness_real_random(lmax: int, seed: int = 42) -> dict[int, np.ndarray]:
    """Gaussian random with reality F^{-m} = (-1)^m conj(F^m)."""
    rng = np.random.RandomState(seed)
    Fs: dict[int, np.ndarray] = {}
    for a in range(lmax + 1):
        size = 2 * a + 1
        F = np.zeros(size, dtype=complex)
        F[a] = rng.randn()
        for m in range(1, a + 1):
            re = rng.randn()
            im = rng.randn()
            F[a + m] = re + 1j * im
            F[a - m] = ((-1) ** m) * (re - 1j * im)
        Fs[a] = F
    return Fs


def test_witness_named(label: str, Fs: dict[int, np.ndarray], l_list: list[int]) -> None:
    print(f'\n=== {label} ===')
    for ell in l_list:
        block = closed_form_block(ell)
        A = build_bootstrap_matrix(ell, Fs, block)
        sv = np.linalg.svd(A, compute_uv=False)
        scale = sv[0]
        for thr in [1e-9, 1e-12, 1e-14]:
            rank = int(np.sum(sv > thr * scale))
            if rank == 2 * ell + 1:
                break
        cond = sv[0] / max(sv[-1], 1e-300)
        marker = 'OK ' if rank == 2 * ell + 1 else 'BAD'
        print(f'  {marker} ell={ell:3d}  rank={rank:3d}/{2 * ell + 1:3d}  cond={cond:.2e}  thr={thr:.0e}')


def main():
    l_list = [8, 10, 15, 20, 30, 40, 50, 70, 100]
    for name in ['W1', 'W2', 'W3', 'W4', 'W5']:
        test_witness(name, l_list)
    Fs_rng = witness_random(100, seed=42)
    test_witness_named('Random complex (seed 42)', Fs_rng, l_list)
    Fs_real = witness_real_random(100, seed=42)
    test_witness_named('Random real-signal (seed 42)', Fs_real, l_list)


if __name__ == '__main__':
    main()
