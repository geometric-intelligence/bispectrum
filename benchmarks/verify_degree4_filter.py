"""Check which seed solutions survive to degree 4.

For each of the 10 seed fibre solutions (on the +v branch), try to find an F4 such that ALL
degree-4 augmented entries match the true signal's values. Uses nonlinear least squares (the
entries are a mix of linear and quadratic in F4).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from bispectrum._cg import clebsch_gordan
from bispectrum.so3_on_s2 import _build_cg_power_index_map, _build_selective_index_map

A0_VAL = 1.0
C_VAL = 1.0

SEED_SOLUTIONS_POS_V = [
    (0.39090528, 0.20888667, 0.32700283, 0.33558389, 0.59889718, 0.19447556, 0.32107748),
    (-0.00010923, 0.48117135, -0.20720501, 0.31128246, 0.56008158, -0.18547439, 0.34601322),
    (0.57591834, 0.43776535, 0.01800310, 0.48187285, 0.10894240, 0.43896959, 0.21120146),
    (0.24583608, 0.44534492, 0.07106802, -0.01351650, 0.68537352, -0.34276345, 0.13478520),
    (0.08780215, 0.56478013, -0.16001445, 0.44198736, 0.31000443, -0.24722088, 0.37170455),
    (0.50000000, 0.37500000, 0.28571429, 0.45454545, 0.25000000, 0.23076923, 0.41176471),
    (0.49490489, 0.37151277, 0.21915283, 0.54338143, 0.19063878, 0.24512194, 0.37278334),
    (0.46736567, 0.34438752, 0.17049480, 0.61936446, 0.18234553, 0.24887675, 0.32169146),
    (0.42567920, 0.33521088, -0.02208087, 0.69190599, 0.16535166, 0.33583614, 0.12983666),
    (0.51745421, 0.49876452, 0.08415742, -0.01024443, 0.44640593, -0.46119515, 0.19329177),
]
WITNESS_IDX = 5

_cg_memo: dict[tuple, float] = {}


def _cg(l1: int, m1: int, l2: int, m2: int, l: int, m: int) -> float:
    if m1 + m2 != m:
        return 0.0
    if abs(m) > l or abs(m1) > l1 or abs(m2) > l2:
        return 0.0
    if l < abs(l1 - l2) or l > l1 + l2:
        return 0.0
    key = (l1, m1, l2, m2, l, m)
    if key not in _cg_memo:
        _cg_memo[key] = float(clebsch_gordan(l1, m1, l2, m2, l, m))
    return _cg_memo[key]


def _make_F(ell: int, real_params: tuple | list | np.ndarray) -> np.ndarray:
    """Build complex SH coefficient vector from real gauge-fixed parameters."""
    F = np.zeros(2 * ell + 1, dtype=complex)
    if ell == 0:
        F[0] = real_params[0]
    elif ell == 1:
        F[1] = real_params[0]
    elif ell == 2:
        y, x, u, v = real_params
        F[2] = y
        F[3] = x
        F[1] = -x
        F[4] = u + 1j * v
        F[0] = u - 1j * v
    elif ell == 3:
        t, p1, q1, p2, q2, p3, q3 = real_params
        F[3] = t
        F[4] = p1 + 1j * q1
        F[2] = -(p1 - 1j * q1)
        F[5] = p2 + 1j * q2
        F[1] = p2 - 1j * q2
        F[6] = p3 + 1j * q3
        F[0] = -(p3 - 1j * q3)
    return F


def _make_F4_from_real(params9: np.ndarray) -> np.ndarray:
    """Build F4 from 9 real params: t, p1, q1, p2, q2, p3, q3, p4, q4."""
    t, p1, q1, p2, q2, p3, q3, p4, q4 = params9
    F = np.zeros(9, dtype=complex)
    F[4] = t
    F[5] = p1 + 1j * q1
    F[3] = -(p1 - 1j * q1)
    F[6] = p2 + 1j * q2
    F[2] = p2 - 1j * q2
    F[7] = p3 + 1j * q3
    F[1] = -(p3 - 1j * q3)
    F[8] = p4 + 1j * q4
    F[0] = p4 - 1j * q4
    return F


def _F4_to_real(F4: np.ndarray) -> np.ndarray:
    """Extract 9 real params from F4."""
    t = F4[4].real
    p1, q1 = F4[5].real, F4[5].imag
    p2, q2 = F4[6].real, F4[6].imag
    p3, q3 = F4[7].real, F4[7].imag
    p4, q4 = F4[8].real, F4[8].imag
    return np.array([t, p1, q1, p2, q2, p3, q3, p4, q4])


def _beta(Fs: dict[int, np.ndarray], l1: int, l2: int, l: int) -> complex:
    F1, F2, Fl = Fs[l1], Fs[l2], Fs[l]
    result = 0.0 + 0j
    for m in range(-l, l + 1):
        for m1 in range(-l1, l1 + 1):
            m2 = m - m1
            if abs(m2) > l2:
                continue
            c = _cg(l1, m1, l2, m2, l, m)
            if c == 0:
                continue
            result += c * F1[m1 + l1] * F2[m2 + l2] * np.conj(Fl[m + l])
    return result


def _P(Fs: dict[int, np.ndarray], l1: int, l2: int, l: int) -> float:
    F1, F2 = Fs[l1], Fs[l2]
    proj = np.zeros(2 * l + 1, dtype=complex)
    for m in range(-l, l + 1):
        for m1 in range(-l1, l1 + 1):
            m2 = m - m1
            if abs(m2) > l2:
                continue
            c = _cg(l1, m1, l2, m2, l, m)
            if c == 0:
                continue
            proj[m + l] += c * F1[m1 + l1] * F2[m2 + l2]
    return float(np.sum(np.abs(proj) ** 2))


def main() -> None:
    print('=' * 70)
    print('Degree-4 Filter: Which seed solutions survive?')
    print('=' * 70)

    F0 = _make_F(0, [A0_VAL])
    F1 = _make_F(1, [C_VAL])
    F2 = _make_F(2, [3 / 7, 2 / 5, 1 / 3, 4 / 9])
    F3_true = _make_F(3, SEED_SOLUTIONS_POS_V[WITNESS_IDX])

    rng = np.random.RandomState(42)
    F4_real_params = rng.randn(9) * 0.5
    F4_true = _make_F4_from_real(F4_real_params)

    Fs_true = {0: F0, 1: F1, 2: F2, 3: F3_true, 4: F4_true}

    bisp4 = _build_selective_index_map(4)
    cgp4 = _build_cg_power_index_map(4)

    d4_bisp = [t for t in bisp4 if max(t) == 4]
    d4_cgp = [t for t in cgp4 if 4 in t[:2]]

    print(f'Degree-4 bisp triples ({len(d4_bisp)}): {d4_bisp}')
    print(f'Degree-4 CGP triples ({len(d4_cgp)}): {d4_cgp}')

    target_bisp = {t: _beta(Fs_true, *t) for t in d4_bisp}
    target_cgp = {t: _P(Fs_true, *t) for t in d4_cgp}

    print('\nTrue degree-4 values:')
    for t, v in target_bisp.items():
        print(f'  β{t} = {v.real:+.8f} + {v.imag:+.8f}j')
    for t, v in target_cgp.items():
        print(f'  P{t} = {v:.8f}')

    def make_residual(F3_fixed: np.ndarray):
        def residual(f4_params: np.ndarray) -> np.ndarray:
            F4 = _make_F4_from_real(f4_params)
            Fs = {0: F0, 1: F1, 2: F2, 3: F3_fixed, 4: F4}
            res = []
            for t in d4_bisp:
                b = _beta(Fs, *t)
                bt = target_bisp[t]
                res.append(b.real - bt.real)
                res.append(b.imag - bt.imag)
            for t in d4_cgp:
                p = _P(Fs, *t)
                res.append(p - target_cgp[t])
            return np.array(res)

        return residual

    print(f'\n{"=" * 70}')
    print('Testing each seed solution...')
    print(f'{"=" * 70}')

    for sol_idx, d3_params in enumerate(SEED_SOLUTIONS_POS_V):
        is_witness = sol_idx == WITNESS_IDX
        F3_spur = _make_F(3, d3_params)
        residual_fn = make_residual(F3_spur)

        if is_witness:
            res0 = residual_fn(F4_real_params)
            print(f'\nSolution {sol_idx} [WITNESS]:')
            print(f'  True F4 residual: {np.linalg.norm(res0):.2e}')

        best_cost = np.inf
        search_rng = np.random.RandomState(sol_idx * 100 + 7)
        for _trial in range(200):
            x0 = search_rng.randn(9) * 1.0
            try:
                result = least_squares(residual_fn, x0, method='lm', max_nfev=1000)
                if result.cost < best_cost:
                    best_cost = result.cost
                    result.x.copy()
            except Exception:  # nosec B110
                pass

        tag = ' [WITNESS]' if is_witness else ''
        if not is_witness:
            print(f'\nSolution {sol_idx}{tag}:')
        print(f'  Best fit cost (200 starts): {best_cost:.2e}')
        print(f'  Best fit ||residual||: {np.sqrt(2 * best_cost):.2e}')

        if best_cost < 1e-20:
            print('  ==> SURVIVES to degree 4')
        else:
            print('  ==> ELIMINATED at degree 4')

    print(f'\n{"=" * 70}')


if __name__ == '__main__':
    main()
