"""Fiber computation: solve Phi_aug(x) = Phi_aug(x0) for ALL real solutions.

Degree-by-degree approach: at each ell, fix lower degrees to witness values,
then search for all real solutions of the per-degree constraints.
Uses damped Gauss-Newton with many random starts.

The augmented invariant includes mandatory even self-coupling entries
beta(ell,ell,l) (l even, 2 <= l <= ell) for global injectivity.
"""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from bispectrum.so3_on_s2 import (
    _build_selective_index_map, load_cg_matrices,
    _build_cg_power_index_map,
)

CG_NP: dict[tuple[int, int], np.ndarray] = {}
A0, C_GAUGE = 1.5, 1.2


def _init_cg(lmax: int):
    cg_torch = load_cg_matrices(lmax)
    for k, v in cg_torch.items():
        CG_NP[k] = v.numpy()


def params_to_fc_np(p: np.ndarray, lmax: int) -> dict[int, np.ndarray]:
    fc: dict[int, np.ndarray] = {}
    fc[0] = np.array([A0 + 0j])
    fc[1] = np.array([0.0 + 0j, C_GAUGE + 0j, 0.0 + 0j])
    idx = 0
    for lv in range(2, lmax + 1):
        coeffs = np.zeros(2 * lv + 1, dtype=np.complex128)
        for m in range(1, lv + 1):
            if lv == 2 and m == 1:
                re = p[idx]; idx += 1; im = 0.0
            else:
                re = p[idx]; im = p[idx + 1]; idx += 2
            coeffs[lv + m] = re + 1j * im
            coeffs[lv - m] = ((-1.0) ** m) * (re - 1j * im)
        coeffs[lv] = p[idx] + 0j; idx += 1
        fc[lv] = coeffs
    return fc


def bispectrum_entry_np(fc, l1, l2, l_val):
    key = (min(l1, l2), max(l1, l2))
    if l1 <= l2:
        outer = np.outer(fc[l1], fc[l2]).ravel()
    else:
        outer = np.outer(fc[l2], fc[l1]).ravel()
    cg = CG_NP[key]
    coupled = outer @ cg
    col_start = 0
    for lv in range(abs(l1 - l2), l1 + l2 + 1):
        sz = 2 * lv + 1
        if lv == l_val:
            proj = coupled[col_start:col_start + sz]
            return np.sum(proj * np.conj(fc[l_val]))
        col_start += sz
    return 0.0 + 0j


def cg_power_entry_np(fc, l1, l2, l_out):
    key = (min(l1, l2), max(l1, l2))
    if l1 <= l2:
        outer = np.outer(fc[l1], fc[l2]).ravel()
    else:
        outer = np.outer(fc[l2], fc[l1]).ravel()
    cg = CG_NP[key]
    coupled = outer @ cg
    col_start = 0
    for lv in range(abs(l1 - l2), l1 + l2 + 1):
        sz = 2 * lv + 1
        if lv == l_out:
            proj = coupled[col_start:col_start + sz]
            return float(np.sum(np.abs(proj) ** 2))
        col_start += sz
    return 0.0


def get_live_bispec(lmax, witness):
    idx_map = _build_selective_index_map(lmax)
    fc = params_to_fc_np(witness, lmax)
    live = []
    for l1, l2, lv in idx_map:
        if max(l1, l2, lv) < 2:
            continue
        beta = bispectrum_entry_np(fc, l1, l2, lv)
        parity = (l1 + l2 + lv) % 2
        val = float(np.imag(beta)) if parity else float(np.real(beta))
        if abs(val) > 1e-10:
            live.append((l1, l2, lv))
    return live


def param_indices_for_ell(ell: int) -> list[int]:
    idx = 0
    for lv in range(2, ell):
        idx += 2 * lv + 1
        if lv == 2:
            idx -= 1
    n_params = 2 * ell + 1
    if ell == 2:
        n_params -= 1
    return list(range(idx, idx + n_params))


def eval_at_ell(full_params: np.ndarray, ell: int,
                live_bispec: list, cg_power: list, lmax: int) -> np.ndarray:
    """Evaluate constraints involving degree ell as the highest."""
    fc = params_to_fc_np(full_params, lmax)
    vals = []
    for l1, l2, lv in live_bispec:
        if max(l1, l2, lv) != ell:
            continue
        beta = bispectrum_entry_np(fc, l1, l2, lv)
        parity = (l1 + l2 + lv) % 2
        vals.append(float(np.imag(beta)) if parity else float(np.real(beta)))
    for l1, l2, lo in cg_power:
        if max(l1, l2) != ell:
            continue
        vals.append(cg_power_entry_np(fc, l1, l2, lo))
    return np.array(vals)


def damped_gauss_newton(residual_fn, x0, max_iter=60, tol=1e-24):
    """Gauss-Newton with backtracking line search."""
    n = len(x0)
    x = x0.copy()
    h = 1e-8

    for it in range(max_iter):
        f0 = residual_fn(x)
        r2 = np.sum(f0 ** 2)
        if r2 < tol:
            return x, r2

        m = len(f0)
        J = np.empty((m, n))
        for j in range(n):
            xp = x.copy(); xp[j] += h
            J[:, j] = (residual_fn(xp) - f0) / h

        try:
            dx = np.linalg.lstsq(J, -f0, rcond=None)[0]
        except np.linalg.LinAlgError:
            return x, r2

        alpha = 1.0
        for _ in range(20):
            x_new = x + alpha * dx
            r2_new = np.sum(residual_fn(x_new) ** 2)
            if r2_new < r2:
                x = x_new
                break
            alpha *= 0.5
        else:
            return x, r2

    return x, np.sum(residual_fn(x) ** 2)


def fiber_search_degree_by_degree(lmax: int, n_starts: int = 3000):
    """Search for all real solutions degree by degree."""
    _init_cg(lmax)
    n_free = sum(2 * l + 1 for l in range(2, lmax + 1)) - 1
    witness = np.array([(i + 1) / 3.0 for i in range(n_free)])

    live_bispec = get_live_bispec(lmax, witness)
    cg_power = _build_cg_power_index_map(lmax)

    print(f"lmax={lmax}: {n_free} params, "
          f"{len(live_bispec)} live bispec, {len(cg_power)} CG power", flush=True)

    all_results = {}

    for ell in range(2, lmax + 1):
        p_idx = param_indices_for_ell(ell)
        d_ell = len(p_idx)

        target = eval_at_ell(witness, ell, live_bispec, cg_power, lmax)
        n_constr = len(target)
        wit_params = witness[p_idx[0]:p_idx[-1] + 1].copy()

        print(f"\n  ell={ell}: {d_ell} unknowns, {n_constr} constraints", flush=True)

        def residual(t):
            p = witness.copy()
            for i, idx in enumerate(p_idx):
                p[idx] = t[i]
            return eval_at_ell(p, ell, live_bispec, cg_power, lmax) - target

        r_wit = residual(wit_params)
        print(f"    Witness residual: {np.sum(r_wit**2):.2e}", flush=True)

        solutions = []
        rng = np.random.RandomState(42 + ell)
        t0 = time.time()

        for trial in range(n_starts):
            scale = rng.choice([0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
            t_init = rng.randn(d_ell) * scale

            t_sol, res = damped_gauss_newton(residual, t_init, max_iter=50)

            if res < 1e-22:
                is_dup = any(np.linalg.norm(t_sol - s) < 1e-5 for s in solutions)
                if not is_dup:
                    solutions.append(t_sol.copy())
                    dist = np.linalg.norm(t_sol - wit_params)
                    print(f"    trial {trial}: NEW sol #{len(solutions)}, "
                          f"dist={dist:.2e}, res={res:.2e}", flush=True)

            if trial % 500 == 499:
                elapsed = time.time() - t0
                print(f"    ... {trial+1}/{n_starts} ({elapsed:.1f}s), "
                      f"{len(solutions)} sol(s)", flush=True)

        elapsed = time.time() - t0
        print(f"    {len(solutions)} solution(s) in {elapsed:.1f}s", flush=True)
        for i, s in enumerate(solutions):
            dist = np.linalg.norm(s - wit_params)
            if dist < 1e-5:
                print(f"      Sol {i}: WITNESS", flush=True)
            else:
                print(f"      Sol {i}: OTHER (dist={dist:.2e})", flush=True)
                print(f"        params = [{', '.join(f'{v:.6f}' for v in s)}]", flush=True)

        all_results[ell] = {
            'd_ell': d_ell,
            'n_constr': n_constr,
            'n_solutions': len(solutions),
            'solutions': solutions,
        }

    return all_results


def main():
    for lmax in [4, 5]:
        print(f"\n{'='*60}", flush=True)
        print(f"FIBER SEARCH: lmax = {lmax}", flush=True)
        print(f"{'='*60}", flush=True)

        results = fiber_search_degree_by_degree(lmax, n_starts=3000)

        print(f"\nSUMMARY lmax={lmax}:", flush=True)
        all_unique = True
        for ell, r in results.items():
            n_sol = r['n_solutions']
            status = "UNIQUE" if n_sol == 1 else f"MULTIPLE ({n_sol})"
            print(f"  ell={ell}: {status} "
                  f"({r['d_ell']} unknowns, {r['n_constr']} constraints)", flush=True)
            if n_sol != 1:
                all_unique = False

        if all_unique:
            print(f"  => Fiber at witness has UNIQUE real preimage", flush=True)
        else:
            print(f"  => Fiber has MULTIPLE real preimages", flush=True)


if __name__ == '__main__':
    main()
