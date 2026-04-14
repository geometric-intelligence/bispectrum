#!/usr/bin/env python
"""Tree-based fiber enumeration for L=4 and L=5.

At each degree ell, given fixed lower-degree coefficients from a parent
branch, uses multi-start damped Gauss-Newton to find ALL per-degree
solutions. Propagates each solution to the next degree. At the final
degree, validates each leaf against the full system.

Only the witness path should survive the full-system check, proving
fiber degree = 1.
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from fiber_search import _init_cg, get_live_bispec, CG_NP, A0, C_GAUGE
from bispectrum.so3_on_s2 import _build_cg_power_index_map


def _build_fc(params, lmax):
    fc = {}
    fc[0] = np.array([A0 + 0j])
    fc[1] = np.array([0.0 + 0j, C_GAUGE + 0j, 0.0 + 0j])
    idx = 0
    for lv in range(2, lmax + 1):
        c = np.zeros(2 * lv + 1, dtype=np.complex128)
        for m in range(1, lv + 1):
            if lv == 2 and m == 1:
                c[lv + m] = params[idx]; c[lv - m] = ((-1)**m) * params[idx]; idx += 1
            else:
                c[lv + m] = params[idx] + 1j * params[idx + 1]
                c[lv - m] = ((-1)**m) * (params[idx] - 1j * params[idx + 1])
                idx += 2
        c[lv] = params[idx] + 0j; idx += 1
        fc[lv] = c
    return fc


def _fc_to_t(fc, ell):
    c = fc[ell]
    parts = []
    for m in range(1, ell + 1):
        if ell == 2 and m == 1:
            parts.append(c[ell + m].real)
        else:
            parts.extend([c[ell + m].real, c[ell + m].imag])
    parts.append(c[ell].real)
    return np.array(parts)


def _update_fc(fc, ell, t):
    c = np.zeros(2 * ell + 1, dtype=np.complex128)
    idx = 0
    for m in range(1, ell + 1):
        if ell == 2 and m == 1:
            c[ell + m] = t[idx]; c[ell - m] = ((-1)**m) * t[idx]; idx += 1
        else:
            c[ell + m] = t[idx] + 1j * t[idx + 1]
            c[ell - m] = ((-1)**m) * (t[idx] - 1j * t[idx + 1])
            idx += 2
    c[ell] = t[idx] + 0j
    fc[ell] = c


def _bispec(fc, l1, l2, lv):
    key = (min(l1, l2), max(l1, l2))
    outer = np.outer(fc[min(l1, l2)], fc[max(l1, l2)]).ravel()
    coupled = outer @ CG_NP[key]
    col = 0
    for l in range(abs(l1 - l2), l1 + l2 + 1):
        sz = 2 * l + 1
        if l == lv:
            return np.sum(coupled[col:col + sz] * np.conj(fc[lv]))
        col += sz
    return 0.0 + 0j


def _cgpow(fc, l1, l2, lo):
    key = (min(l1, l2), max(l1, l2))
    outer = np.outer(fc[min(l1, l2)], fc[max(l1, l2)]).ravel()
    coupled = outer @ CG_NP[key]
    col = 0
    for l in range(abs(l1 - l2), l1 + l2 + 1):
        sz = 2 * l + 1
        if l == lo:
            return float(np.sum(np.abs(coupled[col:col + sz]) ** 2))
        col += sz
    return 0.0


def _pdim(ell):
    return (2 * ell + 1) - (1 if ell == 2 else 0)


def _pstart(ell):
    idx = 0
    for lv in range(2, ell):
        idx += _pdim(lv)
    return idx


def _eval_at(fc, ell, bispec_e, cgp_e, target):
    vals = []
    for l1, l2, lv in bispec_e:
        beta = _bispec(fc, l1, l2, lv)
        par = (l1 + l2 + lv) % 2
        vals.append(float(np.imag(beta)) if par else float(np.real(beta)))
    for l1, l2, lo in cgp_e:
        vals.append(_cgpow(fc, l1, l2, lo))
    return np.array(vals) - target


def _gn_search(fc_parent, ell, bispec_e, cgp_e, target, d, wit_t, n_starts):
    """Multi-start damped GN. Returns list of distinct solutions."""
    h = 1e-8
    solutions = []
    fc = {k: v.copy() for k, v in fc_parent.items()}
    fc_orig_ell = fc_parent.get(ell, np.zeros(2 * ell + 1, dtype=np.complex128)).copy()

    rng = np.random.RandomState(42 + ell + abs(hash(fc_parent[2].tobytes())) % 10000)
    wit_scale = max(np.linalg.norm(wit_t), 1.0)

    for trial in range(n_starts):
        if trial < 5:
            t = wit_t + rng.randn(d) * 0.01 * wit_scale
        elif trial < 30:
            t = wit_t + rng.randn(d) * rng.choice([0.01, 0.1, 0.5]) * wit_scale
        elif trial < 60:
            t = wit_t * (1.0 + rng.randn(d) * rng.choice([0.1, 0.3, 0.5]))
        else:
            scale = rng.choice([0.01, 0.1, 0.5, 1.0, 2.0, 5.0,
                                wit_scale * 0.5, wit_scale, wit_scale * 2])
            t = rng.randn(d) * scale

        for it in range(50):
            _update_fc(fc, ell, t)
            f0 = _eval_at(fc, ell, bispec_e, cgp_e, target)
            r2 = np.sum(f0**2)

            if r2 < 1e-16:
                is_dup = any(np.linalg.norm(t - s) < 1e-4 for s in solutions)
                if not is_dup:
                    solutions.append(t.copy())
                break

            if it >= 10 and r2 > 1e-4:
                break

            J = np.empty((len(f0), d))
            for j in range(d):
                tp = t.copy(); tp[j] += h
                _update_fc(fc, ell, tp)
                J[:, j] = (_eval_at(fc, ell, bispec_e, cgp_e, target) - f0) / h

            try:
                dx = np.linalg.lstsq(J, -f0, rcond=None)[0]
            except np.linalg.LinAlgError:
                break

            alpha = 1.0
            for _ in range(15):
                tn = t + alpha * dx
                _update_fc(fc, ell, tn)
                r2n = np.sum(_eval_at(fc, ell, bispec_e, cgp_e, target)**2)
                if r2n < r2:
                    t = tn
                    break
                alpha *= 0.5
            else:
                break

        fc[ell] = fc_orig_ell.copy()

    return solutions


def tree_search(lmax, n_starts_det=2000, n_starts_over=500):
    _init_cg(lmax)
    n = sum(2 * l + 1 for l in range(2, lmax + 1)) - 1
    witness = np.array([(i + 1) / 3.0 for i in range(n)])
    live = get_live_bispec(lmax, witness)
    cgp = _build_cg_power_index_map(lmax)

    sc = [t for t in live if t[0] == t[1] and t[0] >= 3]
    print(f"lmax={lmax}: n={n}, {len(live)} bispec, {len(cgp)} CG power")
    print(f"  Self-coupling: {sc}")

    fc_wit = _build_fc(witness, lmax)
    full_tgt = []
    for l1, l2, lv in live:
        beta = _bispec(fc_wit, l1, l2, lv)
        par = (l1 + l2 + lv) % 2
        full_tgt.append(float(np.imag(beta)) if par else float(np.real(beta)))
    for l1, l2, lo in cgp:
        full_tgt.append(_cgpow(fc_wit, l1, l2, lo))
    full_tgt = np.array(full_tgt)

    branches = [witness.copy()]
    t0_all = time.time()

    for ell in range(2, lmax + 1):
        d = _pdim(ell)
        ps = _pstart(ell)
        bispec_e = [(l1, l2, lv) for l1, l2, lv in live if max(l1, l2, lv) == ell]
        cgp_e = [(l1, l2, lo) for l1, l2, lo in cgp if max(l1, l2) == ell]
        m_ell = len(bispec_e) + len(cgp_e)
        n_starts = n_starts_det if m_ell <= d + 2 else n_starts_over
        wit_t = _fc_to_t(fc_wit, ell)

        tgt_ell = []
        for l1, l2, lv in bispec_e:
            beta = _bispec(fc_wit, l1, l2, lv)
            par = (l1 + l2 + lv) % 2
            tgt_ell.append(float(np.imag(beta)) if par else float(np.real(beta)))
        for l1, l2, lo in cgp_e:
            tgt_ell.append(_cgpow(fc_wit, l1, l2, lo))
        tgt_ell = np.array(tgt_ell)

        print(f"\n  ell={ell}: d={d} m={m_ell} {len(branches)}br {n_starts}starts")
        sys.stdout.flush()
        t0 = time.time()

        new_branches = []
        for bi, parent in enumerate(branches):
            dist = np.linalg.norm(parent - witness)
            lbl = "wit" if dist < 1e-4 else f"alt d={dist:.1e}"

            fc_p = _build_fc(parent, lmax)
            sols = _gn_search(fc_p, ell, bispec_e, cgp_e, tgt_ell, d, wit_t, n_starts)

            for s in sols:
                child = parent.copy()
                child[ps:ps + d] = s
                new_branches.append(child)

            nw = sum(1 for s in sols if np.linalg.norm(s - wit_t) < 1e-4)
            na = len(sols) - nw
            elapsed_b = time.time() - t0
            print(f"    b{bi} ({lbl}): {len(sols)} ({nw}w {na}a) [{elapsed_b:.0f}s]")
            sys.stdout.flush()

        elapsed = time.time() - t0
        branches = new_branches
        print(f"    => {len(branches)} branches ({elapsed:.0f}s)")
        sys.stdout.flush()

    print(f"\n  Checking {len(branches)} leaves...")
    valid = []
    for b in branches:
        fc_b = _build_fc(b, lmax)
        vals = []
        for l1, l2, lv in live:
            beta = _bispec(fc_b, l1, l2, lv)
            par = (l1 + l2 + lv) % 2
            vals.append(float(np.imag(beta)) if par else float(np.real(beta)))
        for l1, l2, lo in cgp:
            vals.append(_cgpow(fc_b, l1, l2, lo))
        r2 = np.sum((np.array(vals) - full_tgt)**2)
        dist = np.linalg.norm(b - witness)
        if r2 < 1e-10:
            valid.append(b)
            lbl = "WITNESS" if dist < 1e-3 else f"DISTINCT d={dist:.2e}"
            print(f"    valid: r2={r2:.2e} {lbl}")

    total = time.time() - t0_all
    nd = sum(1 for v in valid if np.linalg.norm(v - witness) > 1e-3)
    print(f"\n  RESULT lmax={lmax}: {len(valid)} valid, {nd} distinct ({total:.0f}s)")
    if nd == 0 and len(valid) > 0:
        print("  => UNIQUE FIBER")
    elif len(valid) == 0:
        print("  => ERROR: witness not found")
    else:
        print("  => MULTIPLE PREIMAGES")
    sys.stdout.flush()
    return nd == 0 and len(valid) > 0


if __name__ == "__main__":
    res = {}
    for lmax in [4, 5]:
        print(f"\n{'='*60}")
        print(f"TREE FIBER SEARCH: lmax={lmax}")
        print(f"{'='*60}")
        sys.stdout.flush()
        res[lmax] = tree_search(lmax, n_starts_det=2000, n_starts_over=500)

    print(f"\n{'='*60}")
    print("SUMMARY")
    for lmax, ok in res.items():
        print(f"  L={lmax}: {'PASS' if ok else 'FAIL'}")
