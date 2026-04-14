#!/usr/bin/env python
"""Verify that the augmented invariant (with self-coupling entries) has a
unique fiber at the rational witness for L=4 and L=5."""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from fiber_search import (
    _init_cg, get_live_bispec, params_to_fc_np,
    bispectrum_entry_np, cg_power_entry_np, damped_gauss_newton,
)
from bispectrum.so3_on_s2 import _build_cg_power_index_map


def verify_fiber(lmax: int, n_starts: int = 2000):
    _init_cg(lmax)
    n = sum(2 * l + 1 for l in range(2, lmax + 1)) - 1
    w = np.array([(i + 1) / 3.0 for i in range(n)])
    live = get_live_bispec(lmax, w)
    cgp = _build_cg_power_index_map(lmax)

    print(f"lmax={lmax}: {n} params, {len(live)} live bispec, {len(cgp)} CG power")
    sc_in_live = [t for t in live if t[0] == t[1] and t[0] >= 3]
    print(f"  Self-coupling in live: {sc_in_live}")

    def eval_full(p):
        fc = params_to_fc_np(p, lmax)
        vals = []
        for l1, l2, lv in live:
            beta = bispectrum_entry_np(fc, l1, l2, lv)
            par = (l1 + l2 + lv) % 2
            vals.append(float(np.imag(beta)) if par else float(np.real(beta)))
        for l1, l2, lo in cgp:
            vals.append(cg_power_entry_np(fc, l1, l2, lo))
        return np.array(vals)

    target = eval_full(w)
    print(f"  Total constraints: {len(target)}")
    sys.stdout.flush()

    def res(x):
        return eval_full(x) - target

    rng = np.random.RandomState(42)
    found = []
    t0 = time.time()
    for trial in range(n_starts):
        eps = rng.choice([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
        x0 = w + eps * rng.randn(n)
        x, r = damped_gauss_newton(res, x0, max_iter=80)
        if r < 1e-22:
            dup = any(np.linalg.norm(x - s) < 1e-5 for s in found)
            if not dup:
                found.append(x.copy())
                d = np.linalg.norm(x - w)
                label = "WITNESS" if d < 1e-5 else f"DISTINCT dist={d:.2e}"
                print(f"    trial {trial}: sol #{len(found)} ({label})")
                sys.stdout.flush()
        if trial % 500 == 499:
            print(f"    {trial+1}/{n_starts} ({time.time()-t0:.1f}s)")
            sys.stdout.flush()

    elapsed = time.time() - t0
    n_distinct = sum(1 for s in found if np.linalg.norm(s - w) > 1e-5)
    print(f"\n  RESULT lmax={lmax}: {len(found)} solutions ({n_distinct} distinct from witness) in {elapsed:.1f}s")
    if n_distinct == 0:
        print(f"  => UNIQUE FIBER (global injectivity)")
    else:
        print(f"  => MULTIPLE PREIMAGES (global injectivity FAILS)")
    sys.stdout.flush()
    return n_distinct == 0


if __name__ == "__main__":
    ok4 = verify_fiber(4, n_starts=2000)
    print()
    ok5 = verify_fiber(5, n_starts=1000)
    print(f"\n{'='*40}")
    print(f"L=4: {'PASS' if ok4 else 'FAIL'}")
    print(f"L=5: {'PASS' if ok5 else 'FAIL'}")
