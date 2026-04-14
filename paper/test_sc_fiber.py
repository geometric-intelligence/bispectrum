"""Test: does adding self-coupling bispectral entries resolve the fiber ambiguity?"""
from __future__ import annotations
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from paper.fiber_search import (
    _init_cg, get_live_bispec, param_indices_for_ell,
    params_to_fc_np, bispectrum_entry_np, cg_power_entry_np,
    damped_gauss_newton,
)
from bispectrum.so3_on_s2 import _build_cg_power_index_map


def main():
    lmax = 4
    _init_cg(lmax)
    n = sum(2 * l + 1 for l in range(2, lmax + 1)) - 1
    w = np.array([(i + 1) / 3.0 for i in range(n)])
    live = get_live_bispec(lmax, w)
    cgp = _build_cg_power_index_map(lmax)

    sc_bispec = []
    fc_test = params_to_fc_np(w, lmax)
    for ell in range(2, lmax + 1):
        for l_out in range(0, min(2 * ell, lmax) + 1):
            if l_out > 2 * ell:
                continue
            try:
                beta = bispectrum_entry_np(fc_test, ell, ell, l_out)
                parity = (2 * ell + l_out) % 2
                val = float(np.imag(beta)) if parity else float(np.real(beta))
                if abs(val) > 1e-10:
                    already = (ell, ell, l_out) in live
                    if not already:
                        sc_bispec.append((ell, ell, l_out))
            except Exception:
                pass

    print(f"Added {len(sc_bispec)} self-coupling entries:")
    for t in sc_bispec:
        print(f"  B{t}")

    def eval_full(p):
        fc = params_to_fc_np(p, lmax)
        vals = []
        for l1, l2, lv in live:
            beta = bispectrum_entry_np(fc, l1, l2, lv)
            par = (l1 + l2 + lv) % 2
            vals.append(float(np.imag(beta)) if par else float(np.real(beta)))
        for l1, l2, lv in sc_bispec:
            beta = bispectrum_entry_np(fc, l1, l2, lv)
            par = (l1 + l2 + lv) % 2
            vals.append(float(np.imag(beta)) if par else float(np.real(beta)))
        for l1, l2, lo in cgp:
            vals.append(cg_power_entry_np(fc, l1, l2, lo))
        return np.array(vals)

    target = eval_full(w)
    print(f"Total: {len(target)} constraints")
    sys.stdout.flush()

    def res(x):
        return eval_full(x) - target

    rng = np.random.RandomState(42)
    found = []
    t0 = time.time()
    for trial in range(2000):
        eps = rng.choice([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
        x0 = w + eps * rng.randn(n)
        x, r = damped_gauss_newton(res, x0, max_iter=80)
        if r < 1e-22:
            dup = any(np.linalg.norm(x - s) < 1e-5 for s in found)
            if not dup:
                found.append(x.copy())
                d = np.linalg.norm(x - w)
                print(f"  t{trial}: sol #{len(found)}, dist={d:.2e}")
                sys.stdout.flush()
        if trial % 500 == 499:
            print(f"  {trial+1}/2000 ({time.time()-t0:.1f}s)")
            sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\nRESULT: {len(found)} solutions in {elapsed:.1f}s")
    for i, s in enumerate(found):
        d = np.linalg.norm(s - w)
        label = "WITNESS" if d < 1e-5 else f"DISTINCT (dist={d:.2e})"
        print(f"  Sol {i}: {label}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
