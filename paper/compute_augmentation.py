"""Compute the per-degree CG power augmentation needed and verify full rank.

For each degree ℓ, find the minimum set of CG power entries
||(F_{l1} ⊗ F_ℓ)|_{l_out}||^2 that fills the rank gap.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
from bispectrum.so3_on_s2 import (
    _build_selective_index_map, _bispectrum_entry, load_cg_matrices,
)

CG_DATA = load_cg_matrices(10)
A0, C_GAUGE = 1.5, 1.2

def params_to_fc(p, lmax):
    fc = {}
    fc[0] = torch.complex(torch.tensor([[A0]], dtype=torch.float64),
                           torch.zeros(1, 1, dtype=torch.float64))
    fc[1] = torch.complex(torch.tensor([[0., C_GAUGE, 0.]], dtype=torch.float64),
                           torch.zeros(1, 3, dtype=torch.float64))
    idx = 0
    for lv in range(2, lmax + 1):
        neg, pos = [], []
        for m in range(1, lv + 1):
            if lv == 2 and m == 1:
                re = p[idx]; idx += 1; im = torch.zeros_like(re)
            else:
                re = p[idx]; im = p[idx + 1]; idx += 2
            pos.append(torch.complex(re, im))
            neg.append(((-1.) ** m) * torch.complex(re, -im))
        m0 = torch.complex(p[idx], torch.zeros_like(p[idx])); idx += 1
        fc[lv] = torch.stack(list(reversed(neg)) + [m0] + pos).unsqueeze(0)
    return fc

def cg_project(fc, l1, l2, l_out):
    outer = torch.einsum('bi,bj->bij', fc[l1], fc[l2]).reshape(1, -1)
    cg = CG_DATA[(l1, l2)].to(dtype=outer.dtype)
    tr = outer @ cg
    start = 0
    for lv in range(abs(l1 - l2), l1 + l2 + 1):
        sz = 2 * lv + 1
        if lv == l_out: return tr[0, start:start + sz]
        start += sz

def eval_combined(p, bispec_triples, cg_entries, lmax):
    fc = params_to_fc(p, lmax)
    vals = []
    for l1, l2, lv in bispec_triples:
        beta = _bispectrum_entry(fc, l1, l2, lv, CG_DATA[(l1, l2)])[0]
        parity = (l1 + l2 + lv) % 2
        vals.append(beta.imag if parity else beta.real)
    for l1, l2, lo in cg_entries:
        proj = cg_project(fc, l1, l2, lo)
        vals.append(torch.sum(proj.real ** 2 + proj.imag ** 2))
    return torch.stack(vals)


def main():
    results = []
    outpath = os.path.join(os.path.dirname(__file__), 'augmentation_results.txt')
    f = open(outpath, 'w')

    for lmax in range(4, 9):
        n_free = sum(2 * l + 1 for l in range(2, lmax + 1)) - 1
        torch.manual_seed(42)
        params = torch.randn(n_free, dtype=torch.float64)

        idx_map = _build_selective_index_map(lmax)
        fc_test = params_to_fc(params, lmax)

        live_bispec = []
        for l1, l2, lv in idx_map:
            if max(l1, l2, lv) < 2: continue
            beta = _bispectrum_entry(fc_test, l1, l2, lv, CG_DATA[(l1, l2)])[0]
            parity = (l1 + l2 + lv) % 2
            val = beta.imag.item() if parity else beta.real.item()
            if abs(val) > 1e-10:
                live_bispec.append((l1, l2, lv))

        # Candidate CG power entries: ||(F_{l1} ⊗ F_{l2})|_lo||^2
        # Only include entries that involve degrees in [2, lmax]
        cg_candidates = []
        for l1 in range(lmax + 1):
            for l2 in range(l1, lmax + 1):
                if max(l1, l2) < 2: continue
                if l1 == 0 and True:  # skip (0,l,l) — just ||F_l||^2, redundant with power spec
                    pass
                for lo in range(abs(l1 - l2), min(l1 + l2, lmax) + 1):
                    if l1 == 0 and lo == l2: continue
                    cg_candidates.append((l1, l2, lo))

        # Greedy augmentation
        def get_rank(bispec, cg_ents, p=params, lm=lmax):
            def ev(pp):
                return eval_combined(pp, bispec, cg_ents, lm)
            J = torch.autograd.functional.jacobian(ev, p)
            sv = torch.linalg.svdvals(J)
            tol = 1e-8 * sv[0].item() if sv[0] > 0 else 1e-15
            return int((sv > tol).sum().item())

        r0 = get_rank(live_bispec, [])
        f.write(f"lmax={lmax}: {n_free} unknowns, {len(live_bispec)} live bispec, rank={r0}\n")

        best_aug = []
        best_rank = r0
        remaining = list(cg_candidates)
        for rnd in range(n_free - best_rank):
            best_cand = None
            best_r = best_rank
            for cand in remaining:
                r = get_rank(live_bispec, best_aug + [cand])
                if r > best_r:
                    best_r = r
                    best_cand = cand
            if best_cand is None:
                f.write(f"  round {rnd+1}: no improvement, stopping\n")
                break
            best_aug.append(best_cand)
            remaining.remove(best_cand)
            best_rank = best_r
            f.write(f"  +||(F_{best_cand[0]}⊗F_{best_cand[1]})|_{best_cand[2]}||^2 → rank={best_rank}\n")
            if best_rank >= n_free:
                f.write(f"  FULL RANK with {len(best_aug)} augmentations!\n")
                break

        # Verify at 2 more seeds
        ok = True
        for seed in [123, 999]:
            torch.manual_seed(seed)
            p2 = torch.randn(n_free, dtype=torch.float64)
            r2 = get_rank(live_bispec, best_aug, p2, lmax)
            if r2 < n_free:
                ok = False
                f.write(f"  seed={seed}: rank={r2} FAIL\n")
        if ok:
            f.write(f"  multi-seed: verified ✓\n")

        total = len(live_bispec) + 2 + len(best_aug)  # +2 for l=0,1
        budget = (lmax + 1) ** 2 - 3
        f.write(f"  total entries: {total} (budget was {budget}, overhead {total-budget})\n\n")
        f.flush()

    f.close()
    print(f"Results: {outpath}")


if __name__ == "__main__":
    main()
