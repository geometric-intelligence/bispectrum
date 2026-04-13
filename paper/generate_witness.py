"""Generate witness points proving full Jacobian rank for the augmented invariant.

For each lmax in [4..8], outputs:
1. The witness signal (rational coefficients)
2. The augmented entry set (bispec + CG power)
3. The Jacobian rank at the witness
"""
import sys, os, json
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


def get_rank_and_sv(bispec, cg_ents, params, lmax):
    def ev(p):
        return eval_combined(p, bispec, cg_ents, lmax)
    J = torch.autograd.functional.jacobian(ev, params)
    sv = torch.linalg.svdvals(J)
    tol = 1e-8 * sv[0].item() if sv[0] > 0 else 1e-15
    return int((sv > tol).sum().item()), sv


def greedy_augment(live_bispec, params, lmax, n_free):
    cg_candidates = []
    for l1 in range(lmax + 1):
        for l2 in range(l1, lmax + 1):
            if max(l1, l2) < 2: continue
            for lo in range(abs(l1 - l2), min(l1 + l2, lmax) + 1):
                if l1 == 0 and lo == l2: continue
                cg_candidates.append((l1, l2, lo))

    best_aug = []
    best_rank, _ = get_rank_and_sv(live_bispec, [], params, lmax)

    for _ in range(n_free - best_rank):
        best_cand, best_r = None, best_rank
        for cand in cg_candidates:
            if cand in best_aug: continue
            r, _ = get_rank_and_sv(live_bispec, best_aug + [cand], params, lmax)
            if r > best_r:
                best_r = r
                best_cand = cand
        if best_cand is None: break
        best_aug.append(best_cand)
        best_rank = best_r
        if best_rank >= n_free: break

    return best_aug, best_rank


def main():
    results = {}
    outpath = os.path.join(os.path.dirname(__file__), 'witness_results.txt')
    f = open(outpath, 'w')

    for lmax in [4, 5]:
        n_free = sum(2 * l + 1 for l in range(2, lmax + 1)) - 1

        # Use simple rational-ish witness: p_i = (i+1)/3
        params = torch.tensor([(i+1)/3.0 for i in range(n_free)], dtype=torch.float64)

        idx_map = _build_selective_index_map(lmax)
        fc_test = params_to_fc(params, lmax)

        live_bispec = []
        dead_bispec = []
        for l1, l2, lv in idx_map:
            if max(l1, l2, lv) < 2: continue
            beta = _bispectrum_entry(fc_test, l1, l2, lv, CG_DATA[(l1, l2)])[0]
            parity = (l1 + l2 + lv) % 2
            val = beta.imag.item() if parity else beta.real.item()
            if abs(val) > 1e-10:
                live_bispec.append((l1, l2, lv))
            else:
                dead_bispec.append((l1, l2, lv))

        r0, sv0 = get_rank_and_sv(live_bispec, [], params, lmax)
        f.write(f"lmax={lmax}: {n_free} unknowns\n")
        f.write(f"  Live bispec: {len(live_bispec)}, dead: {dead_bispec}\n")
        f.write(f"  Bispec-only rank: {r0}/{n_free}\n")

        aug, rfinal = greedy_augment(live_bispec, params, lmax, n_free)
        f.write(f"  Augmentation: {len(aug)} CG power entries\n")
        for a in aug:
            f.write(f"    P_{{{a[0]},{a[1]},{a[2]}}}\n")
        f.write(f"  Final rank: {rfinal}/{n_free}\n")

        # Verify at 3 random seeds
        for seed in [42, 123, 999]:
            torch.manual_seed(seed)
            p2 = torch.randn(n_free, dtype=torch.float64)
            r2, _ = get_rank_and_sv(live_bispec, aug, p2, lmax)
            f.write(f"  seed={seed}: rank={r2}\n")

        f.write(f"\n  Witness signal: p = [{', '.join(f'{v:.4f}' for v in params.tolist())}]\n\n")
        f.flush()

        results[lmax] = {
            'n_free': n_free,
            'live_bispec': live_bispec,
            'dead_bispec': dead_bispec,
            'augmentation': aug,
            'bispec_only_rank': r0,
            'augmented_rank': rfinal,
        }

    f.close()
    print(f"Results: {outpath}")


if __name__ == "__main__":
    main()
