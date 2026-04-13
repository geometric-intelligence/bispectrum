"""Verify the augmented selective bispectrum has full Jacobian rank.

Key insight: the scalar bispectrum has dead entries for real signals.
We augment with 5 CG-power-spectrum entries at the seed stage:
  P_{l1,l2,l} := ||(F_{l1} ⊗ F_{l2})|_l||^2
These are degree-4 SO(3)-invariants.

Budget analysis: we DROP the 2 dead entries and replace with 5 CG power
entries, giving a net increase of 3 at the seed stage. To compensate,
we reduce l=4 from 10 to 9 (standard budget) and also check if any l=5+
entries can be trimmed.
"""
import sys
sys.path.insert(0, '../src')
import torch
from bispectrum.so3_on_s2 import (
    _build_selective_index_map, _bispectrum_entry, load_cg_matrices,
)

CG_DATA = load_cg_matrices(8)
A0, C_GAUGE = 1.5, 1.2

AUG_ENTRIES = [(1,2,1), (1,3,2), (2,3,1), (2,3,2), (3,3,2)]


def params_to_fc(p, lmax):
    fc = {}
    fc[0] = torch.complex(torch.tensor([[A0]],dtype=torch.float64), torch.zeros(1,1,dtype=torch.float64))
    fc[1] = torch.complex(torch.tensor([[0.,C_GAUGE,0.]],dtype=torch.float64), torch.zeros(1,3,dtype=torch.float64))
    idx = 0
    for lv in range(2, lmax+1):
        neg, pos = [], []
        for m in range(1, lv+1):
            if lv==2 and m==1: re=p[idx]; idx+=1; im=torch.zeros_like(re)
            else: re=p[idx]; im=p[idx+1]; idx+=2
            pos.append(torch.complex(re,im)); neg.append(((-1.)**m)*torch.complex(re,-im))
        m0 = torch.complex(p[idx], torch.zeros_like(p[idx])); idx+=1
        fc[lv] = torch.stack(list(reversed(neg))+[m0]+pos).unsqueeze(0)
    return fc


def cg_project(fc, l1, l2, l_out):
    outer = torch.einsum('bi,bj->bij', fc[l1], fc[l2]).reshape(1,-1)
    cg = CG_DATA[(l1,l2)].to(dtype=outer.dtype)
    tr = outer @ cg
    start = 0
    for lv in range(abs(l1-l2), l1+l2+1):
        sz = 2*lv+1
        if lv == l_out: return tr[0, start:start+sz]
        start += sz


def eval_augmented(p, lmax, bispec_triples, aug_triples):
    fc = params_to_fc(p, lmax)
    vals = []
    for l1, l2, lv in bispec_triples:
        beta = _bispectrum_entry(fc, l1, l2, lv, CG_DATA[(l1,l2)])[0]
        parity = (l1+l2+lv) % 2
        vals.append(beta.imag if parity else beta.real)
    for l1, l2, lo in aug_triples:
        proj = cg_project(fc, l1, l2, lo)
        vals.append(torch.sum(proj.real**2 + proj.imag**2))
    return torch.stack(vals)


def check_dead(fc, l1, l2, lv):
    beta = _bispectrum_entry(fc, l1, l2, lv, CG_DATA[(l1,l2)])[0]
    parity = (l1+l2+lv) % 2
    val = beta.imag.item() if parity else beta.real.item()
    return abs(val) < 1e-10


def main():
    for lmax in [4, 5, 6, 7, 8]:
        n_free = sum(2*l+1 for l in range(2, lmax+1)) - 1
        torch.manual_seed(42)
        p = torch.randn(n_free, dtype=torch.float64)

        idx_map = _build_selective_index_map(lmax)
        fc_test = params_to_fc(p, lmax)

        live = []
        dead = []
        for l1, l2, lv in idx_map:
            if max(l1,l2,lv) < 2: continue
            if check_dead(fc_test, l1, l2, lv):
                dead.append((l1,l2,lv))
            else:
                live.append((l1,l2,lv))

        def eval_orig(p_in, trips=live, lm=lmax):
            return eval_augmented(p_in, lm, trips, [])

        def eval_aug(p_in, trips=live, aug=AUG_ENTRIES, lm=lmax):
            return eval_augmented(p_in, lm, trips, aug)

        Jo = torch.autograd.functional.jacobian(eval_orig, p)
        svo = torch.linalg.svdvals(Jo)
        ro = int((svo > 1e-8*svo[0]).sum().item())

        Ja = torch.autograd.functional.jacobian(eval_aug, p)
        sva = torch.linalg.svdvals(Ja)
        ra = int((sva > 1e-8*sva[0]).sum().item())

        total_orig = len(live) + 2  # +2 for l=0,1
        total_aug = total_orig + 5
        budget = (lmax+1)**2 - 3

        print(f"lmax={lmax}: unknowns={n_free}, dead={len(dead)}, live_entries={len(live)}")
        print(f"  Dead entries: {dead}")
        print(f"  Original:  {len(live)} entries → rank {ro}/{n_free}")
        print(f"  +5 CG aug: {len(live)+5} entries → rank {ra}/{n_free}")
        print(f"  Full rank: {'YES' if ra==n_free else 'NO'}")
        print(f"  Budget: orig={(lmax+1)**2-3}, with aug={total_aug}, overhead={total_aug-budget}")

        # Verify at 2 more seeds
        ranks_ok = True
        for s in [123, 999]:
            torch.manual_seed(s)
            p2 = torch.randn(n_free, dtype=torch.float64)
            Ja2 = torch.autograd.functional.jacobian(lambda pp: eval_aug(pp), p2)
            sva2 = torch.linalg.svdvals(Ja2)
            ra2 = int((sva2 > 1e-8*sva2[0]).sum().item())
            if ra2 != n_free: ranks_ok = False
        print(f"  Multi-seed: {'all full rank' if ranks_ok else 'FAILED'}")
        print()


if __name__ == "__main__":
    main()
