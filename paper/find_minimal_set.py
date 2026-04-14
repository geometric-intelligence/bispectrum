"""Find the minimal augmentation set and verify full-system rank."""
import sys
sys.path.insert(0, '../src')
import torch
from bispectrum.so3_on_s2 import (
    _build_selective_index_map, _bispectrum_entry, load_cg_matrices,
)
from itertools import combinations

SEED_LMAX = 3
CG_DATA = load_cg_matrices(8)
A0, C_GAUGE = 1.5, 1.2

def make_seed_params(seed=42):
    torch.manual_seed(seed)
    return torch.randn(11, dtype=torch.float64)

def params_to_fc_seed(p):
    fc = {}
    fc[0] = torch.complex(torch.tensor([[A0]], dtype=torch.float64), torch.zeros(1,1,dtype=torch.float64))
    fc[1] = torch.complex(torch.tensor([[0.,C_GAUGE,0.]], dtype=torch.float64), torch.zeros(1,3,dtype=torch.float64))
    idx = 0
    for lv in [2, 3]:
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

def eval_entry(p, entry):
    fc = params_to_fc_seed(p)
    if entry[0] == 'B':
        _, l1, l2, lv = entry
        beta = _bispectrum_entry(fc, l1, l2, lv, CG_DATA[(l1,l2)])[0]
        return beta.imag if (l1+l2+lv)%2 else beta.real
    else:
        _, l1, l2, lo = entry
        proj = cg_project(fc, l1, l2, lo)
        return torch.sum(proj.real**2 + proj.imag**2)

def eval_set(p, entries):
    return torch.stack([eval_entry(p, e) for e in entries])

def rank_of(entries, params):
    J = torch.autograd.functional.jacobian(lambda p: eval_set(p, entries), params)
    sv = torch.linalg.svdvals(J)
    tol = 1e-8 * sv[0].item() if sv[0]>0 else 1e-15
    return int((sv > tol).sum().item()), sv

ALL_BISPEC = [
    ('B',1,1,2),('B',0,2,2),('B',2,2,2),('B',1,2,3),
    ('B',1,3,2),('B',2,3,1),('B',0,3,3),('B',3,3,2),
]
ALL_CG = [
    ('P',1,2,1),('P',1,2,2),('P',1,2,3),
    ('P',1,3,2),('P',1,3,3),('P',1,3,4),
    ('P',2,2,0),('P',2,2,2),('P',2,2,4),
    ('P',2,3,1),('P',2,3,2),('P',2,3,3),('P',2,3,4),('P',2,3,5),
    ('P',3,3,0),('P',3,3,2),('P',3,3,4),('P',3,3,6),
]

def main():
    params = make_seed_params(42)

    print("BASELINE: 8 nonzero scalar bispec")
    r, _ = rank_of(ALL_BISPEC, params)
    print(f"  rank = {r} / 11\n")

    print("SEARCH: n_bispec + n_cg = N, find min N with rank 11")
    for N in range(11, 14):
        found = False
        for nb in range(max(0, N-len(ALL_CG)), min(len(ALL_BISPEC), N)+1):
            nc = N - nb
            if nc < 0 or nc > len(ALL_CG): continue
            for bc in combinations(ALL_BISPEC, nb):
                for cc in combinations(ALL_CG, nc):
                    entries = list(bc) + list(cc)
                    try:
                        r, _ = rank_of(entries, params)
                    except Exception:
                        continue
                    if r == 11:
                        ok = all(rank_of(entries, make_seed_params(s))[0]==11 for s in [123,999,7])
                        if ok:
                            print(f"  N={N}: {nb} bispec + {nc} cg → rank 11 ✓")
                            for e in entries: print(f"    {e}")
                            found = True
                            break
                if found: break
            if found: break
        if found:
            break
        else:
            print(f"  N={N}: no rank-11 set found")

    print("\n" + "="*70)
    print("FULL SYSTEM RANK with augmented seed")
    print("="*70)

    aug = [('P',1,2,1),('P',1,3,2),('P',2,3,1),('P',2,3,2),('P',3,3,2)]

    for lmax in [4, 5, 6]:
        n_free = sum(2*l+1 for l in range(2, lmax+1)) - 1
        torch.manual_seed(42)
        params_full = torch.randn(n_free, dtype=torch.float64)

        def params_to_fc_full(p, lm=lmax):
            fc = {}
            fc[0] = torch.complex(torch.tensor([[A0]],dtype=torch.float64), torch.zeros(1,1,dtype=torch.float64))
            fc[1] = torch.complex(torch.tensor([[0.,C_GAUGE,0.]],dtype=torch.float64), torch.zeros(1,3,dtype=torch.float64))
            idx = 0
            for lv in range(2, lm+1):
                neg, pos = [], []
                for m in range(1, lv+1):
                    if lv==2 and m==1: re=p[idx]; idx+=1; im=torch.zeros_like(re)
                    else: re=p[idx]; im=p[idx+1]; idx+=2
                    pos.append(torch.complex(re,im)); neg.append(((-1.)**m)*torch.complex(re,-im))
                m0 = torch.complex(p[idx], torch.zeros_like(p[idx])); idx+=1
                fc[lv] = torch.stack(list(reversed(neg))+[m0]+pos).unsqueeze(0)
            return fc

        idx_map = _build_selective_index_map(lmax)

        entries = []
        for l1,l2,lv in idx_map:
            if max(l1,l2,lv) < 2: continue
            fc_t = params_to_fc_full(params_full)
            beta = _bispectrum_entry(fc_t, l1, l2, lv, CG_DATA[(l1,l2)])[0]
            parity = (l1+l2+lv) % 2
            val = beta.imag.item() if parity else beta.real.item()
            if abs(val) > 1e-10:
                entries.append(('B', l1, l2, lv))

        def eval_full_orig(p, ents=entries, lm=lmax):
            fc = params_to_fc_full(p, lm)
            vals = []
            for e in ents:
                _, l1, l2, lv = e
                beta = _bispectrum_entry(fc, l1, l2, lv, CG_DATA[(l1,l2)])[0]
                parity = (l1+l2+lv) % 2
                vals.append(beta.imag if parity else beta.real)
            return torch.stack(vals)

        def eval_full_aug(p, ents=entries, a=aug, lm=lmax):
            fc = params_to_fc_full(p, lm)
            vals = []
            for e in ents:
                _, l1, l2, lv = e
                beta = _bispectrum_entry(fc, l1, l2, lv, CG_DATA[(l1,l2)])[0]
                parity = (l1+l2+lv) % 2
                vals.append(beta.imag if parity else beta.real)
            for e in a:
                _, l1, l2, lo = e
                proj = cg_project(fc, l1, l2, lo)
                vals.append(torch.sum(proj.real**2 + proj.imag**2))
            return torch.stack(vals)

        Jo = torch.autograd.functional.jacobian(eval_full_orig, params_full)
        svo = torch.linalg.svdvals(Jo)
        ro = int((svo > 1e-8*svo[0]).sum().item())

        Ja = torch.autograd.functional.jacobian(eval_full_aug, params_full)
        sva = torch.linalg.svdvals(Ja)
        ra = int((sva > 1e-8*sva[0]).sum().item())

        print(f"\nlmax={lmax}: {n_free} unknowns")
        print(f"  Original: {len(entries)} entries, rank={ro}")
        print(f"  +5 CG aug: {len(entries)+5} entries, rank={ra}")
        print(f"  FULL RANK: {'YES ✓' if ra==n_free else 'NO ✗'}")

if __name__ == "__main__":
    main()
