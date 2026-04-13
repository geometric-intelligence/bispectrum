"""Verify bootstrap rank using autograd — no manual matrix construction."""
import sys
sys.path.insert(0, '../src')
import torch
from bispectrum.so3_on_s2 import (
    _build_selective_index_map, _bispectrum_entry, load_cg_matrices,
)

CG_DATA = load_cg_matrices(8)
A0, C_GAUGE = 1.5, 1.2


def params_to_fc(p_fixed, p_target, l_target, lmax):
    """Build coefficients: p_fixed for degrees 0..l_target-1 (frozen),
    p_target for degree l_target (free)."""
    fc = {}
    fc[0] = torch.complex(torch.tensor([[A0]], dtype=torch.float64), torch.zeros(1,1, dtype=torch.float64))
    fc[1] = torch.complex(torch.tensor([[0.,C_GAUGE,0.]], dtype=torch.float64), torch.zeros(1,3, dtype=torch.float64))
    idx = 0
    for lv in range(2, lmax+1):
        n_real = 2*lv+1 if lv != 2 else 2*lv  # gauge: Im(a_2^1)=0
        if lv < l_target:
            raw = p_fixed[idx:idx+n_real]
            idx += n_real
        elif lv == l_target:
            raw = p_target
        else:
            raw = p_fixed[idx:idx+n_real]
            idx += n_real
        neg, pos = [], []
        j = 0
        for m in range(1, lv+1):
            if lv==2 and m==1:
                re = raw[j]; j += 1; im = torch.zeros_like(re)
            else:
                re = raw[j]; im = raw[j+1]; j += 2
            pos.append(torch.complex(re,im)); neg.append(((-1.)**m)*torch.complex(re,-im))
        m0 = torch.complex(raw[j], torch.zeros_like(raw[j])); j += 1
        fc[lv] = torch.stack(list(reversed(neg))+[m0]+pos).unsqueeze(0)
    return fc


def main():
    for lmax in [4, 5, 6, 7]:
        idx_map = _build_selective_index_map(lmax)
        print(f"lmax={lmax}:")

        for l_target in range(4, lmax+1):
            n_target = 2*l_target+1 if l_target != 2 else 2*l_target
            n_fixed = sum(2*l+1 if l!=2 else 2*l for l in range(2, lmax+1)) - n_target

            torch.manual_seed(42)
            p_fixed = torch.randn(n_fixed, dtype=torch.float64)
            p_target = torch.randn(n_target, dtype=torch.float64)

            triples = [(l1,l2,lv) for l1,l2,lv in idx_map if max(l1,l2,lv)==l_target]

            def eval_bootstrap(p_t, trips=triples, lt=l_target, lm=lmax, pf=p_fixed):
                fc = params_to_fc(pf, p_t, lt, lm)
                vals = []
                for l1,l2,lv in trips:
                    beta = _bispectrum_entry(fc, l1, l2, lv, CG_DATA[(l1,l2)])[0]
                    # Use BOTH real and imag (full complex info)
                    vals.append(beta.real)
                    vals.append(beta.imag)
                return torch.stack(vals)

            def eval_bootstrap_parity(p_t, trips=triples, lt=l_target, lm=lmax, pf=p_fixed):
                fc = params_to_fc(pf, p_t, lt, lm)
                vals = []
                for l1,l2,lv in trips:
                    beta = _bispectrum_entry(fc, l1, l2, lv, CG_DATA[(l1,l2)])[0]
                    parity = (l1+l2+lv) % 2
                    vals.append(beta.imag if parity else beta.real)
                return torch.stack(vals)

            # Complex Jacobian (real+imag parts)
            J_full = torch.autograd.functional.jacobian(eval_bootstrap, p_target)
            sv_full = torch.linalg.svdvals(J_full)
            tol = 1e-8 * sv_full[0].item() if sv_full[0]>0 else 1e-15
            rank_full = int((sv_full > tol).sum().item())

            # Parity-corrected Jacobian (real part)
            J_par = torch.autograd.functional.jacobian(eval_bootstrap_parity, p_target)
            sv_par = torch.linalg.svdvals(J_par)
            tol2 = 1e-8 * sv_par[0].item() if sv_par[0]>0 else 1e-15
            rank_par = int((sv_par > tol2).sum().item())

            # Check for dead entries
            fc_test = params_to_fc(p_fixed, p_target, l_target, lmax)
            dead = []
            for l1,l2,lv in triples:
                beta = _bispectrum_entry(fc_test, l1, l2, lv, CG_DATA[(l1,l2)])[0]
                if abs(beta.item()) < 1e-10:
                    dead.append((l1,l2,lv))

            print(f"  ℓ={l_target}: {len(triples)} entries, "
                  f"full(Re+Im) rank={rank_full}/{n_target}, "
                  f"parity rank={rank_par}/{n_target}, "
                  f"dead={dead}")

        print()


if __name__ == "__main__":
    main()
