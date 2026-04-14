"""Deep debug of bootstrap rank — compare real vs complex signals,
gauge-fixed vs ungauged, and identify which m-components are unconstrained."""
import sys
sys.path.insert(0, '../src')
import torch
from bispectrum.so3_on_s2 import (
    _build_selective_index_map, _bispectrum_entry, load_cg_matrices,
)

CG_DATA = load_cg_matrices(8)

def make_complex_fc(lmax, seed=42):
    """Fully generic complex signal (no reality constraint, no gauge)."""
    torch.manual_seed(seed)
    fc = {}
    for l in range(lmax+1):
        r = torch.randn(1, 2*l+1, dtype=torch.float64)
        i = torch.randn(1, 2*l+1, dtype=torch.float64)
        fc[l] = torch.complex(r, i)
    return fc


def eval_bootstrap_complex(params, l_target, fc_lower, triples, lmax):
    """Evaluate bootstrap entries with F_{l_target} parametrized by arbitrary complex params."""
    fc = dict(fc_lower)
    fc[l_target] = torch.complex(
        params[:2*l_target+1].unsqueeze(0),
        params[2*l_target+1:].unsqueeze(0)
    )
    vals = []
    for l1,l2,lv in triples:
        beta = _bispectrum_entry(fc, l1, l2, lv, CG_DATA[(l1,l2)])[0]
        vals.append(beta.real)
        vals.append(beta.imag)
    return torch.stack(vals)


def eval_bootstrap_real(params, l_target, fc_lower, triples, lmax):
    """Evaluate bootstrap entries with real-signal F_{l_target}."""
    fc = dict(fc_lower)
    neg, pos = [], []
    j = 0
    for m in range(1, l_target+1):
        re = params[j]; im = params[j+1]; j += 2
        pos.append(torch.complex(re, im))
        neg.append(((-1.)**m)*torch.complex(re, -im))
    m0 = torch.complex(params[j], torch.zeros_like(params[j])); j += 1
    fc[l_target] = torch.stack(list(reversed(neg))+[m0]+pos).unsqueeze(0)
    vals = []
    for l1,l2,lv in triples:
        beta = _bispectrum_entry(fc, l1, l2, lv, CG_DATA[(l1,l2)])[0]
        vals.append(beta.real)
        vals.append(beta.imag)
    return torch.stack(vals)


def compute_rank(J, label=""):
    sv = torch.linalg.svdvals(J)
    tol = 1e-8 * sv[0].item() if sv[0] > 0 else 1e-15
    r = int((sv > tol).sum().item())
    return r, sv


def main():
    idx_map = _build_selective_index_map(8)

    print("=" * 70)
    print("Bootstrap rank: COMPLEX signals (2*(2l+1) real params per F_l)")
    print("=" * 70)

    for l_target in range(4, 8):
        fc_full = make_complex_fc(8, seed=42)
        fc_lower = {l: fc_full[l] for l in range(l_target)}
        triples = [(l1,l2,lv) for l1,l2,lv in idx_map if max(l1,l2,lv)==l_target]

        n_params = 2 * (2*l_target+1)
        torch.manual_seed(42)
        params = torch.randn(n_params, dtype=torch.float64)

        J = torch.autograd.functional.jacobian(
            lambda p: eval_bootstrap_complex(p, l_target, fc_lower, triples, 8), params)
        r, sv = compute_rank(J)
        print(f"  ℓ={l_target}: {len(triples)} entries × 2 = {2*len(triples)} rows, "
              f"{n_params} params, rank={r}")

    print("\n" + "=" * 70)
    print("Bootstrap rank: REAL signal, NO gauge (2l+1 real params)")
    print("=" * 70)

    for l_target in range(4, 8):
        torch.manual_seed(42)
        # Build real lower-degree coefficients
        fc_lower = {}
        for l in range(l_target):
            neg, pos = [], []
            for m in range(1, l+1):
                re = torch.randn(1, dtype=torch.float64)
                im = torch.randn(1, dtype=torch.float64)
                pos.append(torch.complex(re, im))
                neg.append(((-1.)**m)*torch.complex(re, -im))
            m0 = torch.complex(torch.randn(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
            fc_lower[l] = torch.stack(list(reversed(neg))+[m0]+pos).unsqueeze(0)

        triples = [(l1,l2,lv) for l1,l2,lv in idx_map if max(l1,l2,lv)==l_target]
        n_params = 2*l_target + 1
        params = torch.randn(n_params, dtype=torch.float64)

        J = torch.autograd.functional.jacobian(
            lambda p: eval_bootstrap_real(p, l_target, fc_lower, triples, 8), params)
        r, sv = compute_rank(J)
        print(f"  ℓ={l_target}: {len(triples)} entries, "
              f"{n_params} real params, rank={r}")

    print("\n" + "=" * 70)
    print("Bootstrap rank: REAL signal, GAUGE-FIXED (F_1=(0,c,0), Im(a_2^1)=0)")
    print("=" * 70)

    for l_target in range(4, 8):
        torch.manual_seed(42)
        fc_lower = {}
        fc_lower[0] = torch.complex(torch.tensor([[1.5]], dtype=torch.float64),
                                     torch.zeros(1,1, dtype=torch.float64))
        fc_lower[1] = torch.complex(torch.tensor([[0.,1.2,0.]], dtype=torch.float64),
                                     torch.zeros(1,3, dtype=torch.float64))
        for l in range(2, l_target):
            neg, pos = [], []
            for m in range(1, l+1):
                re = torch.randn(1, dtype=torch.float64)
                if l == 2 and m == 1:
                    im = torch.zeros(1, dtype=torch.float64)
                else:
                    im = torch.randn(1, dtype=torch.float64)
                pos.append(torch.complex(re, im))
                neg.append(((-1.)**m)*torch.complex(re, -im))
            m0 = torch.complex(torch.randn(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
            fc_lower[l] = torch.stack(list(reversed(neg))+[m0]+pos).unsqueeze(0)

        triples = [(l1,l2,lv) for l1,l2,lv in idx_map if max(l1,l2,lv)==l_target]
        n_params = 2*l_target + 1
        params = torch.randn(n_params, dtype=torch.float64)

        J = torch.autograd.functional.jacobian(
            lambda p: eval_bootstrap_real(p, l_target, fc_lower, triples, 8), params)
        r, sv = compute_rank(J)
        print(f"  ℓ={l_target}: {len(triples)} entries, "
              f"{n_params} real params, rank={r}")

    print("\n" + "=" * 70)
    print("FULL bispectrum bootstrap rank (all valid triples)")
    print("=" * 70)

    for l_target in [4, 5, 6]:
        torch.manual_seed(42)
        fc_lower = {}
        fc_lower[0] = torch.complex(torch.tensor([[1.5]], dtype=torch.float64),
                                     torch.zeros(1,1, dtype=torch.float64))
        fc_lower[1] = torch.complex(torch.tensor([[0.,1.2,0.]], dtype=torch.float64),
                                     torch.zeros(1,3, dtype=torch.float64))
        for l in range(2, l_target):
            neg, pos = [], []
            for m in range(1, l+1):
                re = torch.randn(1, dtype=torch.float64)
                if l==2 and m==1: im = torch.zeros(1, dtype=torch.float64)
                else: im = torch.randn(1, dtype=torch.float64)
                pos.append(torch.complex(re, im))
                neg.append(((-1.)**m)*torch.complex(re, -im))
            m0 = torch.complex(torch.randn(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
            fc_lower[l] = torch.stack(list(reversed(neg))+[m0]+pos).unsqueeze(0)

        all_triples = []
        for l1 in range(l_target+1):
            for l2 in range(l1, l_target+1):
                for lv in range(abs(l1-l2), min(l1+l2, l_target)+1):
                    if max(l1,l2,lv) == l_target:
                        all_triples.append((l1,l2,lv))

        n_params = 2*l_target + 1
        params = torch.randn(n_params, dtype=torch.float64)

        J = torch.autograd.functional.jacobian(
            lambda p: eval_bootstrap_real(p, l_target, fc_lower, all_triples, l_target), params)
        r, sv = compute_rank(J)
        print(f"  ℓ={l_target}: {len(all_triples)} entries (all valid), "
              f"{n_params} real params, rank={r}")


if __name__ == "__main__":
    main()
