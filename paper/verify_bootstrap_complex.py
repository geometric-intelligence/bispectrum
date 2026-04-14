"""Verify the COMPLEX rank of the bootstrap coefficient matrix for real signals.

Key insight: the Jacobian rank deficit is about the REAL map β: R^n → R^n.
But the bootstrap uses COMPLEX linear algebra: A F_ℓ = b where A ∈ C^{N×(2ℓ+1)}.
The complex rank of A can be 2ℓ+1 even when the real Jacobian is rank-deficient.
"""
import sys
sys.path.insert(0, '../src')
import torch
from bispectrum.so3_on_s2 import (
    _build_selective_index_map, _bispectrum_entry, load_cg_matrices,
)

CG_DATA = load_cg_matrices(8)
A0, C_GAUGE = 1.5, 1.2


def make_real_fc(lmax, seed=42):
    torch.manual_seed(seed)
    fc = {}
    for l in range(lmax + 1):
        pp, pn = [], []
        for m in range(1, l + 1):
            re = torch.randn(1, dtype=torch.float64)
            im = torch.randn(1, dtype=torch.float64)
            pp.append(torch.complex(re, im))
            pn.append(((-1.0) ** m) * torch.complex(re, -im))
        m0 = torch.complex(torch.randn(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
        fc[l] = torch.stack(list(reversed(pn)) + [m0] + pp, dim=-1)
    return fc


def build_bootstrap_matrix(fc, l_target, triples):
    """Build the complex coefficient matrix A for the bootstrap at l_target.

    For chain entries (l1,l2,l_target): F_{l_target} appears as conj(a_l^m).
    For cross entries (l1,l_target,l_v): F_{l_target} appears as a_{l_target}^{m2}.
    """
    n_cols = 2 * l_target + 1
    rows_conj = []  # entries where conj(F_l) appears
    rows_direct = []  # entries where F_l appears directly

    for l1, l2, lv in triples:
        if lv == l_target and l2 < l_target:
            # Chain: (l1, l2, l_target), linear in conj(a_{l_target}^m)
            row = torch.zeros(n_cols, dtype=torch.complex128)
            for m in range(-l_target, l_target + 1):
                col_idx = m + l_target
                val = torch.tensor(0.0, dtype=torch.complex128)
                for m1 in range(-l1, l1 + 1):
                    m2 = m - m1
                    if abs(m2) > l2:
                        continue
                    cg_val = _get_cg(l1, m1, l2, m2, l_target, m)
                    val += cg_val * fc[l1][0, m1 + l1] * fc[l2][0, m2 + l2]
                row[col_idx] = val
            rows_conj.append(row)

        elif l2 == l_target and lv < l_target:
            # Cross: (l1, l_target, lv), linear in a_{l_target}^{m2}
            row = torch.zeros(n_cols, dtype=torch.complex128)
            for m2 in range(-l_target, l_target + 1):
                col_idx = m2 + l_target
                val = torch.tensor(0.0, dtype=torch.complex128)
                for m1 in range(-l1, l1 + 1):
                    m = m1 + m2
                    if abs(m) > lv:
                        continue
                    cg_val = _get_cg(l1, m1, l_target, m2, lv, m)
                    val += cg_val * fc[l1][0, m1 + l1] * torch.conj(fc[lv][0, m + lv])
                row[col_idx] = val
            rows_direct.append(row)

        elif lv == l_target and l2 >= l_target:
            pass  # self-coupling, skip for now

    # For chain entries, the system is: row · conj(F_l) = beta
    # For cross entries, the system is: row · F_l = beta
    # These are different: one uses conj(F_l), the other F_l.
    # For a unified system, convert conj rows: if row · conj(F) = b,
    # then conj(row) · F = conj(b), so we can use conj(row) for direct system.
    A_rows = []
    for row in rows_conj:
        A_rows.append(torch.conj(row))
    for row in rows_direct:
        A_rows.append(row)

    if not A_rows:
        return torch.zeros(0, n_cols, dtype=torch.complex128)
    return torch.stack(A_rows)


def _get_cg(l1, m1, l2, m2, l, m):
    """Get CG coefficient from precomputed data."""
    if m1 + m2 != m:
        return 0.0
    if l < abs(l1 - l2) or l > l1 + l2:
        return 0.0
    cg_mat = CG_DATA[(l1, l2)]
    row = (m1 + l1) * (2 * l2 + 1) + (m2 + l2)
    start = 0
    for lv in range(abs(l1 - l2), l1 + l2 + 1):
        if lv == l:
            col = start + m + l
            return cg_mat[row, col].item()
        start += 2 * lv + 1
    return 0.0


def main():
    for lmax in [4, 5, 6, 7, 8]:
        idx_map = _build_selective_index_map(lmax)

        print(f"lmax={lmax}:")
        for l_target in range(4, lmax + 1):
            triples = [(l1, l2, lv) for l1, l2, lv in idx_map
                       if (lv == l_target and l2 < l_target) or
                       (l2 == l_target and lv < l_target)]
            expected_rank = 2 * l_target + 1

            # Test at multiple seeds
            ranks = []
            for seed in [42, 123, 999]:
                fc = make_real_fc(lmax, seed)
                A = build_bootstrap_matrix(fc, l_target, triples)
                if A.shape[0] == 0:
                    ranks.append(0)
                    continue
                sv = torch.linalg.svdvals(A)
                tol = 1e-8 * sv[0].item() if sv[0] > 0 else 1e-15
                rank = int((sv > tol).sum().item())
                ranks.append(rank)

            min_r = min(ranks)
            status = "✓" if min_r >= expected_rank else "✗"
            print(f"  ℓ={l_target}: {len(triples)} entries, "
                  f"complex rank={min_r}/{expected_rank} {status}")

    # Also check: does the dead entry contribute a zero row?
    print("\nDead entry row check:")
    for l_target, dead_triple in [(5, (4, 5, 4)), (7, (6, 7, 6))]:
        if l_target > 8: continue
        fc = make_real_fc(l_target, 42)
        A = build_bootstrap_matrix(fc, l_target, [dead_triple])
        if A.shape[0] > 0:
            row_norm = torch.linalg.norm(A[0]).item()
            print(f"  {dead_triple}: ||row||={row_norm:.6e}")
        else:
            print(f"  {dead_triple}: not a bootstrap entry for ℓ={l_target}")


if __name__ == "__main__":
    main()
