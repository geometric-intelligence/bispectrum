"""Systematic analysis of seed-stage Jacobian rank for the SO(3) bispectrum.

Tests which degree-4 augmentation invariants can fix the rank deficiency.
"""
import sys
sys.path.insert(0, '../src')

import torch
from bispectrum.so3_on_s2 import (
    _build_selective_index_map, _bispectrum_entry, load_cg_matrices,
)
from itertools import combinations

LMAX = 6
CG_DATA = load_cg_matrices(LMAX)

A0 = 1.5
C_GAUGE = 1.2


def make_gauge_fixed_params(n_seed_params: int = 11, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n_seed_params, dtype=torch.float64)


def params_to_f23(p: torch.Tensor) -> dict[int, torch.Tensor]:
    """Build F_0 (fixed), F_1 (gauge-fixed), F_2, F_3 from 11 free params."""
    fc: dict[int, torch.Tensor] = {}
    fc[0] = torch.complex(
        torch.tensor([[A0]], dtype=torch.float64),
        torch.zeros(1, 1, dtype=torch.float64),
    )
    fc[1] = torch.complex(
        torch.tensor([[0.0, C_GAUGE, 0.0]], dtype=torch.float64),
        torch.zeros(1, 3, dtype=torch.float64),
    )
    idx = 0
    for l_val in [2, 3]:
        neg_parts: list[torch.Tensor] = []
        pos_parts: list[torch.Tensor] = []
        for m in range(1, l_val + 1):
            if l_val == 2 and m == 1:
                re = p[idx]; idx += 1
                im = torch.zeros_like(re)
            else:
                re = p[idx]; im = p[idx + 1]; idx += 2
            pos_parts.append(torch.complex(re, im))
            neg_parts.append(((-1.0) ** m) * torch.complex(re, -im))
        m0 = torch.complex(p[idx], torch.zeros_like(p[idx])); idx += 1
        coeffs = list(reversed(neg_parts)) + [m0] + pos_parts
        fc[l_val] = torch.stack(coeffs).unsqueeze(0)
    return fc


def cg_project(fc: dict[int, torch.Tensor], l1: int, l2: int, l_out: int) -> torch.Tensor:
    """Compute (F_{l1} ⊗ F_{l2})|_{l_out} as a (2l_out+1) vector."""
    f_l1 = fc[l1]
    f_l2 = fc[l2]
    outer = torch.einsum('bi,bj->bij', f_l1, f_l2).reshape(1, -1)
    cg = CG_DATA[(l1, l2)].to(dtype=outer.dtype)
    transformed = outer @ cg  # shape (1, sum of 2l+1 for valid l)
    start = 0
    for lv in range(abs(l1 - l2), l1 + l2 + 1):
        sz = 2 * lv + 1
        if lv == l_out:
            return transformed[0, start:start + sz]
        start += sz
    raise ValueError(f"l_out={l_out} not in coupling range of {l1}⊗{l2}")


def eval_scalar_bispectrum(p: torch.Tensor, triples: list[tuple[int, int, int]]) -> torch.Tensor:
    """Evaluate scalar bispectral entries using parity-correct component."""
    fc = params_to_f23(p)
    vals = []
    for l1, l2, lv in triples:
        beta = _bispectrum_entry(fc, l1, l2, lv, CG_DATA[(l1, l2)])[0]
        parity = (l1 + l2 + lv) % 2
        vals.append(beta.imag if parity else beta.real)
    return torch.stack(vals)


def eval_cg_power(p: torch.Tensor, quadruples: list[tuple[int, int, int, int]]) -> torch.Tensor:
    """Evaluate CG power spectrum entries: ||(F_{l1}⊗F_{l2})|_l||^2.

    Also supports cross terms: <(F_{l1}⊗F_{l2})|_l, (F_{l3}⊗F_{l4})|_l>.
    quadruples: (l1, l2, l3_or_l, l4_or_sentinel)
    If l4 == -1: power entry ||(F_{l1}⊗F_{l2})|_{l3}||^2
    Otherwise: cross <(F_{l1}⊗F_{l2})|_{l4}, (F_{l3}⊗F_{l2})|_{l4}>... 
    """
    fc = params_to_f23(p)
    vals = []
    for entry in quadruples:
        if len(entry) == 3:
            l1, l2, l_out = entry
            proj = cg_project(fc, l1, l2, l_out)
            vals.append(torch.sum(proj.real ** 2 + proj.imag ** 2))
        else:
            l1a, l2a, l1b, l2b, l_out = entry
            proj_a = cg_project(fc, l1a, l2a, l_out)
            proj_b = cg_project(fc, l1b, l2b, l_out)
            val = torch.sum(proj_a * torch.conj(proj_b))
            parity = (l1a + l2a + l1b + l2b) % 2
            vals.append(val.imag if parity else val.real)
    return torch.stack(vals)


def compute_rank(eval_fn, params: torch.Tensor, tol_factor: float = 1e-8) -> tuple[int, torch.Tensor]:
    J = torch.autograd.functional.jacobian(eval_fn, params)
    sv = torch.linalg.svdvals(J)
    tol = tol_factor * sv[0].item() if sv[0].item() > 0 else 1e-15
    rank = int((sv > tol).sum().item())
    return rank, sv


def main():
    params = make_gauge_fixed_params(seed=42)
    n_unknowns = 11  # 4 (F_2) + 7 (F_3)

    idx_map = _build_selective_index_map(LMAX)
    seed_triples = [(l1, l2, lv) for l1, l2, lv in idx_map if max(l1, l2, lv) <= 3 and max(l1, l2, lv) >= 2]

    print("=" * 70)
    print("PART 1: Current scalar bispectrum seed rank")
    print("=" * 70)

    def eval_seed(p):
        return eval_scalar_bispectrum(p, seed_triples)

    rank, sv = compute_rank(eval_seed, params)
    print(f"Seed triples: {seed_triples}")
    print(f"Count: {len(seed_triples)}, Rank: {rank}, Unknowns: {n_unknowns}")
    print(f"SV: {[f'{s:.3e}' for s in sv.tolist()]}")

    nonzero_triples = []
    for l1, l2, lv in seed_triples:
        fc_test = params_to_f23(params)
        beta = _bispectrum_entry(fc_test, l1, l2, lv, CG_DATA[(l1, l2)])[0]
        parity = (l1 + l2 + lv) % 2
        val = beta.imag.item() if parity else beta.real.item()
        status = "ZERO" if abs(val) < 1e-10 else "ok"
        print(f"  ({l1},{l2},{lv}): parity={parity}, val={val:.4e} [{status}]")
        if abs(val) >= 1e-10:
            nonzero_triples.append((l1, l2, lv))

    def eval_nonzero(p):
        return eval_scalar_bispectrum(p, nonzero_triples)
    rank_nz, _ = compute_rank(eval_nonzero, params)
    print(f"\nNonzero-only: {len(nonzero_triples)} entries, rank={rank_nz}")

    print("\n" + "=" * 70)
    print("PART 2: Available CG power spectrum augmentations")
    print("=" * 70)

    cg_power_candidates = []
    for l1 in range(4):
        for l2 in range(l1, 4):
            for l_out in range(abs(l1 - l2), min(l1 + l2, LMAX) + 1):
                if l1 == 0 and l2 == l_out:
                    continue  # this is just ||F_{l2}||^2, already have power spectrum
                if max(l1, l2) < 2:
                    continue  # only F_0, F_1 — already known, no new info
                cg_power_candidates.append((l1, l2, l_out))

    print(f"CG power spectrum candidates: {len(cg_power_candidates)}")
    for cand in cg_power_candidates:
        l1, l2, l_out = cand
        def eval_single(p, c=cand):
            return eval_cg_power(p, [c])
        try:
            fc_t = params_to_f23(params)
            proj = cg_project(fc_t, l1, l2, l_out)
            val = torch.sum(proj.real**2 + proj.imag**2).item()
        except Exception:
            val = float('nan')
        print(f"  ||(F_{l1}⊗F_{l2})|_{l_out}||^2 = {val:.6f}")

    print("\n" + "=" * 70)
    print("PART 3: Greedy search for minimal augmentation set")
    print("=" * 70)

    best_rank = rank_nz
    best_set: list[tuple[int, int, int]] = []
    remaining = list(cg_power_candidates)

    for round_num in range(n_unknowns - best_rank):
        best_cand = None
        best_new_rank = best_rank
        for cand in remaining:
            def eval_combined(p, extras=[cand]):
                v1 = eval_scalar_bispectrum(p, nonzero_triples)
                v2 = eval_cg_power(p, best_set + extras)
                return torch.cat([v1, v2])
            try:
                r, _ = compute_rank(eval_combined, params)
            except Exception:
                continue
            if r > best_new_rank:
                best_new_rank = r
                best_cand = cand
        if best_cand is None:
            print(f"  Round {round_num+1}: no improvement found, stopping")
            break
        best_set.append(best_cand)
        remaining.remove(best_cand)
        best_rank = best_new_rank
        l1, l2, lo = best_cand
        print(f"  Round {round_num+1}: added ||(F_{l1}⊗F_{l2})|_{lo}||^2 → rank={best_rank}")
        if best_rank >= n_unknowns:
            print(f"  FULL RANK ACHIEVED with {len(best_set)} augmentations!")
            break

    print(f"\nFinal augmentation set ({len(best_set)} entries):")
    for l1, l2, lo in best_set:
        print(f"  ||(F_{l1}⊗F_{l2})|_{lo}||^2")

    print(f"\nTotal entries: {len(nonzero_triples)} scalar bispec + {len(best_set)} CG power = {len(nonzero_triples)+len(best_set)}")
    print(f"Final rank: {best_rank} / {n_unknowns} unknowns")

    print("\n" + "=" * 70)
    print("PART 4: Cross-term augmentations (degree-4 bilinear)")
    print("=" * 70)

    cross_candidates = []
    for l1a in range(4):
        for l2a in range(l1a, 4):
            for l1b in range(4):
                for l2b in range(l1b, 4):
                    if (l1a, l2a) >= (l1b, l2b):
                        continue
                    for l_out in range(max(abs(l1a-l2a), abs(l1b-l2b)),
                                       min(l1a+l2a, l1b+l2b, LMAX)+1):
                        if l_out not in range(abs(l1a-l2a), l1a+l2a+1):
                            continue
                        if l_out not in range(abs(l1b-l2b), l1b+l2b+1):
                            continue
                        if max(l1a, l2a, l1b, l2b) < 2:
                            continue
                        cross_candidates.append((l1a, l2a, l1b, l2b, l_out))

    print(f"Cross-term candidates: {len(cross_candidates)}")

    best_rank_2 = rank_nz
    best_set_2: list = []
    remaining_all = list(cg_power_candidates) + [(c[0], c[1], c[4]) if len(c)==5 else c for c in []] 

    all_candidates = [(c, 'power') for c in cg_power_candidates] + [(c, 'cross') for c in cross_candidates]

    for round_num in range(n_unknowns - best_rank_2):
        best_cand = None
        best_type = None
        best_new_rank = best_rank_2
        for cand, ctype in all_candidates:
            if cand in [x[0] for x in best_set_2]:
                continue
            def eval_combined(p, extra_cand=cand, extra_type=ctype):
                v1 = eval_scalar_bispectrum(p, nonzero_triples)
                power_entries = [x[0] for x in best_set_2 if x[1] == 'power']
                cross_entries = [x[0] for x in best_set_2 if x[1] == 'cross']
                if extra_type == 'power':
                    power_entries = power_entries + [extra_cand]
                else:
                    cross_entries = cross_entries + [extra_cand]
                parts = [v1]
                if power_entries:
                    parts.append(eval_cg_power(p, power_entries))
                if cross_entries:
                    parts.append(eval_cg_power(p, cross_entries))
                return torch.cat(parts)
            try:
                r, _ = compute_rank(eval_combined, params)
            except Exception:
                continue
            if r > best_new_rank:
                best_new_rank = r
                best_cand = cand
                best_type = ctype
        if best_cand is None:
            print(f"  Round {round_num+1}: no improvement, stopping")
            break
        best_set_2.append((best_cand, best_type))
        best_rank_2 = best_new_rank
        print(f"  Round {round_num+1}: added {best_type} {best_cand} → rank={best_rank_2}")
        if best_rank_2 >= n_unknowns:
            print(f"  FULL RANK ACHIEVED!")
            break

    print(f"\nFinal combined set ({len(best_set_2)} entries):")
    for cand, ctype in best_set_2:
        print(f"  [{ctype}] {cand}")
    print(f"Final rank: {best_rank_2} / {n_unknowns}")

    print("\n" + "=" * 70)
    print("PART 5: Verify at multiple random seeds")
    print("=" * 70)
    final_augmentations = best_set if best_rank >= n_unknowns else [x[0] for x in best_set_2 if x[1] == 'power']
    for seed in [42, 123, 999, 7, 2024]:
        p = make_gauge_fixed_params(seed=seed)
        def eval_final(p_inner):
            v1 = eval_scalar_bispectrum(p_inner, nonzero_triples)
            if final_augmentations:
                v2 = eval_cg_power(p_inner, final_augmentations)
                return torch.cat([v1, v2])
            return v1
        r, sv = compute_rank(eval_final, p)
        gap = sv[r-1].item() / max(sv[min(r, len(sv)-1)].item(), 1e-20)
        print(f"  seed={seed}: rank={r}, gap={gap:.2e}")


if __name__ == "__main__":
    main()
