# Research Log: Fixing the SO(3) Bispectrum Completeness Proof

## Problem Statement

The scalar SO(3) bispectrum β_{l1,l2,l} = <(F_{l1}⊗F_{l2})|_l, F_l> has
fundamental Jacobian rank deficiency for real signals on S^2:

- Seed Jacobian (10×11) has rank 6 (not 10 as claimed)
- Two entries (β_{2,2,3}, β_{2,3,2}) are identically zero for real signals
- Additional algebraic dependencies reduce effective rank further
- Even the FULL bispectrum has rank 15 (not 22) for lmax=4

Root cause: for real signals, conj(β_{l1,l2,l}) = (-1)^{l1+l2-l} β,
making entries with odd parity purely imaginary. Entries with l_v ∈ {l1,l2}
and the complementary degree odd vanish identically due to CG cancellation
under the reality constraint |a_l^{-m}|^2 = |a_l^m|^2.

## Approaches Under Investigation

### Approach A: Augment scalar bispectrum with degree-4 "CG power spectrum" invariants

Replace dead/dependent entries with:
  P_{l1,l2,l} := ||(F_{l1}⊗F_{l2})|_l||^2
These are degree-4 SO(3)-invariant and carry information the scalar
bispectrum misses.

### Approach B: Switch to unsummed / matrix-valued bispectrum

Use vector-valued entries instead of scalar contractions. Higher rank
but larger output.

## Critical Finding: Bootstrap Rank for Real vs Complex Signals

Complex rank of bootstrap matrix A (solving A conj(F_ℓ) = β):

| ℓ | COMPLEX signal rank | REAL signal rank | Expected |
|---|---|---|---|
| 4 | 9 | 4 | 9 |
| 5 | 11 | 4 | 11 |
| 6 | 13 | 8 | 13 |
| 7 | 15 | 8 | 15 |
| 8 | 17 | 7 | 17 |

Root cause: the reality constraint a_ℓ^{-m} = (-1)^m conj(a_ℓ^m) creates
algebraic dependencies in A that are NOT present for complex signals.
The Zariski density argument ("real signals are Zariski-dense in C^n")
fails because the reality constraint involves conjugation, which is not
algebraic over C.

Even the FULL bispectrum (all valid triples) has the same rank deficiency.
This is intrinsic to the scalar SO(3) bispectrum for real signals.

## Solution Strategy

Augment the scalar bispectrum with "CG power spectrum" entries at every
degree:
  P_{l1,l2,ℓ} := ||(F_{l1} ⊗ F_{l2})|_ℓ||^2  (degree-4 invariant)
where l1 or l2 = ℓ (so that F_ℓ is involved).

These are QUADRATIC in F_ℓ (when the other factor is known), providing
additional constraints that fill the rank gap.

The augmented invariant set has O(L^2) entries (still matching the
orbit-space dimension up to constant factor).

## Augmentation Verification

Greedy augmentation with CG power entries achieves full Jacobian rank:

### lmax=4 (20 unknowns)
- 18 live scalar bispec entries → rank 10
- +10 CG power entries → rank 20 (FULL RANK) ✓
- Total: 30 entries (overhead: +8 over budget 22)

### lmax=5 (31 unknowns)
- 28 live scalar bispec entries → rank 14
- +17 CG power entries → rank 31 (FULL RANK) ✓
- Total: 47 entries (overhead: +14 over budget 33)

Pattern: the rank deficit is ~50% of unknowns. CG power entries fill it.
Total output: ~1.5 × (L+1)^2 entries = O(L^2), still quadratic.

The augmentation entries follow a systematic rule:
- For each degree ℓ ≥ 2 and each known degree l1 < ℓ:
  add ||(F_{l1} ⊗ F_ℓ)|_{l_out}||^2 for l_out = |l1-ℓ|, ..., l1+ℓ
  (up to budget constraints)

## Proof Strategy

The augmented invariant Φ_aug = (β_sel, P_sel) maps R^n → R^m with:
- β_sel: selective scalar bispectral entries (degree 3)
- P_sel: CG power spectrum entries (degree 4)

Jacobian rank of Φ_aug is n-3 = dim(orbit space) at generic signals.
By the rank theorem, generic fibers are 3-dimensional.
Since SO(3) orbits are 3-dimensional and contained in fibers,
the generic fiber IS the orbit. Hence Φ_aug separates orbits.

## Experiment Log
