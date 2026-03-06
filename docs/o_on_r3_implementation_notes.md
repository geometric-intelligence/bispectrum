# OonR3 Implementation Notes

## Overview

`OonR3` implements the selective G-bispectrum for the octahedral group O (the rotation symmetry group of the cube, |O| = 24), following Algorithm 3 of Mataigne et al. (2024).

## Group Data

The 24 rotation matrices of O are generated from two generators via BFS:

- `R_z(π/2)`: 90° rotation around the z-axis
- `R_{[1,1,1]}(2π/3)`: 120° rotation around the [1,1,1] body diagonal

The elements are sorted lexicographically (flattened 3×3 matrix entries) for a canonical ordering. The identity element ends up at index 23 in this ordering, not index 0.

All group data (elements, Cayley table, inverse map, Kronecker table) was generated using `escnn` and cross-verified against the analytical construction.

## Irreducible Representations

O has 5 real irreps with dimensions (1, 3, 3, 2, 1):

| Irrep   | Dim | Construction                                  |
| ------- | --- | --------------------------------------------- |
| ρ₀ (A₁) | 1   | Trivial: ρ₀(g) = 1                            |
| ρ₁ (T₁) | 3   | Standard: ρ₁(g) = the 3×3 rotation matrix     |
| ρ₂ (T₂) | 3   | Product: ρ₂(g) = ρ₄(g) · ρ₁(g)                |
| ρ₃ (E)  | 2   | Hardcoded from escnn (involves ±1/2, ±√3/2)   |
| ρ₄ (A₂) | 1   | Alternating sign of body-diagonal permutation |

ρ₃ is the only irrep that requires non-integer matrix entries. The 24 matrices are hardcoded as exact rational/irrational values.

## Clebsch-Gordan Matrices

CG matrices are computed via the projection operator method:

```
P_k = (d_k / |G|) Σ_g χ_k(g)* (ρ_i(g) ⊗ ρ_j(g))
```

The eigenvectors of P_k (with eigenvalue 1) form the columns of C corresponding to irrep k. The full CG matrix C is assembled by concatenating these columns in order.

Two CG matrices are needed for the selective bispectrum:

- C₁₁ for ρ₁ ⊗ ρ₁ = ρ₀ ⊕ ρ₁ ⊕ ρ₂ ⊕ ρ₃ (9×9)
- C₁₂ for ρ₁ ⊗ ρ₂ = ρ₁ ⊕ ρ₂ ⊕ ρ₃ ⊕ ρ₄ (9×9)

## DFT Convention

The group Fourier transform uses the left-multiplication convention:

```
F(ρ_k) = Σ_g f(g) · ρ_k(g)
```

Under the group action T_h f(g) = f(h⁻¹g), this gives:

```
F(T_h f)(ρ_k) = ρ_k(h) · F(f)(ρ_k)
```

The inverse DFT is:

```
f(g) = (1/|G|) Σ_k d_k · tr(ρ_k(g)^T · F_k)
```

## Forward Pass

The selective bispectrum computes 4 matrix coefficients (172 scalars total):

| Pair (i,j) | Scalars | Formula                     |
| ---------- | ------- | --------------------------- |
| (ρ₀,ρ₀)    | 1       | F₀³                         |
| (ρ₀,ρ₁)    | 9       | F₀ · F₁ᵀF₁                  |
| (ρ₁,ρ₁)    | 81      | C₁₁ (⊕ Fₖᵀ) C₁₁ᵀ (F₁ ⊗ F₁)  |
| (ρ₁,ρ₂)    | 81      | C₁₂ (⊕ F'ₖᵀ) C₁₂ᵀ (F₁ ⊗ F₂) |

## Inversion: The SO(3) Phase Problem

### Why bootstrap fails for O

For D_n, the bootstrap inversion works because:

1. F₁ is recovered from F₁ᵀF₁ via symmetric square root (up to an O(2) rotation Q)
2. C^T(Q⊗Q)C is block-diagonal for ALL Q ∈ SO(2), since the CG decomposition of 2D irreps respects the continuous group SO(2)

For O, this fails because:

1. F₁ is 3×3, and the symmetric sqrt gives F₁ up to an SO(3) rotation Q
2. C^T(Q⊗Q)C is block-diagonal only for Q ∈ O (the discrete group), NOT for general Q ∈ SO(3)
3. The l=2 sector (ρ₂ and ρ₃) mixes under general SO(3) rotations, causing cross-talk between irreps during extraction

This is a fundamental mathematical obstruction, not an implementation bug.

### What we verified

From the extraction R = C^T β (S⊗S)⁻¹ C (using S = symmetric sqrt of F₁ᵀF₁):

- R R^T IS block-diagonal → correctly gives F_k^T F_k for all k
- The cross-block ratio A⁻¹B (where D² = \[[A,B],[C,D₂₂]\]) depends only on Q
- P' = D₂₂⁻¹C = -P^T (a consistency check, not new information)
- The reconstruction D² from P has 3 free parameters (R_A ∈ O(3), R_Q ∈ O(2)), matching the 3 DOF of SO(3)

### Implemented approach: Bootstrap + Levenberg-Marquardt

1. **Bootstrap initialization**: Recover F₀ exactly (cube root). Recover F₁ᵀF₁ exactly, take symmetric sqrt. Extract F_k^T F_k from R R^T, take symmetric sqrts. This gives exact Fourier *norms* but approximate *phases*.

2. **Levenberg-Marquardt corrections**: Each step computes the Jacobian J of the bispectrum w.r.t. the signal (172 × 24 matrix, via forward-mode AD), then solves (J^T J + μI)⁻¹ J^T r with adaptive damping μ. This is a single linear solve per step, not iterative optimization.

3. **Multi-start**: The bootstrap phases can land in different basins of attraction. Multiple restarts with randomised Fourier phases (random orthogonal rotations of each F_k) ensure convergence.

Default: 10 LM steps × 4 restarts.

## Known Limitations

1. **Inversion speed**: ~2s per sample due to per-sample Jacobian computation. Could be improved with batched Jacobian or analytical Jacobian formulas.

2. **Inversion accuracy**: Some signals require more restarts to converge. The multi-start mechanism handles this but at the cost of increased computation.

3. **Full bispectrum**: `selective=False` is not yet implemented for the forward pass.

4. **Differentiability**: The current `invert` implementation uses `detach()` internally and is not differentiable w.r.t. the input beta. An unrolled-differentiable version would require removing the detach calls and using `create_graph=True` in the Jacobian computation.

## References

- Mataigne, S., Keriven, N., & Peyré, G. (2024). The Selective G-Bispectrum and its Inversion. NeurIPS 2024.
- The escnn library (for verification of group data and irrep matrices).
