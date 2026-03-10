# Selective Bispectrum for SO(3) on S²

Notes on the open problem of finding a selective (O(L) instead of O(L²))
bispectrum for SO(3) acting on S², and how to approach it.

## Why it's hard

For finite groups, Algorithm 1 (Mataigne et al., NeurIPS 2024) does a BFS on
the Kronecker table starting from ρ₀. This terminates because there are
finitely many irreps.

For SO(3), the irreps are the Wigner D-matrices ρₗ for l = 0, 1, 2, ... and
the Kronecker rule is:

```
ρ_{l₁} ⊗ ρ_{l₂} = ⊕_{l=|l₁-l₂|}^{l₁+l₂} ρ_l
```

The BFS never terminates — ρ₁ ⊗ ρₗ always produces ρ\_{l+1}, which is new,
forever. So there is no finite selective subset that covers all irreps.

## Approach 1: Truncated selective bispectrum (most practical)

Fix a band-limit L. Then there are finitely many irreps ρ₀, ..., ρ_L and
the Kronecker table is finite. Run the same BFS as Algorithm 1 on this
truncated table. The generating irrep is ρ₁ (since ρ₁ ⊗ ρₗ always contains
ρ\_{l+1}), so the selective pairs would be:

- β\_{ρ₀, ρ₀} → recover F(ρ₀)
- β\_{ρ₀, ρ₁} → recover |F(ρ₁)|, up to SO(3) indeterminacy
- β\_{ρ₁, ρ₁} → recover F(ρ₂)
- β\_{ρ₁, ρ₂} → recover F(ρ₃)
- ...
- β\_{ρ₁, ρ\_{L-1}} → recover F(ρ_L)

That's L+1 matrix-valued coefficients — O(L) instead of O(L²) for the full
bispectrum. This is the direct analogue of D_n's Algorithm 4.

**Key open question**: does this actually give a complete invariant for
band-limited functions on S²? The D_n proof relies on the finite group
structure. For truncated SO(3), one would need to prove that the truncated
bispectrum still separates orbits, or characterize reconstruction error bounds.

## Approach 2: Bootstrap inversion with Wigner 3j symbols

The D_n inversion proof (Theorem 4.4) bootstraps from F(ρ₁): recover F(ρ_k)
sequentially from β\_{ρ₁, ρ\_{k-1}}. The same structure applies to SO(3) with
ρ₁ as the generating irrep (the spin-1 representation). The CG matrices are
the well-known Wigner 3j symbols. The inversion step would be:

```
F(ρ_{l+1}) = extract from C_{1,l}† [F(ρ₁) ⊗ F(ρ_l)]⁻¹ β_{ρ₁,ρ_l} C_{1,l}
```

Challenges:

- F(ρ_l) is a (2l+1) × (2l+1) matrix (not 2×2 like D_n).
- The tensor product ρ₁ ⊗ ρₗ decomposes into ρ\_{l-1} ⊕ ρ_l ⊕ ρ\_{l+1},
  so the block-diagonal matrix is (3(2l+1)) × (3(2l+1)). Need to extract
  the ρ\_{l+1} block.
- The indeterminacy is now SO(3) (not O(2)), acting on the 3×3 F(ρ₁).
- Conditioning of the matrix inverse degrades with l.

## Approach 3: Stability analysis (the real bottleneck)

Even if the algebra works, the practical question is numerical stability.
Each step inverts F(ρ₁) ⊗ F(ρ_l), and errors compound over L sequential
steps. Key questions:

- Bound the condition number of the Kronecker product at each step.
- Show that for "generic" signals (non-degenerate Fourier coefficients),
  the bootstrap is stable.
- Characterize the reconstruction error as a function of L and the signal's
  spectral decay.

## Approach 4: Direct optimization (bypass algebraic inversion)

Skip algebraic inversion entirely. Given β_sel(Θ), solve:

```
min_{f: S² → R}  ||β_sel(f) - β_sel(Θ)||²
```

This is what the paper does in their adversarial experiment (Eq. 12).
Completeness means this has a unique minimizer up to SO(3). No closed-form
inversion needed, but loses the efficiency of the algebraic approach.

## Recommendation

The most promising path is (1) + (2): truncate at band-limit L, use ρ₁ as
the generating irrep (giving O(L) selective coefficients), and prove the
bootstrap inversion works with Wigner 3j symbols as CG matrices. The hard
part is the stability/conditioning analysis for large L.

This would be a publishable result.

## Literature survey

### Foundational theory

- **Kakarala (2009)**. "The Bispectrum as a Source of Phase-Sensitive
  Invariants for Fourier Descriptors: A Group-Theoretic Approach".
  [arXiv:0902.0196](https://arxiv.org/abs/0902.0196).
  Proves the bispectrum is a *complete* source of invariants for homogeneous
  spaces of compact groups, including SO(3) on S². Provides a constructive
  inversion algorithm. This is the theoretical foundation that guarantees
  the full bispectrum works; the open question is whether a *selective*
  (reduced) subset suffices.

- **Kondor (2008)**. "Group Theoretical Methods in Machine Learning".
  PhD thesis, Columbia. Develops the representation-theoretic framework
  for equivariance and invariance on compact groups. Establishes the
  bispectrum as O(|G|²) complete invariant, motivating the search for
  O(|G|) reductions.

### Orbit recovery (bispectrum completeness for band-limited SO(3))

- **Edidin & Satriano (2023)**. "Band-limited functions on compact Lie
  groups and orbit determination from third-order moments".
  [arXiv:2306.00155](https://arxiv.org/abs/2306.00155).
  Proves that for generic band-limited functions on SO(3) (and other
  compact groups like SU(n), SO(2n+1)), the bispectrum (third moment)
  determines the group orbit. Motivated by cryo-EM. **This is the
  closest result to what we need** — it says the full bispectrum of
  band-limited functions is complete, but does not address selectivity
  (i.e. which subset of bispectral pairs suffices).

- **Bendory, Edidin, Katz & Kreymer (2025)**. "Orbit recovery for
  spherical functions".
  [arXiv:2508.02674](https://arxiv.org/abs/2508.02674).
  Shows degree-3 invariants (bispectrum) suffice for orbit recovery of
  functions on S² under SO(3), with band-limiting + radial discretization.
  For SO(3), proves 3 spherical shells suffice (verifying a conjecture by
  Bandeira et al.). Provides an explicit **frequency marching** algorithm
  — essentially the same bootstrap as our Approach 2. Validates on protein
  structures. **Key reference for implementation.**

- **Edidin & Katz (2024)**. "The reflection invariant bispectrum: signal
  recovery in the dihedral model".
  [arXiv:2408.09599](https://arxiv.org/abs/2408.09599).
  Extends orbit recovery to the dihedral group D_n (non-abelian). Proves
  degree-3 invariants determine generic orbits. Notes that frequency
  marching is impractical for D_n but optimization works well. Directly
  relevant to our DnonDn implementation. Sample complexity ω(σ⁶) matches
  the cyclic case.

### Selective bispectrum (same group, Mataigne et al.)

- **Mataigne, Miolane & Mataigne (2024)**. "The Selective G-Bispectrum
  and its Inversion: Applications to G-Invariant Networks". NeurIPS 2024.
  [arXiv:2407.07655](https://arxiv.org/abs/2407.07655).
  Our main reference. Proves selective bispectrum for C_n, products of
  cyclic groups, D_n, octahedral and full octahedral groups. Does not
  address SO(3) or other compact/infinite groups.

- **Myers & Miolane (2025)**. "The Selective Disk Bispectrum and Its
  Inversion, with Application to Multi-Reference Alignment".
  [arXiv:2511.19706](https://arxiv.org/abs/2511.19706).
  Extends the selective bispectrum to SO(2) acting on the disk (Fourier-
  Bessel basis). Derives the first explicit inverse for the disk
  bispectrum. Applies to multi-reference alignment of rotated images.
  **Natural stepping stone** toward SO(3) on S² — same continuous group
  structure but in 2D.

### Applied: bispectrum in computational chemistry (SNAP / ACE)

- **Thompson et al. (2015)**. "Spectral Neighbor Analysis Potential (SNAP)".
  J. Comp. Phys.
  Uses SO(3) bispectrum components (via hyperspherical harmonics / Wigner
  D-matrices) as rotationally invariant descriptors of atomic environments.
  Band-limited at some `twojmax`. The full bispectrum at band-limit J has
  O(J³) components — a *selective* version would directly reduce descriptor
  size and training cost for ML potentials.

- **Drautz (2019)**. "Atomic Cluster Expansion".
  Phys. Rev. B 99, 014104.
  Generalizes SNAP to arbitrary body order. The 3-body (bispectrum) term
  is the SO(3) bispectrum. ACE provides a systematic completeness
  framework but does not address selectivity in the bispectral sense.

### Equivariant neural networks (related tools)

- **e3nn** (Geiger & Smidt, 2022).
  [docs.e3nn.org](https://docs.e3nn.org/).
  PyTorch library for E(3)-equivariant neural networks. Core operation is
  the CG tensor product of SO(3) irreps — exactly the building block needed
  for SO(3) bispectrum. `e3nn.o3.ReducedTensorProducts` already handles
  CG decomposition and could be used to compute bispectral coefficients.
  Potential implementation substrate for a selective SO(3) bispectrum layer.

## Gaps and opportunities

1. **Edidin et al. prove completeness but not selectivity.** They show the
   full bispectrum of band-limited functions determines orbits under SO(3).
   But they compute ALL O(L³) bispectral pairs. The selective question —
   which O(L) pairs suffice — is exactly our open problem.

2. **Myers & Miolane solve the disk (SO(2)) case.** The step from SO(2)
   on disk to SO(3) on S² is the natural next target.

3. **Frequency marching = our bootstrap.** Bendory et al.'s "frequency
   marching" algorithm for SO(3) orbit recovery is essentially the same
   sequential recovery strategy as Algorithm 4 for D_n. Their stability
   analysis (or lack thereof for large L) is the key gap.

4. **SNAP/ACE would directly benefit.** A selective SO(3) bispectrum at
   band-limit J with O(J) instead of O(J³) components would be immediately
   useful for ML interatomic potentials.

## Deep dive: Myers & Miolane (2025) — The Selective Disk Bispectrum

This paper is the closest precedent to what we want for SO(3) on S².
It solves the analogous problem one dimension down: SO(2) on the disk D.

### Setup

- **Domain**: unit disk D = {x ∈ R² : ||x|| < 1} (NOT a group, NOT
  homogeneous for SO(2) — the origin is fixed).
- **Symmetry**: SO(2) rotations.
- **Basis**: disk harmonics ψ\_{nk}(r,θ) = c\_{nk} J_n(λ\_{nk} r) e^{inθ},
  indexed by angular frequency n ∈ Z and radial index k ∈ Z\_{>0}.
  This is a Fourier-Bessel basis: angular part is standard Fourier,
  radial part is Bessel functions J_n.
- **Equivariance**: rotating f by φ multiplies a\_{nk} by e^{inφ}.
  The angular index n plays the role of the irrep index in C_n.
- **Band-limiting**: keep coefficients with λ\_{nk} ≤ πL/2 (Nyquist).
  Gives m total coefficients with max angular frequency N_m.

### Full disk bispectrum

```
b_{j1, j2, k3} = a_{n_{j1}, k_{j1}} · a_{n_{j2}, k_{j2}} · a*_{n_{j1}+n_{j2}, k3}
```

Same abelian triple-product structure as C_n, but each coefficient
carries an extra radial index k. Complexity: O(m³/N_m) space.

### Selective version — the key construction

They pick only two "generating" first indices:

- j1 = 0 → (n=0, k=1): the DC coefficient a\_{0,1}
- j1 = 2 → (n=1, k=1): the first angular mode a\_{1,1}

Selective coefficients:

1. b\_{0,0,k} = a\_{0,1}² · a\*\_{0,k} for k = 1..K_0
   → recovers all radial modes at angular frequency n=0

2. b\_{2,n,k} = a\_{1,1} · a\_{n,1} · a\*_{n+1,k}
   for n = 0..N_m-1 and k = 1..K_{n+1}
   → bootstraps from angular frequency n to n+1,
   recovering ALL radial modes k at each step

Total: O(m) coefficients instead of O(m³/N_m).
For 112×112 images: 4,957 vs 1,057,052 coefficients.

### Inversion algorithm

Identical bootstrap to C_n's Algorithm 2, extended for radial modes:

1. Get |a\_{0,1}| from b\_{0,0,1} = a\_{0,1}³.
2. Recover all a\_{0,k} from b\_{0,0,k} / a\_{0,1}².
3. Get |a\_{1,1}| from b\_{2,0,1} using a\_{0,1}. Fix arg(a\_{1,1}) = 0
   (SO(2) indeterminacy — same as our C_n case).
4. For n = 0..N_m-1: recover a\_{n+1,k} for all k from
   b\_{2,n,k} / (a\_{1,1} · a\_{n,1}).

Assumption: a\_{n,1} ≠ 0 for n = 0..N_m-1 (generic, holds with any noise).

### Completeness proof

They prove: b^f = b^{f'} iff f' = f ∘ R_φ for some φ ∈ SO(2).
The proof follows directly from the inversion theorem — if you can
recover the signal from the selective coefficients, those coefficients
are a complete invariant.

### Numerical approximation

They leverage a fast approximate DH transform (Marshall & Izzo, 2023)
that runs in O(L² log L) instead of O(L³), giving total complexity
O(L² log L + m) with controlled accuracy bounds:

```
||b̃ - b||_∞ ≤ C ε ||f||_1
```

### Application: multi-reference alignment (MRA)

They derive a noise bias correction for averaging bispectra of noisy
rotated copies (analogous to Bendory 2018 for the translation case).
This enables: observe N noisy rotated copies → average in bispectrum
space → correct bias → invert → recover ground truth image.
This would be impossible without the bispectrum inverse.

### Key insight for SO(3) on S²

The disk bispectrum has a **two-index structure** (n, k) — angular and
radial — while C_n has only one index. The bootstrap walks along the
angular direction (n → n+1) and recovers all radial modes at each step.

For SO(3) on S², spherical harmonics Y_l^m have degree l and order m,
with 2l+1 orders per degree. The analogy is:

| Disk (SO(2))         | Sphere (SO(3))                      |
| -------------------- | ----------------------------------- |
| Angular freq n       | Degree l                            |
| Radial index k       | No radial on S² (present on ball)   |
| Scalar coeff a\_{nk} | Matrix coeff F(ρ_l) of size (2l+1)² |
| e^{inφ} equivariance | D^l(R) equivariance (Wigner D)      |
| n₁ + n₂ = n₃ (CG)    | Wigner 3j symbols (non-trivial CG)  |
| SO(2) indeterminacy  | SO(3) indeterminacy                 |
| Bootstrap: n → n+1   | Bootstrap: l → l+1                  |

The bootstrap structure is identical. The two complications for SO(3):

1. CG decomposition is non-trivial (Wigner 3j instead of δ\_{n1+n2,n3}).
2. Fourier coefficients are matrices, not scalars, so "dividing" means
   matrix inversion with conditioning concerns.

### What this means for us

The Myers & Miolane paper validates the selective approach for a
continuous group on a non-group domain. Their success strongly suggests
the same strategy works for SO(3) on S². The remaining work is:

1. Replace disk harmonics with spherical harmonics.
2. Replace the scalar bootstrap with a matrix bootstrap using Wigner 3j.
3. Prove completeness (or leverage Edidin & Satriano's orbit recovery
   result to argue it).
4. Analyze numerical stability for the matrix inversion at each step.
5. Implement in PyTorch (potentially using e3nn for CG products).
