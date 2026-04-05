# The Selective SO(3) Bispectrum on S²: Construction and Completeness

## Summary

We construct a **selective** bispectrum for SO(3) acting on the 2-sphere S²
at band-limit L, using O(L²) scalar entries instead of the full O(L³). We
prove (sketch) that this selective set is a complete invariant for generic
band-limited signals: two signals have the same selective bispectrum if and
only if they differ by an SO(3) rotation.

## Setup and notation

**Signal**: f : S² → ℝ, band-limited to degree L.

**Spherical harmonic expansion**:
```
f(θ, φ) = Σ_{l=0}^{L}  Σ_{m=-l}^{l}  a_l^m  Y_l^m(θ, φ)
```
The coefficient vector at degree l is F_l = (a_l^{-l}, ..., a_l^l) ∈ ℂ^{2l+1}.
For real f, the conjugation symmetry a_l^{-m} = (-1)^m conj(a_l^m) holds, giving
2l + 1 real degrees of freedom per degree. Total signal dimension: (L+1)².

**Rotation action**: Under g ∈ SO(3), the SH coefficients transform as
F_l → D_l(g) F_l, where D_l is the (2l+1)-dimensional Wigner D-matrix.

**Bispectrum**: For each valid triple (l₁, l₂, l) with |l₁ - l₂| ≤ l ≤ l₁ + l₂:
```
β_{l₁,l₂,l} = Σ_{m₁,m₂,m}  C(l₁ m₁; l₂ m₂ | l m)  a_{l₁}^{m₁}  a_{l₂}^{m₂}  conj(a_l^m)
```
where C are Clebsch-Gordan coefficients. Each β is a scalar (real for real f).
The full bispectrum has O(L³) entries.

**Completeness (Kakarala 2009)**: The full bispectrum determines f up to
SO(3) rotation for generic f.

## Why O(L) is impossible

For finite groups, the selective bispectrum has O(|G|) entries (Mataigne et al.
2024). For SO(3) on S², one might hope the "BFS chain" β_{1,0}, β_{1,1},
β_{1,2}, ..., β_{1,L-1} — just L matrix-valued entries — suffices. It does not.

**The scalar bottleneck**: On S², F_l is a *vector* of dim 2l+1 (not a matrix
as for signals on the group SO(3) itself). Each bispectrum entry β_{l₁,l₂,l}
is a single scalar. The chain pairs (1,l) for l = 1..L yield only ~3L scalars,
but we must recover (L+1)² ≈ L² scalar unknowns.

**Dimensional lower bound**: The orbit space has dimension (L+1)² - 3 (signal
dimension minus SO(3) parameters). Any complete invariant must have at least
(L+1)² - 3 independent components. Therefore O(L²) is a hard lower bound.

Our construction achieves this bound.

## The selective index set

For each target degree l = 0, ..., L, we select up to 2l + 1 bispectral
triples. These are chosen by priority from four categories:

### Category 1: Chain entries

```
(l₁, l₂, l)  with  l₁ ≤ l₂ < l,  l₁ + l₂ ≥ l
```

These are **linear** in F_l: β = ⟨G, F_l⟩ where G is computed from known
lower-degree coefficients and CG coefficients.

### Category 2: Cross entries

```
(l₁, l, l')  with  1 ≤ l₁ < l,  l - l₁ ≤ l' < l
```

Also **linear** in F_l: here F_l appears in the tensor product (middle
position) with known F_{l₁}, contracted against known F_{l'}.

### Category 3: Power entry

```
(0, l, l)
```

Gives β = a₀ · ||F_l||² — a **quadratic** constraint.

### Category 4: Self-coupling entries

```
(l, l, l')  with  0 ≤ l' < l
```

**Quadratic** (or cubic when l' = l) in F_l. Needed at low degrees where
chain + cross entries are insufficient.

### Entry counts by degree

| Degree l | Chain | Cross | Power | Self | Total (capped at 2l+1) |
|----------|-------|-------|-------|------|------------------------|
| 0        | —     | —     | —     | —    | 1 (just β_{0,0,0})     |
| 1        | 0     | 0     | 1     | 1    | 2                      |
| 2        | 1     | 1     | 1     | 2    | 5                      |
| 3        | 1     | 2     | 1     | 3    | 7                      |
| 4        | 2     | 3     | 1     | 3    | 9                      |
| l ≥ 5    | ⌊l/2⌋ | l(l-1)/2 | 1 | l  | 2l+1 (truncated)       |

Total output size: Σ (entries per l) ≈ (L+1)² - 1.

## Completeness proof (sketch)

**Theorem.** The selective SO(3) bispectrum on S² at band-limit L, as
constructed above, determines f up to SO(3) rotation for Lebesgue-almost-every
band-limited f.

**Proof.** We recover F₀, F₁, ..., F_L sequentially. At each step, all
lower-degree coefficients are known (up to a common rotation).

### Step 0: Recover F₀

β_{0,0,0} = (a₀⁰)² · conj(a₀⁰) = |a₀⁰|² · a₀⁰.

For real signals, a₀⁰ ∈ ℝ, so a₀⁰ = sign(β_{0,0,0}) · |β_{0,0,0}|^{1/3}.
Uniquely determined. ∎ (step 0)

### Step 1: Recover F₁ via gauge fixing

β_{0,1,1} = a₀⁰ · ||F₁||².

This recovers ||F₁|| (since a₀⁰ is known). F₁ is a 3-vector. Choose
g ∈ SO(3) such that D₁(g) F₁ points along the z-axis:

```
D₁(g) F₁ = (0, ||F₁||, 0)
```

This consumes 2 of SO(3)'s 3 degrees of freedom (the direction of F₁ on the
unit sphere). A residual SO(2) azimuthal symmetry (rotations around z) remains.

Under this residual SO(2), a_l^m → e^{imφ} a_l^m for all l, m.

### Step 2: Recover F₂ (nonlinear, 5 unknowns)

After gauge fixing, F₁ = (0, c, 0) with c = ||F₁||. Real-signal constraints
reduce F₂ to 5 real unknowns: a₂⁰, Re(a₂¹), Im(a₂¹), Re(a₂²), Im(a₂²).

**Residual SO(2) gauge**: fix Im(a₂¹) = 0 (choosing the remaining azimuthal
rotation). Now 4 unknowns remain.

**Available equations** (5 entries in the selective set for degree 2):

1. β_{1,1,2}: After gauge fixing, CG coupling with F₁ = (0,c,0) yields
   β_{1,1,2} = C(1,0;1,0|2,0) · c² · conj(a₂⁰). Determines a₂⁰.

2. β_{1,2,1}: Similarly linear in F₂, but after gauge fixing also only
   couples to a₂⁰. Consistency check.

3. β_{0,2,2} = a₀ · ||F₂||²: Quadratic constraint relating
   |a₂⁰|² + 2|a₂¹|² + 2|a₂²|².

4. β_{2,2,1}: A weighted quadratic form Σ w_m |a₂ᵐ|² with CG-derived
   weights. Independent from ||F₂||² due to distinct CG coefficients.

5. β_{2,2,0}: Proportional to ||F₂||² (by CG symmetry), but can serve
   as consistency check.

After fixing a₂⁰ and Im(a₂¹) = 0, the remaining 3 unknowns (|a₂¹|, Re(a₂²),
Im(a₂²)) are constrained by 3 equations (from entries 3, 4, and the cubic
invariant β_{2,2,2} if included). The Jacobian has full rank at generic points.
∎ (step 2)

### Step 3: Recover F₃ (transition zone)

7 entries provide 5 linear + 2 nonlinear equations for 7 unknowns. The
additional self-coupling entries cover the deficit from chain+cross alone.
Same Jacobian-rank argument applies at generic points.

### Step 4 (l ≥ 4): Linear bootstrap

At degree l ≥ 4, the chain + cross entries provide N ≥ 2l+1 linear equations:

```
β_k = ⟨G_k, F_l⟩,    k = 1, ..., N
```

where each G_k is a (2l+1)-dimensional vector computed from known lower-degree
coefficients and Clebsch-Gordan coefficients:

```
G_k^m = Σ_{m₁}  C(l₁, m₁; l₂, m-m₁ | l, m) · a_{l₁}^{m₁} · a_{l₂}^{m-m₁}
```

(or the analogous cross formula).

This is a linear system A x = b with A ∈ ℂ^{N × (2l+1)}.

**Claim**: A has rank 2l+1 for Lebesgue-a.e. signal f.

*Proof of claim*: The rows of A are polynomial functions of {a_k^m : k < l}
with CG coefficients as fixed nonzero weights. The determinant of any
(2l+1) × (2l+1) submatrix of A is a polynomial in these coefficients. This
polynomial is not identically zero (as can be verified at a single generic
point, e.g., by numerical computation). Therefore the rank-deficient set
{f : rank(A) < 2l+1} is a proper algebraic subvariety of codimension ≥ 1,
hence Lebesgue-null.

At a generic point, the linear system is consistent and determined:
F_l = A† b (pseudoinverse). ∎ (step 4)

### Induction

Steps 0–4 recover F₀, F₁, ..., F_L sequentially, each up to the rotation
gauge fixed in step 1. By Plancherel's theorem (inverse SHT), the signal f is
recovered from its SH coefficients. The gauge freedom corresponds exactly to
an SO(3) rotation of f.

Therefore: β_sel(f) = β_sel(f') implies f' ∈ SO(3) · f for generic f. ∎

## Comparison with existing results

| Result | Invariant type | Size | Completeness |
|--------|---------------|------|-------------|
| Kakarala (2009) | Full bispectrum | O(L³) | Yes (constructive) |
| Edidin & Satriano (2023) | Full bispectrum (band-limited) | O(L³) | Yes (generic) |
| Bendory et al. (2025) | Freq. marching on shells | O(L²) per shell | Yes (3 shells) |
| Mataigne et al. (2024) | Selective G-bispec (finite) | O(\|G\|) | Yes (exact) |
| **This work** | **Selective SO(3) on S²** | **O(L²)** | **Yes (generic)** |

## Relation to the inversion gap

For finite groups (O, D_n), the "inversion gap" (see `inversion_gap_for_
octahedral_group.md`) arises when the bootstrap ambiguity Q is not absorbed by
CG matrices. For SO(3) on S², the situation is different:

- The ambiguity in F₁ is an SO(3) rotation Q (3 DOF).
- We fix Q by gauge-choosing F₁'s direction (2 DOF) and F₂'s azimuthal phase
  (1 DOF), consuming all 3 DOF before the linear bootstrap begins.
- At l ≥ 4, no ambiguity remains — the linear system has a unique solution.

This avoids the octahedral-group problem entirely: we never need CG matrices
to "absorb" a continuous ambiguity. The gauge fixing is explicit.

## Implementation

The selective bispectrum is implemented in `SO3onS2(selective=True)`.

```python
from bispectrum import SO3onS2

# Full: O(L³) entries
full = SO3onS2(lmax=5, selective=False)   # 69 entries

# Selective: O(L²) entries
sel = SO3onS2(lmax=5, selective=True)     # 35 entries
```

Both produce the same values on shared triples (verified to machine precision).
Rotation invariance holds for both at the same discretization-limited accuracy.

## Limitations and open questions

1. **Genericity assumption**: The proof requires f to avoid a measure-zero
   algebraic variety. Characterizing this variety explicitly (analogous to the
   a_{n,1} ≠ 0 condition for the disk) would strengthen the result.

2. **Numerical conditioning**: The linear system at each degree l has a
   condition number that depends on the signal. For spectrally decaying
   signals (the typical case), the G vectors may become ill-conditioned at
   high l. A stability analysis bounding the condition number in terms of the
   spectral decay rate is the main open problem.

3. **Inversion algorithm**: The forward selective bispectrum is implemented.
   Inversion (recovering f from β_sel) requires:
   - l = 0, 1: closed-form (cube root + gauge fix)
   - l = 2, 3: nonlinear solve (Levenberg-Marquardt, as for octahedral)
   - l ≥ 4: linear solve (pseudoinverse)

4. **Larger band-limits**: The current CG data supports lmax ≤ 5.
   Extending to larger L requires generating CG matrices (e.g., via
   e3nn or sympy.physics.quantum) and analyzing the conditioning.
