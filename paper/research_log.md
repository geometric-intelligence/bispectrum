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

## KEY DISCOVERY: Reflection Obstruction and T_R Resolution

### The Reflection T_R

The azimuthal reflection R: (θ,φ) → (θ,-φ) acts on SH coefficients as:
  a_ℓ^m → (-1)^m a_ℓ^{-m} = conj(a_ℓ^m) for real signals

On the gauge slice (F₁ aligned to z, Im(a₂¹) = 0):
  T_R: v → -v, q_{ℓ,m} → -q_{ℓ,m} for all ℓ ≥ 3, m ≥ 1

T_R preserves the gauge conditions and is NOT in SO(3) (det = -1).

### Parity Classification (Proven Symbolically + Numerically)

For real signals:
  β_{l1,l2,l}(T_R · f) = (-1)^{l1+l2+l} · conj(β_{l1,l2,l}(f))

- **Even parity** (l1+l2+l even): β is real and T_R-invariant
- **Odd parity** (l1+l2+l odd): β is purely imaginary, Im(β) flips under T_R

CG power P is always T_R-invariant (proved via CG symmetry).

### Vanishing Pattern for Odd-Parity Entries

Odd-parity β_{l1,l2,l} is identically ZERO whenever any two indices are equal:
- l1 = l2: CG antisymmetry vs symmetric square
- l1 = l or l2 = l: CG antisymmetry + reality constraint cancellation

**First NONZERO odd-parity entry: β(2,3,4)** with l1+l2+l = 9, all distinct.

Verified numerically at seed-42 signal (using sympy CG reference):
  Im(β(2,3,4)) ≈ 3.26 (nonzero)
  Im(β(2,3,4))(T_R · f) ≈ -3.26 (sign flip)

Other nonzero odd-parity entries at degree 4: β(2,4,3), β(3,4,2).

### Complete Classification (degrees ≤ 5)

All odd-parity entries verified (60+ entries checked):
- **ZERO** for real signals: ALL entries with l1=l2, l2=l, or l1=l
  (includes β(1,1,1), β(2,2,3), β(1,3,3), β(2,3,2), β(3,3,1), etc.)
- **NONZERO** (purely imaginary): entries with ALL THREE indices distinct
  - β(2,3,4), β(2,4,3), β(3,4,2): Im ≈ ±8 to ±12
  - β(2,4,5), β(2,5,4), β(4,5,2): Im ≈ ±1.4 to ±2.1

### Implication for Completeness

- **L ≤ 3**: No nonzero odd-parity entries exist. The augmented bispectrum
  is O(3)-complete but NOT SO(3)-complete. Fibers are 2-point: {f, T_R(f)}.

- **L ≥ 4**: The complex-valued β(2,3,4) breaks T_R via its imaginary part.
  Combined with full Jacobian rank, fibers are 1-point. Full SO(3)-completeness.

## Proof Strategy (Final)

The augmented selective bispectrum Φ_aug: V_L → C^N × R^M separates
SO(3) orbits on R^{(L+1)^2} / SO(3) for L ≥ 4.

Proof structure:
1. Gauge fix using F₁ (2 DOF) and Im(a₂¹) = 0 (1 DOF)
2. Recover F₀ from β(0,0,0), F₁ from β(0,1,1)
3. Seed recovery (ℓ ≤ 3): augmented system has full rank → 0-dim fibers
   with 2 points {f, T_R(f)} (T_R = azimuthal reflection)
4. T_R resolution at ℓ = 4: Im(β(2,3,4)) ≠ 0 generically, flips sign
   under T_R, identifying the correct branch
5. Per-degree bootstrap (ℓ ≥ 5): full-rank linear bootstrap determines
   F_ℓ uniquely given correct lower degrees

## Linear Bootstrap Verification

Verified for ℓ = 4 to 100 (script: benchmarks/verify_linear_bootstrap.py):
- All pass: rank = 2ℓ+1 at deterministic witness (seed 42)
- Condition numbers: 10^1 to 10^4 (well-conditioned)
- For ℓ ≥ 8: uses closed-form tridiagonal family

## Seed System Analysis (Symbolic, via sympy)

### Degree 2 (4 unknowns: y, x, u, v)

| Invariant | Expression | Determines |
|-----------|-----------|-----------|
| β_{1,1,2} | √6·c²·y/3 | y (linear) |
| P_{1,2,1} | c²(3x²+2y²)/5 | x² (then x>0 by gauge) |
| β_{0,2,2} | A₀(2u²+2v²+2x²+y²) | S = u²+v² |
| β_{2,2,2} | (6√14·S·y - 6√21·u·x² - ...)/7 | u (linear in u) |

Remaining ambiguity: v² = S - u² gives |v| but not sign(v).

### Degree 3 (7 unknowns: t, p₁, q₁, p₂, q₂, p₃, q₃)

Key finding: the 3 scalar bispectral entries β(1,2,3), β(1,3,2), β(2,3,1)
are in the Jacobian row span of each other and contribute rank 1 collectively.
This is much worse than the complex case.

The entries β(2,3,2) and β(2,2,3) are identically zero (odd parity, repeated indices).

Full augmented seed system (13 nonzero entries, 11 unknowns):
  - Bispectrum alone (8 nonzero entries): rank 6
  - Adding ALL 5 CG power entries: rank 11 (FULL)
  - CG power entries essential for rank: P(1,2,1), P(1,3,2), P(2,3,1), P(2,3,2), P(3,3,2)
  - Each CG power entry contributes exactly one additional rank direction
Verified at rational witness using exact sympy arithmetic.

## Experiment Log

### 2025-04-13: Parity breakthrough

Discovered and proved the complete parity structure:
- Even-parity β and CG power P are T_R-invariant
- Odd-parity β vanishes iff any two indices coincide
- First nonzero odd-parity entry at degree 4: β(2,3,4)
- This gives all-L proof for L ≥ 4 (no numerical ceiling)

Wrote complete proof in paper/proof_completeness.tex.

### 2026-04-13: Exact implementation verification

Verified proof_completeness.tex against exact implementation entries in so3_on_s2.py:

1. **CG bug found & fixed**: Custom Racah-formula CG had wrong s_min/s_max bounds.
   Switched to sympy.physics.wigner.clebsch_gordan as reference. All parity claims
   confirmed: odd-parity entries ARE purely imaginary (Re ~ 1e-16).

2. **Exact seed system match**: The implementation uses exactly these entries at lmax=3:
   - Bispectrum: β(1,1,2), β(0,2,2), β(2,2,2), β(1,2,3), β(2,3,2)[=0],
     β(1,3,2), β(2,3,1), β(2,2,3)[=0], β(0,3,3), β(3,3,2)
   - CG power: P(1,2,1), P(1,3,2), P(2,3,1), P(2,3,2), P(3,3,2)
   - 13 nonzero entries, 2 identically zero → 13×11 Jacobian, rank 11

3. **Revised rank structure**: Bispectrum alone achieves rank 6 (not 8 as earlier
   claimed with wrong entries). ALL 5 CG power entries are rank-essential,
   each contributing 1 rank direction.

4. **T_R resolution confirmed**: β(2,3,4) is in the selective index map at lmax=4.
   Im(β(2,3,4)) ≈ 3.26 at seed-42 witness, flips sign under T_R.
   Also β(2,4,3) and β(3,4,2) provide redundant parity-breaking.

5. **Output size**: Total augmented output is ~1.6× orbit-space dimension,
   well within Θ(L²).

6. **Critical: Per-degree REAL rank deficiency**: The "linear bootstrap" has
   COMPLEX rank 2ℓ+1 but REAL rank only ~ℓ for real signals! The scalar
   bispectrum ALONE cannot determine F_ℓ for real signals at ANY degree.
   CG power entries are essential at EVERY degree (not just the seed).
   Per-degree real rank structure:
   - ℓ=4: linear=4/9, +self=6/9, +CG=9/9
   - ℓ=5: linear=4/11, +self=6/11, +CG=11/11
   - ℓ=6: linear=8/13, +self=11/13, +CG=13/13

7. **T_R resolution logic**: The proof's T_R argument was corrected.
   The T_R branch doesn't produce T_R(f) at higher degrees; instead,
   the bootstrap on the T_R branch produces a DIFFERENT signal g that
   satisfies all bootstrap entries but FAILS non-bootstrap entries.
   Verified at ℓ=4: self-coupling β(4,4,2) differs by ~20, CG power
   P(2,4,3) differs by ~35.

8. **Parity formula corrected**: β(T_R·f) = (-1)^{l1+l2+l} β(f) for ALL
   signals (not just real). Separate reality constraint gives
   conj(β) = (-1)^{l1+l2+l} β for real signals.

### 2026-04-13: Seed fibre is NOT 2-point — critical proof restructuring

9. **Seed fibre has 10 solutions per v-branch (20 total)**: Exhaustive
   multi-start search (5000 starts per branch) finds EXACTLY 10 distinct
   real solutions per sign(v) branch, forming 10 T_R pairs.
   All 20 solutions have full Jacobian rank 11/11 (simple zeros).
   The degree-3 subsystem (7 real equations in 7 unknowns, degrees
   1+2^5+4, Bezout bound 128) has more real solutions than the
   previously claimed 1 per branch.
   Script: benchmarks/verify_seed_fibre.py

10. **Degree-4 eliminates all spurious seed solutions**: For each of the
    10 seed solutions on the +v branch, multi-start NLS (200 starts)
    tries to find F4 satisfying all 29 real degree-4 constraints
    (12 bispectral × 2 + 5 CG power). ONLY the true seed solution
    achieves zero residual (||r|| ~ 1e-16). All 9 spurious solutions
    have residual >= 1e-2.
    Key mechanism: degree-4 entries that DON'T involve F3 (β(2,2,4),
    β(2,4,2), P(1,4,3), P(1,4,4), P(2,4,2), P(2,4,3), β(4,4,2),
    β(4,4,4)) nearly determine F4 independently. Entries involving
    F3 then impose constraints that are inconsistent for spurious F3.
    Script: benchmarks/verify_degree4_filter.py

11. **Proof restructured**: The seed lemma (lem:ell3) now correctly claims
    only a FINITE fibre (not exactly 2 points). A new lemma
    (lem:degree4-filter) handles fibre reduction at degree 4.
    The T_R resolution (lem:TR-resolve) is now stated as a special case
    of degree-4 elimination, with the rational function structure
    (numerator p, denominator det(A4)^2) made explicit.

12. **Bootstrap scope clarified**: The bootstrap rank verification
    covers ℓ=4..100. For any fixed L, the generic set is the complement
    of L-3 proper algebraic subvarieties (one per degree). The theorem
    is stated for fixed L with L-dependent genericity conditions.
