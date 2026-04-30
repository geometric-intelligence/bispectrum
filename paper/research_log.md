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

### 2026-04-16: STRUCTURAL AUDIT and rescue plan

Auditing the entire `appendix:so3-completeness` proof for non-structural
content. Goal: identify whether the proof can be made fully structural
while preserving O(L²) augmented selective invariant.

#### Catalogue of non-structural claims

| # | Claim | Where | Type of evidence |
|---|-------|-------|------------------|
| C1 | Seed Jacobian (13×11) has rank 11 at rational witness | Lem app-ell3 | sympy at one rational point |
| C2 | Seed fibre has 10 real solutions per v-branch | Rem app-seed-fibre-size | Multi-start NLS, 5000 starts |
| C3 | Degree-4 NLS finds zero residual ONLY for true seed (9 spurious eliminated) | Lem app-degree4-filter | Multi-start NLS, 200 starts × 10 seeds |
| C4 | Complex bootstrap rank = 2ℓ+1 for ℓ ∈ [4, 100] | Lem app-bootstrap-rank | Per-ℓ numerical witness, seed 42 |
| C5 | Real per-degree augmented rank = 2ℓ+1 for ℓ ∈ [4, 30] | Lem app-augmented-perdegree | Per-ℓ numerical witness |
| C6 | T_R branch is degree-4 inconsistent | Lem app-TR-resolve | Same witness as C3, rational-function argument |

#### Structural status per claim

- **C1 (seed Jacobian rank)**: Easily upgradable. A nonzero polynomial
  evaluated at one rational point is generically nonzero. The proof
  already invokes this density argument; only the witness is required
  in the appendix. STRUCTURAL.

- **C2 (10 solutions/branch)**: Not used in the proof; only in the
  remark. Bezout gives a structural upper bound (≤ 32 complex per
  branch from 1·1·1·2·2·2·2·4 = 128 over both branches, or 32 per
  branch using 7 of the 8 entries). Replace the "10" by "≤ 32" in
  the proof. STRUCTURAL after edit.

- **C3 (degree-4 filter)**: This is **the** critical gap. The proof
  enumerates all spurious seeds NUMERICALLY, then verifies each has
  nonzero degree-4 residual via NLS. Two failure modes:
    (a) NLS may miss spurious solutions (no a priori enumeration).
    (b) The polynomial-density argument applies to each spurious
        branch individually, but proves at most that "generically each
        of the 9 KNOWN branches has nonzero residual"; says nothing
        about possible OTHER spurious branches not enumerated.
  Status: NOT STRUCTURAL.

- **C4 (complex bootstrap)**: Per-ℓ verification is a sound
  polynomial-nonzero-at-witness argument FOR EACH FIXED ℓ. The
  generic set is the complement of L−3 proper subvarieties. Sound
  for any FIXED L ≤ 100, but the script must be re-run for each new L.
  Status: STRUCTURAL FOR FIXED L, but lacks a uniform-in-ℓ argument.

- **C5 (real per-degree rank)**: Same as C4 plus the complex→real
  density argument (Remark app-complex-real). The complex rank
  certificate at the deterministic witness extends to a real-rank
  statement once the gauge-fixed reality structure is plugged in.
  Status: same as C4.

- **C6 (T_R branch inconsistency)**: Inherits the C3 gap (it's a
  special case). The "rational function with denominator (det A4)²"
  argument is structurally sound IF the numerator polynomial is
  nonzero at the witness. So this part IS structural-modulo-witness.
  Status: STRUCTURAL given C3 witness.

#### Empirical investigation: can extra CG power entries reduce the
seed fibre below 10?

Tested at the rational witness (`paper/test_seed_augment.py`,
800 starts/branch). Augmenting the seed CG power set with each of:
  - {P(1,3,3)}: still 10/10
  - {P(2,3,3)}: still 10/10
  - {P(1,3,3), P(2,3,3)}: still 10/10
  - {P(2,2,3)} or {P(1,1,3)}: still 10/10 (these don't even involve F3)
  - all four together: still 10/10

**Conclusion**: degree-≤4 SO(3)-invariants on V_0..3 cannot resolve
the 10-fold cover. The 10:1 ramification is intrinsic to the
invariant theory of (V_0..3, SO(3)) at this Hilbert-series level.
ANY structural reduction to a 2-point seed fibre MUST use degree-≥4
input data (i.e., F_4 or higher). This validates the paper's
strategy of pushing fibre reduction to degree 4.

#### Why C3 cannot be patched by adding more low-degree entries

The 10 spurious seeds are real points of the algebraic variety cut
out by all SO(3) invariants of degree ≤4 in F_0..3 (modulo SO(3)).
Equivalently, the SO(3) GIT-quotient V_0..3//SO(3) has a self-cover
of degree (at least) 10 visible at the level of bispectrum+P
invariants. To resolve we MUST use info beyond degree 3.

#### Structural rescue plan

Replace the "numerically-eliminate-9-spurious-seeds" argument with a
classical algebraic-geometry argument applied to the JOINT system at
degrees 0-4. Specifically:

1. **Use the FULL bispectrum at L=4** as the seed. This is a constant
   number of entries (≤ 30), so the asymptotic budget Θ(L²) is
   unaffected. By Edidin–Satriano (2024), the full bispectrum at any
   fixed L separates generic O(L^3) orbits of SO(3) on real
   band-limited signals. Specialised to L=4: full bispectrum at L=4
   determines (F_0..4) up to T_R generically.
   - Status: cite as a known structural theorem.
   - Output cost: O(L²) since we add O(1) extra entries at L=4.

2. **T_R resolved at degree 4** by parity-odd entries β(2,3,4),
   β(2,4,3), β(3,4,2). Their imaginary parts flip sign under T_R
   and are nonzero generically (Cor app-first-odd). STRUCTURAL.

3. **Bootstrap for ℓ ≥ 5** using the selective family T_ℓ.
   Structural rank claim: **for every ℓ ≥ 5, det(A_ℓ) is a nonzero
   polynomial in F_0..ℓ−1**. The current per-ℓ witness shows this for
   ℓ ≤ 100. Need either:
   (a) An explicit closed-form determinant or rank argument uniform
       in ℓ, OR
   (b) A "uniform witness" — a single signal sequence (F_a)_{a≥0}
       (computable in closed form) such that det(A_ℓ) ≠ 0 at this
       sequence for every ℓ. The latter is at most a one-time
       structural lemma.

#### Status of the structural rescue

- Steps 1–2 give a STRUCTURAL replacement for the entire
  "seed + degree-4 filter + T_R resolution" block (Lemmas
  app-ell2, app-ell3, app-degree4-filter, app-TR-resolve), at the
  cost of citing Edidin–Satriano as a black-box theorem.

- Step 3 still requires a structural rank argument uniform in ℓ.
  This is the **only remaining gap** after the rescue.

#### Open structural problem: uniform bootstrap rank

For each ℓ ≥ 5, define
  A_ℓ(F_0,...,F_{ℓ-1}) ∈ C^{(2ℓ+1) × (2ℓ+1)}
to be the bootstrap matrix from the closed-form selective family
T_ℓ (Prop app-entry-counts). We need

  Q_ℓ(F_0,...,F_{ℓ-1}) := det A_ℓ(F_0,...,F_{ℓ-1}) ≢ 0 in C[F_0..ℓ-1].

PROPOSED STRUCTURAL ARGUMENT (uniform witness):

Define the deterministic "geometric" witness
  F_a^m = ζ^{a+m},   for ζ ∈ C generic,  a ≥ 1, |m| ≤ a.

Then Q_ℓ becomes a Laurent polynomial p_ℓ(ζ) in ζ. By the structure
of the closed-form family (chain rows (a,ℓ,ℓ-a), offset rows
(a,ℓ,ℓ-a+1), C-rows (a,ℓ-a,ℓ)), p_ℓ has a leading term
proportional to ∏_a CG[a,*,ℓ,*|...] which is a known Wigner 3j
product.

To finish: show that for some specific ζ_0 (e.g. ζ_0 = 1 or a small
integer) and every ℓ ≥ 5, p_ℓ(ζ_0) ≠ 0. This is a single statement
parametrised by ℓ; it can be reduced to a Wigner-3j non-vanishing
identity provable by (e.g.) the Racah formula.

This reduction was NOT executed in the present session; flagged as
the next structural step.

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

### 2026-04-16 (cont.): RESCUE COMPLETE via Kakarala 1992/2012 Theorem 7

After re-reading Edidin-Satriano 2024 carefully, my earlier rescue plan
was based on a misreading: Edidin-Satriano's `R \ge L+2` theorem is for
the MULTI-SHELL case (signals on R^3 sampled on R radial shells), NOT
for single-S^2 signals. Their single-S^2 statement only cites
[Bandeira-Blum-Smith-Kileel-Perry-Weed-Wein, 1712.10163] as a
COMPUTATIONAL verification for L <= 15. So Edidin-Satriano cannot be
used as a structural black-box for our seed.

HOWEVER: a sharper structural result is already available.

#### Kakarala 2012, Theorem 7 (the actual structural seed)

Kakarala (JMIV 2012, derived from his 1992 PhD thesis) proves:

> Let G be compact, H closed in G, N_H the normalizer of H in G. Let
> r, s ∈ L^1(G) be left-H-invariant with maximal H-rank Fourier
> coefficients. Then a3_r = a3_s iff s(g) = r(xg) for some x ∈ N_H.

Specialised to G = SO(3), H = SO(2)_z (so G/H = S^2), the normalizer
N_H = O(2) is generated by H plus the 180° rotation R_x(π) about the
x-axis. Both lie in SO(3) (R_x(π) is a proper rotation, det = +1).
Maximal H-rank for S^2 functions means F_l ≠ 0 for all l ≤ L
(every Fourier vector is nonzero).

CONSEQUENCE: For generic real-valued S^2 functions band-limited at L,
the FULL bispectrum {β_{l1,l2,l} : 0 ≤ l1 ≤ l2 ≤ L, |l1-l2| ≤ l ≤ l1+l2}
is a COMPLETE SO(3)-invariant. NO reflection ambiguity, NO finite
fibre, NO T_R issue. Edidin-Satriano's "Kakarala recovers up to
reflection" remark in their Section 1 is misleading: the "reflection"
they reference is the N_H/H = Z/2 quotient, but the non-trivial coset
representative R_x(π) is a *proper* SO(3) rotation, so it produces no
genuine residual ambiguity for S^2 functions.

#### Rescue using Kakarala Theorem 7

Replace the entire seed block (degrees 0-4) with a single citation
to Kakarala. Concretely:

(a) Augment the selective bispectrum with the FULL bispectrum at a
    fixed low degree L_0 ≥ 2 (we use L_0 = 4). The number of
    additional entries is bounded by
        |full bispectrum at L_0| ≤ (L_0+1)^3 / 6 + O(L_0^2)
    which is O(1) (for L_0 = 4: ~30 entries). The asymptotic O(L^2)
    output cost is preserved.

(b) By Kakarala Thm 7, the full bispectrum at L_0 = 4 uniquely
    determines (F_0, ..., F_4) up to SO(3), for every signal with
    F_l ≠ 0 for l ∈ {0, 1, 2, 3, 4}. In particular, after gauge
    fixing, (F_0, ..., F_4) is uniquely determined as a point in
    the gauge slice. STRUCTURAL.

This eliminates Lemmas app-ell2, app-ell3, app-degree4-filter,
app-TR-resolve, Remark app-seed-fibre-size, and the entire degree-4
NLS computational certificate. Eight pages of the proof collapse to
"by Kakarala 1992/2012, Theorem 7".

#### What survives

Only the bootstrap rank for l ≥ 5 remains. The current per-l
verification (Lem app-bootstrap-rank for complex rank, l ≤ 100;
Lem app-augmented-perdegree for real rank, l ≤ 30) is structural
"for any fixed L", which is the standard guarantee. The script can
be re-run to extend coverage. A truly uniform-in-l proof remains
open but is not necessary for the headline theorem.

Key empirical observation (test_uniform_witness.py with chain-row
conjugation bug fixed):

  - Random complex signal (no reality constraint), seed 42:
      FULL complex rank 2l+1 verified for l ∈ {8, 10, 15, 20, 30, 40,
      50, 70, 100}. Condition number stays at 10^2 - 10^3 (excellent).
  - Random real signal (with reality constraint), seed 42:
      Linear bootstrap alone has real rank ≈ l (deficient by ≈ l).
      Confirms Remark app-complex-real: chain+cross alone is
      insufficient for real signals; CG power augmentation is
      mandatory at every l.
  - Closed-form analytic witnesses (W1: F_a^m = 1; W2: F_a^m = 1/(1+m^2);
      W3: F_a^m = (1+m)*(1+a); W4: F_a^m = (-1)^a*(1+abs(m));
      W5: F_a^m = exp(i m); — all REAL-projected) FAIL at moderate l.
      No simple closed-form witness gives uniform-in-l rank.

Conclusion: a "uniform witness" structural proof in closed form
appears to require an actual Wigner-3j non-vanishing identity for
the determinant of the closed-form bootstrap family, which is a
genuine computer-algebra task beyond the scope of this session.

#### Final structural map of the proof

| Step | Status |
|------|--------|
| Gauge fixing (lem:app-gauge) | STRUCTURAL |
| F_0, F_1 recovery (lem:app-F0F1) | STRUCTURAL |
| Parity of bispectrum (lem:app-parity, lem:app-P-invariant, prop:app-odd-vanishing, cor:app-first-odd) | STRUCTURAL |
| Seed (degrees 0-4) via FULL bispectrum at L_0=4 + Kakarala Thm 7 | STRUCTURAL (cite Kakarala 1992/2012) |
| Bootstrap rank l ≥ 5 (lem:app-bootstrap-rank, lem:app-augmented-perdegree) | STRUCTURAL FOR EACH FIXED l |
| Inductive recovery (prop:app-TR-propagate, main thm) | STRUCTURAL |

The augmented selective bispectrum, augmented further with the full
bispectrum at L_0 = 4, has output size

  |selective at L| + |full at L_0=4| = Θ(L^2) + Θ(1) = Θ(L^2)

and is a complete SO(3)-invariant on generic real S^2 signals,
with the only "computer-verified per fixed L" component being the
bootstrap rank at l ∈ [5, L]. The seed is fully structural.

#### Concrete LaTeX edit plan for paper.tex

1. In the proof overview (around line 1171), drop "(3) fibre reduction
   at degree 4 via parity-breaking entries" and rephrase as:
   "(3) seed completeness for degrees 0-4 via Kakarala's bispectrum
   completeness theorem applied to the full L_0=4 bispectrum block".

2. Add a new lemma `lem:app-seed-kakarala` immediately after
   `lem:app-F0F1`:
   "For generic real signals satisfying Assumption G1-G3 with F_l ≠ 0
   for l ≤ 4, the full bispectrum at L_0=4 uniquely determines
   (F_0,...,F_4) on the gauge slice. (Kakarala 1992/2012 Theorem 7
   applied to G=SO(3), H=SO(2)_z, with maximal-H-rank coefficients
   = nonzero F_l vectors.)"

3. Delete or downgrade Lemmas app-ell2, app-ell3, app-degree4-filter,
   app-TR-resolve, Remark app-seed-fibre-size to "Historical /
   constructive remark" — the constructive seed solver remains valid
   as an algorithm even though the completeness proof no longer
   depends on it.

4. Update Definition of `Phi_aug` to include the full bispectrum
   at L_0=4 (in addition to the existing selective entries at all l).
   The output count becomes
       N_aug(L) = N_selective(L) + N_full(L_0=4) = Θ(L^2) + 30.

5. Update `thm:app-main` proof Step 4 to read:
   "By Lemma app-seed-kakarala, the seed (F_0,...,F_4) is uniquely
   determined on the gauge slice. The reflection R_x(π) ∈ N_H \ H
   acts on the gauge slice but is itself a proper SO(3) rotation;
   the gauge-fixed parameterisation already breaks this Z/2."
   (No more parity-breaking argument needed for T_R; T_R was already
   resolved by Kakarala because his "ambiguity up to N_H" is
   ambiguity up to a proper SO(3) rotation, not an O(3) reflection.)

6. Drop the "Computational certificates" items 2 (seed fibre
   enumeration) and 3 (degree-4 fibre reduction).

7. In Table around line 1716, update "Edidin & Satriano (2024)" row
   to also mention "(multi-shell only)" since their R ≥ L+2 result
   does not apply to single-shell S^2.

The user-visible cost of this rescue is exactly: a fixed Θ(1)
additional output entries at low degree, and one extra citation
to Kakarala. In return, the proof becomes structurally complete
modulo per-fixed-L bootstrap rank verification.

## Update: dual-citation strengthening of structural seed (post-rescue)

Refined `lem:app-seed-kakarala` to invoke TWO independent
structural arguments instead of relying on Kakarala alone:

  (i)  Representation-theoretic via Kakarala 1992/2012 Theorem 7
       (proved by Iwahori-Sugiura duality on G/H), with N_H ⊂ SO(3)
       so no O(3)-reflection residue.

  (ii) Symbolic Jacobian-rank certificate of Bandeira et al. 2017
       (their orbit-recovery paper, valid for all L₀ ≤ 15, hence
       certainly for L₀ = 4). One-time computer-algebra check
       over a number field, NOT a numerical witness — fully
       structural.

This insulates the seed step from any reader who is uncomfortable
trusting the Kakarala 1992 thesis as the sole source for Theorem 7.

Also added `rem:app-edidin-reflection` explicitly explaining that
Edidin-Satriano's "up to reflection" caveat refers to the
*algorithm* in Kakarala Section 5 (positive-square-root step
introduces an O(3) ambiguity that the recursion propagates), NOT
to the bispectrum data. The full bispectrum at any L₀ ≥ 4 contains
β_{2,3,4} (smallest all-distinct odd-parity entry), which flips
sign under T_R, separating f from T_R · f. So Edidin-Satriano's
caveat is correct *for the algorithm* but does not weaken the
completeness statement of Theorem 7.

Bibtex `bandeira2017estimation` added to references.bib.
Compilation clean (tectonic, no undefined references/citations).

## Update: deep rewrite to a CONDITIONAL completeness theorem (`proof_clean.tex`)

Outcome: the proof in `paper/proof_clean.tex` is now defensible as a
conditional theorem on two explicitly named hypotheses, with no
remaining "Kakarala-proves-the-seed" overclaim and no "local Jacobian
rank ⇒ global uniqueness" inference.

### Why the previous "rescue" was still too strong

After re-reading Kakarala 2012 Theorem 7 carefully against a strict
reviewer pass, the rescue strategy above (apply Kakarala to the
finite L₀=4 block) is NOT a valid proof:

- Theorem 7 requires the Fourier coefficients to have maximal H-rank
  for ALL irreducible representations α of G, equivalently at least
  one nonzero coefficient at every degree.
- A band-limited truncation has a_α = 0 for every α > L_seed by
  construction, so the maximal-H-rank hypothesis fails immediately.
- Kakarala 1992/2012 therefore cannot be invoked as an unconditional
  proof of finite-band seed injectivity. The 30-entry block buys
  inspiration only.

The Bandeira et al. 2017 symbolic Jacobian is computer-algebra
EVIDENCE for the seed map, not a published theorem about scalar
S² bispectrum injectivity at L_seed ≤ 7. It cannot stand in for the
seed proof either, only support it.

### What the rewritten proof actually proves

`thm:main` in `proof_clean.tex` is now stated as

  Conditional on
    Hyp 1 (finite seed injectivity for L_seed ≤ 7) and
    Hyp 2 (Δ_ℓ ≢ 0 for every ℓ ≥ 8),
  Φ_aug separates SO(3)-orbits of generic real-valued band-limited
  signals, with |Φ_aug| = Θ(L²).

The two hypotheses are isolated, named, and explicitly listed as open
problems. The proof itself is unconditional given the hypotheses.

### Per-degree bootstrap is now a SQUARE COMPLEX LINEAR system

For ℓ ≥ 8, the selected block T_ℓ has 2ℓ+1 entries:

  T_ℓ = {(a,ℓ,ℓ-a) : 1 ≤ a ≤ ℓ-1}                 (chain rows, conj F_ℓ)
      ∪ {(a,ℓ,ℓ-a+1) : 2 ≤ a ≤ ℓ-1}               (cross rows, F_ℓ)
      ∪ {(a,ℓ-a,ℓ) : 1 ≤ a ≤ 4}                   (chain rows, conj F_ℓ)

After conjugating the chain equations, all rows are linear in F_ℓ
(over C), giving a square system A_ℓ(F_<ℓ) F_ℓ = b_ℓ in
C^{(2ℓ+1)×(2ℓ+1)}. Inductive uniqueness needs only

  Δ_ℓ := det A_ℓ ≢ 0 as a polynomial in real coords of F_<ℓ.

That is Hypothesis 2. No more nonlinear CG-power systems, no more
"real Jacobian rank ≥ 2ℓ+1 implies F_ℓ = F'_ℓ" overclaim. The
induction step is now literally:

  Φ_aug(f)=Φ_aug(f') ⇒ F_<ℓ = F'_<ℓ
                     ⇒ A_ℓ(F_<ℓ) F_ℓ = A_ℓ(F_<ℓ) F'_ℓ
                     ⇒ F_ℓ = F'_ℓ.

### CG-power is no longer load-bearing in the proof

Definition `def:Phi-aug` sets S_P = ∅ in the formal theorem.
CG-power entries are kept ONLY as optional implementation diagnostics
(`rem:cg-optional`). Section `sec:cgpower` was rewritten:
"Why CG power augmentation is necessary" → "CG-power diagnostics".
The earlier real-Jacobian-rank deficit table (rank-4 vs target 9 etc.)
is now attributed to "an earlier real-valued implementation" of the
selected bispectrum, not to the proof.

The corrected logic: invertibility of A_ℓ over C already forces
uniqueness of F_ℓ on the real-signal slice. There is no real-rank gap
inside the proof itself, so CG-power need not close any gap. Adding
extra invariants cannot destroy completeness, so CG-power is harmless
to ship in the implementation.

### Numerical "L ≤ 100 certificate" downgraded to "evidence"

Floating-point checks of det A_ℓ ≠ 0 for ℓ = 8…100 (singular-value
ratio ≥ 10⁻⁴) are NOT a mathematical certificate. The text now says:

- C2 "Bootstrap determinant evidence":  strong numerical evidence,
  not a certificate.
- "Verified band-limits" corollary removed; replaced by
  `rem:numerical-evidence` saying rigorous proof requires exact
  arithmetic, interval arithmetic, or a perturbation bound certifying
  distance from singularity.

This is honest about what `verify_linear_bootstrap.py` actually
proves: per-(ℓ, fixed witness) numerical full rank. It does NOT
prove Δ_ℓ ≢ 0 as a polynomial — that requires symbolic / interval
work that we have not done.

### Kakarala's role in the document

Demoted from "proves the seed" to "motivates the structural
template". Body text now explicitly says Theorem 7 is invoked as
inspiration for "constant-size full low-degree block + per-degree
recurrence", not as a proof of `hyp:seed`. The bandeira2017estimation
remark is kept as "possible symbolic certificate" — open.

### β_{2,3,4} parity corollary made symbolic and exact

Old draft used a NumPy seed witness (β_{2,3,4} ≈ -0.9149i at seed 42)
and incorrectly cited CG(2,2; 3,2 | 4,4) = 2√(2/5).

Corrected:
- CG(2,2; 3,2 | 4,4) = √(2/5) exactly (verified, every CG ≤ 1).
- Closed-form witness:  a_2^2 = a_3^2 = 1, a_4^4 = i, all other
  degree-2/3/4 coefficients zero, negative-m fixed by reality.
  Only (m_1,m_2,m) ∈ {(2,2,4),(-2,-2,-4)} contribute.
  β_{2,3,4} = √(2/5)·1·1·(-i) + (-√(2/5))·1·1·(+i) = -2i√(2/5) ≠ 0.
  Verified numerically: -1.2649i = -2·0.6325·i. ✓

### Repeated-index vanishing generalised via the symmetric tensor

Old "self-coupling CG symmetry" argument only handled (a, a, ℓ).
Replaced with the full symmetric tensor

  T_{ℓ_1,ℓ_2,ℓ_3} := Σ (3j-symbol)·a_{ℓ_1}^{m_1} a_{ℓ_2}^{m_2} a_{ℓ_3}^{m_3}.

Conversion 3j → CG plus reality identity gives

  β_{ℓ_1,ℓ_2,ℓ} = (-1)^{ℓ_1+ℓ_2}·√(2ℓ+1)·T_{ℓ_1,ℓ_2,ℓ}.

3j is invariant under even column permutations, picks up
(-1)^{ℓ_1+ℓ_2+ℓ_3} under odd ones. Two columns with equal degree
insert the SAME signal coefficients, so swap → T = ±T → T = 0 when
parity is odd. This handles (a,a,ℓ), (a,ℓ,a), (ℓ,a,a) uniformly.

### Genericity phrased on the gauge slice

(G1)–(G5) are polynomial inequalities in gauge-fixed real coords, so
they cut out a Zariski-open V ⊂ G in the gauge slice G of `lem:gauge`.
The generic set in V_L is the SO(3)-saturation SO(3)·V. The previous
phrasing "Zariski-open subset of V_L" was sloppy because aligning F_1
with ẑ is not a global polynomial map on V_L.

### Index-set deduplication

Old `def:Phi-aug` had S_β AND a separate "full low-degree seed block"
in the concatenation, while the Index-sets paragraph said S_β already
contained the seed block — a formal duplicate. Fixed by introducing
S_seed and defining S_β := S_seed ∪ ⋃_{ℓ≥8} T_ℓ. Φ_aug is now a
single concatenation indexed by S_β.

### Title / naming honesty

Old: "Completeness of the Augmented Selective SO(3)-Bispectrum on S²".
New: "Completeness of a Selective SO(3)-Bispectrum on S²
(with optional Clebsch–Gordan power diagnostics)".

The body and abstract clarify: the proof is for S_P = ∅; "augmented"
refers to the implementation variant that ships the optional CG-power
block.

### What is OPEN after this round

1. Hyp 1: Finite seed injectivity for L_seed ≤ 7. Needs a real
   theorem (e.g. closed-form Gröbner / resultant non-vanishing on the
   gauge slice). Bandeira 2017 is supportive evidence only.

2. Hyp 2: Δ_ℓ ≢ 0 uniformly for all ℓ ≥ 8. Needs an exact closed-form
   identity, or interval-arithmetic checks per ℓ. Floating-point
   passes for ℓ ≤ 100 do not constitute proof.

3. The proof is correct CONDITIONAL on (1) and (2). No further gaps
   identified by the latest reviewer pass:
   "Yes — as a conditional proof, this version is basically correct."

### Files

- `paper/proof_clean.tex` — current canonical proof (152 KB PDF, builds
  cleanly under tectonic, no undefined refs, no overfull hboxes).
- `benchmarks/verify_linear_bootstrap.py` — produces ℓ=4..12 numerical
  full-rank evidence for Hyp 2 at fixed seeds. NOT a certificate.
- `paper/exact_certificates.py` — mpmath-based Jacobian rank checks
  for L ∈ {4,5}. Provides triangular local uniqueness only; does NOT
  prove Hyp 1 globally.
