# SO2onD2 Implementation Notes

**Module**: `src/bispectrum/so2_on_d2.py`, `src/bispectrum/_bessel.py`
**Paper**: Myers & Miolane, "The Selective Disk Bispectrum and Its Inversion, with Application to Multi-Reference Alignment", [arXiv:2511.19706](https://arxiv.org/abs/2511.19706), 2025.
**Date**: March 2026

## What this implements

The selective disk bispectrum: a rotation-invariant representation for grayscale images on the unit disk. The group is SO(2) (2D rotations), the domain is D^2 (unit disk). The transform decomposes an image into disk harmonic coefficients, then forms specific triple products that cancel out the rotation phase.

The selective version (Definition 4.2 from the paper) uses O(m) coefficients instead of the full O(m^3/N_m), and is sufficient for complete inversion (recovering the image up to a global rotation).

## Architecture decisions

### Following the existing codebase pattern

The existing modules (`CnonCn`, `DnonDn`, `SO3onS2`) all follow the `DESIGN.md` contract:

- `nn.Module` with no trainable parameters
- Constructor precomputes everything into registered buffers
- `forward(f)` takes raw signals, handles the transform internally
- `invert(beta)` recovers the signal up to group-action indeterminacy
- `output_size` and `index_map` properties
- Class name follows `{Group}on{Domain}` convention

`SO2onD2` mirrors `CnonCn` most closely because SO(2) is commutative, so no Clebsch-Gordan matrices are needed. The bispectrum reduces to scalar triple products of disk harmonic coefficients, just like `CnonCn` uses scalar triple products of DFT coefficients.

The `selective=False` path raises `NotImplementedError`, same pattern as `DnonDn`.

### No new dependencies

The initial plan called for `scipy.special.jv` and `scipy.special.jn_zeros` for Bessel functions. This was rejected to stay consistent with `DESIGN.md` principle 9 ("Minimal dependencies"). Instead, a pure-torch Bessel module was written from scratch (`_bessel.py`), following the same pattern as `_cg.py` for internal utilities.

No external JSON files either. The `SO3onS2` module loads precomputed CG matrices from a JSON file, which works for that case but feels fragile. For SO2onD2, everything (Bessel roots, normalization constants, basis matrices) is computed at construction time.

## The Bessel function implementation (`_bessel.py`)

### `bessel_jn(n, x)` — J_n(x) via forward recurrence

Uses `torch.special.bessel_j0` and `bessel_j1` as base cases, then the standard recurrence:

```
J_{k+1}(x) = (2k/x) * J_k(x) - J_{k-1}(x)
```

Forward recurrence is numerically stable when x >= n (the "dominant" direction), which holds for our use case: we evaluate J_n at points lambda * r where lambda is a Bessel root (lambda >= n) and r in [0, 1].

**Known limitation**: Forward recurrence suffers from catastrophic cancellation near zeros of J_n for n >= 2. At a root of J_2, we're computing a near-zero quantity as the difference of two O(0.1) numbers, losing ~7 digits. This means our "roots" of J_2 are slightly shifted from the true mathematical roots — they're actually the zeros of the *numerically computed* J_2. Everything is self-consistent (the DHT basis evaluates J_n using the same recurrence), so the bispectrum properties still hold. The absolute displacement is ~6e-5 for J_2 roots, negligible for practical purposes.

scipy uses completely different algorithms (continued fractions, Steed's method) that don't suffer from this, but adding scipy as a dependency was vetoed.

### Root finding — the interlacing approach

The first attempt used McMahon's asymptotic expansion for all orders. This works beautifully for J_0 (the expansion is accurate to machine precision even for the first root), but for higher orders the asymptotic regime starts later and the initial guess can be far off.

The second attempt added Newton polishing to the McMahon initial guess. This worked but was fragile — Newton can diverge if the initial guess is in the wrong basin.

The final implementation uses the **interlacing property** of Bessel roots:

```
j_{n-1, k} < j_{n, k} < j_{n-1, k+1}
```

This means if you know all roots of J\_{n-1}, you have guaranteed brackets for every root of J_n. The algorithm:

1. Compute J_0 roots via McMahon + Newton polish (very accurate)
2. For each order n = 1, 2, ..., n_max:
   - Use consecutive roots of J\_{n-1} as brackets [a, b]
   - Find the root of J_n in each bracket via Newton-bisection hybrid
   - These become the brackets for order n+1

This is computed in a single pass (`compute_all_bessel_roots`), sharing intermediate results. The Newton-bisection hybrid within each bracket is guaranteed to converge because we have a sign change.

Performance-wise, the scalar evaluation helpers `_jn_scalar` and `_djn_scalar` avoid creating torch tensors in the inner loop, which matters when you're doing thousands of Newton iterations.

### Precision subtlety

A nasty bug: `_jn_scalar` was originally calling `torch.tensor(x)` without specifying dtype. Python floats are float64, but `torch.tensor(x)` infers `torch.float32` by default (at least on some torch versions / configs). This silently halved the precision of all root-finding, making roots accurate to only ~1e-7 instead of ~1e-14.

Fix: explicitly `torch.tensor(x, dtype=torch.float64)` everywhere in the scalar helpers.

## The disk harmonic transform

### The problem with the complex basis

The natural basis for functions on the disk is:

```
psi_{nk}(r, theta) = c_{nk} * J_n(lambda_{nk} * r) * exp(i * n * theta)
```

For real-valued functions f, the coefficients satisfy the conjugate symmetry a\_{-n,k} = conj(a\_{n,k}). So the "full" basis with both positive and negative n is redundant for real signals: psi\_{n,k} and psi\_{-n,k} are complex conjugates on the real line, making them linearly dependent as columns of the (real) design matrix.

The initial implementation built a complex synthesis matrix Psi of shape (p_disk, m) and computed the DHT as pinv(Psi) @ f. This was a disaster — the condition number was ~6e20, the matrix was rank-deficient (rank 48 out of 50 columns for L=8), and DHT roundtrips had 14-100% error.

### The real basis reformulation

The fix was to reformulate the DHT using a real basis. For a real signal f, the expansion is:

```
f(r, theta) = sum_{k} c_{0k} * J_0(lambda_{0k} * r) * a_{0,k}
            + sum_{n>0, k} 2*c_{nk} * J_n(lambda_{nk} * r) * [Re(a_{n,k}) * cos(n*theta) - Im(a_{n,k}) * sin(n*theta)]
```

This gives a real matrix Phi of shape (p_disk, d_real) where:

- Each (0, k) pair contributes 1 column: c * J_0(lambda * r)
- Each (n, k) pair with n > 0 contributes 2 columns: the cos and sin components

The dimension d_real = K_0 + 2\*(m_nonneg - K_0), which equals the total number of coefficients including +/- n pairs. The analysis (forward DHT) is pinv(Phi) @ f_disk, and synthesis (inverse DHT) is Phi @ x_real.

The `_real_to_complex` and `_complex_to_real` helpers convert between the real parameter vector and complex DH coefficients. For n=0, the coefficient is just a real number. For n>0, a\_{n,k} = x[col_re] + i * x[col_im].

This reformulation dropped the condition number to ~1e3, with only 1-2 unresolvable basis functions per grid (the very highest frequency components that alias on the discrete pixel grid). We use `rcond=1e-10` in `torch.linalg.pinv` to zero out those singular values.

### Bandlimit convention

The paper states bandlimit lambda = pi * L / 2, but this alone doesn't pin down the exact number of basis functions m, because you have to decide how to handle (n, k) pairs where lambda\_{nk} is very close to the threshold.

After much debugging, the convention that reproduces the paper's Table 1 exactly is the `fle-2d` convention from Marshall et al. (2023):

```
ne = floor(L^2 * pi / 4)
```

This is approximately the number of pixels inside the unit disk. All (n, k) pairs are sorted by ascending Bessel root, and the first ne are kept. This determines m, N_m, and K_n implicitly.

One wrinkle: the ne cutoff can split a conjugate pair (n, k) and (-n, k). If the last selected pair has |n| > 0 and its partner was cut off, we drop it too. This ensures the real basis matrix has consistent dimensions. Without this, L=28 gives 316 coefficients instead of 315.

## The selective bispectrum

### Forward pass

The selective bispectrum (Definition 4.2) has two types of coefficients:

**Type 0**: b\_{0,0,k} = a\_{0,1}^2 * conj(a\_{0,k}) for k = 1..K_0

This is analogous to the (0,0) and (0,1) terms in the CnonCn selective bispectrum. The a\_{0,1}^2 factor is real (since a\_{0,k} is always real for real signals), so this extracts the radial profile at zero angular frequency.

**Type 2**: b\_{2,n,k} = a\_{1,1} * a\_{n,1} * conj(a\_{n+1,k}) for n = 0..N_m-1, k = 1..K\_{n+1}

This chains adjacent angular frequencies together: knowing a\_{n,1} lets you recover a\_{n+1,k} from b\_{2,n,k}. The a\_{1,1} factor serves as the "anchor" (its magnitude is known, its phase is absorbed by the rotation indeterminacy).

Total count: N = sum\_{n=0}^{N_m} K_n.

The implementation precomputes index tensors (`_type0_a0k_idx`, `_type2_an1_idx`, `_type2_anp1k_idx`) at construction time, so the forward pass is fully vectorized — no Python loops.

### Invariance

Under rotation by phi, the DH coefficients transform as:

```
a'_{n,k} = exp(i * n * phi) * a_{n,k}
```

For Type 0:
b'_{0,0,k} = (a'_{0,1})^2 * conj(a'_{0,k})
= a_{0,1}^2 * conj(a\_{0,k})
= b\_{0,0,k}

(because exp(i * 0 * phi) = 1 for all n=0 terms.)

For Type 2:
b'_{2,n,k} = a'_{1,1} * a'_{n,1} * conj(a'_{n+1,k})
= exp(i*phi) * a\_{1,1} * exp(i*n*phi) * a\_{n,1} * conj(exp(i*(n+1)*phi) * a\_{n+1,k})
= exp(i*phi) * exp(i*n*phi) * exp(-i\*(n+1)*phi) * a\_{1,1} * a\_{n,1} * conj(a\_{n+1,k})
= exp(i*(1 + n - n - 1)\*phi) * b\_{2,n,k}
= b\_{2,n,k}

The phases cancel exactly. This is verified by `test_analytical_rotation_invariance` (which operates on coefficients directly, no grid discretization involved).

Spatial rotation invariance (rotating the actual image) has additional discretization error from grid_sample interpolation plus the DHT's finite resolution. This error is ~O(1/L) and is tested with a loose tolerance.

## Inversion

### The bootstrap (Theorem 4.4)

The inversion follows the same structure as `CnonCn.invert()`:

1. **Recover a\_{0,1}** from b\_{0,0,1} = |a\_{0,1}|^2 * a\_{0,1} = |a\_{0,1}|^3 * exp(i * arg(a\_{0,1})).
   So |a\_{0,1}| = |b\_{0,0,1}|^{1/3} and the phase is arg(b\_{0,0,1}).
   (For real signals, a\_{0,1} is real, so this just gives a\_{0,1} = b\_{0,0,1}^{1/3}.)

2. **Recover a\_{0,k}** for k >= 2 from b\_{0,0,k} / a\_{0,1}^2 = conj(a\_{0,k}), so a\_{0,k} = conj(b\_{0,0,k} / a\_{0,1}^2).

3. **Recover |a\_{1,1}|** from b\_{2,0,1} = a\_{1,1} * a\_{0,1} * conj(a\_{1,1}) = a\_{0,1} * |a\_{1,1}|^2.
   So |a\_{1,1}| = sqrt(|b\_{2,0,1} / a\_{0,1}|). The phase of a\_{1,1} is the rotation indeterminacy — we set it to 0.

4. **Recover a\_{1,k}** for k >= 2 from b\_{2,0,k} = a\_{1,1} * a\_{0,1} * conj(a\_{1,k}).

5. **Sequential bootstrap for n >= 1**: b\_{2,n,k} = a\_{1,1} * a\_{n,1} * conj(a\_{n+1,k}).
   Since a\_{n,1} was recovered in the previous step, we get a\_{n+1,k} = conj(b\_{2,n,k} / (a\_{1,1} * a\_{n,1})).

After recovering all coefficients, the inverse DHT reconstructs the image.

### Precision considerations

The inversion involves divisions by a\_{1,1} and a\_{n,1}. The paper's assumption that these are nonzero is essential — if any a\_{n,1} = 0, the chain breaks at that angular frequency. In practice with noisy images this is never an issue, but synthetic test signals need to be constructed carefully (adding a constant bias `+ 0.5` to the random coefficients helps).

The full roundtrip (f -> bispectrum -> invert -> f') goes through the DHT twice, so the ~1-2 unresolvable basis functions cause accumulated error. The test asserts that >85% of coefficients are recovered within 20% relative error, rather than demanding exact reconstruction.

## Test structure

45 tests organized into 5 groups:

- **TestBessel** (10 tests): J_0 and J_1 match torch.special, J_n at zero, root locations, roots-are-actual-zeros, monotonicity, batch root computation consistency.

- **TestSO2onD2Construction** (7 tests): instantiation, no trainable params, extra_repr, index_map, coefficient counts against paper Table 1 (L=8: 27, L=16: 105, L=28: 315), explicit bandlimit, full-not-implemented.

- **TestSO2onD2Forward** (7 tests): output shape, dtype, determinism, different-signals-differ, batch-size-one, analytical rotation invariance (coefficient-level), spatial rotation invariance (grid_sample).

- **TestSO2onD2DHTRoundtrip** (4 tests): bandlimited signal roundtrip, single harmonic roundtrip.

- **TestSO2onD2Invert** (6 tests): exact coefficient recovery, direct coefficient-level inversion (no DHT), output shape, output dtype, full-not-implemented, bispectrum roundtrip (bsp(invert(bsp(f))) ≈ bsp(f)).

## Performance characteristics

Construction is slow for large L because of the Bessel root computation (pure Python loops with Newton iterations) and the pseudoinverse computation. For L=28, construction takes ~15s. For L=64, it would take minutes. The forward pass itself is fast (just matrix multiply + vectorized indexing).

If performance becomes a bottleneck, the main optimization targets are:

1. Cache Bessel roots (they only depend on n_max and k_max)
2. Use the fast DHT from Marshall et al. 2023 (O(L^2 log L) instead of O(L^3))
3. Implement Bessel functions in C++/CUDA

## Files

```
src/bispectrum/
├── _bessel.py        # bessel_jn, bessel_jn_zeros, compute_all_bessel_roots
├── so2_on_d2.py      # SO2onD2 module
└── __init__.py        # updated to export SO2onD2

tests/
└── test_so2_on_d2.py  # 45 tests
```

## What's not implemented

- **Full disk bispectrum** (`selective=False`): raises NotImplementedError. Would need O(m^3/N_m) coefficients — straightforward but expensive.
- **MRA correction** (Proposition 4.6): the bias term delta for noisy observations. Deferred to a follow-up.
- **Fast DHT**: the O(L^2 log L) approximation from Marshall et al. 2023. Currently using the naive O(L^3) pseudoinverse approach.
- **GPU optimization**: everything runs on CPU in float64. The module supports `.to(device)` via registered buffers, but the construction-time computation is CPU-only.

## Possible paper issues

Some things noticed during implementation that may warrant further examination:

- The paper's Table 2 shows m=46 for L=8 with bandlimit lambda = pi\*L/2. But the ne = floor(L^2 * pi/4) = 50 convention gives slightly different index sets depending on how ties in Bessel root ordering are broken. The selective coefficient count (27) matches Table 1 exactly, so the conventions are consistent where it matters.

- The inversion theorem requires a\_{n,1} != 0 for all n. This is stated as a mild assumption ("does not impact real-world application as long as there is some noise"), but for deterministic signals it can genuinely fail. A signal that's purely radial (only n=0 components) has all a\_{n,1} = 0 for n > 0, and the inversion chain breaks completely.

- The normalization constant c\_{nk} = 1 / (sqrt(pi) * |J\_{n+1}(lambda\_{nk})|) follows from the orthonormality condition on the disk with measure r dr dtheta. The paper doesn't give this formula explicitly — it's derived from the standard Bessel function integral identity.
