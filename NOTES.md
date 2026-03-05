# Implementation Notes

Developer notes on implementing the G-bispectrum for cyclic and dihedral groups.
These document pitfalls, non-obvious decisions, and lessons learned.

## CnonCn (cyclic groups)

Straightforward. The cyclic bispectrum is just scalar triple products of FFT
coefficients — no CG matrices, no matrix algebra. The only subtlety is in
inversion:

- **SO(2) vs C_n indeterminacy**: The phase of F(rho_1) is fixed to 0 during
  inversion (Algorithm 1, step 2). This absorbs not just discrete C_n shifts
  but a *continuous* SO(2) rotation. The recovered signal is related to the
  original by a continuous cyclic shift, not necessarily an integer one. This
  means `ifft(fhat_recovered)` is complex, not real. Tests compare DFT
  magnitudes, not the spatial signal directly.

## DnonDn (dihedral groups)

This was significantly harder. Notes organized by component.

### CG matrix computation (`_compute_cg`)

We need CG matrices C such that `kron(rho_i(g), rho_j(g)) = C [oplus rho(g)] C^T`
for all g in D_n. The g-invariance repo used `scipy.linalg.schur`; we replaced
it with `torch.linalg.eig` on the 4x4 orthogonal matrix `kron(rho_i(a), rho_j(a))`.

**Pitfall 1 — eigenvalue sign convention**: The complex eigenvectors of an
orthogonal matrix give a basis where the Schur block is R(-theta), not R(+theta).
If you just take Re/Im of the eigenvector for eigenvalue e^{i*theta}, you get
R(-theta) in the Schur form. But we need R(+|theta|) to match the standard irrep
convention rho_k(a) = R(2*pi\*k/n). The fix: negate u2 (the imaginary-part vector)
when theta > 0 to flip the sign. When theta < 0, R(-theta) = R(|theta|) already
has the right sign.

**Pitfall 2 — degenerate real eigenvalues**: When rho_i x rho_j decomposes into
two 1D irreps with the same eigenvalue under the rotation generator a (e.g. rho_0
and rho_01 both have eigenvalue +1 on a), `torch.linalg.eig` gives a
2-dimensional eigenspace but doesn't split it into the two 1D irreps. To resolve
this, restrict the reflection generator Q_x = kron(rho_i(x), rho_j(x)) to the
degenerate subspace and diagonalize it with `torch.linalg.eigh`. The eigenvectors
of Q_x within that subspace separate the 1D irreps (which differ in their x
eigenvalue: +1 vs -1).

**Pitfall 3 — basis alignment for reflections**: Even after getting the correct
rotation blocks, the 2D subspace basis must be rotated so that the reflection
generator x acts as diag(1, -1) (not some other reflection matrix). The reflection
representation restricted to the 2D subspace is a general reflection matrix
`[[cos2a, sin2a],[sin2a, -cos2a]]`; rotating by angle alpha aligns it with
diag(1,-1). Without this, the Fourier coefficients F(rho_k) don't have the
standard form and the bispectrum formula gives wrong results.

### Bispectrum formula (forward pass)

The non-commutative bispectrum (Theorem 3.1) is:

```
beta_{rho1,rho2} = [F(rho1) x F(rho2)] C [oplus F(rho)^T] C^T
```

**Pitfall 4 — matrix multiplication order**: The initial implementation had the
Kronecker product and CG matrix in the wrong order, which broke D_n invariance
completely. The formula must be read carefully: the Kronecker product `F1 x F2`
multiplies from the left, then C, then the block-diagonal Fplus^T, then C^T.
For real irreps, dagger = transpose. Getting this wrong gives a bilinear form
that is NOT invariant under the group action.

Debugging approach: test invariance for a single rotation by a (shift=1) on a
random signal. If max error is O(1) rather than O(1e-7), the formula is wrong.
Compare element by element against a brute-force computation from the definition.

### Group DFT

**Pitfall 5 — einops removal**: The g-invariance code uses
`einops.rearrange(rho, '(c1 c2 w) h -> c1 c2 w h', c1=2, c2=2)` to reshape the
rotation tensor. This was replaced with explicit construction of a (n2d, 2, 2, n)
tensor using cos/sin. The reflection version `rho_ref` is the same but with
column 1 negated (`rho_ref[:, :, 1, :] *= -1`), corresponding to the
diag(1,-1)^m factor.

The DFT is currently O(n^2) because it materializes the full rotation tensor and
does an einsum. This can be replaced with two calls to `torch.fft.fft` (O(n log n))
by observing that each entry of the 2x2 Fourier coefficient matrix is a linear
combination of Re/Im parts of the standard FFT. See the FFT plan for details.

### Inversion (Algorithm 3)

**Step 1 — F(rho_0)**: Same as cyclic case. F(rho_0) = sign(b00) * |b00|^{1/3}.

**Step 2 — F(rho_1) and the O(2) indeterminacy**: This was the hardest part of
the entire implementation. beta\_{rho0,rho1} / F(rho0) = F1^T F1 is a 2x2
symmetric positive semidefinite matrix. Any Q in O(2) gives an equally valid
F1' = Q F1 satisfying F1'^T F1' = F1^T F1. This is the non-abelian analogue of
the phase indeterminacy in the cyclic case, but it's continuous O(2) instead of
SO(2).

We set F1 = V sqrt(Lambda) V^T (the symmetric PSD square root) via `torch.linalg.eigh`.

**Attempted and abandoned**: A grid search over O(2) to find the Q that best
block-diagonalizes `C^T beta_{11} (QF1 x QF1)^{-1} C` (trying to make the
off-diagonal blocks vanish). This involved:

- Coarse grid of 360 rotation angles + 360 reflected angles
- Golden-section refinement on the best candidate
- Batched evaluation of an off-diagonal norm score

This was ultimately removed because:

1. The search is fragile (many local minima, numerical noise)
2. Even with the "correct" Q, the reconstructed bispectrum only matches the
   original if Q happens to equal the true Q — which cannot be determined from
   the bispectrum alone (that's the whole point of the indeterminacy)

**Step 3 — sequential recovery**: For k = 2..n2d, recover F(rho_k) from
beta\_{rho1,rho\_{k-1}} using the known F(rho_1) and F(rho\_{k-1}):

```
oplus F = [C^T beta A^{-1} C]^T   where A = F1 x F_{k-1}
```

Extract the relevant blocks from the block-diagonal matrix.

**Pitfall 6 — overwriting already-known coefficients**: The block-diagonal matrix
from step 3 contains entries for ALL irreps in rho_1 x rho\_{k-1}, including
rho_0, rho_1, and previously recovered rho_j. An earlier version tracked "known"
irreps and skipped them. The final version overwrites everything for internal
consistency — since F1 has an O(2) ambiguity, the "known" values from step 1-2
may be inconsistent with what step 3 produces. Overwriting ensures all Fourier
coefficients are mutually consistent (even if they differ from the original by
the O(2) transformation).

### Testing lessons

**Pitfall 7 — orbit equivalence in test signals**: An early "different signals
should produce different bispectra" test used delta_e = [1,0,0,0,0,0,0,0] and
delta_a = [0,1,0,0,0,0,0,0] for D_4. But delta_a is just delta_e rotated by a —
they're in the same D_4 orbit! Their bispectra are correctly identical, so the
test `assert not torch.allclose(bsp(f1), bsp(f2))` failed. Fix: use signals that
are genuinely in different orbits, e.g. [1,1,0,...,0].

**What the roundtrip tests actually verify**:

- `test_roundtrip_bispectrum`: Only checks the first 5 scalars (beta\_{rho0,rho0}
  and beta\_{rho0,rho1}) for exact roundtrip. These are O(2)-invariant quantities
  (they only depend on F0 and F1^T F1, both of which are uniquely determined).
- `test_roundtrip_fourier_frobenius`: Checks that ||F(rho_k)||\_F matches for all
  k. Frobenius norms are O(2)-invariant (||QF|| = ||F||), so this always holds.
- The bispectrum of the *reconstructed* signal is valid — it's just the bispectrum
  of a D_n-equivalent signal, not necessarily identical to the original bispectrum
  for the beta\_{rho1,rho_k} coefficients.

### Bugs found in g-invariance

1. `n3 = n2 - 1` for odd n gives 0 when n=3 (n2=1), meaning no beta\_{rho1,rho_k}
   coefficients are computed. But beta\_{rho1,rho_1} is needed to recover F(rho_01).
2. `n >= 8` restriction is artificial (came from escnn, not math).
3. The g-invariance repo had zero tests for bispectrum correctness.

## General advice for future group implementations

- Always test G-invariance first. If `bsp(T_g(f)) != bsp(f)` for a random g and
  random f, the formula is wrong. This is the most powerful diagnostic.
- Test the DFT roundtrip (forward + inverse = identity) independently before
  testing the bispectrum. If the DFT is broken, bispectrum tests will fail in
  confusing ways.
- Use float64 for inversion tests. The eigendecomposition and matrix inversions
  accumulate error quickly in float32.
- Understand the group orbits before writing "different signals differ" tests.
- The paper has a typo in Eq. 14 / Algorithm 3: `beta_{rho1,rho2}` in the loop
  body should be `beta_{rho1,rho_{k-1}}`. The code in g-invariance handles this
  correctly despite the paper typo.
