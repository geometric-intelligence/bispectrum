# Response to Reviewer AZ1B

Thank you for recognizing that the package fills an unavailable capability,
could broaden adoption of bispectra, and is well written and thoroughly
evaluated. We agree that most group-specific formulas build on prior theory;
the contribution is to make them usable together as tested, differentiable
PyTorch components. We respectfully disagree, however, that the paper lacks a
substantive algorithmic or empirical contribution.

The work contributes more than a repackaging of formulas: (i) one uniform
autograd-compatible implementation for seven group/domain pairs, including
precomputed CG/DFT/Bessel buffers and selective sparse contractions; (ii) an
optimized and profiled implementation reducing full representations by up to
$512\times$; (iii) a new augmented selective $\mathrm{SO}(3)$-on-$S^2$
construction with an asymptotically optimal $\Theta(L^2)$ coefficient count,
an $\Omega(L^2)$ lower bound, a parity/vanishing analysis, and an explicit
diagnosis and repair of the real-signal rank deficit; and (iv) the first
controlled evaluation of these invariants as pooling layers across planar,
volumetric, and spherical tasks.

The empirical benefit is also not confined to very-low-data subsets. At full
Spherical MNIST data, the augmented spherical invariant reaches 95.0% versus
79.2% for a matched-capacity power-spectrum model, isolating information loss
rather than classifier size. At 10% PCam, it is within 0.4 AUC points of the
best method, and at full-data OrganMNIST3D it matches max pooling at moderate
capacity. The low-data gains are where completeness helps most, but not the
only demonstrated functionality: the library also provides near-exact
invariance without augmentation and preserves orbit information that norm/max
pooling discards.

## Q1: What made the spherical extension technically difficult?

The finite-group selection argument cannot be transferred by replacing a
finite irrep index with $\ell$:

1. **Homogeneous-space rank deficiency.** For signals on a finite group, each
   generic Fourier coefficient is a nonsingular matrix, which supports the
   recursive argument of Mataigne et al. On $S^2$, the coefficient at degree
   $\ell$ is only a vector
   $\mathbf F_\ell\in\mathbb C^{2\ell+1}$. Kakarala's full-rank
   operator-bispectrum theorem therefore does not directly apply.
2. **A linear chain is dimensionally insufficient.** A band-limited real
   spherical signal has $(L+1)^2$ degrees of freedom modulo three rotational
   degrees. We prove that every smooth/polynomial complete invariant therefore
   needs at least $(L+1)^2-3=\Omega(L^2)$ live components; an $O(L)$
   Mataigne-style frequency chain cannot be complete.
3. **Reality and parity break the naive bootstrap.** On real signals, each
   scalar bispectrum entry is either real or purely imaginary, and every
   odd-parity entry with a repeated index vanishes identically. After removing
   these zero rows, the nominal $(2\ell+1)$-row bootstrap block has generic
   rank only approximately $\ell+2$, not $2\ell+1$.
4. **Rank must be repaired without losing selectivity.** Our construction adds
   mandatory even self-couplings and greedily selected degree-4 CG-power
   invariants until the per-degree real Jacobian reaches full rank. This keeps
   $O(\ell)$ entries per degree and hence $\Theta(L^2)$ overall.

These are the reasons the result is an *augmented selective invariant*, rather
than a direct application of the finite-group algorithm. We will foreground
this technical chain in Section 5.

## Q2: What would a completeness proof require?

We conjecture generic completeness; the current Jacobian and reconstruction
tests are evidence, not a proof. A proof would require:

1. prove that the full low-degree seed separates a generic orbit and fixes a
   common rotational gauge (including orientation, using a nonzero odd-parity
   triple);
2. induct on $\ell$, assuming all lower-degree coefficients have been recovered
   in that common gauge;
3. prove that the selected bootstrap, self-coupling, and CG-power equations
   have a **unique generic real solution** for $\mathbf F_\ell$; and
4. characterize the algebraic exceptional set where a seed coefficient
   vanishes, a rank drops, or a nontrivial stabilizer remains.

The unresolved conceptual step is (3). Full generic Jacobian rank gives local
identifiability but does not exclude a second, globally separated solution.
The finite-group matrix-inverse proof does not resolve this because the
spherical coefficients are rank-deficient vectors, and the real-signal parity
identities create extra algebraic dependencies. A proof likely needs either an
explicit rational frequency-marching inverse or an invariant-field argument
showing that the selected polynomials generate the generic orbit-separating
field. We will add this proof roadmap and state “conjectured generic
completeness” explicitly.

## Q3: S2CNN baseline and inference

We agree that the published S2CNN values should appear in Table 4 rather than
only in prose, and will move them there. We did not run S2CNN in the submitted
experiments, so we cannot infer an apples-to-apples speed advantage from its
published accuracies. [[E6: report matched hardware/batch S2CNN and
bispectrum+MLP latency, including whether SHT time is included. If not run,
replace with: “A reliable matched re-run was not feasible during the response
window; we will retain the accuracy comparison, explicitly label it as
published, and remove any implication of a measured speed advantage over
S2CNN.”]]

## Corrections

Thank you for catching the three presentation issues. We will (i) correct the
coefficient notation/cross-reference following Eq. (3), (ii) add the formal
definition that an invariant $\Phi$ is complete on a set $U$ when
$\Phi(f)=\Phi(h)$ iff $h=g\!\cdot\!f$ for some $g\in G$ and $f,h\in U$, and
(iii) change `TorusDnTorus` to the library's actual class name,
`TorusOnTorus`.
