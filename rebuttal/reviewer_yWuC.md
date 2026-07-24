# Response to Reviewer yWuC

Thank you for the careful, technically precise review and for recognizing the
value of the uniform tested interface, the optimal-order coefficient count,
and the low-data results. We agree that the original text did not separate the
theoretical status of the finite-group construction from that of the spherical
construction sharply enough.

## Status of the spherical invariant

**Generic completeness is conjectured, not proved.** Our evidence is empirical:
generic Jacobian-rank tests and signal reconstruction. We will state this
explicitly in the abstract, Section 5, Table 1, and Limitations. We will also
use “augmented selective invariant” for the full output
$\Phi_{\mathrm{sel}}$ and reserve “selective bispectrum” for its degree-3
bispectral subset. The distinction matters because the bootstrap bispectral
block has generic rank approximately $\ell+2$, rather than $2\ell+1$, on the
real-signal slice; even self-couplings and degree-4 CG-power scalars provide
the additional independent constraints.

This status differs from the finite-group and disk modules, whose generic
completeness follows from the cited results under their stated assumptions. We
will add precisely this proven-versus-empirical distinction to Limitations, as
requested.

## Requested spherical evidence

**Feature-set ablation.** We are running the requested controlled ablation with
the same input, MLP, optimizer, and three seeds:

| Features | NR/NR | NR/R |
|---|---:|---:|
| bootstrap bispectral triples | [[E3: base results]] | [[E3]] |
| + even self-couplings | [[E3: self-coupling results]] | [[E3]] |
| + CG-power scalars | [[E3: CG-power results]] | [[E3]] |
| full model (+ 5 auxiliary bispectral entries) | [[E3: full results]] | [[E3]] |

[[E3: one-sentence interpretation of which component changes accuracy.]]
This directly tests whether the Jacobian-motivated augmentation also matters
for classification rather than only for reconstruction.

**Band-limit and out-of-dataset reconstruction.** We agree that validating only
at $L=12$ while classifying at $L=15$ was a gap. We repeated the reconstruction
at $L=15$ using the classifier's feature construction and also tested random
real band-limited signals rather than MNIST alone:
[[E2: sample count, restarts, feature residual, aligned signal residual, success
rate, grid, and SHT floor]]. We will add these results and replace the current
limitation about the $L=12/L=15$ mismatch with the measured $L=15$ scope. These
experiments remain evidence, not a proof, and do not rule out exceptional
signals or near-collisions.

## Portability and runtime

We agree that one H100 is insufficient for a library claim. We benchmarked the
same public implementation on [[E1: devices]], recording PyTorch, CUDA/backend,
batch size, dtype, warm-up, and median synchronized wall-clock:

| Device/backend | Representative modules | Selective forward time |
|---|---|---:|
| CPU [[E1: CPU model, PyTorch]] | [[E1]] | [[E1]] |
| consumer GPU [[E1: model, CUDA/PyTorch]] | [[E1]] | [[E1]] |
| H100 [[E1: CUDA/PyTorch]] | [[E1]] | [[E1]] |

[[E1: concise conclusion, including whether “sub-millisecond” survives and any
backend exceptions.]] We will limit the abstract claim to the devices/settings
actually measured and add the full compatibility matrix to the appendix and
documentation.

## High-capacity OrganMNIST3D result

We agree that “likely overfits” was not established by test accuracy alone. At
$(16,32)$ channels we now report train/validation curves, final test results,
and a controlled regularization sweep for max and bispectral pooling:
[[E4: run count, weight decay/dropout settings, best train/validation/test
numbers, and generalization gaps]]. [[E4: conclusion—state only what the curves
support: optimization failure, overfitting, or capacity/readout limitation.]]
We will replace the speculative sentence in the paper with this evidence. This
experiment is also reported in our response to Reviewer LgCU.

## Baselines and scope

We agree that a stronger spherical invariant baseline would improve the
evaluation. [[E5: report tensor-product/e3nn or spherical-scattering result and
matched setup. If E5 is not completed, replace with: “Implementing and tuning a
new tensor-product or scattering pipeline reliably during the response window
was not feasible. We will add this comparison in the revision and retain the
missing-baseline limitation; we do not claim superiority to those methods.”]]
The revised related-work discussion will also distinguish our construction
from the complete but cubic-size invariants of Edidin--Satriano and the
three-shell frequency-marching result of Bendory et al.; the current comparison
table already records their completeness and scaling.

## Corrections to claims

We accept the reviewer's corrections:

- “selectivity reduces computational cost from $O(|G|^2)$ to $O(|G|)$” will
  become “selectivity reduces the **coefficient count** from
  $O(|G|^2)$ to $O(|G|)$”; Table 1's selective forward cost is
  $O(|G|^2)$;
- “consistently outperform” will be replaced by the narrower observed claim:
  bispectral pooling improves data efficiency in the tested low-data,
  moderate-capacity settings, while matching or trailing alternatives
  elsewhere;
- coverage will be reported consistently as 96.8%; and
- the checklist references to the Section 5 construction will be corrected.

Finally, the hidden phrases noted under “Paper Formatting Concerns” were not
inserted by the authors. They are a NeurIPS-generated watermark placed in
reviewer-facing PDFs to detect prohibited LLM-assisted reviewing; identical
text and placement occur across NeurIPS 2026 submissions. The Program Chairs
can confirm this, and our author PDF/LaTeX sources do not contain the text.
