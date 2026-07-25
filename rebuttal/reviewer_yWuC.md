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

**Feature-set ablation.** We ran the requested controlled ablation. Omitted
features are masked to zero, so all four variants share the identical
768-channel input, MLP, parameter count (231,818), optimizer, and three seeds
(mean $\pm$ population std, accuracy %):

| Features (active real channels) | NR/NR | NR/R |
|---|---:|---:|
| bootstrap bispectral triples (496) | $94.09 \pm 0.08$ | $94.09 \pm 0.18$ |
| + even self-couplings (604) | $94.48 \pm 0.13$ | $94.49 \pm 0.15$ |
| + CG-power scalars (758) | $95.00 \pm 0.06$ | $95.00 \pm 0.03$ |
| full model (+ 5 auxiliary bispectral entries, 768) | $94.78 \pm 0.06$ | $94.76 \pm 0.07$ |

Each augmentation stage changes accuracy by more than cross-seed variability:
the Jacobian-motivated even self-couplings add $+0.4$ points and the CG-power
scalars a further $+0.5$; the five auxiliary entries of the production model
are not needed for accuracy ($-0.2$). NR/R equals NR/NR at every stage, so
rotation invariance holds regardless of feature set. This directly tests
whether the Jacobian-motivated augmentation also matters for classification
rather than only for reconstruction—it does.

**Band-limit and out-of-dataset reconstruction.** We agree that validating only
at $L=12$ while classifying at $L=15$ was a gap. We repeated the reconstruction
at $L=15$ (128x256 grid) with the classifier's feature construction, 4
optimization restarts per signal and 12 alignment restarts, on 16 Spherical
MNIST signals (8 digits, identity + one random rotation each) and 16 random
Gaussian band-limited signals. With pre-registered thresholds (feature residual
$\le 10^{-2}$, aligned image residual $\le 0.3$): on MNIST signals, all 16
recoveries converged in feature space (median residual $1.4\times10^{-3}$, IQR
$[1.2, 3.5]\times10^{-3}$), the invariance residual under rotation was
$1.9\times10^{-3}$ (median), and 62.5% additionally passed the aligned-image
threshold (median aligned residual 0.27, IQR $[0.21, 0.52]$; successful cases
sit at the SHT discretization floor). On random signals the optimizer plateaued
just above the feature threshold (median $1.4\times10^{-2}$, IQR
$[1.1, 1.7]\times10^{-2}$; 3/16 below threshold) and no case passed alignment
(median aligned residual 1.19), so the joint success rate is 0% and we report
it as such rather than adjusting thresholds post hoc; we cannot currently
distinguish optimization difficulty from a genuine failure mode on this family.
We will add both results and replace the current $L=12/L=15$ limitation with
the measured $L=15$ scope. These experiments remain evidence, not a proof, and
do not rule out exceptional signals or near-collisions.

## Portability and runtime

We agree that one H100 is insufficient for a library claim. We benchmarked the
same public implementation (paper-scale selective configurations, batch 16,
float32, 10 warm-up iterations, median synchronized wall-clock over 600 timed
forwards, including the SHT inside SO(3)-on-$S^2$) on an NVIDIA A100 80GB
(PyTorch 2.11.0, CUDA 13.0) and its x86-64 host CPU (same PyTorch build):

| Device/backend | Fastest module | SO(3)-on-$S^2$ ($L=16$) | Slowest module |
|---|---:|---:|---:|
| A100, CUDA 13.0 | 0.098 ms (torus) | 0.232 ms | 0.812 ms (octahedral) |
| x86-64 CPU | 0.082 ms (cyclic) | 402 ms | 402 ms (SO(3)-on-$S^2$) |

All seven modules execute correctly on both backends with no unsupported
operations. "Sub-millisecond" survives on the GPU for every module at batch
16; on CPU only the small finite-group modules are sub-millisecond, while
grid-based modules run at 58--402 ms per batch. We will therefore scope the
abstract's claim to GPU execution, limit it to the devices/settings actually
measured, and add the full per-module compatibility matrix (including
consumer-GPU and Apple-silicon numbers) to the appendix and documentation.

## High-capacity OrganMNIST3D result

We agree that “likely overfits” was not established by test accuracy alone. At
$(16,32)$ channels we now report train/validation curves, final test results,
and a controlled regularization sweep: 7 configurations $\times$ 3 seeds = 21
runs (max and bispectral baselines at weight decay $10^{-4}$; bispectral with
weight decay $10^{-3}$/$10^{-2}$, dropout 0.2/0.5, and $10^{-3}$+0.2),
validation-AUC model selection, test evaluated once after training. At the
best-validation epoch, max pooling reaches $84.9\pm9.7$% train /
$76.7\pm5.8$% test; the bispectral baseline $73.7\pm6.3$% train /
$66.8\pm4.9$% test; the best regularized bispectral model (weight decay
$10^{-3}$) $80.5\pm3.7$% train / $72.5\pm1.0$% test. Train-test gaps are
small and uniform (0.067--0.087) for every configuration. The curves do not
show the overfitting signature: bispectral train accuracy is *lower* than max
pooling's, train tracks validation throughout, and stronger regularization
raises train and test together without closing the gap to max pooling. The
supported diagnosis is an optimization/capacity limitation of the bispectral
head at this width, not memorization; we will replace the speculative sentence
in the paper with this evidence. This experiment is also reported in our
response to Reviewer LgCU.

## Baselines and scope

We agree that a stronger spherical invariant baseline would improve the
evaluation. Implementing and tuning a new tensor-product or scattering
pipeline reliably during the response window was not feasible. We will add
this comparison in the revision and retain the missing-baseline limitation;
we do not claim superiority to those methods.
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
