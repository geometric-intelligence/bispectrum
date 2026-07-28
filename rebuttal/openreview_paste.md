<!--
OpenReview paste file - NeurIPS 2026 Submission 10902
Four blocks below: one global Official Comment plus one response per reviewer.
Copy ONLY the text between the BEGIN/END markers into the corresponding
OpenReview reply box.
Format: Markdown + MathJax ($...$), per OpenReview FAQ. Limit: 10,000 chars each.
Character counts are filled in at the top of each block.
-->

<!-- ================================================================
GLOBAL Official Comment - post on the main submission thread
(visible to all reviewers and the AC).
Title: "Global response: new experiments, completeness status, and planned
revisions"
Character count: 4,281 / 5,000 (Official Comment box is capped at 5,000,
unlike the 10,000-char rebuttal boxes - little headroom, trim before adding)
================================================================ -->

<<<BEGIN GLOBAL>>>
We thank the reviewers and the Area Chair for the careful assessment. The reviews agree that the library is well engineered, clearly presented, and fills a practical gap: it provides the first tested, differentiable, uniform interface to selective bispectral invariants across seven group/domain pairs. The main points of disagreement concern the status of the new spherical construction and the breadth of its empirical validation. We will revise the paper to make the distinction unambiguous:

- Completeness of the finite-group and disk constructions follows from prior theory under their stated genericity assumptions.
- Generic completeness of our augmented selective $\mathrm{SO}(3)$-on-$S^2$ invariant is a **conjecture**, supported - but not proved - by reconstruction and Jacobian-rank evidence.
- The $\mathrm{SO}(3)$ output is an *augmented invariant*: its bispectral component is supplemented by degree-4 CG-power scalars to repair the rank deficit on real signals.

During the response period we ran the experiments the reviewers requested. Each individual response carries the full details, and the headline results are:

1. **Reconstruction at the deployed band-limit** ($L=15$, $128\times256$ grid): all 16 Spherical MNIST signals recovered in feature space (median residual $1.4\times10^{-3}$, 100% under the pre-registered $10^{-2}$ threshold) with 62.5% joint success after $\mathrm{SO}(3)$ alignment. On 16 random band-limited signals the optimizer plateaued just above the feature threshold (median $1.4\times10^{-2}$) and alignment did not succeed, which we report as-is - empirical evidence on natural signals, inconclusive on random ones (details in our response to Reviewer yWuC).
2. **Parameter-matched feature ablation** on Spherical MNIST: bootstrap triples 94.09% $\to$ + even self-couplings 94.48% $\to$ + CG-power scalars 95.00%, with each step exceeding cross-seed variability and rotated-test accuracy identical to unrotated within noise at every stage (Reviewer yWuC).
3. **Timings** at batch 16, float32 on an NVIDIA A100 (PyTorch 2.11, CUDA 13.0), its x86-64 host CPU, and an Apple M5 Pro (CPU and MPS): the six paper-scale configurations run in 0.098-0.812 ms on the A100, 0.082-402 ms on the x86 CPU, 0.016-1.811 ms on the M5 Pro CPU, and 0.062-0.719 ms on MPS. Every configuration executes on every backend, so "sub-millisecond" holds for both tested GPU backends and we will scope the abstract's claim to those settings (Reviewer yWuC).
4. **Train/validation curves and a regularization sweep** for the high-capacity OrganMNIST3D model (21 runs): no overfitting signature - train accuracy tracks validation for both poolings (train-test gap $\approx$ 0.07-0.08) - and the best regularization improves bispectral test accuracy from 66.8% to 72.5% without closing the gap to max pooling (76.7%), supporting an optimization/capacity diagnosis rather than memorization (Reviewers LgCU and yWuC).

The revision will also correct the finite-group complexity statement (selectivity reduces the **coefficient count** from $O(|G|^2)$ to $O(|G|)$, while the selective forward cost in Table 1 stays $O(|G|^2)$), use 96.8% consistently for coverage, soften the abstract's empirical claim, fix the checklist cross-references and reported typos, and move the published S2CNN results into the Spherical MNIST table. For the camera-ready we further commit to a fairly tuned tensor-product/scattering spherical baseline and a matched S2CNN re-run on the same hardware and measurement protocol as our timing benchmarks.

**Clarification about the hidden PDF text.** The hidden phrases reported by Reviewer yWuC were not inserted by the authors. They are a conference-side watermark added to reviewer-facing PDFs to detect prohibited LLM-generated reviews: the NeurIPS organizing committee acknowledged the intervention in a statement to [The Transmitter](https://www.thetransmitter.org/publishing/scientists-decry-conferences-use-of-hidden-prompts-to-snare-ai-peer-reviews/), and reviewers have publicly reported the identical text across their NeurIPS 2026 review stacks. Our author-generated PDF and LaTeX sources do not contain this text, and we respectfully ask that this venue-added watermark not be treated as an author formatting or integrity concern.
<<<END GLOBAL>>>

<!-- ================================================================
RESPONSE 1 of 3 - Reviewer LgCU (Rating 5: Accept, Confidence 2)
Character count: 3,218 / 10,000
================================================================ -->

<<<BEGIN LgCU>>>
Thank you for the positive assessment of the paper's quality, clarity, originality, and potential value to the PyTorch ecosystem. We agree that the 6.3M-parameter OrganMNIST3D result needs evidence beyond the submitted attribution to "likely overfitting."

### OrganMNIST3D high-capacity ablation

We reran the $(16,32)$-channel max-pooling and bispectral models with identical data splits and three seeds, logging train/validation curves and final test results, and swept regularization while holding the backbone and optimizer schedule fixed (7 configurations $\times$ 3 seeds, validation-AUC model selection, accuracy % at the best-validation epoch, mean $\pm$ std):

| Pooling | Weight decay / dropout | Train ACC | Val ACC | Test ACC |
| --- | --- | ---: | ---: | ---: |
| max | baseline: $10^{-4}$ / 0 | $84.9 \pm 9.7$ | $84.9 \pm 7.6$ | $76.7 \pm 5.8$ |
| bispectrum | baseline: $10^{-4}$ / 0 | $73.7 \pm 6.3$ | $77.6 \pm 3.5$ | $66.8 \pm 4.9$ |
| bispectrum | best: $10^{-3}$ / 0 | $80.5 \pm 3.7$ | $81.0 \pm 2.0$ | $72.5 \pm 1.0$ |

The submitted single-seed Table 11 values (68.5% and 78.5%) fall within these cross-seed ranges. A train/validation gap never opens: train accuracy stays at or below validation accuracy for both poolings across all 100 epochs, and the train-test gap is small and uniform (0.067-0.087) in every one of the seven configurations (the remaining four - weight decay $10^{-2}$, dropout 0.2/0.5, and combined - reach 70.2-71.0% test). Stronger regularization raises the bispectral model's train and test accuracy together (+5.7 test points at weight decay $10^{-3}$) but does not close the gap to max pooling. This rules out overfitting as the explanation: the evidence supports an optimization/capacity limitation of the bispectral model at this width.

We will add the curves and sweep to the appendix and replace "likely overfits" with the optimization/capacity conclusion supported by these measurements. The revision will also state the scope plainly: at $(8,16)$ channels the methods are statistically equivalent ($74.5 \pm 0.6$% bispectrum versus $74.3 \pm 1.5$% max), and the high-capacity result identifies a regime where max pooling is stronger. We do not claim a universal advantage for complete pooling.

### Terminology

We also accept the clarity suggestion. Section 2 will add short operational definitions before the formulas:

- A **Wigner-$D$ matrix** is the matrix representing a 3D rotation on the degree-$\ell$ spherical-harmonic coefficients.
- **Clebsch-Gordan coefficients** give the change of basis decomposing the tensor product of two irreducible representations. Contracting with them is what makes each bispectral scalar rotation invariant.
- The **parity rule** states that, for real spherical signals, a scalar bispectrum entry is real at even total degree and purely imaginary at odd total degree. Odd-parity entries with a repeated degree vanish identically.

Finally, the revision will sharpen the status statement throughout: generic completeness of the finite-group construction is theoretically supported, while generic completeness of our augmented $\mathrm{SO}(3)$-on-$S^2$ invariant is conjectured and supported empirically rather than proved.
<<<END LgCU>>>

<!-- ================================================================
RESPONSE 2 of 3 - Reviewer yWuC (Rating 4: Borderline Accept, Confidence 3)
Character count: 9,099 / 10,000
================================================================ -->

<<<BEGIN yWuC>>>
Thank you for the careful, technically precise review and for recognizing the value of the uniform tested interface, the optimal-order coefficient count, and the low-data results. You are right that the original text did not separate the theoretical status of the finite-group construction from that of the spherical construction sharply enough.

### Status of the spherical invariant

**Generic completeness is conjectured, not proved.** Our evidence is empirical: generic Jacobian-rank tests and signal reconstruction. We will state this explicitly in the abstract, Section 5, Table 1, and Limitations. We will also use "augmented selective invariant" for the full output $\Phi_{\mathrm{sel}}$ and reserve "selective bispectrum" for its degree-3 bispectral subset. The distinction matters because the bootstrap bispectral block has generic rank approximately $\ell+2$, rather than $2\ell+1$, on the real-signal slice. Even self-couplings and degree-4 CG-power scalars provide the additional independent constraints.

This status differs from the finite-group and disk modules, whose generic completeness follows from the cited results under their stated assumptions. We will add precisely this proven-versus-empirical distinction to Limitations, as requested.

### Requested spherical evidence

**Feature-set ablation.** We ran the requested controlled ablation. Omitted features are masked to zero, so all four variants share the identical 768-channel input, MLP, parameter count (231,818), optimizer, and three seeds (mean $\pm$ population std, accuracy %):

| Features (active real channels) | NR/NR | NR/R |
| --- | ---: | ---: |
| bootstrap bispectral triples (496) | $94.09 \pm 0.08$ | $94.09 \pm 0.18$ |
| + even self-couplings (604) | $94.48 \pm 0.13$ | $94.49 \pm 0.15$ |
| + CG-power scalars (758) | $95.00 \pm 0.06$ | $95.00 \pm 0.03$ |
| full model (+ 5 auxiliary bispectral entries, 768) | $94.78 \pm 0.06$ | $94.76 \pm 0.07$ |

Each augmentation stage changes accuracy by more than cross-seed variability: the Jacobian-motivated even self-couplings add +0.4 points and the CG-power scalars a further +0.5. The five auxiliary entries of the production model are not needed for accuracy (-0.2). NR/R equals NR/NR at every stage, so rotation invariance holds regardless of feature set. This directly tests whether the Jacobian-motivated augmentation also matters for classification rather than only for reconstruction - it does.

**Band-limit and out-of-dataset reconstruction.** Validating only at $L=12$ while classifying at $L=15$ was a real gap. We repeated the reconstruction at $L=15$ ($128\times256$ grid) with the classifier's feature construction, 4 optimization restarts per signal and 12 alignment restarts, on 16 Spherical MNIST signals (8 digits, identity + one random rotation each) and 16 random Gaussian band-limited signals. With pre-registered thresholds (feature residual $\le 10^{-2}$, aligned image residual $\le 0.3$): on MNIST signals, all 16 recoveries converged in feature space (median residual $1.4\times10^{-3}$, IQR $[1.2, 3.5]\times10^{-3}$), the invariance residual under rotation was $1.9\times10^{-3}$ (median), and 62.5% additionally passed the aligned-image threshold (median aligned residual 0.27, IQR $[0.21, 0.52]$, with successful cases sitting at the SHT discretization floor). On random signals the optimizer plateaued just above the feature threshold (median $1.4\times10^{-2}$, IQR $[1.1, 1.7]\times10^{-2}$, only 3/16 below threshold) and no case passed alignment (median aligned residual 1.19), so the joint success rate is 0% and we report it as such rather than adjusting thresholds post hoc. We cannot currently distinguish optimization difficulty from a genuine failure mode on this family. We will add both results and replace the current $L=12$/$L=15$ limitation with the measured $L=15$ scope. These experiments remain evidence, not a proof, and do not rule out exceptional signals or near-collisions.

### Portability and runtime

We agree that one H100 is insufficient for a library claim. We benchmarked the same public implementation (paper-scale selective configurations, batch 16, float32, 10 warm-up iterations, median synchronized wall-clock over 600 timed forwards, including the SHT inside $\mathrm{SO}(3)$-on-$S^2$) on an NVIDIA A100 80GB (PyTorch 2.11.0, CUDA 13.0), its x86-64 host CPU, and an Apple M5 Pro (PyTorch 2.11.0, MPS and CPU):

| Device/backend | Fastest module | $\mathrm{SO}(3)$-on-$S^2$ ($L=16$) | Slowest module |
| --- | ---: | ---: | ---: |
| A100, CUDA 13.0 | 0.098 ms (torus) | 0.232 ms | 0.812 ms (octahedral) |
| x86-64 CPU | 0.082 ms (cyclic) | 402 ms | 402 ms (spherical) |
| M5 Pro, MPS | 0.062 ms (torus) | 0.641 ms | 0.719 ms (disk) |
| M5 Pro, CPU | 0.016 ms (cyclic) | 1.811 ms | 1.811 ms (spherical) |

All six benchmarked configurations execute correctly on CUDA, MPS, and both CPUs with no unsupported operations. The seventh module, $\mathrm{SO}(2)$-on-$S^1$, is a thin wrapper that inherits the cyclic module's computation exactly, so the cyclic row covers it. "Sub-millisecond" holds for every configuration on both tested GPU backends at batch 16. On the M5 Pro CPU, five modules are sub-millisecond and the spherical module takes 1.811 ms. The x86 host is substantially slower on grid-based modules. We will scope the abstract's claim to the tested GPU settings and add the full per-module timing and compatibility matrix to the appendix and documentation.

### High-capacity OrganMNIST3D result

"Likely overfits" was indeed not established by test accuracy alone. At $(16,32)$ channels we now report train/validation curves, final test results, and a controlled regularization sweep: 7 configurations $\times$ 3 seeds = 21 runs (max and bispectral baselines at weight decay $10^{-4}$, plus bispectral with weight decay $10^{-3}$/$10^{-2}$, dropout 0.2/0.5, and $10^{-3}$+0.2), validation-AUC model selection, test evaluated once after training. At the best-validation epoch, max pooling reaches $84.9 \pm 9.7$% train / $76.7 \pm 5.8$% test, the bispectral baseline $73.7 \pm 6.3$% train / $66.8 \pm 4.9$% test, and the best regularized bispectral model (weight decay $10^{-3}$) $80.5 \pm 3.7$% train / $72.5 \pm 1.0$% test. The submitted single-seed Table 11 values (68.5% and 78.5%) fall within these cross-seed ranges. Train-test gaps are small and uniform (0.067-0.087) for every configuration. The curves do not show the overfitting signature: bispectral train accuracy is *lower* than max pooling's, train tracks validation throughout, and stronger regularization raises train and test together without closing the gap to max pooling. The supported diagnosis is an optimization/capacity limitation of the bispectral model at this width rather than memorization. We will replace the speculative sentence in the paper with this evidence. This experiment is also reported in our response to Reviewer LgCU.

### Baselines and scope

A stronger spherical invariant baseline would clearly improve the evaluation. Within the response window we prioritized the ablation and reconstruction experiments requested above, and could not also tune a new tensor-product or scattering pipeline to a fair standard. We commit to adding this tuned comparison to the camera-ready and retain the missing-baseline limitation until then. We do not claim superiority to those methods. The revised related-work discussion will also distinguish our construction from the complete but cubic-size invariants of Edidin-Satriano and the three-shell frequency-marching result of Bendory et al. The current comparison table already records their completeness and scaling.

### Corrections to claims

We accept the reviewer's corrections:

- "selectivity reduces computational cost from $O(|G|^2)$ to $O(|G|)$" will become "selectivity reduces the **coefficient count** from $O(|G|^2)$ to $O(|G|)$" (Table 1's selective forward cost is $O(|G|^2)$).
- "consistently outperform" will be replaced by the narrower observed claim: bispectral pooling improves data efficiency in the tested low-data, moderate-capacity settings, while matching or trailing alternatives elsewhere.
- Coverage will be reported consistently as 96.8%.
- The checklist references to the Section 5 construction will be corrected.

Finally, the hidden phrases noted under "Paper Formatting Concerns" were not inserted by the authors. They are a conference-side watermark added to reviewer-facing PDFs to detect prohibited LLM-generated reviews: the NeurIPS organizing committee acknowledged the intervention in a statement to [The Transmitter](https://www.thetransmitter.org/publishing/scientists-decry-conferences-use-of-hidden-prompts-to-snare-ai-peer-reviews/), reviewers have publicly reported the identical text across their entire NeurIPS 2026 review stacks (including in reviewer-facing copies of their own submissions), and the committee has been informing reviewers who notice it not to penalize individual papers. Our author-generated PDF and LaTeX sources do not contain this text, and the Program Chairs can confirm the watermark.
<<<END yWuC>>>

<!-- ================================================================
RESPONSE 3 of 3 - Reviewer AZ1B (Rating 2: Reject, Confidence 3)
Character count: 6,669 / 10,000
================================================================ -->

<<<BEGIN AZ1B>>>
Thank you for recognizing that the package fills an unavailable capability, could broaden adoption of bispectra, and is well written and thoroughly evaluated. We agree that most group-specific formulas build on prior theory. The contribution is to make them usable together as tested, differentiable PyTorch components. We respectfully disagree, however, that the paper lacks a substantive algorithmic or empirical contribution.

The work contributes more than a repackaging of formulas: (i) one uniform autograd-compatible implementation for seven group/domain pairs, including precomputed CG/DFT/Bessel buffers and selective sparse contractions, (ii) profiled kernel-level optimizations that shrink the full representations by up to $512\times$, (iii) a new augmented selective $\mathrm{SO}(3)$-on-$S^2$ construction with an asymptotically optimal $\Theta(L^2)$ coefficient count, an $\Omega(L^2)$ lower bound, a parity/vanishing analysis, and an explicit diagnosis and repair of the real-signal rank deficit, and (iv) the first controlled evaluation of these invariants as pooling layers across planar, volumetric, and spherical tasks.

The empirical benefit is also not confined to very-low-data subsets. At full Spherical MNIST data, the augmented spherical invariant reaches 95.0% versus 79.2% for a matched-capacity power-spectrum model, isolating information loss rather than classifier size. At 10% PCam, it is within 0.4 AUC points of the best method, and at full-data OrganMNIST3D it matches max pooling at moderate capacity. The low-data gains are where completeness helps most, but not the only demonstrated functionality: the library also provides near-exact invariance without augmentation and preserves orbit information that norm/max pooling discards.

During the response period we added two measurements that bear directly on the contribution's substance. A parameter-matched feature ablation on Spherical MNIST shows each component of the augmented construction adds accuracy beyond cross-seed variability (three seeds: bootstrap triples 94.09%, + even self-couplings 94.48%, + CG-power scalars 95.00%), so each component of the augmentation is functionally necessary. We also benchmarked six paper-scale module configurations on an NVIDIA A100, an x86-64 CPU, and an Apple M5 Pro (CPU and MPS): every configuration runs on every backend, sub-millisecond on both GPUs. Details are in our response to Reviewer yWuC.

### Q1: What made the spherical extension technically difficult?

The finite-group selection argument cannot be transferred by replacing a finite irrep index with $\ell$:

1. **Homogeneous-space rank deficiency.** For signals on a finite group, each generic Fourier coefficient is a nonsingular matrix, which supports the recursive argument of Mataigne et al. On $S^2$, the coefficient at degree $\ell$ is only a vector $\mathbf{F}_\ell \in \mathbb{C}^{2\ell+1}$. Kakarala's full-rank operator-bispectrum theorem therefore does not directly apply.
2. **A linear chain is dimensionally insufficient.** A band-limited real spherical signal has $(L+1)^2$ degrees of freedom modulo three rotational degrees. We prove that every smooth/polynomial complete invariant therefore needs at least $(L+1)^2-3=\Omega(L^2)$ live components, so an $O(L)$ Mataigne-style frequency chain cannot be complete.
3. **Reality and parity break the naive bootstrap.** On real signals, each scalar bispectrum entry is either real or purely imaginary, and every odd-parity entry with a repeated index vanishes identically. After removing these zero rows, the nominal $(2\ell+1)$-row bootstrap block has generic rank only approximately $\ell+2$, not $2\ell+1$.
4. **Rank must be repaired without losing selectivity.** Our construction adds mandatory even self-couplings and greedily selected degree-4 CG-power invariants until the per-degree real Jacobian reaches full rank. This keeps $O(\ell)$ entries per degree and hence $\Theta(L^2)$ overall.

These are the reasons the result is an *augmented selective invariant*, rather than a direct application of the finite-group algorithm. We will foreground this technical chain in Section 5.

### Q2: What would a completeness proof require?

We conjecture generic completeness. The current Jacobian and reconstruction tests provide evidence but fall short of a proof. A proof would proceed in four steps:

1. Prove that the full low-degree seed separates a generic orbit and fixes a common rotational gauge (including orientation, using a nonzero odd-parity triple).
2. Induct on $\ell$, assuming all lower-degree coefficients have been recovered in that common gauge.
3. Prove that the selected bootstrap, self-coupling, and CG-power equations have a **unique generic real solution** for $\mathbf{F}_\ell$.
4. Characterize the algebraic exceptional set where a seed coefficient vanishes, a rank drops, or a nontrivial stabilizer remains.

The unresolved conceptual step is (3). Full generic Jacobian rank gives local identifiability but does not exclude a second, globally separated solution. The finite-group matrix-inverse proof does not resolve this because the spherical coefficients are rank-deficient vectors, and the real-signal parity identities create extra algebraic dependencies. A proof likely needs either an explicit rational frequency-marching inverse or an invariant-field argument showing that the selected polynomials generate the generic orbit-separating field. We will add this proof roadmap and state "conjectured generic completeness" explicitly.

### Q3: S2CNN baseline and inference

The published S2CNN values should indeed appear in Table 4 rather than only in prose, and we will move them there. We did not run S2CNN in the submitted experiments, so we cannot infer an apples-to-apples speed advantage from its published accuracies, and porting the original implementation to a current PyTorch/CUDA stack was not achievable within the response window. We commit to a matched S2CNN re-run for the camera-ready, on the same hardware, precision, and measurement protocol as our timing benchmarks, and will report the resulting accuracy and measured inference times. Until then we will label the accuracy comparison as published and remove any implication of a measured speed advantage over S2CNN.

### Corrections

Thank you for catching the three presentation issues. We will (i) correct the coefficient notation/cross-reference following Eq. (3), (ii) add the formal definition that an invariant $\Phi$ is complete on a set $U$ when $\Phi(f)=\Phi(h)$ iff $h=g\cdot f$ for some $g\in G$ and $f,h\in U$, and (iii) change `TorusDnTorus` to the library's actual class name, `TorusOnTorus`.
<<<END AZ1B>>>
