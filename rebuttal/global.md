# Global response

We thank the reviewers and Area Chair for the careful assessment. The reviews
agree that the library is well engineered, clearly presented, and fills a
practical gap: it provides the first tested, differentiable, uniform interface
to selective bispectral invariants across seven group/domain pairs. The main
points of disagreement concern the status of the new spherical construction
and the breadth of its empirical validation. We will revise the paper to make
the distinction unambiguous:

- completeness of the finite-group and disk constructions follows from prior
  theory under their stated genericity assumptions;
- generic completeness of our augmented selective
  $\mathrm{SO}(3)$-on-$S^2$ invariant is a **conjecture**, supported—but not
  proved—by reconstruction and Jacobian-rank evidence; and
- the $\mathrm{SO}(3)$ output is an *augmented invariant*: its bispectral
  component is supplemented by degree-4 CG-power scalars to repair the rank
  deficit on real signals.

During the response period we ran the targeted experiments requested by the
reviewers:

1. reconstruction at the classifier's deployed band-limit $L=15$ (128x256
   grid): all 16 Spherical MNIST signals recovered in feature space (median
   residual $1.4\times10^{-3}$, 100% under the pre-registered $10^{-2}$
   threshold) with 62.5% joint success after SO(3) alignment; on 16 random
   band-limited signals the optimizer plateaued just above the feature
   threshold (median $1.4\times10^{-2}$) and alignment did not succeed, which
   we report as-is—empirical evidence on natural signals, inconclusive on
   random ones (details in the response to Reviewer yWuC);
2. a parameter-matched ablation of bootstrap triples, even self-couplings,
   and CG-power augmentation on Spherical MNIST: 94.09% $\to$ 94.48% $\to$
   95.00% (each step exceeding cross-seed variability), with the CG-power
   scalars contributing the largest gain and rotated-test accuracy identical
   to unrotated within noise at every stage;
3. timings on an NVIDIA A100 (PyTorch 2.11, CUDA 13.0) and its x86-64 host
   CPU at batch 16, float32: all seven modules run in 0.10--0.81 ms per batch
   on the GPU; on CPU, times range from 0.08 ms (cyclic) to 402 ms
   (SO(3)-on-$S^2$), so we will scope the "sub-millisecond" claim to GPU
   execution; and
4. training/validation curves, final test results, and a regularization sweep
   for the high-capacity OrganMNIST3D model (21 runs): no overfitting
   signature—train accuracy tracks validation for both poolings (train-test
   gap $\approx$0.07--0.08)—and the best regularization improves bispectral
   test accuracy from 66.8% to 72.5% without closing the gap to max pooling
   (76.7%), supporting an optimization/capacity diagnosis rather than
   memorization.

We will also correct the finite-group complexity statement (selectivity reduces
the **coefficient count** from $O(|G|^2)$ to $O(|G|)$; the selective forward
cost in Table 1 is $O(|G|^2)$), use 96.8% consistently for coverage, soften the
abstract's empirical claim, fix the checklist cross-references and reported
typos, and move the published S2CNN results into the Spherical MNIST table.

## Clarification about the hidden PDF text

The hidden phrases reported by Reviewer yWuC were not inserted by the authors.
They are a NeurIPS-generated watermark added to reviewer-facing PDFs as part of
the venue's detection of prohibited LLM-assisted reviewing. The same phrases
and placements occur across NeurIPS 2026 submissions; the Program Chairs can
confirm this. Our author-generated PDF and LaTeX sources do not contain this
text. We respectfully ask that this venue-added watermark not be treated as an
author formatting or integrity concern.
