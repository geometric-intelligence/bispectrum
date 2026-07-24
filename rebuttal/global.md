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

During the response period we are also running the targeted experiments
requested by the reviewers:

1. reconstruction at the classifier's deployed band-limit $L=15$ and on
   random band-limited signals: [[E2: reconstruction summary]];
2. an ablation of bootstrap triples, even self-couplings, and CG-power
   augmentation: [[E3: classification-ablation summary]];
3. CPU/consumer-GPU/H100 timings and backend versions:
   [[E1: portability summary]]; and
4. training/validation curves, final test results, and regularization controls
   for the 6.3M-parameter OrganMNIST3D model: [[E4: overfitting summary]].

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
