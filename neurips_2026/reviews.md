OpenReview
.net
Search articles, authors and reviews...
Notifications53
Activity
Tasks
Johan Mathe 
back arrowGo to NeurIPS 2026 Conference homepage
: Selective 
-Bispectra Made Practical
Download PDF
Johan Mathe, Adele Myers, Simon Mataigne, Nina Miolane 
30 Apr 2026 (modified: 27 May 2026)
NeurIPS 2026 Conference Submission
Conference, Senior Area Chairs, Area Chairs, Reviewers, Authors
Revisions
CC BY 4.0
TL;DR: We introduce \texttt{bispectrum}, a PyTorch library for efficient, differentiable, and complete group-invariant representations across diverse geometric domains.
Abstract:
Many machine learning tasks are invariant under the action of a group 
 of transformations: signal classification can be invariant under translations, image classification under 2D rotations, and spherical-image classification under 3D rotations. The 
-bispectrum is a principled complete invariant of a signal )retaining all signal's information up to the group action) with proven benefits in machine learning and as a pooling layer in deep networks. However, its deployment has been hampered by high computational cost and a patchwork of group-specific implementations. We present bispectrum, an open-source, fully unit-tested PyTorch library that implements selective 
-bispectra for seven different group actions, as differentiable modules that can be directly incorporated into machine learning pipelines and deep learning architectures. For finite groups 
, selectivity reduces the computational cost from 
 to 
. For planar rotations, we leverage the disk bispectrum. For spherical 3D rotations, we introduce an augmented selective bispectrum at band-limit 
 which reduces the cost from 
 to 
 coefficients. We profile the entire library (for

Checklist Confirmation: I confirm that I have included a paper checklist in the paper PDF.
Supplementary Material:  zip
Responsible Reviewing: We acknowledge the responsible reviewing obligations as authors.
Primary Area: Deep learning advancements (e.g., architectures, optimizers, representation learning)
Secondary Area: Computer vision (e.g., object detection, segmentation, image classification, image generation, video)
Contribution Type: General: Most submissions will fall into this type.
Academic Integrity: I acknowledge that I have read the NeurIPS Handbook and commit to adhering to all policies in the Handbook (https://neurips.cc/Conferences/2026/MainTrackHandbook), the NeurIPS Code of Conduct and the NeurIPS Academic Integrity Policy.
LLM Usage: Editing (e.g., grammar, spelling, word choice), Facilitating or running experiments
Declaration: I confirm that the above information is accurate.
Reviewer Nomination:  Nina Miolane
Submission Number: 10902
Discussion
Filter by reply type...
Filter by author...
Search keywords...

Sort: Newest First
15 / 15 replies shown
Add:
Citation Checker by Program Chairs
Citation Checkerby Program Chairs04 Aug 2026, 11:34 (modified: 04 Sept 2026, 22:13)Program Chairs, Senior Area Chairs, Area Chairs, AuthorsRevisions
Comment:
Automatically flagged references
Here we list the references that were automatically flagged by our checker. Note that our checker may miss many hallucinated references, and some of the listed references may have been flagged erroneously. It is therefore crucial that you check them manually, following the guidance above. We rely on the Area Chairs to decide whether these constitute violations of responsible authorship.

For papers falling into either of the following categories, we expect your default recommendation to be a rejection on the grounds of irresponsible authorship. This is strong presumption rather than a strict rule; as the Area Chair, you retain discretion, and may recommend otherwise where the specific circumstances of a paper warrant it:

The paper contains at least two hallucinated references according to the criteria described below the report.
The paper contains one hallucinated reference and there is some additional evidence of AI slop.
If a paper has only one problematic reference, but no additional concerns about the quality of the paper or its presentation, please simply alert the authors in your meta-review.

Please complete the form “Hallucinated References Investigation” to communicate the results of your investigation prior to completing your meta reviews.

Citation Checker Report
Reference Check Summary
0/35 problematic references.

Hallucinated/problematic reference criteria
These are adopted from ICLR and ICML 2026 hallucinated reference guidance:

If you are unable to find any document or publication with a title anywhere near the one provided in the reference, it should be considered hallucinated. Minor differences in title should not be considered evidence of a hallucinated reference.
If the reference's authors are only slightly incorrect, e.g., a few subtle misspellings, missing a few authors, or wrong authorship order, we do not consider this to imply a hallucinated reference, but please let the submission's authors know in your meta-review that they should fix this issue.
If the author list has additional issues, e.g., added authors that are not actually authors of that publication or missing a large fraction of authors, then the reference should be considered hallucinated.
If the reference's venue is a real venue but is incorrect for that reference, we do not consider this to imply a hallucinated reference, but please tell the submission's authors in your meta-review, these include wrong journal name, arXiv ID, etc.
If there is no evidence that the reference's venue exists, the reference should be considered hallucinated.
Finally, we note that there are many generic references that refer to existing artifacts (e.g. "GPT-4o, OpenAI et al. 2024") but do not refer to any real document. These are fine.
Even with the above guidelines in mind, there will likely be some borderline cases. Please bring those up with your SAC if you need additional guidance.
Participate in the discussion
Official Commentby Area Chair 1SUy01 Aug 2026, 00:47Program Chairs, Reviewer AZ1B, Reviewer LgCU, Reviewer yWuC, Senior Area Chairs, Area Chairs, Reviewers, Reviewers Submitted, Authors
Comment:
Dear Reviewers,

Thank you for your time and effort in the review process.

As the authors have submitted their formal response, it would be appreciated if you could read these and parcitipate in the dicussion.

Thank you for your continued engagement!

Best,

AC

Official Comment by Authors
Official Commentby Authors (Simon Mataigne, Nina Miolane, Adele Myers, Johan Mathe)31 Jul 2026, 10:50 (modified: 31 Jul 2026, 10:52)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers, AuthorsRevisions
Comment:
Dear Area Chair,

Thank you for the meta-review and for your engagement with our submission. We posted our responses and a global summary on July 28, covering the four experiments run during the response period (feature ablation, reconstruction at the deployed band-limit, multi-backend timings, and the OrganMNIST3D regularization sweep) together with the completeness-status clarification and the planned corrections.

As of today, reviewers not yet engaged with the discussion, which closes Monday (AoE). We have followed up with them directly, but we would also be grateful if you could encourage them to consider our responses, if and when you judge that appropriate. We remain available for any remaining questions before the deadline.

With thanks for handling our submission, The Authors

Global response: new experiments, completeness status, and planned revisions
Official Commentby Authors (Simon Mataigne, Nina Miolane, Adele Myers, Johan Mathe)28 Jul 2026, 10:56 (modified: 28 Jul 2026, 10:59)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers, Reviewers Submitted, AuthorsRevisions
Comment:
We thank the reviewers and the Area Chair for the careful assessment. The reviews agree that the library is well engineered, clearly presented, and fills a practical gap: it provides the first tested, differentiable, uniform interface to selective bispectral invariants across seven group/domain pairs. The main points of disagreement concern the status of the new spherical construction and the breadth of its empirical validation. We will revise the paper to make the distinction unambiguous:

Completeness of the finite-group and disk constructions follows from prior theory under their stated genericity assumptions.
Generic completeness of our augmented selective 
-on-
 invariant is a conjecture, supported - but not proved - by reconstruction and Jacobian-rank evidence.
The 
 output is an augmented invariant: its bispectral component is supplemented by degree-4 CG-power scalars to repair the rank deficit on real signals.
During the response period we ran the experiments the reviewers requested. Each individual response carries the full details, and the headline results are:

Reconstruction at the deployed band-limit (
, 
 grid): all 16 Spherical MNIST signals recovered in feature space (median residual 
, 100% under the pre-registered 
 threshold) with 62.5% joint success after 
 alignment. On 16 random band-limited signals the optimizer plateaued just above the feature threshold (median 
) and alignment did not succeed, which we report as-is - empirical evidence on natural signals, inconclusive on random ones (details in our response to Reviewer yWuC).
Parameter-matched feature ablation on Spherical MNIST: bootstrap triples 94.09% 
 + even self-couplings 94.48% 
 + CG-power scalars 95.00%, with each step exceeding cross-seed variability and rotated-test accuracy identical to unrotated within noise at every stage (Reviewer yWuC).
Timings at batch 16, float32 on an NVIDIA A100 (PyTorch 2.11, CUDA 13.0), its x86-64 host CPU, and an Apple M5 Pro (CPU and MPS): the six paper-scale configurations run in 0.098-0.812 ms on the A100, 0.082-402 ms on the x86 CPU, 0.016-1.811 ms on the M5 Pro CPU, and 0.062-0.719 ms on MPS. Every configuration executes on every backend, so "sub-millisecond" holds for both tested GPU backends and we will scope the abstract's claim to those settings (Reviewer yWuC).
Train/validation curves and a regularization sweep for the high-capacity OrganMNIST3D model (21 runs): no overfitting signature - train accuracy tracks validation for both poolings (train-test gap 
 0.07-0.08) - and the best regularization improves bispectral test accuracy from 66.8% to 72.5% without closing the gap to max pooling (76.7%), supporting an optimization/capacity diagnosis rather than memorization (Reviewers LgCU and yWuC).
The revision will also correct the finite-group complexity statement (selectivity reduces the coefficient count from 
 to 
, while the selective forward cost in Table 1 stays 
), use 96.8% consistently for coverage, soften the abstract's empirical claim, fix the checklist cross-references and reported typos, and move the published S2CNN results into the Spherical MNIST table. For the camera-ready we further commit to a fairly tuned tensor-product/scattering spherical baseline and a matched S2CNN re-run on the same hardware and measurement protocol as our timing benchmarks.

Clarification about the hidden PDF text. The hidden phrases reported by Reviewer yWuC were not inserted by the authors. They are a conference-side watermark added to reviewer-facing PDFs to detect prohibited LLM-generated reviews: the NeurIPS organizing committee acknowledged the intervention in a statement to The Transmitter, and reviewers have publicly reported the identical text across their NeurIPS 2026 review stacks. Our author-generated PDF and LaTeX sources do not contain this text, and we respectfully ask that this venue-added watermark not be treated as an author formatting or integrity concern.

Official Review of Submission10902 by Reviewer LgCU
Official Reviewby Reviewer LgCU26 Jun 2026, 17:42 (modified: 21 Aug 2026, 01:13)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer LgCURevisions
Summary:
This paper introduce a open-soruce PyTorch library called bispectrum, which implements selective G-bispectra. The new library also introudce a new augmented selective SO(3)-invariant for band-limited spherical signals that reduces the coefficient count from O(L³) to O(L²).

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Quality: I think the paper have good quality. Despite that the SO(3)-on-S² invariant is only empirical complete. Clarity: I think the paper is really clear. despite it may benefit from explain some terms in the paper, like Clebsch–Gordan, Wigner-D, parity lemmas. Significance: I have difficulty to access the significance of the library, like how many researchers will think this library is useful. But I feel it is still a good addition to the current pytorch family. Originality: I don't think the originality is too relevant for a library.

Quality: 3: good
Clarity: 3: good
Significance: 3: good
Originality: 4: excellent
Questions:
For the table11, the OrganMNIST3D results, the performance is droped to 0.685 at (16, 32) channels. The paper state that the bispectrum model likely overfits the 971 training samples, could you do more ablation for us to understand more about it?

Limitations:
Yes

Confidence: 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
N/A

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Rebuttal by Authors
Rebuttalby Authors (Simon Mataigne, Nina Miolane, Adele Myers, Johan Mathe)27 Jul 2026, 19:04 (modified: 28 Jul 2026, 06:14)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
Thank you for the positive assessment of the paper's quality, clarity, originality, and potential value to the PyTorch ecosystem. We agree that the 6.3M-parameter OrganMNIST3D result needs evidence beyond the submitted attribution to "likely overfitting."

OrganMNIST3D high-capacity ablation
We reran the 
-channel max-pooling and bispectral models with identical data splits and three seeds, logging train/validation curves and final test results, and swept regularization while holding the backbone and optimizer schedule fixed (7 configurations 
 3 seeds, validation-AUC model selection, accuracy % at the best-validation epoch, mean 
 std):

Pooling	Weight decay / dropout	Train ACC	Val ACC	Test ACC
max	baseline: 
 / 0	
bispectrum	baseline: 
 / 0	
bispectrum	best: 
 / 0	
The submitted single-seed Table 11 values (68.5% and 78.5%) fall within these cross-seed ranges. A train/validation gap never opens: train accuracy stays at or below validation accuracy for both poolings across all 100 epochs, and the train-test gap is small and uniform (0.067-0.087) in every one of the seven configurations (the remaining four - weight decay 
, dropout 0.2/0.5, and combined - reach 70.2-71.0% test). Stronger regularization raises the bispectral model's train and test accuracy together (+5.7 test points at weight decay 
) but does not close the gap to max pooling. This rules out overfitting as the explanation: the evidence supports an optimization/capacity limitation of the bispectral model at this width.

We will add the curves and sweep to the appendix and replace "likely overfits" with the optimization/capacity conclusion supported by these measurements. The revision will also state the scope plainly: at 
 channels the methods are statistically equivalent (
% bispectrum versus 
% max), and the high-capacity result identifies a regime where max pooling is stronger. We do not claim a universal advantage for complete pooling.

Terminology
We also accept the clarity suggestion. Section 2 will add short operational definitions before the formulas:

A Wigner-
 matrix is the matrix representing a 3D rotation on the degree-
 spherical-harmonic coefficients.
Clebsch-Gordan coefficients give the change of basis decomposing the tensor product of two irreducible representations. Contracting with them is what makes each bispectral scalar rotation invariant.
The parity rule states that, for real spherical signals, a scalar bispectrum entry is real at even total degree and purely imaginary at odd total degree. Odd-parity entries with a repeated degree vanish identically.
Finally, the revision will sharpen the status statement throughout: generic completeness of the finite-group construction is theoretically supported, while generic completeness of our augmented 
-on-
 invariant is conjectured and supported empirically rather than proved.

Official Comment by Authors
Official Commentby Authors (Simon Mataigne, Nina Miolane, Adele Myers, Johan Mathe)31 Jul 2026, 10:45Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:
Dear Reviewer LgCU,

With the discussion period closing Monday (August 3, AoE), we wanted to check in briefly. Our response takes up your question about the high-capacity OrganMNIST3D result: a new 21-run regularization sweep with per-epoch train/validation curves rules out overfitting and supports an optimization/capacity explanation instead. Section 2 will also gain the short operational definitions you suggested.

Your feedback has genuinely improved the paper, and we are grateful for it. If our response settles your question, a brief acknowledgment would be much appreciated. If anything remains unclear, we would be glad to address it before the deadline.

With thanks and best regards, The Authors

Official Review of Submission10902 by Reviewer yWuC
Official Reviewby Reviewer yWuC25 Jun 2026, 10:17 (modified: 10 Aug 2026, 17:25)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer yWuCRevisions
Summary:
bispectrum is a unit-tested PyTorch library implementing selective 
-bispectral invariants as differentiable pooling modules for seven group/domain pairs under a uniform API. Most modules repackage prior constructions (Sanborn et al. 2023; Mataigne et al. 2024; Myers & Miolane 2025; Kakarala 2012). The one new construction is an augmented selective 
-on-
 invariant of size 
 (vs. 
 for the full spherical bispectrum), with completeness supported empirically rather than proven. Modules are profiled on an H100 and evaluated as terminal pooling layers on PCam (
), OrganMNIST3D (octahedral 
), and Spherical MNIST (
), against norm, gated, Fourier-ELU, max pooling, and augmented-CNN baselines.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths.

A single tested differentiable interface for invariants previously scattered across group-specific implementations. Line coverage is 96.8%; the selective forward pass is sub-millisecond at the tested band-limits.
The 
-on-
 construction has coefficient count 
, matching the 
 orbit-space dimension and the 
 lower bound (Proposition 7).
Low-data results: the bispectrum leads at 1% PCam (0.912 vs. 0.873/0.876) and 5% OrganMNIST3D (33.2% vs. 18.9% max pool); on Spherical MNIST it reaches 95.0% vs. 79.2% for the matched-capacity power spectrum.
Weaknesses.

Section 5 status. The spherical invariant is augmented: the bootstrap block 
 is rank-deficient on real signals (rank 
 vs. 
, Remark 18), so CG-power scalars are added to recover rank. Completeness is empirical only. The bispectrum-vs-augmentation split and the proven-vs-empirical split should be stated more sharply.
Band-limit mismatch. Completeness is probed at 
 (footnote 1) while the classifier runs at 
. The deployed band-limit is not the one validated.
Empirical scope. The spherical evidence rests on Spherical MNIST; the greedy index selection is validated in one setting; all datasets are small. At matched capacity the bispectrum is competitive, not best (Fourier-ELU leads PCam at 0.945; max pool matches/leads OrganMNIST3D). The abstract's "consistently outperform" overstates this.
High-capacity failure. At 
/6.3M params the bispectrum drops to 68.5% vs. 78.5% max pool (Table 11), attributed to overfitting; the advantage is data-regime-specific.
Benchmarking. All timings are on one H100. There are no CPU or consumer-GPU numbers and no CUDA-version compatibility statement, which matters for a library contribution.
Baselines. No comparison to tensor-product/e3nn-style or scattering invariants, or to other complete spherical invariants (Bendory et al. 2025; Edidin & Satriano 2024). The Cohen 
CNN row uses published numbers, not a re-run.
Imprecision. The abstract states selectivity reduces finite-group "computational cost from 
 to 
"; Table 1 shows selective forward cost is 
 — only the coefficient count is 
. Minor: coverage is reported as both 96% and 96.8%; some checklist cross-references point to the wrong sections (the Section 5 construction is cited as Section 6).
Writing is otherwise clear.

Quality: 3: good
Clarity: 3: good
Significance: 3: good
Originality: 3: good
Questions:
Is generic completeness of the 
-on-
 invariant conjectured or only empirical? A statement suffices; a proof is not expected.
Add a classification ablation for the spherical construction: bootstrap triples only 
 + even self-couplings 
 + CG-power scalars. This isolates what drives accuracy and tests whether the greedy selection generalizes.
Add spherical evidence beyond Spherical MNIST (e.g. random band-limited signals or a second dataset) and report reconstruction at 
, the band-limit used by the classifier.
Do the sub-millisecond runtimes hold on a consumer GPU and/or CPU, and across CUDA versions/backends?
Add one stronger invariant baseline (tensor-product/e3nn-style readout or scattering), at least on Spherical MNIST.
The bispectrum degrades at 6.3M params on OrganMNIST3D (971 samples) while max pool improves. Can you separate overfitting from a capacity limitation — e.g. train/test curves or a regularization sweep at 
, or high-capacity results on a larger dataset?
Limitations:
Mostly addressed: the paper notes the missing completeness proof, the missing 
 inversion API, conditioning at large band-limits, and missing e3nn/MACE/scattering comparisons. Add two points: (i) the finite-group selective bispectrum has theoretical support whereas the spherical invariant is empirical, and the two should not be presented at equal confidence; (ii) completeness is validated at 
, not the deployed 
.

Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
None

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Rebuttal by Authors
Rebuttalby Authors (Simon Mataigne, Nina Miolane, Adele Myers, Johan Mathe)27 Jul 2026, 19:04 (modified: 28 Jul 2026, 06:14)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
Thank you for the careful, technically precise review and for recognizing the value of the uniform tested interface, the optimal-order coefficient count, and the low-data results. You are right that the original text did not separate the theoretical status of the finite-group construction from that of the spherical construction sharply enough.

Status of the spherical invariant
Generic completeness is conjectured, not proved. Our evidence is empirical: generic Jacobian-rank tests and signal reconstruction. We will state this explicitly in the abstract, Section 5, Table 1, and Limitations. We will also use "augmented selective invariant" for the full output 
 and reserve "selective bispectrum" for its degree-3 bispectral subset. The distinction matters because the bootstrap bispectral block has generic rank approximately 
, rather than 
, on the real-signal slice. Even self-couplings and degree-4 CG-power scalars provide the additional independent constraints.

This status differs from the finite-group and disk modules, whose generic completeness follows from the cited results under their stated assumptions. We will add precisely this proven-versus-empirical distinction to Limitations, as requested.

Requested spherical evidence
Feature-set ablation. We ran the requested controlled ablation. Omitted features are masked to zero, so all four variants share the identical 768-channel input, MLP, parameter count (231,818), optimizer, and three seeds (mean 
 population std, accuracy %):

Features (active real channels)	NR/NR	NR/R
bootstrap bispectral triples (496)	
+ even self-couplings (604)	
+ CG-power scalars (758)	
full model (+ 5 auxiliary bispectral entries, 768)	
Each augmentation stage changes accuracy by more than cross-seed variability: the Jacobian-motivated even self-couplings add +0.4 points and the CG-power scalars a further +0.5. The five auxiliary entries of the production model are not needed for accuracy (-0.2). NR/R equals NR/NR at every stage, so rotation invariance holds regardless of feature set. This directly tests whether the Jacobian-motivated augmentation also matters for classification rather than only for reconstruction - it does.

Band-limit and out-of-dataset reconstruction. Validating only at 
 while classifying at 
 was a real gap. We repeated the reconstruction at 
 (
 grid) with the classifier's feature construction, 4 optimization restarts per signal and 12 alignment restarts, on 16 Spherical MNIST signals (8 digits, identity + one random rotation each) and 16 random Gaussian band-limited signals. With pre-registered thresholds (feature residual 
, aligned image residual 
): on MNIST signals, all 16 recoveries converged in feature space (median residual 
, IQR 
), the invariance residual under rotation was 
 (median), and 62.5% additionally passed the aligned-image threshold (median aligned residual 0.27, IQR 
, with successful cases sitting at the SHT discretization floor). On random signals the optimizer plateaued just above the feature threshold (median 
, IQR 
, only 3/16 below threshold) and no case passed alignment (median aligned residual 1.19), so the joint success rate is 0% and we report it as such rather than adjusting thresholds post hoc. We cannot currently distinguish optimization difficulty from a genuine failure mode on this family. We will add both results and replace the current 
/
 limitation with the measured 
 scope. These experiments remain evidence, not a proof, and do not rule out exceptional signals or near-collisions.

Portability and runtime
We agree that one H100 is insufficient for a library claim. We benchmarked the same public implementation (paper-scale selective configurations, batch 16, float32, 10 warm-up iterations, median synchronized wall-clock over 600 timed forwards, including the SHT inside 
-on-
) on an NVIDIA A100 80GB (PyTorch 2.11.0, CUDA 13.0), its x86-64 host CPU, and an Apple M5 Pro (PyTorch 2.11.0, MPS and CPU):

Device/backend	Fastest module	
-on-
 (
)	Slowest module
A100, CUDA 13.0	0.098 ms (torus)	0.232 ms	0.812 ms (octahedral)
x86-64 CPU	0.082 ms (cyclic)	402 ms	402 ms (spherical)
M5 Pro, MPS	0.062 ms (torus)	0.641 ms	0.719 ms (disk)
M5 Pro, CPU	0.016 ms (cyclic)	1.811 ms	1.811 ms (spherical)
All six benchmarked configurations execute correctly on CUDA, MPS, and both CPUs with no unsupported operations. The seventh module, 
-on-
, is a thin wrapper that inherits the cyclic module's computation exactly, so the cyclic row covers it. "Sub-millisecond" holds for every configuration on both tested GPU backends at batch 16. On the M5 Pro CPU, five modules are sub-millisecond and the spherical module takes 1.811 ms. The x86 host is substantially slower on grid-based modules. We will scope the abstract's claim to the tested GPU settings and add the full per-module timing and compatibility matrix to the appendix and documentation.

High-capacity OrganMNIST3D result
"Likely overfits" was indeed not established by test accuracy alone. At 
 channels we now report train/validation curves, final test results, and a controlled regularization sweep: 7 configurations 
 3 seeds = 21 runs (max and bispectral baselines at weight decay 
, plus bispectral with weight decay 
/
, dropout 0.2/0.5, and 
+0.2), validation-AUC model selection, test evaluated once after training. At the best-validation epoch, max pooling reaches 
% train / 
% test, the bispectral baseline 
% train / 
% test, and the best regularized bispectral model (weight decay 
) 
% train / 
% test. The submitted single-seed Table 11 values (68.5% and 78.5%) fall within these cross-seed ranges. Train-test gaps are small and uniform (0.067-0.087) for every configuration. The curves do not show the overfitting signature: bispectral train accuracy is lower than max pooling's, train tracks validation throughout, and stronger regularization raises train and test together without closing the gap to max pooling. The supported diagnosis is an optimization/capacity limitation of the bispectral model at this width rather than memorization. We will replace the speculative sentence in the paper with this evidence. This experiment is also reported in our response to Reviewer LgCU.

Baselines and scope
A stronger spherical invariant baseline would clearly improve the evaluation. Within the response window we prioritized the ablation and reconstruction experiments requested above, and could not also tune a new tensor-product or scattering pipeline to a fair standard. We commit to adding this tuned comparison to the camera-ready and retain the missing-baseline limitation until then. We do not claim superiority to those methods. The revised related-work discussion will also distinguish our construction from the complete but cubic-size invariants of Edidin-Satriano and the three-shell frequency-marching result of Bendory et al. The current comparison table already records their completeness and scaling.

Corrections to claims
We accept the reviewer's corrections:

"selectivity reduces computational cost from 
 to 
" will become "selectivity reduces the coefficient count from 
 to 
" (Table 1's selective forward cost is 
).
"consistently outperform" will be replaced by the narrower observed claim: bispectral pooling improves data efficiency in the tested low-data, moderate-capacity settings, while matching or trailing alternatives elsewhere.
Coverage will be reported consistently as 96.8%.
The checklist references to the Section 5 construction will be corrected.
Finally, the hidden phrases noted under "Paper Formatting Concerns" were not inserted by the authors. They are a conference-side watermark added to reviewer-facing PDFs to detect prohibited LLM-generated reviews: the NeurIPS organizing committee acknowledged the intervention in a statement to The Transmitter, reviewers have publicly reported the identical text across their entire NeurIPS 2026 review stacks (including in reviewer-facing copies of their own submissions), and the committee has been informing reviewers who notice it not to penalize individual papers. Our PDF and LaTeX sources do not contain this text, and the Program Chairs can confirm the watermark.

Official Comment by Authors
Official Commentby Authors (Simon Mataigne, Nina Miolane, Adele Myers, Johan Mathe)31 Jul 2026, 10:46Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:
Dear Reviewer yWuC,

As the discussion window closes Monday (AoE), we would like to follow up on our response. It answers your questions with new measurements: the requested feature-set ablation (each augmentation stage adds accuracy beyond seed noise), reconstruction re-run at the deployed band-limit 
, and timings across an A100, an x86 CPU, and an Apple M5 Pro (CPU and MPS) showing every benchmarked configuration runs on every backend. The corrections you identified are all adopted, and we have committed to the stronger baseline for the camera-ready.

We found the review unusually helpful, and the paper is stronger for it - thank you. If the response resolves your concerns, we would appreciate a short confirmation and, if you consider it justified, a revised score. Should any point remain open, we are happy to clarify it before the deadline.

Best regards, The Authors

Official Review of Submission10902 by Reviewer AZ1B
Official Reviewby Reviewer AZ1B22 Jun 2026, 04:58 (modified: 04 Aug 2026, 06:19)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer AZ1BRevisions
Summary:
This paper introduces a novel efficient implementation of bispectra for several groups with a PyTorch-compatible API. Bispectra are group invariants that are complete, i.e. if two bispectra agree then the signal from which they were computed agree up to a group transformation. Bispectra can be used either on top of an equivariant neural network to make equivariant features invariant (demonstrated with experiments on finite groups) or as a feature extractor on top of which a non-equivariant network can be trained, yielding model that is overall invariant (demonstrated with experiments on sphericalMNIST). The paper uses a subset of the total bisprectrum that is still complete to achieve efficiency.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths
The paper presents a novel software package to compute bispectra for a number of groups and domains in a unified way. Such an implementation is currently not available and the presented package could improve adoption of bispectra in geometric deep learning and is therefore a valuable contribution to the community. The paper therefore enables usage of bispectra by more applied researchers unfamiliar with the mathematical and algorithmic background and more detailed studies of the different uses of bispectra in (equivariant) neural networks.

The paper also provides thorough studies of the efficiency of the provided package and some ablations of performance of bispectra.

Furthermore, the paper is well-written and points out contributions and limitations clearly.

Weaknesses
The biggest weakness of this paper is the lack of a significant conceptual, theoretical, empirical or algorithmic contribution. As Table 2 summarizes, all implemented algorithms to compute selective bispectra were previously available, with 
 on 
 being the only exception. However, as the paper notes, this constitutes a moderate extension of an algorithm by Mataigne et al. (2024). The crucial completeness of the bispectrum is not proven analytically but only tested empirically, although proofs exist for the other used bispectra (Table 7).

The experiments provided do show a benefit of the bispectrum, but only in the very low data regime, limiting the scope of the applicability of the proposed package. This is consistent across both tasks in Sections 6.1 and 6.2 and noted by the authors. Therefore, I rated the paper as "not good" on significance.

Conclusion
The proposed software package seems like a valuable contribution to the community that could boost the adoption of bispectra. However, since the main contribution of the paper are implementations of existing algorithms in a consistent API and the paper is weak in conceptual, theoretical, empirical or algorithmic contributions, in it’s current form, it is unsuitable for NeurIPS. This is the basis for my rating on originality below. I therefore recommend rejection.

Quality: 3: good
Clarity: 4: excellent
Significance: 2: not good
Originality: 2: not good
Questions:
Could you outline what key technical difficulties you had to overcome in order to extend the selective bispectrum to SO(3) as compared to the algorithms available in the literature?
Can you outline the steps necessary to prove the completeness of the selective SO(3) bispectrum? Are there any conceptual obstacles for which it is unclear how to overcome them?
In the spherical MNIST experiments, you currently left the S2CNN-results out of Table 4 and wrote them in the main text. However these seem like a significant baseline. I would expect that a central advantage of the bispectra in comparison to S2CNNs lies in the faster inference. Did you measure inference times for these models?
I also spotted the following typos:

The coefficients 
 referred to in l. 100 following do not appear in (3)
It would be good to state the formal definition of completeness (l. 208) in Section 2
Table 2: TorusDnTorus should be TorusonTorus
Limitations:
yes

Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
No concerns.

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Rebuttal by Authors
Rebuttalby Authors (Simon Mataigne, Nina Miolane, Adele Myers, Johan Mathe)27 Jul 2026, 19:07 (modified: 28 Jul 2026, 06:14)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
Thank you for recognizing that the package fills an unavailable capability, could broaden adoption of bispectra, and is well written and thoroughly evaluated. We agree that most group-specific formulas build on prior theory. The contribution is to make them usable together as tested, differentiable PyTorch components. We respectfully disagree, however, that the paper lacks a substantive algorithmic or empirical contribution.

The work contributes more than a repackaging of formulas: (i) one uniform autograd-compatible implementation for seven group/domain pairs, including precomputed CG/DFT/Bessel buffers and selective sparse contractions, (ii) profiled kernel-level optimizations that shrink the full representations by up to 
, (iii) a new augmented selective 
-on-
 construction with an asymptotically optimal 
 coefficient count, an 
 lower bound, a parity/vanishing analysis, and an explicit diagnosis and repair of the real-signal rank deficit, and (iv) the first controlled evaluation of these invariants as pooling layers across planar, volumetric, and spherical tasks.

The empirical benefit is also not confined to very-low-data subsets. At full Spherical MNIST data, the augmented spherical invariant reaches 95.0% versus 79.2% for a matched-capacity power-spectrum model, isolating information loss rather than classifier size. At 10% PCam, it is within 0.4 AUC points of the best method, and at full-data OrganMNIST3D it matches max pooling at moderate capacity. The low-data gains are where completeness helps most, but not the only demonstrated functionality: the library also provides near-exact invariance without augmentation and preserves orbit information that norm/max pooling discards.

During the response period we added two measurements that bear directly on the contribution's substance. A parameter-matched feature ablation on Spherical MNIST shows each component of the augmented construction adds accuracy beyond cross-seed variability (three seeds: bootstrap triples 94.09%, + even self-couplings 94.48%, + CG-power scalars 95.00%), so each component of the augmentation is functionally necessary. We also benchmarked six paper-scale module configurations on an NVIDIA A100, an x86-64 CPU, and an Apple M5 Pro (CPU and MPS): every configuration runs on every backend, sub-millisecond on both GPUs. Details are in our response to Reviewer yWuC.

Q1: What made the spherical extension technically difficult?
The finite-group selection argument cannot be transferred by replacing a finite irrep index with 
:

Homogeneous-space rank deficiency. For signals on a finite group, each generic Fourier coefficient is a nonsingular matrix, which supports the recursive argument of Mataigne et al. On 
, the coefficient at degree 
 is only a vector 
. Kakarala's full-rank operator-bispectrum theorem therefore does not directly apply.
A linear chain is dimensionally insufficient. A band-limited real spherical signal has 
 degrees of freedom modulo three rotational degrees. We prove that every smooth/polynomial complete invariant therefore needs at least 
 live components, so an 
 Mataigne-style frequency chain cannot be complete.
Reality and parity break the naive bootstrap. On real signals, each scalar bispectrum entry is either real or purely imaginary, and every odd-parity entry with a repeated index vanishes identically. After removing these zero rows, the nominal 
-row bootstrap block has generic rank only approximately 
, not 
.
Rank must be repaired without losing selectivity. Our construction adds mandatory even self-couplings and greedily selected degree-4 CG-power invariants until the per-degree real Jacobian reaches full rank. This keeps 
 entries per degree and hence 
 overall.
These are the reasons the result is an augmented selective invariant, rather than a direct application of the finite-group algorithm. We will foreground this technical chain in Section 5.

Q2: What would a completeness proof require?
We conjecture generic completeness. The current Jacobian and reconstruction tests provide evidence but fall short of a proof. A proof would proceed in four steps:

Prove that the full low-degree seed separates a generic orbit and fixes a common rotational gauge (including orientation, using a nonzero odd-parity triple).
Induct on 
, assuming all lower-degree coefficients have been recovered in that common gauge.
Prove that the selected bootstrap, self-coupling, and CG-power equations have a unique generic real solution for 
.
Characterize the algebraic exceptional set where a seed coefficient vanishes, a rank drops, or a nontrivial stabilizer remains.
The unresolved conceptual step is (3). Full generic Jacobian rank gives local identifiability but does not exclude a second, globally separated solution. The finite-group matrix-inverse proof does not resolve this because the spherical coefficients are rank-deficient vectors, and the real-signal parity identities create extra algebraic dependencies. A proof likely needs either an explicit rational frequency-marching inverse or an invariant-field argument showing that the selected polynomials generate the generic orbit-separating field. We will add this proof roadmap and state "conjectured generic completeness" explicitly.

Q3: S2CNN baseline and inference
The published S2CNN values should indeed appear in Table 4 rather than only in prose, and we will move them there. We did not run S2CNN in the submitted experiments, so we cannot infer an apples-to-apples speed advantage from its published accuracies, and porting the original implementation to a current PyTorch/CUDA stack was not achievable within the response window. We commit to a matched S2CNN re-run for the camera-ready, on the same hardware, precision, and measurement protocol as our timing benchmarks, and will report the resulting accuracy and measured inference times. Until then we will label the accuracy comparison as published and remove any implication of a measured speed advantage over S2CNN.

Corrections
Thank you for catching the three presentation issues. We will (i) correct the coefficient notation/cross-reference following Eq. (3), (ii) add the formal definition that an invariant 
 is complete on a set 
 when 
 iff 
 for some 
 and 
, and (iii) change TorusDnTorus to the library's actual class name, TorusOnTorus.

Official Comment by Authors
Official Commentby Authors (Simon Mataigne, Nina Miolane, Adele Myers, Johan Mathe)31 Jul 2026, 10:46Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:
Dear Reviewer AZ1B,

Since the discussion period ends Monday (AoE), we wanted to follow up on our response to your review. It addresses your two main questions in concrete terms: what made the 
 extension technically difficult (four specific obstacles, none present in the finite-group case) and what a completeness proof would require (a four-step roadmap with the open step identified). A new parameter-matched ablation also shows that each component of the augmented construction contributes measurable accuracy, which we hope speaks to the substance of the contribution.

We appreciate the technical depth of your questions - engaging with them has sharpened the paper. If our answers change your assessment, we would be grateful for an updated evaluation. If not, we would value hearing which concerns remain so we can address them before the deadline.

Thank you again, The Authors

Official Comment by Reviewer AZ1B
Official Commentby Reviewer AZ1B01 Aug 2026, 03:32Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:
Thank you for the detailed reply, and in particular for the directness of Q2. The four-step roadmap and the identification of step (3) as the unresolved conceptual obstacle is a clearer statement of the theoretical status than the submission gives. I would encourage stating "conjectured generic completeness" in the paper as you propose.

I have reconsidered how I framed my original weakness, and I want to restate it more precisely rather than as a general claim about contribution type.

On originality. The asymptotic efficiency gains the paper leads with are, as the paper itself documents, results of Mataigne et al. (2024) and Myers & Miolane (2025). The 
 reduction and the coefficient-count reduction in Figure 5 are those prior selectivity results, applied. What is new on top is implementation work: precomputed buffers, sparse contractions, a uniform module interface. I take this to be real and useful engineering, and the 
 spherical construction.

I have read Q1 carefully and accept that the spherical case is not a mechanical index substitution: the rank deficiency on the homogeneous space, the dimensional insufficiency of a linear chain, and the parity-induced rank collapse are genuine obstacles. My difficulty is with what the construction delivers. It is not shown to be complete, and Table 7 records Bendory et al. (2025) at 
 per shell with completeness established. So the construction is neither the first at this size nor accompanied by the guarantee that distinguishes the bispectrum from cheaper incomplete invariants in the first place, but this is the paper's own stated motivation. Taken together, I do not find the originality of the submitted work strong.

On significance. A unified evaluation earns its weight through the insights it produces. The insight here is largely negative and already anticipated: complete invariants help when data is scarce and are matched or overtaken otherwise (Table 11 at 6.3M params).

The one strong learned-equivariant comparison, S2CNN, is transcribed from Cohen et al. rather than run: the two systems are unmatched on capacity, preprocessing, optimization, and hardware. Within roughly a point either way the outcome is a wash, so it supports no conclusion in either direction. More broadly, clear separation appears against norm pooling and the power spectrum, both incomplete by construction and the latter provably so by the paper's own Proposition 7. Therefore, these comparisons demonstrate a consequence the paper has already proved. Max pooling is also incomplete, yet ties at moderate capacity on OrganMNIST3D (73.0 vs 72.6) and leads substantially at 6.3M parameters (78.5 vs 68.5). Fourier-ELU, which Table 1 marks as complete modulo aliasing, edges the bispectrum on PCam (0.945 vs 0.941). The tighter invariance gap (0.1 vs 2.0pp) is a real advantage and I do not discount it, but it is not the claim the paper foregrounds.

The added ablation establishes that each component of the augmentation is necessary, which I did not doubt, and the cross-platform benchmarks establish portability. Neither bears on the completeness gap or the baseline question.

Score. I am raising my score to 3 in recognition of the engineering quality and the thoroughness of the response.

 Replying to Official Comment by Reviewer AZ1B
Official Comment by Authors
Official Commentby Authors (Simon Mataigne, Nina Miolane, Adele Myers, Johan Mathe)03 Aug 2026, 20:33Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:
Dear Reviewer AZ1B,

Thank you for the careful re-reading, for restating your concerns so precisely, and for raising your score. We will state "conjectured generic completeness" in the paper as discussed. We also take your point that the decisive open items are the completeness guarantee and a matched learned-equivariant comparison - both are now explicit commitments, with the proof roadmap added to Section 5 and the matched S2CNN re-run planned for the camera-ready. We are grateful for the time you invested in this exchange.

Best regards, The Authors

About OpenReview
Contact
FAQ
Hosting a Venue
Sponsors
Terms of Use / Privacy Policy
All Venues
Donate
News
OpenReview is a long-term project to advance science through improved peer review with legal nonprofit status. We gratefully acknowledge the support of the OpenReview Sponsors. © 2026 OpenReview

