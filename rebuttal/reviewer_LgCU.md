# Response to Reviewer LgCU

Thank you for the positive assessment of the paper's quality, clarity,
originality, and potential value to the PyTorch ecosystem. We agree that the
6.3M-parameter OrganMNIST3D result needs evidence beyond the submitted
attribution to “likely overfitting.”

## OrganMNIST3D high-capacity ablation

We reran the $(16,32)$-channel max-pooling and bispectral models with identical
data splits and three seeds, logging train/validation curves and final test
results, and swept regularization while holding the backbone and optimizer
schedule fixed (7 configurations $\times$ 3 seeds; validation-AUC model
selection; accuracy % at the best-validation epoch, mean $\pm$ std):

| Pooling | Weight decay / dropout | Train ACC | Val ACC | Test ACC |
|---|---|---:|---:|---:|
| max | baseline: $10^{-4}$ / 0 | $84.9 \pm 9.7$ | $84.9 \pm 7.6$ | $76.7 \pm 5.8$ |
| bispectrum | baseline: $10^{-4}$ / 0 | $73.7 \pm 6.3$ | $77.6 \pm 3.5$ | $66.8 \pm 4.9$ |
| bispectrum | best: $10^{-3}$ / 0 | $80.5 \pm 3.7$ | $81.0 \pm 2.0$ | $72.5 \pm 1.0$ |

A train/validation gap never opens: train accuracy stays at or below
validation accuracy for both poolings across all 100 epochs, and the
train-test gap is small and uniform (0.067--0.087) in every one of the seven
configurations (the remaining four—weight decay $10^{-2}$, dropout 0.2/0.5,
and combined—reach 70.2--71.0% test). Stronger regularization raises the
bispectral model's train and test accuracy together (+5.7 test points at
weight decay $10^{-3}$) but does not close the gap to max pooling. This rules
out overfitting as the explanation: the evidence supports an
optimization/capacity limitation of the bispectral model at this width.

We will add the curves and sweep to the appendix and replace “likely overfits”
with the optimization/capacity conclusion supported by these measurements. We will also make the
scope explicit: at $(8,16)$ channels the methods are statistically equivalent
($74.5\pm0.6\%$ bispectrum versus $74.3\pm1.5\%$ max), whereas the submitted
high-capacity result is not evidence for a universal advantage of complete
pooling.

## Terminology

We also accept the clarity suggestion. Section 2 will add short operational
definitions before the formulas:

- a **Wigner-$D$ matrix** is the matrix representing a 3D rotation on the
  degree-$\ell$ spherical-harmonic coefficients;
- **Clebsch--Gordan coefficients** give the change of basis decomposing the
  tensor product of two irreducible representations; contracting with them is
  what makes each bispectral scalar rotation invariant; and
- the **parity rule** states that, for real spherical signals, a scalar
  bispectrum entry is real at even total degree and purely imaginary at odd
  total degree; odd-parity entries with a repeated degree vanish identically.

Finally, we will sharpen the status statement throughout: generic completeness
of the finite-group construction is theoretically supported, while generic
completeness of our augmented $\mathrm{SO}(3)$-on-$S^2$ invariant is
conjectured and supported empirically, not proved.
