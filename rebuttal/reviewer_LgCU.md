# Response to Reviewer LgCU

Thank you for the positive assessment of the paper's quality, clarity,
originality, and potential value to the PyTorch ecosystem. We agree that the
6.3M-parameter OrganMNIST3D result needs evidence beyond the submitted
attribution to “likely overfitting.”

## OrganMNIST3D high-capacity ablation

We reran the $(16,32)$-channel max-pooling and bispectral models with identical
data splits and three seeds, logging train/validation curves and final test
results, and swept regularization while holding the backbone and optimizer
schedule fixed:

| Pooling | Weight decay / dropout | Train ACC | Val ACC | Test ACC |
|---|---|---:|---:|---:|
| max | [[E4]] | [[E4]] | [[E4]] | [[E4]] |
| bispectrum | baseline: $10^{-4}$ / 0 | [[E4]] | [[E4]] | [[E4]] |
| bispectrum | [[E4: best regularization]] | [[E4]] | [[E4]] | [[E4]] |

[[E4: report when the train/validation gap opens and whether stronger
regularization closes it. State only the supported diagnosis: overfitting,
optimization failure, or a capacity/readout limitation.]]

We will add the curves and sweep to the appendix and replace “likely overfits”
with the conclusion supported by these measurements. We will also make the
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
