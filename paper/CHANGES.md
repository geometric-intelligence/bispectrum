# Changes Since Submission

Every experiment in the camera-ready version was re-run after the author response, when we found bugs in the library, the training protocol, and the pooling head.
Several numbers therefore differ from the submitted version and from the author response. This
document records what changed and why.

## Library (`bispectrum` 0.4.0)

- **`SO3onS2`.** For $\ell \ge 8$ the chain and cross rows of the selection
  encoded the same real equation (see the remark "Distinct equations per degree" in the
  appendix), so the linear rows had rank at most $\ell + 1 < 2\ell + 1$ and, even with the
  self-couplings and CG-power entries, the per-degree Jacobian fell short of full rank by
  $\lfloor(\ell-6)/2\rfloor$ at every $\ell \ge 8$ (20 dimensions in total at $L=15$).
  Identically zero entries were also kept. The corrected selection has full per-degree rank,
  checked numerically (Section "The Augmented Selective Spherical Bispectrum"), and 380
  coefficients at $L=15$ (submitted: 384) and 426 at $L=16$ (submitted: 430). The submitted
  reconstruction figure used the rank-deficient map.
- **`DnonDn`.** The fold block was dropped for odd $n \ge 5$, leaving Jacobian
  rank $2n - 1$, so the map was not locally injective. Odd $n$ gain the missing coefficients.
  Even $n$, including the $D_{32}$ benchmark, are unchanged.
- **`SO2onDisk`.** The Bessel zeros were off by $\sim 5 \times 10^{-7}$ and
  the basis had condition number $\sim 10^{16}$. The default band limit is now the Nyquist
  limit, which changes the disk row of the benchmark table from 105 to 76 coefficients.

## Training protocol

- **Sample floor.** An epoch was one pass over the $n$ training examples, so at
  $n=100$ an epoch was about two optimizer steps. Every epoch now draws at least 10,000
  examples. This changes every low-data result. For example, the Spherical MNIST bispectrum at
  $n=100$ goes from 11.7% to 62.7%.
- **Fixed epochs.** Early stopping fired on the accuracy dips at SGDR warm restarts,
  and batch sizes differed between models. Runs now train for a fixed number of epochs ending
  a cosine cycle, at one batch size per dataset (tuned per model on Spherical MNIST), and are
  evaluated at the best-validation checkpoint (appendix "Experimental Details").
- **Also changed.** Aug. CNN is trained with augmentation at every training-set size.
  The rotated test is the primary metric. Norm pooling is added as an incomplete-invariant
  baseline. We use 10, 3–5, and 5 seeds on OrganMNIST3D, PCam, and Spherical MNIST, report the
  sample standard deviation, and test differences with Welch $t$-tests. S2CNN and a
  tensor-product network are re-run under our protocol with matched tuning
  (Spherical MNIST figure).

## Pooling head

The submitted pooling head computed $\mathrm{proj}(\mathrm{ReLU}(\mathrm{signlog}(\beta)))$,
with $\mathrm{signlog}$ the signed logarithm of the real and imaginary parts and $\mathrm{proj}$
the learned $1\times 1$ projection, so its ReLU zeroes every negative $G$-bispectral
coefficient before the projection. The camera-ready head computes
$\mathrm{ReLU}(\mathrm{proj}(\mathrm{signlog}(\beta)))$. It was chosen on validation AUC alone
(table below). Spherical MNIST never used this head.
On $O$ the truncated invariant keeps full Jacobian rank
(200 of 200 random inputs) yet costs $\approx 5$ pp, while on $C_8$ it is rank-deficient
(24 of 200) yet costs nothing measurable, so Jacobian rank does not explain the gap. We suspect that the
surviving subset changes with every input, so the fixed projection reads different coordinates
at every voxel, and the zeroed entries pass no gradient. The rule we adopt is to apply only fixed
injective maps (real–imaginary split, signed log) before the first learned layer. Because we compared four heads for the bispectrum but only one per baseline,
the table also reports the submitted head. Norm pooling has no truncated variant, because its outputs are
non-negative and a ReLU leaves them unchanged.

**Moving the ReLU after the projection adds 4–5 pp on OrganMNIST3D at full data.** Test
accuracy (OrganMNIST3D, %) or AUC (PCam) at the best-validation checkpoint, mean ± std over 3
seeds. The submission head applied the ReLU to the signed-log bispectrum before the $1\times 1$
projection, which zeroes every negative coefficient. The camera-ready head (bold) was chosen on
validation AUC alone, by mean rank over the five settings. It makes no significant difference
at $n=100$ or on PCam. $p$: Welch test against the submission head.

| Head | OrganMNIST3D 6.3M | OrganMNIST3D 463K | OrganMNIST3D 463K, $n=100$ | PCam $n=100$ | PCam $n=12.5$K |
| --- | --- | --- | --- | --- | --- |
| Submission: ReLU, then projection | 73.9 ± 1.5 | 73.6 ± 2.0 | 61.0 ± 0.9 | .812 ± .024 | .939 ± .011 |
| **Projection, then ReLU** | **77.9 ± 1.9** | **78.6 ± 0.3** | **61.1 ± 2.6** | **.824 ± .021** | **.936 ± .007** |
| $p$ vs. submission | 0.045 | 0.049 | 0.93 | 0.55 | 0.73 |
| + power spectrum | 78.1 ± 1.2 | 76.9 ± 2.4 | 62.4 ± 2.7 | .813 ± .031 | .940 ± .008 |
| + power spectrum, cube root | 77.5 ± 0.7 | 79.0 ± 3.7 | 62.7 ± 2.8 | .821 ± .003 | .936 ± .005 |
| Max pool | 76.0 ± 2.1 | 73.1 ± 0.6 | 63.1 ± 1.8 | – | – |
| Norm pool | 68.2 ± 3.7 | 63.3 ± 2.7 | 57.9 ± 3.0 | .802 ± .038 | .933 ± .011 |
| Aug. CNN | – | – | – | .813 ± .042 | .928 ± .009 |

## Spherical feature ablation

The table below repeats the ablation of the augmented selective spherical bispectrum
on the corrected `SO3onS2`. Only the Clebsch–Gordan power spectra measurably help
accuracy. The self-couplings do not change accuracy and are included for the Jacobian rank.
The classifier uses all 760 real features.

**Of the augmentation steps, only the Clebsch–Gordan power spectra measurably help.** Spherical
MNIST accuracy (%) as the augmented selective spherical bispectrum at $L=15$ is built up. Each
row adds entries to the one above, with the MLP width fixed and the other entries masked to
zero. Mean ± std over 5 seeds, at the default training settings (batch 256, learning rate
$10^{-3}$, 30 epochs), whereas the Spherical MNIST figure uses tuned settings. $p$: Welch test
of NR/R against the row above.

| Entries | Real features | NR/NR | NR/R | $p$ |
| --- | ---: | --- | --- | --- |
| Base bispectral entries | 480 | 95.20 ± 0.09 | 95.18 ± 0.11 | – |
| + even self-couplings $\beta_{\ell,\ell,\ell'}$ | 584 | 95.23 ± 0.10 | 95.19 ± 0.08 | 0.87 |
| + Clebsch–Gordan power spectra | 738 | 95.50 ± 0.05 | 95.51 ± 0.05 | $<10^{-3}$ |
| + degree power, other self-couplings (full) | 760 | 95.58 ± 0.10 | 95.58 ± 0.08 | 0.14 |

## Errata

The following statements from the submission or the author response no longer hold.

- **OrganMNIST3D at 6.3M parameters.** We attributed max pooling's lead to an
  optimization or capacity limitation of the bispectral model (best regularized 72.5% vs.
  76.7%). The cause was the pooling head. With the corrected head and 10 seeds, bispectral pooling
  reaches 77.6% vs. 76.0% ($p=0.10$). Likewise, the submitted tie with max pooling at
  0.46M parameters (72.6% vs. 73.0%) becomes 78.5% vs. 73.6% ($p < 10^{-3}$).
- **Spherical feature ablation.** We stated that each augmentation stage changes
  accuracy by more than the cross-seed variability. Only the Clebsch–Gordan power spectra do
  (spherical feature ablation table above).
- **Reconstruction of random signals at $L=15$.** We reported gradient descent
  plateauing just above the acceptance threshold, with 3 of 16 random signals below it. On the
  corrected invariant gradient descent stalls at a 7.5% median invariant error with 0 of 16
  below it, and the image-space residual we reported measured the rotation aligner rather than
  the invariant. The degree-by-degree solver recovers all 16 random signals
  (Section "Reconstruction" and the appendix on recovering signals).
- **CPU latency.** We reported 0.082–402 ms on x86. That run used one thread per core
  on a shared 128-core host. With the thread count fixed, every module except
  `SO3onS2` is sub-millisecond and `SO3onS2` takes 5.5–45 ms on x86
  (portability table).
- **Spherical MNIST against the power spectrum.** We reported 95.0% vs. 79.2% for a
  matched-capacity power spectrum. The numbers are now 95.8% (tuned) vs. 77.7% at 11K
  parameters and 80.1% at 550K (appendix on the capacity of the power spectrum). The conclusion holds.
- **Counts.** The spherical classifier has 760 input features and 229,770
  parameters (submitted: 768 and 231,818), and the test coverage of the library is 98.5%
  (submitted: 96.8%).
