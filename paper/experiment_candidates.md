# Experiment Candidates: Bispectral Nonlinearity in Equivariant CNNs

*March 2026 — Research notes for NeurIPS paper experiment selection*

______________________________________________________________________

## Core Narrative

Equivariant CNNs (e2cnn/escnn) achieve rotation equivariance by design, but rely on
ad-hoc nonlinearities to extract invariant features. Every existing option has a
fundamental limitation:

| Nonlinearity         | Exact?               | Complete?                   | Cost                  | Problem                                     |
| -------------------- | -------------------- | --------------------------- | --------------------- | ------------------------------------------- |
| **Norm**             | Yes                  | **No** — discards all phase | O(L)                  | Slow convergence, loses discriminative info |
| **Gate**             | Yes                  | **No** — scalar bottleneck  | O(2L)                 | Wastes half channels on trivial gate fields |
| **FourierPointwise** | **No** — approximate | Yes                         | O(L log L)            | Aliasing from harmonic truncation           |
| **TensorProduct**    | Yes                  | Yes                         | O(L³)                 | Polynomial degree explodes with depth       |
| **Bispectrum**       | Yes                  | **Yes**                     | O(L²), selective→O(L) | **Fills the gap**                           |

The bispectrum is the unique nonlinearity that is simultaneously exact, complete, and
tractable. The selective bispectrum (Mataigne et al. 2024) brings the cost down to O(L),
matching the gate nonlinearity while being provably complete.

______________________________________________________________________

## Primary Experiment: PatchCamelyon (PCam)

**Dataset**: 327,680 color images (96×96), binary classification (metastatic tissue
detection in histopathology). Standard, well-cited benchmark.

**Why rotation invariance matters**: Tissue orientation under the microscope is arbitrary.
Veeling et al. (MICCAI 2018) showed P4M-DenseNet reduces error by 27% relative over
baseline (15.93% → 11.64%).

**Why the bispectrum specifically helps**: Histopathology textures have oriented
sub-structures (collagen fibers, gland lumen shapes, mitotic spindle orientations) that
distinguish benign from malignant. The norm nonlinearity discards exactly this
directional information. The bispectrum preserves it while maintaining rotation
invariance.

**Experiment design**:

```
Baseline 1: Standard CNN (DenseNet/WRN) + rotation augmentation
Baseline 2: escnn C8-WRN + norm nonlinearity (Weiler & Cesa 2019)
Baseline 3: escnn C8-WRN + gated nonlinearity
Baseline 4: escnn C8-WRN + FourierELU (Franzen & Wand 2021)
Ours:       escnn C8-WRN + bispectral nonlinearity (CnonCn or SO2onS1)
```

**Metrics to report**:

1. **Accuracy / AUC** — expect bispectrum ≥ gate > norm > FourierELU
2. **Rotation robustness** — evaluate on rotated test set; expect bispectrum ≈ norm >> baseline
3. **Data efficiency** — learning curves at 1%, 5%, 10%, 50%, 100% training data; equivariant + complete invariant should shine in low-data regime
4. **Channel efficiency** — selective bispectrum O(|G|) vs full O(|G|²) vs gate (2× channels)

**Existing results to beat**:

| Method                 | Accuracy | Source       |
| ---------------------- | -------- | ------------ |
| DenseNet baseline (Z2) | 84.1%    | Veeling 2018 |
| P4-DenseNet            | 87.6%    | Veeling 2018 |
| P4M-DenseNet           | 88.4%    | Veeling 2018 |
| Attentive P4M-DenseNet | 89.1%    | Romero 2020  |

**Integration path**: escnn's `EquivariantModule` interface makes this a clean drop-in.
The `CnonCn` / `SO2onS1` bispectrum module computes the invariant from Fourier
coefficients output by equivariant conv layers. Implement as an `EquivariantModule`
subclass with `in_type`, `out_type`, and `forward(GeometricTensor) -> GeometricTensor`.

**Why PCam wins over alternatives**:

- Large enough for NeurIPS (327k images)
- Rotation invariance is physically justified
- Phase information is discriminative (texture directionality matters)
- Existing equivariant baselines to beat
- The nonlinearity is the only variable — clean ablation
- No domain expertise needed for reviewers

______________________________________________________________________

## Secondary Experiment: Galaxy10 DECals

**Dataset**: 17,736 images, 10 morphology classes from DESI Legacy Imaging Surveys.

**Why**: Galaxies have perfect rotation symmetry. D16-GCNN achieves 95.52% vs ~89%
baseline (+6.5%). Spiral arm winding direction and bar orientation are discriminative
features that require phase information — exactly what norm discards and bispectrum
preserves.

**Existing results**:

| Method                | Accuracy      | Source      |
| --------------------- | ------------- | ----------- |
| Standard CNN baseline | ~89-90%       | Pandya 2023 |
| D16-GCNN              | 95.52 ± 0.18% | Pandya 2023 |

**What we expect**: Bispectrum should improve on D16-GCNN because galaxy morphology
classes (spiral vs barred spiral, ring vs lens) differ precisely in phase relationships
between angular harmonics.

**References**:

- Pandya et al. 2023, "E(2) Equivariant NNs for Galaxy Morphology", arXiv:2311.01500
- Code: https://github.com/snehjp2/GCNNMorphology

______________________________________________________________________

## Other Candidates Considered (and why they rank lower)

### Cryo-EM Viewing Direction Classification

Zhao & Singer (2014) already use exactly the SO2onDisk bispectrum pipeline
(Fourier-Bessel + bispectrum), achieving 99% accuracy at SNR=1/100. But their pipeline
is classical (non-learned). A differentiable SO2onDisk layer for cryo-EM would be a
genuine first.

**Why it ranks lower**: Requires cryo-EM domain expertise and datasets (EMPIAR). Higher
barrier. Better as a follow-up paper.

**References**:

- Zhao & Singer 2014, PMC4014198
- ASPIRE software: http://spr.math.princeton.edu/

### Retinal RNFL Analysis (Ophthalmology)

OCT peripapillary scans are literally signals on a disk/circle. Optic disc torsion
causes rotation artifacts that confound diagnosis. Fourier analysis already published
(AUC 0.93) but no one has used Fourier-Bessel + bispectrum.

**Why it ranks lower**: Very domain-specific. Better for MICCAI than NeurIPS.

**References**:

- Fourier analysis of RNFL: Nature Scientific Reports, doi:10.1038/s41598-020-67334-6

### MoNuSeg Nucleus Segmentation

Oreiller et al.'s bispectral U-Net performs slightly worse than standard U-Net (F-score
0.716 vs 0.732). A Dec 2024 benchmark (arXiv:2412.09182) found equivariance doesn't
help much on nucleus segmentation because nuclei are approximately circular. Risk of
negative result.

### Molecular ML (MACE / NequIP / SNAP)

The bispectrum is already the standard descriptor in this field (SNAP, SOAP, ACE). MACE's
symmetric contractions are already a generalized bispectrum. Would be competing with the
ACE/MACE team on their home turf. The novelty would be "cleaner implementation" rather
than "new capability."

### Crystallography (TorusOnTorus)

Equivariant GNNs (EquiformerV2) already dominate. The torus bispectrum could help with
periodic boundary conditions, but the connection is less direct and the community already
uses established tools.

______________________________________________________________________

## Closest Prior Art

### Oreiller et al. (MIDL 2022) — Bispectral U-Net

- **Architecture**: U-Net with BCHConv2D layers (SO(2) bispectrum), 2 downsampling levels
- **Group**: SO(2) via circular harmonics, bandwidth N=0..8
- **Dataset**: MoNuSeg (24 images, 10 random splits)
- **Result**: F-score 0.716 ± 0.033 vs standard U-Net 0.732 ± 0.033
- **Key finding**: Matched accuracy but near-perfect rotation robustness (RMSE ~10⁻⁵ vs ~8% for standard)
- **Limitation**: SO(2) only, no reflections, full (not selective) bispectrum
- **Code**: https://github.com/voreille/2d_bispectrum_cnn (TensorFlow, already in third_party/)

### Sanborn et al. (ICLR 2023) — Bispectral Neural Networks

- **Focus**: Learning the group from data (commutative groups only)
- **RotMNIST**: 98.1% accuracy (vs E2CNN 99.3%)
- **Key property**: Completeness + adversarial robustness
- **Code**: https://github.com/sophiaas/bispectral-networks

### Franzen & Wand (NeurIPS 2021) — General SO(2) Nonlinearities

- **Approach**: FFT → pointwise nonlinearity → IFT
- **Best MNIST-rot error**: 0.685% with ReLU
- **Limitation**: Approximate equivariance; aliasing from grid discretization

______________________________________________________________________

## Key References

| Paper                       | Venue     | Relevance                                         |
| --------------------------- | --------- | ------------------------------------------------- |
| Veeling et al. 2018         | MICCAI    | PCam benchmark, P4M-DenseNet                      |
| Weiler & Cesa 2019          | NeurIPS   | escnn, norm/gate nonlinearities                   |
| Franzen & Wand 2021         | NeurIPS   | General SO(2) nonlinearities                      |
| Kondor et al. 2018          | NeurIPS   | CG-Nets: CG product as sole nonlinearity          |
| Sanborn et al. 2023         | ICLR      | Bispectral Neural Networks, completeness          |
| Mataigne et al. 2024        | NeurIPS   | Selective bispectrum (our prior work)             |
| Oreiller et al. 2022        | MIDL      | Bispectral U-Net baseline                         |
| Kakarala 2012               | JMIV      | Bispectrum completeness theory                    |
| "Price of Freedom" 2025     | ICLR      | Tensor product expressivity/efficiency tradeoffs  |
| Pandya et al. 2023          | NeurIPS-W | Galaxy10 D16-GCNN                                 |
| arXiv:2412.09182 (Dec 2024) | arXiv     | Rotation-equivariant U-Net benchmark (5 datasets) |
| Chevalley et al. 2024       | MICCAI-W  | Bispectral 3D U-Net                               |
| Batatia et al. 2024         | Nature MI | Design space of equivariant potentials            |
