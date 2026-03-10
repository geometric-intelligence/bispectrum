# Steerable Bispectrum: Analysis & Bibliography

## What is the Steerable Bispectrum?

The steerable bispectrum is a **learnable CNN layer** that achieves local rotation invariance by:

1. Convolving the image with filters expressed on a steerable basis: `profile_i(r) · e^{ikθ}` (learned radial profile × circular harmonic)
2. Computing the bispectrum triple product from the per-pixel steerable filter responses: `Θ̂_{k1} · Θ̂_{k2} · Θ̂*_{k1+k2}`

The bispectrum cancels the rotation phase `e^{ikφ}`, making the output rotation-invariant at each pixel.

## Relation to Our Library

### Naming Convention

Following our `{Group}on{Domain}` convention, the steerable bispectrum would be `SteerableSO2onR2`:

- **Group**: SO(2) (2D rotations)
- **Domain**: R² (the plane / pixel grid)
- **Prefix `Steerable`**: indicates learned filter weights (unlike our parameter-free modules)

| Class              | Learnable? | What it computes      |
| ------------------ | ---------- | --------------------- |
| `SO2onS1`          | No         | Bispectrum of f       |
| `SO2onD2`          | No         | Bispectrum of f       |
| `SteerableSO2onR2` | Yes        | Bispectrum of (w ∗ f) |

### Which existing module does it correspond to?

The per-pixel bispectrum formula is exactly `CnonCn` (equivalently `SO2onS1`): at each pixel, you have `n_harmonics` complex Fourier coefficients indexed by angular frequency k, and the triple product `f̂_{k1} · f̂_{k2} · f̂*_{k1+k2}` is the abelian bispectrum on Z/nZ.

It is **not** `SO2onD2`, because the bispectrum only couples angular frequencies — the radial structure is absorbed into the convolution, not into the bispectrum itself. `SO2onD2` would require bispectral coefficients mixing across radial orders (Bessel indices).

### Canonical difference from our modules

Our modules are **parameter-free transforms** (signal in → invariant descriptor out). The steerable bispectrum is a **learnable layer** (signal + learned filter weights → invariant features). The `Θ̂_l` depend on both the signal f and the learned weights `ŵ^{(l)}`.

### Is it really a bispectrum?

Yes, the formula is the textbook abelian bispectrum. But the `Θ̂_k` are not proper Fourier coefficients in the harmonic analysis sense — they're steerable filter responses that depend on learned, non-orthogonal radial profiles. The rotation invariance property holds, but the completeness/injectivity/invertibility guarantees of the proper bispectrum do not.

### Connection to our tex document

Section "Steerable Bispectrum: After Convolution on Steerable Basis" in `bispectral_signatures_of_data.tex` formalizes exactly this idea for Z/nZ, deriving the output coefficients `Θ̂_l` from convolution on a steerable basis and noting you can compute the bispectrum directly from them.

The tex document also shows where this breaks for non-abelian groups: for D₄ via steerable CNNs, the intertwiner basis has L=3 elements but D₄ has M=5 irreps, so you can't recover true Fourier coefficients from the steerable CNN output. This subtlety doesn't arise for the abelian SO(2) case.

### Could we speed up their code?

No. The bispectrum step (element-wise triple product) is already trivial. The bottleneck is the convolution (`CHConv2D`). Also: framework mismatch (theirs is TensorFlow, ours is PyTorch).

### The gap we could fill

The Depeursinge group established the steerable bispectrum as a practical CNN layer but never addressed **selectivity** or **inversion**. A `SteerableSO2onR2` module could apply selective bispectral indices to steerable filter responses, giving both the practical CNN benefits (locality, learned radial profiles) and theoretical guarantees (minimal coefficients, invertibility).

## Reference Implementation

Cloned to `third_party/2d_bispectrum_cnn` from https://github.com/voreille/2d_bispectrum_cnn

Key files:

- `src/models/layers.py`: `BCHConv2D` (bispectrum layer), `CHConv2D` (circular harmonic convolution), `ECHConv2D` (power spectrum baseline)
- `src/models/models.py`: `BispectUnetLight`, `SpectUnetLight` (U-Net architectures using these layers)

## Bibliography

### Foundational Theory

1. **Kakarala (1992)** — *Triple Correlation on Groups*. PhD thesis, UC Irvine.
   The foundational result: the bispectrum is a complete invariant for functions on compact groups, up to group action.

2. **Kakarala (2012)** — *The Bispectrum as a Source of Phase-Sensitive Invariants for Fourier Descriptors: A Group-Theoretic Approach*. JMIV 44, 341–353. [Springer](https://link.springer.com/article/10.1007/s10851-012-0330-6)
   Unified derivation for all compact groups + constructive recovery algorithm for SO(3). Shows bispectrum discriminates shapes that power spectrum cannot.

### Steerable / Equivariant CNN Foundations

3. **Worrall et al. (2017)** — *Harmonic Networks: Deep Translation and Rotation Equivariance*. CVPR 2017. [arXiv:1612.04642](https://arxiv.org/abs/1612.04642)
   Origin of circular harmonic filters (`e^{ikθ}`) in CNNs. SO(2)-equivariant feature maps with magnitude pooling (power spectrum) for invariance — not bispectrum. This is the `ECHConv2D` (spectrum-only) baseline.

4. **Weiler et al. (2018)** — *Learning Steerable Filters for Rotation Equivariant CNNs*. CVPR 2018. [PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Weiler_Learning_Steerable_Filters_CVPR_2018_paper.pdf)
   General framework: filters as linear combinations of atomic steerable filters. Arbitrary angular resolution. Does not use bispectrum for invariance.

5. **Weiler & Cesa (2019)** — *General E(2)-Equivariant Steerable CNNs*. NeurIPS 2019. [arXiv:1911.08251](https://arxiv.org/abs/1911.08251)
   General theory solving kernel constraints for arbitrary E(2) subgroups. Unifies H-Nets, GCNNs, and steerable CNNs. The `e2cnn` library. Uses norm/gated nonlinearities, not bispectrum.

### The Steerable Bispectrum Line (Depeursinge group)

6. **Andrearczyk, Oreiller, Depeursinge (2019)** — *Exploring Local Rotation Invariance in 3D CNNs with Steerable Filters*. MIDL 2019. [PMLR](https://proceedings.mlr.press/v102/andrearczyk19a.html)
   3D steerable filters with SH × learnable radial profiles, pooled by magnitude (power spectrum). Notes that magnitude pooling loses phase.

7. **Oreiller, Andrearczyk, Fageot, Prior, Depeursinge (2020)** — *3D Solid Spherical Bispectrum CNNs for Biomedical Texture Analysis*. [arXiv:2004.13371](https://arxiv.org/abs/2004.13371)
   **The key paper.** Introduces the steerable bispectrum CNN: project onto circular/spherical harmonics, then compute the bispectrum instead of magnitude pooling. Shows bispectrum > power spectrum for texture discrimination. Theoretical foundation of the `2d_bispectrum_cnn` repo.

8. **Oreiller et al. (2022)** — *Robust Multi-Organ Nucleus Segmentation Using a Locally Rotation Invariant Bispectral U-Net*. MIDL 2022. [PMLR](https://proceedings.mlr.press/v172/oreiller22a.html)
   Application paper for the `2d_bispectrum_cnn` repo. Bispectral U-Net on histopathology (MoNuSeg). Shows rotation robustness without data augmentation.

9. **voreille/ssbcnn** — [GitHub](https://github.com/voreille/ssbcnn)
   Follow-up extending to scale + rotation steerable bispectrum.

### Related: Bispectrum in Equivariant Neural Networks

10. **Thomas, Smidt et al. (2018)** — *Tensor Field Networks*. [arXiv:1802.08219](https://arxiv.org/abs/1802.08219)
    SO(3)-equivariant networks on point clouds using SH filters × learnable radial functions. Same filter factorization as the steerable bispectrum, but uses tensor product nonlinearities (Clebsch-Gordan) instead of bispectrum pooling.

11. **Mataigne et al. (2024)** — *The Selective G-Bispectrum and its Inversion*. NeurIPS 2024. [arXiv:2407.07655](https://arxiv.org/abs/2407.07655)
    Selective O(|G|) bispectrum with inversion for finite groups. The non-steerable (parameter-free) approach implemented in our library.
