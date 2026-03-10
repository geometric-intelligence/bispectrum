# Paper Plan: The Steerable Bispectrum for Non-Abelian Groups

**Target venue**: NeurIPS 2026
**Working title**: *Bispectral Invariants from Steerable Convolutions: Bridging Equivariant Networks and Complete Invariant Theory*

## 1. One-Paragraph Pitch

Steerable CNNs (Weiler & Cesa 2019) achieve equivariance by design but rely on ad-hoc nonlinearities (norms, gates) to extract invariant features — losing information. The bispectrum (Kakarala 1992, Mataigne et al. 2024) provides a mathematically complete invariant with reconstruction guarantees — but has only been used with proper Fourier coefficients, not inside learned architectures. For abelian groups (SO(2)), combining the two is trivial: the Depeursinge group (Oreiller et al. 2020, 2022) showed this works as a CNN layer. For non-abelian groups (D_n, O), nobody has done it — because steerable CNN outputs live in an intertwiner space whose dimension L doesn't match the number of irreps M, so the standard bispectrum formula doesn't apply. We solve this problem.

## 2. The Problem

### 2.1 What exists

**Steerable CNNs** decompose filters on an intertwiner basis for a group H acting on R²:
$$\\Psi(x) = \\sum\_{l=1}^L \\hat{w}^{(l)} Y_l(x)$$
where Y_l ∈ Hom^H(π, ρ). The convolution output is:
$$\\Theta(x_0) = \\sum\_{l=1}^L \\hat{\\Theta}\_l \\cdot Y_l(x_0)$$

For abelian H = SO(2): the basis Y_l = e^{ilθ} is indexed by irreps, so L = M and Θ̂_l ARE the Fourier coefficients. The bispectrum β\_{k1,k2} = Θ̂\_{k1} · Θ̂\_{k2} · Θ̂\*\_{k1+k2} works directly. This is what the Depeursinge group implements.

For non-abelian H (e.g. D₄): the intertwiner basis has L = dim Hom^H(π, ρ₀) elements. For D₄ with ρ₀ = trivial: L = 3. But D₄ has M = 5 irreps. **L ≠ M**. The Θ̂_l are NOT Fourier coefficients. You cannot plug them into the bispectrum formula.

### 2.2 What's missing

A bispectral invariant that can be computed from the L intertwiner coefficients {Θ̂_l} of a steerable convolution output, for non-abelian H, with:

- Proven invariance to the action of H
- Completeness guarantees (or characterization of what information is lost)
- Selectivity (minimal number of coefficients)
- A practical CNN layer implementation

## 3. Proposed Contributions

### Contribution A: Bispectrum from Intertwiner Coefficients (Theory)

**Approach 1 — Lift to Fourier coefficients**: Express the L intertwiner coefficients in terms of the M Fourier coefficients. Since the intertwiner basis Y_l can be decomposed on the irreps of H, there exists a (possibly rectangular) matrix A such that:
$$\\hat{f}_\\rho = A_\\rho \\cdot [\\hat{\\Theta}\_1, ..., \\hat{\\Theta}\_L]^T$$
When L < M, this is underdetermined — some Fourier components are not recoverable from the intertwiner coefficients. The bispectrum computed from the recoverable components gives a **partial bispectrum** that is still H-invariant but may not be complete.

**What to prove**:

- (P1) Characterize the kernel of the map Θ̂ → F̂: which Fourier components are invisible to the steerable convolution with trivial output representation ρ₀?
- (P2) Show that choosing different output representations ρ (not just trivial) in different layers gives access to different Fourier components — and that a sufficient set of ρ's recovers all M components.
- (P3) Derive the bispectrum formula in terms of {Θ̂_l^{(ρ)}} for multiple output representations ρ.

**Approach 2 — Direct triple correlation** (already started in our tex, lines 2043–2174): Define a triple correlation directly for vector-valued signals with the induced representation action, prove invariance (done in tex), compute its Fourier transform to get the bispectrum (done in tex, lines 2194–2312). The result already has the standard form:
$$\\beta(f)_{\\rho_1, \\rho_2} = \[\\mathcal{F}(f)_{\\rho_1} \\otimes \\mathcal{F}(f)_{\\rho_2}\] \\cdot C_{\\rho_1,\\rho_2} \\bigoplus\_{\\rho \\in \\rho_1 \\otimes \\rho_2} \\mathcal{F}(f)_\\rho^\\dagger \\cdot C_{\\rho_1,\\rho_2}^\\dagger$$

**What to prove**:

- (P4) Show that F(f)\_ρ can be computed from the steerable convolution output by appropriately choosing the output representation ρ — i.e., that for each irrep ρ of H, running a steerable conv with output representation ρ yields F(f)\_ρ directly.
- (P5) If so, the standard bispectrum + selectivity (Mataigne et al. 2024) applies immediately — we just need multiple "heads" in the conv layer, one per irrep of H.

### Contribution B: Selective Steerable Bispectrum (Theory)

Once Contribution A establishes which bispectral formula to use, apply the selectivity results from Mataigne et al. 2024:

- For D_n: selective bispectrum needs ⌊(n-1)/2⌋ + 2 matrix-valued coefficients instead of O(|D_n|²)
- For O (octahedral): 4 matrix-valued coefficients

**What to prove**:

- (P6) The selectivity theorem (Mataigne et al. 2024, Theorem 4.3) applies to the steerable bispectrum, provided the Fourier coefficients are correctly extracted from the steerable conv output.
- (P7) Derive the per-pixel output channel count for the selective steerable bispectrum as a function of the group H. Compare with naive (full) bispectrum channel count.

### Contribution C: SteerableBispectrum Layer (Implementation)

A PyTorch `nn.Module` that:

1. Takes feature maps from a steerable conv layer (e.g., from e2cnn)
2. Extracts Fourier coefficients from the steerable representation
3. Computes the selective bispectrum using precomputed CG matrices
4. Outputs H-invariant feature maps

Interface sketch:

```python
class SteerableBispectrum(nn.Module):
    def __init__(self, group: str, in_type, selective: bool = True):
        """
        Args:
            group: "C4", "D4", "SO2", etc.
            in_type: e2cnn FieldType describing input representation
            selective: use selective O(|H|) or full O(|H|²) bispectrum
        """

    def forward(self, x: GeometricTensor) -> torch.Tensor:
        """
        Returns: (batch, H, W, n_bispectral_coeffs) invariant feature maps
        """
```

This should interoperate with `e2cnn` / `escnn` for the equivariant conv layers, and use our existing bispectrum library (`DnonDn`, `OctaonOcta`) for the bispectral computation.

## 4. Proofs Needed (Summary)

| ID  | Statement                                                | Difficulty                | Status                                     |
| --- | -------------------------------------------------------- | ------------------------- | ------------------------------------------ |
| P1  | Kernel of intertwiner → Fourier map for ρ₀               | Medium                    | Open                                       |
| P2  | Multiple ρ's recover all Fourier components              | Medium                    | Open (conjecture from rep theory)          |
| P3  | Bispectrum formula in terms of multi-ρ Θ̂                 | Medium                    | Open                                       |
| P4  | Steerable conv with output ρ yields F(f)\_ρ              | Key result                | Partially done in tex (needs verification) |
| P5  | Standard bispectrum applies to multi-head steerable conv | Easy (follows from P4)    | Open                                       |
| P6  | Selectivity applies to steerable setting                 | Easy (follows from P4+P5) | Open                                       |
| P7  | Channel count comparison table                           | Easy (combinatorics)      | Open                                       |

**Critical path**: P4 → P5 → P6. If P4 holds, everything else follows from existing results.

## 5. Experiments

### Experiment 1: Rotation-Invariant Classification (CIFAR-10/100 rotated)

**Setup**: Compare on rotated CIFAR-10/100:

- Baseline 1: Standard CNN + data augmentation
- Baseline 2: Steerable CNN (e2cnn) with norm nonlinearity (Weiler & Cesa 2019)
- Baseline 3: Steerable CNN with SO(2) bispectrum nonlinearity (Oreiller et al. 2020)
- **Ours**: Steerable CNN with D_n / D₄ selective bispectrum nonlinearity

**What we expect**: The D₄ bispectrum preserves more information than SO(2) (handles reflections), and is more efficient than the norm nonlinearity (which discards phase entirely). The selective version should match or beat the full bispectrum with fewer channels.

**Why it's impactful**: Direct comparison of invariant pooling strategies on a standard benchmark. Nobody has compared bispectrum vs norm/gate nonlinearities in steerable CNNs head-to-head.

### Experiment 2: Medical Image Segmentation (MoNuSeg / DRIVE)

**Setup**: Reproduce the Oreiller et al. 2022 experiments with:

- Their SO(2) bispectral U-Net (BCHConv2D baseline)
- Our D₄ selective bispectral U-Net
- Standard equivariant U-Net with norm nonlinearity

**What we expect**: D₄ handles reflections (common in histopathology — cells have no preferred chirality), giving better rotation+reflection robustness than SO(2)-only bispectrum.

**Why it's impactful**: Direct improvement on their published results, on their own benchmark, using the same architecture but with a stronger invariant.

### Experiment 3: Scaling with Group Size / Angular Resolution

**Setup**: Measure feature map memory and inference time as a function of:

- Number of harmonics n (for SO(2) case: n = 4, 8, 16, 32, 64)
- Group order |H| (for finite groups: C₄, C₈, D₄, D₈, D₁₆)

Compare: full bispectrum O(|H|²) vs selective bispectrum O(|H|) vs norm O(|H|) channel counts.

**What we expect**: Selective bispectrum is the only option that is both complete (unlike norm) and tractable (unlike full bispectrum) at large |H|.

**Why it's impactful**: Shows the selectivity result has practical consequences for steerable CNN design. Makes the case that selective bispectrum is the right nonlinearity for high-resolution equivariant networks.

## 6. Paper Outline

1. **Introduction** (1 page)

   - Steerable CNNs achieve equivariance but invariant extraction is a bottleneck
   - Bispectrum provides complete invariants but only for proper Fourier coefficients
   - We bridge the gap for non-abelian groups

2. **Background** (1.5 pages)

   - Steerable CNNs, intertwiners, kernel constraint (Weiler & Cesa 2019)
   - G-Bispectrum, selectivity, inversion (Mataigne et al. 2024)
   - Steerable bispectrum for SO(2) (Oreiller et al. 2020)

3. **The Steerable Bispectrum for Non-Abelian Groups** (3 pages)

   - The L ≠ M problem (Section 3.1)
   - Multi-representation extraction of Fourier coefficients (Section 3.2, Proofs P1–P4)
   - Bispectrum formula from steerable outputs (Section 3.3, Proof P5)
   - Selective steerable bispectrum (Section 3.4, Proofs P6–P7)

4. **Implementation** (1 page)

   - SteerableBispectrum layer
   - Integration with e2cnn/escnn
   - Channel count analysis

5. **Experiments** (2 pages)

   - Rotated CIFAR classification
   - Medical image segmentation
   - Scaling analysis

6. **Conclusion** (0.5 pages)

Total: ~9 pages + references + appendix (NeurIPS format)

## 7. Bibliography

### Foundational

- Kakarala (1992). *Triple Correlation on Groups*. PhD thesis, UC Irvine.
- Kakarala (2012). *The Bispectrum as a Source of Phase-Sensitive Invariants for Fourier Descriptors: A Group-Theoretic Approach*. JMIV 44.
- Freeman & Adelson (1991). *The Design and Use of Steerable Filters*. IEEE TPAMI 13(9).

### Steerable / Equivariant CNNs

- Cohen & Welling (2016). *Group Equivariant Convolutional Networks*. ICML 2016.
- Worrall et al. (2017). *Harmonic Networks: Deep Translation and Rotation Equivariance*. CVPR 2017.
- Weiler et al. (2018). *Learning Steerable Filters for Rotation Equivariant CNNs*. CVPR 2018.
- Weiler & Cesa (2019). *General E(2)-Equivariant Steerable CNNs*. NeurIPS 2019.
- Thomas et al. (2018). *Tensor Field Networks*. arXiv:1802.08219.
- Cesa, Lang, Weiler (2022). *A Program to Build E(N)-Equivariant Steerable CNNs*. ICLR 2022.

### Steerable Bispectrum (Depeursinge group)

- Andrearczyk, Oreiller, Depeursinge (2019). *Exploring Local Rotation Invariance in 3D CNNs with Steerable Filters*. MIDL 2019.
- Oreiller et al. (2020). *3D Solid Spherical Bispectrum CNNs for Biomedical Texture Analysis*. arXiv:2004.13371.
- Oreiller et al. (2022). *Robust Multi-Organ Nucleus Segmentation Using a Locally Rotation Invariant Bispectral U-Net*. MIDL 2022.

### Selective Bispectrum

- Mataigne et al. (2024). *The Selective G-Bispectrum and its Inversion: Applications to G-Invariant Networks*. NeurIPS 2024.
- Myers & Miolane (2025). *The Selective Disk Bispectrum and Its Inversion*. arXiv:2511.19706.

### Nonlinearities in Equivariant Networks

- Kondor & Trivedi (2018). *On the Generalization of Equivariance and Convolution in Neural Networks to the Action of Compact Groups*. ICML 2018.
- Lang & Weiler (2020). *A Wigner-Eckart Theorem for Group Equivariant Convolution Kernels*. ICLR 2021.
- Brandstetter et al. (2022). *Geometric and Physical Quantities Improve E(3) Equivariant Message Passing*. ICLR 2022.

## 8. Timeline (Speculative)

| Phase                            | Duration         | Deliverable                                                |
| -------------------------------- | ---------------- | ---------------------------------------------------------- |
| Theory: P4 proof (critical path) | 3–4 weeks        | Theorem + proof that steerable conv → Fourier coefficients |
| Theory: P5–P7 (follow from P4)   | 1–2 weeks        | Selectivity theorem for steerable setting                  |
| Code: SteerableBispectrum layer  | 2–3 weeks        | PyTorch module + e2cnn integration                         |
| Experiments: CIFAR + MoNuSeg     | 3–4 weeks        | Tables + figures                                           |
| Writing                          | 2–3 weeks        | Full draft                                                 |
| **Total**                        | **~12–16 weeks** |                                                            |

## 9. Risk Assessment

**Main risk**: P4 might not hold as stated. The steerable conv with output representation ρ might not directly yield the Fourier coefficient F(f)\_ρ — there could be a mixing between spatial (Z²) and fiber (H) components that prevents clean extraction.

**Mitigation**: Even if full Fourier recovery is impossible from a single conv layer, a multi-layer extraction (stack of steerable convs with different ρ's) might work. Alternatively, the partial bispectrum from the recoverable components might already be practically useful — test empirically.

**Fallback**: If the non-abelian theory doesn't close, the paper can pivot to contributions B+C only: the selective steerable bispectrum for SO(2) with Fourier-Bessel basis (our SO2onD2 as a conv layer), benchmarked against the Depeursinge group's approach. This is less novel but still publishable.
