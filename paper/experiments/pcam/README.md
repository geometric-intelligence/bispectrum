# PatchCamelyon Experiment: Bispectral Nonlinearity in Equivariant CNNs

## Context

Equivariant CNNs (Cohen & Welling 2016, Weiler & Cesa 2019) achieve rotation
equivariance by construction, but they must eventually extract
**rotation-invariant** features for classification. The choice of invariant map
is a fundamental design decision with no consensus solution:

| Method | Nonlinearity | Invariant map | Complete? | Exact equivariance? | Cost per layer | Key reference |
| -------------- | ----------------------------------------- | ------------------------------------------ | ------------------------------ | ----------------------- | -------------- | ---------------------------------- |
| **Standard** | ReLU | Data augmentation (flips, 90° rot) | N/A | N/A | O(C) | Cohen & Welling, ICML 2016 |
| **Norm** | ReLU(‖x‖\_G − bias) · x/‖x‖\_G | GroupMaxPool | No — discards phase | Yes | O(C) | Weiler & Cesa, NeurIPS 2019 |
| **Gate** | σ(‖x\_gate‖\_G) · x\_feat | GroupMaxPool | No — scalar bottleneck | Yes | O(2C) | Weiler & Cesa, NeurIPS 2019 |
| **FourierELU** | FFT → upsample → ELU → downsample → IFFT | GroupMaxPool | Approximately | No — aliasing from truncation | O(C log C) | Franzen & Wand, NeurIPS 2021 |
| **Bispectrum** | ReLU (regular repr) | Selective bispectrum + 1×1 proj | **Yes — provably invertible** | **Yes** | O(C) | Kakarala 2012; Mataigne et al. 2024 |

The bispectrum is the **unique complete rotation invariant**: it preserves all
signal information except the absolute orientation (Kakarala 2012). The
selective bispectrum (Mataigne et al. 2024) reduces its cost from O(|G|^2) to
O(|G|), matching the norm nonlinearity in channel count while being provably
complete.

This experiment tests whether completeness matters in practice: does preserving
phase information via the bispectrum improve classification accuracy on a task
where rotation invariance is physically justified?

## Why PatchCamelyon?

PatchCamelyon (PCam) is a binary classification task (metastatic tissue
detection) on 327,680 histopathology patches (96x96 RGB). It is the ideal
testbed because:

1. **Rotation invariance is physically justified.** Tissue orientation under
   the microscope is arbitrary — there is no "up" direction.

2. **Phase information is discriminative.** Histopathology textures contain
   oriented sub-structures (collagen fibers, gland lumen shapes, mitotic
   spindle orientations) that distinguish benign from malignant tissue. The
   norm nonlinearity discards exactly this directional information. The
   bispectrum preserves it.

3. **Equivariance is proven to help.** Veeling et al. (MICCAI 2018) showed
   P4M-DenseNet reduces error by 27% relative over a standard DenseNet
   (15.93% → 11.64%).

4. **Large enough for statistically significant results.** 327k images with
   standard train/val/test splits.

## Experiment design

Five model variants share the same equivariant DenseNet backbone, differing
only in the nonlinearity / invariant pooling:

```
Baseline 1 (standard):     Vanilla DenseNet + geometric data augmentation
Baseline 2 (norm):         Equivariant DenseNet + NormReLU nonlinearity
Baseline 3 (gate):         Equivariant DenseNet + gated nonlinearity
Baseline 4 (fourier_elu):  Equivariant DenseNet + FFT→ELU→IFFT nonlinearity
Ours       (bispectrum):   Equivariant DenseNet + bispectral invariant pooling
```

Each configuration is run with both C8 (8 discrete rotations) and D4 (4
rotations + reflections) groups, across 3 random seeds.

### Metrics

- **AUC-ROC** — primary metric (standard for PCam)
- **Accuracy** — secondary
- **Rotation robustness** — AUC evaluated on test images rotated by 12
  arbitrary angles (0°, 15°, 30°, ..., 315°). Equivariant models should
  produce consistent AUC; the standard CNN should degrade.
- **Data efficiency** — learning curves at 1%, 5%, 10%, 50%, 100% of training
  data. The bispectrum's completeness should shine in low-data regimes.

### What we expect

1. **Bispectrum >= gate > norm > standard** on AUC, because the bispectrum
   preserves phase information that norm/gate discard.
2. **Bispectrum ≈ norm >> standard** on rotation robustness, because both are
   exactly equivariant (unlike FourierELU which is approximate).
3. **Bispectrum advantage grows in low-data regime**, because completeness
   reduces the need for the network to learn invariant features from data.
4. The **selective** bispectrum (O(|G|) coefficients) should match the full
   bispectrum in accuracy while being faster — validating the selectivity
   result of Mataigne et al. (2024).

### Existing baselines to beat

| Method                 | Accuracy | Source              |
| ---------------------- | -------- | ------------------- |
| DenseNet baseline (Z2) | 84.1%    | Veeling et al. 2018 |
| P4-DenseNet            | 87.6%    | Veeling et al. 2018 |
| P4M-DenseNet           | 88.4%    | Veeling et al. 2018 |
| Attentive P4M-DenseNet | 89.1%    | Romero et al. 2020  |

## How to run

### Prerequisites

```bash
uv pip install h5py torchvision
# The bispectrum library must be importable:
uv pip install -e /path/to/bispectrum
```

### Single run

```bash
# Standard CNN baseline
uv run train.py --model standard --data_dir ./pcam_data

# Bispectrum with C8 group
uv run train.py --model bispectrum --group c8 --data_dir ./pcam_data

# Norm nonlinearity with D4 group
uv run train.py --model norm --group d4 --data_dir ./pcam_data

# Data-efficiency experiment (10% of training data)
uv run train.py --model bispectrum --group c8 --train_fraction 0.1 --data_dir ./pcam_data
```

### Full sweep (all 5 baselines x 3 seeds)

```bash
uv run train.py --sweep --group c8 --data_dir ./pcam_data
```

This produces a summary table and saves all results to
`./pcam_results/sweep_results.json`.

### Key arguments

| Argument           | Default      | Description                                                            |
| ------------------ | ------------ | ---------------------------------------------------------------------- |
| `--model`          | `bispectrum` | One of: `standard`, `norm`, `gate`, `fourier_elu`, `bispectrum`        |
| `--group`          | `c8`         | Symmetry group: `c8` (8 rotations) or `d4` (4 rotations + reflections) |
| `--epochs`         | `100`        | Max training epochs                                                    |
| `--batch_size`     | `64`         | Batch size                                                             |
| `--lr`             | `1e-3`       | Learning rate (AdamW)                                                  |
| `--patience`       | `15`         | Early stopping patience (on validation AUC)                            |
| `--growth_rate`    | `12`         | DenseNet growth rate k                                                 |
| `--block_config`   | `4 4 4`      | Layers per dense block                                                 |
| `--train_fraction` | `1.0`        | Fraction of training data (for data-efficiency curves)                 |

Data is auto-downloaded from Zenodo on first run (~7 GB total).

## Architecture

The equivariant DenseNet is built from scratch (no escnn dependency):

```
Input: (B, 3, 96, 96)
    |
LiftingConv2d(3 → 24, 3x3)           → (B, 24, |G|, 96, 96)
EquivBatchNorm + nonlinearity
    |
DenseBlock_1(k=12, 4 layers)          → (B, 72, |G|, 96, 96)
Transition_1(compression=0.5)          → (B, 36, |G|, 48, 48)
    |
DenseBlock_2(k=12, 4 layers)          → (B, 84, |G|, 48, 48)
Transition_2                           → (B, 42, |G|, 24, 24)
    |
DenseBlock_3(k=12, 4 layers)          → (B, 90, |G|, 24, 24)
    |
EquivBatchNorm + ReLU
InvariantPool                          → (B, 90, 24, 24)
GlobalAvgPool                          → (B, 90)
Linear(90 → 1)                        → (B,)
```

The `InvariantPool` step is the experimental variable:

- **GroupMaxPool**: max over group dim (baselines 2–4)
- **BispectrumPool**: selective bispectrum via `CnonCn`/`DnonDn` (ours)

Group-equivariant convolutions use `grid_sample`-based kernel rotation.
Pointwise ReLU is equivariant for the regular representation of finite groups
(the group action permutes channels).

## File structure

```
paper/experiments/pcam/
    data.py    — PCam dataset, auto-download, transforms
    model.py   — Equivariant layers, nonlinearities, DenseNet, model factory
    train.py   — Training loop, evaluation, rotation robustness, CLI
```

## Related work

### Equivariant CNNs

- Cohen & Welling, "Group Equivariant Convolutional Networks", ICML 2016
- Weiler & Cesa, "General E(2)-Equivariant Steerable CNNs", NeurIPS 2019
- Cesa, Lang & Weiler, "A Program to Build E(N)-Equivariant Steerable CNNs", ICLR 2022

### Nonlinearities in equivariant networks

- Franzen & Wand, "General Nonlinearities in SO(2)-Equivariant CNNs", NeurIPS 2021
- Kondor, Lin & Trivedi, "Clebsch-Gordan Nets", NeurIPS 2018
- "The Price of Freedom: Expressivity/Efficiency Tradeoffs in Equivariant Tensor Products", ICLR 2025

### Bispectral approaches

- Kakarala, "The Bispectrum as a Source of Phase-Sensitive Invariants", JMIV 2012
- Mataigne et al., "The Selective G-Bispectrum and its Inversion", NeurIPS 2024
- Oreiller et al., "Robust Multi-Organ Nucleus Segmentation Using a Locally Rotation Invariant Bispectral U-Net", MIDL 2022
- Sanborn et al., "Bispectral Neural Networks", ICLR 2023
- Chevalley et al., "A Bispectral 3D U-Net for Rotation Robustness in Medical Segmentation", MICCAI-W 2024

### PCam benchmark

- Veeling et al., "Rotation Equivariant CNNs for Digital Pathology", MICCAI 2018
- Romero et al., "Attentive Group Equivariant Convolutional Networks", ICML 2020
