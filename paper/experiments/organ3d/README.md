# OrganMNIST3D — O-Equivariant 3D ResNet with Bispectral Pooling

3D classification experiment on OrganMNIST3D (11 organ classes, 28³ voxels, 1-channel CT) comparing invariant pooling strategies in octahedral-equivariant 3D CNNs.

## Motivation

The PCam experiment demonstrates bispectral pooling for 2D cyclic/dihedral groups. This experiment extends the idea to 3D using the chiral octahedral group O (|O| = 24), testing whether complete invariant pooling via the selective G-bispectrum (`OctaonOcta`) outperforms simpler alternatives (max/norm pooling) on 3D medical volumes.

## Architecture

All equivariant variants share the same O-equivariant 3D ResNet backbone:

```
Stem:     LiftingConv3d(1 → C0, 3³) + EquivBN + ReLU         [28³]
Stage 1:  2× BasicBlock(C0) → AvgPool3d(2)                    [14³]
Stage 2:  GroupConv3d(C0→C1, 1³) proj + 2× BasicBlock(C1) → AvgPool3d(2)  [7³]
Stage 3:  2× BasicBlock(C1) + EquivBN + ReLU                  [7³]
Pool:     {BispectrumPool3d | GroupMaxPool3d | GroupNormPool3d}
Head:     GlobalAvgPool3d → Linear(head_dim, 11)
```

Default channels: `C0=4, C1=8`. Exact voxel permutations (no interpolation) for kernel rotations.

| Variant    | Intermediate nonlin | Final pool        | Params |
|------------|--------------------|--------------------|--------|
| standard   | ReLU               | GlobalAvgPool3d    | ~16K   |
| max_pool   | ReLU               | GroupMaxPool3d     | ~374K  |
| norm_pool  | NormReLU3d         | GroupNormPool3d    | ~375K  |
| bispectrum | ReLU               | BispectrumPool3d   | ~463K  |

The `standard` variant is a plain 3D ResNet with geometric augmentation (random octahedral rotations at train time).

## Dataset

[OrganMNIST3D](https://medmnist.com/) — 1,742 CT organ volumes (train=971, val=161, test=610).

```bash
pip install medmnist
```

## Running

Single model:

```bash
source /home/johmathe/bispectrum/.venv/bin/activate
export PYTHONPATH="/home/johmathe/bispectrum/src:$PYTHONPATH"
python train.py --model bispectrum --data_dir ./organ3d_data
```

Full sweep:

```bash
./run_sweep.sh              # Phase A: 4 models × seed 42, skip rotation
./run_sweep.sh --phase-b    # Phase B: 4 models × seeds {123, 456}, with rotation eval
```

Analyze results:

```bash
python analyze_results.py --results_dir ./organ3d_results
```

## Key CLI arguments

| Flag              | Default      | Description                              |
|-------------------|--------------|------------------------------------------|
| `--model`         | `bispectrum` | One of: standard, max_pool, norm_pool, bispectrum |
| `--channels`      | `4 8`        | Channel widths (C0, C1)                  |
| `--epochs`        | `100`        |                                          |
| `--batch_size`    | `32`         |                                          |
| `--lr`            | `1e-3`       |                                          |
| `--patience`      | `15`         | Early stopping on val AUC                |
| `--skip_rotation` | `false`      | Skip 24-rotation robustness eval         |
| `--sweep`         | `false`      | Run all 4 models × 3 seeds              |
| `--dry_run`       | `false`      | Print model info, exit                   |

## Rotation robustness

All variants are evaluated on all 24 octahedral rotations of the test set. Equivariant models with bispectral/norm/max pooling should show near-zero variance across rotations, while the standard CNN should degrade.

## Files

```
data.py              — OrganMNIST3D dataset + augmentations
model.py             — O-equivariant 3D ResNet (4 variants)
train.py             — Training loop, metrics, rotation eval, sweep
run_sweep.sh         — Shell script for full experimental sweep
analyze_results.py   — Load results, print tables, generate plots
```
