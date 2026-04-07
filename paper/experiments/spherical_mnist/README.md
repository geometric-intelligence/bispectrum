# Spherical MNIST — SO(3) Bispectrum on S²

Classification experiment on MNIST digits projected onto the sphere, following
the evaluation protocol from Cohen et al. (2018) "Spherical CNNs".

## Motivation

The PCam and OrganMNIST3D experiments demonstrate bispectral pooling for finite
groups (C8, D4, octahedral O). This experiment extends to **SO(3)** — the full
continuous rotation group — by computing the bispectrum of spherical harmonic
coefficients on S².

## Protocol (Cohen et al. 2018)

MNIST digits are projected onto the sphere via stereographic projection from the
north pole. Two dataset variants:

- **NR** (Non-Rotated): digit on northern hemisphere, fixed orientation.
- **R** (Rotated): each digit randomly rotated by an independent SO(3) rotation.

We train on one variant and evaluate on both, producing the accuracy table:

| Method     | NR/NR | R/R   | NR/R  |
|------------|-------|-------|-------|
| Standard   | high  | low   | ~10%  |
| Bispectrum | high  | high  | high  |

A rotation-invariant model should achieve similar accuracy regardless of
train/test rotation mismatch.

## Models

| Variant          | Features                      | Invariant?       | Params |
|------------------|-------------------------------|------------------|--------|
| `standard`       | CNN on equirectangular image   | No               | ~185K  |
| `power_spectrum` | \|\|F_l\|\|² per SH degree → MLP   | Yes (incomplete) | ~11K   |
| `bispectrum`     | SO3onS2 bispectrum → MLP      | Yes (complete)   | ~165K  |

## Running

Single model:

```bash
source /home/johmathe/bispectrum/.venv/bin/activate
export PYTHONPATH="/home/johmathe/bispectrum/src:$PYTHONPATH"
python train.py --model bispectrum --train_mode NR
```

Full sweep:

```bash
./run_sweep.sh
```

Analyze results:

```bash
python analyze_results.py --results_dir ./smnist_results
```

## Key CLI arguments

| Flag               | Default      | Description                                   |
|--------------------|--------------|-----------------------------------------------|
| `--model`          | `bispectrum` | One of: standard, power_spectrum, bispectrum   |
| `--train_mode`     | `NR`         | Training data: NR or R                         |
| `--lmax`           | `15`         | Max spherical harmonic degree                  |
| `--nlat`           | `64`         | Latitude grid points                           |
| `--nlon`           | `128`        | Longitude grid points                          |
| `--epochs`         | `50`         |                                                |
| `--batch_size`     | `256`        |                                                |
| `--patience`       | `10`         | Early stopping on val accuracy                 |
| `--full_bispectrum`| `false`      | Use O(L³) full bispectrum instead of O(L²)     |
| `--n_rotations`    | `10`         | Random rotations for robustness eval           |

## Files

```
data.py              — MNIST → sphere projection, rotation, caching
model.py             — Bispectrum, power spectrum, and CNN classifiers
train.py             — Training loop, Cohen protocol eval, sweep
run_sweep.sh         — Shell script for full experimental sweep
analyze_results.py   — Load results, print tables, generate plots
```
