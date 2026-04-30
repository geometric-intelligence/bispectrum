# Spherical MNIST — SO(3) Bispectrum Reconstruction Demo

Reconstruct spherical MNIST digits from their `SO3onS2` bispectrum by
gradient descent in image space, then **rotate the recovered signal back
to the target via Procrustes-style SO(3) alignment**. The aligned recon
visually matches the target, demonstrating that the bispectrum is a
**complete SO(3)-invariant** on `S^2`: the original signal is recovered
**up to an SO(3) "phase change"** (the Wigner-D action on spherical
harmonic coefficients), which is exactly the orbit indeterminacy of any
complete invariant.

## What it shows

For each digit `f`:

1. **Invariance.** Generate `K` random rotations and check that
   `||beta(R_k . f) - beta(f)|| / ||beta(f)||` is at the SHT
   discretisation floor (~7e-3 at `nlat=64, nlon=128, lmax=12`).
2. **Round-trip recovery.** Initialise `f_hat` as a random Gaussian
   image, then minimise
   `||beta(f_hat) - beta(target)||^2 / ||beta(target)||^2`
   with Adam (cosine-annealed LR) + `--n_recon_restarts` random restarts;
   the per-target best-loss recon is kept.
3. **Phase change is real, and the phase is recoverable.** The raw
   image-space error `||f_hat - target|| / ||target||` is `O(1)` — the
   recon lives in a different orbit representative. After solving for
   the optimal `R \in SO(3)` (Adam over a quaternion parameterisation
   with multi-restart) the aligned error
   `||R . f_hat - target|| / ||target||` collapses to the SHT-rotation
   floor. The orbits figure shows target / raw recon / aligned recon
   side-by-side on actual spheres so this is visually obvious.

## Running

```bash
source /home/johmathe/bispectrum/.venv/bin/activate
export PYTHONPATH="/home/johmathe/bispectrum/src:$PYTHONPATH"

python reconstruct.py --n_digits 8 --n_rotations 2
```

Default settings (`lmax=12`, `n_steps=8000`, `n_recon_restarts=4`,
`align_n_restarts=12`) give a clean figure in ~20 min on a single GPU.
For a faster sanity check use `--n_digits 3 --n_recon_restarts 1
--n_steps 3000 --align_n_restarts 4`.

To regenerate **only the compact 2x2 paper figure** (~5 min), skip the
comprehensive sweep:

```bash
python reconstruct.py --paper_only --paper_digits 0 1
```

Outputs land in `figures/`:

| File                            | Description                                                          |
|---------------------------------|----------------------------------------------------------------------|
| `orbits.{pdf,png}`              | `(n_digits x 3(1+K))` grid: target / raw recon / aligned recon spheres |
| `paper_orbits.{pdf,png}`        | Compact `(len(--paper_digits) x 2)` figure for the NeurIPS paper     |
| `convergence.{pdf,png}`         | Median + IQR of relative bispectrum residual vs. step                |
| `invariance_vs_recon.{pdf,png}` | Per-pair scatter of invariance vs. recon residual                    |
| `results.json`                  | All scalar metrics + per-step traces                                 |

## Key CLI arguments

| Flag                       | Default | Notes                                                                   |
|----------------------------|---------|-------------------------------------------------------------------------|
| `--n_digits`               | `8`     | Digits sampled from the spherical MNIST test cache (one per class first)|
| `--n_rotations`            | `2`     | Independent random `R_k` per digit                                      |
| `--lmax`                   | `12`    | Recon at the default `64x128` grid is bounded by an SHT discretisation floor that scales with `lmax`. `lmax=12` keeps the recon comfortably below the floor; `lmax=15` (classifier setting) plateaus near it and degrades the alignment quality |
| `--nlat / --nlon`          | `64/128`| Spherical grid; must match a pre-built cache file                       |
| `--n_steps`                | `8000`  | Adam iterations per reconstruction                                      |
| `--lr`                     | `5e-2`  | Initial Adam LR (cosine-annealed to `lr * 1e-2`)                        |
| `--n_recon_restarts`       | `4`     | Random Adam restarts per recon; per-sample best is kept                 |
| `--align_n_restarts`       | `12`    | Quaternion restarts per alignment (`0` disables alignment)              |
| `--align_n_steps`          | `200`   | Adam steps per alignment restart                                        |
| `--align_lr`               | `1e-1`  | Initial alignment Adam LR (cosine-annealed)                             |
| `--render`                 | `sphere`| `sphere` = orthographic 3D view; `equirectangular` = old flat layout    |
| `--view_size`              | `128`   | Orthographic view resolution                                            |
| `--elev_deg / --azim_deg`  | `25/30` | Fallback / fixed-view camera direction (degrees)                        |
| `--fixed_view`             | off     | Disable per-panel auto-centering on the signal centroid (use a single shared camera direction instead) |
| `--paper_digits`           | `0 1`   | `digit_idx` values to use for the compact `paper_orbits.pdf` figure    |
| `--paper_figure_path`      | auto    | Override output path; defaults to `<output_dir>/paper_orbits.pdf`       |
| `--paper_only`             | off     | Run only the digits in `--paper_digits` and emit only the paper figure (fast regeneration path) |
| `--full_bispectrum`        | off     | `O(L^3)` full bispectrum instead of selective `O(L^2)`                  |
| `--no_bandlimit_project`   | off     | Disable the per-step `IRealSHT(RealSHT(.))` projection                  |
| `--seed`                   | `0`     | Controls digit selection, rotations, Gaussian init, and alignment seeds |
| `--data_dir`               | `../spherical_mnist/smnist_data` | Where the cached `test_NLATxNLON.pt` lives |

## Implementation notes

- **Differentiability.** `SO3onS2.forward` skips the CUDA-graph fast path
  whenever `torch.is_grad_enabled()`, so optimisation runs through the
  pure-PyTorch sparse gather-multiply-scatter kernels and Adam can
  backprop end-to-end through the SHT and the Clebsch-Gordan contraction.
- **Band-limit projection.** The bispectrum only constrains SH
  coefficients up to `lmax`. Without `IRealSHT(RealSHT(.))` between Adam
  steps, the `f_hat` would carry Gaussian junk in the high-frequency
  null space (visually noisy, but irrelevant to the loss). The projection
  step is essentially free and keeps the reconstructions visually clean.
- **Cosine-annealed LR + best-loss tracking.** Adam without LR decay
  oscillates around the minimum; we anneal to `lr * 1e-2` and keep the
  per-target best (lowest-loss) `f_hat` across all steps so late-stage
  jitter cannot worsen the result.
- **Multi-restart reconstruction.** The bispectrum loss landscape has
  shallow local minima for some digits. Running Adam from
  `--n_recon_restarts` independent Gaussian inits and keeping the
  per-sample best dramatically tightens the residual on hard cases at
  ~linear cost.
- **SO(3) alignment.** We parameterise `R \in SO(3)` as a unit quaternion
  (auto-normalised on the fly to dodge the `(\alpha, \beta, \gamma) =
  (0, 0, 0)` Euler gimbal-lock singularity that yields NaN gradients
  through the `arctan2` / `acos` chain in the spherical sampler), and
  minimise `||rotate(f_hat, R(q)) - target||^2` with Adam + cosine
  annealing. Multi-restart with Haar-uniform `randn(4)/||.||` quaternions
  is essential because the loss has many local minima from the digit's
  pseudo-symmetries.
- **Sphere rendering.** Each panel in `orbits.png` is an orthographic
  projection of the spherical signal. By default each camera points at
  the panel's positive-mass centroid (the digit ends up centered in the
  disk), with a faint lat/lon graticule overlaid so the eye can pick up
  rotation differences between panels. Aligned and target panels share
  the same view because they're the same signal up to alignment quality.
  Use `--fixed_view` (with `--elev_deg / --azim_deg`) to lock all panels
  to one direction. Equirectangular plotting is still available via
  `--render equirectangular`.
- **No `SO3onS2.invert()`.** The library currently raises
  `NotImplementedError` for SO(3) bispectrum inversion (`DESIGN.md`
  TODO-M4). This script is the empirical demo of feasibility, not a
  proposed inversion API.

## Files

```
reconstruct.py        - End-to-end script (CLI + figures + JSON)
README.md             - This file
figures/              - Output PDFs / PNGs / JSON (gitignored)
```
