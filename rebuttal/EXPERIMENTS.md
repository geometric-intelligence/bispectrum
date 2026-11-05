# Camera-ready experiment runbook

Run this on the GPU machine from the repository root. E1--E4 are the experiments added for
the camera-ready: device timings, reconstruction at L=15, the Spherical MNIST feature-set
ablation and the OrganMNIST3D regularization sweep.

Outputs go to `rebuttal/results/` and `experiments/*/*_results*/`, which are symlinks into
the gitignored `results/` store (see `RESULTS.md`). On a fresh clone, run
`python rebuttal/results_store.py link` first, and run `adopt` after any step that creates
a new results directory.

## 0. Reproduce the environment

```bash
git status --short
git rev-parse HEAD
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,experiments]"
mkdir -p rebuttal/results/environment
python - <<'PY' | tee rebuttal/results/environment/python.txt
import platform
import torch

print("Python:", platform.python_version())
print("Platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
PY
nvidia-smi | tee rebuttal/results/environment/nvidia-smi.txt
```

Run the normal tests before the experiments:

```bash
python -m pytest -q
```

The E2--E4 sweep runners are restart-safe: completed `results.json` files are
skipped. E1 overwrites the requested output JSON.

## E1. Device/backend forward timings

Purpose: forward latency across devices and backends. The script benchmarks the exact
paper-scale selective configurations at batch 16, including the SHT inside
`SO3onS2`, and records the environment in each JSON.

On a Linux GPU host (CUDA, then the host CPU at each of `BENCH_THREADS`, default
`1 8`):

```bash
CUDA_VISIBLE_DEVICES=0 bash rebuttal/run_camera_ready.sh timings <tag>  # e.g. h100, a100
```

This writes `<tag>_cuda.json` and `<tag>_host_cpu_t<n>.json`. Pin the CPU
threads: the default (one per core) gave 402 ms for `SO3onS2` on a shared
128-core host, against 5.5 ms at 8 threads. On a consumer NVIDIA GPU, use a tag
such as `rtx4090`.

On the Mac (bispectrum >= 0.4.1; 0.4.0 fails on 4 of the 6 modules under MPS):

```bash
bash rebuttal/run_device_bench_mac.sh  # m5pro_mps.json, m5pro_host_cpu_t<n>.json
```

MPS has no float64, so the MPS rows cast each module to float32
(`environment.module_cast_dtype`); CPU and CUDA keep the float64 tables as built.
Unsupported operations are retained as explicit errors rather than silently
dropped.

With all device JSON files in `rebuttal/results/device_bench/`:

```bash
python rebuttal/analyze_device_benchmarks.py \
  rebuttal/results/device_bench/*_cuda.json \
  rebuttal/results/device_bench/*_mps.json \
  rebuttal/results/device_bench/*_host_cpu_t*.json \
  --output rebuttal/results/device_bench/combined.json \
  | tee rebuttal/results/device_bench/combined.md
```

Expected runtime: a few minutes per device; first-time CG/Bessel cache
construction can dominate.

Bring back:

- every device JSON and log;
- `combined.json` and `combined.md`; and
- the environment files from step 0.

When reporting, give hardware, PyTorch/backend versions, batch, dtype, median
and IQR. Say “sub-millisecond” only for rows that actually satisfy it.

## E2. Reconstruction at L=15 and on random signals

Purpose: signal recovery from the selective invariant at a higher degree and on
signals that are not digits. The script uses `L=15` on a
128x256 grid, four reconstruction restarts, and 12 alignment restarts. It
tests eight Spherical MNIST signals (identity + one rotation each) and 16
independent Gaussian real band-limited signals. The random signals are
generated directly in spherical-harmonic coefficient space.

First run a one-signal smoke test:

```bash
python experiments/spherical_mnist_reconstruction/reconstruct.py \
  --signal_source random --n_digits 1 --n_rotations 0 \
  --lmax 15 --nlat 128 --nlon 256 --n_steps 20 \
  --n_recon_restarts 1 --align_n_restarts 0 \
  --device cuda --skip_figures \
  --output_dir rebuttal/results/reconstruction_smoke
```

Then run the full experiment:

```bash
bash rebuttal/run_reconstruction_evidence.sh
```

The runner creates the 128x256 Spherical MNIST test cache if absent. Expect
several GPU-hours; inspect the first 300-step log before leaving it unattended.
The summary pre-registers two transparent operational thresholds:
feature residual \<= 1e-2 and aligned image residual \<= 0.3. Report the full
median/IQR as primary evidence; the thresholded success rate is secondary.
These results remain empirical evidence, not a completeness proof.

Bring back:

- `rebuttal/results/reconstruction/{smnist_l15,random_l15}/results.json`;
- both `state.pt` files (for later figures);
- both logs; and
- `rebuttal/results/reconstruction/summary.json`.

The runner then solves the same targets again with the degree-by-degree solver,
`experiments/spherical_mnist_reconstruction/degree_marching.py`: Levenberg--Marquardt
seeded at degree 4, extended one degree at a time, on CPU in float64, about 10 minutes per
source with 12 parallel targets (`MARCH_JOBS`, `MARCH_THREADS`). This stage was designed after
the Adam results, so it is post hoc; Adam stays the pre-registered baseline. It writes
`{smnist,random}_l15_marching/results.json` and renders Fig. 1 from the Spherical MNIST run to
`smnist_l15_marching/paper_orbits.pdf`, or to `PAPER_FIGURE` when set.

When reporting, include sample count, grid, restarts, feature residual median/IQR,
aligned residual median/IQR, invariance residual for rotated MNIST, and joint
success rate. If optimization fails on many random signals, report that result
without changing thresholds post hoc.

## E3. Spherical feature-set classification ablation

Purpose: which parts of the selective feature set carry the classification signal. The
implementation masks omitted
features to zero while retaining the full 768-channel MLP input. Therefore all
four variants have exactly the same classifier and parameter count:

1. `bootstrap`: 248 bootstrap triples (496 active real channels);
2. `bootstrap_self`: +54 mandatory even self-couplings (604 channels);
3. `bootstrap_self_cg`: +77 CG-power scalars (758 channels); and
4. `full`: +3 ordinary power and 2 budget-selected self-couplings (768
   channels, the production model).

Check dimensions before training:

```bash
for features in bootstrap bootstrap_self bootstrap_self_cg full; do
  python experiments/spherical_mnist/train.py \
    --model bispectrum \
    --bispectrum_features "$features" \
    --run_label "bispectrum_${features}" \
    --dry_run
done
```

Run the 4-feature-set x 3-seed canonical-training experiment:

```bash
bash rebuttal/run_smnist_ablation.sh
```

This yields NR/NR and the key NR/R result in twelve runs. If time permits and
all twelve primary runs succeeded, add R/R with:

```bash
TRAIN_MODES=R bash rebuttal/run_smnist_ablation.sh
```

The first invocation may spend substantial time building the 64x128 Spherical
MNIST caches. Expected runtime depends strongly on SHT throughput; use the
first completed seed to estimate the remaining wall time.

Bring back:

- every `smnist_ablation/*/results.json`;
- `smnist_ablation/summary.json`; and
- the full job log.

When reporting, give active real-channel count, identical total parameter
count, mean +/- population standard deviation over three seeds for NR/NR and
NR/R, and a one-sentence interpretation. Do not describe a small difference as
causal unless it exceeds cross-seed variability.

## E4. OrganMNIST3D high-capacity regularization

Purpose: whether the OrganMNIST3D pooling gap at high capacity is overfitting. The sweep
reruns the
$(16,32)$ max and bispectral baselines with epoch-level train/validation
history, then tests five stronger-regularization settings for the bispectral
model. Model selection remains validation-AUC based; test data are evaluated
only after training.

Runs use the camera-ready protocol (70 epochs, batch 16, no early stopping);
each takes ~9 GB of GPU memory, so three fit on a 40 GB A100.

Run 7 configurations x 3 seeds = 21 runs:

```bash
bash rebuttal/run_organ3d_regularization.sh
# or one run per GPU slot:
CUDA_VISIBLE_DEVICES=3 CONFIG_LABELS=bsp_base SEEDS=42 bash rebuttal/run_organ3d_regularization.sh
```

The configurations are max baseline; bispectrum baseline; weight decay
1e-3/1e-2; dropout 0.2/0.5; and weight decay 1e-3 + dropout 0.2. The analyzer
selects each run's best validation-AUC epoch, reports the corresponding
train/validation accuracy and final held-out test accuracy, and produces:

- `organ3d_regularization/analysis/summary.json`;
- `organ3d_regularization/analysis/curves_and_sweep.pdf`; and
- the PNG companion.

Bring back those analysis files, every per-run `results.json`, and the job log.
When reporting, state only the diagnosis supported by the curves:

- high train accuracy + low validation/test accuracy, improved by
  regularization: overfitting;
- low train and validation accuracy: optimization/capacity bottleneck; or
- high train accuracy with no regularization improvement: memorization is
  present, but regularization does not explain/repair the pooling gap.

Do not use “overfitting” merely because test accuracy is lower.

## Collect the results

```bash
python rebuttal/results_store.py adopt      # new result directories into the store
python rebuttal/results_store.py manifest   # then commit rebuttal/results_manifest.tsv
```

Publishing the store is described in `RESULTS.md`.
