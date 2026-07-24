# Rebuttal experiment runbook

Run this on the GPU machine from the repository root. E1--E4 are ordered by
rebuttal value. Do not block the response on E5/E6.

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

Purpose: answer yWuC Q4 and fill `[[E1]]`. The script benchmarks the exact
paper-scale selective configurations at batch 16, including the SHT inside
`SO3onS2`, and records the environment in each JSON.

On the H100:

```bash
mkdir -p rebuttal/results/device_bench
python rebuttal/benchmark_devices.py \
  --device cuda \
  --output rebuttal/results/device_bench/h100_cuda.json \
  | tee rebuttal/results/device_bench/h100_cuda.log
python rebuttal/benchmark_devices.py \
  --device cpu \
  --output rebuttal/results/device_bench/h100_host_cpu.json \
  | tee rebuttal/results/device_bench/h100_host_cpu.log
```

On a consumer NVIDIA GPU, run the same first command but write
`consumer_cuda.json`. On the Mac, use `--device mps` and `--device cpu` and
write `mac_mps.json`/`mac_cpu.json`. Unsupported operations are retained as
explicit errors rather than silently dropped.

After copying all device JSON files into `rebuttal/results/device_bench/`:

```bash
python rebuttal/analyze_device_benchmarks.py \
  rebuttal/results/device_bench/*_cuda.json \
  rebuttal/results/device_bench/*_mps.json \
  rebuttal/results/device_bench/*_cpu.json \
  --output rebuttal/results/device_bench/combined.json \
  | tee rebuttal/results/device_bench/combined.md
```

If a glob has no matches, list only the files that exist. Expected runtime:
roughly 5--20 minutes per device; first-time CG/Bessel cache construction can
dominate.

Bring back:

- every device JSON and log;
- `combined.json` and `combined.md`; and
- the environment files from step 0.

For `[[E1]]`, report hardware, PyTorch/backend versions, batch, dtype, median
and IQR. Say “sub-millisecond” only for rows that actually satisfy it.

## E2. Reconstruction at L=15 and on random signals

Purpose: answer yWuC Q3 and fill `[[E2]]`. The script uses `L=15` on a
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
feature residual <= 1e-2 and aligned image residual <= 0.3. Report the full
median/IQR as primary evidence; the thresholded success rate is secondary.
These results remain empirical evidence, not a completeness proof.

Bring back:

- `rebuttal/results/reconstruction/{smnist_l15,random_l15}/results.json`;
- both `state.pt` files (for later figures);
- both logs; and
- `rebuttal/results/reconstruction/summary.json`.

For `[[E2]]`, include sample count, grid, restarts, feature residual median/IQR,
aligned residual median/IQR, invariance residual for rotated MNIST, and joint
success rate. If optimization fails on many random signals, report that result
without changing thresholds post hoc.

## E3. Spherical feature-set classification ablation

Purpose: answer yWuC Q2 and fill `[[E3]]`. The implementation masks omitted
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

For `[[E3]]`, report active real-channel count, identical total parameter
count, mean +/- population standard deviation over three seeds for NR/NR and
NR/R, and a one-sentence interpretation. Do not describe a small difference as
causal unless it exceeds cross-seed variability.

## E4. OrganMNIST3D high-capacity regularization

Purpose: answer LgCU and yWuC Q6 and fill `[[E4]]`. The sweep reruns the
$(16,32)$ max and bispectral baselines with epoch-level train/validation
history, then tests five stronger-regularization settings for the bispectral
model. Model selection remains validation-AUC based; test data are evaluated
only after training.

Memory smoke tests:

```bash
python experiments/organ3d/train.py \
  --model max_pool --channels 16 32 --batch_size 8 \
  --data_dir experiments/organ3d/organ3d_data --memory_check
python experiments/organ3d/train.py \
  --model bispectrum --channels 16 32 --batch_size 4 \
  --data_dir experiments/organ3d/organ3d_data --memory_check
```

Run 7 configurations x 3 seeds = 21 runs:

```bash
bash rebuttal/run_organ3d_regularization.sh
```

The configurations are max baseline; bispectrum baseline; weight decay
1e-3/1e-2; dropout 0.2/0.5; and weight decay 1e-3 + dropout 0.2. The analyzer
selects each run's best validation-AUC epoch, reports the corresponding
train/validation accuracy and final held-out test accuracy, and produces:

- `organ3d_regularization/analysis/summary.json`;
- `organ3d_regularization/analysis/curves_and_sweep.pdf`; and
- the PNG companion.

Bring back those analysis files, every per-run `results.json`, and the job log.
For `[[E4]]`, state only the diagnosis supported by the curves:

- high train accuracy + low validation/test accuracy, improved by
  regularization: overfitting;
- low train and validation accuracy: optimization/capacity bottleneck; or
- high train accuracy with no regularization improvement: memorization is
  present, but regularization does not explain/repair the pooling gap.

Do not use “overfitting” merely because test accuracy is lower.

## E5/E6. Optional baselines

Do not implement these during the response window unless E1--E4 are complete
and at least one full day remains:

- E5: tensor-product/e3nn or spherical-scattering baseline;
- E6: matched S2CNN inference benchmark.

Neither has an existing apples-to-apples pipeline in this repository. A rushed
implementation would be less credible than the explicit scope statements
already drafted. If skipped, use the fallback text inside `[[E5]]` and
`[[E6]]`; do not claim a measured speed advantage over S2CNN.

## Package results for transfer

```bash
git rev-parse HEAD > rebuttal/results/GIT_SHA
tar -czf rebuttal_results_10902.tar.gz rebuttal/results
shasum -a 256 rebuttal_results_10902.tar.gz
```

After transferring the archive back, replace `[[E1]]`--`[[E4]]` in the four
drafts, choose the E5/E6 fallback text if those experiments were skipped, and
recount characters before posting.
