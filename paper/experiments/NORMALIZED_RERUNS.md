# Normalized Experiment Reruns

Use `run_normalized_experiments.sh` from `paper/experiments` to validate and run the normalized experiment matrix. The script activates the `bispectrum` conda environment and the repository `.venv`, writes all outputs under `paper/experiments/normalized_results`, and excludes `so2_disk`.

First run memory checks only:

```bash
cd paper/experiments
bash run_normalized_experiments.sh check
```

This enumerates every planned `(dataset, model, train_mode, train_size)` memory shape at `CHECK_SEED=42` and runs a one-batch forward/backward probe. Seeds do not change tensor shapes, so `check` mode intentionally does not iterate over `SEEDS`. The manifest is written to:

```text
paper/experiments/normalized_results/memory_manifest.jsonl
```

Only after the memory manifest is clean, launch training manually:

```bash
cd paper/experiments
bash run_normalized_experiments.sh run
```

`run` mode now gates every training command on a passing entry in `memory_manifest.jsonl`. If the run matrix changes (new model, batch size, train size, etc.), wipe and regenerate the manifest first:

```bash
cd paper/experiments
rm -f normalized_results/memory_manifest.jsonl
bash run_normalized_experiments.sh check
```

Without a `status=ok` entry that matches the planned config, `run` aborts (or skips with `FAIL_FAST=0`).

The default training seed set is five seeds. `SEEDS` affects `run` mode; `check` mode uses `CHECK_SEED`:

```bash
CHECK_SEED=42 bash run_normalized_experiments.sh check
SEEDS="42 123 456 789 101112" bash run_normalized_experiments.sh run
```

To extend headline cells to ten seeds, override `SEEDS` for `run` mode and rerun in resume mode. Completed `results.json` files are skipped by default:

```bash
SEEDS="42 123 456 789 101112 2024 2025 31415 27182 16180" \
  bash run_normalized_experiments.sh run
```

Validate result coverage and seed counts:

```bash
cd paper/experiments
bash run_normalized_experiments.sh analyze
```

The analyzer fails if required protocol cells are missing or have fewer than five seeds. It writes:

```text
paper/experiments/normalized_results/normalized_summary.json
paper/experiments/normalized_results/normalized_table.tex
```

## Data-efficiency normalization

Data-efficiency curves are now normalized by **absolute training sample count** instead of dataset-specific fractions, so the x-axis (`N`) is directly comparable across PCam, OrganMNIST3D, and Spherical MNIST. Default sweeps:

| Dataset | Sizes (N) | Full-set N |
| --- | --- | --- |
| PCam | `100 500 2500 12500 full` | 262,144 |
| OrganMNIST3D | `100 500 full` | 971 |
| Spherical MNIST | `100 500 2500 12500 full` | 60,000 |

`full` is encoded by omitting `--train_size`; subset selection is reproducible via `torch.Generator(seed)` and the JSON manifests under `normalized_results/subsets/<dataset>_train_n<N>_seed<SEED>.json`. Output directory tags are `n_<N>` (e.g. `pcam/n_500/...`) or `n_full` for full-set runs.

## Useful overrides

```bash
MEMORY_HEADROOM=0.80 bash run_normalized_experiments.sh check
FAIL_FAST=0 bash run_normalized_experiments.sh check
RESUME=0 bash run_normalized_experiments.sh run
PCAM_SIZES="2500 12500 full" bash run_normalized_experiments.sh check
ORGAN_SIZES="100 full" bash run_normalized_experiments.sh check
SMNIST_SIZES="500 12500 full" bash run_normalized_experiments.sh check
```

After runs complete, regenerate dataset-specific figures from the normalized result roots and use `normalized_table.tex` as the source of truth for paper table cells. Paper captions should report the protocol, seed count, training set size N, parameter budget, and memory-checked batch size.
