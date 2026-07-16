# Experiments

Three benchmark experiments comparing G-bispectrum pooling against baseline invariant/equivariant pooling strategies, plus the reconstruction demo. Results feed the 3x3 grid figure built by `make_grid_figure.py`.

## Comparison protocol (all experiments)

- Invariant models are trained on canonical (non-augmented) data (`train_mode C`).
- The non-equivariant CNN baseline is trained with G-augmentation (`train_mode R`) and labeled "Aug. CNN" in the figures.
- Every run records both the canonical test metric and the rotated (OOD) test metric; line plots report the rotated metric, mean ± std over seeds 42/123/456.
- Cohen et al. (2018) S²CNN appears in the Spherical MNIST row as a published-reference dashed line (cited, not re-run).

## Setup (on the GPU machine)

```bash
git clone <repo-url> bispectrum && cd bispectrum
git checkout <branch>
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,experiments]"
```

Datasets download automatically on first use (PCam from Zenodo ≈ 8 GB, OrganMNIST3D via `medmnist`, MNIST via `torchvision`).

## Run matrix

Each script is resumable: completed runs (existing `results.json`) are skipped, so re-running after an interruption is safe.

| # | Experiment | Script | Runs | Feeds |
|---|------------|--------|------|-------|
| 1 | PCam | `pcam/run_matched_sweep.sh` (then `--phase-b`) | 5 CNN configs (R) + 5x5 equivariant (C) + 7 so2_disk, x3 seeds | row 1 params |
| 2 | PCam | `pcam/run_data_pareto_sweep.sh` (then `--phase-b`) | 6 models x 5 sizes x 3 seeds at ~100K params | row 1 bars + data |
| 3 | Organ3D | `organ3d/run_sweep.sh` | 4 models x 3 seeds at channels (4,8) | row 2 bars + curve anchors |
| 4 | Organ3D | `organ3d/run_wider_multiseed.sh` | 4 models x 2 wider channel configs x 3 seeds | row 2 params |
| 5 | Organ3D | `organ3d/run_dataeff_multiseed.sh` | 4 models x 4 sizes x 3 seeds | row 2 data |
| 6 | SMNIST | `spherical_mnist/run_sweep.sh` | 3 models x 2 modes x 3 seeds | row 3 bars + Cohen table |
| 7 | SMNIST | `spherical_mnist/run_capacity_sweep.sh` | 2 models x 5 widths x 3 seeds (C) | row 3 params |
| 8 | SMNIST | `spherical_mnist/run_data_efficiency.sh` | 3 models x 4 sizes x 3 seeds | row 3 data |

Expected results directories (created next to each script):

```
pcam/pcam_results_pareto/            # 1
pcam/pcam_results_data_pareto/       # 2 (n_100/ ... n_full/ subdirs)
organ3d/organ3d_results/             # 3, 4, 5 (shared)
spherical_mnist/smnist_results/      # 6, 8 (shared)
spherical_mnist/smnist_results_capacity/  # 7
```

## Running on the GPU machine (tmux)

One detached session per experiment family; each logs to a file. The three families are independent — run them on separate GPUs/machines if available (`CUDA_VISIBLE_DEVICES=<n>` before `bash` to pin a GPU).

```bash
cd ~/bispectrum/experiments

tmux new-session -d -s pcam 'cd pcam && { bash run_matched_sweep.sh && bash run_matched_sweep.sh --phase-b && bash run_data_pareto_sweep.sh && bash run_data_pareto_sweep.sh --phase-b; } 2>&1 | tee pcam_sweeps.log'

tmux new-session -d -s organ3d 'cd organ3d && { bash run_sweep.sh && bash run_wider_multiseed.sh && bash run_dataeff_multiseed.sh; } 2>&1 | tee organ3d_sweeps.log'

tmux new-session -d -s smnist 'cd spherical_mnist && { bash run_sweep.sh && bash run_capacity_sweep.sh && bash run_data_efficiency.sh; } 2>&1 | tee smnist_sweeps.log'
```

Monitor with `tmux attach -t pcam` (detach: `Ctrl-b d`) or `tail -f <experiment>/<name>_sweeps.log`.

## Pulling results back and building the figure

From the analysis machine:

```bash
REMOTE=user@gpu-machine:~/bispectrum/experiments
rsync -avz --include='*/' --include='results.json' --exclude='*' \
    "$REMOTE/pcam/pcam_results_pareto/"        experiments/pcam/pcam_results_pareto/
rsync -avz --include='*/' --include='results.json' --exclude='*' \
    "$REMOTE/pcam/pcam_results_data_pareto/"   experiments/pcam/pcam_results_data_pareto/
rsync -avz --include='*/' --include='results.json' --exclude='*' \
    "$REMOTE/organ3d/organ3d_results/"         experiments/organ3d/organ3d_results/
rsync -avz --include='*/' --include='results.json' --exclude='*' \
    "$REMOTE/spherical_mnist/smnist_results/"  experiments/spherical_mnist/smnist_results/
rsync -avz --include='*/' --include='results.json' --exclude='*' \
    "$REMOTE/spherical_mnist/smnist_results_capacity/" experiments/spherical_mnist/smnist_results_capacity/

python experiments/make_grid_figure.py           # real data -> experiments/grid_figure/
python experiments/make_grid_figure.py --mock    # layout preview -> experiments/grid_mockup/
```

Outputs: per-panel PDFs (`grid_r{row}c{col}_*.pdf`), per-row legends, an assembled contact sheet (PNG + PDF), and `caption.txt` with the figure caption including the S²CNN justification.
