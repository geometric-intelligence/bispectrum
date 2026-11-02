#!/usr/bin/env bash
# Camera-ready reruns on bispectrum 0.4.0 with the per-epoch sample floor.
#
#   bash rebuttal/run_camera_ready.sh archive [--dry_run]  # move stale results aside
#   bash rebuttal/run_camera_ready.sh smoke <pcam|organ3d|smnist>...
#   bash rebuttal/run_camera_ready.sh launch bongo         # tmux: gpu0 PCam, gpu1 OrganMNIST3D
#   bash rebuttal/run_camera_ready.sh launch h100          # tmux: gpu0 Spherical MNIST
#   bash rebuttal/run_camera_ready.sh <pcam|organ3d|smnist>  # one pipeline, foreground
#   bash rebuttal/run_camera_ready.sh analyze              # grid figure + significance + timings
#
# Pipelines (every sweep skips runs whose results.json exists, so re-running
# resumes):
#   pcam     bongo GPU0  data-Pareto n in {100, 500, 2500} x 6 models x 5 seeds
#   organ3d  bongo GPU1  data efficiency n in {50..500} + widths x 10 seeds,
#                        max_pool batch-16 control, A100 device timings
#   smnist   H100        data efficiency, Table 4 sweep, capacity, feature
#                        ablation x 5 seeds; reconstruction; H100 device timings
#
# Environment: SEEDS5 / SEEDS10 override the seed sets; SKIP_TESTS=1 skips
# pytest at launch; SESSION names the tmux session.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_ROOT/rebuttal/run_camera_ready.sh"
export PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
SESSION="${SESSION:-camera-ready}"
RESULTS="$REPO_ROOT/rebuttal/results"
LOG_DIR="$RESULTS/logs_camera_ready"
ENV_DIR="$RESULTS/environment_camera_ready/$(hostname -s)"
SMOKE_DIR="${SMOKE_DIR:-${TMPDIR:-/tmp}/camera_ready_smoke}"

SEEDS5="${SEEDS5:-42 123 456 789 101112}"
SEEDS10="${SEEDS10:-42 123 456 789 101112 1 2 3 4 5}"
SMOKE_MIN_VAL_ACC="${SMOKE_MIN_VAL_ACC:-0.2}"

EXP="$REPO_ROOT/experiments"
SMNIST_DATA="${SMNIST_DATA_DIR:-$EXP/spherical_mnist/smnist_data}"

banner() {
  echo ""
  echo "=== $* — $(date) ==="
}

# ---------------------------------------------------------------------------
# Preflight: the library in the venv must be the 0.4.0 map (editable install
# of this checkout), results must carry the floor provenance, CUDA must work.
# ---------------------------------------------------------------------------
preflight() {
  banner "preflight on $(hostname -s)"
  "$PYTHON" - <<'PY'
import sys
from pathlib import Path

import torch

import bispectrum

version = tuple(int(part) for part in bispectrum.__version__.split('.')[:3])
print('bispectrum', bispectrum.__version__, 'from', Path(bispectrum.__file__).parent)
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
if version < (0, 4, 0):
    sys.exit('bispectrum < 0.4.0: run `uv pip install -e ".[dev,experiments]"` in this checkout')
if not torch.cuda.is_available():
    sys.exit('CUDA unavailable: torch wheel does not match the driver')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}:', torch.cuda.get_device_name(i))
PY
}

# results.json written by a smoke run must show the floor and the library version.
check_results() {
  local results_path=$1 expected_epoch_samples=$2 min_val_acc=${3:-0}
  "$PYTHON" - "$results_path" "$expected_epoch_samples" "$min_val_acc" <<'PY'
import json
import sys
from pathlib import Path

path, expected, min_acc = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
run = json.loads(path.read_text())
version = tuple(int(part) for part in str(run.get('bispectrum_version', '0')).split('.')[:3])
problems = []
if run.get('epoch_samples') != expected:
    problems.append(f"epoch_samples={run.get('epoch_samples')} != {expected}")
if version < (0, 4, 0):
    problems.append(f"bispectrum_version={run.get('bispectrum_version')} < 0.4.0")
acc = run.get('best_val_accuracy')
if acc is not None and acc < min_acc:
    problems.append(f'best_val_accuracy={acc:.3f} < {min_acc}')
print(f'{path}: epoch_samples={run.get("epoch_samples")} version={run.get("bispectrum_version")} '
      f'best_val={acc if acc is None else round(acc, 3)}')
if problems:
    sys.exit('smoke gate failed: ' + '; '.join(problems))
PY
}

# ---------------------------------------------------------------------------
# Smoke gates: one short run per dataset into $SMOKE_DIR, then the checks.
# ---------------------------------------------------------------------------
smoke_pcam() {
  banner "smoke: PCam bispectrum n=100, 1 epoch"
  rm -rf "$SMOKE_DIR/pcam"
  (cd "$EXP/pcam" && "$PYTHON" train.py --model bispectrum --growth_rate 4 --train_mode C \
    --train_size 100 --epochs 1 --patience 1 --batch_size 128 --skip_rotation \
    --output_dir "$SMOKE_DIR/pcam" --seed 42)
  check_results "$(find "$SMOKE_DIR/pcam" -name results.json)" 10000
}

smoke_organ3d() {
  banner "smoke: OrganMNIST3D bispectrum n=50, 1 epoch"
  rm -rf "$SMOKE_DIR/organ3d"
  (cd "$EXP/organ3d" && "$PYTHON" train.py --model bispectrum --channels 4 8 --train_mode C \
    --train_size 50 --epochs 1 --patience 1 --batch_size 16 --skip_rotation \
    --data_dir ./organ3d_data --output_dir "$SMOKE_DIR/organ3d" --seed 42)
  check_results "$(find "$SMOKE_DIR/organ3d" -name results.json)" 971
}

smoke_smnist() {
  banner "smoke: Spherical MNIST dry runs (expect 760 invariant dims for the full model)"
  for features in bootstrap bootstrap_self bootstrap_self_cg full; do
    (cd "$EXP/spherical_mnist" && "$PYTHON" train.py --model bispectrum \
      --bispectrum_features "$features" --run_label "bispectrum_${features}" --dry_run)
  done
  banner "smoke: Spherical MNIST bispectrum n=100, 5 epochs (was 9.6% val acc before the floor)"
  rm -rf "$SMOKE_DIR/smnist"
  (cd "$EXP/spherical_mnist" && "$PYTHON" train.py --model bispectrum --train_mode C \
    --train_size 100 --epochs 5 --patience 5 --skip_rotation \
    --data_dir "$SMNIST_DATA" --output_dir "$SMOKE_DIR/smnist" --seed 42)
  check_results "$(find "$SMOKE_DIR/smnist" -name results.json)" 10000 "$SMOKE_MIN_VAL_ACC"
}

# ---------------------------------------------------------------------------
# Device timings (same protocol as the rebuttal run, new library version).
# ---------------------------------------------------------------------------
device_timings() {
  local tag=$1
  banner "device timings: $tag"
  mkdir -p "$RESULTS/device_bench"
  "$PYTHON" "$REPO_ROOT/rebuttal/benchmark_devices.py" --device cuda \
    --output "$RESULTS/device_bench/${tag}_cuda.json" 2>&1 | tee "$RESULTS/device_bench/${tag}_cuda.log"
  "$PYTHON" "$REPO_ROOT/rebuttal/benchmark_devices.py" --device cpu \
    --output "$RESULTS/device_bench/${tag}_host_cpu.json" 2>&1 | tee "$RESULTS/device_bench/${tag}_host_cpu.log"
}

# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
pcam_pipeline() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  preflight
  smoke_pcam
  banner "PCam data-Pareto: n in {100, 500, 2500} x 6 models x 5 seeds (90 runs)"
  SEEDS="$SEEDS5" SIZES="100 500 2500" RUN_SO2_DISK=0 bash "$EXP/pcam/run_data_pareto_sweep.sh" --phase-b
  banner "[pcam] DONE"
}

organ3d_pipeline() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
  preflight
  smoke_organ3d
  banner "OrganMNIST3D data efficiency: n in {50, 100, 250, 500} x 4 models x 10 seeds"
  SEEDS="$SEEDS10" bash "$EXP/organ3d/run_dataeff_multiseed.sh"
  banner "OrganMNIST3D full data: widths (4,8), (8,16), (16,32) x 4 models x 10 seeds"
  SEEDS="$SEEDS10" WIDTHS="4_8 8_16 16_32" bash "$EXP/organ3d/run_wider_multiseed.sh"
  banner "OrganMNIST3D max_pool batch-16 control: n in {100, 500} x 10 seeds"
  SEEDS="$SEEDS10" bash "$EXP/organ3d/run_bs16_control.sh"
  device_timings a100
  banner "[organ3d] DONE"
}

smnist_pipeline() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  preflight
  smoke_smnist
  banner "Spherical MNIST data efficiency: n in {100, 500, 2500, 12500} x 3 models x 5 seeds"
  SEEDS="$SEEDS5" bash "$EXP/spherical_mnist/run_data_efficiency.sh"
  banner "Spherical MNIST Table 4 sweep: 3 models x {C, R} x 5 seeds, full data"
  SEEDS="$SEEDS5" bash "$EXP/spherical_mnist/run_sweep.sh"
  banner "Spherical MNIST capacity sweep: 10 widths x 5 seeds"
  SEEDS="$SEEDS5" bash "$EXP/spherical_mnist/run_capacity_sweep.sh"
  banner "Spherical MNIST feature ablation: 4 feature sets x 5 seeds"
  SEEDS="$SEEDS5" bash "$REPO_ROOT/rebuttal/run_smnist_ablation.sh"
  banner "Reconstruction evidence (smnist_l15, random_l15)"
  bash "$REPO_ROOT/rebuttal/run_reconstruction_evidence.sh"
  device_timings h100
  banner "[smnist] DONE"
}

# ---------------------------------------------------------------------------
# Analysis: grid figure, significance table, combined device timings.
# ---------------------------------------------------------------------------
analyze() {
  cd "$REPO_ROOT"
  banner "grid figure"
  "$PYTHON" experiments/make_grid_figure.py
  banner "significance table"
  "$PYTHON" experiments/significance.py --output "$RESULTS/significance.md"
  banner "ablation summary"
  "$PYTHON" rebuttal/analyze_smnist_ablation.py --results-dir "$RESULTS/smnist_ablation" \
    --output "$RESULTS/smnist_ablation/summary.json"
  banner "device timings"
  local jsons=()
  for f in "$RESULTS"/device_bench/*_cuda.json "$RESULTS"/device_bench/*_host_cpu.json; do
    [[ -f "$f" ]] && jsons+=("$f")
  done
  if (( ${#jsons[@]} )); then
    "$PYTHON" rebuttal/analyze_device_benchmarks.py "${jsons[@]}" \
      --output "$RESULTS/device_bench/combined.json" | tee "$RESULTS/device_bench/combined.md"
  else
    echo "no device_bench/*.json yet"
  fi
}

record_environment() {
  mkdir -p "$ENV_DIR" "$LOG_DIR"
  cd "$REPO_ROOT"
  git rev-parse HEAD | tee "$ENV_DIR/git_sha.txt"
  git status --short | tee "$ENV_DIR/git_status.txt"
  "$PYTHON" - <<'PY' | tee "$ENV_DIR/python.txt"
import platform

import torch

import bispectrum

print("Python:", platform.python_version())
print("Platform:", platform.platform())
print("bispectrum:", bispectrum.__version__)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:", torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
  nvidia-smi | tee "$ENV_DIR/nvidia-smi.txt"
}

launch() {
  local machine=$1
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION" >&2
    exit 1
  fi
  record_environment
  preflight
  if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
    banner "test suite (SKIP_TESTS=1 to skip)"
    "$PYTHON" -m pytest -q tests/ 2>&1 | tee "$LOG_DIR/pytest_$(hostname -s).log"
  fi

  case "$machine" in
    bongo)
      tmux new-session -d -s "$SESSION" -n pcam
      tmux send-keys -t "$SESSION:pcam" \
        "CUDA_VISIBLE_DEVICES=0 bash '$SCRIPT' pcam 2>&1 | tee -a '$LOG_DIR/pcam_gpu0.log'" C-m
      tmux new-window -t "$SESSION" -n organ3d
      tmux send-keys -t "$SESSION:organ3d" \
        "CUDA_VISIBLE_DEVICES=1 bash '$SCRIPT' organ3d 2>&1 | tee -a '$LOG_DIR/organ3d_gpu1.log'" C-m
      ;;
    h100)
      tmux new-session -d -s "$SESSION" -n smnist
      tmux send-keys -t "$SESSION:smnist" \
        "CUDA_VISIBLE_DEVICES=0 bash '$SCRIPT' smnist 2>&1 | tee -a '$LOG_DIR/smnist_gpu0.log'" C-m
      ;;
    *)
      echo "usage: $0 launch <bongo|h100>" >&2
      exit 1
      ;;
  esac
  echo ""
  echo "Launched '$SESSION'. Monitor with:"
  echo "  tmux attach -t $SESSION        (Ctrl-b n to switch windows)"
  echo "  tail -f $LOG_DIR/*.log"
}

case "${1:-}" in
  archive)
    shift
    cd "$REPO_ROOT" && "$PYTHON" rebuttal/archive_stale_results.py "$@"
    ;;
  smoke)
    shift
    preflight
    for target in "$@"; do "smoke_$target"; done
    ;;
  pcam) pcam_pipeline ;;
  organ3d) organ3d_pipeline ;;
  smnist) smnist_pipeline ;;
  launch) launch "${2:-}" ;;
  analyze) analyze ;;
  *)
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 1
    ;;
esac
