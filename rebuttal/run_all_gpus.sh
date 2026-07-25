#!/usr/bin/env bash
# Launch all rebuttal experiments (E1-E4) across both GPUs in one tmux session.
#
#   bash rebuttal/run_all_gpus.sh            # env record + pytest, then launch tmux
#   SKIP_TESTS=1 bash rebuttal/run_all_gpus.sh
#
# GPU 0: E1 device benchmarks -> E2 reconstruction evidence
# GPU 1: E3 sMNIST feature ablation -> E4 organ3d regularization
#
# All sweep runners are restart-safe (existing results.json are skipped), so
# re-running this script resumes where it left off.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_ROOT/rebuttal/run_all_gpus.sh"
export PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
SESSION="${SESSION:-rebuttal}"
RESULTS="$REPO_ROOT/rebuttal/results"
LOG_DIR="$RESULTS/logs"

gpu0_pipeline() {
  export CUDA_VISIBLE_DEVICES=0
  cd "$REPO_ROOT"
  mkdir -p "$RESULTS/device_bench"

  echo "=== [GPU0] E1: device benchmarks ==="
  "$PYTHON" rebuttal/benchmark_devices.py \
    --device cuda \
    --output "$RESULTS/device_bench/a100_cuda.json" \
    2>&1 | tee "$RESULTS/device_bench/a100_cuda.log"
  "$PYTHON" rebuttal/benchmark_devices.py \
    --device cpu \
    --output "$RESULTS/device_bench/a100_host_cpu.json" \
    2>&1 | tee "$RESULTS/device_bench/a100_host_cpu.log"
  "$PYTHON" rebuttal/analyze_device_benchmarks.py \
    "$RESULTS/device_bench/a100_cuda.json" \
    "$RESULTS/device_bench/a100_host_cpu.json" \
    --output "$RESULTS/device_bench/combined.json" \
    | tee "$RESULTS/device_bench/combined.md"

  echo "=== [GPU0] E2: reconstruction smoke test ==="
  "$PYTHON" experiments/spherical_mnist_reconstruction/reconstruct.py \
    --signal_source random --n_digits 1 --n_rotations 0 \
    --lmax 15 --nlat 128 --nlon 256 --n_steps 20 \
    --n_recon_restarts 1 --align_n_restarts 0 \
    --device cuda --skip_figures \
    --output_dir "$RESULTS/reconstruction_smoke"

  echo "=== [GPU0] E2: full reconstruction evidence ==="
  bash rebuttal/run_reconstruction_evidence.sh

  echo "=== [GPU0] DONE ==="
}

gpu1_pipeline() {
  export CUDA_VISIBLE_DEVICES=1
  cd "$REPO_ROOT"

  echo "=== [GPU1] E3: dimension dry runs ==="
  for features in bootstrap bootstrap_self bootstrap_self_cg full; do
    "$PYTHON" experiments/spherical_mnist/train.py \
      --model bispectrum \
      --bispectrum_features "$features" \
      --run_label "bispectrum_${features}" \
      --dry_run
  done

  echo "=== [GPU1] E3: sMNIST ablation sweep (12 runs) ==="
  bash rebuttal/run_smnist_ablation.sh

  echo "=== [GPU1] E4: organ3d memory smoke tests ==="
  "$PYTHON" experiments/organ3d/train.py \
    --model max_pool --channels 16 32 --batch_size 8 \
    --data_dir experiments/organ3d/organ3d_data --memory_check
  "$PYTHON" experiments/organ3d/train.py \
    --model bispectrum --channels 16 32 --batch_size 4 \
    --data_dir experiments/organ3d/organ3d_data --memory_check

  echo "=== [GPU1] E4: organ3d regularization sweep (21 runs) ==="
  bash rebuttal/run_organ3d_regularization.sh

  echo "=== [GPU1] DONE ==="
}

case "${1:-launch}" in
  gpu0) gpu0_pipeline ;;
  gpu1) gpu1_pipeline ;;
  launch)
    mkdir -p "$LOG_DIR" "$RESULTS/environment"
    cd "$REPO_ROOT"

    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION" >&2
      exit 1
    fi

    echo "=== Step 0: environment record ==="
    git rev-parse HEAD | tee "$RESULTS/environment/git_sha.txt"
    "$PYTHON" - <<'PY' | tee "$RESULTS/environment/python.txt"
import platform
import torch

print("Python:", platform.python_version())
print("Platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:", torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
    nvidia-smi | tee "$RESULTS/environment/nvidia-smi.txt"

    if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
      echo "=== Step 0: test suite (SKIP_TESTS=1 to skip) ==="
      "$PYTHON" -m pytest -q 2>&1 | tee "$LOG_DIR/pytest.log"
    fi

    echo "=== Launching tmux session '$SESSION' ==="
    tmux new-session -d -s "$SESSION" -n gpu0
    tmux send-keys -t "$SESSION:gpu0" \
      "bash '$SCRIPT' gpu0 2>&1 | tee -a '$LOG_DIR/gpu0_e1_e2.log'" C-m
    tmux new-window -t "$SESSION" -n gpu1
    tmux send-keys -t "$SESSION:gpu1" \
      "bash '$SCRIPT' gpu1 2>&1 | tee -a '$LOG_DIR/gpu1_e3_e4.log'" C-m

    echo
    echo "Launched. Monitor with:"
    echo "  tmux attach -t $SESSION        (Ctrl-b n to switch GPU windows)"
    echo "  tail -f $LOG_DIR/gpu0_e1_e2.log"
    echo "  tail -f $LOG_DIR/gpu1_e3_e4.log"
    ;;
  *)
    echo "usage: $0 [launch|gpu0|gpu1]" >&2
    exit 1
    ;;
esac
