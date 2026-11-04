#!/usr/bin/env bash
# Spherical MNIST e3nn baselines for the camera-ready (E5 tensor product, E6 matched S2CNN).
#
#   bash rebuttal/run_smnist_baselines.sh tmux <session> <subcommand>...
#   bash rebuttal/run_smnist_baselines.sh train     # l8a100: wait for the camera-ready runner,
#                                                   # tune, select, final seeds
#   bash rebuttal/run_smnist_baselines.sh bench     # h100: wait for an idle GPU, time all models
#   bash rebuttal/run_smnist_baselines.sh collect   # h100: rsync results from l8a100
#   bash rebuttal/run_smnist_baselines.sh analyze   # h100: Table 4 rows with the baselines
set -euo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
SMNIST="$REPO_ROOT/experiments/spherical_mnist"
OUT="$REPO_ROOT/rebuttal/results/smnist_baselines"
LOG_DIR="$OUT/logs"
CACHE="$SMNIST/smnist_data/spherical_cache"
CAMERA_READY_RESULTS="${CAMERA_READY_RESULTS:-$HOME/bispectrum/experiments/spherical_mnist/smnist_results}"
SEEDS=(42 123 456 789 101112)
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" PYTHONUNBUFFERED=1

banner() { echo "=== $* — $(date -u +%FT%TZ)"; }

wait_for_camera_ready() {
  while pgrep -f 'gpu_job_runner.py --manifest .*camera_ready_jobs/manifest.json' >/dev/null; do
    sleep 300
  done
  banner "camera-ready runner gone on $(hostname -s)"
}

check_cache() {
  local missing=0 f
  for f in train_64x128 val_64x128 test_64x128 test_64x128_rotated_s777; do
    [[ -f "$CACHE/$f.pt" ]] || { echo "missing $CACHE/$f.pt" >&2; missing=1; }
  done
  for s in "${SEEDS[@]}"; do
    for f in train val; do
      [[ -f "$CACHE/${f}_64x128_rotated_s$s.pt" ]] || { echo "missing ${f}_64x128_rotated_s$s.pt" >&2; missing=1; }
    done
  done
  return "$missing"
}

run_stage() {
  local stage=$1
  banner "stage $stage"
  "$PYTHON" "$REPO_ROOT/rebuttal/gpu_job_runner.py" --manifest "$OUT/$stage.json" --machine l8a100 \
    --state_dir "$OUT/state/l8a100"
}

train() {
  check_cache
  "$PYTHON" -c 'import e3nn' || { echo 'e3nn missing: uv pip install e3nn' >&2; exit 1; }
  wait_for_camera_ready
  "$PYTHON" "$REPO_ROOT/rebuttal/smnist_baselines_manifest.py" tune --out "$OUT/tune.json"
  run_stage tune
  "$PYTHON" "$REPO_ROOT/rebuttal/smnist_baselines_manifest.py" final --out "$OUT/final.json"
  run_stage final
  banner "train DONE"
}

bench() {
  wait_for_camera_ready
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
    sleep 60
  done
  banner "GPU idle on $(hostname -s); timing"
  "$PYTHON" "$REPO_ROOT/rebuttal/benchmark_smnist_models.py" --tp_grid \
    --output "$OUT/timing_$(hostname -s).json"
  banner "bench DONE"
}

collect() {
  local host=${COLLECT_HOST:-l8a100} remote=${REMOTE_ROOT:-bispectrum-crb}
  for d in smnist_results_baselines smnist_results_baselines_tuning; do
    rsync -a "$host:$remote/experiments/spherical_mnist/$d/" "$SMNIST/$d/"
  done
  rsync -a --exclude logs/ "$host:$remote/rebuttal/results/smnist_baselines/" "$OUT/"
  banner "collected from $host"
}

analyze() {
  (cd "$SMNIST" && "$PYTHON" analyze_results.py \
    --results_dir "$CAMERA_READY_RESULTS" ./smnist_results_baselines \
    --output "$OUT/smnist_analysis_baselines.png") | tee "$OUT/table4_with_baselines.txt"
}

case "${1:-}" in
  tmux)
    session=${2:?usage: $0 tmux <session> <subcommand>...}
    shift 2
    if tmux has-session -t "=$session" 2>/dev/null; then
      echo "tmux session '$session' already exists" >&2
      exit 1
    fi
    mkdir -p "$LOG_DIR"
    tmux new-session -d -s "$session" -c "$REPO_ROOT" \
      "bash $(printf '%q ' "$SCRIPT" "$@") 2>&1 | tee -a $(printf '%q' "$LOG_DIR/$session.log"); echo \"=== $session exited \${PIPESTATUS[0]}\" | tee -a $(printf '%q' "$LOG_DIR/$session.log"); sleep 86400"
    echo "tmux session '$session' on $(hostname -s): $*  (log: $LOG_DIR/$session.log)"
    ;;
  train | bench | collect | analyze) "$1" ;;
  *)
    sed -n '2,9p' "$SCRIPT"
    exit 1
    ;;
esac
