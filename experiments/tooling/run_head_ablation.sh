#!/usr/bin/env bash
# Bispectral pooling head ablation (experiments/tooling/run_head_ablation.py) under tmux.
#
#   bash experiments/tooling/run_head_ablation.sh launch    # tmux session "head-ablation": smoke, run, analyze
#   bash experiments/tooling/run_head_ablation.sh smoke     # gates only: tests, manifest, two 1-epoch runs
#   bash experiments/tooling/run_head_ablation.sh run       # the 90 jobs on $GPUS x $WORKERS_PER_GPU slots
#   bash experiments/tooling/run_head_ablation.sh analyze   # summary.md / summary.json
#
# Environment: GPUS (default 0-7), WORKERS_PER_GPU (2), OMP_NUM_THREADS (8), PYTHON
# (.venv/bin/python of this checkout), SESSION (head-ablation).
#
# `launch` refuses to start if the session exists, records the git commit, working-tree
# status, pip freeze and nvidia-smi under experiments/tooling/results/head_ablation/env/<host>/ and
# opens two windows: `pipeline` (smoke && run && analyze, logged to logs/pipeline.log)
# and `monitor` (watch nvidia-smi). Everything long-running happens inside the session,
# so it survives the SSH connection. `pipeline` is that window's command.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/experiments/tooling/run_head_ablation.sh"
export PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
SESSION="${SESSION:-head-ablation}"
GPUS="${GPUS:-0-7}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

RESULTS="$REPO_ROOT/experiments/tooling/results/head_ablation"
LOG_DIR="$RESULTS/logs"
ENV_DIR="$RESULTS/env/$(hostname -s)"
SMOKE_DIR="${SMOKE_DIR:-${TMPDIR:-/tmp}/head_ablation_smoke}"
EXP="$REPO_ROOT/experiments"

banner() {
  echo ""
  echo "=== $* — $(date) ==="
}

# A smoke results.json must come from this head code (bispectrum_head recorded), the
# fixed-epoch protocol and the per-epoch sample floor.
check_results() {
  local results_path=$1 expected_epoch_samples=$2 expected_head=$3
  "$PYTHON" - "$results_path" "$expected_epoch_samples" "$expected_head" <<'PY'
import json
import sys
from pathlib import Path

path, expected, head = Path(sys.argv[1]), int(sys.argv[2]), json.loads(sys.argv[3])
run = json.loads(path.read_text())
problems = []
if run.get('epoch_samples') != expected:
    problems.append(f"epoch_samples={run.get('epoch_samples')} != {expected}")
if run.get('early_stopping_patience') != 0:
    problems.append(f"early_stopping_patience={run.get('early_stopping_patience')} != 0")
if run.get('epochs_run') != run.get('epochs'):
    problems.append(f"epochs_run={run.get('epochs_run')} != epochs={run.get('epochs')}")
if run.get('bispectrum_head') != head:
    problems.append(f"bispectrum_head={run.get('bispectrum_head')} != {head}")
mem = run.get('peak_train_reserved_bytes') or 0
print(f'{path}: epoch_samples={run.get("epoch_samples")} head={run.get("bispectrum_head")} '
      f'best_val_auc={run.get("best_val_auc", 0):.3f} peak_mem={mem / 2**30:.1f} GiB '
      f'wall={run.get("wall_time_s", 0):.0f}s')
if problems:
    sys.exit('smoke gate failed: ' + '; '.join(problems))
PY
}

smoke() {
  banner "smoke on $(hostname -s): tests"
  (cd "$REPO_ROOT" && "$PYTHON" -m pytest -q tests/test_experiment_bispectrum_head.py tests/test_experiment_protocol.py)

  banner "smoke: job manifest"
  local n_jobs
  n_jobs=$("$PYTHON" "$REPO_ROOT/experiments/tooling/run_head_ablation.py" --dry_run | "$PYTHON" -c 'import json,sys; print(len(json.load(sys.stdin)))')
  echo "manifest: $n_jobs jobs"
  [[ "$n_jobs" == 90 ]] || { echo "expected 90 jobs"; exit 1; }

  banner "smoke: PCam bispectrum (default head) n=100, 1 epoch"
  rm -rf "$SMOKE_DIR"
  (cd "$EXP/pcam" && CUDA_VISIBLE_DEVICES="${GPUS%%[-,]*}" "$PYTHON" train.py --model bispectrum --growth_rate 4 \
    --train_mode C --train_size 100 --epochs 1 --batch_size 64 --skip_rotation \
    --output_dir "$SMOKE_DIR/pcam" --seed 42)
  check_results "$(find "$SMOKE_DIR/pcam" -name results.json)" 10000 \
    '{"nonlin": "post", "power": false, "scale": "log"}'

  banner "smoke: OrganMNIST3D bispectrum (power + cbrt) n=50, 1 epoch"
  (cd "$EXP/organ3d" && CUDA_VISIBLE_DEVICES="${GPUS%%[-,]*}" "$PYTHON" train.py --model bispectrum --channels 4 8 \
    --bispectrum_power --bispectrum_scale cbrt --train_mode C --train_size 50 --epochs 1 \
    --batch_size 16 --skip_rotation --data_dir ./organ3d_data --output_dir "$SMOKE_DIR/organ3d" --seed 42)
  check_results "$(find "$SMOKE_DIR/organ3d" -name results.json)" 971 \
    '{"nonlin": "post", "power": true, "scale": "cbrt"}'
  banner "smoke: PASSED"
}

run() {
  banner "head ablation: 90 jobs on GPUs $GPUS x $WORKERS_PER_GPU workers"
  "$PYTHON" "$REPO_ROOT/experiments/tooling/run_head_ablation.py" --gpus "$GPUS" --workers_per_gpu "$WORKERS_PER_GPU" \
    --omp_threads "$OMP_NUM_THREADS"
}

analyze() {
  banner "analysis"
  "$PYTHON" "$REPO_ROOT/experiments/tooling/analyze_head_ablation.py" --results_root "$RESULTS"
}

# smoke && run && analyze, everything logged; what the tmux `pipeline` window runs.
pipeline() {
  mkdir -p "$LOG_DIR"
  local status=0
  { smoke && run && analyze; } 2>&1 | tee -a "$LOG_DIR/pipeline.log" || status=$?
  echo "=== pipeline exited $status — $(date)" | tee -a "$LOG_DIR/pipeline.log"
  return "$status"
}

record_env() {
  mkdir -p "$ENV_DIR"
  (cd "$REPO_ROOT" && git rev-parse HEAD > "$ENV_DIR/git_commit.txt" && git status --short > "$ENV_DIR/git_status.txt")
  (cd "$REPO_ROOT" && git diff > "$ENV_DIR/git_diff.patch") || true
  "$PYTHON" -m pip freeze > "$ENV_DIR/pip_freeze.txt" 2>/dev/null || "$PYTHON" -m pip list > "$ENV_DIR/pip_freeze.txt" 2>/dev/null || true
  nvidia-smi > "$ENV_DIR/nvidia_smi.txt" 2>/dev/null || true
  {
    echo "host: $(hostname)"; echo "date: $(date -u +%FT%TZ)"; echo "python: $PYTHON"
    echo "gpus: $GPUS x $WORKERS_PER_GPU workers"; echo "omp_threads: $OMP_NUM_THREADS"; nproc
  } > "$ENV_DIR/launch.txt"
}

launch() {
  if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION"
    exit 1
  fi
  [[ -x "$PYTHON" ]] || { echo "no python at $PYTHON"; exit 1; }
  mkdir -p "$LOG_DIR"
  record_env
  local env_prefix="GPUS=$GPUS WORKERS_PER_GPU=$WORKERS_PER_GPU OMP_NUM_THREADS=$OMP_NUM_THREADS PYTHON=$PYTHON"
  tmux new-session -d -s "$SESSION" -n pipeline -c "$REPO_ROOT" \
    "env $env_prefix bash '$SCRIPT' pipeline; sleep 86400"
  tmux new-window -t "$SESSION" -n monitor "watch -n 60 nvidia-smi"
  echo "launched tmux session '$SESSION' on $(hostname -s): commit $(cat "$ENV_DIR/git_commit.txt")"
  echo "  logs:    $LOG_DIR/pipeline.log, $LOG_DIR/runner.log"
  echo "  attach:  tmux attach -t $SESSION"
}

case "${1:-}" in
  smoke) smoke ;;
  run) run ;;
  analyze) analyze ;;
  pipeline) pipeline ;;
  launch) launch ;;
  *)
    sed -n '2,16p' "$SCRIPT"
    exit 1
    ;;
esac
