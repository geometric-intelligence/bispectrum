#!/usr/bin/env bash
# Spherical MNIST: give the bispectrum the baselines' optimizer grid (camera-ready fairness).
#
#   bash rebuttal/run_smnist_bispectrum_grid.sh tmux <session>   # detached, survives logout
#   bash rebuttal/run_smnist_bispectrum_grid.sh run              # tune -> select -> final seeds
#
# Grid = union of the S2CNN+BN and tensor-product optimizer values: batch {32, 256} x
# lr {1e-3, 3e-3, 5e-3}, one C-trained seed (42), selected on best_val_accuracy exactly like
# rebuttal/smnist_baselines_manifest.py; the winner then runs at every seed and train mode.
# Restartable: any run whose results.json exists is skipped.
set -uo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
SMNIST="$REPO_ROOT/experiments/spherical_mnist"
OUT="$REPO_ROOT/rebuttal/results/smnist_bispectrum_grid"
LOG_DIR="$OUT/logs"
TUNE_DIR=./smnist_results_bispectrum_tuning
FINAL_DIR=./smnist_results_bispectrum_tuned
SEEDS=(42 123 456 789 101112)
MODES=(C R)
BATCHES=(32 256)
LRS=(1e-3 3e-3 5e-3)
COMMON=(--model bispectrum --epochs 30 --data_dir ./smnist_data)
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4

banner() { echo "=== $* — $(date -u +%FT%TZ)"; }

run_one() {
  local out_dir=$1 label=$2 mode=$3 seed=$4 bs=$5 lr=$6
  local res="$SMNIST/$out_dir/${label}_${mode}_seed${seed}/results.json"
  if [[ -f "$res" ]]; then
    echo "skip (done): ${label}_${mode}_seed${seed}"
    return 0
  fi
  local log="$LOG_DIR/${label}_${mode}_seed${seed}.log"
  echo "start: ${label}_${mode}_seed${seed}"
  (cd "$SMNIST" && "$PYTHON" train.py "${COMMON[@]}" --run_label "$label" --train_mode "$mode" \
    --seed "$seed" --batch_size "$bs" --lr "$lr" --output_dir "$out_dir") >"$log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 && -f "$res" ]]; then
    echo "done: ${label}_${mode}_seed${seed}"
  else
    echo "FAILED (exit $rc): ${label}_${mode}_seed${seed} (log: $log)"
  fi
  return $rc
}

tune() {
  banner "stage tune"
  local pids=() bs lr
  for bs in "${BATCHES[@]}"; do
    for lr in "${LRS[@]}"; do
      run_one "$TUNE_DIR" "bispectrum_bs${bs}_lr${lr}" C 42 "$bs" "$lr" &
      pids+=($!)
    done
  done
  local failed=0 p
  for p in "${pids[@]}"; do wait "$p" || failed=1; done
  return $failed
}

select_config() {
  banner "stage select"
  "$PYTHON" - "$SMNIST/$TUNE_DIR" "$OUT/selection.json" <<'PY'
import json, sys
from pathlib import Path

root, out = Path(sys.argv[1]), Path(sys.argv[2])
scores = {}
for bs in (32, 256):
    for lr in ('1e-3', '3e-3', '5e-3'):
        label = f'bispectrum_bs{bs}_lr{lr}'
        path = root / f'{label}_C_seed42' / 'results.json'
        if not path.exists():
            sys.exit(f'missing tuning result: {path}')
        scores[label] = float(json.loads(path.read_text())['best_val_accuracy'])
best = max(scores, key=scores.__getitem__)
bs, lr = best.removeprefix('bispectrum_bs').split('_lr')
out.write_text(json.dumps({'selected': best, 'batch_size': int(bs), 'lr': lr,
                           'val_accuracy': scores}, indent=2) + '\n')
for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
    print(f'  {k}: val {v:.4f}{"  <- selected" if k == best else ""}')
PY
}

final() {
  banner "stage final"
  local sel bs lr label
  sel=$("$PYTHON" -c "import json; d=json.load(open('$OUT/selection.json')); print(d['selected'], d['batch_size'], d['lr'])")
  read -r label bs lr <<<"$sel"
  local pids=() mode seed
  for mode in "${MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_one "$FINAL_DIR" "$label" "$mode" "$seed" "$bs" "$lr" &
      pids+=($!)
    done
  done
  local failed=0 p
  for p in "${pids[@]}"; do wait "$p" || failed=1; done
  return $failed
}

run() {
  mkdir -p "$LOG_DIR"
  banner "bispectrum grid on $(hostname -s), git $(git -C "$REPO_ROOT" rev-parse --short HEAD)"
  tune || { banner "tune had failures; rerun to retry"; exit 1; }
  select_config || exit 1
  final || { banner "final had failures; rerun to retry"; exit 1; }
  banner "bispectrum grid DONE"
}

case "${1:-}" in
  tmux)
    session=${2:?usage: $0 tmux <session>}
    if tmux has-session -t "=$session" 2>/dev/null; then
      echo "tmux session '$session' already exists" >&2
      exit 1
    fi
    mkdir -p "$LOG_DIR"
    tmux new-session -d -s "$session" -c "$REPO_ROOT" \
      "bash $(printf '%q' "$SCRIPT") run 2>&1 | tee -a $(printf '%q' "$LOG_DIR/$session.log"); echo \"=== $session exited \${PIPESTATUS[0]}\" | tee -a $(printf '%q' "$LOG_DIR/$session.log"); sleep 86400"
    echo "tmux session '$session' on $(hostname -s) (log: $LOG_DIR/$session.log)"
    ;;
  run) run ;;
  *)
    sed -n '2,10p' "$SCRIPT"
    exit 1
    ;;
esac
