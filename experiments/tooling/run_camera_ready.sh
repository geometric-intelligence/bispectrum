#!/usr/bin/env bash
# Camera-ready reruns on bispectrum 0.4.0: per-epoch sample floor, a fixed epoch
# budget that ends at an SGDR cycle end (no early stopping), one batch size per
# dataset (see tests/test_experiment_protocol.py).
#
#   bash experiments/tooling/run_camera_ready.sh archive [--dry_run]  # move stale results aside
#   bash experiments/tooling/run_camera_ready.sh smoke <pcam|organ3d|smnist>...
#   bash experiments/tooling/run_camera_ready.sh launch <bongo|h100|custom>  # one tmux window per GPU
#   bash experiments/tooling/run_camera_ready.sh <pipeline>           # one pipeline, foreground
#   bash experiments/tooling/run_camera_ready.sh analyze              # grid, significance, paper figures -> results/camera_ready
#   bash experiments/tooling/run_camera_ready.sh tmux <session> <subcommand>...  # any of these in tmux
#
# Job runner (PCam + OrganMNIST3D across machines; Spherical MNIST stays a pipeline):
#   bash experiments/tooling/run_camera_ready.sh capture <dir> <pipeline>...  # train.py calls -> <dir>/*.tsv
#   bash experiments/tooling/run_camera_ready.sh manifest build|memory|costs|assign|validate ...  # -> $MANIFEST
#   bash experiments/tooling/run_camera_ready.sh launch_jobs <l8a100|bongo|h100>  # tmux camera-ready-runs
#   bash experiments/tooling/run_camera_ready.sh collect              # rsync + validate, on the H100
#
# Pipelines (every sweep skips runs whose results.json exists, so re-running
# resumes):
#   smnist        Spherical MNIST data efficiency, Table 4 sweep, capacity,
#                 feature ablation x 5 seeds
#   organ3d       OrganMNIST3D data efficiency n in {50..500} + widths x 10 seeds,
#                 $ORGAN3D_STREAMS concurrent streams on one GPU
#   pcam          PCam data-Pareto n in {100, 500, 2500} x 6 models x 5 seeds
#   pcam_12500    PCam data-Pareto n = 12500 x 7 models x 3 seeds
#   pcam_full     PCam data-Pareto full data x 7 models x 3 seeds
#   pcam_matched  PCam growth-rate sweep at n = 12500, seeds $MATCHED_SEEDS
#   reconstruction, timings <tag>   no training; kept from the 0.4.0 run
#
# launch runs the pipelines listed in GPU<i>_PIPELINES sequentially in one tmux
# window per GPU, with the extra environment in GPU<i>_ENV (e.g. "SEEDS3=123
# MODELS=fourier_elu RUN_SO2_DISK=0") to shard a pipeline across GPUs. Shards
# must not share a (seed, model) pair: their skip checks would race. Defaults:
# bongo GPU0 "pcam pcam_12500", GPU1 "organ3d pcam_full" (A100 timings first,
# on the idle machine); h100 GPU0 "smnist pcam_matched"; custom has none.
#
# Environment: SEEDS3 / SEEDS5 / SEEDS10 / MATCHED_SEEDS override the seed sets;
# PCAM_SIZES / ORGAN3D_WIDTHS narrow a pipeline; MODELS_FILTER passes through to
# run_matched_sweep.sh; BENCH_THREADS lists the pinned CPU thread counts for
# device timings; SKIP_TESTS=1 skips pytest at launch; SESSION names the tmux session.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/experiments/tooling/run_camera_ready.sh"
export PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
SESSION="${SESSION:-camera-ready}"
RESULTS="$REPO_ROOT/experiments/tooling/results"
LOG_DIR="$RESULTS/logs_camera_ready"
ENV_DIR="$RESULTS/environment_camera_ready/$(hostname -s)"
JOBS_DIR="$RESULTS/camera_ready_jobs"
export MANIFEST="${MANIFEST:-$JOBS_DIR/manifest.json}"
RUNS_SESSION="${RUNS_SESSION:-camera-ready-runs}"
SMOKE_DIR="${SMOKE_DIR:-${TMPDIR:-/tmp}/camera_ready_smoke}"

SEEDS3="${SEEDS3:-42 123 456}"
SEEDS5="${SEEDS5:-42 123 456 789 101112}"
SEEDS10="${SEEDS10:-42 123 456 789 101112 1 2 3 4 5}"
PCAM_SIZES="${PCAM_SIZES:-100 500 2500}"
ORGAN3D_WIDTHS="${ORGAN3D_WIDTHS:-4_8 8_16 16_32}"
ORGAN3D_STREAMS="${ORGAN3D_STREAMS:-3}"
MATCHED_SEEDS="${MATCHED_SEEDS:-$SEEDS3}"
BENCH_THREADS="${BENCH_THREADS:-1 8}"
SMOKE_MIN_VAL_ACC="${SMOKE_MIN_VAL_ACC:-0.2}"
# One BLAS/OpenMP pool per CPU core oversubscribes the host once several runs
# share it; 8 threads per run is faster on bongo even for a single run.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

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
  [[ -z "${CAPTURE_JOBS:-}" ]] || return 0
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

# results.json written by a smoke run must show the floor, the library version
# and the fixed-epoch protocol.
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
if run.get('early_stopping_patience') != 0:
    problems.append(f"early_stopping_patience={run.get('early_stopping_patience')} != 0")
if run.get('epochs_run') != run.get('epochs'):
    problems.append(f"epochs_run={run.get('epochs_run')} != epochs={run.get('epochs')}")
head = run.get('bispectrum_head')
if run.get('dataset') in ('pcam', 'organ3d') and run.get('model') == 'bispectrum' and (
    not head or head.get('nonlin') != 'post'
):
    problems.append(f'bispectrum_head={head}: expected the fixed head (nonlin=post)')
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
  [[ -z "${CAPTURE_JOBS:-}" ]] || return 0
  banner "smoke: PCam bispectrum n=100, 1 epoch"
  rm -rf "$SMOKE_DIR/pcam"
  (cd "$EXP/pcam" && "$PYTHON" train.py --model bispectrum --growth_rate 4 --train_mode C \
    --train_size 100 --epochs 1 --skip_rotation \
    --output_dir "$SMOKE_DIR/pcam" --seed 42)
  check_results "$(find "$SMOKE_DIR/pcam" -name results.json)" 10000
}

smoke_organ3d() {
  [[ -z "${CAPTURE_JOBS:-}" ]] || return 0
  banner "smoke: OrganMNIST3D bispectrum n=50, 1 epoch"
  rm -rf "$SMOKE_DIR/organ3d"
  (cd "$EXP/organ3d" && "$PYTHON" train.py --model bispectrum --channels 4 8 --train_mode C \
    --train_size 50 --epochs 1 --skip_rotation \
    --data_dir ./organ3d_data --output_dir "$SMOKE_DIR/organ3d" --seed 42)
  check_results "$(find "$SMOKE_DIR/organ3d" -name results.json)" 971
}

smoke_smnist() {
  [[ -z "${CAPTURE_JOBS:-}" ]] || return 0
  banner "smoke: Spherical MNIST dry runs (expect 760 invariant dims for the full model)"
  for features in bootstrap bootstrap_self bootstrap_self_cg full; do
    (cd "$EXP/spherical_mnist" && "$PYTHON" train.py --model bispectrum \
      --bispectrum_features "$features" --run_label "bispectrum_${features}" --dry_run)
  done
  banner "smoke: Spherical MNIST bispectrum n=100, 10 epochs (was 9.6% val acc before the floor)"
  rm -rf "$SMOKE_DIR/smnist"
  (cd "$EXP/spherical_mnist" && "$PYTHON" train.py --model bispectrum --train_mode C \
    --train_size 100 --epochs 10 --skip_rotation \
    --data_dir "$SMNIST_DATA" --output_dir "$SMOKE_DIR/smnist" --seed 42)
  check_results "$(find "$SMOKE_DIR/smnist" -name results.json)" 10000 "$SMOKE_MIN_VAL_ACC"
}

# ---------------------------------------------------------------------------
# Device timings (same protocol as the first E1 run, new library version).
# ---------------------------------------------------------------------------
device_timings() {
  local tag=$1 threads
  banner "device timings: $tag (CPU threads: $BENCH_THREADS)"
  mkdir -p "$RESULTS/device_bench"
  "$PYTHON" "$REPO_ROOT/experiments/tooling/benchmark_devices.py" --device cuda \
    --output "$RESULTS/device_bench/${tag}_cuda.json" 2>&1 | tee "$RESULTS/device_bench/${tag}_cuda.log"
  for threads in $BENCH_THREADS; do
    "$PYTHON" "$REPO_ROOT/experiments/tooling/benchmark_devices.py" --device cpu --threads "$threads" \
      --output "$RESULTS/device_bench/${tag}_host_cpu_t${threads}.json" 2>&1 |
      tee "$RESULTS/device_bench/${tag}_host_cpu_t${threads}.log"
  done
}

# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
pcam_pipeline() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  preflight
  smoke_pcam
  banner "PCam data-Pareto: n in {$PCAM_SIZES} x 6 models x seeds {$SEEDS5}"
  SEEDS="$SEEDS5" SIZES="$PCAM_SIZES" RUN_SO2_DISK=0 bash "$EXP/pcam/run_data_pareto_sweep.sh" --phase-b
  banner "[pcam] DONE"
}

pcam_size_pipeline() {
  local size=$1
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  preflight
  banner "PCam data-Pareto: n = $size x 7 models x seeds {$SEEDS3}"
  SEEDS="$SEEDS3" SIZES="$size" bash "$EXP/pcam/run_data_pareto_sweep.sh" --phase-b
  banner "[pcam_$size] DONE"
}

pcam_matched_pipeline() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  preflight
  for seed in $MATCHED_SEEDS; do
    banner "PCam growth-rate sweep at n = 12500: seed $seed (models: ${MODELS_FILTER:-all})"
    bash "$EXP/pcam/run_matched_sweep.sh" --seed "$seed"
  done
  banner "[pcam_matched] DONE"
}

organ3d_stream() {
  local seeds=$1
  SEEDS="$seeds" bash "$EXP/organ3d/run_dataeff_multiseed.sh"
  SEEDS="$seeds" WIDTHS="$ORGAN3D_WIDTHS" bash "$EXP/organ3d/run_wider_multiseed.sh"
}

# OrganMNIST3D runs leave the GPU half idle, so seeds are dealt round-robin to
# concurrent streams; streams never share a seed, so their skip checks do not race.
organ3d_pipeline() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
  preflight
  smoke_organ3d
  local seeds groups=() pids=() i failed=0
  read -ra seeds <<<"$SEEDS10"
  for i in "${!seeds[@]}"; do
    groups[i % ORGAN3D_STREAMS]+="${seeds[i]} "
  done
  banner "OrganMNIST3D: n in {50, 100, 250, 500} + widths {$ORGAN3D_WIDTHS} x 4 models, streams: ${groups[*]/%/|}"
  for i in "${!groups[@]}"; do
    organ3d_stream "${groups[i]}" >"$LOG_DIR/organ3d_stream$i.log" 2>&1 &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do
    wait "${pids[i]}" || { echo "stream $i failed: $LOG_DIR/organ3d_stream$i.log" >&2; failed=1; }
  done
  (( failed == 0 )) || exit 1
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
  SEEDS="$SEEDS5" bash "$REPO_ROOT/experiments/tooling/run_smnist_ablation.sh"
  banner "[smnist] DONE"
}

reconstruction_pipeline() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  preflight
  banner "Reconstruction evidence (smnist_l15, random_l15)"
  bash "$REPO_ROOT/experiments/tooling/run_reconstruction_evidence.sh"
}

# ---------------------------------------------------------------------------
# Analysis: grid figure, significance table, combined device timings.
# ---------------------------------------------------------------------------
analyze() {
  local out="$RESULTS/camera_ready"
  cd "$REPO_ROOT"
  mkdir -p "$out"
  banner "grid figure"
  "$PYTHON" experiments/make_grid_figure.py --output_dir "$out/grid_figure"
  banner "significance table"
  "$PYTHON" experiments/significance.py --output "$out/significance.md"
  "$PYTHON" experiments/significance.py --canonical --output "$out/significance_canonical.md"
  banner "ablation summary"
  "$PYTHON" experiments/tooling/analyze_smnist_ablation.py --results-dir "$RESULTS/smnist_ablation" \
    --output "$RESULTS/smnist_ablation/summary.json"
  banner "device timings"
  local jsons=()
  for f in "$RESULTS"/device_bench/*_cuda.json "$RESULTS"/device_bench/*_mps.json \
    "$RESULTS"/device_bench/*_host_cpu_t*.json; do
    [[ -f "$f" ]] && jsons+=("$f")
  done
  if (( ${#jsons[@]} )); then
    "$PYTHON" experiments/tooling/analyze_device_benchmarks.py "${jsons[@]}" \
      --output "$RESULTS/device_bench/combined.json" | tee "$RESULTS/device_bench/combined.md"
  else
    echo "no device_bench/*.json yet"
  fi
  # The PCam scripts write next to themselves; OrganMNIST3D writes data_efficiency.png to the cwd.
  banner "paper figures -> $out"
  "$PYTHON" experiments/pcam/analyze_pareto.py | tee "$out/pcam_pareto.txt"
  mv experiments/pcam/pareto_analysis.png experiments/pcam/pareto_analysis.pdf "$out/"
  "$PYTHON" experiments/pcam/analyze_data_pareto.py | tee "$out/pcam_data_pareto.txt"
  mv experiments/pcam/data_pareto_analysis.png "$out/"
  (cd "$out" && "$PYTHON" "$REPO_ROOT/experiments/organ3d/analyze_results.py" \
    --results_dir "$REPO_ROOT/experiments/organ3d/organ3d_results" \
    --output_plot organ3d_analysis.png) | tee "$out/organ3d.txt"
  "$PYTHON" experiments/spherical_mnist/analyze_results.py \
    --results_dir experiments/spherical_mnist/smnist_results \
    --output "$out/smnist_analysis.png" | tee "$out/smnist.txt"
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
  # "=" forces an exact match; a bare name also matches any session it prefixes.
  if tmux has-session -t "=$SESSION" 2>/dev/null; then
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
      : "${GPU0_PIPELINES:=pcam pcam_12500}" "${GPU1_PIPELINES:=organ3d pcam_full}"
      if [[ ! -f "$RESULTS/device_bench/a100_cuda.json" ]]; then
        device_timings a100
      fi
      ;;
    h100)
      : "${GPU0_PIPELINES:=smnist pcam_matched}"
      ;;
    custom) ;;
    *)
      echo "usage: $0 launch <bongo|h100|custom>" >&2
      exit 1
      ;;
  esac
  local n_gpus i pipelines_var env_var window launched=()
  n_gpus=$(nvidia-smi -L | wc -l)
  for (( i = 0; i < n_gpus; i++ )); do
    pipelines_var="GPU${i}_PIPELINES"
    env_var="GPU${i}_ENV"
    [[ -n "${!pipelines_var:-}" ]] || continue
    window="gpu$i"
    if (( ${#launched[@]} == 0 )); then
      tmux new-session -d -s "$SESSION" -n "$window"
    else
      tmux new-window -t "$SESSION" -n "$window"
    fi
    tmux send-keys -t "$SESSION:$window" \
      "env CUDA_VISIBLE_DEVICES=$i ${!env_var:-} bash '$SCRIPT' chain ${!pipelines_var} 2>&1 | tee -a '$LOG_DIR/gpu$i.log'" C-m
    launched+=("$window: ${!env_var:-} ${!pipelines_var}")
  done
  if (( ${#launched[@]} == 0 )); then
    echo "no GPU<i>_PIPELINES set for the $n_gpus GPUs on this host" >&2
    exit 1
  fi
  echo ""
  echo "Launched '$SESSION'. Monitor with:"
  printf '  %s\n' "${launched[@]}"
  echo "  tmux attach -t $SESSION        (Ctrl-b n to switch windows)"
  echo "  tail -f $LOG_DIR/*.log"
}

# ---------------------------------------------------------------------------
# Job runner: the train.py calls of the PCam / OrganMNIST3D pipelines are
# captured once, assigned across machines (camera_ready_manifest.py) and run
# with memory-aware packing on every GPU of a machine (gpu_job_runner.py).
# ---------------------------------------------------------------------------
CAPTURABLE="pcam pcam_12500 pcam_full pcam_matched organ3d"

capture() {
  local out_dir pipeline
  out_dir="${1:?usage: $0 capture <out_dir> <pipeline>...}"
  shift
  mkdir -p "$out_dir" "$LOG_DIR"
  out_dir=$(cd "$out_dir" && pwd)
  for pipeline in "$@"; do
    if [[ " $CAPTURABLE " != *" $pipeline "* ]]; then
      echo "capture: '$pipeline' is not one of: $CAPTURABLE" >&2
      exit 1
    fi
    : >"$out_dir/$pipeline.tsv"
    CAPTURE_JOBS="$out_dir/$pipeline.tsv" ORGAN3D_STREAMS=1 bash "$SCRIPT" "$pipeline" >/dev/null
    echo "$pipeline: $(wc -l <"$out_dir/$pipeline.tsv") jobs -> $out_dir/$pipeline.tsv"
  done
}

# Runs inside the camera-ready-runs tmux session (see launch_jobs).
machine_jobs() {
  local machine=$1
  record_environment
  preflight
  if [[ "$machine" == h100 ]]; then
    bash "$SCRIPT" smnist
  fi
  smoke_pcam
  smoke_organ3d
  banner "job runner: $machine ($MANIFEST)"
  # shellcheck disable=SC2086  # RUNNER_ARGS is a flag list
  "$PYTHON" "$REPO_ROOT/experiments/tooling/gpu_job_runner.py" --manifest "$MANIFEST" --machine "$machine" \
    --state_dir "$JOBS_DIR/state/$machine" ${RUNNER_ARGS:-}
}

# Variables forwarded into tmux sessions: a running tmux server keeps the
# environment it was started with, not the caller's.
PASS_ENV=(PYTHON MANIFEST SEEDS3 SEEDS5 SEEDS10 PCAM_SIZES ORGAN3D_WIDTHS MATCHED_SEEDS
  MODELS_FILTER SMNIST_DATA_DIR SMOKE_DIR OMP_NUM_THREADS SKIP_TESTS RUNNER_ARGS COLLECT_HOSTS)

env_prefix() {
  local var out="env"
  for var in "${PASS_ENV[@]}"; do
    if [[ -n "${!var+x}" ]]; then
      out+=" $var=$(printf '%q' "${!var}")"
    fi
  done
  echo "$out"
}

# new_session <session> <window> <subcommand>...: run this script's subcommand
# in a detached tmux session; the window stays open after it exits.
new_session() {
  local session=$1 window=$2
  shift 2
  # "=" forces an exact match; a bare name also matches any session it prefixes.
  if tmux has-session -t "=$session" 2>/dev/null; then
    echo "tmux session '$session' already exists on $(hostname -s); attach with: tmux attach -t $session" >&2
    exit 1
  fi
  mkdir -p "$LOG_DIR"
  tmux new-session -d -s "$session" -n "$window" -c "$REPO_ROOT" \
    "$(env_prefix) bash $(printf '%q ' "$SCRIPT" _step "$session" "$@"); sleep 86400"
  echo "tmux session '$session' on $(hostname -s): $*"
  echo "  log: $LOG_DIR/$session.log   attach: tmux attach -t $session"
}

_step() {
  local session=$1 status=0
  shift
  local log="$LOG_DIR/$session.log"
  mkdir -p "$LOG_DIR"
  echo "=== $session started on $(hostname -s) at $(git -C "$REPO_ROOT" rev-parse --short HEAD): $* — $(date)" | tee -a "$log"
  bash "$SCRIPT" "$@" 2>&1 | tee -a "$log" || status=$?
  echo "=== $session exited $status — $(date)" | tee -a "$log"
  return "$status"
}

launch_jobs() {
  local machine=${1:?usage: $0 launch_jobs <l8a100|bongo|h100>} window=jobs
  if [[ ! -f "$MANIFEST" ]]; then
    echo "no manifest at $MANIFEST (experiments/tooling/camera_ready_manifest.py assign)" >&2
    exit 1
  fi
  if [[ "$machine" == h100 ]]; then
    window=smnist-then-jobs
  fi
  new_session "$RUNS_SESSION" "$window" machine_jobs "$machine"
  tmux new-window -t "=$RUNS_SESSION:" -n monitor "watch -n 60 nvidia-smi"
  echo "  runner state: $JOBS_DIR/state/$machine (done/, failed/, logs/, progress.json)"
}

collect() {
  local host root
  cd "$REPO_ROOT"
  for host in ${COLLECT_HOSTS:-l8a100 bongo}; do
    for root in experiments/pcam/pcam_results_data_pareto experiments/pcam/pcam_results_pareto \
      experiments/organ3d/organ3d_results experiments/tooling/results/camera_ready_jobs/state; do
      if ! ssh "$host" test -d "bispectrum/$root"; then
        echo "$host: no $root"
        continue
      fi
      banner "rsync $host:$root"
      mkdir -p "$root"
      rsync -a --ignore-existing --stats "$host:bispectrum/$root/" "$root/" | grep -E '^Number of (regular )?files transferred'
    done
  done
  banner "validate against $MANIFEST"
  "$PYTHON" experiments/tooling/camera_ready_manifest.py validate --manifest "$MANIFEST"
  banner "stale results (expect none)"
  "$PYTHON" experiments/tooling/archive_stale_results.py --dry_run
}

case "${1:-}" in
  archive)
    shift
    cd "$REPO_ROOT" && "$PYTHON" experiments/tooling/archive_stale_results.py "$@"
    ;;
  smoke)
    shift
    preflight
    for target in "$@"; do "smoke_$target"; done
    ;;
  pcam) pcam_pipeline ;;
  pcam_12500) pcam_size_pipeline 12500 ;;
  pcam_full) pcam_size_pipeline full ;;
  pcam_matched) pcam_matched_pipeline ;;
  organ3d) organ3d_pipeline ;;
  smnist) smnist_pipeline ;;
  reconstruction) reconstruction_pipeline ;;
  timings) preflight && device_timings "${2:?usage: $0 timings <tag>}" ;;
  chain)
    shift
    for pipeline in "$@"; do bash "$SCRIPT" "$pipeline"; done
    ;;
  launch) launch "${2:-}" ;;
  capture)
    shift
    capture "$@"
    ;;
  manifest)
    shift
    cd "$REPO_ROOT" && "$PYTHON" experiments/tooling/camera_ready_manifest.py "$@"
    ;;
  machine_jobs) machine_jobs "${2:?usage: $0 machine_jobs <machine>}" ;;
  launch_jobs) launch_jobs "${2:-}" ;;
  collect) collect ;;
  tmux)
    shift
    new_session "${1:?usage: $0 tmux <session> <subcommand>...}" main "${@:2}"
    ;;
  _step)
    shift
    _step "$@"
    ;;
  analyze) analyze ;;
  *)
    awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0" >&2
    exit 1
    ;;
esac
