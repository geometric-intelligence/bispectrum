#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
RECON="$REPO_ROOT/experiments/spherical_mnist_reconstruction/reconstruct.py"
DATA_DIR="${SMNIST_DATA_DIR:-$REPO_ROOT/experiments/spherical_mnist/smnist_data}"
OUTPUT_ROOT="${OUTPUT_DIR:-$REPO_ROOT/rebuttal/results/reconstruction}"
mkdir -p "$OUTPUT_ROOT"

COMMON=(
  --lmax 15
  --nlat 128
  --nlon 256
  --n_steps 8000
  --lr 5e-2
  --n_recon_restarts 4
  --align_n_restarts 12
  --align_n_steps 200
  --align_lr 1e-1
  --device cuda
  --skip_figures
)

if [[ ! -f "$DATA_DIR/spherical_cache/test_128x256.pt" ]]; then
  (
    cd "$REPO_ROOT/experiments/spherical_mnist"
    "$PYTHON" -c \
      "from data import SphericalMNISTDataset; SphericalMNISTDataset('test', '$DATA_DIR', 128, 256)"
  )
fi

if [[ ! -f "$OUTPUT_ROOT/smnist_l15/results.json" ]]; then
  "$PYTHON" "$RECON" \
    "${COMMON[@]}" \
    --signal_source smnist \
    --n_digits 8 \
    --n_rotations 1 \
    --seed 2026 \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_ROOT/smnist_l15" \
    2>&1 | tee "$OUTPUT_ROOT/smnist_l15.log"
fi

if [[ ! -f "$OUTPUT_ROOT/random_l15/results.json" ]]; then
  "$PYTHON" "$RECON" \
    "${COMMON[@]}" \
    --signal_source random \
    --n_digits 16 \
    --n_rotations 0 \
    --seed 2026 \
    --output_dir "$OUTPUT_ROOT/random_l15" \
    2>&1 | tee "$OUTPUT_ROOT/random_l15.log"
fi

"$PYTHON" "$REPO_ROOT/rebuttal/analyze_reconstruction.py" \
  "$OUTPUT_ROOT/smnist_l15/results.json" \
  "$OUTPUT_ROOT/random_l15/results.json" \
  --output "$OUTPUT_ROOT/summary.json"

# Post hoc: degree-by-degree solver on the same targets (CPU, float64), one process per target.
MARCH="$REPO_ROOT/experiments/spherical_mnist_reconstruction/degree_marching.py"
MARCH_JOBS="${MARCH_JOBS:-12}"
MARCH_THREADS="${MARCH_THREADS:-2}"

march() {
  local source=$1 n_signals=$2 n_rotations=$3
  local out="$OUTPUT_ROOT/${source}_l15_marching"
  local last=$((n_signals * (n_rotations + 1) - 1))
  [[ -f "$out/results.json" ]] && return
  mkdir -p "$out/logs"
  seq 0 "$last" | xargs -P "$MARCH_JOBS" -I{} sh -c \
    "OMP_NUM_THREADS=$MARCH_THREADS '$PYTHON' '$MARCH' solve --signal_source $source \
      --n_signals $n_signals --n_rotations $n_rotations --seed 2026 --data_dir '$DATA_DIR' \
      --n_threads $MARCH_THREADS --targets {} --output_dir '$out' > '$out/logs/target_{}.log' 2>&1"
  "$PYTHON" "$MARCH" merge --output_dir "$out"
  for i in $(seq 0 "$last"); do cat "$out/logs/target_$i.log"; done > "$out/run.log"
}

march smnist 8 1
march random 16 0

"$PYTHON" "$MARCH" plot \
  --results "$OUTPUT_ROOT/smnist_l15_marching/results.json" \
  --rot_idx 0 \
  --paper_figure_path "$REPO_ROOT/paper/experiments/spherical_mnist_reconstruction/figures/paper_orbits.pdf"
