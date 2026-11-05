#!/bin/bash
# Spherical MNIST sweep: 3 models x 2 training modes x 3 seeds = 18 runs.
#
# Models and approximate parameter counts (lmax=15, selective bispectrum):
#   standard:        ~185K
#   power_spectrum:   ~11K
#   bispectrum:      ~165K
#
# First run builds and caches the spherical projections (~2 min for NR,
# ~5 min for R due to per-image SO(3) rotations). Subsequent runs load
# from cache.
#
# Usage (run in tmux):
#   ./run_sweep.sh
#   SEEDS="42 123 456 789 101112" MODELS="bispectrum" ./run_sweep.sh
#
# SEEDS / MODELS / TRAIN_MODES are space-separated env overrides; defaults below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# shellcheck disable=SC2206  # word splitting is the point of these overrides
MODELS=(${MODELS:-standard power_spectrum bispectrum})
# train.py canonicalizes NR -> C in output dir names; use C here so the
# resume/skip check matches the actual results directories.
# shellcheck disable=SC2206
TRAIN_MODES=(${TRAIN_MODES:-C R})
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-42 123 456})
OUTPUT_DIR="./smnist_results"
# Same protocol for every model: 30 epochs (SGDR cycle end 10 + 20), no early stopping
# (best-validation checkpoint), batch size 256 (train.py default).
COMMON="--epochs 30"

run_single() {
    local model=$1 mode=$2 seed=$3
    local out_dir="${OUTPUT_DIR}/${model}_${mode}_seed${seed}"
    if [[ -f "${out_dir}/results.json" ]]; then
        echo "SKIP (already done): model=$model mode=$mode seed=$seed"
        return 0
    fi
    echo ""
    echo "============================================================"
    echo "  model=$model  mode=$mode  seed=$seed  $(date)"
    echo "============================================================"
    python train.py --model "$model" --train_mode "$mode" \
        --output_dir "$OUTPUT_DIR" --seed "$seed" $COMMON
}

echo "=== Full sweep: ${#MODELS[@]} models x ${#TRAIN_MODES[@]} modes x ${#SEEDS[@]} seeds ==="
for seed in "${SEEDS[@]}"; do
    for mode in "${TRAIN_MODES[@]}"; do
        for model in "${MODELS[@]}"; do
            run_single "$model" "$mode" "$seed"
        done
    done
done

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "  Results in $OUTPUT_DIR"
echo "============================================================"
