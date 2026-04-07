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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /home/johmathe/bispectrum/.venv/bin/activate
export PYTHONPATH="/home/johmathe/bispectrum/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

MODELS=(standard power_spectrum bispectrum)
TRAIN_MODES=(NR R)
OUTPUT_DIR="./smnist_results"
COMMON="--patience 10 --epochs 50"

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

echo "=== Full sweep: 3 models x 2 modes x 3 seeds ==="
for seed in 42 123 456; do
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
