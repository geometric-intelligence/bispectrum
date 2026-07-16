#!/bin/bash
# Data efficiency sweep for the grid figure's data-efficiency curves.
#
# Protocol (consistent-comparison):
#   - Invariant models (power_spectrum, bispectrum) are trained canonical (C).
#   - The standard CNN is trained on SO(3)-rotated data (R) — it is the
#     "Aug. CNN" baseline in the figures.
#   - test_r (rotated test set) is always recorded; the figure curves plot
#     rotated test accuracy. --skip_rotation only skips the extra
#     per-rotation robustness stats, which the curves do not need.
#
# Full-training-set points come from run_sweep.sh.
#
# Usage (run in tmux):
#   ./run_data_efficiency.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SIZES=(100 500 2500 12500)
OUTPUT_DIR="./smnist_results"
COMMON="--patience 10 --epochs 50"

train_mode_for() {
    local model=$1
    if [[ "$model" == "standard" ]]; then
        echo "R"
    else
        echo "C"
    fi
}

run_single() {
    local model=$1 seed=$2 size=$3
    local mode
    mode=$(train_mode_for "$model")
    local out_dir="${OUTPUT_DIR}/${model}_${mode}_seed${seed}_n${size}"
    if [[ -f "${out_dir}/results.json" ]]; then
        echo "SKIP (already done): model=$model mode=$mode seed=$seed size=$size"
        return 0
    fi
    echo ""
    echo "============================================================"
    echo "  model=$model  mode=$mode  seed=$seed  size=$size  $(date)"
    echo "============================================================"
    python train.py --model "$model" --train_mode "$mode" \
        --output_dir "$OUTPUT_DIR" --seed "$seed" \
        --train_size "$size" \
        --skip_rotation \
        $COMMON
}

for seed in 42 123 456; do
    for size in "${SIZES[@]}"; do
        for model in standard power_spectrum bispectrum; do
            run_single "$model" "$seed" "$size"
        done
    done
done

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "  Results in $OUTPUT_DIR"
echo "============================================================"
