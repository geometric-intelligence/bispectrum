#!/bin/bash
# Data efficiency sweep: 3 models x 2 modes x 3 seeds x 2 sample-count steps = 36 runs.
# Full-set already done in run_sweep.sh.
#
# Usage (run in tmux):
#   ./run_data_efficiency.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /home/johmathe/bispectrum/.venv/bin/activate
export PYTHONPATH="/home/johmathe/bispectrum/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

MODELS=(standard power_spectrum bispectrum)
TRAIN_MODES=(NR R)
SIZES=(500 6000)
OUTPUT_DIR="./smnist_results"
COMMON="--patience 10 --epochs 50"

run_single() {
    local model=$1 mode=$2 seed=$3 size=$4
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
        for mode in "${TRAIN_MODES[@]}"; do
            for model in "${MODELS[@]}"; do
                run_single "$model" "$mode" "$seed" "$size"
            done
        done
    done
done

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "  Results in $OUTPUT_DIR"
echo "============================================================"
