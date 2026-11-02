#!/bin/bash
# Batch-size control for the OrganMNIST3D low-data comparison.
#
# In the grid protocol the bispectrum model trains at batch 16 while the
# pooling baselines train at batch 32 (run_dataeff_multiseed.sh), so at
# n=100/500 the bispectrum takes 2x more optimizer steps per epoch. This
# control reruns max_pool at batch 16 so the bispectrum-vs-max_pool gap can be
# attributed to the invariant rather than to the step budget.
#
# Results go to a separate directory: results['model'] stays 'max_pool', so
# mixing them into ./organ3d_results would double-count in the grid loader.
#
# Usage (run in tmux):
#   ./run_bs16_control.sh
#   SEEDS="42 123 456" SIZES="100" ./run_bs16_control.sh
#
# SEEDS / SIZES / MODELS / BATCH_SIZE are space-separated env overrides.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="${OUTPUT_DIR:-./organ3d_results_bs16_control}"
COMMON="--patience 15 --epochs 100 --data_dir ./organ3d_data"
BATCH_SIZE="${BATCH_SIZE:-16}"
# shellcheck disable=SC2206  # word splitting is the point of these overrides
SEEDS=(${SEEDS:-42 123 456 789 101112 1 2 3 4 5})
# shellcheck disable=SC2206
SIZES=(${SIZES:-100 500})
# shellcheck disable=SC2206
MODELS=(${MODELS:-max_pool})

run_single() {
    local model=$1 seed=$2 size=$3
    local channels="4 8"
    local mode="C"
    local out_dir="${OUTPUT_DIR}/${model}_${mode}_ch4_8_seed${seed}_n${size}"

    if [[ -f "${out_dir}/results.json" ]]; then
        echo "SKIP (already done): model=$model mode=$mode seed=$seed size=$size"
        return 0
    fi

    echo ""
    echo "============================================================"
    echo "  model=$model  mode=$mode  seed=$seed  size=$size  bs=$BATCH_SIZE  $(date)"
    echo "============================================================"
    python train.py --model "$model" --channels $channels --train_mode "$mode" \
        --output_dir "$OUTPUT_DIR" --seed "$seed" --batch_size "$BATCH_SIZE" \
        --train_size "$size" $COMMON
}

for seed in "${SEEDS[@]}"; do
    for size in "${SIZES[@]}"; do
        for model in "${MODELS[@]}"; do
            run_single "$model" "$seed" "$size"
        done
    done
done

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "============================================================"
