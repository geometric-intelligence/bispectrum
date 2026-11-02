#!/bin/bash
# Data efficiency: 3 seeds x 4 sample-count steps x 4 models at channels (4,8).
# Full-training-set points come from run_sweep.sh.
#
# Protocol (consistent-comparison):
#   - Invariant models (max_pool, norm_pool, bispectrum) are trained
#     canonical (C).
#   - The standard 3D CNN is trained with octahedral augmentation (R) — it is
#     the "Aug. 3D CNN" baseline in the figures.
#   - Rotation (OOD) evaluation is always on: the figure curves plot rotated
#     test accuracy.
#
# Usage (run in tmux):
#   ./run_dataeff_multiseed.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="./organ3d_results"
COMMON="--patience 15 --epochs 100 --data_dir ./organ3d_data"

train_mode_for() {
    local model=$1
    if [[ "$model" == "standard" ]]; then
        echo "R"
    else
        echo "C"
    fi
}

batch_size_for() {
    local model=$1
    if [[ "$model" == "bispectrum" ]]; then echo 16
    elif [[ "$model" == "standard" ]]; then echo 64
    else echo 32; fi
}

run_single() {
    local model=$1 seed=$2 size=$3
    local channels="4 8"
    local mode
    mode=$(train_mode_for "$model")
    local size_tag="_n${size}"
    local out_dir="${OUTPUT_DIR}/${model}_${mode}_ch4_8_seed${seed}${size_tag}"

    if [[ -f "${out_dir}/results.json" ]]; then
        echo "SKIP (already done): model=$model mode=$mode seed=$seed size=$size"
        return 0
    fi

    local bs
    bs=$(batch_size_for "$model")

    echo ""
    echo "============================================================"
    echo "  model=$model  mode=$mode  seed=$seed  size=$size  bs=$bs  $(date)"
    echo "============================================================"
    python train.py --model "$model" --channels $channels --train_mode "$mode" \
        --output_dir "$OUTPUT_DIR" --seed "$seed" --batch_size "$bs" \
        --train_size "$size" $COMMON
}

for seed in 42 123 456; do
    for size in 50 100 250 500; do
        for model in standard max_pool norm_pool bispectrum; do
            run_single "$model" "$seed" "$size"
        done
    done
done

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "============================================================"
