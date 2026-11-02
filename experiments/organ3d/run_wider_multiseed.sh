#!/bin/bash
# Capacity (parameter-efficiency) sweep: 3 seeds x 3 channel widths x 4 models.
# Together with run_sweep.sh (channels 4,8) this builds the
# accuracy-vs-params curves for the grid figure.
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
#   ./run_wider_multiseed.sh
#   SEEDS="789 101112" WIDTHS="16_32" MODELS="max_pool bispectrum" ./run_wider_multiseed.sh
#
# SEEDS / WIDTHS / MODELS are space-separated env overrides; defaults below.
# WIDTHS entries use "_" between channel counts (e.g. 8_16 -> --channels 8 16).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="./organ3d_results"
COMMON="--patience 15 --epochs 100 --data_dir ./organ3d_data"
# shellcheck disable=SC2206  # word splitting is the point of these overrides
SEEDS=(${SEEDS:-42 123 456})
# shellcheck disable=SC2206
WIDTHS=(${WIDTHS:-8_16 16_32})
# shellcheck disable=SC2206
MODELS=(${MODELS:-standard max_pool norm_pool bispectrum})

train_mode_for() {
    local model=$1
    if [[ "$model" == "standard" ]]; then
        echo "R"
    else
        echo "C"
    fi
}

batch_size_for() {
    local model=$1 channels=$2
    local c1
    c1=$(echo "$channels" | awk '{print $NF}')
    if [[ "$model" == "bispectrum" ]]; then
        if (( c1 >= 32 )); then echo 4
        elif (( c1 >= 16 )); then echo 8
        else echo 16; fi
    elif [[ "$model" == "standard" ]]; then
        echo 64
    else
        if (( c1 >= 32 )); then echo 8
        elif (( c1 >= 16 )); then echo 16
        else echo 32; fi
    fi
}

run_single() {
    local model=$1 seed=$2 channels=$3
    local ch_tag="${channels// /_}"
    local mode
    mode=$(train_mode_for "$model")
    local out_dir="${OUTPUT_DIR}/${model}_${mode}_ch${ch_tag}_seed${seed}"

    if [[ -f "${out_dir}/results.json" ]]; then
        echo "SKIP (already done): model=$model mode=$mode seed=$seed channels=$channels"
        return 0
    fi

    local bs
    bs=$(batch_size_for "$model" "$channels")

    echo ""
    echo "============================================================"
    echo "  model=$model  mode=$mode  seed=$seed  channels=$channels  bs=$bs  $(date)"
    echo "============================================================"
    python train.py --model "$model" --channels $channels --train_mode "$mode" \
        --output_dir "$OUTPUT_DIR" --seed "$seed" --batch_size "$bs" \
        $COMMON
}

for seed in "${SEEDS[@]}"; do
    for width in "${WIDTHS[@]}"; do
        channels="${width//_/ }"
        for model in "${MODELS[@]}"; do
            run_single "$model" "$seed" "$channels"
        done
    done
done

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "  Results in $OUTPUT_DIR"
echo "============================================================"
