#!/bin/bash
# Data-efficiency Pareto sweep: run each model at matched ~100K params across
# multiple training set fractions to build AUC-vs-data curves.
#
# Protocol (consistent-comparison):
#   - Invariant models are trained in canonical mode (C).
#   - The standard CNN is trained with rotation augmentation (R) — it is the
#     "Aug. CNN" baseline in the figures.
#   - Rotation (OOD) evaluation is always on: the figure curves plot rotated
#     test AUC.
#
# Matched growth rates (from find_growth_rates.py, target ~100K params):
#   standard:    gr=12 → 102K
#   norm:        gr=4  → 110K
#   gate:        gr=3  → 136K
#   fourier_elu: gr=4  → 110K
#   norm_pool:   gr=4  → ~110K (same backbone as fourier_elu, paramless pool)
#   bispectrum:  gr=4  → 128K
#   so2_disk:    bl=30 → ~100K (MLP auto-sized)
#
# Usage (run in tmux):
#   ./run_data_pareto_sweep.sh              # Phase A (seed 42)
#   ./run_data_pareto_sweep.sh --phase-b    # Phase B (seeds 123, 456)
#   SEEDS="789 101112" SIZES="100 500 2500" RUN_SO2_DISK=0 ./run_data_pareto_sweep.sh --phase-b
#
# SEEDS (phase B only) / SIZES / MODELS are space-separated env overrides;
# RUN_SO2_DISK=0 skips the SO2onDisk MLP runs. Defaults below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

declare -A MODEL_GR
MODEL_GR[standard]=12
MODEL_GR[norm]=4
MODEL_GR[gate]=3
MODEL_GR[fourier_elu]=4
MODEL_GR[norm_pool]=4
MODEL_GR[bispectrum]=4

# shellcheck disable=SC2206  # word splitting is the point of these overrides
MODELS=(${MODELS:-standard norm gate fourier_elu norm_pool bispectrum})
# Absolute training-set sizes; "full" maps to PCam's 262144 examples.
# shellcheck disable=SC2206
SIZES=(${SIZES:-100 500 2500 12500 full})
# Phase B seeds (phase A is always seed 42).
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-123 456})
RUN_SO2_DISK=${RUN_SO2_DISK:-1}
SO2_DISK_BL=30

BASE_OUTPUT_DIR="./pcam_results_data_pareto"
# Same protocol for every model: a fixed epoch budget that ends at an SGDR cycle end, no early
# stopping (best-validation checkpoint), one batch size. 64 is the largest batch at which every
# model/growth rate in this and run_matched_sweep.sh fits on an 80 GB GPU.
COMMON="--batch_size 64"

# Full-data epochs are ~10-45 min each, so full data stops at the first cycle end (10);
# subsets get 10 + 20 epochs.
epochs_for() {
    local size=$1
    if [[ "$size" == "full" ]]; then
        echo 10
    else
        echo 30
    fi
}

train_mode_for() {
    local model=$1
    if [[ "$model" == "standard" ]]; then
        echo "R"
    else
        echo "C"
    fi
}

size_args() {
    local size=$1
    if [[ "$size" == "full" ]]; then
        echo ""
    else
        echo "--train_size $size"
    fi
}

size_tag() {
    local size=$1
    if [[ "$size" == "full" ]]; then
        echo "n_full"
    else
        echo "n_${size}"
    fi
}

run_single() {
    local model=$1 size=$2 seed=$3
    local gr=${MODEL_GR[$model]}
    local mode
    mode=$(train_mode_for "$model")
    local tag
    tag=$(size_tag "$size")
    local output_dir="${BASE_OUTPUT_DIR}/${tag}"
    local suffix=""
    if [[ "$size" != "full" ]]; then
        suffix="_n${size}"
    fi
    local out_dir="${output_dir}/${model}_c8_gr${gr}_${mode}_seed${seed}${suffix}"
    if [[ -f "${out_dir}/results.json" ]]; then
        echo "SKIP (already done): model=$model size=$size seed=$seed"
        return 0
    fi
    local size_arg
    size_arg=$(size_args "$size")
    echo ""
    echo "============================================================"
    echo "  model=$model  gr=$gr  mode=$mode  size=$size  seed=$seed  $(date)"
    echo "============================================================"
    python train.py --model "$model" --growth_rate "$gr" --train_mode "$mode" \
        --output_dir "$output_dir" --seed "$seed" --epochs "$(epochs_for "$size")" \
        $size_arg $COMMON
}

run_so2_disk() {
    local size=$1 seed=$2
    local tag
    tag=$(size_tag "$size")
    local output_dir="${BASE_OUTPUT_DIR}/${tag}"
    local suffix=""
    if [[ "$size" != "full" ]]; then
        suffix="_n${size}"
    fi
    local out_dir="${output_dir}/so2_disk_bl${SO2_DISK_BL}_C_seed${seed}${suffix}"
    if [[ -f "${out_dir}/results.json" ]]; then
        echo "SKIP (already done): model=so2_disk size=$size seed=$seed"
        return 0
    fi
    local size_arg
    size_arg=$(size_args "$size")
    echo ""
    echo "============================================================"
    echo "  model=so2_disk  bl=$SO2_DISK_BL  size=$size  seed=$seed  $(date)"
    echo "============================================================"
    python train.py --model so2_disk --bandlimit "$SO2_DISK_BL" --train_mode C \
        --output_dir "$output_dir" --seed "$seed" --epochs "$(epochs_for "$size")" \
        $size_arg $COMMON
}

if [[ "${1:-}" == "--phase-b" ]]; then
    echo "=== PHASE B: remaining seeds (${SEEDS[*]}) ==="
    for size in "${SIZES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for model in "${MODELS[@]}"; do
                run_single "$model" "$size" "$seed"
            done
            if [[ "$RUN_SO2_DISK" == "1" ]]; then
                run_so2_disk "$size" "$seed"
            fi
        done
    done
else
    echo "=== PHASE A: seed 42 ==="
    for size in "${SIZES[@]}"; do
        for model in "${MODELS[@]}"; do
            run_single "$model" "$size" 42
        done
        if [[ "$RUN_SO2_DISK" == "1" ]]; then
            run_so2_disk "$size" 42
        fi
    done
fi

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "  Results in $BASE_OUTPUT_DIR"
echo "============================================================"
