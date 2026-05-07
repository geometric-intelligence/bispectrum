#!/bin/bash
# Data-efficiency Pareto sweep: run each model at matched ~100K params across
# multiple training set fractions to build AUC-vs-data curves.
#
# Matched growth rates (from find_growth_rates.py, target ~100K params):
#   standard:    gr=12 → 102K
#   norm:        gr=4  → 110K
#   gate:        gr=3  → 136K
#   fourier_elu: gr=4  → 110K
#   bispectrum:  gr=4  → 128K
#   so2_disk:    bl=30 → ~100K (MLP auto-sized)
#
# Phase A (36 runs, single seed, skip rotation): ~4-6 hours
# Phase B (72 runs, 2 more seeds, with rotation): ~10-15 hours
#
# Usage (run in tmux):
#   ./run_data_pareto_sweep.sh              # Phase A
#   ./run_data_pareto_sweep.sh --phase-b    # Phase B

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
MODEL_GR[bispectrum]=4

MODELS=(standard norm gate fourier_elu bispectrum)
# Absolute training-set sizes; "full" maps to PCam's 262144 examples.
SIZES=(100 500 2500 12500 full)
SO2_DISK_BL=10

BASE_OUTPUT_DIR="./pcam_results_data_pareto"
COMMON="--patience 10 --epochs 50"

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

batch_size_for() {
    local model=$1 gr=$2
    case "$model" in
        standard)    echo 1024 ;;
        norm)        echo 128 ;;
        gate)        echo 128 ;;
        fourier_elu) echo 64 ;;
        bispectrum)  echo 128 ;;
        *)           echo 128 ;;
    esac
}

run_single() {
    local model=$1 size=$2 seed=$3 extra=${4:-}
    local gr=${MODEL_GR[$model]}
    local tag
    tag=$(size_tag "$size")
    local output_dir="${BASE_OUTPUT_DIR}/${tag}"
    local suffix=""
    if [[ "$size" != "full" ]]; then
        suffix="_n${size}"
    fi
    local out_dir="${output_dir}/${model}_c8_gr${gr}_seed${seed}${suffix}"
    if [[ -f "${out_dir}/results.json" ]]; then
        echo "SKIP (already done): model=$model size=$size seed=$seed"
        return 0
    fi
    local bs
    bs=$(batch_size_for "$model" "$gr")
    local size_arg
    size_arg=$(size_args "$size")
    echo ""
    echo "============================================================"
    echo "  model=$model  gr=$gr  size=$size  seed=$seed  bs=$bs  $(date)"
    echo "============================================================"
    python train.py --model "$model" --growth_rate "$gr" \
        --output_dir "$output_dir" --seed "$seed" --batch_size "$bs" \
        $size_arg $COMMON $extra
}

run_so2_disk() {
    local size=$1 seed=$2 extra=${3:-}
    local tag
    tag=$(size_tag "$size")
    local output_dir="${BASE_OUTPUT_DIR}/${tag}"
    local suffix=""
    if [[ "$size" != "full" ]]; then
        suffix="_n${size}"
    fi
    local out_dir="${output_dir}/so2_disk_bl${SO2_DISK_BL}_seed${seed}${suffix}"
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
    python train.py --model so2_disk --bandlimit "$SO2_DISK_BL" \
        --output_dir "$output_dir" --seed "$seed" --batch_size 256 \
        $size_arg $COMMON $extra
}

if [[ "${1:-}" == "--phase-b" ]]; then
    echo "=== PHASE B: remaining seeds (123, 456) with rotation eval ==="
    for size in "${SIZES[@]}"; do
        for seed in 123 456; do
            for model in "${MODELS[@]}"; do
                run_single "$model" "$size" "$seed"
            done
            run_so2_disk "$size" "$seed"
        done
    done
else
    echo "=== PHASE A: single seed (42), skip rotation ==="
    for size in "${SIZES[@]}"; do
        for model in "${MODELS[@]}"; do
            run_single "$model" "$size" 42 "--skip_rotation"
        done
        run_so2_disk "$size" 42 "--skip_rotation"
    done
fi

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "  Results in $BASE_OUTPUT_DIR"
echo "============================================================"
