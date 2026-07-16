#!/bin/bash
# Pareto sweep: run each model at multiple growth_rates to build AUC-vs-params curves.
#
# Protocol (consistent-comparison):
#   - Invariant models (norm, gate, fourier_elu, norm_pool, bispectrum, so2_disk)
#     are trained in canonical mode (C).
#   - The standard CNN is trained with rotation augmentation (R) — it is the
#     "Aug. CNN" baseline in the figures.
#   - Rotation (OOD) evaluation is always on: the figure curves plot rotated
#     test AUC, so every run must record test_r / rotation_robustness.
#
# Param counts (from find_growth_rates.py):
#
#   standard:    gr=6→30K, gr=12→102K, gr=20→267K, gr=30→582K, gr=35→786K
#   norm:        gr=3→69K, gr=4→110K,  gr=6→222K,  gr=8→372K,  gr=12→791K
#   gate:        gr=3→136K, gr=4→218K, gr=6→440K,  gr=8→741K,  gr=12→1.58M
#   fourier_elu: gr=3→69K, gr=4→110K,  gr=6→222K,  gr=8→372K,  gr=12→790K
#   norm_pool:   ~same as fourier_elu (identical backbone, paramless pool)
#   bispectrum:  gr=3→80K, gr=4→128K,  gr=6→258K,  gr=8→433K,  gr=12→920K
#   so2_disk:    all ~100K (MLP auto-sized), bandlimit controls feature quality
#
# Phase A (seed 42, all configs): first full curves
# Phase B (seeds 123, 456): error bars
#
# Usage (run in tmux):
#   ./run_matched_sweep.sh              # Phase A
#   ./run_matched_sweep.sh --phase-b    # Phase B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

STANDARD_GRS=(6 12 20 30 35)
EQUIVARIANT_GRS=(3 4 6 8 12)
SO2_DISK_BLS=(10 15 20 25 30 40 50)
EQUIVARIANT_MODELS=(norm gate fourier_elu norm_pool bispectrum)

OUTPUT_DIR="./pcam_results_pareto"
TRAIN_SIZE=12500
COMMON="--train_size ${TRAIN_SIZE} --patience 10 --epochs 50"
N_TAG="_n${TRAIN_SIZE}"

batch_size_for() {
    local model=$1 gr=$2
    if [[ "$model" == "fourier_elu" || "$model" == "bispectrum" || "$model" == "norm" || "$model" == "norm_pool" ]]; then
        if (( gr >= 8 )); then echo 64
        else echo 128
        fi
    elif [[ "$model" == "gate" ]] && (( gr >= 8 )); then
        echo 128
    else
        echo 256
    fi
}

run_single() {
    local model=$1 gr=$2 seed=$3 mode=${4:-C}
    local out_dir="${OUTPUT_DIR}/${model}_c8_gr${gr}_${mode}_seed${seed}${N_TAG}"
    if [[ -f "${out_dir}/results.json" ]]; then
        echo "SKIP (already done): model=$model gr=$gr mode=$mode seed=$seed"
        return 0
    fi
    local bs
    bs=$(batch_size_for "$model" "$gr")
    echo ""
    echo "============================================================"
    echo "  model=$model  gr=$gr  mode=$mode  seed=$seed  bs=$bs  $(date)"
    echo "============================================================"
    python train.py --model "$model" --growth_rate "$gr" --train_mode "$mode" \
        --output_dir "$OUTPUT_DIR" --seed "$seed" --batch_size "$bs" $COMMON
}

run_so2_disk() {
    local bl=$1 seed=$2
    local out_dir="${OUTPUT_DIR}/so2_disk_bl${bl}_C_seed${seed}${N_TAG}"
    if [[ -f "${out_dir}/results.json" ]]; then
        echo "SKIP (already done): model=so2_disk bl=$bl seed=$seed"
        return 0
    fi
    echo ""
    echo "============================================================"
    echo "  model=so2_disk  bl=$bl  seed=$seed  $(date)"
    echo "============================================================"
    python train.py --model so2_disk --bandlimit "$bl" --train_mode C \
        --output_dir "$OUTPUT_DIR" --seed "$seed" --batch_size 256 $COMMON
}

run_all_for_seed() {
    local seed=$1
    for gr in "${STANDARD_GRS[@]}"; do
        run_single standard "$gr" "$seed" R
    done
    for model in "${EQUIVARIANT_MODELS[@]}"; do
        for gr in "${EQUIVARIANT_GRS[@]}"; do
            run_single "$model" "$gr" "$seed" C
        done
    done
    for bl in "${SO2_DISK_BLS[@]}"; do
        run_so2_disk "$bl" "$seed"
    done
}

if [[ "${1:-}" == "--phase-b" ]]; then
    echo "=== PHASE B: remaining seeds (123, 456) ==="
    for seed in 123 456; do
        run_all_for_seed "$seed"
    done
else
    echo "=== PHASE A: seed 42, all configs ==="
    run_all_for_seed 42
fi

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "  Results in $OUTPUT_DIR"
echo "============================================================"
