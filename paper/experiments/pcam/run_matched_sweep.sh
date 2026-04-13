#!/bin/bash
# Pareto sweep: run each model at multiple growth_rates to build AUC-vs-params curves.
#
# Param counts (from find_growth_rates.py):
#
#   standard:    gr=6→30K, gr=12→102K, gr=20→267K, gr=30→582K, gr=35→786K
#   norm:        gr=3→69K, gr=4→110K,  gr=6→222K,  gr=8→372K,  gr=12→791K
#   gate:        gr=3→136K, gr=4→218K, gr=6→440K,  gr=8→741K,  gr=12→1.58M
#   fourier_elu: gr=3→69K, gr=4→110K,  gr=6→222K,  gr=8→372K,  gr=12→790K
#   bispectrum:  gr=3→80K, gr=4→128K,  gr=6→258K,  gr=8→433K,  gr=12→920K
#   so2_disk:    all ~100K (MLP auto-sized), bandlimit controls feature quality
#
# Phase A (25 CNN + N so2_disk runs, single seed, skip rotation): ~4-6 hours
# Phase B (remaining seeds with rotation): ~10-15 hours
#
# Usage (run in tmux):
#   ./run_matched_sweep.sh              # Phase A
#   ./run_matched_sweep.sh --phase-b    # Phase B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /home/johmathe/bispectrum/.venv/bin/activate
export PYTHONPATH="/home/johmathe/bispectrum/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

STANDARD_GRS=(6 12 20 30 35)
EQUIVARIANT_GRS=(3 4 6 8 12)
SO2_DISK_BLS=(10 15 20 25 30 40 50)

OUTPUT_DIR="./pcam_results_pareto"
COMMON="--train_fraction 0.1 --patience 10 --epochs 50"

batch_size_for() {
    local model=$1 gr=$2
    if [[ "$model" == "fourier_elu" || "$model" == "bispectrum" || "$model" == "norm" ]]; then
        if (( gr >= 8 )); then echo 64
        elif (( gr >= 6 )); then echo 128
        else echo 256
        fi
    elif [[ "$model" == "gate" ]] && (( gr >= 8 )); then
        echo 128
    else
        echo 256
    fi
}

run_single() {
    local model=$1 gr=$2 seed=$3 extra=${4:-}
    local out_dir="${OUTPUT_DIR}/${model}_c8_gr${gr}_seed${seed}"
    if [[ -f "${out_dir}/results.json" ]]; then
        echo "SKIP (already done): model=$model gr=$gr seed=$seed"
        return 0
    fi
    local bs
    bs=$(batch_size_for "$model" "$gr")
    echo ""
    echo "============================================================"
    echo "  model=$model  gr=$gr  seed=$seed  bs=$bs  $(date)"
    echo "============================================================"
    python train.py --model "$model" --growth_rate "$gr" \
        --output_dir "$OUTPUT_DIR" --seed "$seed" --batch_size "$bs" $COMMON $extra
}

run_so2_disk() {
    local bl=$1 seed=$2 extra=${3:-}
    local out_dir="${OUTPUT_DIR}/so2_disk_bl${bl}_seed${seed}"
    if [[ -f "${out_dir}/results.json" ]]; then
        echo "SKIP (already done): model=so2_disk bl=$bl seed=$seed"
        return 0
    fi
    echo ""
    echo "============================================================"
    echo "  model=so2_disk  bl=$bl  seed=$seed  $(date)"
    echo "============================================================"
    python train.py --model so2_disk --bandlimit "$bl" \
        --output_dir "$OUTPUT_DIR" --seed "$seed" --batch_size 256 $COMMON $extra
}

if [[ "${1:-}" == "--phase-b" ]]; then
    echo "=== PHASE B: remaining seeds (123, 456) with rotation eval ==="
    for seed in 123 456; do
        for gr in "${STANDARD_GRS[@]}"; do
            run_single standard "$gr" "$seed"
        done
        for model in norm gate fourier_elu bispectrum; do
            for gr in "${EQUIVARIANT_GRS[@]}"; do
                run_single "$model" "$gr" "$seed"
            done
        done
        for bl in "${SO2_DISK_BLS[@]}"; do
            run_so2_disk "$bl" "$seed"
        done
    done
else
    echo "=== PHASE A: single seed (42), skip rotation ==="
    for gr in "${STANDARD_GRS[@]}"; do
        run_single standard "$gr" 42 "--skip_rotation"
    done
    for model in norm gate fourier_elu bispectrum; do
        for gr in "${EQUIVARIANT_GRS[@]}"; do
            run_single "$model" "$gr" 42 "--skip_rotation"
        done
    done
    for bl in "${SO2_DISK_BLS[@]}"; do
        run_so2_disk "$bl" 42 "--skip_rotation"
    done
fi

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "  Results in $OUTPUT_DIR"
echo "============================================================"
