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
#
# Phase A (30 runs, single seed, skip rotation): ~3-5 hours
# Phase B (60 runs, 2 more seeds, with rotation): ~8-12 hours
#
# Usage (run in tmux):
#   ./run_data_pareto_sweep.sh              # Phase A
#   ./run_data_pareto_sweep.sh --phase-b    # Phase B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /home/johmathe/bispectrum/.venv/bin/activate
export PYTHONPATH="/home/johmathe/bispectrum/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

declare -A MODEL_GR
MODEL_GR[standard]=12
MODEL_GR[norm]=4
MODEL_GR[gate]=3
MODEL_GR[fourier_elu]=4
MODEL_GR[bispectrum]=4

MODELS=(standard norm gate fourier_elu bispectrum)
FRACTIONS=(0.01 0.05 0.1 0.25 0.5 1.0)

BASE_OUTPUT_DIR="./pcam_results_data_pareto"
COMMON="--patience 10 --epochs 50"

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
    local model=$1 frac=$2 seed=$3 extra=${4:-}
    local gr=${MODEL_GR[$model]}
    local output_dir="${BASE_OUTPUT_DIR}/frac_${frac}"
    local out_dir="${output_dir}/${model}_c8_gr${gr}_seed${seed}"
    if [[ -f "${out_dir}/results.json" ]]; then
        echo "SKIP (already done): model=$model frac=$frac seed=$seed"
        return 0
    fi
    local bs
    bs=$(batch_size_for "$model" "$gr")
    echo ""
    echo "============================================================"
    echo "  model=$model  gr=$gr  frac=$frac  seed=$seed  bs=$bs  $(date)"
    echo "============================================================"
    python train.py --model "$model" --growth_rate "$gr" \
        --output_dir "$output_dir" --seed "$seed" --batch_size "$bs" \
        --train_fraction "$frac" $COMMON $extra
}

if [[ "${1:-}" == "--phase-b" ]]; then
    echo "=== PHASE B: remaining seeds (123, 456) with rotation eval ==="
    for frac in "${FRACTIONS[@]}"; do
        for seed in 123 456; do
            for model in "${MODELS[@]}"; do
                run_single "$model" "$frac" "$seed"
            done
        done
    done
else
    echo "=== PHASE A: single seed (42), skip rotation ==="
    for frac in "${FRACTIONS[@]}"; do
        for model in "${MODELS[@]}"; do
            run_single "$model" "$frac" 42 "--skip_rotation"
        done
    done
fi

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "  Results in $BASE_OUTPUT_DIR"
echo "============================================================"
