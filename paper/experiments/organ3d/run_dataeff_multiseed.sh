#!/bin/bash
# Data efficiency: 3 seeds x 4 fractions x 3 models
# Seed 42 already done — this runs seeds 123, 456

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /home/johmathe/bispectrum/.venv/bin/activate
export PYTHONPATH="/home/johmathe/bispectrum/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="./organ3d_results"
COMMON="--patience 15 --epochs 100 --data_dir ./organ3d_data"

batch_size_for() {
    local model=$1
    if [[ "$model" == "bispectrum" ]]; then echo 16
    elif [[ "$model" == "standard" ]]; then echo 64
    else echo 32; fi
}

run_single() {
    local model=$1 seed=$2 frac=$3
    local channels="4 8"
    local frac_tag="_frac${frac}"
    local out_dir="${OUTPUT_DIR}/${model}_ch4_8_seed${seed}${frac_tag}"

    if [[ -f "${out_dir}/results.json" ]]; then
        echo "SKIP (already done): model=$model seed=$seed frac=$frac"
        return 0
    fi

    local bs
    bs=$(batch_size_for "$model")

    echo ""
    echo "============================================================"
    echo "  model=$model  seed=$seed  frac=$frac  bs=$bs  $(date)"
    echo "============================================================"
    python train.py --model "$model" --channels $channels \
        --output_dir "$OUTPUT_DIR" --seed "$seed" --batch_size "$bs" \
        --train_fraction "$frac" $COMMON
}

for seed in 123 456; do
    for frac in 0.05 0.10 0.25 0.50; do
        for model in standard max_pool bispectrum; do
            run_single "$model" "$seed" "$frac"
        done
    done
done

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "============================================================"
