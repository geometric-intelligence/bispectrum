#!/bin/bash
# OrganMNIST3D sweep: run all 4 model variants x 3 seeds.
#
# Param counts (default channels 4,8):
#   standard:    ~16K
#   max_pool:    ~374K
#   norm_pool:   ~375K
#   bispectrum:  ~463K
#
# Full sweep: 4 models x 3 seeds with rotation eval (~2 hours)
#
# Usage (run in tmux):
#   ./run_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /home/johmathe/bispectrum/.venv/bin/activate
export PYTHONPATH="/home/johmathe/bispectrum/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

MODELS=(standard max_pool norm_pool bispectrum)
OUTPUT_DIR="./organ3d_results"
COMMON="--patience 15 --epochs 100 --data_dir ./organ3d_data"

batch_size_for() {
    local model=$1
    if [[ "$model" == "bispectrum" ]]; then
        echo 16
    elif [[ "$model" == "standard" ]]; then
        echo 64
    else
        echo 32
    fi
}

run_single() {
    local model=$1 seed=$2 extra=${3:-}
    local channels="4 8"
    local out_dir="${OUTPUT_DIR}/${model}_ch4_8_seed${seed}"
    if [[ -f "${out_dir}/results.json" ]]; then
        echo "SKIP (already done): model=$model seed=$seed"
        return 0
    fi
    local bs
    bs=$(batch_size_for "$model")
    echo ""
    echo "============================================================"
    echo "  model=$model  seed=$seed  bs=$bs  $(date)"
    echo "============================================================"
    python train.py --model "$model" --channels $channels \
        --output_dir "$OUTPUT_DIR" --seed "$seed" --batch_size "$bs" $COMMON $extra
}

echo "=== Full sweep: all seeds with rotation eval ==="
for seed in 42 123 456; do
    for model in "${MODELS[@]}"; do
        run_single "$model" "$seed"
    done
done

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "  Results in $OUTPUT_DIR"
echo "============================================================"
