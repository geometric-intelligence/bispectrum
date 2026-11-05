#!/bin/bash
# OrganMNIST3D main sweep: all 4 model variants x 3 seeds at channels (4,8).
#
# Protocol (consistent-comparison):
#   - Invariant models (max_pool, norm_pool, bispectrum) are trained
#     canonical (C).
#   - The standard 3D CNN is trained with random octahedral augmentation (R)
#     — it is the "Aug. 3D CNN" baseline in the figures.
#   - Rotation (OOD) evaluation is always on: the figures plot rotated test
#     accuracy next to canonical test accuracy.
#
# Param counts (default channels 4,8):
#   standard:    ~16K
#   max_pool:    ~374K
#   norm_pool:   ~375K
#   bispectrum:  ~463K
#
# Usage (run in tmux):
#   ./run_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

MODELS=(standard max_pool norm_pool bispectrum)
OUTPUT_DIR="./organ3d_results"
# Same protocol for every model: a fixed epoch budget that ends at an SGDR cycle end
# (10 + 20 + 40 epochs), no early stopping (best-validation checkpoint), one batch size.
COMMON="--epochs 70 --batch_size 16 --data_dir ./organ3d_data"

train_mode_for() {
    local model=$1
    if [[ "$model" == "standard" ]]; then
        echo "R"
    else
        echo "C"
    fi
}

run_single() {
    local model=$1 seed=$2
    local channels="4 8"
    local mode
    mode=$(train_mode_for "$model")
    local out_dir="${OUTPUT_DIR}/${model}_${mode}_ch4_8_seed${seed}"
    if [[ -f "${out_dir}/results.json" ]]; then
        echo "SKIP (already done): model=$model mode=$mode seed=$seed"
        return 0
    fi
    echo ""
    echo "============================================================"
    echo "  model=$model  mode=$mode  seed=$seed  $(date)"
    echo "============================================================"
    python train.py --model "$model" --channels $channels --train_mode "$mode" \
        --output_dir "$OUTPUT_DIR" --seed "$seed" $COMMON
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
