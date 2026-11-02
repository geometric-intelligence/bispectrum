#!/bin/bash
# Capacity (parameter-efficiency) sweep: power_spectrum and bispectrum at
# multiple MLP widths x 3 seeds, C-trained, always evaluated on the rotated
# test set. Builds the accuracy-vs-params curves for the grid figure.
#
# Param counts (lmax=15):
#   power_spectrum: h=128->3.7K  h=256->11K  h=512->39K  h=1024->144K  h=2048->550K
#   bispectrum:     h=32->25K    h=64->52K   h=128->108K h=256->232K   h=512->529K
#
# The largest power-spectrum width (550K) matches the largest bispectrum
# budget (529K), so the incomplete-vs-complete invariant comparison is
# capacity-controlled along the whole curve. The standard CNN has a fixed
# architecture (185K) and appears as a single point from the main sweep.
#
# Usage (run in tmux):
#   ./run_capacity_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="./smnist_results_capacity"
COMMON="--epochs 50 --patience 10 --output_dir ${OUTPUT_DIR}"

PS_HIDDENS=(128 256 512 1024 2048)
BSP_HIDDENS=(32 64 128 256 512)
SEEDS=(42 123 456)

run_single() {
    local model=$1 hidden=$2 seed=$3
    local label="${model}_h${hidden}"
    local out_dir="${OUTPUT_DIR}/${label}_C_seed${seed}"
    if [[ -f "${out_dir}/results.json" ]]; then
        echo "SKIP (already done): model=$model hidden=$hidden seed=$seed"
        return 0
    fi
    echo ""
    echo "============================================================"
    echo "  model=$model  hidden=$hidden  seed=$seed  $(date)"
    echo "============================================================"
    python train.py --model "$model" --hidden "$hidden" \
        --run_label "$label" --train_mode C --seed "$seed" $COMMON
}

for seed in "${SEEDS[@]}"; do
    for hidden in "${PS_HIDDENS[@]}"; do
        run_single power_spectrum "$hidden" "$seed"
    done
    for hidden in "${BSP_HIDDENS[@]}"; do
        run_single bispectrum "$hidden" "$seed"
    done
done

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "  Results in $OUTPUT_DIR"
echo "============================================================"
