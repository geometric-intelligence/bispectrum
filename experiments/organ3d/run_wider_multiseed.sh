#!/bin/bash
# Wider channels multi-seed: seeds 123, 456 for (8,16) and (16,32).
# Seed 42 already done in run_tier1_sweep.sh Part B.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="./organ3d_results"
COMMON="--patience 15 --epochs 100 --data_dir ./organ3d_data"

batch_size_for() {
    local model=$1 channels=$2
    local c1
    c1=$(echo "$channels" | awk '{print $NF}')
    if [[ "$model" == "bispectrum" ]]; then
        if (( c1 >= 32 )); then echo 4
        elif (( c1 >= 16 )); then echo 8
        else echo 16; fi
    else
        if (( c1 >= 32 )); then echo 8
        elif (( c1 >= 16 )); then echo 16
        else echo 32; fi
    fi
}

run_single() {
    local model=$1 seed=$2 channels=$3
    local ch_tag="${channels// /_}"
    local out_dir="${OUTPUT_DIR}/${model}_ch${ch_tag}_seed${seed}"

    if [[ -f "${out_dir}/results.json" ]]; then
        echo "SKIP (already done): model=$model seed=$seed channels=$channels"
        return 0
    fi

    local bs
    bs=$(batch_size_for "$model" "$channels")

    echo ""
    echo "============================================================"
    echo "  model=$model  seed=$seed  channels=$channels  bs=$bs  $(date)"
    echo "============================================================"
    python train.py --model "$model" --channels $channels \
        --output_dir "$OUTPUT_DIR" --seed "$seed" --batch_size "$bs" \
        $COMMON
}

for seed in 123 456; do
    for channels in "8 16" "16 32"; do
        for model in max_pool bispectrum; do
            run_single "$model" "$seed" "$channels"
        done
    done
done

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "  Results in $OUTPUT_DIR"
echo "============================================================"
