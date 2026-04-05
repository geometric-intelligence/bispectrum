#!/bin/bash
# Tier 1 experiments: data efficiency + wider channels
#
# Part A: Data efficiency — train at {5%, 10%, 25%, 50%} of data
#         (100% already done in the main sweep)
#         Models: standard, max_pool, bispectrum (skip norm_pool — it's unstable)
#         Single seed (42) for speed; rotation eval on all.
#
# Part B: Wider channels — (8,16) and (16,32)
#         Models: max_pool, bispectrum (the comparison that matters)
#         Single seed (42); rotation eval on all.
#
# Usage (run in tmux):
#   ./run_tier1_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /home/johmathe/bispectrum/.venv/bin/activate
export PYTHONPATH="/home/johmathe/bispectrum/src:${PYTHONPATH:-}"
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
    elif [[ "$model" == "standard" ]]; then
        echo 64
    else
        if (( c1 >= 32 )); then echo 8
        elif (( c1 >= 16 )); then echo 16
        else echo 32; fi
    fi
}

run_single() {
    local model=$1 seed=$2 channels=$3 frac=$4
    local ch_tag="${channels// /_}"
    local frac_tag=""
    if [[ "$frac" != "1.0" ]]; then
        frac_tag="_frac${frac}"
    fi
    local out_dir="${OUTPUT_DIR}/${model}_ch${ch_tag}_seed${seed}${frac_tag}"

    if [[ -f "${out_dir}/results.json" ]]; then
        echo "SKIP (already done): model=$model seed=$seed channels=$channels frac=$frac"
        return 0
    fi

    local bs
    bs=$(batch_size_for "$model" "$channels")

    echo ""
    echo "============================================================"
    echo "  model=$model  seed=$seed  channels=$channels  frac=$frac  bs=$bs  $(date)"
    echo "============================================================"
    python train.py --model "$model" --channels $channels \
        --output_dir "$OUTPUT_DIR" --seed "$seed" --batch_size "$bs" \
        --train_fraction "$frac" $COMMON
}

echo "============================================================"
echo "  PART A: Data efficiency sweep"
echo "============================================================"
for frac in 0.05 0.10 0.25 0.50; do
    for model in standard max_pool bispectrum; do
        run_single "$model" 42 "4 8" "$frac"
    done
done

echo ""
echo "============================================================"
echo "  PART B: Wider channels sweep"
echo "============================================================"
for channels in "8 16" "16 32"; do
    for model in max_pool bispectrum; do
        run_single "$model" 42 "$channels" "1.0"
    done
done

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "  Results in $OUTPUT_DIR"
echo "============================================================"
