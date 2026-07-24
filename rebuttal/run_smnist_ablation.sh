#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
TRAIN="$REPO_ROOT/experiments/spherical_mnist/train.py"
DATA_DIR="${SMNIST_DATA_DIR:-$REPO_ROOT/experiments/spherical_mnist/smnist_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/rebuttal/results/smnist_ablation}"
TRAIN_MODES="${TRAIN_MODES:-C}"

for features in bootstrap bootstrap_self bootstrap_self_cg full; do
  for mode in $TRAIN_MODES; do
    for seed in 42 123 456; do
      label="bispectrum_${features}"
      result="$OUTPUT_DIR/${label}_${mode}_seed${seed}/results.json"
      if [[ -f "$result" ]]; then
        echo "SKIP existing $result"
        continue
      fi
      "$PYTHON" "$TRAIN" \
        --model bispectrum \
        --bispectrum_features "$features" \
        --run_label "$label" \
        --train_mode "$mode" \
        --seed "$seed" \
        --lmax 15 \
        --nlat 64 \
        --nlon 128 \
        --hidden 256 \
        --epochs 50 \
        --patience 10 \
        --batch_size 256 \
        --lr 1e-3 \
        --weight_decay 1e-4 \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --skip_rotation
    done
  done
done

"$PYTHON" "$REPO_ROOT/rebuttal/analyze_smnist_ablation.py" \
  --results-dir "$OUTPUT_DIR" \
  --output "$OUTPUT_DIR/summary.json"
