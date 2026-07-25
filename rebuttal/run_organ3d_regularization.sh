#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
TRAIN="$REPO_ROOT/experiments/organ3d/train.py"
DATA_DIR="${ORGAN3D_DATA_DIR:-$REPO_ROOT/experiments/organ3d/organ3d_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/rebuttal/results/organ3d_regularization}"
# Optional space-separated list of config labels to run (for splitting the
# sweep across GPUs). Empty means run everything.
CONFIG_LABELS="${CONFIG_LABELS:-}"

CONFIGS=(
  "max_base|max_pool|1e-4|0.0"
  "bsp_base|bispectrum|1e-4|0.0"
  "bsp_wd1e-3|bispectrum|1e-3|0.0"
  "bsp_wd1e-2|bispectrum|1e-2|0.0"
  "bsp_drop0.2|bispectrum|1e-4|0.2"
  "bsp_drop0.5|bispectrum|1e-4|0.5"
  "bsp_wd1e-3_drop0.2|bispectrum|1e-3|0.2"
)

for config in "${CONFIGS[@]}"; do
  IFS='|' read -r label model weight_decay dropout <<<"$config"
  if [[ -n "$CONFIG_LABELS" && " $CONFIG_LABELS " != *" $label "* ]]; then
    continue
  fi
  if [[ "$model" == "bispectrum" ]]; then
    batch_size=4
  else
    batch_size=8
  fi
  for seed in 42 123 456; do
    result="$OUTPUT_DIR/${label}_C_ch16_32_seed${seed}/results.json"
    if [[ -f "$result" ]]; then
      echo "SKIP existing $result"
      continue
    fi
    "$PYTHON" "$TRAIN" \
      --model "$model" \
      --run_label "$label" \
      --channels 16 32 \
      --train_mode C \
      --seed "$seed" \
      --epochs 100 \
      --patience 15 \
      --batch_size "$batch_size" \
      --lr 1e-3 \
      --weight_decay "$weight_decay" \
      --dropout "$dropout" \
      --data_dir "$DATA_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --skip_rotation
  done
done

# Only analyze once all 21 runs exist (relevant when the sweep is split
# across GPUs: the last finisher triggers the analysis).
n_done="$(find "$OUTPUT_DIR" -name results.json | wc -l)"
if [[ "$n_done" -eq 21 ]]; then
  "$PYTHON" "$REPO_ROOT/rebuttal/analyze_organ3d_regularization.py" \
    --results-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR/analysis"
else
  echo "Skipping analysis: $n_done/21 results present."
fi
