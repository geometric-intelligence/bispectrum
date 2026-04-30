#!/usr/bin/env bash
# Normalized experiment launcher.
#
# Default mode is `check`: enumerate every planned run and execute one-batch
# memory probes. Use `run` only after the check manifest is clean.

set -euo pipefail

MODE="${1:-check}"
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="${ROOT_DIR}/paper/experiments"
RESULTS_ROOT="${RESULTS_ROOT:-${EXPERIMENT_DIR}/normalized_results}"
MEMORY_MANIFEST="${MEMORY_MANIFEST:-${RESULTS_ROOT}/memory_manifest.jsonl}"
SUBSET_DIR="${SUBSET_DIR:-${RESULTS_ROOT}/subsets}"
MEMORY_HEADROOM="${MEMORY_HEADROOM:-0.85}"
FAIL_FAST="${FAIL_FAST:-1}"
RESUME="${RESUME:-1}"
SEEDS=(${SEEDS:-42 123 456 789 101112})
CHECK_SEED="${CHECK_SEED:-42}"
PCAM_SIZES=(${PCAM_SIZES:-100 500 2500 12500 full})
ORGAN_SIZES=(${ORGAN_SIZES:-100 500 full})
SMNIST_SIZES=(${SMNIST_SIZES:-100 500 2500 12500 full})
TRAIN_MODES=(${TRAIN_MODES:-C R})

usage() {
    cat <<EOF
Usage: $(basename "$0") [check|run|analyze]

Environment overrides:
  SEEDS="42 123 456 789 101112"
  CHECK_SEED=42
  PCAM_SIZES="100 500 2500 12500 full"
  ORGAN_SIZES="100 500 full"
  SMNIST_SIZES="100 500 2500 12500 full"
  RESULTS_ROOT="${RESULTS_ROOT}"
  MEMORY_HEADROOM=0.85
  FAIL_FAST=1
  RESUME=1

Modes:
  check    Run one representative-seed memory check per config and write memory_manifest.jsonl.
  run      Run full training commands, skipping completed results when RESUME=1.
  analyze  Validate seed/protocol coverage from completed results.
EOF
}

activate_env() {
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate bispectrum
    elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
        conda activate bispectrum
    elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
        conda activate bispectrum
    else
        echo "WARN: conda was not found; continuing with .venv only." >&2
    fi
    source "${ROOT_DIR}/.venv/bin/activate"
    export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
}

run_or_fail() {
    echo "+ $*"
    if [[ "${FAIL_FAST}" == "1" ]]; then
        "$@"
    else
        "$@" || echo "WARN: command failed: $*" >&2
    fi
}

maybe_skip() {
    local out_dir=$1
    if [[ "${MODE}" == "run" && "${RESUME}" == "1" && -f "${out_dir}/results.json" ]]; then
        echo "SKIP completed: ${out_dir}"
        return 0
    fi
    return 1
}

manifest_gate() {
    local rc=0
    python "${EXPERIMENT_DIR}/check_manifest.py" --manifest "$MEMORY_MANIFEST" "$@" || rc=$?
    if [[ $rc -eq 0 ]]; then
        return 0
    fi
    if [[ "${FAIL_FAST}" == "1" ]]; then
        echo "FAIL: manifest gate refused to launch (rc=${rc}); rerun 'check' first." >&2
        exit "$rc"
    fi
    echo "WARN: manifest gate refused to launch (rc=${rc}); skipping." >&2
    return 1
}

pcam_batch_size() {
    case "$1" in
        standard) echo 1024 ;;
        fourier_elu) echo 64 ;;
        *) echo 128 ;;
    esac
}

organ_batch_size() {
    case "$1" in
        standard) echo 64 ;;
        bispectrum) echo 16 ;;
        *) echo 32 ;;
    esac
}

smnist_batch_size() {
    echo 256
}

# Bucket a run under "n_full" or "n_<size>" without leaking the literal "full" sentinel.
size_bucket() {
    local size=$1
    if [[ "$size" == "full" ]]; then
        echo "n_full"
    else
        echo "n_${size}"
    fi
}

# Output dir suffix used by train.py: empty for full, _n<size> for subsets.
size_suffix() {
    local size=$1
    if [[ "$size" == "full" ]]; then
        echo ""
    else
        echo "_n${size}"
    fi
}

run_pcam() {
    local model=$1 mode=$2 size=$3 seed=$4
    local gr
    case "$model" in
        standard) gr=12 ;;
        norm) gr=4 ;;
        gate) gr=3 ;;
        fourier_elu) gr=4 ;;
        bispectrum) gr=4 ;;
        *) echo "unknown PCam model: $model" >&2; exit 1 ;;
    esac
    local bs
    bs=$(pcam_batch_size "$model")
    local bucket
    bucket=$(size_bucket "$size")
    local suffix
    suffix=$(size_suffix "$size")
    local out_root="${RESULTS_ROOT}/pcam/${bucket}"
    local out_dir="${out_root}/${model}_c8_gr${gr}_${mode}_seed${seed}${suffix}"
    local size_args=()
    local gate_size_args=()
    if [[ "$size" != "full" ]]; then
        size_args=(--train_size "$size")
        gate_size_args=(--train_size "$size")
    fi
    local common=(
        python train.py
        --model "$model"
        --group c8
        --geometry_group c8
        --growth_rate "$gr"
        --train_mode "$mode"
        --seed "$seed"
        --batch_size "$bs"
        --output_dir "$out_root"
        --data_dir ./pcam_data
        --subset_dir "$SUBSET_DIR"
        --patience 10
        --epochs 50
    )
    common+=("${size_args[@]}")
    (
        cd "${EXPERIMENT_DIR}/pcam"
        if [[ "${MODE}" == "check" ]]; then
            run_or_fail "${common[@]}" --memory_check --memory_manifest "$MEMORY_MANIFEST" --memory_headroom "$MEMORY_HEADROOM"
        elif ! maybe_skip "$out_dir"; then
            if ! manifest_gate \
                --dataset pcam \
                --model "$model" \
                --group c8 \
                --train_mode "$mode" \
                --batch_size "$bs" \
                --growth_rate "$gr" \
                "${gate_size_args[@]}"; then
                return 0
            fi
            run_or_fail "${common[@]}"
        fi
    )
}

run_organ() {
    local model=$1 mode=$2 size=$3 seed=$4
    local bs
    bs=$(organ_batch_size "$model")
    local bucket
    bucket=$(size_bucket "$size")
    local suffix
    suffix=$(size_suffix "$size")
    local out_root="${RESULTS_ROOT}/organ3d/${bucket}"
    local out_dir="${out_root}/${model}_${mode}_ch4_8_seed${seed}${suffix}"
    local size_args=()
    local gate_size_args=()
    if [[ "$size" != "full" ]]; then
        size_args=(--train_size "$size")
        gate_size_args=(--train_size "$size")
    fi
    local common=(
        python train.py
        --model "$model"
        --channels 4 8
        --head_dim 64
        --train_mode "$mode"
        --seed "$seed"
        --batch_size "$bs"
        --output_dir "$out_root"
        --data_dir ./organ3d_data
        --subset_dir "$SUBSET_DIR"
        --patience 15
        --epochs 100
    )
    common+=("${size_args[@]}")
    (
        cd "${EXPERIMENT_DIR}/organ3d"
        if [[ "${MODE}" == "check" ]]; then
            run_or_fail "${common[@]}" --memory_check --memory_manifest "$MEMORY_MANIFEST" --memory_headroom "$MEMORY_HEADROOM"
        elif ! maybe_skip "$out_dir"; then
            if ! manifest_gate \
                --dataset organ3d \
                --model "$model" \
                --train_mode "$mode" \
                --batch_size "$bs" \
                --channels 4 8 \
                --head_dim 64 \
                "${gate_size_args[@]}"; then
                return 0
            fi
            run_or_fail "${common[@]}"
        fi
    )
}

run_smnist() {
    local label=$1 mode=$2 size=$3 seed=$4
    local model=$label
    local hidden=256
    case "$label" in
        standard) model=standard; hidden=256 ;;
        power_spectrum) model=power_spectrum; hidden=256 ;;
        power_spectrum_matched) model=power_spectrum; hidden=1312 ;;
        bispectrum) model=bispectrum; hidden=256 ;;
        *) echo "unknown Spherical MNIST model: $label" >&2; exit 1 ;;
    esac
    local bs
    bs=$(smnist_batch_size "$label")
    local bucket
    bucket=$(size_bucket "$size")
    local suffix
    suffix=$(size_suffix "$size")
    local out_root="${RESULTS_ROOT}/spherical_mnist/${bucket}"
    local out_label="$model"
    if [[ "$label" == "power_spectrum_matched" ]]; then
        out_label="power_spectrum_matched"
    fi
    local out_dir="${out_root}/${out_label}_${mode}_seed${seed}${suffix}"
    local lmax=15
    local size_args=()
    local gate_size_args=()
    if [[ "$size" != "full" ]]; then
        size_args=(--train_size "$size")
        gate_size_args=(--train_size "$size")
    fi
    local common=(
        python train.py
        --model "$model"
        --run_label "$label"
        --train_mode "$mode"
        --seed "$seed"
        --hidden "$hidden"
        --lmax "$lmax"
        --batch_size "$bs"
        --output_dir "$out_root"
        --data_dir ./smnist_data
        --subset_dir "$SUBSET_DIR"
        --test_rotation_seed 777
        --patience 10
        --epochs 50
    )
    common+=("${size_args[@]}")
    (
        cd "${EXPERIMENT_DIR}/spherical_mnist"
        if [[ "${MODE}" == "check" ]]; then
            run_or_fail "${common[@]}" --memory_check --memory_manifest "$MEMORY_MANIFEST" --memory_headroom "$MEMORY_HEADROOM"
        elif ! maybe_skip "$out_dir"; then
            if ! manifest_gate \
                --dataset spherical_mnist \
                --model "$label" \
                --train_mode "$mode" \
                --batch_size "$bs" \
                --hidden "$hidden" \
                --lmax "$lmax" \
                "${gate_size_args[@]}"; then
                return 0
            fi
            run_or_fail "${common[@]}"
        fi
    )
}

main() {
    case "$MODE" in
        check|run|analyze) ;;
        -h|--help|help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac

    mkdir -p "$RESULTS_ROOT" "$SUBSET_DIR"
    activate_env

    if [[ "$MODE" == "check" ]]; then
        : > "$MEMORY_MANIFEST"
    fi

    if [[ "$MODE" == "analyze" ]]; then
        run_or_fail python "${EXPERIMENT_DIR}/analyze_normalized_results.py" \
            "${RESULTS_ROOT}/pcam" "${RESULTS_ROOT}/organ3d" "${RESULTS_ROOT}/spherical_mnist" \
            --min_seeds 5 \
            --output "${RESULTS_ROOT}/normalized_summary.json" \
            --fail_on_warn
        run_or_fail python "${EXPERIMENT_DIR}/export_normalized_latex.py" \
            --summary "${RESULTS_ROOT}/normalized_summary.json" \
            --output "${RESULTS_ROOT}/normalized_table.tex"
        exit 0
    fi

    local seeds_to_run=("${SEEDS[@]}")
    if [[ "$MODE" == "check" ]]; then
        seeds_to_run=("$CHECK_SEED")
        echo "Memory check mode uses CHECK_SEED=${CHECK_SEED}; seeds do not change tensor shapes."
    fi

    for seed in "${seeds_to_run[@]}"; do
        for mode in "${TRAIN_MODES[@]}"; do
            for size in "${PCAM_SIZES[@]}"; do
                for model in standard norm gate fourier_elu bispectrum; do
                    run_pcam "$model" "$mode" "$size" "$seed"
                done
            done
            for size in "${ORGAN_SIZES[@]}"; do
                for model in standard max_pool bispectrum; do
                    run_organ "$model" "$mode" "$size" "$seed"
                done
            done
            for size in "${SMNIST_SIZES[@]}"; do
                for model in standard power_spectrum power_spectrum_matched bispectrum; do
                    run_smnist "$model" "$mode" "$size" "$seed"
                done
            done
        done
    done
}

main "$@"
