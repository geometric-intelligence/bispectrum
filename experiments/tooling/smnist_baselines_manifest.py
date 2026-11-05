#!/usr/bin/env python3
"""Job manifests for the Spherical MNIST e3nn baselines (camera-ready E5/E6).

    tune             one C-trained seed per S2CNN+BN and tensor-product config, plus the faithful
                     S2CNN (Cohen et al. optimizer: Adam 5e-3, batch 32) at every seed and train
                     mode
    final            the best-validation S2CNN+BN and tensor-product configs at every seed and
                     train mode; writes the selection next to the manifest
    data_efficiency  the selected configs, and the bispectrum's (batch 32, lr 1e-3), at every
                     training-set size and seed: NR-trained, plus R-trained S2CNN+BN (the only
                     model that is not exactly invariant)

All manifests feed ``gpu_job_runner.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CWD = 'experiments/spherical_mnist'
TUNE_DIR = './smnist_results_baselines_tuning'
FINAL_DIR = './smnist_results_baselines'
DATA_DIR = './smnist_results_baselines_data'
SEEDS = (42, 123, 456, 789, 101112)
TRAIN_MODES = ('C', 'R')
TUNE_SEED = 42
# Same sizes as experiments/spherical_mnist/run_data_efficiency.sh; full data is the final stage.
DATA_SIZES = (100, 500, 2500, 12500)
FULL_TRAIN = 50_000
MIN_EPOCH_SAMPLES = 10_000  # train.py default
VAL_SIZE = 10_000
COMMON = ('--epochs', '30', '--data_dir', './smnist_data')
MACHINES: dict[str, dict[str, Any]] = {
    'l8a100': {'gpus': 8, 'gpu_mem_gib': 39.0, 'max_workers_per_gpu': 3, 'speed': 1.0},
}
MEM_GIB = 3.0

S2CNN_COHEN: tuple[str, ...] = ('--model', 's2cnn', '--batch_size', '32', '--lr', '5e-3')
# Selected on the H100 from {32, 256} x {1e-3, 3e-3, 5e-3} (smnist_results_bispectrum_tuned).
BISPECTRUM_TUNED: tuple[str, ...] = ('--model', 'bispectrum', '--batch_size', '32', '--lr', '1e-3')
FAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    's2cnn_bn': {
        f's2cnn_bn_bs{bs}_lr{lr}': (
            '--model',
            's2cnn',
            '--s2cnn_batch_norm',
            '--batch_size',
            str(bs),
            '--lr',
            lr,
        )
        for bs in (32, 256)
        for lr in ('1e-3', '5e-3')
    },
    'tensor_product': {
        f'tp_c{c}_l{lmax}_lr{lr}': (
            '--model',
            'tensor_product',
            '--tp_channels',
            str(c),
            '--tp_lmax',
            str(lmax),
            '--lr',
            lr,
        )
        for c, lmax in ((16, 4), (16, 6), (32, 4), (24, 6))
        for lr in ('1e-3', '3e-3')
    },
}


def est_hours(flags: tuple[str, ...], train_size: int | None = None) -> float:
    if 'tensor_product' in flags:
        channels = int(flags[flags.index('--tp_channels') + 1])
        full = 1.0 if channels <= 16 else 1.5
    else:
        full = 0.35 if flags[flags.index('--batch_size') + 1] == '32' else 0.15
    if train_size is None:
        return full
    # Per epoch: max(n, floor) training samples (~3x a forward) plus the validation pass.
    work = 3 * max(train_size, MIN_EPOCH_SAMPLES) + VAL_SIZE
    return round(full * work / (3 * FULL_TRAIN + VAL_SIZE), 3)


def job(
    run_label: str,
    flags: tuple[str, ...],
    mode: str,
    seed: int,
    output_dir: str,
    train_size: int | None = None,
) -> dict[str, Any]:
    subset = ('--train_size', str(train_size), '--skip_rotation') if train_size else ()
    argv = [
        'train.py',
        *flags,
        '--run_label',
        run_label,
        '--train_mode',
        mode,
        '--seed',
        str(seed),
        '--output_dir',
        output_dir,
        *subset,
        *COMMON,
    ]
    digest = hashlib.sha1(f'{CWD}\0{" ".join(argv)}'.encode(), usedforsecurity=False)
    n_tag = f'_n{train_size}' if train_size else ''
    return {
        'id': digest.hexdigest()[:12],
        'label': f'{run_label}_{mode}_seed{seed}{n_tag}',
        'cwd': CWD,
        'argv': argv,
        'mem_gib': MEM_GIB,
        'est_hours': est_hours(flags, train_size),
        'machine': 'l8a100',
        'pipeline': 'smnist_baselines',
    }


def every_seed(run_label: str, flags: tuple[str, ...]) -> list[dict[str, Any]]:
    return [job(run_label, flags, mode, seed, FINAL_DIR) for mode in TRAIN_MODES for seed in SEEDS]


def tune_jobs() -> list[dict[str, Any]]:
    jobs = [
        job(label, flags, 'C', TUNE_SEED, TUNE_DIR)
        for configs in FAMILIES.values()
        for label, flags in configs.items()
    ]
    return jobs + every_seed('s2cnn', S2CNN_COHEN)


def data_efficiency_jobs(selection: dict[str, Any]) -> list[dict[str, Any]]:
    """Every config is the one selected at full data, reused unchanged at every size."""
    families: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
        ('s2cnn_bn', tuple(selection['s2cnn_bn']['flags']), TRAIN_MODES),
        ('tensor_product', tuple(selection['tensor_product']['flags']), ('C',)),
        ('bispectrum_bs32_lr1e-3', BISPECTRUM_TUNED, ('C',)),
    ]
    return [
        job(label, flags, mode, seed, DATA_DIR, train_size=size)
        for label, flags, modes in families
        for size in DATA_SIZES
        for mode in modes
        for seed in SEEDS
    ]


def select(tune_root: Path) -> dict[str, Any]:
    """Best ``best_val_accuracy`` per family over its tuning runs (all must be finished)."""
    selection: dict[str, Any] = {}
    for family, configs in FAMILIES.items():
        scores: dict[str, float] = {}
        for label in configs:
            path = tune_root / f'{label}_C_seed{TUNE_SEED}' / 'results.json'
            if not path.exists():
                raise SystemExit(f'missing tuning result: {path}')
            scores[label] = float(json.loads(path.read_text())['best_val_accuracy'])
        best = max(scores, key=scores.__getitem__)
        selection[family] = {
            'selected': best,
            'flags': list(configs[best]),
            'val_accuracy': scores,
        }
    return selection


def write(path: Path, jobs: list[dict[str, Any]], stage: str) -> None:
    manifest = {
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'stage': stage,
        'machines': {
            m: {**spec, 'slots': spec['gpus'] * spec['max_workers_per_gpu']}
            for m, spec in MACHINES.items()
        },
        'jobs': jobs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(manifest, indent=1) + '\n')
    tmp.replace(path)
    print(f'{path}: {len(jobs)} jobs')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest='cmd', required=True)
    tune = sub.add_parser('tune')
    tune.add_argument('--out', type=Path, required=True)
    final = sub.add_parser('final')
    final.add_argument('--out', type=Path, required=True)
    final.add_argument('--tune_root', type=Path, default=REPO_ROOT / CWD / TUNE_DIR)
    data = sub.add_parser('data_efficiency')
    data.add_argument('--out', type=Path, required=True)
    data.add_argument(
        '--selection', type=Path, default=None, help='default: selection.json next to --out'
    )
    args = parser.parse_args(argv)

    if args.cmd == 'tune':
        write(args.out, tune_jobs(), 'tune')
        return 0
    if args.cmd == 'data_efficiency':
        selection_path = args.selection or args.out.parent / 'selection.json'
        selection = json.loads(selection_path.read_text())
        write(args.out, data_efficiency_jobs(selection), 'data_efficiency')
        return 0
    selection = select(args.tune_root)
    (args.out.parent / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    for family, chosen in selection.items():
        print(
            f'{family}: {chosen["selected"]} (val {chosen["val_accuracy"][chosen["selected"]]:.4f})'
        )
    jobs = [
        j
        for family, chosen in selection.items()
        for j in every_seed(family, tuple(chosen['flags']))
    ]
    write(args.out, jobs, 'final')
    return 0


if __name__ == '__main__':
    sys.exit(main())
