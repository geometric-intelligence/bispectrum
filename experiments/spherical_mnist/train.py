#!/usr/bin/env python3
"""Training script for the Spherical MNIST experiment.

Implements the Cohen et al. (2018) evaluation protocol:
  - Train on non-rotated (NR) or rotated (R) spherical MNIST
  - Test on both NR and R test sets
  - Report NR/NR, R/R, NR/R accuracy table

Usage examples:

    # Bispectrum model, trained on non-rotated data
    python train.py --model bispectrum --train_mode NR

    # Standard CNN, trained on rotated data
    python train.py --model standard --train_mode R

    # Full sweep: 3 models x 2 modes x 3 seeds
    python train.py --sweep
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from data import get_dataloaders
from model import build_model

from bispectrum import random_rotation_matrix, rotate_spherical_function

NUM_CLASSES = 10


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ''
    except (OSError, subprocess.SubprocessError):
        return ''


def _cost_record(device: torch.device, wall_time_s: float, epochs_run: int) -> dict:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        return {
            'wall_time_s': float(wall_time_s),
            'epochs_run': int(epochs_run),
            'peak_train_alloc_bytes': int(torch.cuda.max_memory_allocated(device)),
            'peak_train_reserved_bytes': int(torch.cuda.max_memory_reserved(device)),
            'gpu_name': torch.cuda.get_device_name(device),
            'cuda_total_memory_bytes': int(props.total_memory),
            'git_sha': _git_sha(),
            'argv': list(sys.argv[1:]),
        }
    return {
        'wall_time_s': float(wall_time_s),
        'epochs_run': int(epochs_run),
        'peak_train_alloc_bytes': 0,
        'peak_train_reserved_bytes': 0,
        'gpu_name': None,
        'cuda_total_memory_bytes': 0,
        'git_sha': _git_sha(),
        'argv': list(sys.argv[1:]),
    }


def _canonical_train_mode(train_mode: str) -> str:
    return 'C' if train_mode in {'C', 'NR'} else 'R'


def _invariant_dim(model: torch.nn.Module) -> int | None:
    bispectrum = getattr(model, 'bispectrum', None)
    if bispectrum is not None:
        return int(getattr(bispectrum, 'output_size', 0)) * 2
    lmax = getattr(model, 'lmax', None)
    if isinstance(lmax, int):
        return lmax + 1
    return None


def compute_metrics(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Compute accuracy and loss on a dataset."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction='sum')
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]

    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
    }


def evaluate_rotation_robustness(
    model: torch.nn.Module,
    test_dataset,
    device: torch.device,
    n_rotations: int = 10,
    batch_size: int = 256,
) -> dict:
    """Evaluate model consistency under random SO(3) rotations.

    Applies the SAME rotation to ALL test images for each trial, testing whether predictions are
    truly rotation-invariant.
    """
    model.eval()
    images = test_dataset.images
    labels = test_dataset.labels
    results: dict = {}

    for rot_idx in range(n_rotations):
        torch.manual_seed(77777 + rot_idx)
        R = random_rotation_matrix()

        correct = 0
        total = 0

        for start in range(0, len(images), batch_size):
            batch_imgs = images[start : start + batch_size]
            batch_lbls = labels[start : start + batch_size]

            rotated = rotate_spherical_function(batch_imgs, R)
            rotated = rotated.to(device, dtype=torch.float32)
            batch_lbls = batch_lbls.to(device)

            with torch.no_grad(), torch.amp.autocast('cuda'):
                logits = model(rotated)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_lbls).sum().item()
            total += batch_lbls.shape[0]

        results[f'rot_{rot_idx:02d}'] = correct / total

    accs = [v for k, v in results.items() if k.startswith('rot_')]
    results['mean_accuracy'] = sum(accs) / len(accs)
    results['std_accuracy'] = (
        sum((a - results['mean_accuracy']) ** 2 for a in accs) / len(accs)
    ) ** 0.5
    return results


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    num_batches = len(loader)
    t_epoch = time.time()

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * labels.shape[0]
        n += labels.shape[0]

        if (step + 1) % 50 == 0 or step == 0:
            elapsed = time.time() - t_epoch
            ms_per_step = elapsed / (step + 1) * 1000
            eta = (num_batches - step - 1) * elapsed / (step + 1)
            print(
                f'  step {step + 1}/{num_batches} | '
                f'loss={loss.item():.4f} | '
                f'{ms_per_step:.0f} ms/step | '
                f'ETA {eta:.0f}s',
                end='\r',
            )

    print(' ' * 80, end='\r')
    return total_loss / n


def memory_check(args: argparse.Namespace) -> dict:
    """Run a one-batch forward/backward pass and record peak CUDA memory."""
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_mode = _canonical_train_mode(args.train_mode)
    train_loader, _, _, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        nlat=args.nlat,
        nlon=args.nlon,
        train_rotated=train_mode == 'R',
        train_size=args.train_size,
        seed=args.seed,
        test_rotation_seed=args.test_rotation_seed,
        subset_dir=args.subset_dir,
    )
    model = build_model(
        model_type=args.model,
        lmax=args.lmax,
        nlat=args.nlat,
        nlon=args.nlon,
        selective=not args.full_bispectrum,
        hidden=args.hidden,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    invariant_dim = _invariant_dim(model)
    n_tag = f'_n{args.train_size}' if args.train_size and args.train_size > 0 else ''
    run_label = args.run_label or args.model
    output_dir = Path(args.output_dir) / f'{run_label}_{train_mode}_seed{args.seed}{n_tag}'

    status = 'ok'
    error = ''
    allocated = 0
    reserved = 0
    total = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        total = torch.cuda.get_device_properties(device).total_memory

    try:
        model.train()
        images, labels = next(iter(train_loader))
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        if torch.cuda.is_available():
            allocated = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
            if total and reserved / total > args.memory_headroom:
                status = 'fail'
                error = f'reserved memory exceeds {args.memory_headroom:.0%} headroom'
    except RuntimeError as exc:
        status = 'fail'
        error = str(exc)

    record = {
        'dataset': 'spherical_mnist',
        'model': args.run_label or args.model,
        'base_model': args.model,
        'train_mode': train_mode,
        'train_size': args.train_size,
        'train_examples': len(train_loader.dataset),
        'seed': args.seed,
        'batch_size': args.batch_size,
        'effective_batch_size': args.batch_size,
        'lmax': args.lmax,
        'hidden': args.hidden,
        'output_dir': str(output_dir),
        'n_params': n_params,
        'invariant_dim': invariant_dim,
        'test_rotation_seed': args.test_rotation_seed,
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
        'total_memory_bytes': total,
        'max_memory_allocated_bytes': allocated,
        'max_memory_reserved_bytes': reserved,
        'memory_headroom': args.memory_headroom,
        'status': status,
        'error': error,
    }
    print(json.dumps(record))
    if args.memory_manifest:
        manifest = Path(args.memory_manifest)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open('a') as f:
            f.write(json.dumps(record) + '\n')
    if status != 'ok':
        raise SystemExit(1)
    return record


def train(args: argparse.Namespace) -> dict:
    """Run a single training experiment.

    Returns final metrics dict.
    """
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(
        model_type=args.model,
        lmax=args.lmax,
        nlat=args.nlat,
        nlon=args.nlon,
        selective=not args.full_bispectrum,
        hidden=args.hidden,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    invariant_dim = _invariant_dim(model)
    print(f'Model: {args.model} (lmax={args.lmax}), {n_params:,} params')

    if args.dry_run:
        print(
            json.dumps(
                {
                    'model': args.run_label or args.model,
                    'base_model': args.model,
                    'lmax': args.lmax,
                    'train_mode': _canonical_train_mode(args.train_mode),
                    'n_params': n_params,
                    'invariant_dim': invariant_dim,
                }
            )
        )
        return {'n_params': n_params}

    train_mode = _canonical_train_mode(args.train_mode)
    train_rotated = train_mode == 'R'
    train_loader, val_loader, test_nr_loader, test_r_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        nlat=args.nlat,
        nlon=args.nlon,
        train_rotated=train_rotated,
        train_size=args.train_size,
        seed=args.seed,
        test_rotation_seed=args.test_rotation_seed,
        subset_dir=args.subset_dir,
    )

    print(f'Device: {device}')
    print(
        f'Data: train={len(train_loader.dataset)}, '
        f'val={len(val_loader.dataset)}, '
        f'test_nr={len(test_nr_loader.dataset)}, '
        f'test_r={len(test_r_loader.dataset)}'
    )

    torch.set_float32_matmul_precision('high')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler('cuda')
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10 * steps_per_epoch,
        T_mult=2,
    )

    best_val_acc = 0.0
    patience_counter = 0
    n_tag = f'_n{args.train_size}' if args.train_size and args.train_size > 0 else ''
    run_label = args.run_label or args.model
    out_dir = Path(args.output_dir) / f'{run_label}_{train_mode}_seed{args.seed}{n_tag}'
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.time()
    epochs_run = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            scaler,
        )
        val_metrics = compute_metrics(model, val_loader, device)
        elapsed = time.time() - t0
        epochs_run = epoch

        print(
            f'Epoch {epoch:3d}/{args.epochs} | '
            f'train_loss={train_loss:.4f} | '
            f'val_loss={val_metrics["loss"]:.4f} | '
            f'val_acc={val_metrics["accuracy"]:.4f} | '
            f'{elapsed:.1f}s'
        )

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / 'best_model.pt')
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f'Early stopping at epoch {epoch} (patience={args.patience})')
            break

    cost = _cost_record(device, time.time() - t_start, epochs_run)

    model.load_state_dict(torch.load(out_dir / 'best_model.pt', weights_only=True))

    test_nr = compute_metrics(model, test_nr_loader, device)
    test_r = compute_metrics(model, test_r_loader, device)
    print(f'\nTest NR: {test_nr}')
    print(f'Test R:  {test_r}')

    rot_robustness: dict = {}
    if not args.skip_rotation:
        test_nr_ds = test_nr_loader.dataset
        rot_robustness = evaluate_rotation_robustness(
            model,
            test_nr_ds,
            device,
            n_rotations=args.n_rotations,
        )
        print(
            f'Rotation robustness: mean_acc={rot_robustness["mean_accuracy"]:.4f}, '
            f'std_acc={rot_robustness["std_accuracy"]:.4f}'
        )

    results = {
        'dataset': 'spherical_mnist',
        'model': run_label,
        'base_model': args.model,
        'train_mode': train_mode,
        'test_rotation_seed': args.test_rotation_seed,
        'seed': args.seed,
        'lmax': args.lmax,
        'n_params': n_params,
        'invariant_dim': invariant_dim,
        'train_size': args.train_size,
        'train_examples': len(train_loader.dataset),
        'best_val_accuracy': best_val_acc,
        'test_c': test_nr,
        'test': test_nr,
        'test_nr': test_nr,
        'test_r': test_r,
        'rotation_robustness': rot_robustness,
        **cost,
    }
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_sweep(args: argparse.Namespace):
    """Run the full experimental sweep: 3 models x 2 modes x 3 seeds."""
    models = ['standard', 'power_spectrum', 'bispectrum']
    train_modes = ['C', 'R']
    seeds = [42, 123, 456]
    all_results = []

    for model_name in models:
        for mode in train_modes:
            for seed in seeds:
                args.model = model_name
                args.train_mode = mode
                args.seed = seed
                print(f'\n{"=" * 60}')
                print(f'Running: model={model_name}, mode={mode}, seed={seed}')
                print(f'{"=" * 60}')
                results = train(args)
                all_results.append(results)

    print(f'\n{"=" * 70}')
    print('SUMMARY (canonical/random protocol)')
    print(f'{"=" * 70}')
    print(f'{"Model":<18} {"Params":>8} {"C/C":>8} {"R/R":>8} {"C/R":>8} {"Rot \u03c3":>8}')
    print('-' * 70)

    grouped = defaultdict(list)
    for r in all_results:
        grouped[(r['model'], r['train_mode'])].append(r)

    for model_name in models:
        nr_runs = grouped.get((model_name, 'C'), [])
        r_runs = grouped.get((model_name, 'R'), [])

        nr_nr = sum(r['test_nr']['accuracy'] for r in nr_runs) / max(len(nr_runs), 1)
        r_r = sum(r['test_r']['accuracy'] for r in r_runs) / max(len(r_runs), 1)
        nr_r = sum(r['test_r']['accuracy'] for r in nr_runs) / max(len(nr_runs), 1)

        rot_stds = [
            r['rotation_robustness'].get('std_accuracy', 0)
            for r in nr_runs
            if r.get('rotation_robustness')
        ]
        rot_std = sum(rot_stds) / max(len(rot_stds), 1)

        n_params = nr_runs[0]['n_params'] if nr_runs else 0

        print(
            f'{model_name:<18} {n_params:>8,} '
            f'{nr_nr:>8.4f} {r_r:>8.4f} {nr_r:>8.4f} {rot_std:>8.6f}'
        )

    out_path = Path(args.output_dir) / 'sweep_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nFull results saved to {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Spherical MNIST experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--model',
        choices=['standard', 'power_spectrum', 'bispectrum'],
        default='bispectrum',
    )
    parser.add_argument(
        '--run_label',
        type=str,
        default=None,
        help='Optional result label when the same base model is run at another budget.',
    )
    parser.add_argument(
        '--train_mode',
        choices=['C', 'NR', 'R'],
        default='C',
        help='Training data variant: C/NR=canonical/non-rotated, R=rotated.',
    )
    parser.add_argument('--data_dir', type=str, default='./smnist_data')
    parser.add_argument('--output_dir', type=str, default='./smnist_results')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lmax', type=int, default=15)
    parser.add_argument('--nlat', type=int, default=64)
    parser.add_argument('--nlon', type=int, default=128)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument(
        '--train_size',
        type=int,
        default=None,
        help='Subset training data to N examples for data-efficiency experiments. '
        'Omit (or pass <=0) to use the full training set.',
    )
    parser.add_argument(
        '--test_rotation_seed',
        type=int,
        default=777,
        help='Fixed seed for the rotated test-set cache shared across model seeds.',
    )
    parser.add_argument(
        '--subset_dir',
        type=str,
        default=None,
        help='Directory where train-subset index manifests are written.',
    )
    parser.add_argument(
        '--full_bispectrum',
        action='store_true',
        help='Use full O(L^3) bispectrum instead of selective O(L^2).',
    )
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='Run all 3 models x 2 modes x 3 seeds.',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Build model, print param count, and exit.',
    )
    parser.add_argument(
        '--memory_check',
        action='store_true',
        help='Run a one-batch forward/backward memory check and exit.',
    )
    parser.add_argument(
        '--memory_manifest',
        type=str,
        default=None,
        help='Optional JSONL path for memory_check records.',
    )
    parser.add_argument(
        '--memory_headroom',
        type=float,
        default=0.85,
        help='Fail memory_check when reserved CUDA memory exceeds this fraction.',
    )
    parser.add_argument(
        '--skip_rotation',
        action='store_true',
        help='Skip rotation robustness evaluation.',
    )
    parser.add_argument(
        '--n_rotations',
        type=int,
        default=10,
        help='Number of random rotations for robustness eval.',
    )

    args = parser.parse_args()

    if args.memory_check:
        memory_check(args)
    elif args.sweep:
        run_sweep(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
