#!/usr/bin/env python3
"""Training script for the PCam equivariant DenseNet experiment.

Usage examples:

    # Standard CNN baseline with geometric augmentation
    python train.py --model standard --data_dir ./pcam_data

    # Equivariant DenseNet with bispectral invariant pooling (C8)
    python train.py --model bispectrum --group c8 --data_dir ./pcam_data

    # Equivariant DenseNet with norm nonlinearity (D4)
    python train.py --model norm --group d4 --data_dir ./pcam_data

    # Data-efficiency experiment: 2500 training examples
    python train.py --model bispectrum --group c8 --train_size 2500

    # SO2onDisk disk bispectrum baseline (no backbone)
    python train.py --model so2_disk --bandlimit 30 --data_dir ./pcam_data

    # Run all 6 baselines × 3 seeds (full sweep)
    python train.py --sweep --data_dir ./pcam_data
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
from data import apply_group_element, get_dataloaders, gpu_augment, gpu_normalize_only
from model import build_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5, check=False,
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


def compute_metrics(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Compute accuracy, AUC, and loss on a dataset."""
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    n = 0

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            images = gpu_normalize_only(images)
            labels = labels.to(device, non_blocking=True).float()
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='sum')
            total_loss += loss.item()
            n += labels.shape[0]

            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    # Accuracy
    preds = (all_probs > 0.5).long()
    accuracy = (preds == all_labels.long()).float().mean().item()

    # AUC (manual computation to avoid sklearn dependency)
    auc = _compute_auc(all_labels, all_probs)

    return {
        'loss': total_loss / n,
        'accuracy': accuracy,
        'auc': auc,
    }


def _compute_auc(labels: torch.Tensor, probs: torch.Tensor) -> float:
    """Compute AUC-ROC without sklearn (vectorised Wilcoxon-Mann-Whitney)."""
    sorted_indices = probs.argsort(descending=True)
    sorted_labels = labels[sorted_indices].long()

    n_pos = sorted_labels.sum().item()
    n_neg = len(sorted_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp_cumsum = sorted_labels.cumsum(0)
    neg_mask = sorted_labels == 0
    auc_sum = tp_cumsum[neg_mask].sum().item()
    return auc_sum / (n_pos * n_neg)


def evaluate_rotation_robustness(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    geometry_group: str = 'c8',
) -> dict:
    """Evaluate model consistency across all finite-group test transforms."""
    n_elements = 8

    model.eval()
    results = {}

    for element_idx in range(n_elements):
        all_probs = []
        all_labels = []
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                images = apply_group_element(images, geometry_group, element_idx)
                images = gpu_normalize_only(images)
                logits = model(images)
                all_probs.append(torch.sigmoid(logits).cpu())
                all_labels.append(labels)

        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels).float()
        probs_binary = (all_probs > 0.5).long()
        accuracy = (probs_binary == all_labels.long()).float().mean().item()
        results[f'g{element_idx:02d}'] = {
            'accuracy': accuracy,
            'auc': _compute_auc(all_labels, all_probs),
        }

    aucs = [v['auc'] for v in results.values()]
    accs = [v['accuracy'] for v in results.values()]
    results['mean_auc'] = sum(aucs) / len(aucs)
    results['std_auc'] = (sum((a - results['mean_auc']) ** 2 for a in aucs) / len(aucs)) ** 0.5
    results['mean_accuracy'] = sum(accs) / len(accs)
    results['std_accuracy'] = (
        sum((a - results['mean_accuracy']) ** 2 for a in accs) / len(accs)
    ) ** 0.5

    return results


def _mean_group_metrics(rotation_metrics: dict) -> dict[str, float]:
    return {
        'loss': 0.0,
        'accuracy': float(rotation_metrics.get('mean_accuracy', 0.0)),
        'auc': float(rotation_metrics.get('mean_auc', 0.0)),
    }


def _invariant_dim(model: torch.nn.Module) -> int | None:
    pool = getattr(model, 'invariant_pool', None)
    features_per_channel = getattr(pool, 'features_per_channel', None)
    channels = getattr(model, '_channels_before_fc', None)
    if isinstance(features_per_channel, int) and isinstance(channels, int):
        return features_per_channel * channels
    classifier = getattr(model, 'classifier', None)
    in_features = getattr(classifier, 'in_features', None)
    return int(in_features) if isinstance(in_features, int) else None


def memory_check(args: argparse.Namespace) -> dict:
    """Run a one-batch forward/backward pass and record peak CUDA memory."""
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(
        nonlin_type=args.model,
        group=args.group,
        growth_rate=args.growth_rate,
        block_config=tuple(args.block_config),
        bandlimit=getattr(args, 'bandlimit', None),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    invariant_dim = _invariant_dim(model)
    geometry_group = args.geometry_group or args.group
    n_tag = f'_n{args.train_size}' if args.train_size and args.train_size > 0 else ''
    output_dir = (
        Path(args.output_dir)
        / f'{args.model}_{args.group}_gr{args.growth_rate}_{args.train_mode}_seed{args.seed}{n_tag}'
    )
    train_loader, _, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        augment_geometry=args.train_mode == 'R',
        train_size=args.train_size,
        seed=args.seed,
        subset_dir=args.subset_dir,
    )

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
        images = gpu_augment(
            images.to(device, non_blocking=True),
            args.train_mode == 'R',
            geometry_group=geometry_group,
        )
        labels = labels.to(device, non_blocking=True).float()
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
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
        'dataset': 'pcam',
        'model': args.model,
        'group': args.group,
        'train_mode': args.train_mode,
        'train_size': args.train_size,
        'train_examples': len(train_loader.dataset),
        'seed': args.seed,
        'batch_size': args.batch_size,
        'effective_batch_size': args.batch_size,
        'growth_rate': args.growth_rate,
        'output_dir': str(output_dir),
        'n_params': n_params,
        'invariant_dim': invariant_dim,
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


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    augment_geometry: bool = False,
    geometry_group: str = 'd4',
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    num_batches = len(loader)
    t_epoch = time.time()
    for step, (images, labels) in enumerate(loader):
        images = gpu_augment(
            images.to(device, non_blocking=True),
            augment_geometry,
            geometry_group=geometry_group,
        )
        labels = labels.to(device, non_blocking=True).float()

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * labels.shape[0]
        n += labels.shape[0]

        if (step + 1) % 100 == 0 or step == 0:
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


def train(args: argparse.Namespace) -> dict:
    """Run a single training experiment.

    Returns final metrics dict.
    """
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(
        nonlin_type=args.model,
        group=args.group,
        growth_rate=args.growth_rate,
        block_config=tuple(args.block_config),
        bandlimit=getattr(args, 'bandlimit', None),
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    invariant_dim = _invariant_dim(model)
    if args.model == 'so2_disk':
        print(f'Model: {args.model} (bandlimit={args.bandlimit}), {n_params:,} params')
    else:
        print(f'Model: {args.model} (group={args.group}, gr={args.growth_rate}), {n_params:,} params')

    if args.dry_run:
        info: dict = {
            'model': args.model,
            'train_mode': args.train_mode,
            'n_params': n_params,
            'invariant_dim': invariant_dim,
        }
        if args.model == 'so2_disk':
            info['bandlimit'] = args.bandlimit
        else:
            info.update({'group': args.group, 'growth_rate': args.growth_rate})
        print(json.dumps(info))
        return {'n_params': n_params}

    print(f'Device: {device}')

    augment_geo = args.train_mode == 'R'
    geometry_group = args.geometry_group or args.group
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        augment_geometry=augment_geo,
        train_size=args.train_size,
        seed=args.seed,
        subset_dir=args.subset_dir,
    )
    print(
        f'Data: train={len(train_loader.dataset)}, '
        f'val={len(val_loader.dataset)}, test={len(test_loader.dataset)}'
    )

    if args.compile:
        model = torch.compile(model)

    torch.set_float32_matmul_precision('high')

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda')
    steps_per_epoch = len(train_loader)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10 * steps_per_epoch, T_mult=2)

    best_auc = 0.0
    patience_counter = 0
    n_tag = f'_n{args.train_size}' if args.train_size and args.train_size > 0 else ''
    if args.model == 'so2_disk':
        out_dir = Path(args.output_dir) / f'{args.model}_bl{args.bandlimit:.0f}_{args.train_mode}_seed{args.seed}{n_tag}'
    else:
        out_dir = Path(args.output_dir) / f'{args.model}_{args.group}_gr{args.growth_rate}_{args.train_mode}_seed{args.seed}{n_tag}'
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.time()
    epochs_run = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler,
            augment_geometry=augment_geo,
            geometry_group=geometry_group,
        )
        val_metrics = compute_metrics(model, val_loader, device)
        elapsed = time.time() - t0
        epochs_run = epoch

        print(
            f'Epoch {epoch:3d}/{args.epochs} | '
            f'train_loss={train_loss:.4f} | '
            f'val_loss={val_metrics["loss"]:.4f} | '
            f'val_acc={val_metrics["accuracy"]:.4f} | '
            f'val_auc={val_metrics["auc"]:.4f} | '
            f'{elapsed:.1f}s'
        )

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / 'best_model.pt')
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f'Early stopping at epoch {epoch} (patience={args.patience})')
            break

    cost = _cost_record(device, time.time() - t_start, epochs_run)

    model.load_state_dict(torch.load(out_dir / 'best_model.pt', weights_only=True))
    test_c = compute_metrics(model, test_loader, device)
    print(f'\nTest C results: {test_c}')

    if not args.skip_rotation:
        rot_metrics = evaluate_rotation_robustness(model, test_loader, device, geometry_group)
        print(
            f'Rotation robustness: mean_auc={rot_metrics["mean_auc"]:.4f}, '
            f'std_auc={rot_metrics["std_auc"]:.4f}'
        )
        test_r = _mean_group_metrics(rot_metrics)
    else:
        rot_metrics = {}
        test_r = {}

    results: dict = {
        'dataset': 'pcam',
        'model': args.model,
        'group': args.group,
        'seed': args.seed,
        'train_mode': args.train_mode,
        'geometry_group': geometry_group,
        'growth_rate': args.growth_rate,
        'n_params': n_params,
        'invariant_dim': invariant_dim,
        'train_size': args.train_size,
        'train_examples': len(train_loader.dataset),
        'best_val_auc': best_auc,
        'test': test_c,
        'test_c': test_c,
        'test_r': test_r,
        'rotation_robustness': rot_metrics,
        **cost,
    }
    if args.model == 'so2_disk':
        results['bandlimit'] = args.bandlimit
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_sweep(args: argparse.Namespace):
    """Run the full experimental sweep."""
    models = ['standard', 'norm', 'gate', 'fourier_elu', 'bispectrum']
    train_modes = ['C', 'R']
    seeds = [42, 123, 456]
    all_results = []

    for model_name in models:
        for train_mode in train_modes:
            for seed in seeds:
                args.model = model_name
                args.train_mode = train_mode
                args.seed = seed
                print(f'\n{"=" * 60}')
                print(
                    f'Running: model={model_name}, group={args.group}, '
                    f'train_mode={train_mode}, seed={seed}'
                )
                print(f'{"=" * 60}')
                results = train(args)
                all_results.append(results)

    # Summary table.
    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"=" * 60}')
    print(f'{"Model":<15} {"Params":>10} {"Test AUC":>10} {"Test Acc":>10} {"Rot Std":>10}')
    print('-' * 60)

    grouped = defaultdict(list)
    for r in all_results:
        grouped[(r['model'], r['train_mode'])].append(r)

    for model_name in models:
        for train_mode in train_modes:
            runs = grouped[(model_name, train_mode)]
            aucs = [r['test_c']['auc'] for r in runs]
            accs = [r['test_c']['accuracy'] for r in runs]
            rot_stds = [
                r['rotation_robustness'].get('std_auc', 0.0)
                for r in runs
                if r.get('rotation_robustness')
            ]
            n_params = runs[0]['n_params']

            mean_auc = sum(aucs) / len(aucs)
            std_auc = (sum((a - mean_auc) ** 2 for a in aucs) / len(aucs)) ** 0.5
            mean_acc = sum(accs) / len(accs)
            mean_rot = sum(rot_stds) / max(len(rot_stds), 1)

            print(
                f'{model_name + "/" + train_mode:<15} {n_params:>10,} '
                f'{mean_auc:.4f}±{std_auc:.4f} '
                f'{mean_acc:.4f}       '
                f'{mean_rot:.6f}'
            )

    # Save all results.
    out_path = Path(args.output_dir) / 'sweep_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nFull results saved to {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='PCam equivariant DenseNet experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--model',
        choices=['standard', 'norm', 'gate', 'fourier_elu', 'bispectrum', 'so2_disk'],
        default='bispectrum',
        help='Nonlinearity / model variant.',
    )
    parser.add_argument('--group', choices=['c8', 'd4'], default='c8')
    parser.add_argument('--data_dir', '--data-dir', type=str, default='./pcam_data')
    parser.add_argument('--output_dir', '--output-dir', type=str, default='./pcam_results')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--growth_rate', type=int, default=12)
    parser.add_argument(
        '--bandlimit',
        type=float,
        default=30.0,
        help='Bandlimit for SO2onDisk (only used when --model=so2_disk).',
    )
    parser.add_argument(
        '--block_config',
        type=int,
        nargs='+',
        default=[4, 4, 4],
        help='Number of layers in each dense block.',
    )
    parser.add_argument(
        '--train_size',
        type=int,
        default=None,
        help='Subset training data to N examples for data-efficiency experiments. '
             'Omit (or pass <=0) to use the full training set.',
    )
    parser.add_argument(
        '--train_mode',
        choices=['C', 'R'],
        default='C',
        help='Training protocol: C=canonical, R=random finite-group augmentation.',
    )
    parser.add_argument(
        '--geometry_group',
        choices=['c8', 'd4'],
        default=None,
        help='Group used for train/test geometric transforms. Defaults to --group.',
    )
    parser.add_argument(
        '--subset_dir',
        type=str,
        default=None,
        help='Directory where train-subset index manifests are written.',
    )
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='Run all 6 baselines × 3 seeds.',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Build model, print param count, and exit (no data loading or training).',
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
        help='Skip the 12-angle rotation robustness evaluation.',
    )
    parser.add_argument(
        '--compile',
        action='store_true',
        help='Use torch.compile for faster training (H100/A100).',
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
