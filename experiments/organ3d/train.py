#!/usr/bin/env python3
"""Training script for the OrganMNIST3D equivariant 3D ResNet experiment.

Usage examples:

    # Standard CNN baseline with geometric augmentation
    python train.py --model standard --data_dir ./organ3d_data

    # Equivariant 3D ResNet with bispectral invariant pooling
    python train.py --model bispectrum --data_dir ./organ3d_data

    # Equivariant 3D ResNet with max pooling over group dim
    python train.py --model max_pool --data_dir ./organ3d_data

    # Run all 4 baselines x 3 seeds (full sweep)
    python train.py --sweep --data_dir ./organ3d_data
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
from data import apply_octahedral_element, get_dataloaders, gpu_augment_3d
from model import build_model

NUM_CLASSES = 11


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


def compute_metrics(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Compute accuracy, macro AUC, and loss on a dataset."""
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    n = 0

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction='sum')
            total_loss += loss.item()
            n += labels.shape[0]

            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    preds = all_probs.argmax(dim=-1)
    accuracy = (preds == all_labels).float().mean().item()

    auc = _compute_macro_auc(all_labels, all_probs)

    return {
        'loss': total_loss / n,
        'accuracy': accuracy,
        'auc': auc,
    }


def _compute_auc_binary(labels: torch.Tensor, scores: torch.Tensor) -> float:
    """Compute binary AUC-ROC (Wilcoxon-Mann-Whitney)."""
    sorted_indices = scores.argsort(descending=True)
    sorted_labels = labels[sorted_indices].long()

    n_pos = sorted_labels.sum().item()
    n_neg = len(sorted_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp_cumsum = sorted_labels.cumsum(0)
    neg_mask = sorted_labels == 0
    auc_sum = tp_cumsum[neg_mask].sum().item()
    return auc_sum / (n_pos * n_neg)


def _compute_macro_auc(labels: torch.Tensor, probs: torch.Tensor) -> float:
    """Compute macro-averaged one-vs-rest AUC across all classes."""
    aucs = []
    for c in range(probs.shape[1]):
        binary_labels = (labels == c).float()
        if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
            continue
        auc = _compute_auc_binary(binary_labels, probs[:, c])
        aucs.append(auc)
    return sum(aucs) / len(aucs) if aucs else 0.5


def evaluate_rotation_robustness(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model consistency across all 24 octahedral rotations.

    For each rotation, rotate all test volumes and measure AUC. An equivariant model should produce
    consistent AUC regardless of rotation.
    """
    model.eval()
    results = {}

    for g_idx in range(24):
        all_probs = []
        all_labels = []
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                images = apply_octahedral_element(images, g_idx)

                logits = model(images)
                all_probs.append(torch.softmax(logits, dim=-1).cpu())
                all_labels.append(labels)

        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)

        acc = (all_probs.argmax(dim=-1) == all_labels).float().mean().item()
        auc = _compute_macro_auc(all_labels, all_probs)
        results[f'g{g_idx:02d}'] = {'accuracy': acc, 'auc': auc}

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


def _invariant_dim_info(model: torch.nn.Module) -> dict[str, int | None]:
    """Return raw and projected invariant feature widths feeding the head.

    For BispectrumPool3d, ``proj.in_channels = num_channels * features_per_channel``
    is the raw bispectrum width and ``proj.out_channels = head_dim`` is what
    the linear classifier sees. For non-bispectrum pools the head input
    equals the raw width.
    """
    pool = getattr(model, 'invariant_pool', None)
    proj = getattr(pool, 'proj', None)
    raw: int | None = None
    projected: int | None = None
    if proj is not None:
        in_ch = getattr(proj, 'in_channels', None)
        out_ch = getattr(proj, 'out_channels', None)
        if isinstance(in_ch, int):
            raw = int(in_ch)
        if isinstance(out_ch, int):
            projected = int(out_ch)
    fc_in = getattr(model, '_fc_in', None)
    if isinstance(fc_in, int):
        if projected is None:
            projected = int(fc_in)
        if raw is None:
            raw = int(fc_in)
    return {'raw_invariant_dim': raw, 'projected_dim': projected}


def _invariant_dim(model: torch.nn.Module) -> int | None:
    info = _invariant_dim_info(model)
    return info.get('projected_dim') or info.get('raw_invariant_dim')


def memory_check(args: argparse.Namespace) -> dict:
    """Run a one-batch forward/backward pass and record peak CUDA memory."""
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    channels = tuple(args.channels)
    model = build_model(
        nonlin_type=args.model,
        channels=channels,
        head_dim=args.head_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dim_info = _invariant_dim_info(model)
    invariant_dim = dim_info.get('projected_dim') or dim_info.get('raw_invariant_dim')
    n_tag = f'_n{args.train_size}' if args.train_size and args.train_size > 0 else ''
    output_dir = (
        Path(args.output_dir)
        / f'{args.model}_{args.train_mode}_ch{"_".join(map(str, channels))}_seed{args.seed}{n_tag}'
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
        images = gpu_augment_3d(
            images.to(device, non_blocking=True),
            args.train_mode == 'R',
        )
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
        'dataset': 'organ3d',
        'model': args.model,
        'train_mode': args.train_mode,
        'train_size': args.train_size,
        'train_examples': len(train_loader.dataset),
        'seed': args.seed,
        'batch_size': args.batch_size,
        'effective_batch_size': args.batch_size,
        'channels': list(channels),
        'head_dim': args.head_dim,
        'output_dir': str(output_dir),
        'n_params': n_params,
        'invariant_dim': invariant_dim,
        'raw_invariant_dim': dim_info.get('raw_invariant_dim'),
        'projected_dim': dim_info.get('projected_dim'),
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
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    num_batches = len(loader)
    t_epoch = time.time()

    for step, (images, labels) in enumerate(loader):
        images = gpu_augment_3d(images.to(device, non_blocking=True), augment_geometry)
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

        if (step + 1) % 10 == 0 or step == 0:
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

    channels = tuple(args.channels)
    model = build_model(
        nonlin_type=args.model,
        channels=channels,
        head_dim=args.head_dim,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dim_info = _invariant_dim_info(model)
    invariant_dim = dim_info.get('projected_dim') or dim_info.get('raw_invariant_dim')
    print(f'Model: {args.model} (channels={channels}), {n_params:,} params')

    if args.dry_run:
        print(
            json.dumps(
                {
                    'model': args.model,
                    'train_mode': args.train_mode,
                    'channels': list(channels),
                    'n_params': n_params,
                    'invariant_dim': invariant_dim,
                    'raw_invariant_dim': dim_info.get('raw_invariant_dim'),
                    'projected_dim': dim_info.get('projected_dim'),
                }
            )
        )
        return {'n_params': n_params}

    print(f'Device: {device}')

    augment_geo = args.train_mode == 'R'
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda')
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10 * steps_per_epoch, T_mult=2
    )

    best_auc = 0.0
    patience_counter = 0
    n_tag = f'_n{args.train_size}' if args.train_size and args.train_size > 0 else ''
    out_dir = (
        Path(args.output_dir)
        / f'{args.model}_{args.train_mode}_ch{"_".join(map(str, channels))}_seed{args.seed}{n_tag}'
    )
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
            augment_geometry=augment_geo,
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
        rot_metrics = evaluate_rotation_robustness(model, test_loader, device)
        print(
            f'Rotation robustness: mean_auc={rot_metrics["mean_auc"]:.4f}, '
            f'std_auc={rot_metrics["std_auc"]:.4f}, '
            f'mean_acc={rot_metrics["mean_accuracy"]:.4f}, '
            f'std_acc={rot_metrics["std_accuracy"]:.4f}'
        )
        test_r = _mean_group_metrics(rot_metrics)
    else:
        rot_metrics = {}
        test_r = {}

    results = {
        'dataset': 'organ3d',
        'model': args.model,
        'train_mode': args.train_mode,
        'seed': args.seed,
        'channels': list(channels),
        'head_dim': args.head_dim,
        'n_params': n_params,
        'invariant_dim': invariant_dim,
        'raw_invariant_dim': dim_info.get('raw_invariant_dim'),
        'projected_dim': dim_info.get('projected_dim'),
        'train_size': args.train_size,
        'train_examples': len(train_loader.dataset),
        'best_val_auc': best_auc,
        'test': test_c,
        'test_c': test_c,
        'test_r': test_r,
        'rotation_robustness': rot_metrics,
        **cost,
    }
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_sweep(args: argparse.Namespace):
    """Run the full experimental sweep."""
    models = ['standard', 'max_pool', 'norm_pool', 'bispectrum']
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
                print(f'Running: model={model_name}, train_mode={train_mode}, seed={seed}')
                print(f'{"=" * 60}')
                results = train(args)
                all_results.append(results)

    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"=" * 60}')
    print(f'{"Model":<15} {"Params":>10} {"Test AUC":>12} {"Test Acc":>12} {"Rot σ_AUC":>12}')
    print('-' * 65)

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

    out_path = Path(args.output_dir) / 'sweep_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nFull results saved to {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='OrganMNIST3D equivariant 3D ResNet experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--model',
        choices=['standard', 'max_pool', 'norm_pool', 'bispectrum'],
        default='bispectrum',
        help='Model variant / pooling method.',
    )
    parser.add_argument('--data_dir', '--data-dir', type=str, default='./organ3d_data')
    parser.add_argument('--output_dir', '--output-dir', type=str, default='./organ3d_results')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--channels',
        type=int,
        nargs='+',
        default=[4, 8],
        help='Channel widths (C0, C1) for stages.',
    )
    parser.add_argument('--head_dim', type=int, default=64)
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
        help='Training protocol: C=canonical, R=random octahedral augmentation.',
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
        help='Run all 4 models x 3 seeds.',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Build model, print param count, exit.',
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
        help='Skip the 24-rotation robustness evaluation.',
    )
    parser.add_argument(
        '--compile',
        action='store_true',
        help='Use torch.compile for faster training.',
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
