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
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from data import get_dataloaders, gpu_augment_3d
from model import build_model, _make_octahedral_group

NUM_CLASSES = 11


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

    For each rotation, rotate all test volumes and measure AUC.
    An equivariant model should produce consistent AUC regardless of rotation.
    """
    elements, _, _ = _make_octahedral_group()
    elements = elements.to(device)

    half = 13.5
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(28, device=device, dtype=torch.float32) - half,
            torch.arange(28, device=device, dtype=torch.float32) - half,
            torch.arange(28, device=device, dtype=torch.float32) - half,
            indexing='ij',
        ),
        dim=-1,
    ).reshape(-1, 3)

    model.eval()
    results = {}

    for g_idx in range(24):
        R = elements[g_idx]
        new_coords = (R @ coords.T).T
        ni = (new_coords[:, 0] + half).round().long().clamp(0, 27)
        nj = (new_coords[:, 1] + half).round().long().clamp(0, 27)
        nk = (new_coords[:, 2] + half).round().long().clamp(0, 27)

        all_probs = []
        all_labels = []
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                B = images.shape[0]
                if g_idx != 23:  # g23 is identity
                    rotated = images[:, 0].reshape(B, -1)[:, ni * 28 * 28 + nj * 28 + nk]
                    images = rotated.reshape(B, 1, 28, 28, 28)

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
    results['std_accuracy'] = (sum((a - results['mean_accuracy']) ** 2 for a in accs) / len(accs)) ** 0.5

    return results


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
    """Run a single training experiment. Returns final metrics dict."""
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
    print(f'Model: {args.model} (channels={channels}), {n_params:,} params')

    if args.dry_run:
        print(json.dumps({
            'model': args.model,
            'channels': list(channels),
            'n_params': n_params,
        }))
        return {'n_params': n_params}

    print(f'Device: {device}')

    augment_geo = args.model == 'standard'
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        augment_geometry=augment_geo,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    print(
        f'Data: train={len(train_loader.dataset)}, '
        f'val={len(val_loader.dataset)}, test={len(test_loader.dataset)}'
    )

    if args.compile:
        model = torch.compile(model)

    torch.set_float32_matmul_precision('high')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler('cuda')
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10 * steps_per_epoch, T_mult=2
    )

    best_auc = 0.0
    patience_counter = 0
    out_dir = Path(args.output_dir) / f'{args.model}_ch{"_".join(map(str, channels))}_seed{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler,
            augment_geometry=augment_geo,
        )
        val_metrics = compute_metrics(model, val_loader, device)
        elapsed = time.time() - t0

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

    model.load_state_dict(torch.load(out_dir / 'best_model.pt', weights_only=True))
    test_metrics = compute_metrics(model, test_loader, device)
    print(f'\nTest results: {test_metrics}')

    if not args.skip_rotation:
        rot_metrics = evaluate_rotation_robustness(model, test_loader, device)
        print(
            f'Rotation robustness: mean_auc={rot_metrics["mean_auc"]:.4f}, '
            f'std_auc={rot_metrics["std_auc"]:.4f}, '
            f'mean_acc={rot_metrics["mean_accuracy"]:.4f}, '
            f'std_acc={rot_metrics["std_accuracy"]:.4f}'
        )
    else:
        rot_metrics = {'mean_auc': 0.0, 'std_auc': 0.0, 'mean_accuracy': 0.0, 'std_accuracy': 0.0}

    results = {
        'model': args.model,
        'seed': args.seed,
        'channels': list(channels),
        'head_dim': args.head_dim,
        'n_params': n_params,
        'train_fraction': args.train_fraction,
        'best_val_auc': best_auc,
        'test': test_metrics,
        'rotation_robustness': rot_metrics,
    }
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_sweep(args: argparse.Namespace):
    """Run the full experimental sweep."""
    models = ['standard', 'max_pool', 'norm_pool', 'bispectrum']
    seeds = [42, 123, 456]
    all_results = []

    for model_name in models:
        for seed in seeds:
            args.model = model_name
            args.seed = seed
            print(f'\n{"=" * 60}')
            print(f'Running: model={model_name}, seed={seed}')
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
        grouped[r['model']].append(r)

    for model_name in models:
        runs = grouped[model_name]
        aucs = [r['test']['auc'] for r in runs]
        accs = [r['test']['accuracy'] for r in runs]
        rot_stds = [r['rotation_robustness']['std_auc'] for r in runs]
        n_params = runs[0]['n_params']

        mean_auc = sum(aucs) / len(aucs)
        std_auc = (sum((a - mean_auc) ** 2 for a in aucs) / len(aucs)) ** 0.5
        mean_acc = sum(accs) / len(accs)
        mean_rot = sum(rot_stds) / len(rot_stds)

        print(
            f'{model_name:<15} {n_params:>10,} '
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
        '--channels', type=int, nargs='+', default=[4, 8],
        help='Channel widths (C0, C1) for stages.',
    )
    parser.add_argument('--head_dim', type=int, default=64)
    parser.add_argument(
        '--train_fraction', type=float, default=1.0,
        help='Fraction of training data to use.',
    )
    parser.add_argument(
        '--sweep', action='store_true',
        help='Run all 4 models x 3 seeds.',
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Build model, print param count, exit.',
    )
    parser.add_argument(
        '--skip_rotation', action='store_true',
        help='Skip the 24-rotation robustness evaluation.',
    )
    parser.add_argument(
        '--compile', action='store_true',
        help='Use torch.compile for faster training.',
    )

    args = parser.parse_args()

    if args.sweep:
        run_sweep(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
