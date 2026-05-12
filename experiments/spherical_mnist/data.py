"""Spherical MNIST dataset.

Projects MNIST digits onto the sphere via stereographic projection from
the north pole. Supports both non-rotated (NR) and randomly rotated (R)
variants for the Cohen et al. (2018) evaluation protocol.

Usage:
    train_loader, val_loader, test_nr_loader, test_r_loader = get_dataloaders(
        data_dir="./smnist_data", nlat=64, nlon=128,
    )
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets as tv_datasets

from bispectrum import random_rotation_matrix, rotate_spherical_function

NLAT = 64
NLON = 128


def _build_projection_grid(nlat: int, nlon: int) -> torch.Tensor:
    """Stereographic projection grid from S^2 equiangular grid to MNIST plane.

    For each (theta, phi) on the equiangular grid, computes (x, y) in [-1, 1]
    via inverse stereographic projection from the north pole.  Points on the
    southern hemisphere (r > 1) land outside [-1, 1] and are zeroed by
    grid_sample with padding_mode='zeros'.

    Returns:
        (nlat, nlon, 2) tensor for use with F.grid_sample.
    """
    theta = torch.linspace(0, torch.pi, nlat, dtype=torch.float32)
    phi = torch.linspace(0, 2 * torch.pi, nlon + 1, dtype=torch.float32)[:-1]
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')

    r = torch.tan(theta_grid / 2)
    x = r * torch.cos(phi_grid)
    y = r * torch.sin(phi_grid)

    return torch.stack([x, y], dim=-1)


def project_to_sphere(
    images: torch.Tensor,
    grid: torch.Tensor,
    batch_size: int = 2000,
) -> torch.Tensor:
    """Project MNIST images onto the sphere via stereographic projection.

    Args:
        images: (N, 1, 28, 28) float tensor in [0, 1].
        grid: (nlat, nlon, 2) projection grid.
        batch_size: images processed per grid_sample call.

    Returns:
        (N, nlat, nlon) spherical signals.
    """
    grid_exp = grid.unsqueeze(0)

    chunks = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i : i + batch_size]
        g = grid_exp.expand(batch.shape[0], -1, -1, -1)
        proj = F.grid_sample(
            batch,
            g,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        chunks.append(proj.squeeze(1))

    return torch.cat(chunks, dim=0)


def apply_random_rotations(
    spherical_images: torch.Tensor,
    seed: int = 42,
) -> torch.Tensor:
    """Apply independent random SO(3) rotations to each spherical image.

    Args:
        spherical_images: (N, nlat, nlon) float tensor.
        seed: base seed for reproducible per-image rotations.

    Returns:
        (N, nlat, nlon) rotated spherical signals.
    """
    N = spherical_images.shape[0]
    result = torch.empty_like(spherical_images)

    for i in range(N):
        torch.manual_seed(seed * 100_000 + i)
        R = random_rotation_matrix()
        result[i] = rotate_spherical_function(
            spherical_images[i : i + 1],
            R,
        ).squeeze(0)

        if (i + 1) % 5000 == 0 or i == 0:
            print(f'  Rotating: {i + 1}/{N}', end='\r')

    print(f'  Rotating: {N}/{N} done.         ')
    return result


class SphericalMNISTDataset(Dataset):
    """MNIST digits projected onto the sphere.

    Precomputes and caches the spherical projection (and optional rotation) so that subsequent runs
    load instantly.
    """

    def __init__(
        self,
        split: str,
        data_dir: str,
        nlat: int = NLAT,
        nlon: int = NLON,
        rotated: bool = False,
        seed: int = 42,
    ):
        cache_dir = Path(data_dir) / 'spherical_cache'
        rot_tag = f'_rotated_s{seed}' if rotated else ''
        cache_path = cache_dir / f'{split}_{nlat}x{nlon}{rot_tag}.pt'

        if cache_path.exists():
            print(f'Loading cached {split} (rotated={rotated}) ...', end=' ')
            data = torch.load(cache_path, weights_only=True)
            self.images = data['images']
            self.labels = data['labels']
            print(f'{len(self.labels)} samples.')
            return

        print(f'Building {split} dataset (rotated={rotated}) ...')
        is_train = split in ('train', 'val')
        mnist = tv_datasets.MNIST(root=data_dir, train=is_train, download=True)
        raw_data = mnist.data.float() / 255.0
        raw_targets = mnist.targets

        if split == 'train':
            raw_data = raw_data[:50000]
            raw_targets = raw_targets[:50000]
        elif split == 'val':
            raw_data = raw_data[50000:]
            raw_targets = raw_targets[50000:]

        grid = _build_projection_grid(nlat, nlon)
        self.images = project_to_sphere(raw_data.unsqueeze(1), grid)
        self.labels = raw_targets.clone()

        if rotated:
            self.images = apply_random_rotations(self.images, seed=seed)

        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({'images': self.images, 'labels': self.labels}, cache_path)
        print(f'  Cached to {cache_path}')

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.images[idx], self.labels[idx].item()


def get_dataloaders(
    data_dir: str,
    batch_size: int = 256,
    nlat: int = NLAT,
    nlon: int = NLON,
    train_rotated: bool = False,
    train_size: int | None = None,
    seed: int = 42,
    test_rotation_seed: int = 777,
    subset_dir: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Build train / val / test_NR / test_R data loaders.

    Args:
        train_size: Absolute number of training examples to use. ``None`` or
            a value >= the full training set means use everything.

    Returns:
        (train_loader, val_loader, test_nr_loader, test_r_loader)
    """
    train_ds = SphericalMNISTDataset(
        'train',
        data_dir,
        nlat,
        nlon,
        rotated=train_rotated,
        seed=seed,
    )
    val_ds = SphericalMNISTDataset(
        'val',
        data_dir,
        nlat,
        nlon,
        rotated=train_rotated,
        seed=seed,
    )
    test_nr_ds = SphericalMNISTDataset(
        'test',
        data_dir,
        nlat,
        nlon,
        rotated=False,
    )
    test_r_ds = SphericalMNISTDataset(
        'test',
        data_dir,
        nlat,
        nlon,
        rotated=True,
        seed=test_rotation_seed,
    )

    n_full = len(train_ds)
    if train_size is not None and 0 < train_size < n_full:
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n_full, generator=rng)[:train_size].tolist()
        if subset_dir is not None:
            out_dir = Path(subset_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f'spherical_mnist_train_n{train_size}_seed{seed}.json'
            out_path.write_text(json.dumps({'indices': indices}, indent=2))
        train_ds = Subset(train_ds, indices)

    loader_rng = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        generator=loader_rng,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_nr_loader = DataLoader(
        test_nr_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_r_loader = DataLoader(
        test_r_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader, test_nr_loader, test_r_loader
