"""PatchCamelyon (PCam) dataset loading and transforms.

Downloads the official HDF5 splits from
https://github.com/basveeling/pcam and wraps them as PyTorch Datasets.

Usage:
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="./pcam_data", batch_size=64, augment_geometry=True,
    )
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader, Dataset, Subset

PCAM_URLS = {
    'train_x': 'https://zenodo.org/records/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz',
    'train_y': 'https://zenodo.org/records/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz',
    'val_x': 'https://zenodo.org/records/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz',
    'val_y': 'https://zenodo.org/records/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz',
    'test_x': 'https://zenodo.org/records/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz',
    'test_y': 'https://zenodo.org/records/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz',
}


def _download_and_extract(url: str, dest_dir: str) -> str:
    """Download *url* into *dest_dir*, gunzip if needed, return h5 path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    gz_name = url.split('/')[-1]
    gz_path = dest_dir / gz_name
    h5_path = dest_dir / gz_name.replace('.gz', '')

    if h5_path.exists():
        return str(h5_path)

    if not gz_path.exists():
        print(f'Downloading {gz_name} ...')
        # Use urllib so we don't add a requests dependency.
        import urllib.request

        urllib.request.urlretrieve(url, str(gz_path))  # nosec B310

    print(f'Extracting {gz_name} ...')
    with gzip.open(gz_path, 'rb') as f_in, open(h5_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()
    return str(h5_path)


class PCamDataset(Dataset):
    """PatchCamelyon dataset fully loaded into RAM.

    Images are 96x96 RGB, stored as a float32 (N, 3, H, W) tensor in [0, 1].
    Labels are a 1-D int64 tensor.
    """

    def __init__(
        self,
        images_h5: str,
        labels_h5: str,
        transform: torch.nn.Module | None = None,
        normalize: bool = True,
    ):
        self.transform = transform

        print(f'Loading {images_h5} into RAM ...', end=' ', flush=True)
        with h5py.File(images_h5, 'r') as f:
            raw = torch.from_numpy(f['x'][:])
            self.images = raw.permute(0, 3, 1, 2).float().div_(255.0)
        with h5py.File(labels_h5, 'r') as f:
            self.labels = torch.from_numpy(f['y'][:].flat[:]).long()

        if normalize:
            mean = torch.tensor(_MEAN).reshape(1, 3, 1, 1)
            std = torch.tensor(_STD).reshape(1, 3, 1, 1)
            self.images.sub_(mean).div_(std)

        print(f'{len(self.labels)} samples loaded.')

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.images[idx]
        label = self.labels[idx].item()
        if self.transform is not None:
            img = self.transform(img)
        return img, label


_MEAN = [0.7008, 0.5384, 0.6916]
_STD = [0.2350, 0.2774, 0.2129]


def gpu_augment(
    images: torch.Tensor,
    augment_geometry: bool = False,
) -> torch.Tensor:
    """Apply augmentations as GPU batch operations on [0, 1] tensors, then normalize.

    Much faster than per-sample CPU transforms through DataLoader workers.
    """
    B = images.shape[0]
    dev = images.device

    if augment_geometry:
        # Random horizontal flip (per-sample).
        mask = torch.rand(B, 1, 1, 1, device=dev) > 0.5
        images = torch.where(mask, images.flip(-1), images)

        # Random vertical flip (per-sample).
        mask = torch.rand(B, 1, 1, 1, device=dev) > 0.5
        images = torch.where(mask, images.flip(-2), images)

        # Uniform random k×90° rotation (per-sample).
        k = torch.randint(0, 4, (B,), device=dev)
        for rot in range(1, 4):
            rot_mask = (k == rot).reshape(B, 1, 1, 1)
            images = torch.where(rot_mask, torch.rot90(images, rot, [-2, -1]), images)

    # Brightness: multiply by uniform [1-b, 1+b].
    b = 0.1
    brightness = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 * b - b)
    images = images * brightness

    # Contrast: lerp toward per-image mean gray.
    c = 0.1
    gray_mean = images.mean(dim=(-3, -2, -1), keepdim=True)
    contrast = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 * c - c)
    images = contrast * images + (1 - contrast) * gray_mean

    # Saturation: lerp toward grayscale.
    s = 0.1
    gray = 0.2989 * images[:, 0:1] + 0.5870 * images[:, 1:2] + 0.1140 * images[:, 2:3]
    saturation = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 * s - s)
    images = saturation * images + (1 - saturation) * gray

    images = images.clamp(0, 1)

    # Normalize.
    mean = torch.tensor(_MEAN, device=dev).reshape(1, 3, 1, 1)
    std = torch.tensor(_STD, device=dev).reshape(1, 3, 1, 1)
    return (images - mean) / std


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    augment_geometry: bool = True,
    train_fraction: float = 1.0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test data loaders.

    Data is fully loaded into RAM. Augmentation runs on GPU via ``gpu_augment``,
    so all DataLoaders use ``num_workers=0`` (no CPU-side transforms).

    Args:
        data_dir: Directory to download / find HDF5 files.
        batch_size: Batch size.
        augment_geometry: Whether to apply geometric augmentation (flips,
            90-degree rotations) to training data. Typically True for
            the standard CNN baseline and False for equivariant models.
        train_fraction: Fraction of training data to use (for data-efficiency
            experiments). 1.0 = full training set.
        seed: Random seed for subsetting.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Download if needed.
    paths = {}
    for key, url in PCAM_URLS.items():
        paths[key] = _download_and_extract(url, data_dir)

    train_ds = PCamDataset(paths['train_x'], paths['train_y'], normalize=False)
    val_ds = PCamDataset(paths['val_x'], paths['val_y'], normalize=True)
    test_ds = PCamDataset(paths['test_x'], paths['test_y'], normalize=True)

    # Optionally subset training data for data-efficiency experiments.
    if train_fraction < 1.0:
        n = len(train_ds)
        k = max(1, int(n * train_fraction))
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n, generator=rng)[:k].tolist()
        train_ds = Subset(train_ds, indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
