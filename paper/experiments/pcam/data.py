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
from torchvision import transforms

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
    """PatchCamelyon dataset backed by HDF5 files.

    Images are 96x96 RGB uint8.  Labels are binary (0/1).
    """

    def __init__(
        self,
        images_h5: str,
        labels_h5: str,
        transform: transforms.Compose | None = None,
    ):
        self.images_h5 = images_h5
        self.labels_h5 = labels_h5
        self.transform = transform

        # Open read-only; h5py supports lazy loading so this is cheap.
        self._img_file = h5py.File(images_h5, 'r')
        self._lbl_file = h5py.File(labels_h5, 'r')
        self.images = self._img_file['x']
        self.labels = self._lbl_file['y']

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.images[idx]  # (96, 96, 3) uint8
        label = int(self.labels[idx].flat[0])

        # Convert to float tensor (C, H, W) in [0, 1].
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def close(self):
        self._img_file.close()
        self._lbl_file.close()


_MEAN = [0.7008, 0.5384, 0.6916]
_STD = [0.2350, 0.2774, 0.2129]


def train_transform(augment_geometry: bool = True) -> transforms.Compose:
    """Training transforms.

    Args:
        augment_geometry: If True, apply random flips and 90-degree rotations.
            Set False for equivariant models where geometric augmentation is
            redundant.
    """
    ops = []
    if augment_geometry:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation(degrees=(90, 90))], p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=(180, 180))], p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=(270, 270))], p=0.5),
        ]
    ops += [
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
    return transforms.Compose(ops)


def eval_transform() -> transforms.Normalize:
    return transforms.Normalize(mean=_MEAN, std=_STD)


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    augment_geometry: bool = True,
    num_workers: int = 4,
    train_fraction: float = 1.0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test data loaders.

    Args:
        data_dir: Directory to download / find HDF5 files.
        batch_size: Batch size.
        augment_geometry: Whether to apply geometric augmentation (flips,
            90-degree rotations) to training data. Typically True for
            the standard CNN baseline and False for equivariant models.
        num_workers: Number of dataloader workers.
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

    train_ds = PCamDataset(
        paths['train_x'],
        paths['train_y'],
        transform=train_transform(augment_geometry),
    )
    val_ds = PCamDataset(paths['val_x'], paths['val_y'], transform=eval_transform())
    test_ds = PCamDataset(paths['test_x'], paths['test_y'], transform=eval_transform())

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
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
