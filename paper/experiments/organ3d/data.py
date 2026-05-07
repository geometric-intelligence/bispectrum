"""OrganMNIST3D dataset loading and transforms.

Downloads the official OrganMNIST3D from the MedMNIST collection via the
``medmnist`` package and wraps it as PyTorch Datasets.

Usage:
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="./organ3d_data", batch_size=32,
    )
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

DATASET_MEAN = 0.5009
DATASET_STD = 0.2802


class OrganMNIST3DDataset(Dataset):
    """OrganMNIST3D dataset fully loaded into RAM.

    Volumes are 28x28x28 single-channel CT, stored as float32 (N, 1, 28, 28, 28)
    tensors in [0, 1].  Labels are 1-D int64 tensor with values in 0..10.
    """

    def __init__(self, split: str, data_dir: str, normalize: bool = True):
        import medmnist

        Path(data_dir).mkdir(parents=True, exist_ok=True)
        ds = medmnist.OrganMNIST3D(
            split=split, download=True, root=data_dir, size=28,
        )

        imgs = []
        labels = []
        for img, lbl in ds:
            imgs.append(np.array(img))
            labels.append(int(lbl.squeeze()))

        self.images = torch.from_numpy(np.stack(imgs)).float()
        if self.images.ndim == 4:
            self.images = self.images.unsqueeze(1)

        self.labels = torch.tensor(labels, dtype=torch.long)

        if normalize:
            self.images.sub_(DATASET_MEAN).div_(DATASET_STD)

        print(f'OrganMNIST3D [{split}]: {len(self.labels)} samples loaded.')

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.images[idx], self.labels[idx].item()


_OCTA_ROTATIONS_3x3 = None


def _get_octahedral_rotations() -> torch.Tensor:
    """Return the 24 octahedral rotation matrices (3x3, integer-valued)."""
    global _OCTA_ROTATIONS_3x3
    if _OCTA_ROTATIONS_3x3 is not None:
        return _OCTA_ROTATIONS_3x3

    from bispectrum.octa_on_octa import _ELEMENTS_3x3

    _OCTA_ROTATIONS_3x3 = _ELEMENTS_3x3.clone()
    return _OCTA_ROTATIONS_3x3


def gpu_augment_3d(
    images: torch.Tensor,
    augment_geometry: bool = False,
) -> torch.Tensor:
    """Apply augmentations as GPU batch operations on [0, 1] tensors, then normalize.

    Args:
        images: (B, 1, D, H, W) float tensors in [0, 1].
        augment_geometry: apply random octahedral rotations.
    """
    B = images.shape[0]
    dev = images.device

    if augment_geometry:
        rots = _get_octahedral_rotations().to(dev)
        idx = torch.randint(0, 24, (B,), device=dev)
        for b in range(B):
            vol = images[b, 0]
            R = rots[idx[b]].long()
            coords = torch.stack(
                torch.meshgrid(
                    torch.arange(28, device=dev) - 13.5,
                    torch.arange(28, device=dev) - 13.5,
                    torch.arange(28, device=dev) - 13.5,
                    indexing='ij',
                ),
                dim=-1,
            ).reshape(-1, 3).float()
            new_coords = (R.float() @ coords.T).T
            ni = (new_coords[:, 0] + 13.5).long().clamp(0, 27)
            nj = (new_coords[:, 1] + 13.5).long().clamp(0, 27)
            nk = (new_coords[:, 2] + 13.5).long().clamp(0, 27)
            images[b, 0] = vol[ni, nj, nk].reshape(28, 28, 28)

    b_factor = 0.1
    brightness = 1.0 + (torch.rand(B, 1, 1, 1, 1, device=dev) * 2 * b_factor - b_factor)
    images = images * brightness

    c_factor = 0.1
    gray_mean = images.mean(dim=(-3, -2, -1), keepdim=True)
    contrast = 1.0 + (torch.rand(B, 1, 1, 1, 1, device=dev) * 2 * c_factor - c_factor)
    images = contrast * images + (1 - contrast) * gray_mean

    images = images.clamp(0, 1)

    return (images - DATASET_MEAN) / DATASET_STD


def apply_octahedral_element(images: torch.Tensor, element_idx: int) -> torch.Tensor:
    """Apply one deterministic octahedral group element to a normalized batch."""
    if element_idx == 23:
        return images

    dev = images.device
    rots = _get_octahedral_rotations().to(dev)
    R = rots[element_idx].long()
    half = 13.5
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(28, device=dev) - half,
            torch.arange(28, device=dev) - half,
            torch.arange(28, device=dev) - half,
            indexing='ij',
        ),
        dim=-1,
    ).reshape(-1, 3).float()
    new_coords = (R.float() @ coords.T).T
    ni = (new_coords[:, 0] + half).long().clamp(0, 27)
    nj = (new_coords[:, 1] + half).long().clamp(0, 27)
    nk = (new_coords[:, 2] + half).long().clamp(0, 27)
    B = images.shape[0]
    rotated = images[:, 0].reshape(B, -1)[:, ni * 28 * 28 + nj * 28 + nk]
    return rotated.reshape(B, 1, 28, 28, 28)


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    augment_geometry: bool = True,
    train_size: int | None = None,
    seed: int = 42,
    subset_dir: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test data loaders.

    Data is fully loaded into RAM. Augmentation runs on GPU via ``gpu_augment_3d``.

    Args:
        train_size: Absolute number of training examples to use. ``None`` or
            a value >= the full training set means use everything.
    """
    data_dir = str(Path(data_dir).resolve())

    train_ds = OrganMNIST3DDataset('train', data_dir, normalize=False)
    val_ds = OrganMNIST3DDataset('val', data_dir, normalize=True)
    test_ds = OrganMNIST3DDataset('test', data_dir, normalize=True)

    n_full = len(train_ds)
    if train_size is not None and 0 < train_size < n_full:
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n_full, generator=rng)[:train_size].tolist()
        if subset_dir is not None:
            out_dir = Path(subset_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f'organ3d_train_n{train_size}_seed{seed}.json'
            out_path.write_text(json.dumps({'indices': indices}, indent=2))
        train_ds = Subset(train_ds, indices)

    loader_rng = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True,
        generator=loader_rng, num_workers=0, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
