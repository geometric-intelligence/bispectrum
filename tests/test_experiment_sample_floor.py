"""Per-epoch sample floor used by the experiment data loaders.

``experiments/*/data.py`` draw ``max(len(train_ds), min(min_epoch_samples, n_full))``
samples per epoch through a ``RandomSampler`` so that a ``--train_size 100`` run
does not collapse to one optimizer step per epoch. Two properties are tested:

1. When the training set is at least as large as the floor, the sampler yields
   exactly the index sequence ``DataLoader(shuffle=True)`` would, so results
   produced before the floor existed remain comparable for those sizes.
2. Below the floor, every example is drawn the same number of times per epoch
   (whole permutations, no replacement).

The first group of tests only needs torch and runs in CI. The second group
imports the experiment modules by path and is skipped when their optional
dependencies (torchvision, h5py) are absent.
"""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

FLOOR = 10_000
EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / 'experiments'


def _floor_sampler(
    dataset: TensorDataset, n_full: int, min_epoch_samples: int, seed: int
) -> RandomSampler:
    """Mirror of the sampler construction in ``experiments/*/data.py``."""
    rng = torch.Generator().manual_seed(seed)
    return RandomSampler(
        dataset,
        num_samples=max(len(dataset), min(min_epoch_samples, n_full)),
        generator=rng,
    )


def _indices(loader: DataLoader, epochs: int) -> list[int]:
    out: list[int] = []
    for _ in range(epochs):
        for (batch,) in loader:
            out.extend(int(i) for i in batch)
    return out


def _reference_loader(n: int, batch_size: int, seed: int) -> DataLoader:
    """The pre-floor loader: ``shuffle=True`` seeded through ``generator``."""
    ds = TensorDataset(torch.arange(n))
    rng = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=min(batch_size, n), shuffle=True, generator=rng)


def _floor_loader(n: int, n_full: int, batch_size: int, seed: int) -> DataLoader:
    ds = TensorDataset(torch.arange(n))
    rng = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(ds, num_samples=max(n, min(FLOOR, n_full)), generator=rng)
    return DataLoader(ds, batch_size=min(batch_size, n), sampler=sampler, generator=rng)


@pytest.mark.parametrize(
    ('n', 'n_full'),
    [
        (12_500, 262_144),  # PCam n=12.5k subset, above the floor
        (10_000, 60_000),  # exactly at the floor
        (971, 971),  # OrganMNIST3D full set: floor clips to n_full
        (60_000, 60_000),  # Spherical MNIST full set
    ],
)
def test_floor_matches_shuffle_when_at_or_above_floor(n: int, n_full: int) -> None:
    """At or above the floor the sampler reproduces ``shuffle=True`` exactly."""
    ref = _indices(_reference_loader(n, 256, seed=42), epochs=3)
    new = _indices(_floor_loader(n, n_full, 256, seed=42), epochs=3)
    assert new == ref


@pytest.mark.parametrize(('n', 'n_full'), [(100, 60_000), (500, 262_144), (2_500, 60_000)])
def test_floor_cycles_small_subsets_uniformly(n: int, n_full: int) -> None:
    """Below the floor every index is drawn ``FLOOR // n`` times per epoch."""
    loader = _floor_loader(n, n_full, 256, seed=42)
    assert len(loader.sampler) == FLOOR
    idx = _indices(loader, epochs=1)
    assert len(idx) == FLOOR
    counts = torch.bincount(torch.tensor(idx), minlength=n)
    assert counts.min().item() == counts.max().item() == FLOOR // n


def test_floor_clips_to_full_dataset_size() -> None:
    """A floor larger than the full set must not oversample full-data runs."""
    ds = TensorDataset(torch.arange(971))
    sampler = _floor_sampler(ds, n_full=971, min_epoch_samples=FLOOR, seed=0)
    assert len(sampler) == 971


def test_steps_per_epoch_scale_with_floor() -> None:
    """The LR schedule uses ``len(train_loader)``; it must follow the floor."""
    small = _floor_loader(100, 60_000, 256, seed=0)
    large = _floor_loader(12_500, 60_000, 256, seed=0)
    assert len(small) == -(-FLOOR // 100)  # ceil(10_000 / 100) = 100 steps
    assert len(large) == -(-12_500 // 256)  # unchanged: ceil(12_500 / 256) = 49 steps


# ---------------------------------------------------------------------------
# Signature checks against the real experiment modules (optional deps).
# ---------------------------------------------------------------------------


def _load_experiment_module(name: str) -> ModuleType:
    path = EXPERIMENTS_DIR / name / 'data.py'
    if not path.exists():
        pytest.skip(f'{path} not present')
    spec = importlib.util.spec_from_file_location(f'_exp_{name}_data', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _experiment_modules() -> Iterator[tuple[str, tuple[str, ...]]]:
    yield 'pcam', ('h5py', 'torchvision')
    yield 'organ3d', ()
    yield 'spherical_mnist', ('torchvision',)


@pytest.mark.parametrize(('name', 'deps'), list(_experiment_modules()))
def test_get_dataloaders_exposes_min_epoch_samples(name: str, deps: tuple[str, ...]) -> None:
    """Each experiment loader takes ``min_epoch_samples`` with the shared default."""
    for dep in deps:
        pytest.importorskip(dep)
    module = _load_experiment_module(name)
    params = inspect.signature(module.get_dataloaders).parameters
    assert 'min_epoch_samples' in params
    assert params['min_epoch_samples'].default == FLOOR
