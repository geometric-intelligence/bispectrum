"""SO3-equivariant bispectrum module for spherical functions.

This module provides the SO3onS2 class, a torch.nn.Module that computes the bispectrum of spherical
harmonic coefficients. The bispectrum is invariant under SO(3) rotations.
"""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from bispectrum.spherical import get_full_sh_coefficients, pad_sh_coefficients

if TYPE_CHECKING:
    from collections.abc import Sequence


def _unwrap_npz_dict(value: np.ndarray | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.shape == ():
        item = value.item()
        if isinstance(item, dict):
            return item
    if isinstance(value, dict):
        return value
    return None


def _infer_npz_limits(
    keys: list[str],
    metadata: dict[str, Any] | None,
    matrix_info: dict[str, Any] | None,
) -> tuple[int, int]:
    l1_max: int | None = None
    l2_max: int | None = None

    if metadata is not None:
        if 'l1_max' in metadata:
            l1_max = int(metadata['l1_max'])
        if 'l2_max' in metadata:
            l2_max = int(metadata['l2_max'])

    if (l1_max is None or l2_max is None) and matrix_info:
        l1_values = [int(info['l1']) for info in matrix_info.values()]
        l2_values = [int(info['l2']) for info in matrix_info.values()]
        if l1_values:
            l1_max = max(l1_values)
        if l2_values:
            l2_max = max(l2_values)

    if l1_max is None or l2_max is None:
        l1_max = max(int(key.split('_')[0]) for key in keys)
        l2_max = max(int(key.split('_')[1]) for key in keys)

    return l1_max, l2_max


def _load_cg_npz(cg_path: Path) -> tuple[dict[tuple[int, int], torch.Tensor], int, int]:
    matrices: dict[tuple[int, int], torch.Tensor] = {}

    with np.load(cg_path, allow_pickle=True) as data:
        keys = [key for key in data.files if '_' in key and not key.startswith('_')]
        metadata = _unwrap_npz_dict(data.get('_metadata'))
        matrix_info = _unwrap_npz_dict(data.get('_matrix_info'))
        l1_max, l2_max = _infer_npz_limits(keys, metadata, matrix_info)

        for key in keys:
            l1, l2 = (int(part) for part in key.split('_'))
            matrices[(l1, l2)] = torch.tensor(data[key], dtype=torch.float64)

    return matrices, l1_max, l2_max


def _load_cg_json(cg_path: Path) -> tuple[dict[tuple[int, int], torch.Tensor], int, int]:
    with open(cg_path) as f:
        cg_data = json.load(f)

    l1_max = int(cg_data['metadata']['l1_max'])
    l2_max = int(cg_data['metadata']['l2_max'])
    matrices = {}

    for key, entry in cg_data['matrices'].items():
        l1, l2 = (int(part) for part in key.split('_'))
        matrices[(l1, l2)] = torch.tensor(entry['matrix'], dtype=torch.float64)

    return matrices, l1_max, l2_max


def _load_cg_matrices(cg_path: Path) -> tuple[dict[tuple[int, int], torch.Tensor], int, int]:
    if cg_path.suffix == '.npz':
        return _load_cg_npz(cg_path)
    if cg_path.suffix == '.json':
        return _load_cg_json(cg_path)
    raise ValueError(f'Unsupported CG file type: {cg_path.suffix}')


class SO3onS2(nn.Module):
    """Bispectrum operator for spherical harmonic coefficients.

    Computes the SO(3)-invariant bispectrum for all (l1, l2) pairs where
    l1 <= l2 <= lmax. The bispectrum is defined as:

        beta_{l1,l2}[l] = (F_l1 ⊗ F_l2) @ C_{l1,l2} @ F_hat_l^†

    where C_{l1,l2} is the Clebsch-Gordan matrix.

    Args:
        lmax: Maximum spherical harmonic degree. Default is 10.
        cg_path: Path to .npz/.json file containing Clebsch-Gordan matrices.
            If None, uses the bundled cg_lmax10.npz file.

    Example:
        >>> bsp = SO3onS2(lmax=10)
        >>> # coeffs: (batch, lmax + 1, mmax) complex tensor from RealSHT
        >>> output = bsp(coeffs)  # (batch, num_bispectrum_values)
    """

    def __init__(self, lmax: int = 10, cg_path: str | Path | None = None) -> None:
        super().__init__()
        self.lmax = lmax

        if cg_path is None:
            cg_path = importlib.resources.files('bispectrum') / 'data' / 'cg_lmax10.npz'

        cg_path = Path(cg_path)
        matrices, l1_max, l2_max = _load_cg_matrices(cg_path)
        if lmax > l1_max or lmax > l2_max:
            raise ValueError(f'lmax={lmax} exceeds CG limits (l1_max={l1_max}, l2_max={l2_max})')

        # Build index map: maps flat output index to (l1, l2, l) tuple
        self._index_map: list[tuple[int, int, int]] = []
        for l1 in range(lmax + 1):
            for l2 in range(l1, lmax + 1):
                l_min = abs(l1 - l2)
                l_max_pair = l1 + l2
                for l in range(l_min, l_max_pair + 1):
                    if l <= lmax:  # Only include if we have coefficients for this l
                        self._index_map.append((l1, l2, l))

        # Register CG matrices as buffers for all (l1, l2) pairs
        for l1 in range(lmax + 1):
            for l2 in range(l1, lmax + 1):
                matrix = matrices.get((l1, l2))
                if matrix is None:
                    raise KeyError(f'Missing CG matrix for (l1={l1}, l2={l2}) in {cg_path}')
                self.register_buffer(f'cg_{l1}_{l2}', matrix)

    @property
    def index_map(self) -> Sequence[tuple[int, int, int]]:
        """Map from flat output index to (l1, l2, l) tuple.

        Returns:
            Sequence of (l1, l2, l) tuples, one per output dimension.
        """
        return self._index_map

    @property
    def output_size(self) -> int:
        """Number of bispectrum values in the output.

        Returns:
            Length of the flat output tensor's last dimension.
        """
        return len(self._index_map)

    def _get_cg_matrix(self, l1: int, l2: int) -> torch.Tensor:
        """Get the Clebsch-Gordan matrix for (l1, l2) pair.

        Args:
            l1: First angular momentum degree.
            l2: Second angular momentum degree (must be >= l1).

        Returns:
            CG matrix of shape ((2l1+1)*(2l2+1), (2l1+1)*(2l2+1)).
        """
        return getattr(self, f'cg_{l1}_{l2}')

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Compute bispectrum for input spherical harmonic coefficients.

        Args:
            coeffs: Complex tensor of shape (batch, lmax + 1, mmax) containing
                SH coefficients for m >= 0 (output of RealSHT).

        Returns:
            Complex tensor of shape (batch, output_size) containing bispectrum
            values for all (l1, l2, l) combinations. Use index_map property
            to decode which index corresponds to which (l1, l2, l).
        """
        batch_size = coeffs.shape[0]

        # Expand to full coefficients (all m from -l to l)
        f_coeffs = get_full_sh_coefficients(coeffs)

        # Allocate output tensor
        result = torch.zeros(
            batch_size, self.output_size, dtype=coeffs.dtype, device=coeffs.device
        )

        # Track which (l1, l2) pairs we've computed
        computed_transforms: dict[tuple[int, int], torch.Tensor] = {}

        # Compute bispectrum for each output index
        for out_idx, (l1, l2, l) in enumerate(self._index_map):
            # Get or compute the transformed tensor product for this (l1, l2) pair
            if (l1, l2) not in computed_transforms:
                f_l1 = f_coeffs[l1]  # (batch, 2*l1+1)
                f_l2 = f_coeffs[l2]  # (batch, 2*l2+1)

                # Compute batch-wise tensor product
                outer = torch.einsum('bi,bj->bij', f_l1, f_l2)
                tensor_product = outer.reshape(batch_size, -1)

                # Get CG matrix and move to correct device/dtype
                cg_matrix = self._get_cg_matrix(l1, l2)
                cg_matrix = cg_matrix.to(device=tensor_product.device, dtype=tensor_product.dtype)

                # Apply CG matrix
                computed_transforms[(l1, l2)] = tensor_product @ cg_matrix

            transformed = computed_transforms[(l1, l2)]

            # Get F_l coefficients and compute inner product
            if l in f_coeffs:
                f_l = f_coeffs[l]
                f_hat_l = pad_sh_coefficients(f_l, l1, l2, l)
                result[:, out_idx] = torch.sum(transformed * torch.conj(f_hat_l), dim=-1)

        return result

    def extra_repr(self) -> str:
        """Extra representation for printing the module."""
        return f'lmax={self.lmax}, output_size={self.output_size}'
