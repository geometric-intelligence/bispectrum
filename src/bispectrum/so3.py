"""SO3-equivariant bispectrum module for spherical functions.

This module provides the SO3onS2 class, a torch.nn.Module that computes the bispectrum of spherical
harmonic coefficients. The bispectrum is invariant under SO(3) rotations.
"""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from bispectrum.spherical import get_full_sh_coefficients, pad_sh_coefficients

if TYPE_CHECKING:
    from collections.abc import Sequence


class SO3onS2(nn.Module):
    """Bispectrum operator for spherical harmonic coefficients.

    Computes the SO(3)-invariant bispectrum for all (l1, l2) pairs where
    l1 <= l2 <= lmax // 2. The bispectrum is defined as:

        beta_{l1,l2}[l] = (F_l1 ⊗ F_l2) @ C_{l1,l2} @ F_hat_l^†

    where C_{l1,l2} is the Clebsch-Gordan matrix and l ranges from |l1-l2| to l1+l2.

    By constraining l1, l2 <= lmax // 2, we ensure that l <= l1 + l2 <= lmax,
    so all required coefficients are available.

    Args:
        lmax: Maximum spherical harmonic degree for input coefficients. Default is 5.
        cg_path: Path to JSON file containing Clebsch-Gordan matrices.
            If None, uses the bundled cg_lmax5.json file.

    Example:
        >>> bsp = SO3onS2(lmax=4)
        >>> # coeffs: (batch, lmax, mmax) complex tensor from RealSHT
        >>> output = bsp(coeffs)  # (batch, num_bispectrum_values)
    """

    def __init__(self, lmax: int = 5, cg_path: str | Path | None = None) -> None:
        super().__init__()
        self.lmax = lmax
        self.l1_max = lmax // 2
        self.l2_max = lmax // 2

        # Load CG matrices from JSON
        if cg_path is None:
            cg_path = importlib.resources.files('bispectrum') / 'data' / 'cg_lmax5.json'

        with open(cg_path) as f:
            cg_data = json.load(f)

        # Validate that CG data has the matrices we need
        json_l1_max = cg_data['metadata']['l1_max']
        json_l2_max = cg_data['metadata']['l2_max']
        if self.l1_max > json_l1_max or self.l2_max > json_l2_max:
            raise ValueError(
                f'l1_max={self.l1_max}, l2_max={self.l2_max} exceed JSON limits '
                f'(l1_max={json_l1_max}, l2_max={json_l2_max})'
            )

        # Build index map: maps flat output index to (l1, l2, l) tuple
        # Since l1, l2 <= lmax // 2, we have l <= l1 + l2 <= lmax
        self._index_map: list[tuple[int, int, int]] = []
        for l1 in range(self.l1_max + 1):
            for l2 in range(l1, self.l2_max + 1):
                l_min = abs(l1 - l2)
                l_max_pair = l1 + l2
                for l in range(l_min, l_max_pair + 1):
                    self._index_map.append((l1, l2, l))

        # Register CG matrices as buffers for all (l1, l2) pairs
        for l1 in range(self.l1_max + 1):
            for l2 in range(l1, self.l2_max + 1):
                key = f'{l1}_{l2}'
                matrix_data = cg_data['matrices'][key]['matrix']
                matrix = torch.tensor(matrix_data, dtype=torch.float64)
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
            coeffs: Complex tensor of shape (batch, lmax, mmax) containing
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
            f_l = f_coeffs[l]
            f_hat_l = pad_sh_coefficients(f_l, l1, l2, l)
            result[:, out_idx] = torch.sum(transformed * torch.conj(f_hat_l), dim=-1)

        return result

    def extra_repr(self) -> str:
        """Extra representation for printing the module."""
        return f'lmax={self.lmax}, l1_max={self.l1_max}, l2_max={self.l2_max}, output_size={self.output_size}'
