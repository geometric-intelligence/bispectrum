"""SO2-equivariant bispectrum module for spherical functions.

This module provides the SO2onS2 class, a torch.nn.Module that computes the bispectrum of spherical
harmonic coefficients under SO(2) (z-axis) rotations. The output is invariant under azimuthal
rotations.
"""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from bispectrum.spherical import compute_padding_indices, get_full_sh_coefficients

if TYPE_CHECKING:
    from collections.abc import Sequence


class SO2onS2(nn.Module):
    """SO(2)-invariant bispectrum operator for spherical harmonic coefficients.

    Computes an SO(2)-invariant bispectrum by retaining only the m=0 coupled component
    for each (l1, l2, l) tuple. Under z-axis rotations, each coefficient transforms as
    F_l^m -> e^{-im alpha} F_l^m, so the m=0 component is invariant.

    Args:
        lmax: Maximum spherical harmonic degree. Default is 5.
        cg_path: Path to JSON file containing Clebsch-Gordan matrices.
            If None, uses the bundled cg_lmax5.json file.
    """

    def __init__(self, lmax: int = 5, cg_path: str | Path | None = None) -> None:
        super().__init__()
        self.lmax = lmax

        if cg_path is None:
            cg_path = importlib.resources.files('bispectrum') / 'data' / 'cg_lmax5.json'

        with open(cg_path) as f:
            cg_data = json.load(f)

        json_l1_max = cg_data['metadata']['l1_max']
        json_l2_max = cg_data['metadata']['l2_max']
        if lmax > json_l1_max or lmax > json_l2_max:
            raise ValueError(
                f'lmax={lmax} exceeds JSON limits (l1_max={json_l1_max}, l2_max={json_l2_max})'
            )

        self._index_map: list[tuple[int, int, int]] = []
        for l1 in range(lmax + 1):
            for l2 in range(l1, lmax + 1):
                l_min = abs(l1 - l2)
                l_max_pair = l1 + l2
                for l in range(l_min, l_max_pair + 1):
                    if l <= lmax:
                        self._index_map.append((l1, l2, l))

        for l1 in range(lmax + 1):
            for l2 in range(l1, lmax + 1):
                key = f'{l1}_{l2}'
                matrix_data = cg_data['matrices'][key]['matrix']
                matrix = torch.tensor(matrix_data, dtype=torch.float64)
                self.register_buffer(f'cg_{l1}_{l2}', matrix)

    @property
    def index_map(self) -> Sequence[tuple[int, int, int]]:
        """Map from flat output index to (l1, l2, l) tuple."""
        return self._index_map

    @property
    def output_size(self) -> int:
        """Number of bispectrum values in the output."""
        return len(self._index_map)

    def _get_cg_matrix(self, l1: int, l2: int) -> torch.Tensor:
        """Get the Clebsch-Gordan matrix for (l1, l2) pair."""
        return getattr(self, f'cg_{l1}_{l2}')

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Compute SO(2)-invariant bispectrum for input SH coefficients.

        Args:
            coeffs: Complex tensor of shape (batch, lmax, mmax) containing
                SH coefficients for m >= 0 (output of RealSHT).

        Returns:
            Complex tensor of shape (batch, output_size) containing bispectrum
            values for all (l1, l2, l) combinations. Use index_map to decode.
        """
        batch_size = coeffs.shape[0]

        f_coeffs = get_full_sh_coefficients(coeffs)

        result = torch.zeros(
            batch_size, self.output_size, dtype=coeffs.dtype, device=coeffs.device
        )

        computed_transforms: dict[tuple[int, int], torch.Tensor] = {}

        for out_idx, (l1, l2, l) in enumerate(self._index_map):
            if (l1, l2) not in computed_transforms:
                f_l1 = f_coeffs[l1]
                f_l2 = f_coeffs[l2]

                outer = torch.einsum('bi,bj->bij', f_l1, f_l2)
                tensor_product = outer.reshape(batch_size, -1)

                cg_matrix = self._get_cg_matrix(l1, l2)
                cg_matrix = cg_matrix.to(device=tensor_product.device, dtype=tensor_product.dtype)

                computed_transforms[(l1, l2)] = tensor_product @ cg_matrix

            if l not in f_coeffs:
                continue

            transformed = computed_transforms[(l1, l2)]
            f_l = f_coeffs[l]
            n_p, _ = compute_padding_indices(l1, l2, l)
            m_zero_index = n_p + l
            result[:, out_idx] = transformed[:, m_zero_index] * torch.conj(f_l[:, l])

        return result

    def extra_repr(self) -> str:
        """Extra representation for printing the module."""
        return f'lmax={self.lmax}, output_size={self.output_size}'
