"""Dihedral group bispectrum on D_n.

Implements the G-Bispectrum for the dihedral group D_n acting on itself.

D_n = <a, x | a^n = x^2 = e, xax = a^{-1}> has |D_n| = 2n elements.
Irreps include 2D rotation/reflection representations rho_k and 1D sign
representations, with CG matrices computed analytically from the explicit
irrep formulas.

Reference: Mataigne et al. (2024) Algorithm 3.
"""

import torch
import torch.nn as nn


class DnonDn(nn.Module):
    """Bispectrum of D_n acting on D_n.

    Signal f: D_n -> R has length 2n (one value per group element).
    The group Fourier transform and CG matrices are computed from the
    explicit irrep formulas — no external CG library needed.

    Args:
        n: Polygon order (|D_n| = 2n).
        selective: If True, use selective bispectrum (Algorithm 3).
            If False, use full bispectrum.
    """

    def __init__(self, n: int, selective: bool = True) -> None:
        super().__init__()
        self.n = n
        self.selective = selective

        # TODO: build irrep catalog, CG matrices, and index map
        # 2D irreps rho_k for k = 1..floor((n-1)/2)
        # 1D irreps: rho_0 (trivial), rho_01, and for even n: rho_02, rho_03
        # Selective: floor((n-1)/2) + 2 matrix-valued coefficients (~4|D_n| scalars)
        self._index_map: list[tuple[int, ...]] = []

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Compute the D_n-bispectrum of a signal on D_n.

        Args:
            f: Real-valued signal. Shape: (batch, 2n).

        Returns:
            Complex bispectrum tensor. Shape: (batch, output_size).
        """
        # TODO: implement — group FT then bispectrum via CG matrices
        raise NotImplementedError('DnonDn.forward() not yet implemented.')

    def invert(self, beta: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Recover signal from bispectrum coefficients (up to D_n action).

        Implements Algorithm 3 from Mataigne et al. (2024).
        """
        # TODO: implement inversion bootstrap for D_n
        raise NotImplementedError('DnonDn.invert() not yet implemented.')

    @property
    def output_size(self) -> int:
        """Number of bispectral coefficients in the output."""
        return len(self._index_map)

    @property
    def index_map(self) -> list[tuple[int, ...]]:
        """Maps flat output index -> irrep index tuple."""
        return list(self._index_map)

    def extra_repr(self) -> str:
        return f'n={self.n}, selective={self.selective}, output_size={self.output_size}'
