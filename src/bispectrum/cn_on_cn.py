"""Cyclic group bispectrum on Z/nZ.

Implements the G-Bispectrum for the cyclic group C_n acting on itself.

For signal f: Z/nZ -> R:
    beta(f)_{k1,k2} = f_hat[k1] * f_hat[k2] * conj(f_hat[(k1+k2) mod n])

where f_hat is the DFT of f.

Reference: Kakarala (2009), Mataigne et al. (2024) Algorithms 1-2.
"""

import torch
import torch.nn as nn


class CnonCn(nn.Module):
    """Bispectrum of C_n acting on Z/nZ.

    C_n is commutative, so no Clebsch-Gordan matrices are needed — the
    bispectrum reduces to scalar triple products of DFT coefficients.

    Args:
        n: Group order / signal length.
        selective: If True, use selective O(n) bispectrum (Algorithm 1).
            If False, use full O(n^2) bispectrum.
    """

    def __init__(self, n: int, selective: bool = True) -> None:
        super().__init__()
        self.n = n
        self.selective = selective

        if selective:
            # Algorithm 1 (Mataigne 2024): {(0,0), (0,1), (1,1), (1,2), ..., (1,n-2)}
            self._index_map: list[tuple[int, int]] = [(0, 0), (0, 1)] + [
                (1, k) for k in range(1, n - 1)
            ]
        else:
            self._index_map = [(k1, k2) for k1 in range(n) for k2 in range(n)]

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Compute the C_n-bispectrum of a signal on Z/nZ.

        Args:
            f: Real-valued signal. Shape: (batch, n).

        Returns:
            Complex bispectrum tensor. Shape: (batch, output_size).
        """
        # TODO: implement — DFT then triple products over self._index_map
        raise NotImplementedError('CnonCn.forward() not yet implemented.')

    def invert(self, beta: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Recover signal from bispectrum coefficients (up to cyclic shift).

        Implements Algorithm 1 / Algorithm 2 from Mataigne et al. (2024).
        """
        # TODO: implement inversion bootstrap
        raise NotImplementedError('CnonCn.invert() not yet implemented.')

    @property
    def output_size(self) -> int:
        """Number of bispectral coefficients in the output."""
        return len(self._index_map)

    @property
    def index_map(self) -> list[tuple[int, int]]:
        """Maps flat output index -> (k1, k2) frequency pair."""
        return list(self._index_map)

    def extra_repr(self) -> str:
        return f'n={self.n}, selective={self.selective}, output_size={self.output_size}'
