"""Cyclic group bispectrum on Z/nZ.

Implements the G-Bispectrum for the cyclic group C_n acting on itself.

For signal f: Z/nZ -> R:
    beta(f)_{k1,k2} = f_hat[k1] * f_hat[k2] * conj(f_hat[(k1+k2) mod n])

where f_hat is the DFT of f.

Reference: Kakarala (2009), Mataigne et al. (2024) Algorithms 2-3.
"""

import torch
import torch.nn as nn


class CnonCn(nn.Module):
    """Bispectrum of C_n acting on Z/nZ.

    C_n is commutative, so no Clebsch-Gordan matrices are needed — the
    bispectrum reduces to scalar triple products of DFT coefficients.

    Reference: Mataigne et al., "The Selective G-Bispectrum and its Inversion:
    Applications to G-Invariant Networks", NeurIPS 2024.
    Forward uses Theorem 4.1; inversion uses Algorithm 2 (Appendix C).

    Args:
        n: Group order / signal length.
        selective: If True, use selective O(n) bispectrum.
            If False, use full O(n^2) bispectrum.
    """

    def __init__(self, n: int, selective: bool = True) -> None:
        super().__init__()
        self.n = n
        self.selective = selective

        if selective:
            # Selective indices: {(0,0), (0,1), (1,1), (1,2), ..., (1,n-2)}
            self._index_map: list[tuple[int, int]] = [(0, 0), (0, 1)] + [
                (1, k) for k in range(1, n - 1)
            ]
        else:
            self._index_map = [(k1, k2) for k1 in range(n) for k2 in range(k1, n)]

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Compute the C_n-bispectrum of a signal on Z/nZ.

        Args:
            f: Real-valued signal. Shape: (batch, n).

        Returns:
            Complex bispectrum tensor. Shape: (batch, output_size).
        """
        n = self.n
        fhat = torch.fft.fft(f, dim=-1)  # (batch, n), complex

        if self.selective:
            return self._forward_selective(fhat, n)
        return self._forward_full(fhat, n)

    def _forward_selective(self, fhat: torch.Tensor, n: int) -> torch.Tensor:
        """Selective bispectrum: n coefficients sufficient for inversion.

        Indices: {(0,0), (0,1), (1,1), (1,2), ..., (1,n-2)}.
        """
        batch = fhat.shape[0]
        beta = torch.zeros(batch, n, dtype=fhat.dtype, device=fhat.device)
        beta[:, 0] = fhat[:, 0] * fhat[:, 0] * torch.conj(fhat[:, 0])
        beta[:, 1] = fhat[:, 0] * fhat[:, 1] * torch.conj(fhat[:, 1])
        beta[:, 2:] = fhat[:, 1:2] * fhat[:, 1 : n - 1] * torch.conj(fhat[:, 2:n])
        return beta

    def _forward_full(self, fhat: torch.Tensor, n: int) -> torch.Tensor:
        """Full bispectrum: upper-triangular n*(n+1)/2 coefficients."""
        batch = fhat.shape[0]
        out_size = n * (n + 1) // 2
        beta = torch.zeros(batch, out_size, dtype=fhat.dtype, device=fhat.device)

        idx = 0
        for k1 in range(n):
            k2_range = torch.arange(k1, n, device=fhat.device)
            count = len(k2_range)
            beta[:, idx : idx + count] = (
                fhat[:, k1 : k1 + 1] * fhat[:, k1:n] * torch.conj(fhat[:, (k2_range + k1) % n])
            )
            idx += count

        return beta

    def invert(self, beta: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Recover the signal from selective bispectrum coefficients.

        Implements Algorithm 2 (Appendix C) from Mataigne et al., NeurIPS 2024.
        The recovered signal is determined up to a continuous SO(2) phase
        indeterminacy (not just discrete C_n shifts).

        Args:
            beta: Selective bispectrum. Shape: (batch, n), complex.

        Returns:
            Reconstructed signal (complex). Shape: (batch, n).
            Related to the original by a continuous cyclic shift.

        Raises:
            NotImplementedError: If selective=False.
        """
        if not self.selective:
            raise NotImplementedError(
                'Inversion is only implemented for the selective bispectrum. Use selective=True.'
            )

        n = self.n
        batch = beta.shape[0]
        fhat = torch.zeros(batch, n, dtype=beta.dtype, device=beta.device)

        # Step 1: recover fhat[0] from beta_{0,0} = |fhat[0]|^3 * exp(i*arg(fhat[0]))
        fhat[:, 0] = torch.abs(beta[:, 0]) ** (1.0 / 3.0) * torch.exp(1j * torch.angle(beta[:, 0]))

        # Step 2: recover |fhat[1]| from beta_{0,1} = fhat[0]*|fhat[1]|^2
        # Phase of fhat[1] fixed to 0 (absorbed by the SO(2) indeterminacy)
        fhat[:, 1] = torch.sqrt(torch.abs(beta[:, 1] / fhat[:, 0]))

        # Step 3: sequential recovery for k = 1..n-2
        # beta[k+1] corresponds to beta_{1,k}, and
        # fhat[k+1] = conj(beta_{1,k} / (fhat[1] * fhat[k]))
        for k in range(1, n - 1):
            fhat[:, k + 1] = torch.conj(beta[:, k + 1] / (fhat[:, 1] * fhat[:, k]))

        return torch.fft.ifft(fhat, dim=-1)

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
