"""SO(3) bispectrum on S^2.

Implements the G-Bispectrum for SO(3) acting on the 2-sphere.

beta(f)_{l1,l2}^{(l)} = (F_l1 ⊗ F_l2) · C_{l1,l2} · F_l^†

where F_l are the degree-l SH coefficient vectors and C denotes the
Clebsch-Gordan matrices for SO(3).

Reference: Kakarala (1992), Cohen et al.
"""

import torch
import torch.nn as nn
from torch_harmonics import RealSHT

from bispectrum._cg import load_cg_matrices


class SO3onS2(nn.Module):
    """Bispectrum of SO(3) acting on S^2.

    Takes a real-valued signal on the sphere f(theta, phi) sampled on an
    equiangular grid, computes the spherical harmonic transform internally,
    and returns the bispectrum coefficients.

    Args:
        lmax: Maximum spherical harmonic degree.
        nlat: Number of latitude grid points.
        nlon: Number of longitude grid points.
        selective: Reserved for future use. Selective bispectrum for SO(3)
            is an open problem (see DESIGN.md TODO-M1).
    """

    def __init__(
        self,
        lmax: int = 5,
        nlat: int = 64,
        nlon: int = 128,
        selective: bool = True,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.nlat = nlat
        self.nlon = nlon
        self.selective = selective

        # RealSHT with lmax=L outputs coefficients for l=0..L-1,
        # so we need sht_lmax = lmax + 1 to get l=0..lmax.
        sht_lmax = lmax + 1
        self._sht = RealSHT(
            nlat, nlon, lmax=sht_lmax, mmax=sht_lmax, grid='equiangular', norm='ortho'
        )

        # Load CG matrices and register as buffers
        cg_data = load_cg_matrices(lmax)
        self._index_map: list[tuple[int, int, int]] = []

        for l1 in range(lmax + 1):
            for l2 in range(l1, lmax + 1):
                key = (l1, l2)
                if key not in cg_data:
                    continue
                self.register_buffer(f'cg_{l1}_{l2}', cg_data[key])
                for l in range(abs(l1 - l2), min(l1 + l2, lmax) + 1):
                    self._index_map.append((l1, l2, l))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Compute the SO(3)-bispectrum of a signal on S^2.

        Args:
            f: Real-valued signal on S^2. Shape: (batch, nlat, nlon).

        Returns:
            Complex bispectrum tensor. Shape: (batch, output_size).
        """
        coeffs = self._sht(f)
        f_coeffs = _get_full_sh_coefficients(coeffs)

        batch_size = f.shape[0]
        result = torch.zeros(batch_size, self.output_size, dtype=coeffs.dtype, device=f.device)

        for idx, (l1, l2, l) in enumerate(self._index_map):
            cg = getattr(self, f'cg_{l1}_{l2}')
            result[:, idx] = _bispectrum_entry(f_coeffs, l1, l2, l, cg)

        return result

    def invert(self, beta: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Inversion is an open problem for SO(3) on S^2.

        Raises:
            NotImplementedError: Selective bispectrum and inversion for
                continuous groups remain open mathematical problems.
        """
        raise NotImplementedError(
            'Inversion for SO(3) on S^2 is an open mathematical problem. '
            'See DESIGN.md TODO-M1 and TODO-M4.'
        )

    @property
    def output_size(self) -> int:
        """Number of bispectral coefficients in the output."""
        return len(self._index_map)

    @property
    def index_map(self) -> list[tuple[int, int, int]]:
        """Maps flat output index -> (l1, l2, l) triple."""
        return list(self._index_map)

    def extra_repr(self) -> str:
        return (
            f'lmax={self.lmax}, nlat={self.nlat}, nlon={self.nlon}, output_size={self.output_size}'
        )


# ---------------------------------------------------------------------------
# Private helpers (folded from the old spherical.py)
# ---------------------------------------------------------------------------


def _get_full_sh_coefficients(
    coeffs_positive_m: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Extend SHT output (m >= 0) to full m range using F_l^{-m} = (-1)^m conj(F_l^m)."""
    batch_size, lmax, mmax = coeffs_positive_m.shape
    result: dict[int, torch.Tensor] = {}

    for l_val in range(lmax):
        m_max_for_l = min(l_val, mmax - 1)
        full_coeffs = torch.zeros(
            batch_size,
            2 * l_val + 1,
            dtype=coeffs_positive_m.dtype,
            device=coeffs_positive_m.device,
        )

        full_coeffs[:, l_val] = coeffs_positive_m[:, l_val, 0]

        for m in range(1, m_max_for_l + 1):
            full_coeffs[:, l_val + m] = coeffs_positive_m[:, l_val, m]
            full_coeffs[:, l_val - m] = ((-1) ** m) * torch.conj(coeffs_positive_m[:, l_val, m])

        result[l_val] = full_coeffs

    return result


def _compute_padding_indices(l1: int, l2: int, l_val: int) -> tuple[int, int]:
    """Compute (n_p, n_s) zero-padding sizes for F_l in the coupled basis."""
    l_min = abs(l1 - l2)
    l_max = l1 + l2
    n_p = sum(2 * q + 1 for q in range(l_min, l_val))
    n_s = sum(2 * q + 1 for q in range(l_val + 1, l_max + 1))
    return n_p, n_s


def _pad_sh_coefficients(f_l: torch.Tensor, l1: int, l2: int, l_val: int) -> torch.Tensor:
    """Zero-pad F_l to size (2l1+1)(2l2+1) in the coupled basis."""
    batch_size = f_l.shape[0]
    total_size = (2 * l1 + 1) * (2 * l2 + 1)
    n_p, _ = _compute_padding_indices(l1, l2, l_val)
    padded = torch.zeros(batch_size, total_size, dtype=f_l.dtype, device=f_l.device)
    padded[:, n_p : n_p + 2 * l_val + 1] = f_l
    return padded


def _bispectrum_entry(
    f_coeffs: dict[int, torch.Tensor],
    l1: int,
    l2: int,
    l_val: int,
    cg_matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute scalar bispectrum entry beta_{l1,l2}[l].

    beta = (F_l1 ⊗ F_l2) @ C_{l1,l2} @ F_hat_l^†
    """
    f_l1 = f_coeffs[l1]
    f_l2 = f_coeffs[l2]

    if l_val not in f_coeffs:
        return torch.zeros(f_l1.shape[0], dtype=f_l1.dtype, device=f_l1.device)

    batch_size = f_l1.shape[0]

    outer = torch.einsum('bi,bj->bij', f_l1, f_l2)
    tensor_product = outer.reshape(batch_size, -1)

    cg = cg_matrix.to(device=tensor_product.device, dtype=tensor_product.dtype)
    transformed = tensor_product @ cg

    f_hat_l = _pad_sh_coefficients(f_coeffs[l_val], l1, l2, l_val)
    return torch.sum(transformed * torch.conj(f_hat_l), dim=-1)
