"""SO(3) bispectrum on the unit ball B^3.

Implements the G-Bispectrum for SO(3) acting on the unit ball
B^3 = {(r, theta, phi) : r in [0,1]}.

The solid spherical harmonic basis is:
    Phi_{l,m}(r, theta, phi) = r^l * Y_l^m(theta, phi)

The radially-integrated SH coefficients are:
    A_{l,m} = integral_0^1 a_{l,m}(r) * r^{l+2} dr

where a_{l,m}(r) are the SH coefficients of f restricted to the shell
at radius r. The r^{l+2} factor combines the solid harmonic radial
weight r^l with the spherical Jacobian r^2.

The bispectrum formula is identical to SO3onS2:
    beta(f)_{l1,l2}^{(l)} = (A_l1 ⊗ A_l2) · C_{l1,l2} · A_l^†

Reference: Mathe & Miolane, "Bispectral Signatures of Data" (internal draft),
Section "Domain = (S^2 x R+, SO(3))".
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_harmonics import RealSHT

from bispectrum._cg import load_cg_matrices
from bispectrum.so3_on_s2 import _bispectrum_entry, _get_full_sh_coefficients


def _gauss_legendre_on_unit(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauss-Legendre quadrature nodes and weights on [0, 1].

    Args:
        n: Number of quadrature points.

    Returns:
        nodes: Tensor of shape (n,) with values in (0, 1).
        weights: Tensor of shape (n,) summing to 0.5 (integral of 1 over [0,1]).
    """
    import numpy as np

    np_nodes, np_weights = np.polynomial.legendre.leggauss(n)
    nodes = torch.tensor((np_nodes + 1.0) / 2.0, dtype=torch.float64)
    weights = torch.tensor(np_weights / 2.0, dtype=torch.float64)
    return nodes, weights


def _build_spherical_sample_grid(
    nr: int,
    radii: torch.Tensor,
    nlat: int,
    nlon: int,
) -> torch.Tensor:
    """Build a 3D sampling grid that maps spherical (r, theta, phi) points into normalized [-1,
    1]^3 Cartesian coordinates for grid_sample.

    The output grid has shape (nr * nlat, nlon, 3) representing (x, y, z)
    in [-1, 1]^3 coordinates, suitable for 3D grid_sample with
    input of shape (batch, 1, L, L, L).

    We treat the voxel grid as occupying [-1, 1]^3, so a point at
    Cartesian (x, y, z) with ||(x,y,z)|| <= 1 maps directly.

    Args:
        nr: Number of radial shells.
        radii: Radial positions, shape (nr,), values in (0, 1).
        nlat: Number of latitude points.
        nlon: Number of longitude points.

    Returns:
        grid: Tensor of shape (1, nr*nlat, nlon, 3) for use with
            5D grid_sample (batch, C, D, H, W) -> sample at (D, H, W).
    """
    theta = torch.linspace(0, math.pi, nlat, dtype=torch.float64)
    phi = torch.linspace(0, 2 * math.pi, nlon + 1, dtype=torch.float64)[:-1]

    # (nr, nlat, nlon) grids
    r_g, th_g, ph_g = torch.meshgrid(radii, theta, phi, indexing='ij')

    sin_th = torch.sin(th_g)
    x = r_g * sin_th * torch.cos(ph_g)
    y = r_g * sin_th * torch.sin(ph_g)
    z = r_g * torch.cos(th_g)

    # grid_sample expects (N, D_out, H_out, W_out, 3) with values in [-1,1]
    # We'll reshape (nr, nlat, nlon) -> (1, nr, nlat, nlon, 3)
    grid = torch.stack([x, y, z], dim=-1)  # (nr, nlat, nlon, 3)
    return grid.unsqueeze(0)  # (1, nr, nlat, nlon, 3)


class SO3onB3(nn.Module):
    """Bispectrum of SO(3) acting on the unit ball B^3.

    Takes a real-valued volumetric signal f of shape (batch, L, L, L),
    computes radially-integrated solid spherical harmonic coefficients
    internally, and returns the bispectrum coefficients.

    The domain B^3 = S^2 x [0,1] is not homogeneous for SO(3) — the
    radial coordinate is invariant under rotation. The Fourier basis
    consists of solid spherical harmonics Phi_{l,m}(r, theta, phi) =
    r^l Y_l^m(theta, phi).

    The bispectrum formula is identical to SO3onS2 (same CG matrices),
    applied to the radially-integrated coefficients A_{l,m}.

    Selective bispectrum and inversion are open problems (DESIGN.md
    TODO-M3, TODO-M4).

    Args:
        lmax: Maximum spherical harmonic degree.
        L: Voxel grid resolution. Input shape is (batch, L, L, L).
        nlat: Number of latitude points for the spherical grid.
            Defaults to 2*(lmax+1).
        nlon: Number of longitude points for the spherical grid.
            Defaults to 2*nlat.
        nr: Number of radial quadrature points. Defaults to L//2.
        selective: If True, raises NotImplementedError. Selective bispectrum
            for SO(3) on the ball is an open mathematical problem
            (DESIGN.md TODO-M3).
    """

    _SHT_GRID = 'equiangular'

    def __init__(
        self,
        lmax: int = 5,
        L: int = 32,
        nlat: int | None = None,
        nlon: int | None = None,
        nr: int | None = None,
        selective: bool = False,
    ) -> None:
        super().__init__()
        if selective:
            raise NotImplementedError(
                'Selective bispectrum for SO(3) on B^3 is an open mathematical '
                'problem. Use selective=False for the full bispectrum.'
            )
        self.lmax = lmax
        self.L = L
        self.selective = selective

        if nlat is None:
            nlat = 2 * (lmax + 1)
        if nlon is None:
            nlon = 2 * nlat
        if nr is None:
            nr = max(L // 2, 4)

        self.nlat = nlat
        self.nlon = nlon
        self.nr = nr

        sht_lmax = lmax + 1
        # The sampling grid in _build_spherical_sample_grid uses
        # linspace(0, pi, nlat), which matches the Clenshaw-Curtis
        # (equiangular) quadrature nodes. Changing _SHT_GRID requires
        # updating the sampling grid to match the new quadrature.
        self._sht = RealSHT(
            nlat,
            nlon,
            lmax=sht_lmax,
            mmax=sht_lmax,
            grid=self._SHT_GRID,
            norm='ortho',
        )

        radii, quad_weights = _gauss_legendre_on_unit(nr)
        self.register_buffer('_radii', radii)
        self.register_buffer('_quad_weights', quad_weights)

        # Radial weights: for each l, weight_i = quad_w_i * r_i^{l+2}
        # Shape: (lmax+1, nr)
        radial_weights = torch.zeros(lmax + 1, nr, dtype=torch.float64)
        for l_val in range(lmax + 1):
            radial_weights[l_val] = quad_weights * radii ** (l_val + 2)
        self.register_buffer('_radial_weights', radial_weights)

        sample_grid = _build_spherical_sample_grid(nr, radii, nlat, nlon)
        self.register_buffer('_sample_grid', sample_grid)

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

    def _cartesian_to_spherical_grid(self, f: torch.Tensor) -> torch.Tensor:
        """Resample a Cartesian voxel grid onto spherical shells.

        Args:
            f: Voxel data, shape (batch, L, L, L).

        Returns:
            Spherical shell data, shape (batch, nr, nlat, nlon).
        """
        batch = f.shape[0]
        # grid_sample needs 5D: (N, C, D, H, W)
        f_5d = f.unsqueeze(1).float()  # (batch, 1, L, L, L)

        grid = self._sample_grid.expand(batch, -1, -1, -1, -1).float()

        sampled = F.grid_sample(
            f_5d,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        # sampled: (batch, 1, nr, nlat, nlon)
        return sampled.squeeze(1).to(f.dtype)  # (batch, nr, nlat, nlon)

    def _solid_sht(self, f_shells: torch.Tensor) -> dict[int, torch.Tensor]:
        """Compute radially-integrated solid SH coefficients.

        Args:
            f_shells: Signal on spherical shells, shape (batch, nr, nlat, nlon).

        Returns:
            Dict mapping l -> A_l tensor of shape (batch, 2l+1), complex.
        """
        batch, nr = f_shells.shape[0], f_shells.shape[1]

        # SHT each shell: reshape to (batch*nr, nlat, nlon), apply SHT
        flat = f_shells.reshape(batch * nr, self.nlat, self.nlon)
        coeffs_flat = self._sht(flat.float())  # (batch*nr, lmax, mmax), complex
        # Reshape back: (batch, nr, lmax, mmax)
        sht_lmax = coeffs_flat.shape[-2]
        sht_mmax = coeffs_flat.shape[-1]
        coeffs = coeffs_flat.reshape(batch, nr, sht_lmax, sht_mmax)

        # Radial integration per l: A_{l,m} = sum_i coeffs[batch, i, l, m] * radial_weights[l, i]
        # Each degree l has its own radial weight profile because of the r^l factor.
        rw = self._radial_weights.to(device=coeffs.device, dtype=coeffs.dtype)
        integrated = torch.zeros(
            batch,
            sht_lmax,
            sht_mmax,
            dtype=coeffs.dtype,
            device=coeffs.device,
        )
        for l_val in range(min(self.lmax + 1, sht_lmax)):
            # w: (nr,), coeffs[:, :, l_val, :]: (batch, nr, mmax)
            w = rw[l_val]
            integrated[:, l_val, :] = torch.einsum(
                'r,brm->bm',
                w,
                coeffs[:, :, l_val, :],
            )

        return _get_full_sh_coefficients(integrated)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Compute the SO(3)-bispectrum of a signal on B^3.

        Args:
            f: Real-valued volumetric signal. Shape: (batch, L, L, L).

        Returns:
            Complex bispectrum tensor. Shape: (batch, output_size).
        """
        f_shells = self._cartesian_to_spherical_grid(f)
        f_coeffs = self._solid_sht(f_shells)

        batch_size = f.shape[0]
        result = torch.zeros(
            batch_size,
            self.output_size,
            dtype=torch.complex64,
            device=f.device,
        )

        for idx, (l1, l2, l) in enumerate(self._index_map):
            cg = getattr(self, f'cg_{l1}_{l2}')
            result[:, idx] = _bispectrum_entry(f_coeffs, l1, l2, l, cg)

        return result

    def invert(self, beta: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Inversion is an open problem for SO(3) on B^3.

        Raises:
            NotImplementedError: Selective bispectrum and inversion for
                SO(3) on the ball remain open mathematical problems.
        """
        raise NotImplementedError(
            'Inversion for SO(3) on B^3 is an open mathematical problem. '
            'See DESIGN.md TODO-M3 and TODO-M4.'
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
            f'lmax={self.lmax}, L={self.L}, nlat={self.nlat}, nlon={self.nlon}, '
            f'nr={self.nr}, output_size={self.output_size}'
        )
