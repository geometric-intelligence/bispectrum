"""Tests for SO2onS2 module."""

import torch
from torch_harmonics import RealSHT

from bispectrum import SO2onS2


class TestSO2onS2RotationInvariance:
    """Test that SO2onS2 bispectrum is invariant under z-axis rotations."""

    def test_z_axis_rotation_invariance(self):
        """Test SO2onS2 invariance via coefficient phase shifts."""
        nlat, nlon = 32, 64
        batch_size = 2
        lmax = 3
        sht_lmax = lmax + 1

        sht = RealSHT(nlat, nlon, lmax=sht_lmax, mmax=sht_lmax, grid='equiangular', norm='ortho')
        bsp = SO2onS2(lmax=lmax)

        f = torch.randn(batch_size, nlat, nlon, dtype=torch.float64)
        f_coeffs = sht(f.float()).to(torch.complex128)

        alpha = torch.rand(1).item() * 2 * torch.pi
        mmax = f_coeffs.shape[2]
        m = torch.arange(mmax, device=f_coeffs.device, dtype=torch.float64)
        phase = torch.exp(-1j * m * alpha).to(f_coeffs.dtype)
        f_coeffs_rot = f_coeffs * phase

        beta = bsp(f_coeffs)
        beta_rot = bsp(f_coeffs_rot)

        torch.testing.assert_close(beta, beta_rot, atol=5e-3, rtol=0.0)
