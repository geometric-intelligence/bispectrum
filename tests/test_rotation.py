"""Tests for rotation utilities."""

import torch

from bispectrum import random_rotation_matrix, rotate_spherical_function


class TestRandomRotationMatrix:
    def test_is_orthogonal(self):
        R = random_rotation_matrix()
        identity = torch.eye(3, dtype=torch.float64)
        torch.testing.assert_close(R @ R.T, identity, atol=1e-10, rtol=1e-10)

    def test_determinant_is_one(self):
        R = random_rotation_matrix()
        det = torch.det(R)
        torch.testing.assert_close(
            det, torch.tensor(1.0, dtype=torch.float64), atol=1e-10, rtol=1e-10
        )

    def test_different_each_call(self):
        R1 = random_rotation_matrix()
        R2 = random_rotation_matrix()
        assert not torch.allclose(R1, R2)


class TestRotateSphericalFunction:
    def test_identity_rotation(self):
        """Identity rotation resamples at the grid nodes: exact to float64 rounding.

        The sampling grid is built in float64, so for R = I every sample point
        coincides with a grid node up to ~1e-16 and bilinear interpolation must
        return the node value.  Casting the grid to float32 before ``grid_sample``
        used to shift the sample points by ~1e-6 pixel and leak ~1e-5 errors.
        """
        nlat, nlon = 32, 64
        f = torch.randn(2, nlat, nlon, dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float64)
        f_rotated = rotate_spherical_function(f, R)
        assert f_rotated.dtype == torch.float64
        # Exclude poles where phi is undefined
        torch.testing.assert_close(f_rotated[:, 1:-1, :], f[:, 1:-1, :], atol=1e-12, rtol=0.0)

    def test_identity_rotation_float32_input(self):
        """Float32 input is interpolated in float32 and returned as float32."""
        nlat, nlon = 32, 64
        f = torch.randn(2, nlat, nlon, dtype=torch.float32)
        R = torch.eye(3, dtype=torch.float64)
        f_rotated = rotate_spherical_function(f, R)
        assert f_rotated.dtype == torch.float32
        torch.testing.assert_close(f_rotated[:, 1:-1, :], f[:, 1:-1, :], atol=1e-4, rtol=0.0)

    def test_rotation_inverse_roundtrip_float64(self):
        """R then R^T returns a smooth band-limited function to interpolation accuracy.

        Bilinear interpolation on a 64x128 grid is O(h^2) ~ 1e-3 for an l<=2 harmonic; float32
        casts would not change this bound, so the test only guards the float64 code path from
        regressing to something worse.
        """
        nlat, nlon = 64, 128
        theta = torch.linspace(0, torch.pi, nlat, dtype=torch.float64)
        phi = torch.linspace(0, 2 * torch.pi, nlon + 1, dtype=torch.float64)[:-1]
        th, ph = torch.meshgrid(theta, phi, indexing='ij')
        f = (torch.sin(th) * torch.cos(ph) + 0.5 * torch.cos(th) ** 2).unsqueeze(0)
        R = random_rotation_matrix()
        f_back = rotate_spherical_function(rotate_spherical_function(f, R), R.T)
        torch.testing.assert_close(f_back[:, 2:-2, :], f[:, 2:-2, :], atol=5e-3, rtol=0.0)

    def test_output_shape(self):
        nlat, nlon = 32, 64
        f = torch.randn(3, nlat, nlon, dtype=torch.float64)
        R = random_rotation_matrix()
        f_rotated = rotate_spherical_function(f, R)
        assert f_rotated.shape == f.shape
