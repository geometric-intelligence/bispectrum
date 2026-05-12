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
        nlat, nlon = 32, 64
        f = torch.randn(2, nlat, nlon, dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float64)
        f_rotated = rotate_spherical_function(f, R)
        # Exclude poles where interpolation is unreliable
        torch.testing.assert_close(f_rotated[:, 1:-1, :], f[:, 1:-1, :], atol=0.01, rtol=0.01)

    def test_output_shape(self):
        nlat, nlon = 32, 64
        f = torch.randn(3, nlat, nlon, dtype=torch.float64)
        R = random_rotation_matrix()
        f_rotated = rotate_spherical_function(f, R)
        assert f_rotated.shape == f.shape
