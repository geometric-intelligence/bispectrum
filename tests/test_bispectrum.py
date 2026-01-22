"""Tests for bispectrum computation and rotation invariance.

These tests verify that the bispectrum is invariant under 3D rotations of the input spherical
function, as proven mathematically in the lecture notes.
"""

import pytest
import torch
from torch_harmonics import RealSHT

from bispectrum import (
    bispectrum,
    compute_padding_indices,
    get_full_sh_coefficients,
    pad_sh_coefficients,
    random_rotation_matrix,
    rotate_spherical_function,
)
from bispectrum.clebsch_gordan import clebsch_gordan


class TestGetFullSHCoefficients:
    """Tests for get_full_sh_coefficients function."""

    def test_output_shape(self):
        """Test that output has correct shapes for each l."""
        batch_size = 4
        lmax = 5
        mmax = 5
        coeffs = torch.randn(batch_size, lmax, mmax, dtype=torch.complex128)

        result = get_full_sh_coefficients(coeffs)

        for l in range(lmax):
            assert l in result
            assert result[l].shape == (batch_size, 2 * l + 1)

    def test_m_zero_preserved(self):
        """Test that m=0 coefficients are preserved unchanged."""
        batch_size = 2
        lmax = 4
        mmax = 4
        coeffs = torch.randn(batch_size, lmax, mmax, dtype=torch.complex128)

        result = get_full_sh_coefficients(coeffs)

        for l in range(lmax):
            # m=0 is at index l in the full array
            torch.testing.assert_close(result[l][:, l], coeffs[:, l, 0])

    def test_conjugate_symmetry(self):
        """Test that F_l^{-m} = (-1)^m * conj(F_l^m)."""
        batch_size = 2
        lmax = 4
        mmax = 4
        coeffs = torch.randn(batch_size, lmax, mmax, dtype=torch.complex128)

        result = get_full_sh_coefficients(coeffs)

        for l in range(1, lmax):
            for m in range(1, min(l, mmax - 1) + 1):
                # Positive m at index l + m
                f_pos = result[l][:, l + m]
                # Negative m at index l - m
                f_neg = result[l][:, l - m]
                # Check relation
                expected = ((-1) ** m) * torch.conj(f_pos)
                torch.testing.assert_close(f_neg, expected)


class TestComputePaddingIndices:
    """Tests for compute_padding_indices function."""

    def test_sum_equals_total(self):
        """Test that n_p + (2l+1) + n_s = (2l1+1)(2l2+1)."""
        for l1 in range(5):
            for l2 in range(5):
                total = (2 * l1 + 1) * (2 * l2 + 1)
                for l in range(abs(l1 - l2), l1 + l2 + 1):
                    n_p, n_s = compute_padding_indices(l1, l2, l)
                    assert n_p + (2 * l + 1) + n_s == total

    def test_first_block(self):
        """Test that first block (l = |l1-l2|) has n_p = 0."""
        for l1 in range(5):
            for l2 in range(5):
                l_min = abs(l1 - l2)
                n_p, _ = compute_padding_indices(l1, l2, l_min)
                assert n_p == 0

    def test_last_block(self):
        """Test that last block (l = l1+l2) has n_s = 0."""
        for l1 in range(5):
            for l2 in range(5):
                l_max = l1 + l2
                _, n_s = compute_padding_indices(l1, l2, l_max)
                assert n_s == 0


class TestPadSHCoefficients:
    """Tests for pad_sh_coefficients function."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size = 3
        l1, l2, l = 2, 3, 2
        f_l = torch.randn(batch_size, 2 * l + 1, dtype=torch.complex128)

        result = pad_sh_coefficients(f_l, l1, l2, l)

        expected_size = (2 * l1 + 1) * (2 * l2 + 1)
        assert result.shape == (batch_size, expected_size)

    def test_coefficients_placed_correctly(self):
        """Test that coefficients are placed at correct position."""
        batch_size = 2
        l1, l2, l = 2, 2, 2
        f_l = torch.randn(batch_size, 2 * l + 1, dtype=torch.complex128)

        result = pad_sh_coefficients(f_l, l1, l2, l)

        n_p, n_s = compute_padding_indices(l1, l2, l)
        # Check that coefficients are in the right place
        torch.testing.assert_close(result[:, n_p : n_p + 2 * l + 1], f_l)
        # Check that padding is zeros
        if n_p > 0:
            assert torch.all(result[:, :n_p] == 0)
        if n_s > 0:
            assert torch.all(result[:, n_p + 2 * l + 1 :] == 0)


class TestClebschGordanPlaceholder:
    """Tests for clebsch_gordan placeholder."""

    def test_raises_not_implemented(self):
        """Test that placeholder raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            clebsch_gordan(1, 1)


class TestRandomRotationMatrix:
    """Tests for random_rotation_matrix function."""

    def test_is_orthogonal(self):
        """Test that R @ R^T = I."""
        R = random_rotation_matrix()
        identity = torch.eye(3, dtype=torch.float64)
        torch.testing.assert_close(R @ R.T, identity, atol=1e-10, rtol=1e-10)

    def test_determinant_is_one(self):
        """Test that det(R) = +1 (proper rotation)."""
        R = random_rotation_matrix()
        det = torch.det(R)
        torch.testing.assert_close(
            det, torch.tensor(1.0, dtype=torch.float64), atol=1e-10, rtol=1e-10
        )

    def test_different_each_call(self):
        """Test that different calls produce different matrices."""
        R1 = random_rotation_matrix()
        R2 = random_rotation_matrix()
        # Very unlikely to be equal
        assert not torch.allclose(R1, R2)


class TestRotateSphericalFunction:
    """Tests for rotate_spherical_function."""

    def test_identity_rotation(self):
        """Test that identity rotation preserves the function away from poles."""
        nlat, nlon = 32, 64
        batch_size = 2
        f = torch.randn(batch_size, nlat, nlon, dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float64)

        f_rotated = rotate_spherical_function(f, R)

        # Exclude poles (first and last latitude rows) where phi is undefined
        # and interpolation is unreliable
        f_interior = f[:, 1:-1, :]
        f_rotated_interior = f_rotated[:, 1:-1, :]

        # Should be approximately equal (some interpolation error expected)
        torch.testing.assert_close(f_rotated_interior, f_interior, atol=0.01, rtol=0.01)

    def test_output_shape(self):
        """Test that output has same shape as input."""
        nlat, nlon = 32, 64
        batch_size = 3
        f = torch.randn(batch_size, nlat, nlon, dtype=torch.float64)
        R = random_rotation_matrix()

        f_rotated = rotate_spherical_function(f, R)

        assert f_rotated.shape == f.shape


class TestBispectrumRotationInvariance:
    """Test that bispectrum is invariant under rotations.

    NOTE: This test requires a working Clebsch-Gordan implementation.
    It is marked as expected to fail until CG coefficients are provided.
    """

    @pytest.mark.xfail(reason='Clebsch-Gordan coefficients not yet implemented')
    def test_bispectrum_rotation_invariance(self):
        """Test that bispectrum is invariant under rotations.

        Steps:
        1. Create a random spherical function f
        2. Compute bispectrum beta(f)
        3. Generate random rotation R
        4. Rotate f -> R*f
        5. Compute bispectrum beta(R*f)
        6. Assert beta(f) ≈ beta(R*f) for all (l1, l2, l)
        """
        # Setup parameters
        nlat, nlon = 64, 128
        batch_size = 2
        lmax = 8  # Maximum degree to test
        l1, l2 = 2, 3  # Test specific degree pair

        # Create SHT transform
        # REMINDER: RealSHT outputs COMPLEX coefficients for a REAL function!
        sht = RealSHT(nlat, nlon, lmax=lmax, mmax=lmax, grid='equiangular', norm='ortho')

        # Create a random real-valued spherical function
        f = torch.randn(batch_size, nlat, nlon, dtype=torch.float64)

        # Compute SH coefficients (complex, for m >= 0)
        f_coeffs_raw = sht(f.float()).to(torch.complex128)

        # Extend to all m values
        f_coeffs = get_full_sh_coefficients(f_coeffs_raw)

        # Compute bispectrum
        beta_f = bispectrum(f_coeffs, l1, l2, clebsch_gordan)

        # Generate random rotation
        R = random_rotation_matrix()

        # Rotate the function
        f_rotated = rotate_spherical_function(f, R)

        # Compute SH coefficients of rotated function
        f_rotated_coeffs_raw = sht(f_rotated.float()).to(torch.complex128)
        f_rotated_coeffs = get_full_sh_coefficients(f_rotated_coeffs_raw)

        # Compute bispectrum of rotated function
        beta_f_rotated = bispectrum(f_rotated_coeffs, l1, l2, clebsch_gordan)

        # Assert invariance
        torch.testing.assert_close(
            beta_f,
            beta_f_rotated,
            atol=1e-3,
            rtol=1e-3,
            msg='Bispectrum should be invariant under rotation',
        )
