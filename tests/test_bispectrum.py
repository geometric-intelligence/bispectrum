"""Tests for bispectrum computation and rotation invariance.

These tests verify that the bispectrum is invariant under 3D rotations of the input spherical
function, as proven mathematically in the lecture notes.
"""

import pytest
import torch
from torch_harmonics import RealSHT

from bispectrum import (
    SO3onS2,
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


class TestSO3onS2:
    """Tests for SO3onS2 module."""

    def test_instantiation(self):
        """Test that SO3onS2 can be instantiated with default parameters."""
        bsp = SO3onS2()
        assert bsp.lmax == 5
        assert bsp.output_size > 0

    def test_instantiation_custom_lmax(self):
        """Test instantiation with custom lmax."""
        bsp = SO3onS2(lmax=3)
        assert bsp.lmax == 3

    def test_lmax_exceeds_json_raises(self):
        """Test that lmax exceeding JSON limits raises ValueError."""
        with pytest.raises(ValueError, match='exceed JSON limits'):
            SO3onS2(lmax=100)

    def test_index_map_structure(self):
        """Test that index_map contains valid (l1, l2, l) tuples."""
        bsp = SO3onS2(lmax=4)

        for l1, l2, l in bsp.index_map:
            # l1 <= l2 (by design)
            assert l1 <= l2
            # l in valid range
            assert abs(l1 - l2) <= l <= l1 + l2
            # l1, l2 within bounds
            assert l1 <= bsp.l1_max
            assert l2 <= bsp.l2_max
            # l is automatically <= lmax since l <= l1 + l2 <= l1_max + l2_max = lmax
            assert l <= bsp.lmax

    def test_output_size_matches_index_map(self):
        """Test that output_size equals len(index_map)."""
        bsp = SO3onS2(lmax=4)
        assert bsp.output_size == len(bsp.index_map)

    def test_cg_buffers_registered(self):
        """Test that CG matrices are registered as buffers."""
        bsp = SO3onS2(lmax=4)

        # Check that buffers exist for all (l1, l2) pairs with l1 <= l2 <= l1_max/l2_max
        for l1 in range(bsp.l1_max + 1):
            for l2 in range(l1, bsp.l2_max + 1):
                buffer_name = f'cg_{l1}_{l2}'
                assert hasattr(bsp, buffer_name), f'Missing buffer {buffer_name}'
                buffer = getattr(bsp, buffer_name)
                expected_size = (2 * l1 + 1) * (2 * l2 + 1)
                assert buffer.shape == (expected_size, expected_size)

    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size = 4
        lmax = 3
        # Input tensor needs shape (batch, lmax+1, mmax) to have coefficients for l=0..lmax
        tensor_lmax = lmax + 1

        bsp = SO3onS2(lmax=lmax)
        coeffs = torch.randn(batch_size, tensor_lmax, tensor_lmax, dtype=torch.complex128)

        output = bsp(coeffs)

        assert output.shape == (batch_size, bsp.output_size)
        assert output.dtype == torch.complex128

    def test_forward_deterministic(self):
        """Test that forward pass is deterministic."""
        batch_size = 2
        lmax = 3
        # Input tensor needs shape (batch, lmax+1, mmax) to have coefficients for l=0..lmax
        tensor_lmax = lmax + 1

        bsp = SO3onS2(lmax=lmax)
        coeffs = torch.randn(batch_size, tensor_lmax, tensor_lmax, dtype=torch.complex128)

        output1 = bsp(coeffs)
        output2 = bsp(coeffs)

        torch.testing.assert_close(output1, output2)

    def test_device_movement(self):
        """Test that module can be moved to different devices."""
        bsp = SO3onS2(lmax=2)

        # Check buffers are on CPU by default
        assert bsp.cg_0_0.device.type == 'cpu'

        # Move to CPU explicitly (should work even without GPU)
        bsp_cpu = bsp.to('cpu')
        assert bsp_cpu.cg_0_0.device.type == 'cpu'

    def test_extra_repr(self):
        """Test that extra_repr provides useful info."""
        bsp = SO3onS2(lmax=4)
        repr_str = repr(bsp)
        assert 'lmax=4' in repr_str
        assert 'output_size=' in repr_str

    def test_is_nn_module(self):
        """Test that SO3onS2 is a proper nn.Module."""
        import torch.nn as nn

        bsp = SO3onS2(lmax=2)
        assert isinstance(bsp, nn.Module)

        # Check that it has no trainable parameters (only buffers)
        assert sum(p.numel() for p in bsp.parameters()) == 0

        # Check that buffers are tracked
        buffer_count = sum(1 for _ in bsp.buffers())
        assert buffer_count > 0


class TestSO3onS2RotationInvariance:
    """Test that SO3onS2 bispectrum is invariant under rotations."""

    def test_rotation_invariance(self):
        """Test bispectrum rotation invariance using SO3onS2 module."""
        # Setup parameters
        nlat, nlon = 64, 128
        batch_size = 2
        lmax = 4  # Max degree for SO3onS2
        print('\n=== Shape debugging ===')
        print(f'nlat={nlat}, nlon={nlon}, batch_size={batch_size}, lmax={lmax}')

        # RealSHT with lmax=L outputs coefficients for l=0..L-1
        # So to have coefficients for l=0..lmax, we need sht_lmax = lmax + 1
        sht_lmax = lmax + 1
        print(f'sht_lmax={sht_lmax}')

        # Create SHT transform
        sht = RealSHT(nlat, nlon, lmax=sht_lmax, mmax=sht_lmax, grid='equiangular', norm='ortho')

        # Create SO3onS2 module
        bsp = SO3onS2(lmax=lmax)

        # Create a random real-valued spherical function
        f = torch.randn(batch_size, nlat, nlon, dtype=torch.float64)
        print(f'f (input spatial): {f.shape}')

        # Compute SH coefficients
        f_coeffs = sht(f.float()).to(torch.complex128)
        print(f'f_coeffs (SH coefficients): {f_coeffs.shape}')

        # Compute bispectrum
        beta_f = bsp(f_coeffs)
        print(f'beta_f (bispectrum): {beta_f.shape}')

        # Generate random rotation
        R = random_rotation_matrix()
        print(f'R (rotation matrix): {R.shape}')

        # Rotate the function
        f_rotated = rotate_spherical_function(f, R)
        print(f'f_rotated (rotated spatial): {f_rotated.shape}')

        # Compute SH coefficients of rotated function
        f_rotated_coeffs = sht(f_rotated.float()).to(torch.complex128)
        print(f'f_rotated_coeffs (rotated SH coefficients): {f_rotated_coeffs.shape}')

        # Compute bispectrum of rotated function
        beta_f_rotated = bsp(f_rotated_coeffs)
        print(f'beta_f_rotated (rotated bispectrum): {beta_f_rotated.shape}')
        print('=== End shapes ===\n')

        # Assert invariance (allowing for numerical tolerance due to interpolation)
        torch.testing.assert_close(
            beta_f.abs(),
            beta_f_rotated.abs(),
            atol=0.1,
            rtol=0.1,
            msg='Bispectrum magnitude should be approximately invariant under rotation',
        )
