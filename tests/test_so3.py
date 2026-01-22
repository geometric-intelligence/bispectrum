"""Tests for SO3onS2 module."""

import pytest
import torch
from torch_harmonics import RealSHT

from bispectrum import SO3onS2, random_rotation_matrix, rotate_spherical_function


class TestSO3onS2:
    """Tests for SO3onS2 module."""

    def test_instantiation(self):
        """Test that SO3onS2 can be instantiated with default parameters."""
        bsp = SO3onS2()
        assert bsp.lmax == 10
        assert bsp.output_size > 0

    def test_instantiation_custom_lmax(self):
        """Test instantiation with custom lmax."""
        bsp = SO3onS2(lmax=3)
        assert bsp.lmax == 3

    def test_lmax_exceeds_cg_raises(self):
        """Test that lmax exceeding CG limits raises ValueError."""
        with pytest.raises(ValueError, match='exceeds CG limits'):
            SO3onS2(lmax=100)

    def test_index_map_structure(self):
        """Test that index_map contains valid (l1, l2, l) tuples."""
        bsp = SO3onS2(lmax=3)

        for l1, l2, l in bsp.index_map:
            assert l1 <= l2
            assert abs(l1 - l2) <= l <= l1 + l2
            assert l1 <= bsp.lmax
            assert l2 <= bsp.lmax
            assert l <= bsp.lmax

    def test_output_size_matches_index_map(self):
        """Test that output_size equals len(index_map)."""
        bsp = SO3onS2(lmax=4)
        assert bsp.output_size == len(bsp.index_map)

    def test_cg_buffers_registered(self):
        """Test that CG matrices are registered as buffers."""
        bsp = SO3onS2(lmax=2)

        for l1 in range(3):
            for l2 in range(l1, 3):
                buffer_name = f'cg_{l1}_{l2}'
                assert hasattr(bsp, buffer_name), f'Missing buffer {buffer_name}'
                buffer = getattr(bsp, buffer_name)
                expected_size = (2 * l1 + 1) * (2 * l2 + 1)
                assert buffer.shape == (expected_size, expected_size)

    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size = 4
        lmax = 3
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
        tensor_lmax = lmax + 1

        bsp = SO3onS2(lmax=lmax)
        coeffs = torch.randn(batch_size, tensor_lmax, tensor_lmax, dtype=torch.complex128)

        output1 = bsp(coeffs)
        output2 = bsp(coeffs)

        torch.testing.assert_close(output1, output2)

    def test_device_movement(self):
        """Test that module can be moved to different devices."""
        bsp = SO3onS2(lmax=2)

        assert bsp.cg_0_0.device.type == 'cpu'

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

        assert sum(p.numel() for p in bsp.parameters()) == 0

        buffer_count = sum(1 for _ in bsp.buffers())
        assert buffer_count > 0


class TestSO3onS2RotationInvariance:
    """Test that SO3onS2 bispectrum is invariant under rotations."""

    def test_rotation_invariance(self):
        """Test bispectrum rotation invariance using SO3onS2 module."""
        nlat, nlon = 64, 128
        batch_size = 2
        lmax = 4

        sht_lmax = lmax + 1

        sht = RealSHT(nlat, nlon, lmax=sht_lmax, mmax=sht_lmax, grid='equiangular', norm='ortho')
        bsp = SO3onS2(lmax=lmax)

        f = torch.randn(batch_size, nlat, nlon, dtype=torch.float64)
        f_coeffs = sht(f.float()).to(torch.complex128)

        beta_f = bsp(f_coeffs)

        R = random_rotation_matrix()

        f_rotated = rotate_spherical_function(f, R)
        f_rotated_coeffs = sht(f_rotated.float()).to(torch.complex128)

        beta_f_rotated = bsp(f_rotated_coeffs)

        torch.testing.assert_close(
            beta_f.abs(),
            beta_f_rotated.abs(),
            atol=0.1,
            rtol=0.1,
            msg='Bispectrum magnitude should be approximately invariant under rotation',
        )
