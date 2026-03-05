"""Tests for SO3onS2 bispectrum module."""

import pytest
import torch

from bispectrum import SO3onS2, random_rotation_matrix, rotate_spherical_function


class TestSO3onS2:
    """Tests for SO3onS2 module."""

    def test_instantiation(self):
        bsp = SO3onS2(lmax=3, nlat=32, nlon=64)
        assert bsp.lmax == 3
        assert bsp.nlat == 32
        assert bsp.nlon == 64
        assert bsp.output_size > 0

    def test_instantiation_defaults(self):
        bsp = SO3onS2()
        assert bsp.lmax == 5
        assert bsp.nlat == 64
        assert bsp.nlon == 128

    def test_lmax_exceeds_json_raises(self):
        with pytest.raises(ValueError, match='exceeds JSON limits'):
            SO3onS2(lmax=100, nlat=64, nlon=128)

    def test_index_map_structure(self):
        bsp = SO3onS2(lmax=3, nlat=32, nlon=64)
        for l1, l2, l in bsp.index_map:
            assert l1 <= l2
            assert abs(l1 - l2) <= l <= l1 + l2
            assert l1 <= bsp.lmax
            assert l2 <= bsp.lmax
            assert l <= bsp.lmax

    def test_output_size_matches_index_map(self):
        bsp = SO3onS2(lmax=4, nlat=32, nlon=64)
        assert bsp.output_size == len(bsp.index_map)

    def test_cg_buffers_registered(self):
        bsp = SO3onS2(lmax=2, nlat=32, nlon=64)
        for l1 in range(3):
            for l2 in range(l1, 3):
                buffer_name = f'cg_{l1}_{l2}'
                assert hasattr(bsp, buffer_name), f'Missing buffer {buffer_name}'
                buffer = getattr(bsp, buffer_name)
                expected_size = (2 * l1 + 1) * (2 * l2 + 1)
                assert buffer.shape == (expected_size, expected_size)

    def test_forward_output_shape(self):
        nlat, nlon = 32, 64
        batch_size = 4
        lmax = 3
        bsp = SO3onS2(lmax=lmax, nlat=nlat, nlon=nlon)
        f = torch.randn(batch_size, nlat, nlon)
        output = bsp(f)
        assert output.shape == (batch_size, bsp.output_size)
        assert output.is_complex()

    def test_forward_deterministic(self):
        nlat, nlon = 32, 64
        bsp = SO3onS2(lmax=3, nlat=nlat, nlon=nlon)
        f = torch.randn(2, nlat, nlon)
        torch.testing.assert_close(bsp(f), bsp(f))

    def test_device_movement(self):
        bsp = SO3onS2(lmax=2, nlat=32, nlon=64)
        assert bsp.cg_0_0.device.type == 'cpu'
        bsp_cpu = bsp.to('cpu')
        assert bsp_cpu.cg_0_0.device.type == 'cpu'

    def test_no_trainable_parameters(self):
        bsp = SO3onS2(lmax=2, nlat=32, nlon=64)
        assert isinstance(bsp, torch.nn.Module)
        assert sum(p.numel() for p in bsp.parameters()) == 0
        assert sum(1 for _ in bsp.buffers()) > 0

    def test_invert_raises(self):
        bsp = SO3onS2(lmax=2, nlat=32, nlon=64)
        with pytest.raises(NotImplementedError, match='open mathematical problem'):
            bsp.invert(torch.zeros(1, bsp.output_size))

    def test_extra_repr(self):
        bsp = SO3onS2(lmax=4, nlat=32, nlon=64)
        repr_str = repr(bsp)
        assert 'lmax=4' in repr_str
        assert 'nlat=32' in repr_str
        assert 'output_size=' in repr_str


class TestSO3onS2RotationInvariance:
    """Test that SO3onS2 bispectrum is invariant under rotations."""

    def test_rotation_invariance(self):
        nlat, nlon = 64, 128
        batch_size = 2
        lmax = 4
        bsp = SO3onS2(lmax=lmax, nlat=nlat, nlon=nlon)

        f = torch.randn(batch_size, nlat, nlon, dtype=torch.float64)
        beta_f = bsp(f.float())

        R = random_rotation_matrix()
        f_rotated = rotate_spherical_function(f, R)
        beta_f_rotated = bsp(f_rotated.float())

        torch.testing.assert_close(
            beta_f.abs(),
            beta_f_rotated.abs(),
            atol=0.1,
            rtol=0.1,
            msg='Bispectrum magnitude should be approximately invariant under rotation',
        )
