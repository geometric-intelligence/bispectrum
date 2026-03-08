"""Tests for SO3onB3 bispectrum module."""

import pytest
import torch

from bispectrum import SO3onB3, random_rotation_matrix, rotate_voxel_function


class TestSO3onB3:
    """Tests for SO3onB3 module construction and properties."""

    def test_instantiation(self):
        bsp = SO3onB3(lmax=3, L=16)
        assert bsp.lmax == 3
        assert bsp.L == 16
        assert bsp.output_size > 0

    def test_instantiation_defaults(self):
        bsp = SO3onB3()
        assert bsp.lmax == 5
        assert bsp.L == 32
        assert bsp.nlat == 12
        assert bsp.nlon == 24

    def test_index_map_structure(self):
        bsp = SO3onB3(lmax=3, L=16)
        for l1, l2, l in bsp.index_map:
            assert l1 <= l2
            assert abs(l1 - l2) <= l <= l1 + l2
            assert l1 <= bsp.lmax
            assert l2 <= bsp.lmax
            assert l <= bsp.lmax

    def test_output_size_matches_index_map(self):
        bsp = SO3onB3(lmax=4, L=16)
        assert bsp.output_size == len(bsp.index_map)

    def test_cg_buffers_registered(self):
        bsp = SO3onB3(lmax=2, L=16)
        for l1 in range(3):
            for l2 in range(l1, 3):
                buffer_name = f'cg_{l1}_{l2}'
                assert hasattr(bsp, buffer_name), f'Missing buffer {buffer_name}'
                buffer = getattr(bsp, buffer_name)
                expected_size = (2 * l1 + 1) * (2 * l2 + 1)
                assert buffer.shape == (expected_size, expected_size)

    def test_no_trainable_parameters(self):
        bsp = SO3onB3(lmax=2, L=16)
        assert isinstance(bsp, torch.nn.Module)
        assert sum(p.numel() for p in bsp.parameters()) == 0
        assert sum(1 for _ in bsp.buffers()) > 0

    def test_extra_repr(self):
        bsp = SO3onB3(lmax=3, L=16)
        repr_str = repr(bsp)
        assert 'lmax=3' in repr_str
        assert 'L=16' in repr_str
        assert 'output_size=' in repr_str


class TestSO3onB3Forward:
    """Tests for the forward pass."""

    def test_output_shape(self):
        L = 16
        batch_size = 4
        lmax = 3
        bsp = SO3onB3(lmax=lmax, L=L)
        f = torch.randn(batch_size, L, L, L)
        output = bsp(f)
        assert output.shape == (batch_size, bsp.output_size)

    def test_output_is_complex(self):
        L = 16
        bsp = SO3onB3(lmax=3, L=L)
        f = torch.randn(2, L, L, L)
        output = bsp(f)
        assert output.is_complex()

    def test_forward_deterministic(self):
        L = 16
        bsp = SO3onB3(lmax=3, L=L)
        f = torch.randn(2, L, L, L)
        torch.testing.assert_close(bsp(f), bsp(f))

    def test_different_signals_differ(self):
        L = 16
        bsp = SO3onB3(lmax=3, L=L)
        f1 = torch.randn(1, L, L, L)
        f2 = torch.randn(1, L, L, L)
        out1 = bsp(f1)
        out2 = bsp(f2)
        assert not torch.allclose(out1, out2)

    def test_batch_size_one(self):
        L = 16
        bsp = SO3onB3(lmax=2, L=L)
        f = torch.randn(1, L, L, L)
        output = bsp(f)
        assert output.shape == (1, bsp.output_size)


class TestSO3onB3Invariance:
    """Test that SO3onB3 bispectrum is invariant under SO(3) rotations."""

    def test_rotation_invariance(self):
        L = 32
        lmax = 3
        bsp = SO3onB3(lmax=lmax, L=L, nr=16)

        torch.manual_seed(42)
        f = torch.randn(2, L, L, L)
        beta_f = bsp(f)

        R = random_rotation_matrix()
        f_rotated = rotate_voxel_function(f, R)
        beta_f_rotated = bsp(f_rotated)

        torch.testing.assert_close(
            beta_f.abs(),
            beta_f_rotated.abs(),
            atol=0.15,
            rtol=0.15,
            msg='Bispectrum magnitude should be approximately invariant under rotation',
        )


class TestSO3onB3Invert:
    """Test that invert raises NotImplementedError."""

    def test_invert_raises(self):
        bsp = SO3onB3(lmax=2, L=16)
        with pytest.raises(NotImplementedError, match='open mathematical problem'):
            bsp.invert(torch.zeros(1, bsp.output_size))
