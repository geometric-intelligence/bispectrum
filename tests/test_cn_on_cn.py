"""Tests for CnonCn bispectrum module."""

import pytest
import torch

from bispectrum import CnonCn


class TestCnonCn:
    """Tests for CnonCn stub module."""

    def test_instantiation(self):
        bsp = CnonCn(n=8)
        assert bsp.n == 8
        assert bsp.selective is True

    def test_instantiation_full(self):
        bsp = CnonCn(n=8, selective=False)
        assert bsp.selective is False

    def test_selective_output_size(self):
        n = 8
        bsp = CnonCn(n=n, selective=True)
        assert bsp.output_size == n

    def test_full_output_size(self):
        n = 8
        bsp = CnonCn(n=n, selective=False)
        assert bsp.output_size == n * n

    def test_selective_index_map(self):
        bsp = CnonCn(n=8, selective=True)
        idx = bsp.index_map
        assert idx[0] == (0, 0)
        assert idx[1] == (0, 1)
        assert idx[2] == (1, 1)
        assert idx[-1] == (1, 6)

    def test_no_trainable_parameters(self):
        bsp = CnonCn(n=8)
        assert isinstance(bsp, torch.nn.Module)
        assert sum(p.numel() for p in bsp.parameters()) == 0

    def test_forward_not_implemented(self):
        bsp = CnonCn(n=8)
        with pytest.raises(NotImplementedError):
            bsp(torch.randn(2, 8))

    def test_invert_not_implemented(self):
        bsp = CnonCn(n=8)
        with pytest.raises(NotImplementedError):
            bsp.invert(torch.randn(2, 8))

    def test_extra_repr(self):
        bsp = CnonCn(n=8)
        assert 'n=8' in repr(bsp)
        assert 'selective=True' in repr(bsp)
