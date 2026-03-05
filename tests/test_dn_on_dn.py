"""Tests for DnonDn bispectrum module."""

import pytest
import torch

from bispectrum import DnonDn


class TestDnonDn:
    """Tests for DnonDn stub module."""

    def test_instantiation(self):
        bsp = DnonDn(n=4)
        assert bsp.n == 4
        assert bsp.selective is True

    def test_instantiation_full(self):
        bsp = DnonDn(n=4, selective=False)
        assert bsp.selective is False

    def test_no_trainable_parameters(self):
        bsp = DnonDn(n=4)
        assert isinstance(bsp, torch.nn.Module)
        assert sum(p.numel() for p in bsp.parameters()) == 0

    def test_forward_not_implemented(self):
        bsp = DnonDn(n=4)
        with pytest.raises(NotImplementedError):
            bsp(torch.randn(2, 8))

    def test_invert_not_implemented(self):
        bsp = DnonDn(n=4)
        with pytest.raises(NotImplementedError):
            bsp.invert(torch.randn(2, 4))

    def test_extra_repr(self):
        bsp = DnonDn(n=4)
        assert 'n=4' in repr(bsp)
