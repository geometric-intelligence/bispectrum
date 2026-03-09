"""Tests for SO2onS1 bispectrum module.

SO2onS1 is a thin subclass of CnonCn. These tests verify the subclass relationship holds and that
the continuous-group wrapper behaves identically to the underlying discrete implementation.
"""

import pytest
import torch

from bispectrum import CnonCn, SO2onS1

ATOL = 1e-4
RTOL = 1e-4


class TestSO2onS1IsACnonCn:
    def test_isinstance(self):
        bsp = SO2onS1(n=8)
        assert isinstance(bsp, CnonCn)

    def test_is_nn_module(self):
        bsp = SO2onS1(n=8)
        assert isinstance(bsp, torch.nn.Module)

    def test_no_trainable_parameters(self):
        bsp = SO2onS1(n=8)
        assert sum(p.numel() for p in bsp.parameters()) == 0


class TestSO2onS1MatchesCnonCn:
    @pytest.mark.parametrize('n', [2, 4, 8, 16])
    def test_forward_identical_selective(self, n: int):
        torch.manual_seed(n)
        f = torch.randn(4, n)
        so2 = SO2onS1(n=n, selective=True)
        cn = CnonCn(n=n, selective=True)
        torch.testing.assert_close(so2(f), cn(f))

    @pytest.mark.parametrize('n', [2, 4, 8, 16])
    def test_forward_identical_full(self, n: int):
        torch.manual_seed(n)
        f = torch.randn(4, n)
        so2 = SO2onS1(n=n, selective=False)
        cn = CnonCn(n=n, selective=False)
        torch.testing.assert_close(so2(f), cn(f))

    def test_output_size_matches(self):
        for n in [3, 5, 8, 16]:
            assert SO2onS1(n=n).output_size == CnonCn(n=n).output_size

    def test_index_map_matches(self):
        for n in [3, 5, 8, 16]:
            assert SO2onS1(n=n).index_map == CnonCn(n=n).index_map


class TestSO2onS1Invariance:
    @pytest.mark.parametrize('n', [3, 5, 8, 16, 32])
    def test_rotation_invariance(self, n: int):
        """SO(2) rotation = cyclic shift on discretized S^1."""
        torch.manual_seed(n)
        bsp = SO2onS1(n=n)
        f = torch.randn(4, n)
        beta = bsp(f)
        for shift in range(1, n):
            f_rotated = torch.roll(f, shift, dims=-1)
            torch.testing.assert_close(bsp(f_rotated), beta, atol=ATOL, rtol=RTOL)


class TestSO2onS1Invert:
    @pytest.mark.parametrize('n', [3, 5, 8, 16])
    def test_invert_roundtrip_bispectrum(self, n: int):
        torch.manual_seed(n)
        bsp = SO2onS1(n=n, selective=True)
        f = torch.randn(4, n)
        beta = bsp(f)
        f_rec = bsp.invert(beta)
        beta_rec = bsp(f_rec)
        torch.testing.assert_close(beta, beta_rec, atol=ATOL, rtol=RTOL)

    def test_invert_not_implemented_full(self):
        bsp = SO2onS1(n=8, selective=False)
        with pytest.raises(NotImplementedError):
            bsp.invert(torch.randn(2, 36) + 0j)
