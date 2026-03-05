"""Tests for CnonCn bispectrum module."""

import pytest
import torch

from bispectrum import CnonCn

ATOL = 1e-4
RTOL = 1e-4


class TestCnonCnConstruction:
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
        assert bsp.output_size == n * (n + 1) // 2

    def test_selective_index_map(self):
        bsp = CnonCn(n=8, selective=True)
        idx = bsp.index_map
        assert idx[0] == (0, 0)
        assert idx[1] == (0, 1)
        assert idx[2] == (1, 1)
        assert idx[-1] == (1, 6)

    def test_full_index_map_upper_triangular(self):
        bsp = CnonCn(n=4, selective=False)
        idx = bsp.index_map
        assert idx == [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 2),
            (2, 3),
            (3, 3),
        ]

    def test_no_trainable_parameters(self):
        bsp = CnonCn(n=8)
        assert isinstance(bsp, torch.nn.Module)
        assert sum(p.numel() for p in bsp.parameters()) == 0

    def test_extra_repr(self):
        bsp = CnonCn(n=8)
        assert 'n=8' in repr(bsp)
        assert 'selective=True' in repr(bsp)


class TestCnonCnForward:
    @pytest.mark.parametrize('n', [4, 5, 7, 8, 16, 32])
    def test_output_shape_selective(self, n: int):
        bsp = CnonCn(n=n, selective=True)
        f = torch.randn(3, n)
        out = bsp(f)
        assert out.shape == (3, n)

    @pytest.mark.parametrize('n', [4, 5, 7, 8, 16, 32])
    def test_output_shape_full(self, n: int):
        bsp = CnonCn(n=n, selective=False)
        f = torch.randn(3, n)
        out = bsp(f)
        assert out.shape == (3, n * (n + 1) // 2)

    def test_output_dtype_float32(self):
        bsp = CnonCn(n=8)
        f = torch.randn(2, 8)
        out = bsp(f)
        assert out.is_complex()

    def test_output_dtype_float64(self):
        bsp = CnonCn(n=8)
        f = torch.randn(2, 8, dtype=torch.float64)
        out = bsp(f)
        assert out.dtype == torch.complex128

    def test_deterministic(self):
        bsp = CnonCn(n=8)
        f = torch.randn(4, 8)
        torch.testing.assert_close(bsp(f), bsp(f))

    @pytest.mark.parametrize('n', [4, 5, 7, 8, 16, 32])
    def test_cyclic_shift_invariance_selective(self, n: int):
        torch.manual_seed(n)
        bsp = CnonCn(n=n, selective=True)
        f = torch.randn(4, n)
        beta = bsp(f)
        for shift in range(1, n):
            f_shifted = torch.roll(f, shift, dims=-1)
            torch.testing.assert_close(bsp(f_shifted), beta, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize('n', [4, 5, 7, 8, 16, 32])
    def test_cyclic_shift_invariance_full(self, n: int):
        torch.manual_seed(n)
        bsp = CnonCn(n=n, selective=False)
        f = torch.randn(4, n)
        beta = bsp(f)
        for shift in range(1, n):
            f_shifted = torch.roll(f, shift, dims=-1)
            torch.testing.assert_close(bsp(f_shifted), beta, atol=ATOL, rtol=RTOL)

    def test_different_signals_differ(self):
        bsp = CnonCn(n=8)
        f1 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        f2 = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        beta1 = bsp(f1)
        beta2 = bsp(f2)
        assert not torch.allclose(beta1, beta2)

    def test_selective_matches_formula(self):
        """Verify selective coefficients match the explicit formula."""
        n = 8
        bsp = CnonCn(n=n, selective=True)
        f = torch.randn(2, n)
        fhat = torch.fft.fft(f, dim=-1)
        beta = bsp(f)

        # beta_{0,0} = fhat[0]^2 * conj(fhat[0])
        expected_00 = fhat[:, 0] * fhat[:, 0] * torch.conj(fhat[:, 0])
        torch.testing.assert_close(beta[:, 0], expected_00, atol=1e-6, rtol=1e-6)

        # beta_{0,1} = fhat[0] * fhat[1] * conj(fhat[1])
        expected_01 = fhat[:, 0] * fhat[:, 1] * torch.conj(fhat[:, 1])
        torch.testing.assert_close(beta[:, 1], expected_01, atol=1e-6, rtol=1e-6)

        # beta_{1,k} = fhat[1] * fhat[k] * conj(fhat[k+1]) for k=1..n-2
        for k in range(1, n - 1):
            expected = fhat[:, 1] * fhat[:, k] * torch.conj(fhat[:, k + 1])
            torch.testing.assert_close(beta[:, k + 1], expected, atol=1e-6, rtol=1e-6)

    def test_batch_size_one(self):
        bsp = CnonCn(n=8)
        f = torch.randn(1, 8)
        out = bsp(f)
        assert out.shape == (1, 8)


class TestCnonCnInvert:
    @pytest.mark.parametrize('n', [4, 5, 7, 8, 16, 32])
    def test_invert_roundtrip_dft_magnitudes(self, n: int):
        """DFT magnitudes of the reconstructed signal must match the original."""
        torch.manual_seed(n)
        bsp = CnonCn(n=n, selective=True)
        f = torch.randn(4, n)
        beta = bsp(f)
        f_rec = bsp.invert(beta)

        fhat_orig = torch.fft.fft(f, dim=-1)
        fhat_rec = torch.fft.fft(f_rec, dim=-1)
        torch.testing.assert_close(fhat_orig.abs(), fhat_rec.abs(), atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize('n', [4, 5, 7, 8, 16, 32])
    def test_invert_roundtrip_bispectrum(self, n: int):
        """Bispectrum of the reconstructed signal must match the original."""
        torch.manual_seed(n)
        bsp = CnonCn(n=n, selective=True)
        f = torch.randn(4, n)
        beta = bsp(f)
        f_rec = bsp.invert(beta)
        beta_rec = bsp(f_rec)
        torch.testing.assert_close(beta, beta_rec, atol=ATOL, rtol=RTOL)

    def test_invert_not_implemented_full(self):
        bsp = CnonCn(n=8, selective=False)
        with pytest.raises(NotImplementedError):
            bsp.invert(torch.randn(2, 36) + 0j)

    def test_invert_output_shape(self):
        bsp = CnonCn(n=8, selective=True)
        beta = bsp(torch.randn(3, 8))
        f_rec = bsp.invert(beta)
        assert f_rec.shape == (3, 8)

    def test_invert_output_is_complex(self):
        """Result is complex due to continuous SO(2) phase indeterminacy."""
        bsp = CnonCn(n=8, selective=True)
        beta = bsp(torch.randn(2, 8))
        f_rec = bsp.invert(beta)
        assert f_rec.is_complex()
