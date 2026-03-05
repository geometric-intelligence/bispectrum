"""Tests for DnonDn bispectrum module."""

import pytest
import torch

from bispectrum import DnonDn

ATOL = 1e-4
RTOL = 1e-4


def _rotate(f: torch.Tensor, n: int, shift: int = 1) -> torch.Tensor:
    """Apply rotation by a^shift: cyclic-shift each half independently."""
    f_rot = torch.roll(f[:, :n], shift, dims=-1)
    f_ref = torch.roll(f[:, n:], shift, dims=-1)
    return torch.cat([f_rot, f_ref], dim=-1)


def _reflect(f: torch.Tensor, n: int) -> torch.Tensor:
    """Apply reflection x:  (T_x f)(a^l) = f(a^{-l} x),  (T_x f)(a^l x) = f(a^{-l}).

    The reflection swaps the two halves and reverses the cyclic order.
    """
    f_rot = f[:, :n]
    f_ref = f[:, n:]
    new_ref = torch.cat([f_rot[:, :1], f_rot[:, 1:].flip(dims=(-1,))], dim=-1)
    new_rot = torch.cat([f_ref[:, :1], f_ref[:, 1:].flip(dims=(-1,))], dim=-1)
    return torch.cat([new_rot, new_ref], dim=-1)


class TestDnonDnConstruction:
    def test_instantiation(self):
        bsp = DnonDn(n=4)
        assert bsp.n == 4
        assert bsp.selective is True

    def test_instantiation_full(self):
        bsp = DnonDn(n=4, selective=False)
        assert bsp.selective is False

    def test_n_must_be_gt_2(self):
        with pytest.raises(ValueError, match='n must be > 2'):
            DnonDn(n=2)
        with pytest.raises(ValueError, match='n must be > 2'):
            DnonDn(n=1)

    @pytest.mark.parametrize(
        'n, expected',
        [
            (3, 1 + 4 + 16 * 1),
            (4, 1 + 4 + 16 * 1),
            (5, 1 + 4 + 16 * 1),
            (7, 1 + 4 + 16 * 2),
            (8, 1 + 4 + 16 * 3),
            (16, 1 + 4 + 16 * 7),
        ],
    )
    def test_selective_output_size(self, n: int, expected: int):
        bsp = DnonDn(n=n)
        assert bsp.output_size == expected

    def test_no_trainable_parameters(self):
        bsp = DnonDn(n=8)
        assert isinstance(bsp, torch.nn.Module)
        assert sum(p.numel() for p in bsp.parameters()) == 0

    def test_extra_repr(self):
        bsp = DnonDn(n=8)
        assert 'n=8' in repr(bsp)
        assert 'selective=True' in repr(bsp)

    def test_index_map_length(self):
        for n in (3, 4, 5, 7, 8):
            bsp = DnonDn(n=n)
            assert len(bsp.index_map) == bsp.output_size


class TestDnonDnForward:
    @pytest.mark.parametrize('n', [3, 4, 5, 7, 8, 16])
    def test_output_shape(self, n: int):
        bsp = DnonDn(n=n)
        f = torch.randn(3, 2 * n)
        out = bsp(f)
        assert out.shape == (3, bsp.output_size)

    def test_output_dtype_float32(self):
        bsp = DnonDn(n=8)
        out = bsp(torch.randn(2, 16))
        assert out.dtype == torch.float32

    def test_output_dtype_float64(self):
        bsp = DnonDn(n=8)
        out = bsp(torch.randn(2, 16, dtype=torch.float64))
        assert out.dtype == torch.float64

    def test_deterministic(self):
        bsp = DnonDn(n=8)
        f = torch.randn(4, 16)
        torch.testing.assert_close(bsp(f), bsp(f))

    @pytest.mark.parametrize('n', [3, 4, 5, 7, 8, 16])
    def test_rotation_invariance(self, n: int):
        torch.manual_seed(n)
        bsp = DnonDn(n=n)
        f = torch.randn(4, 2 * n)
        beta = bsp(f)
        for shift in range(1, n):
            f_shifted = _rotate(f, n, shift)
            torch.testing.assert_close(bsp(f_shifted), beta, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize('n', [3, 4, 5, 7, 8, 16])
    def test_reflection_invariance(self, n: int):
        torch.manual_seed(n + 100)
        bsp = DnonDn(n=n)
        f = torch.randn(4, 2 * n)
        beta = bsp(f)
        f_ref = _reflect(f, n)
        torch.testing.assert_close(bsp(f_ref), beta, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize('n', [3, 4, 5, 7, 8])
    def test_combined_rotation_reflection_invariance(self, n: int):
        torch.manual_seed(n + 200)
        bsp = DnonDn(n=n)
        f = torch.randn(4, 2 * n)
        beta = bsp(f)
        f_transformed = _reflect(_rotate(f, n, 2), n)
        torch.testing.assert_close(bsp(f_transformed), beta, atol=ATOL, rtol=RTOL)

    def test_different_signals_differ(self):
        bsp = DnonDn(n=4)
        f1 = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0]])
        f2 = torch.tensor([[1.0, 1, 0, 0, 0, 0, 0, 0]])  # not in same orbit
        assert not torch.allclose(bsp(f1), bsp(f2))

    def test_forward_not_implemented_full(self):
        bsp = DnonDn(n=4, selective=False)
        with pytest.raises(NotImplementedError):
            bsp(torch.randn(2, 8))

    def test_batch_size_one(self):
        bsp = DnonDn(n=8)
        f = torch.randn(1, 16)
        out = bsp(f)
        assert out.shape == (1, bsp.output_size)

    def test_dft_roundtrip(self):
        """_group_dft followed by _inverse_dft recovers the signal."""
        n = 8
        bsp = DnonDn(n=n)
        f = torch.randn(4, 2 * n, dtype=torch.float64)
        fhat = bsp._group_dft(f)
        f_rec = bsp._inverse_dft(fhat)
        torch.testing.assert_close(f_rec, f, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize('n', [3, 4, 5, 7, 8, 16])
    def test_dft_roundtrip_various_n(self, n: int):
        bsp = DnonDn(n=n)
        f = torch.randn(4, 2 * n, dtype=torch.float64)
        fhat = bsp._group_dft(f)
        f_rec = bsp._inverse_dft(fhat)
        torch.testing.assert_close(f_rec, f, atol=1e-10, rtol=1e-10)


class TestDnonDnInvert:
    @pytest.mark.parametrize('n', [3, 4, 5, 7, 8])
    def test_roundtrip_bispectrum(self, n: int):
        """β_{00} and β_{01} must roundtrip exactly; β_{1k} up to O(2)."""
        torch.manual_seed(n + 42)
        bsp = DnonDn(n=n)
        f = torch.randn(4, 2 * n, dtype=torch.float64)
        beta = bsp(f)
        f_rec = bsp.invert(beta)
        beta_rec = bsp(f_rec)
        # β_{ρ0,ρ0} and β_{ρ0,ρ1} are determined by F_0 and F_1^T F_1
        # which are recovered exactly (the O(2) ambiguity cancels).
        torch.testing.assert_close(beta[:, :5], beta_rec[:, :5], atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize('n', [3, 4, 5, 7, 8])
    def test_roundtrip_fourier_frobenius(self, n: int):
        """Frobenius norms of Fourier coefs must match (O(2) indeterminacy)."""
        torch.manual_seed(n + 99)
        bsp = DnonDn(n=n)
        f = torch.randn(4, 2 * n, dtype=torch.float64)
        beta = bsp(f)
        f_rec = bsp.invert(beta)

        fhat_orig = bsp._group_dft(f)
        fhat_rec = bsp._group_dft(f_rec)

        n2d = bsp._n2d
        for k_idx in range(1, n2d + 1):
            frob_orig = torch.linalg.norm(fhat_orig[:, :, :, k_idx], dim=(-2, -1))
            frob_rec = torch.linalg.norm(fhat_rec[:, :, :, k_idx], dim=(-2, -1))
            torch.testing.assert_close(frob_orig, frob_rec, atol=ATOL, rtol=RTOL)

    def test_invert_output_shape(self):
        n = 8
        bsp = DnonDn(n=n)
        beta = bsp(torch.randn(3, 2 * n))
        f_rec = bsp.invert(beta)
        assert f_rec.shape == (3, 2 * n)

    def test_invert_output_is_real(self):
        bsp = DnonDn(n=8)
        beta = bsp(torch.randn(2, 16))
        f_rec = bsp.invert(beta)
        assert not f_rec.is_complex()

    def test_invert_not_implemented_full(self):
        bsp = DnonDn(n=8, selective=False)
        with pytest.raises(NotImplementedError):
            bsp.invert(torch.randn(2, 4))


class TestDnonDnPerformance:
    """Benchmark tests for DFT performance.

    Run with ``pytest -s`` to see timings.
    """

    @pytest.mark.parametrize('n', [64, 256, 1024])
    def test_group_dft_timing(self, n: int) -> None:
        import time

        bsp = DnonDn(n=n)
        f = torch.randn(16, 2 * n, dtype=torch.float64)
        # warm-up
        for _ in range(3):
            bsp._group_dft(f)

        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            bsp._group_dft(f)
        elapsed = (time.perf_counter() - t0) / iters
        print(f'\n  _group_dft  n={n:>5d}  {elapsed * 1e3:.3f} ms')

    @pytest.mark.parametrize('n', [64, 256, 1024])
    def test_inverse_dft_timing(self, n: int) -> None:
        import time

        bsp = DnonDn(n=n)
        f = torch.randn(16, 2 * n, dtype=torch.float64)
        fhat = bsp._group_dft(f)
        # warm-up
        for _ in range(3):
            bsp._inverse_dft(fhat)

        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            bsp._inverse_dft(fhat)
        elapsed = (time.perf_counter() - t0) / iters
        print(f'\n  _inverse_dft  n={n:>5d}  {elapsed * 1e3:.3f} ms')
