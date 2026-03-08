"""Tests for TorusOnTorus bispectrum module."""

import math

import pytest
import torch

from bispectrum import CnonCn
from bispectrum.torus_on_torus import TorusOnTorus

ATOL = 1e-4
RTOL = 1e-4

NS_2D = [(2, 2), (3, 3), (4, 4), (3, 5), (5, 3), (4, 6)]
NS_3D = [(2, 2, 2), (2, 3, 4), (3, 3, 3)]
NS_ALL = [(2, 2), (3, 5), (4, 4), (2, 2, 2), (2, 3, 4)]


class TestTorusOnTorusConstruction:
    def test_empty_ns_raises(self):
        with pytest.raises(ValueError):
            TorusOnTorus(ns=())

    def test_zero_dim_raises(self):
        with pytest.raises(ValueError):
            TorusOnTorus(ns=(0, 4))
        with pytest.raises(ValueError):
            TorusOnTorus(ns=(4, -1))

    def test_selective_requires_dims_ge_2(self):
        with pytest.raises(ValueError):
            TorusOnTorus(ns=(1, 4), selective=True)
        with pytest.raises(ValueError):
            TorusOnTorus(ns=(4, 1), selective=True)

    def test_dim1_full_works(self):
        bsp = TorusOnTorus(ns=(1, 4), selective=False)
        assert bsp.group_order == 4

    def test_instantiation_2d(self):
        bsp = TorusOnTorus(ns=(4, 6))
        assert bsp.ns == (4, 6)
        assert bsp.selective is True
        assert bsp.ndim == 2
        assert bsp.group_order == 24

    def test_instantiation_3d(self):
        bsp = TorusOnTorus(ns=(2, 3, 4))
        assert bsp.ndim == 3
        assert bsp.group_order == 24

    @pytest.mark.parametrize('ns', NS_ALL)
    def test_selective_output_size(self, ns: tuple[int, ...]):
        bsp = TorusOnTorus(ns=ns, selective=True)
        assert bsp.output_size == math.prod(ns)

    @pytest.mark.parametrize('ns', [(2, 2), (3, 3), (2, 3)])
    def test_full_output_size(self, ns: tuple[int, ...]):
        G = math.prod(ns)
        bsp = TorusOnTorus(ns=ns, selective=False)
        assert bsp.output_size == G * (G + 1) // 2

    def test_index_map_format(self):
        bsp = TorusOnTorus(ns=(3, 4))
        idx = bsp.index_map
        assert len(idx) == bsp.output_size
        for k1, k2 in idx:
            assert isinstance(k1, tuple)
            assert isinstance(k2, tuple)
            assert len(k1) == 2
            assert len(k2) == 2

    def test_no_trainable_parameters(self):
        bsp = TorusOnTorus(ns=(4, 4))
        assert isinstance(bsp, torch.nn.Module)
        assert sum(p.numel() for p in bsp.parameters()) == 0

    def test_extra_repr(self):
        bsp = TorusOnTorus(ns=(4, 6))
        r = repr(bsp)
        assert 'ns=(4, 6)' in r
        assert 'selective=True' in r


class TestTorusOnTorusForward:
    @pytest.mark.parametrize('ns', NS_ALL)
    def test_output_shape_selective(self, ns: tuple[int, ...]):
        bsp = TorusOnTorus(ns=ns, selective=True)
        f = torch.randn(3, *ns)
        out = bsp(f)
        assert out.shape == (3, bsp.output_size)

    @pytest.mark.parametrize('ns', [(2, 2), (3, 3), (2, 3)])
    def test_output_shape_full(self, ns: tuple[int, ...]):
        bsp = TorusOnTorus(ns=ns, selective=False)
        f = torch.randn(3, *ns)
        out = bsp(f)
        assert out.shape == (3, bsp.output_size)

    def test_output_dtype_float32(self):
        bsp = TorusOnTorus(ns=(4, 4))
        f = torch.randn(2, 4, 4)
        out = bsp(f)
        assert out.is_complex()

    def test_output_dtype_float64(self):
        bsp = TorusOnTorus(ns=(4, 4))
        f = torch.randn(2, 4, 4, dtype=torch.float64)
        out = bsp(f)
        assert out.dtype == torch.complex128

    def test_deterministic(self):
        bsp = TorusOnTorus(ns=(4, 4))
        f = torch.randn(4, 4, 4)
        torch.testing.assert_close(bsp(f), bsp(f))

    @pytest.mark.parametrize('ns', NS_2D)
    def test_translation_invariance_2d_selective(self, ns: tuple[int, ...]):
        torch.manual_seed(42)
        bsp = TorusOnTorus(ns=ns, selective=True)
        f = torch.randn(4, *ns)
        beta = bsp(f)
        for s1 in range(ns[0]):
            for s2 in range(ns[1]):
                f_shifted = torch.roll(f, shifts=(s1, s2), dims=(1, 2))
                torch.testing.assert_close(bsp(f_shifted), beta, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize('ns', [(2, 2), (3, 3), (2, 3)])
    def test_translation_invariance_2d_full(self, ns: tuple[int, ...]):
        torch.manual_seed(42)
        bsp = TorusOnTorus(ns=ns, selective=False)
        f = torch.randn(2, *ns)
        beta = bsp(f)
        for s1 in range(ns[0]):
            for s2 in range(ns[1]):
                f_shifted = torch.roll(f, shifts=(s1, s2), dims=(1, 2))
                torch.testing.assert_close(bsp(f_shifted), beta, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize('ns', NS_3D)
    def test_translation_invariance_3d_selective(self, ns: tuple[int, ...]):
        torch.manual_seed(42)
        bsp = TorusOnTorus(ns=ns, selective=True)
        f = torch.randn(2, *ns)
        beta = bsp(f)
        for _ in range(8):
            shifts = tuple(torch.randint(0, n, (1,)).item() for n in ns)
            dims = tuple(range(1, 1 + len(ns)))
            f_shifted = torch.roll(f, shifts=shifts, dims=dims)
            torch.testing.assert_close(bsp(f_shifted), beta, atol=ATOL, rtol=RTOL)

    def test_different_signals_differ(self):
        bsp = TorusOnTorus(ns=(4, 4))
        f1 = torch.zeros(1, 4, 4)
        f1[0, 0, 0] = 1.0
        f2 = torch.zeros(1, 4, 4)
        f2[0, 0, 0] = 1.0
        f2[0, 0, 1] = 2.0
        assert not torch.allclose(bsp(f1), bsp(f2))

    def test_selective_matches_formula(self):
        """Verify selective coefficients match the explicit bispectrum formula."""
        ns = (3, 4)
        bsp = TorusOnTorus(ns=ns, selective=True)
        f = torch.randn(2, *ns)
        fhat = torch.fft.fftn(f, dim=(1, 2))
        beta = bsp(f)

        for i in range(min(8, bsp.output_size)):
            k1, k2 = bsp.index_map[i]
            k_sum = tuple((a + b) % n for a, b, n in zip(k1, k2, ns, strict=False))
            expected = (
                fhat[:, k1[0], k1[1]]
                * fhat[:, k2[0], k2[1]]
                * torch.conj(fhat[:, k_sum[0], k_sum[1]])
            )
            torch.testing.assert_close(beta[:, i], expected, atol=1e-6, rtol=1e-6)

    def test_batch_size_one(self):
        bsp = TorusOnTorus(ns=(4, 4))
        f = torch.randn(1, 4, 4)
        out = bsp(f)
        assert out.shape == (1, bsp.output_size)

    def test_forward_rejects_wrong_ndim(self):
        bsp = TorusOnTorus(ns=(4, 4))
        with pytest.raises(ValueError):
            bsp(torch.randn(4, 4))

    def test_forward_rejects_wrong_shape(self):
        bsp = TorusOnTorus(ns=(4, 4))
        with pytest.raises(ValueError):
            bsp(torch.randn(2, 3, 4))


class TestTorusOnTorusInvert:
    @pytest.mark.parametrize('ns', NS_ALL)
    def test_invert_roundtrip_dft_magnitudes(self, ns: tuple[int, ...]):
        """DFT magnitudes of the reconstructed signal must match the original."""
        torch.manual_seed(42)
        bsp = TorusOnTorus(ns=ns, selective=True)
        f = torch.randn(4, *ns)
        beta = bsp(f)
        f_rec = bsp.invert(beta)

        spatial_dims = tuple(range(1, 1 + len(ns)))
        fhat_orig = torch.fft.fftn(f, dim=spatial_dims)
        fhat_rec = torch.fft.fftn(f_rec, dim=spatial_dims)
        torch.testing.assert_close(fhat_orig.abs(), fhat_rec.abs(), atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize('ns', NS_ALL)
    def test_invert_roundtrip_bispectrum(self, ns: tuple[int, ...]):
        """Bispectrum of the reconstructed signal must match the original."""
        torch.manual_seed(42)
        bsp = TorusOnTorus(ns=ns, selective=True)
        f = torch.randn(4, *ns)
        beta = bsp(f)
        f_rec = bsp.invert(beta)
        beta_rec = bsp(f_rec)
        torch.testing.assert_close(beta, beta_rec, atol=ATOL, rtol=RTOL)

    def test_invert_not_implemented_full(self):
        bsp = TorusOnTorus(ns=(3, 3), selective=False)
        with pytest.raises(NotImplementedError):
            bsp.invert(torch.randn(2, 45) + 0j)

    def test_invert_output_shape_2d(self):
        bsp = TorusOnTorus(ns=(3, 4), selective=True)
        beta = bsp(torch.randn(3, 3, 4))
        f_rec = bsp.invert(beta)
        assert f_rec.shape == (3, 3, 4)

    def test_invert_output_shape_3d(self):
        bsp = TorusOnTorus(ns=(2, 3, 4), selective=True)
        beta = bsp(torch.randn(2, 2, 3, 4))
        f_rec = bsp.invert(beta)
        assert f_rec.shape == (2, 2, 3, 4)

    def test_invert_rejects_wrong_shape(self):
        bsp = TorusOnTorus(ns=(3, 4), selective=True)
        with pytest.raises(ValueError):
            bsp.invert(torch.randn(2, 11) + 0j)

    def test_invert_rejects_wrong_ndim(self):
        bsp = TorusOnTorus(ns=(3, 4), selective=True)
        with pytest.raises(ValueError):
            bsp.invert(torch.randn(12) + 0j)

    def test_invert_raises_on_zero_dc(self):
        bsp = TorusOnTorus(ns=(2, 2), selective=True)
        f = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]])
        beta = bsp(f)
        with pytest.raises(ValueError, match=r'fhat\[0\]'):
            bsp.invert(beta)

    def test_invert_raises_on_zero_generator(self):
        bsp = TorusOnTorus(ns=(2, 2), selective=True)
        f = torch.tensor([[[1.0, 2.0], [1.0, 2.0]]])
        beta = bsp(f)
        with pytest.raises(ValueError, match=r'fhat\[e_'):
            bsp.invert(beta)

    def test_invert_output_is_complex(self):
        bsp = TorusOnTorus(ns=(3, 4), selective=True)
        beta = bsp(torch.randn(2, 3, 4))
        f_rec = bsp.invert(beta)
        assert f_rec.is_complex()


class TestTorusOnTorusCnonCnConsistency:
    """Verify TorusOnTorus(ns=(n,)) is numerically equivalent to CnonCn(n=n)."""

    @pytest.mark.parametrize('n', [2, 3, 4, 5, 7, 8, 16])
    def test_forward_matches_cn(self, n: int):
        torch.manual_seed(n)
        bsp_cn = CnonCn(n=n, selective=True)
        bsp_torus = TorusOnTorus(ns=(n,), selective=True)
        f = torch.randn(4, n)
        beta_cn = bsp_cn(f)
        beta_torus = bsp_torus(f)
        torch.testing.assert_close(beta_cn, beta_torus, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize('n', [2, 3, 4, 5, 7, 8, 16])
    def test_invert_matches_cn(self, n: int):
        torch.manual_seed(n)
        bsp_cn = CnonCn(n=n, selective=True)
        bsp_torus = TorusOnTorus(ns=(n,), selective=True)
        f = torch.randn(4, n)
        beta = bsp_cn(f)
        f_rec_cn = bsp_cn.invert(beta)
        f_rec_torus = bsp_torus.invert(beta)
        torch.testing.assert_close(f_rec_cn, f_rec_torus, atol=1e-6, rtol=1e-6)
