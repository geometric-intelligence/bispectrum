"""Tests targeting coverage gaps in bispectrum package.

These tests exercise edge cases and rarely-hit code paths to bring coverage close to 100%.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from bispectrum._bessel import (
    _bisect_newton,
    _jn_scalar,
    _mcmahon_zeros_j0,
    bessel_jn,
    bessel_jn_zeros,
    compute_all_bessel_roots,
)
from bispectrum._cg import (
    _load_cache,
    _save_cache,
    compute_reduced_cg_parallel,
    compute_sparse_cg_entry,
    compute_sparse_cg_parallel,
)
from bispectrum.dn_on_dn import DnonDn, _compute_cg
from bispectrum.rotation import create_spherical_grid
from bispectrum.torus_on_torus import TorusOnTorus


class TestBesselEdgeCases:
    """Cover edge cases in _bessel.py."""

    def test_jn_scalar_zero_input(self):
        """_jn_scalar(n>=2, 0) returns 0 (line 66)."""
        assert _jn_scalar(3, 0.0) == 0.0
        assert _jn_scalar(5, 0.0) == 0.0

    def test_bisect_newton_exact_at_a(self):
        """When f(a) is already zero, return a immediately (line 89)."""
        root = bessel_jn_zeros(0, 1)[0].item()
        result = _bisect_newton(0, root, root + 1.0)
        assert abs(result - root) < 1e-12

    def test_bisect_newton_exact_at_b(self):
        """When f(b) is already zero, return b immediately (line 91)."""
        root = bessel_jn_zeros(0, 1)[0].item()
        result = _bisect_newton(0, root - 1.0, root)
        assert abs(result - root) < 1e-12

    def test_bisect_newton_no_sign_change(self):
        """Fa * fb > 0 returns midpoint (line 92-93)."""
        result = _bisect_newton(0, 0.1, 0.5)
        assert abs(result - 0.3) < 1e-10

    def test_bisect_newton_convergence(self):
        """Exercise the full Newton-bisection loop (lines 96-124)."""
        root = _bisect_newton(0, 2.0, 3.0)
        expected = bessel_jn_zeros(0, 1)[0].item()
        assert abs(root - expected) < 1e-10

    def test_mcmahon_zeros_empty(self):
        """_mcmahon_zeros_j0(0) returns [] (line 130)."""
        assert _mcmahon_zeros_j0(0) == []

    def test_compute_all_bessel_roots_large_n_max(self):
        """Exercise the num_brackets <= 0 branch (lines 223-225)."""
        roots = compute_all_bessel_roots(50, 1)
        assert 0 in roots
        for order in range(50 + 1):
            assert order in roots

    def test_bessel_jn_zeros_single_order(self):
        """bessel_jn_zeros for a single order."""
        zeros = bessel_jn_zeros(2, 5)
        assert zeros.shape == (5,)
        for z in zeros:
            assert abs(bessel_jn(2, z.unsqueeze(0)).item()) < 1e-10

    def test_bessel_jn_zeros_empty(self):
        """bessel_jn_zeros with 0 zeros returns empty."""
        zeros = bessel_jn_zeros(0, 0)
        assert zeros.shape == (0,)


class TestSparseCG:
    """Cover compute_sparse_cg_entry and compute_sparse_cg_parallel."""

    def test_sparse_cg_entry_basic(self):
        """Exercise compute_sparse_cg_entry (lines 497-573)."""
        m1_idx, m_idx, cg_vals = compute_sparse_cg_entry(1, 1, 0)
        assert len(m1_idx) == len(m_idx) == len(cg_vals)
        assert len(cg_vals) > 0

    def test_sparse_cg_entry_higher_l(self):
        """Larger l values to exercise more loop iterations."""
        m1_idx, m_idx, cg_vals = compute_sparse_cg_entry(3, 4, 5)
        assert len(cg_vals) > 0
        assert all(0 <= m1 <= 2 * 3 for m1 in m1_idx)

    def test_sparse_cg_entry_l_val_zero(self):
        """l_val=0 case — single m=0 column, so m_idx = m + l_val = 0."""
        m1_idx, m_idx, cg_vals = compute_sparse_cg_entry(2, 2, 0)
        assert len(cg_vals) > 0
        # l_val=0 means only m=0 is valid, so all m_idx should be 0 (= m + l_val = 0 + 0)
        assert all(m == 0 for m in m_idx)

    def test_sparse_cg_parallel_empty(self):
        """Empty input returns empty list (line 593-594)."""
        result = compute_sparse_cg_parallel([])
        assert result == []

    def test_sparse_cg_parallel_single(self):
        """Single entry uses sequential path (line 608-609)."""
        entries = [(0, 1, 1, 2, False)]
        result = compute_sparse_cg_parallel(entries, max_workers=1)
        assert len(result) == 1
        m1_idx, m_idx, cg_vals = result[0]
        assert len(cg_vals) > 0

    def test_sparse_cg_parallel_multi(self):
        """Multiple entries use threaded path (lines 611-620)."""
        entries = [
            (0, 1, 1, 0, False),
            (1, 1, 1, 1, False),
            (2, 1, 1, 2, False),
            (3, 2, 1, 1, False),
        ]
        result = compute_sparse_cg_parallel(entries, max_workers=4)
        assert len(result) == 4
        for _m1_idx, _m_idx, cg_vals in result:
            assert len(cg_vals) > 0


class TestReducedCGParallel:
    """Cover compute_reduced_cg_parallel multi-threaded path."""

    def test_empty_groups(self):
        """Empty input returns empty dict."""
        result = compute_reduced_cg_parallel([])
        assert result == {}

    def test_single_group_sequential(self):
        """Single group uses sequential path (line 650-651)."""
        groups = [(0, 1, 1, [0, 1, 2])]
        result = compute_reduced_cg_parallel(groups, max_workers=1)
        assert 0 in result
        assert result[0].shape[0] == (2 * 1 + 1) * (2 * 1 + 1)

    def test_multi_group_parallel(self):
        """Multiple groups use threaded path (lines 652-665)."""
        groups = [
            (0, 1, 1, [0, 1, 2]),
            (1, 2, 1, [1, 2, 3]),
            (2, 2, 2, [0, 1, 2, 3, 4]),
        ]
        result = compute_reduced_cg_parallel(groups, max_workers=3)
        assert len(result) == 3
        for gid in range(3):
            assert gid in result


class TestCGCache:
    """Cover disk cache miss/validation paths in _cg.py."""

    def test_load_cache_missing_file(self):
        """_load_cache returns None when file doesn't exist (line 697-698)."""
        result = _load_cache(9999)
        assert result is None

    def test_load_cache_wrong_lmax(self):
        """_load_cache returns None when stored lmax doesn't match (line 701-702)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir) / 'cg_lmax5.pt'
            data = {'__lmax__': torch.tensor(999), '1_1': torch.eye(3)}
            torch.save(data, tmppath)
            with patch('bispectrum._cg._cache_path', return_value=tmppath):
                result = _load_cache(5)
                assert result is None

    def test_save_and_load_roundtrip(self):
        """_save_cache + _load_cache roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('bispectrum._cg._CACHE_DIR', Path(tmpdir)):
                matrices = {(1, 1): torch.randn(3, 3)}
                _save_cache(5, matrices)
                loaded = _load_cache(5)
                assert loaded is not None
                torch.testing.assert_close(loaded[(1, 1)], matrices[(1, 1)])


class TestDnonDnEdgeCases:
    """Cover edge cases in dn_on_dn.py."""

    def test_n3_zero_early_return(self):
        """N=3 odd gives n3=max(n2d-1,1)=1; n=3 should work.

        Actually need n where n3=0 which doesn't happen per formula. Test n=3 (small odd) for the
        small-n code paths.
        """
        bsp = DnonDn(n=3, selective=True)
        f = torch.randn(2, 6, dtype=torch.float64)
        out = bsp(f)
        assert out.shape == (2, bsp.output_size)

    def test_even_n_large(self):
        """Even n exercises all CG decomposition branches."""
        bsp = DnonDn(n=8, selective=True)
        f = torch.randn(4, 16, dtype=torch.float64)
        out = bsp(f)
        assert out.shape == (4, bsp.output_size)

    def test_odd_n_various(self):
        """Odd n values for different code paths."""
        for n in [5, 7, 9, 11]:
            bsp = DnonDn(n=n, selective=True)
            f = torch.randn(2, 2 * n, dtype=torch.float64)
            out = bsp(f)
            assert out.shape == (2, bsp.output_size)

    def test_compute_cg_all_pairs(self):
        """Exercise _compute_cg for various (i,j,n) to hit all branches."""
        for n in [4, 5, 6, 7, 8]:
            n2d = (n - 1) // 2
            for i in range(1, n2d + 1):
                for j in range(1, n2d + 1):
                    C, blocks = _compute_cg(i, j, n)
                    assert C.shape == (4, 4)
                    CtC = C.T @ C
                    assert torch.allclose(CtC, torch.eye(4, dtype=torch.float64), atol=1e-8)

    def test_build_fplus_method(self):
        """_build_fplus still works for various decomposition patterns."""
        for n in [4, 6, 8]:
            bsp = DnonDn(n=n, selective=True)
            f = torch.randn(2, 2 * n, dtype=torch.float64)
            fhat = bsp._group_dft(f)
            for m in range(bsp._n3):
                fplus = bsp._build_fplus(fhat, m)
                assert fplus.shape == (2, 4, 4)

    def test_inversion_runs_n4(self):
        """Inversion for even n=4 runs without error and returns correct shape."""
        bsp = DnonDn(n=4, selective=True)
        f = torch.randn(3, 8, dtype=torch.float64)
        beta = bsp(f)
        f_rec = bsp.invert(beta)
        assert f_rec.shape == (3, 8)
        assert f_rec.dtype == torch.float64

    def test_inversion_runs_n6(self):
        """Inversion for even n=6 exercises the sequential recovery loop."""
        bsp = DnonDn(n=6, selective=True)
        f = torch.randn(2, 12, dtype=torch.float64)
        beta = bsp(f)
        f_rec = bsp.invert(beta)
        assert f_rec.shape == (2, 12)


class TestSO3onS2EdgeCases:
    """Cover edge cases in so3_on_s2.py."""

    def test_selective_false_forward(self):
        """Full (non-selective) forward exercises _forward_python path."""
        from bispectrum import SO3onS2

        bsp = SO3onS2(lmax=2, nlat=16, nlon=32, selective=False)
        f = torch.randn(1, 16, 32)
        out = bsp(f)
        assert out.shape[0] == 1
        assert out.shape[1] == bsp.output_size

    def test_cg_power_map_property(self):
        """Access cg_power_map property (line 1036)."""
        from bispectrum import SO3onS2

        bsp = SO3onS2(lmax=4, nlat=16, nlon=32, selective=True)
        cg_map = bsp.cg_power_map
        assert isinstance(cg_map, list)
        if len(cg_map) > 0:
            assert len(cg_map[0]) == 3

    def test_index_map_property(self):
        """Access index_map property (line 1031)."""
        from bispectrum import SO3onS2

        bsp = SO3onS2(lmax=4, nlat=16, nlon=32, selective=True)
        idx_map = bsp.index_map
        assert isinstance(idx_map, list)
        assert len(idx_map) == bsp.output_size

    def test_extra_repr(self):
        """extra_repr method."""
        from bispectrum import SO3onS2

        bsp = SO3onS2(lmax=3, nlat=16, nlon=32, selective=True)
        r = bsp.extra_repr()
        assert 'lmax=3' in r
        assert 'selective=True' in r

    def test_sparse_cache_miss_forces_computation(self):
        """Force sparse CG computation by clearing cache (lines 440-462)."""
        from bispectrum import SO3onS2

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('bispectrum.so3_on_s2._CACHE_DIR', Path(tmpdir)):
                bsp = SO3onS2(lmax=3, nlat=16, nlon=32, selective=True)
                f = torch.randn(1, 16, 32)
                out = bsp(f)
                assert out.shape == (1, bsp.output_size)

    def test_full_mode_properties(self):
        """Full mode output_size and index_map."""
        from bispectrum import SO3onS2

        bsp = SO3onS2(lmax=2, nlat=16, nlon=32, selective=False)
        assert bsp.output_size > 0
        assert len(bsp.index_map) == bsp.output_size


class TestOctaOnOctaEdgeCases:
    """Cover edge cases in octa_on_octa.py."""

    def test_full_mode_raises(self):
        """Selective=False raises NotImplementedError (line 525)."""
        from bispectrum import OctaonOcta

        bsp = OctaonOcta(selective=False)
        f = torch.randn(2, 24)
        with pytest.raises(NotImplementedError):
            bsp(f)

    def test_wrong_input_shape(self):
        """Wrong input shape raises ValueError (line 529)."""
        from bispectrum import OctaonOcta

        bsp = OctaonOcta(selective=True)
        with pytest.raises(ValueError):
            bsp(torch.randn(2, 10))

    def test_extra_repr(self):
        """extra_repr is callable."""
        from bispectrum import OctaonOcta

        bsp = OctaonOcta(selective=True)
        r = bsp.extra_repr()
        assert 'selective=True' in r


class TestRotationEdgeCases:
    """Cover edge cases in rotation.py."""

    def test_unsupported_grid_type(self):
        """Unsupported grid raises ValueError (line 107)."""
        with pytest.raises(ValueError, match='Unsupported grid type'):
            create_spherical_grid(16, 32, grid='gaussian')


class TestTorusOnTorusEdgeCases:
    """Cover edge cases in torus_on_torus.py."""

    def test_large_group_full_warning(self):
        """Full bispectrum on large group issues warning (line 84).

        Use ns=(101,) which gives |G|=101 < 10000, so no warning. Instead use a 2D torus with
        product > 10000 but mock _build_full_indices to avoid actually building the giant index
        arrays.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            # ns=(102, 102) gives |G|=10404 > 10000, triggering the warning.
            # Patch _build_full_indices to return minimal data so __init__ completes fast.
            with patch.object(
                TorusOnTorus,
                '_build_full_indices',
                return_value=([0], [0], [0], [((0,), (0,))]),
            ):
                TorusOnTorus(ns=(102, 102), selective=False)
            assert any('Consider selective=True' in str(warning.message) for warning in w)
