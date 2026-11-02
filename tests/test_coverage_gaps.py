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
    _bessel_jn_multi,
    _bisect_newton_batch,
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
    load_cg_matrices,
)
from bispectrum.dn_on_dn import DnonDn, _compute_cg
from bispectrum.rotation import create_spherical_grid
from bispectrum.torus_on_torus import TorusOnTorus


class TestPackageVersion:
    def test_version_is_string(self):
        import bispectrum

        assert isinstance(bispectrum.__version__, str)
        assert bispectrum.__version__

    def test_version_fallback_when_not_installed(self):
        """Importing from a source tree without metadata yields the sentinel version."""
        import importlib
        import importlib.metadata

        import bispectrum

        def _missing(name: str) -> str:
            raise importlib.metadata.PackageNotFoundError(name)

        with patch('importlib.metadata.version', _missing):
            importlib.reload(bispectrum)
            assert bispectrum.__version__ == '0.0.0+unknown'
        importlib.reload(bispectrum)
        assert bispectrum.__version__ != '0.0.0+unknown'


class TestBesselEdgeCases:
    """Cover edge cases in _bessel.py."""

    def test_jn_multi_no_orders(self):
        """An empty order tuple returns an empty list without touching x."""
        assert _bessel_jn_multi((), torch.tensor([1.0])) == []

    def test_bisect_newton_batch_exact_endpoints(self):
        """Brackets whose endpoint is already a root return that endpoint."""
        root = bessel_jn_zeros(0, 1)[0].item()
        a = torch.tensor([root, root - 1.0], dtype=torch.float64)
        b = torch.tensor([root + 1.0, root], dtype=torch.float64)
        result = _bisect_newton_batch(0, a, b)
        torch.testing.assert_close(
            result, torch.tensor([root, root], dtype=torch.float64), atol=1e-12, rtol=0.0
        )

    def test_bisect_newton_batch_no_sign_change(self):
        """Fa * fb > 0 returns the midpoint."""
        a = torch.tensor([0.1], dtype=torch.float64)
        b = torch.tensor([0.5], dtype=torch.float64)
        assert abs(_bisect_newton_batch(0, a, b).item() - 0.3) < 1e-10

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

    def test_save_cache_oserror_swallowed(self):
        """A failing cache write must not break CG computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch('bispectrum._cg._CACHE_DIR', Path(tmpdir)),
                patch('torch.save', side_effect=OSError('disk full')),
            ):
                _save_cache(5, {(1, 1): torch.eye(3)})
            assert not list(Path(tmpdir).iterdir())


class TestCGSelectionRules:
    """Triples outside the triangle inequality produce no CG terms."""

    def test_sparse_entry_triangle_violation_is_empty(self):
        m1_idx, m_idx, cg_vals = compute_sparse_cg_entry(1, 1, 3)
        assert len(m1_idx) == len(m_idx) == len(cg_vals) == 0

    def test_sparse_entry_valid_triple_is_nonempty(self):
        _, _, cg_vals = compute_sparse_cg_entry(1, 1, 2)
        assert len(cg_vals) > 0


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


class TestSO3CudaGraphPaths:
    """Cover CUDA graph code paths without requiring a GPU."""

    @staticmethod
    def _fake_cache(f: torch.Tensor, output: torch.Tensor) -> dict:
        from unittest.mock import Mock

        from bispectrum.so3_on_s2 import SO3onS2

        return {
            'graph': Mock(),
            'static_input': torch.empty_like(f),
            'static_output': output,
            'key': SO3onS2._cuda_graph_key(f),
        }

    def test_cuda_graph_cache_hit_replays(self):
        """Cached graph with matching (shape, dtype, device) replays and returns clone."""
        from bispectrum import SO3onS2

        bsp = SO3onS2(lmax=3, nlat=16, nlon=32, selective=True)
        f = torch.randn(2, 16, 32)
        fake_output = torch.randn(2, bsp.output_size)
        bsp._cuda_graph_cache = self._fake_cache(f, fake_output)
        result = bsp._forward_cuda_graph(f, 2, bsp.output_size)
        bsp._cuda_graph_cache['graph'].replay.assert_called_once()
        assert result.shape == fake_output.shape
        assert result is not fake_output

    def test_cuda_graph_batch_size_mismatch_returns_none(self):
        """Cache exists but batch_size differs -> returns None."""
        from bispectrum import SO3onS2

        bsp = SO3onS2(lmax=3, nlat=16, nlon=32, selective=True)
        f8 = torch.randn(8, 16, 32)
        bsp._cuda_graph_cache = self._fake_cache(f8, torch.randn(8, bsp.output_size))
        result = bsp._forward_cuda_graph(torch.randn(2, 16, 32), 2, bsp.output_size)
        assert result is None
        bsp._cuda_graph_cache['graph'].replay.assert_not_called()

    def test_cuda_graph_dtype_mismatch_returns_none(self):
        """A graph captured for float32 must not be replayed on a float64 input of the same shape.

        ``static_input.copy_(f)`` would silently downcast and the replay would return a float32
        result computed from truncated data; the cache was previously keyed on batch size only.
        """
        from bispectrum import SO3onS2

        bsp = SO3onS2(lmax=3, nlat=16, nlon=32, selective=True)
        f32 = torch.randn(2, 16, 32, dtype=torch.float32)
        bsp._cuda_graph_cache = self._fake_cache(f32, torch.randn(2, bsp.output_size))
        f64 = torch.randn(2, 16, 32, dtype=torch.float64)
        result = bsp._forward_cuda_graph(f64, 2, bsp.output_size)
        assert result is None
        bsp._cuda_graph_cache['graph'].replay.assert_not_called()

    def test_cuda_graph_grid_shape_mismatch_returns_none(self):
        """Same batch size but a different (nlat, nlon) must not hit the cache."""
        from bispectrum import SO3onS2

        bsp = SO3onS2(lmax=3, nlat=16, nlon=32, selective=True)
        f = torch.randn(2, 16, 32)
        bsp._cuda_graph_cache = self._fake_cache(f, torch.randn(2, bsp.output_size))
        result = bsp._forward_cuda_graph(torch.randn(2, 32, 64), 2, bsp.output_size)
        assert result is None

    def test_cuda_graph_capture_failure_disables(self):
        """Exception during graph capture sets sentinel and returns None."""
        from bispectrum import SO3onS2

        bsp = SO3onS2(lmax=3, nlat=16, nlon=32, selective=True)
        with patch('torch.cuda.Stream', side_effect=RuntimeError('no CUDA')):
            result = bsp._forward_cuda_graph(torch.randn(2, 16, 32), 2, bsp.output_size)
        assert result is None
        assert bsp._cuda_graph_cache == {'key': None}
        # Sentinel stays disabled on the next call too.
        assert bsp._forward_cuda_graph(torch.randn(2, 16, 32), 2, bsp.output_size) is None

    def test_reset_cuda_graph_cache(self):
        """reset_cuda_graph_cache removes the cache attribute."""
        from bispectrum import SO3onS2

        bsp = SO3onS2(lmax=3, nlat=16, nlon=32, selective=True)
        bsp._cuda_graph_cache = {'key': None}
        bsp.reset_cuda_graph_cache()
        assert not hasattr(bsp, '_cuda_graph_cache')

    def test_reset_cuda_graph_cache_noop_when_absent(self):
        """reset_cuda_graph_cache is safe when no cache exists."""
        from bispectrum import SO3onS2

        bsp = SO3onS2(lmax=3, nlat=16, nlon=32, selective=True)
        bsp.reset_cuda_graph_cache()

    @pytest.mark.parametrize('selective', [True, False])
    def test_cuda_graph_capture_wiring_on_cpu(self, selective: bool):
        """Run the capture path with torch.cuda stream/graph APIs stubbed out.

        The forward captured inside the (mocked) graph context runs eagerly on CPU, so this checks
        that the static input/output buffers are wired to the right forward and that the first call
        returns the captured output.
        """
        from unittest.mock import MagicMock

        from bispectrum import SO3onS2

        bsp = SO3onS2(lmax=3, nlat=16, nlon=32, selective=selective)
        f = torch.randn(2, 16, 32)
        expected = bsp(f)

        graph = MagicMock()
        with (
            patch('torch.cuda.Stream', return_value=MagicMock()),
            patch('torch.cuda.current_stream', return_value=MagicMock()),
            patch('torch.cuda.stream', return_value=MagicMock()),
            patch('torch.cuda.CUDAGraph', return_value=graph),
            patch('torch.cuda.graph', return_value=MagicMock()),
        ):
            out = bsp._forward_cuda_graph(f, 2, bsp.output_size)

        assert out is not None
        torch.testing.assert_close(out, expected)
        cache = bsp._cuda_graph_cache
        assert cache['key'] == SO3onS2._cuda_graph_key(f)
        assert cache['static_input'].shape == f.shape
        torch.testing.assert_close(cache['static_output'], expected)
        graph.replay.assert_called_once()

        # A second call with the same signature replays instead of re-capturing.
        out2 = bsp._forward_cuda_graph(f.clone(), 2, bsp.output_size)
        assert graph.replay.call_count == 2
        torch.testing.assert_close(out2, expected)


def _dense_forward(bsp, f: torch.Tensor, num_entries: int) -> torch.Tensor:
    """Run the dense (reduced-CG) forward of an SO3onS2 module on ``f``."""
    from bispectrum.so3_on_s2 import _get_full_sh_coefficients

    coeffs = bsp._sht(f)
    f_coeffs = _get_full_sh_coefficients(coeffs)
    return bsp._forward_python(f_coeffs, coeffs.dtype, f.device, f.shape[0], num_entries)


class TestSO3TablePaths:
    """Exercise the dense-table and lazy-index code paths of SO3onS2 on CPU.

    Dense tables are built on *selective* modules (which only carry sparse
    tables) so each build is the first registration of the ``_cg_red_*``
    buffers; results are checked against the sparse forward.
    """

    def test_build_group_tables_defaults_to_index_map(self):
        """all_triples=None means 'the bispectral entries only' (no CG power)."""
        from bispectrum import SO3onS2

        bsp = SO3onS2(lmax=3, nlat=16, nlon=32, selective=True)
        f = torch.randn(2, 16, 32, dtype=torch.float64)
        bsp._build_group_tables(load_cg_matrices(3))
        dense = _dense_forward(bsp, f, bsp.n_bispec)
        torch.testing.assert_close(dense, bsp(f)[:, : bsp.n_bispec], atol=1e-10, rtol=1e-10)

    def test_dense_tables_match_sparse_for_selective_map(self):
        """The dense forward agrees with the sparse one, CG power entries included."""
        from bispectrum import SO3onS2

        bsp = SO3onS2(lmax=4, nlat=16, nlon=32, selective=True)
        f = torch.randn(2, 16, 32, dtype=torch.float64)
        bsp._build_group_tables(load_cg_matrices(4), bsp.index_map)
        dense = _dense_forward(bsp, f, bsp.output_size)
        torch.testing.assert_close(dense, bsp(f), atol=1e-10, rtol=1e-10)

    def test_reduced_cg_disk_cache_roundtrip(self):
        """cg_data=None computes reduced CG blocks, caches them, and reloads on the next build."""
        from bispectrum import SO3onS2

        def fresh():
            return SO3onS2(lmax=2, nlat=16, nlon=32, selective=True)

        f = torch.randn(1, 16, 32, dtype=torch.float64)
        expected = fresh()(f)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('bispectrum.so3_on_s2._CACHE_DIR', Path(tmpdir)):
                bsp = fresh()
                bsp._build_group_tables(None, bsp.index_map)
                files = list(Path(tmpdir).glob('cg_reduced_*.pt'))
                assert len(files) == 1
                torch.testing.assert_close(_dense_forward(bsp, f, bsp.output_size), expected)

                # Second build hits the cache file instead of recomputing.
                bsp = fresh()
                with patch('bispectrum.so3_on_s2.compute_reduced_cg_parallel') as never:
                    bsp._build_group_tables(None, bsp.index_map)
                never.assert_not_called()
                torch.testing.assert_close(_dense_forward(bsp, f, bsp.output_size), expected)

                # A corrupt cache file is ignored and recomputed.
                files[0].write_text('not a torch file')
                bsp = fresh()
                bsp._build_group_tables(None, bsp.index_map)
                torch.testing.assert_close(_dense_forward(bsp, f, bsp.output_size), expected)

    def test_reduced_cg_cache_write_failure_is_nonfatal(self):
        from bispectrum import SO3onS2

        bsp = SO3onS2(lmax=2, nlat=16, nlon=32, selective=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch('bispectrum.so3_on_s2._CACHE_DIR', Path(tmpdir)),
                patch('torch.save', side_effect=OSError('disk full')),
            ):
                bsp._build_group_tables(None, bsp.index_map)
            assert not list(Path(tmpdir).iterdir())
        f = torch.randn(1, 16, 32, dtype=torch.float64)
        torch.testing.assert_close(_dense_forward(bsp, f, bsp.output_size), bsp(f))

    def test_lazy_sparse_indices_from_sht_to_flat(self):
        """_sht_to_flat and the dict-based _forward_sparse build the gather tables on demand."""
        from bispectrum import SO3onS2
        from bispectrum.so3_on_s2 import _get_full_sh_coefficients

        bsp = SO3onS2(lmax=3, nlat=16, nlon=32, selective=True)
        f = torch.randn(2, 16, 32, dtype=torch.float64)
        coeffs = bsp._sht(f)
        assert not hasattr(bsp, '_flat_src_l')
        flat = bsp._sht_to_flat(coeffs)
        assert flat.shape == (2, (bsp.lmax + 1) ** 2)

        bsp2 = SO3onS2(lmax=3, nlat=16, nlon=32, selective=True)
        f_coeffs = _get_full_sh_coefficients(coeffs)
        out = bsp2._forward_sparse(f_coeffs, coeffs.dtype, f.device, 2, bsp2.output_size)
        torch.testing.assert_close(out, bsp(f), atol=1e-12, rtol=1e-12)

    def test_entries_without_cg_terms_evaluate_to_zero(self):
        """Triples violating the triangle inequality have no CG terms and give exactly 0."""
        from bispectrum import SO3onS2

        triples = [(1, 1, 2), (1, 1, 3)]
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch('bispectrum.so3_on_s2._CACHE_DIR', Path(tmpdir)),
                patch('bispectrum.so3_on_s2._build_selective_index_map', return_value=triples),
                patch('bispectrum.so3_on_s2._build_cg_power_index_map', return_value=triples),
            ):
                bsp = SO3onS2(lmax=2, nlat=16, nlon=32, selective=True)
        assert bsp.index_map == triples + triples
        out = bsp(torch.randn(3, 16, 32, dtype=torch.float64))
        assert out.shape == (3, 4)
        assert torch.all(out[:, 1] == 0) and torch.all(out[:, 3] == 0)
        assert torch.all(out[:, 0] != 0) and torch.all(out[:, 2] != 0)


class TestSO3SparseCacheIO:
    """Cover sparse CG cache save/load/validation paths."""

    def test_save_sparse_cache_oserror_swallowed(self):
        """OSError during save is silently swallowed."""
        from bispectrum.so3_on_s2 import SO3onS2

        with patch('bispectrum.so3_on_s2._CACHE_DIR', Path('/nonexistent/readonly')):
            SO3onS2._save_sparse_cache(
                Path('/nonexistent/readonly/foo.pt'),
                torch.tensor([1.0]),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0, 1], dtype=torch.int64),
                [[1, 1, 0, 0]],
            )

    def test_load_sparse_cache_corrupt_file(self):
        """Corrupt/unloadable file returns None."""
        from bispectrum.so3_on_s2 import SO3onS2

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'corrupt.pt'
            path.write_text('not a torch file')
            result = SO3onS2._load_sparse_cache(path, [(0, 1, 1, 0, False)])
            assert result is None

    def test_load_sparse_cache_length_mismatch(self):
        """Cache with wrong number of entries returns None."""
        from bispectrum.so3_on_s2 import SO3onS2

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.pt'
            torch.save(
                {
                    'cg_vals': torch.tensor([1.0]),
                    'm1_idx': torch.tensor([0], dtype=torch.int32),
                    'm_idx': torch.tensor([0], dtype=torch.int32),
                    'offsets': torch.tensor([0, 1], dtype=torch.int64),
                    'entry_meta': [[1, 1, 0, 0], [2, 2, 1, 0]],
                },
                path,
            )
            result = SO3onS2._load_sparse_cache(path, [(0, 1, 1, 0, False)])
            assert result is None

    def test_load_sparse_cache_meta_content_mismatch(self):
        """Cache with matching length but different triples returns None."""
        from bispectrum.so3_on_s2 import SO3onS2

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.pt'
            torch.save(
                {
                    'cg_vals': torch.tensor([1.0]),
                    'm1_idx': torch.tensor([0], dtype=torch.int32),
                    'm_idx': torch.tensor([0], dtype=torch.int32),
                    'offsets': torch.tensor([0, 1], dtype=torch.int64),
                    'entry_meta': [[3, 3, 2, 0]],
                },
                path,
            )
            entries = [(0, 1, 1, 0, False)]
            result = SO3onS2._load_sparse_cache(path, entries)
            assert result is None

    def test_load_sparse_cache_valid_roundtrip(self):
        """Valid cache loads successfully."""
        from bispectrum.so3_on_s2 import SO3onS2

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.pt'
            meta = [[1, 1, 0, 0]]
            cg = torch.tensor([0.5774], dtype=torch.float64)
            m1 = torch.tensor([1], dtype=torch.int32)
            m_idx = torch.tensor([0], dtype=torch.int32)
            offsets = torch.tensor([0, 1], dtype=torch.int64)
            SO3onS2._save_sparse_cache(path, cg, m1, m_idx, offsets, meta)
            entries = [(0, 1, 1, 0, False)]
            result = SO3onS2._load_sparse_cache(path, entries)
            assert result is not None
            loaded_cg, loaded_m1, loaded_m, loaded_off, loaded_meta = result
            torch.testing.assert_close(loaded_cg, cg)
            assert loaded_meta == meta


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
