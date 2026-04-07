"""Tests for bispectrum._cg: Clebsch-Gordan coefficient utilities."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from bispectrum._cg import (
    _LOG_FACT,
    _compute_cg_columns_vectorized,
    _compute_cg_matrix_fast,
    _ensure_log_fact,
    _load_cache,
    _save_cache,
    clebsch_gordan,
    compute_cg_columns,
    compute_cg_matrices,
    compute_cg_matrix,
    compute_reduced_cg_parallel,
    load_cg_matrices,
    wigner3j,
)


class TestEnsureLogFact:
    def test_extends_table(self):
        _ensure_log_fact(20)
        assert len(_LOG_FACT) > 20

    def test_correctness(self):
        _ensure_log_fact(10)
        for n in range(1, 11):
            assert abs(_LOG_FACT[n] - math.lgamma(n + 1)) < 1e-12

    def test_idempotent(self):
        _ensure_log_fact(15)
        old_len = len(_LOG_FACT)
        _ensure_log_fact(10)
        assert len(_LOG_FACT) == old_len


class TestWigner3j:
    def test_selection_rule_m_sum(self):
        assert wigner3j(1, 1, 1, 1, 1, 0) == 0.0

    def test_selection_rule_m_exceeds_j(self):
        assert wigner3j(1, 1, 0, 2, -2, 0) == 0.0

    def test_selection_rule_triangle(self):
        assert wigner3j(1, 1, 4, 0, 0, 0) == 0.0
        assert wigner3j(5, 1, 1, 0, 0, 0) == 0.0

    def test_known_value_000(self):
        val = wigner3j(0, 0, 0, 0, 0, 0)
        assert abs(val - 1.0) < 1e-12

    def test_known_value_110(self):
        val = wigner3j(1, 1, 0, 0, 0, 0)
        expected = (-1) ** (1) / math.sqrt(3)
        assert abs(val - expected) < 1e-12

    def test_known_value_112(self):
        val = wigner3j(1, 1, 2, 0, 0, 0)
        expected = math.sqrt(2.0 / 15.0)
        assert abs(val - expected) < 1e-12

    def test_symmetry_column_swap(self):
        """(j1 j2 j3; m1 m2 m3) = (-1)^{j1+j2+j3} (j2 j1 j3; m2 m1 m3)."""
        for j1, j2, j3 in [(1, 2, 2), (2, 1, 3), (1, 1, 2)]:
            for m1 in range(-j1, j1 + 1):
                for m2 in range(-j2, j2 + 1):
                    m3 = -(m1 + m2)
                    if abs(m3) > j3:
                        continue
                    lhs = wigner3j(j1, j2, j3, m1, m2, m3)
                    rhs = ((-1) ** (j1 + j2 + j3)) * wigner3j(j2, j1, j3, m2, m1, m3)
                    assert abs(lhs - rhs) < 1e-12, (
                        f'Symmetry failed for ({j1},{j2},{j3},{m1},{m2},{m3})'
                    )

    def test_sign_flip_all_m(self):
        """(j1 j2 j3; -m1 -m2 -m3) = (-1)^{j1+j2+j3} (j1 j2 j3; m1 m2 m3)."""
        for j1, j2, j3 in [(1, 2, 2), (2, 3, 3)]:
            for m1 in range(-j1, j1 + 1):
                for m2 in range(-j2, j2 + 1):
                    m3 = -(m1 + m2)
                    if abs(m3) > j3:
                        continue
                    lhs = wigner3j(j1, j2, j3, -m1, -m2, -m3)
                    rhs = ((-1) ** (j1 + j2 + j3)) * wigner3j(j1, j2, j3, m1, m2, m3)
                    assert abs(lhs - rhs) < 1e-12

    def test_large_j(self):
        """Smoke test that large j doesn't overflow."""
        val = wigner3j(10, 10, 10, 0, 0, 0)
        assert math.isfinite(val)
        assert val != 0.0


class TestClebschGordan:
    def test_trivial_coupling(self):
        """<0 0; l m | l m> = 1 for all l, m."""
        for l in range(4):
            for m in range(-l, l + 1):
                assert abs(clebsch_gordan(0, 0, l, m, l, m) - 1.0) < 1e-12

    def test_known_cg_1010_20(self):
        val = clebsch_gordan(1, 0, 1, 0, 2, 0)
        expected = math.sqrt(2.0 / 3.0)
        assert abs(val - expected) < 1e-12

    def test_known_cg_1010_00(self):
        val = clebsch_gordan(1, 0, 1, 0, 0, 0)
        expected = -1.0 / math.sqrt(3.0)
        assert abs(val - expected) < 1e-12

    def test_selection_rule_m1_plus_m2(self):
        """<l1 m1; l2 m2 | l m> = 0 when m != m1+m2."""
        assert abs(clebsch_gordan(1, 1, 1, 0, 2, 0)) < 1e-14

    def test_orthonormality_sum_over_m1_m2(self):
        """Sum_{m1,m2} |<l1 m1; l2 m2 | l m>|^2 = 1 for all valid l,m."""
        l1, l2 = 2, 1
        for l_val in range(abs(l1 - l2), l1 + l2 + 1):
            for m in range(-l_val, l_val + 1):
                total = 0.0
                for m1 in range(-l1, l1 + 1):
                    m2 = m - m1
                    if abs(m2) > l2:
                        continue
                    total += clebsch_gordan(l1, m1, l2, m2, l_val, m) ** 2
                assert abs(total - 1.0) < 1e-10, (
                    f'Normalization failed for l1={l1}, l2={l2}, l={l_val}, m={m}: {total}'
                )


class TestComputeCgMatrix:
    @pytest.mark.parametrize('l1,l2', [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)])
    def test_unitarity(self, l1: int, l2: int):
        C = compute_cg_matrix(l1, l2)
        d = (2 * l1 + 1) * (2 * l2 + 1)
        assert C.shape == (d, d)
        I = C.T @ C
        torch.testing.assert_close(I, torch.eye(d, dtype=torch.float64), atol=1e-10, rtol=0)

    @pytest.mark.parametrize('l1,l2', [(0, 0), (0, 1), (1, 1), (2, 1)])
    def test_orthogonality_rows(self, l1: int, l2: int):
        C = compute_cg_matrix(l1, l2)
        I = C @ C.T
        torch.testing.assert_close(
            I, torch.eye(C.shape[0], dtype=torch.float64), atol=1e-10, rtol=0
        )

    def test_shape(self):
        C = compute_cg_matrix(2, 3)
        assert C.shape == (5 * 7, 5 * 7)
        assert C.dtype == torch.float64

    def test_l1_equals_zero(self):
        C = compute_cg_matrix(0, 2)
        expected = torch.eye(5, dtype=torch.float64)
        torch.testing.assert_close(C, expected, atol=1e-12, rtol=0)


class TestComputeCgMatrixFast:
    @pytest.mark.parametrize('l1,l2', [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)])
    def test_matches_reference(self, l1: int, l2: int):
        C_ref = compute_cg_matrix(l1, l2)
        C_fast = _compute_cg_matrix_fast(l1, l2)
        torch.testing.assert_close(C_fast, C_ref, atol=1e-12, rtol=0)

    @pytest.mark.parametrize('l1,l2', [(0, 0), (1, 1), (2, 2), (3, 3)])
    def test_unitarity(self, l1: int, l2: int):
        C = _compute_cg_matrix_fast(l1, l2)
        d = C.shape[0]
        torch.testing.assert_close(C.T @ C, torch.eye(d, dtype=torch.float64), atol=1e-10, rtol=0)


class TestComputeCgColumns:
    def test_subset_of_full_matrix(self):
        l1, l2 = 2, 3
        C_full = compute_cg_matrix(l1, l2)
        l_vals = [2, 3]
        C_cols = compute_cg_columns(l1, l2, l_vals)

        d = (2 * l1 + 1) * (2 * l2 + 1)
        c = sum(2 * lv + 1 for lv in l_vals)
        assert C_cols.shape == (d, c)

        col_offset_full = 0
        col_offset_red = 0
        for l_val in range(abs(l1 - l2), l1 + l2 + 1):
            dl = 2 * l_val + 1
            if l_val in l_vals:
                torch.testing.assert_close(
                    C_cols[:, col_offset_red : col_offset_red + dl],
                    C_full[:, col_offset_full : col_offset_full + dl],
                    atol=1e-12,
                    rtol=0,
                )
                col_offset_red += dl
            col_offset_full += dl

    def test_all_l_vals_equals_full(self):
        l1, l2 = 1, 2
        all_l = list(range(abs(l1 - l2), l1 + l2 + 1))
        C_cols = compute_cg_columns(l1, l2, all_l)
        C_full = compute_cg_matrix(l1, l2)
        torch.testing.assert_close(C_cols, C_full, atol=1e-12, rtol=0)

    def test_single_l_val(self):
        l1, l2 = 2, 2
        C_cols = compute_cg_columns(l1, l2, [2])
        assert C_cols.shape == (25, 5)
        col_norms = C_cols.norm(dim=0)
        assert (col_norms > 1e-14).all()

    def test_shape_for_empty_ish_request(self):
        l1, l2 = 1, 1
        C_cols = compute_cg_columns(l1, l2, [0])
        assert C_cols.shape == (9, 1)


class TestComputeCgColumnsVectorized:
    @pytest.mark.parametrize('l1,l2', [(0, 1), (1, 1), (1, 2), (2, 2), (2, 3)])
    def test_matches_scalar_version(self, l1: int, l2: int):
        all_l = list(range(abs(l1 - l2), l1 + l2 + 1))
        C_scalar = compute_cg_columns(l1, l2, all_l)
        C_vec = _compute_cg_columns_vectorized(l1, l2, all_l)
        torch.testing.assert_close(C_vec, C_scalar, atol=1e-12, rtol=0)

    def test_subset_matches(self):
        l1, l2 = 2, 3
        l_vals = [1, 3, 5]
        C_scalar = compute_cg_columns(l1, l2, l_vals)
        C_vec = _compute_cg_columns_vectorized(l1, l2, l_vals)
        torch.testing.assert_close(C_vec, C_scalar, atol=1e-12, rtol=0)


class TestComputeCgMatrices:
    def test_all_pairs_present(self):
        lmax = 3
        matrices = compute_cg_matrices(lmax)
        for l1 in range(lmax + 1):
            for l2 in range(l1, lmax + 1):
                assert (l1, l2) in matrices

    def test_shapes(self):
        lmax = 2
        matrices = compute_cg_matrices(lmax)
        for (l1, l2), C in matrices.items():
            d = (2 * l1 + 1) * (2 * l2 + 1)
            assert C.shape == (d, d)

    def test_unitarity(self):
        matrices = compute_cg_matrices(2)
        for (_l1, _l2), C in matrices.items():
            d = C.shape[0]
            torch.testing.assert_close(
                C.T @ C, torch.eye(d, dtype=torch.float64), atol=1e-10, rtol=0
            )

    def test_lmax_zero(self):
        matrices = compute_cg_matrices(0)
        assert len(matrices) == 1
        assert (0, 0) in matrices
        torch.testing.assert_close(
            matrices[(0, 0)],
            torch.ones(1, 1, dtype=torch.float64),
            atol=1e-12,
            rtol=0,
        )


class TestComputeReducedCgParallel:
    def test_empty_groups(self):
        result = compute_reduced_cg_parallel([])
        assert result == {}

    def test_single_group(self):
        groups = [(0, 1, 2, [1, 2, 3])]
        result = compute_reduced_cg_parallel(groups, max_workers=1)
        assert 0 in result
        expected = compute_cg_columns(1, 2, [1, 2, 3])
        torch.testing.assert_close(result[0], expected, atol=1e-12, rtol=0)

    def test_multiple_groups(self):
        groups = [
            (0, 1, 1, [0, 1, 2]),
            (1, 2, 2, [0, 2, 4]),
            (2, 0, 3, [3]),
        ]
        result = compute_reduced_cg_parallel(groups, max_workers=2)
        assert len(result) == 3
        for gid, l1, l2, l_vals in groups:
            expected = _compute_cg_columns_vectorized(l1, l2, l_vals)
            torch.testing.assert_close(result[gid], expected, atol=1e-12, rtol=0)

    def test_sequential_fallback(self):
        groups = [(0, 1, 2, [1, 2])]
        result = compute_reduced_cg_parallel(groups, max_workers=1)
        assert 0 in result
        expected = _compute_cg_columns_vectorized(1, 2, [1, 2])
        torch.testing.assert_close(result[0], expected, atol=1e-12, rtol=0)


class TestDiskCache:
    def test_save_and_load_roundtrip(self):
        matrices = compute_cg_matrices(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / 'cg_cache'
            with patch('bispectrum._cg._CACHE_DIR', cache_dir):
                _save_cache(2, matrices)
                loaded = _load_cache(2)
            assert loaded is not None
            for key in matrices:
                torch.testing.assert_close(loaded[key], matrices[key], atol=1e-14, rtol=0)

    def test_load_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / 'no_such_dir'
            with patch('bispectrum._cg._CACHE_DIR', cache_dir):
                assert _load_cache(99) is None

    def test_load_wrong_lmax_returns_none(self):
        matrices = compute_cg_matrices(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / 'cg_cache'
            with patch('bispectrum._cg._CACHE_DIR', cache_dir):
                _save_cache(2, matrices)
                assert _load_cache(3) is None

    def test_load_corrupted_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / 'cg_cache'
            cache_dir.mkdir(parents=True)
            bad_file = cache_dir / 'cg_lmax5.pt'
            bad_file.write_bytes(b'not a valid torch file')
            with patch('bispectrum._cg._CACHE_DIR', cache_dir):
                assert _load_cache(5) is None


class TestLoadCgMatrices:
    def test_returns_correct_keys(self):
        matrices = load_cg_matrices(3)
        for l1 in range(4):
            for l2 in range(l1, 4):
                assert (l1, l2) in matrices

    def test_unitarity(self):
        matrices = load_cg_matrices(2)
        for (_l1, _l2), C in matrices.items():
            d = C.shape[0]
            torch.testing.assert_close(
                C.T @ C, torch.eye(d, dtype=torch.float64), atol=1e-10, rtol=0
            )

    def test_uses_cache_on_second_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / 'cg_cache'
            with patch('bispectrum._cg._CACHE_DIR', cache_dir):
                m1 = load_cg_matrices(2)
                assert (cache_dir / 'cg_lmax2.pt').exists()
                m2 = load_cg_matrices(2)
                for key in m1:
                    torch.testing.assert_close(m1[key], m2[key], atol=1e-14, rtol=0)


class TestWorkerComputeCgColumns:
    def test_worker_matches_vectorized(self):
        from bispectrum._cg import _worker_compute_cg_columns

        args = (42, 2, 3, [1, 3, 5])
        gid, result = _worker_compute_cg_columns(args)
        assert gid == 42
        expected = _compute_cg_columns_vectorized(2, 3, [1, 3, 5])
        torch.testing.assert_close(result, expected, atol=1e-12, rtol=0)


class TestCrossValidationWithKnownValues:
    """Cross-validate computed CG matrices against known analytical identities."""

    def test_cg_completeness(self):
        """Sum_{l,m} <l1,m1;l2,m2|l,m>^2 = 1 for all valid m1,m2."""
        for l1 in range(4):
            for l2 in range(l1, 4):
                for m1 in range(-l1, l1 + 1):
                    for m2 in range(-l2, l2 + 1):
                        m = m1 + m2
                        total = 0.0
                        for l_val in range(abs(l1 - l2), l1 + l2 + 1):
                            if abs(m) > l_val:
                                continue
                            total += clebsch_gordan(l1, m1, l2, m2, l_val, m) ** 2
                        assert abs(total - 1.0) < 1e-10, (
                            f'CG completeness failed for l1={l1}, l2={l2}, m1={m1}, m2={m2}: {total}'
                        )

    def test_clebsch_gordan_max_stretched(self):
        """<l1 l1; l2 l2 | l1+l2 l1+l2> = 1."""
        for l1 in range(5):
            for l2 in range(5):
                val = clebsch_gordan(l1, l1, l2, l2, l1 + l2, l1 + l2)
                assert abs(val - 1.0) < 1e-12, f'Stretched CG failed for l1={l1}, l2={l2}: {val}'

    @pytest.mark.parametrize('lmax', [3, 4, 5])
    def test_analytical_vs_fast_all_pairs(self, lmax: int):
        """Every fast-computed matrix should match the reference implementation."""
        for l1 in range(lmax + 1):
            for l2 in range(l1, lmax + 1):
                C_ref = compute_cg_matrix(l1, l2)
                C_fast = _compute_cg_matrix_fast(l1, l2)
                torch.testing.assert_close(
                    C_fast,
                    C_ref,
                    atol=1e-11,
                    rtol=0,
                    msg=f'Fast vs reference mismatch for ({l1},{l2})',
                )
