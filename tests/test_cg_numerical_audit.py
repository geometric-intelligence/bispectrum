"""High-degree CG accuracy, equivariance and cache-version regressions."""

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

import bispectrum._cg as cg
import bispectrum.so3_on_s2 as spherical


@pytest.mark.parametrize(
    ('ell', 'reference'),
    [(32, 0.018652473651465083), (64, 0.009399178242503595), (96, 0.006282433505226595)],
)
def test_sparse_cg_high_degree_independent_reference_and_normalization(ell: int, reference: float):
    """References are exact SymPy Wigner 3j values rounded from 80 digits."""
    m1, m, values = cg.compute_sparse_cg_entry(ell, ell, ell)
    central = values[(m1 == ell) & (m == ell)].sum()
    assert central == pytest.approx(math.sqrt(2 * ell + 1) * reference, abs=1e-15, rel=1e-13)
    assert np.max(np.abs(values)) <= 1
    column_norms = np.bincount(m, weights=values**2, minlength=2 * ell + 1)
    np.testing.assert_allclose(column_norms, 1, atol=1e-14, rtol=1e-14)


def test_high_degree_exact_selection_zero():
    # Odd-parity repeated-index triples vanish exactly, not by subtracting huge floats.
    assert cg.wigner3j(33, 33, 33, 0, 0, 0) == 0
    m1, m, _ = cg.compute_sparse_cg_entry(33, 33, 33)
    assert not np.any((m1 == 33) & (m == 33))


def test_high_degree_reduced_and_vectorized_columns_use_exact_coefficients():
    ell = 32
    expected = math.sqrt(65) * 0.018652473651465083
    for compute in (cg.compute_cg_columns, cg._compute_cg_columns_vectorized):
        matrix = compute(ell, ell, [ell])
        assert matrix[ell * 65 + ell, ell].item() == pytest.approx(expected, abs=1e-15, rel=1e-13)
        torch.testing.assert_close(
            matrix.T @ matrix, torch.eye(65, dtype=torch.float64), atol=1e-14, rtol=1e-14
        )


def test_fast_matrix_exact_path_remains_unitary():
    # This small tensor product crosses the exact-sum threshold without a large allocation.
    matrix = cg._compute_cg_matrix_fast(1, 32)
    torch.testing.assert_close(
        matrix.T @ matrix, torch.eye(195, dtype=torch.float64), atol=1e-12, rtol=1e-12
    )


def test_high_degree_cubic_is_invariant_to_independent_y_rotation():
    """Use an angular-momentum matrix exponential, independent of the CG implementation."""
    ell = 32
    generator = torch.Generator().manual_seed(709)
    coefficients = torch.randn(65, dtype=torch.complex128, generator=generator)
    coefficients[ell] = coefficients[ell].real.clone()
    for m in range(1, ell + 1):
        coefficients[ell - m] = (-1) ** m * coefficients[ell + m].conj()
    magnetic = torch.arange(-ell, ell, dtype=torch.float64)
    off = 0.5 * torch.sqrt((ell - magnetic) * (ell + magnetic + 1))
    angular_generator = torch.diag(off, diagonal=1) - torch.diag(off, diagonal=-1)
    rotation = torch.matrix_exp(0.41 * angular_generator).to(torch.complex128)
    rotated = rotation @ coefficients
    i_np, m_np, values_np = cg.compute_sparse_cg_entry(ell, ell, ell)
    i, m, values = (
        torch.from_numpy(i_np).long(),
        torch.from_numpy(m_np).long(),
        torch.from_numpy(values_np),
    )
    other = m - i + ell
    expected = (values * coefficients[i] * coefficients[other] * coefficients[m].conj()).sum()
    actual = (values * rotated[i] * rotated[other] * rotated[m].conj()).sum()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_old_cg_cache_namespace_is_not_loaded(tmp_path: Path):
    """Dense, sparse and reduced tables share the versioned numerical namespace."""
    assert cg._CG_CACHE_VERSION == 2
    assert cg._CACHE_DIR.name == 'cg_v2'
    assert spherical._CACHE_DIR == cg._CACHE_DIR
    legacy = tmp_path / 'cg_lmax0.pt'
    torch.save({'__lmax__': torch.tensor(0), '0_0': torch.zeros(1, 1)}, legacy)
    with patch('bispectrum._cg._CACHE_DIR', tmp_path / 'cg_v2'):
        assert cg._load_cache(0) is None
        matrices = cg.load_cg_matrices(0)
        torch.testing.assert_close(matrices[(0, 0)], torch.ones(1, 1, dtype=torch.float64))
        assert cg._cache_path(0).exists()
    assert legacy.exists()
