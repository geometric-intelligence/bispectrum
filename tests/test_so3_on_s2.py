"""Tests for SO3onS2 bispectrum module."""

from collections import Counter

import pytest
import torch

from bispectrum import SO3onS2, random_rotation_matrix, rotate_spherical_function
from bispectrum._cg import load_cg_matrices
from bispectrum.so3_on_s2 import (
    _bispectrum_entry,
    _build_full_index_map,
    _build_selective_index_map,
    _get_full_sh_coefficients,
)


def _linear_rows_for_target(
    index_map: list[tuple[int, int, int]], ell: int
) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    for l1, l2, l_val in index_map:
        if l_val == ell and l1 < ell and l2 < ell:
            rows.append((l1, l2, l_val))
        elif l2 == ell and l1 < ell and l_val < ell:
            rows.append((l1, l2, l_val))
    return rows


def _expected_linear_block(ell: int) -> list[tuple[int, int, int]]:
    explicit_blocks: dict[int, list[tuple[int, int, int]]] = {
        4: [
            (1, 3, 4),
            (2, 2, 4),
            (2, 3, 4),
            (3, 3, 4),
            (1, 4, 3),
            (2, 4, 2),
            (3, 4, 1),
            (2, 4, 3),
            (3, 4, 2),
            (3, 4, 3),
        ],
        5: [
            (1, 4, 5),
            (2, 3, 5),
            (2, 4, 5),
            (3, 4, 5),
            (1, 5, 4),
            (2, 5, 3),
            (3, 5, 2),
            (4, 5, 1),
            (2, 5, 4),
            (3, 5, 4),
            (4, 5, 4),
        ],
        6: [
            (1, 5, 6),
            (2, 4, 6),
            (3, 3, 6),
            (3, 4, 6),
            (1, 6, 5),
            (2, 6, 4),
            (3, 6, 3),
            (4, 6, 2),
            (5, 6, 1),
            (2, 6, 5),
            (3, 6, 5),
            (4, 6, 5),
            (5, 6, 5),
        ],
        7: [
            (1, 6, 7),
            (2, 5, 7),
            (3, 4, 7),
            (4, 5, 7),
            (1, 7, 6),
            (2, 7, 5),
            (3, 7, 4),
            (4, 7, 3),
            (5, 7, 2),
            (6, 7, 1),
            (2, 7, 6),
            (3, 7, 6),
            (4, 7, 6),
            (5, 7, 6),
            (6, 7, 6),
        ],
    }
    if ell in explicit_blocks:
        return explicit_blocks[ell]

    block: list[tuple[int, int, int]] = []
    for a in range(1, ell):
        block.append((a, ell, ell - a))
    for a in range(2, ell):
        block.append((a, ell, ell - a + 1))
    for a in range(1, 5):
        block.append((a, ell - a, ell))
    return block


class TestSO3onS2:
    """Tests for SO3onS2 module."""

    def test_instantiation(self):
        bsp = SO3onS2(lmax=3, nlat=32, nlon=64)
        assert bsp.lmax == 3
        assert bsp.nlat == 32
        assert bsp.nlon == 64
        assert bsp.output_size > 0

    def test_instantiation_defaults(self):
        bsp = SO3onS2()
        assert bsp.lmax == 5
        assert bsp.nlat == 64
        assert bsp.nlon == 128

    def test_large_lmax_computes_analytically(self):
        """CG matrices are now computed analytically, so any lmax works."""
        bsp = SO3onS2(lmax=6, nlat=32, nlon=64)
        assert bsp.output_size > 0

    def test_index_map_structure(self):
        bsp = SO3onS2(lmax=3, nlat=32, nlon=64)
        for l1, l2, l in bsp.index_map:
            assert l1 <= l2
            assert abs(l1 - l2) <= l <= l1 + l2
            assert l1 <= bsp.lmax
            assert l2 <= bsp.lmax
            assert l <= bsp.lmax

    def test_output_size_matches_index_map(self):
        bsp = SO3onS2(lmax=4, nlat=32, nlon=64)
        assert bsp.output_size == len(bsp.index_map)

    def test_reduced_cg_buffers_registered(self):
        bsp = SO3onS2(lmax=2, nlat=32, nlon=64)
        # Each group gets a _cg_red_{gid} buffer with d rows and c ≤ d columns.
        assert len(bsp._group_data) > 0
        for gid, (l1, l2, c, _) in enumerate(bsp._group_data):
            buf = getattr(bsp, f'_cg_red_{gid}')
            d = (2 * l1 + 1) * (2 * l2 + 1)
            assert buf.shape[0] == d
            assert buf.shape[1] == c
            assert c <= d

    def test_forward_output_shape(self):
        nlat, nlon = 32, 64
        batch_size = 4
        lmax = 3
        bsp = SO3onS2(lmax=lmax, nlat=nlat, nlon=nlon)
        f = torch.randn(batch_size, nlat, nlon)
        output = bsp(f)
        assert output.shape == (batch_size, bsp.output_size)
        assert output.is_complex()

    def test_forward_deterministic(self):
        nlat, nlon = 32, 64
        bsp = SO3onS2(lmax=3, nlat=nlat, nlon=nlon)
        f = torch.randn(2, nlat, nlon)
        torch.testing.assert_close(bsp(f), bsp(f))

    def test_device_movement(self):
        bsp = SO3onS2(lmax=2, nlat=32, nlon=64)
        assert bsp._cg_red_0.device.type == 'cpu'
        bsp_cpu = bsp.to('cpu')
        assert bsp_cpu._cg_red_0.device.type == 'cpu'

    def test_no_trainable_parameters(self):
        bsp = SO3onS2(lmax=2, nlat=32, nlon=64)
        assert isinstance(bsp, torch.nn.Module)
        assert sum(p.numel() for p in bsp.parameters()) == 0
        assert sum(1 for _ in bsp.buffers()) > 0

    def test_invert_raises(self):
        bsp = SO3onS2(lmax=2, nlat=32, nlon=64)
        with pytest.raises(NotImplementedError, match='open mathematical problem'):
            bsp.invert(torch.zeros(1, bsp.output_size))

    def test_extra_repr(self):
        bsp = SO3onS2(lmax=4, nlat=32, nlon=64)
        repr_str = repr(bsp)
        assert 'lmax=4' in repr_str
        assert 'nlat=32' in repr_str
        assert 'output_size=' in repr_str


class TestGetFullSHCoefficients:
    """Direct tests for _get_full_sh_coefficients."""

    def test_l0_coefficient(self):
        batch = 3
        coeffs = torch.randn(batch, 4, 4, dtype=torch.complex64)
        result = _get_full_sh_coefficients(coeffs)
        assert 0 in result
        assert result[0].shape == (batch, 1)
        torch.testing.assert_close(result[0][:, 0], coeffs[:, 0, 0])

    def test_negative_m_conjugation(self):
        batch = 2
        lmax = 3
        mmax = lmax
        coeffs = torch.randn(batch, lmax + 1, mmax + 1, dtype=torch.complex128)
        result = _get_full_sh_coefficients(coeffs)
        for l_val in range(1, lmax + 1):
            full = result[l_val]
            for m in range(1, l_val + 1):
                expected_neg_m = ((-1) ** m) * torch.conj(full[:, l_val + m])
                torch.testing.assert_close(full[:, l_val - m], expected_neg_m, atol=1e-12, rtol=0)

    def test_output_sizes(self):
        lmax = 3
        coeffs = torch.randn(2, lmax + 1, lmax + 1, dtype=torch.complex64)
        result = _get_full_sh_coefficients(coeffs)
        for l_val in range(lmax + 1):
            assert l_val in result
            assert result[l_val].shape == (2, 2 * l_val + 1)

    def test_positive_m_preserved(self):
        batch = 2
        lmax = 3
        coeffs = torch.randn(batch, lmax + 1, lmax + 1, dtype=torch.complex128)
        result = _get_full_sh_coefficients(coeffs)
        for l_val in range(lmax + 1):
            for m in range(min(l_val, lmax) + 1):
                torch.testing.assert_close(
                    result[l_val][:, l_val + m],
                    coeffs[:, l_val, m],
                    atol=1e-12,
                    rtol=0,
                )


class TestBispectrumEntry:
    """Direct tests for _bispectrum_entry."""

    def test_output_shape(self):
        batch = 4
        f_coeffs = {
            0: torch.randn(batch, 1, dtype=torch.complex128),
            1: torch.randn(batch, 3, dtype=torch.complex128),
        }
        cg = torch.eye(3, dtype=torch.complex128)
        result = _bispectrum_entry(f_coeffs, 0, 1, 1, cg)
        assert result.shape == (batch,)

    def test_zero_when_l_missing(self):
        batch = 3
        f_coeffs = {
            0: torch.randn(batch, 1, dtype=torch.complex128),
            1: torch.randn(batch, 3, dtype=torch.complex128),
        }
        cg = torch.eye(3, dtype=torch.complex128)
        result = _bispectrum_entry(f_coeffs, 0, 1, 5, cg)
        torch.testing.assert_close(result, torch.zeros(batch, dtype=torch.complex128))

    def test_trivial_cg_l0_l0(self):
        """For l1=l2=l=0, CG is 1x1 identity: beta = f0 * f0 * conj(f0) = |f0|^2 * f0."""
        batch = 2
        f0 = torch.randn(batch, 1, dtype=torch.complex128)
        f_coeffs = {0: f0}
        cg = torch.ones(1, 1, dtype=torch.complex128)
        result = _bispectrum_entry(f_coeffs, 0, 0, 0, cg)
        expected = f0[:, 0] * f0[:, 0] * torch.conj(f0[:, 0])
        torch.testing.assert_close(result, expected, atol=1e-12, rtol=0)


class TestSO3onS2RotationInvariance:
    """Test that SO3onS2 bispectrum is invariant under rotations."""

    def test_rotation_invariance(self):
        nlat, nlon = 64, 128
        batch_size = 2
        lmax = 4
        bsp = SO3onS2(lmax=lmax, nlat=nlat, nlon=nlon)

        f = torch.randn(batch_size, nlat, nlon, dtype=torch.float64)
        beta_f = bsp(f.float())

        R = random_rotation_matrix()
        f_rotated = rotate_spherical_function(f, R)
        beta_f_rotated = bsp(f_rotated.float())

        torch.testing.assert_close(
            beta_f.abs(),
            beta_f_rotated.abs(),
            atol=0.1,
            rtol=0.1,
            msg='Bispectrum magnitude should be approximately invariant under rotation',
        )


class TestSelectiveSO3onS2:
    """Tests for the selective (O(L²)) bispectrum mode."""

    def test_output_size_is_quadratic(self):
        """Selective output size: (lmax+1)² - 4 for lmax in {2,3}, (lmax+1)² - 3 for lmax >= 4."""
        for lmax in [2, 3]:
            bsp = SO3onS2(lmax=lmax, nlat=32, nlon=64, selective=True)
            expected = (lmax + 1) ** 2 - 4
            assert bsp.output_size == expected, (
                f'lmax={lmax}: got {bsp.output_size}, expected {expected}'
            )
        for lmax in [4, 5, 6]:
            bsp = SO3onS2(lmax=lmax, nlat=32, nlon=64, selective=True)
            expected = (lmax + 1) ** 2 - 3
            assert bsp.output_size == expected, (
                f'lmax={lmax}: got {bsp.output_size}, expected {expected}'
            )

    def test_output_smaller_than_full(self):
        bsp_full = SO3onS2(lmax=5, nlat=32, nlon=64, selective=False)
        bsp_sel = SO3onS2(lmax=5, nlat=32, nlon=64, selective=True)
        assert bsp_sel.output_size < bsp_full.output_size

    def test_index_map_validity(self):
        """All selective triples satisfy l1 <= l2 and the triangle inequality."""
        for lmax in [1, 2, 3, 4, 5]:
            bsp = SO3onS2(lmax=lmax, nlat=32, nlon=64, selective=True)
            for l1, l2, l in bsp.index_map:
                assert l1 <= l2, f'l1 > l2: {(l1, l2, l)}'
                assert abs(l1 - l2) <= l <= l1 + l2, f'triangle: {(l1, l2, l)}'
                assert l <= lmax, f'exceeds lmax: {(l1, l2, l)}'

    def test_selective_entries_match_full(self):
        """Every selective entry must match the full bispectrum on the same triple."""
        full = SO3onS2(lmax=5, nlat=64, nlon=128, selective=False)
        sel = SO3onS2(lmax=5, nlat=64, nlon=128, selective=True)
        f = torch.randn(3, 64, 128)

        beta_full = full(f)
        beta_sel = sel(f)

        full_lookup = {triple: i for i, triple in enumerate(full.index_map)}
        for i, triple in enumerate(sel.index_map):
            j = full_lookup[triple]
            torch.testing.assert_close(
                beta_sel[:, i],
                beta_full[:, j],
                atol=1e-10,
                rtol=0,
            )

    def test_forward_shape(self):
        nlat, nlon, batch = 32, 64, 4
        bsp = SO3onS2(lmax=3, nlat=nlat, nlon=nlon, selective=True)
        output = bsp(torch.randn(batch, nlat, nlon))
        assert output.shape == (batch, bsp.output_size)
        assert output.is_complex()

    def test_forward_deterministic(self):
        bsp = SO3onS2(lmax=3, nlat=32, nlon=64, selective=True)
        f = torch.randn(2, 32, 64)
        torch.testing.assert_close(bsp(f), bsp(f))

    def test_rotation_invariance(self):
        bsp = SO3onS2(lmax=4, nlat=64, nlon=128, selective=True)
        f = torch.randn(2, 64, 128, dtype=torch.float64)
        beta_f = bsp(f.float())

        R = random_rotation_matrix()
        f_rot = rotate_spherical_function(f, R)
        beta_rot = bsp(f_rot.float())

        torch.testing.assert_close(
            beta_f.abs(),
            beta_rot.abs(),
            atol=0.1,
            rtol=0.1,
            msg='Selective bispectrum magnitude should be rotation-invariant',
        )

    def test_no_trainable_parameters(self):
        bsp = SO3onS2(lmax=3, nlat=32, nlon=64, selective=True)
        assert sum(p.numel() for p in bsp.parameters()) == 0
        assert sum(1 for _ in bsp.buffers()) > 0

    def test_extra_repr_shows_selective(self):
        bsp = SO3onS2(lmax=3, nlat=32, nlon=64, selective=True)
        assert 'selective=True' in repr(bsp)

    def test_degree_coverage(self):
        """Each degree 0..lmax must have at least one entry."""
        for lmax in [1, 2, 3, 4, 5]:
            bsp = SO3onS2(lmax=lmax, nlat=32, nlon=64, selective=True)
            degrees_present = set()
            for l1, l2, l in bsp.index_map:
                degrees_present.update([l1, l2, l])
            for deg in range(lmax + 1):
                assert deg in degrees_present, f'degree {deg} missing for lmax={lmax}'

    def test_seed_entries_present(self):
        """The seed entries (0,0,0) and (0,1,1) must always be present."""
        for lmax in [1, 2, 5]:
            idx = _build_selective_index_map(lmax)
            assert (0, 0, 0) in idx
            if lmax >= 1:
                assert (0, 1, 1) in idx


class TestBuildSelectiveIndexMap:
    """Unit tests for the _build_selective_index_map helper."""

    def test_lmax_zero(self):
        idx = _build_selective_index_map(0)
        assert idx == [(0, 0, 0)]

    def test_lmax_one(self):
        idx = _build_selective_index_map(1)
        assert idx == [(0, 0, 0), (0, 1, 1)], (
            f'lmax=1 should have exactly (0,0,0) and (0,1,1), got {idx}'
        )
        assert (1, 1, 0) not in idx, 'beta_{1,1,0} is redundant and must be excluded'

    def test_no_beta_110(self):
        """beta_{1,1,0} = -beta_{0,1,1}/sqrt(3) for real signals, so it must never appear in the
        selective index set (same pattern as beta_{2,2,0})."""
        for lmax in range(1, 8):
            idx = _build_selective_index_map(lmax)
            assert (1, 1, 0) not in idx, f'(1,1,0) found for lmax={lmax}'

    def test_beta_222_at_l2(self):
        """beta_{2,2,2} (symmetric self-coupling) replaces the excluded beta_{1,1,0} and must be
        present for lmax >= 2."""
        for lmax in range(2, 8):
            idx = _build_selective_index_map(lmax)
            assert (2, 2, 2) in idx, f'(2,2,2) missing for lmax={lmax}'

    def test_no_beta_121(self):
        """beta_{1,2,1} collapses to a scalar multiple of beta_{1,1,2} after gauge-fixing, so it
        must never appear in the selective set."""
        for lmax in range(2, 8):
            idx = _build_selective_index_map(lmax)
            assert (1, 2, 1) not in idx, f'(1,2,1) found for lmax={lmax}'

    def test_l4_overdetermined(self):
        """At l=4, all 10 chain+cross candidates are kept (overdetermined system)."""
        for lmax in range(4, 8):
            idx = _build_selective_index_map(lmax)
            l4_entries = [t for t in idx if max(t) == 4]
            assert len(l4_entries) == 10, f'l=4 should have 10 entries, got {len(l4_entries)}'
            assert (3, 4, 3) in idx, f'(3,4,3) missing for lmax={lmax}'

    def test_no_duplicates(self):
        for lmax in range(6):
            idx = _build_selective_index_map(lmax)
            assert len(idx) == len(set(idx)), f'duplicates for lmax={lmax}'

    def test_budget_respected(self):
        """No degree should have more than its budget (10 at l=4, 2l+1 otherwise)."""
        for lmax in range(8):
            idx = _build_selective_index_map(lmax)
            counts = Counter()
            for l1, l2, l in idx:
                counts[max(l1, l2, l)] += 1
            for deg, count in counts.items():
                budget = 10 if deg == 4 else 2 * deg + 1
                assert count <= budget, f'degree {deg} has {count} entries > budget {budget}'

    def test_linear_blocks_match_proved_family(self):
        for ell in range(4, 11):
            idx = _build_selective_index_map(ell)
            actual = _linear_rows_for_target(idx, ell)
            expected = _expected_linear_block(ell)
            assert actual == expected


class TestBatchedForwardCorrectness:
    """Verify the batched forward pass matches the scalar _bispectrum_entry reference."""

    @pytest.mark.parametrize('lmax', [1, 2, 3, 4, 5])
    def test_batched_matches_scalar_full(self, lmax: int):
        nlat, nlon = 32, 64
        bsp = SO3onS2(lmax=lmax, nlat=nlat, nlon=nlon, selective=False)
        torch.manual_seed(lmax)
        f = torch.randn(2, nlat, nlon)

        beta_batched = bsp(f)
        coeffs = bsp._sht(f)
        f_coeffs = _get_full_sh_coefficients(coeffs)
        cg_data = load_cg_matrices(lmax)

        for i, (l1, l2, l_val) in enumerate(bsp.index_map):
            beta_ref = _bispectrum_entry(f_coeffs, l1, l2, l_val, cg_data[(l1, l2)])
            torch.testing.assert_close(
                beta_batched[:, i],
                beta_ref,
                atol=1e-6,
                rtol=1e-5,
                msg=f'Mismatch at ({l1},{l2},{l_val}) for lmax={lmax}',
            )

    @pytest.mark.parametrize('lmax', [1, 2, 3, 4, 5])
    def test_batched_matches_scalar_selective(self, lmax: int):
        nlat, nlon = 32, 64
        bsp = SO3onS2(lmax=lmax, nlat=nlat, nlon=nlon, selective=True)
        torch.manual_seed(lmax + 100)
        f = torch.randn(2, nlat, nlon)

        beta_batched = bsp(f)
        coeffs = bsp._sht(f)
        f_coeffs = _get_full_sh_coefficients(coeffs)
        cg_data = load_cg_matrices(lmax)

        for i, (l1, l2, l_val) in enumerate(bsp.index_map):
            beta_ref = _bispectrum_entry(f_coeffs, l1, l2, l_val, cg_data[(l1, l2)])
            torch.testing.assert_close(
                beta_batched[:, i],
                beta_ref,
                atol=1e-6,
                rtol=1e-5,
                msg=f'Mismatch at ({l1},{l2},{l_val}) for lmax={lmax} selective',
            )

    def test_output_nonzero(self):
        """Bispectrum of a non-zero signal must be non-zero."""
        for selective in [False, True]:
            bsp = SO3onS2(lmax=3, nlat=32, nlon=64, selective=selective)
            torch.manual_seed(0)
            f = torch.randn(1, 32, 64)
            beta = bsp(f)
            assert beta.abs().max() > 1e-10, f'Bispectrum is all zeros (selective={selective})'


class TestCompletenessNumerical:
    """Numerical tests for the completeness claim."""

    def test_discriminativeness(self):
        """Two random signals (not rotation-related) must produce different bispectra."""
        bsp = SO3onS2(lmax=4, nlat=32, nlon=64, selective=True)
        torch.manual_seed(42)
        f1 = torch.randn(1, 32, 64)
        f2 = torch.randn(1, 32, 64)
        beta1 = bsp(f1)
        beta2 = bsp(f2)
        diff = (beta1 - beta2).abs().max()
        assert diff > 1e-6, 'Two random signals gave the same bispectrum'

    @pytest.mark.parametrize('lmax', [3, 4])
    def test_jacobian_rank_finite_diff(self, lmax: int):
        """Jacobian of the selective bispectrum w.r.t. real SH coefficient parameters should have
        well-defined numerical rank.

        Uses autograd for exact Jacobian computation. The generic rank of
        the scalar bispectrum for real-valued signals on S^2 is lower than
        (L+1)^2 - 3 because the conjugacy constraint f_l^{-m} = (-1)^m
        conj(f_l^m) introduces polynomial identities among bispectral
        entries. The full bispectrum rank follows:
          even L = 2K:  (K+1)(K+2)(2K+3) / 6
          odd  L = 2K+1: (K+1)(K+2)(K+3) / 3
        """
        cg_data = load_cg_matrices(lmax)
        idx_map = _build_selective_index_map(lmax)
        torch.manual_seed(42)

        n_params = (lmax + 1) ** 2
        params = torch.randn(n_params, dtype=torch.float64)

        def _params_to_coeffs(p: torch.Tensor) -> dict[int, torch.Tensor]:
            f_coeffs: dict[int, torch.Tensor] = {}
            idx = 0
            for l_val in range(lmax + 1):
                neg_parts: list[torch.Tensor] = []
                pos_parts: list[torch.Tensor] = []
                for m in range(1, l_val + 1):
                    re, im = p[idx], p[idx + 1]
                    pos_parts.append(torch.complex(re, im))
                    neg_parts.append(((-1.0) ** m) * torch.complex(re, -im))
                    idx += 2
                m0 = torch.complex(p[idx], torch.zeros_like(p[idx]))
                idx += 1
                coeffs = list(reversed(neg_parts)) + [m0] + pos_parts
                f_coeffs[l_val] = torch.stack(coeffs).unsqueeze(0)
            return f_coeffs

        def _eval_bispec(p: torch.Tensor) -> torch.Tensor:
            fc = _params_to_coeffs(p)
            return torch.stack(
                [
                    _bispectrum_entry(fc, l1, l2, lv, cg_data[(l1, l2)])[0].real
                    for l1, l2, lv in idx_map
                ]
            )

        J = torch.autograd.functional.jacobian(_eval_bispec, params)
        sv = torch.linalg.svdvals(J)

        ratios = sv[:-1] / torch.clamp(sv[1:], min=1e-20)
        rank = int(ratios.argmax()) + 1
        gap = ratios.max().item()

        assert gap > 1e6, f'No clean SVD gap for lmax={lmax}: max ratio={gap:.2e} at rank={rank}'
        assert rank >= lmax + 1, (
            f'Rank {rank} too low for lmax={lmax} '
            f'(must exceed power-spectrum contribution of {lmax + 1})'
        )


class TestBuildFullIndexMap:
    """Tests for _build_full_index_map."""

    def test_all_triples_valid(self):
        lmax = 3
        cg_data = load_cg_matrices(lmax)
        idx_map = _build_full_index_map(lmax, cg_data)
        for l1, l2, l_val in idx_map:
            assert l1 <= l2
            assert abs(l1 - l2) <= l_val <= l1 + l2
            assert l_val <= lmax
            assert (l1, l2) in cg_data

    def test_no_duplicates(self):
        lmax = 4
        cg_data = load_cg_matrices(lmax)
        idx_map = _build_full_index_map(lmax, cg_data)
        assert len(idx_map) == len(set(idx_map))

    def test_missing_cg_pair_skipped(self):
        """When a (l1, l2) pair is absent from cg_data, its triples are skipped."""
        lmax = 3
        cg_data = load_cg_matrices(lmax)
        full_map = _build_full_index_map(lmax, cg_data)

        dropped_key = (1, 2)
        partial_cg = {k: v for k, v in cg_data.items() if k != dropped_key}
        partial_map = _build_full_index_map(lmax, partial_cg)

        dropped = [t for t in full_map if t not in partial_map]
        assert len(dropped) > 0, 'Removing a CG pair should drop some triples'
        for l1, l2, _ in dropped:
            assert (l1, l2) == dropped_key
        for l1, l2, _ in partial_map:
            assert (l1, l2) != dropped_key

    def test_lmax_zero(self):
        cg_data = load_cg_matrices(0)
        idx_map = _build_full_index_map(0, cg_data)
        assert idx_map == [(0, 0, 0)]

    def test_empty_cg_data(self):
        idx_map = _build_full_index_map(3, {})
        assert idx_map == []


class TestSelectiveLmax0:
    """Edge case: selective mode with lmax=0."""

    def test_lmax0_selective(self):
        bsp = SO3onS2(lmax=0, nlat=16, nlon=32, selective=True)
        assert bsp.output_size == 1
        f = torch.randn(2, 16, 32)
        beta = bsp(f)
        assert beta.shape == (2, 1)
        assert beta.abs().max() > 0


class TestForwardEdgeCases:
    """Edge cases for the batched forward pass."""

    def test_empty_index_map(self):
        """Forward with zero output entries returns empty tensor."""
        bsp = SO3onS2(lmax=2, nlat=32, nlon=64)
        bsp._index_map = []
        bsp._group_data = []
        f = torch.randn(3, 32, 64)
        result = bsp(f)
        assert result.shape == (3, 0)

    def test_batch_size_one(self):
        bsp = SO3onS2(lmax=3, nlat=32, nlon=64, selective=True)
        f = torch.randn(1, 32, 64)
        result = bsp(f)
        assert result.shape == (1, bsp.output_size)
        assert result.is_complex()
        assert result.abs().max() > 1e-10

    def test_float64_input(self):
        """Forward with double-precision input."""
        bsp = SO3onS2(lmax=2, nlat=32, nlon=64)
        f = torch.randn(2, 32, 64, dtype=torch.float64)
        result = bsp(f)
        assert result.shape == (2, bsp.output_size)
        assert result.abs().max() > 1e-10

    def test_large_batch(self):
        bsp = SO3onS2(lmax=2, nlat=32, nlon=64, selective=True)
        f = torch.randn(32, 32, 64)
        result = bsp(f)
        assert result.shape == (32, bsp.output_size)

    def test_group_independence(self):
        """Each (l1, l2) group contributes independently to the result."""
        bsp = SO3onS2(lmax=3, nlat=32, nlon=64, selective=True)
        f = torch.randn(2, 32, 64)
        beta = bsp(f)

        groups_seen = set()
        for l1, l2, _ in bsp.index_map:
            groups_seen.add((l1, l2))
        assert len(groups_seen) > 1, 'Need multiple groups for this test'

        for l1, l2 in groups_seen:
            group_indices = [i for i, (a, b, _) in enumerate(bsp.index_map) if (a, b) == (l1, l2)]
            group_result = beta[:, group_indices]
            assert group_result.abs().max() > 0, f'Group ({l1},{l2}) produced all zeros'
