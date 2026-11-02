"""Tests for SO2onDisk selective disk bispectrum module."""

import math

import pytest
import torch
import torch.nn.functional as F

from bispectrum import SO2onDisk
from bispectrum._bessel import (
    _bisect_newton_batch,
    _mcmahon_zeros_j0,
    bessel_jn,
    bessel_jn_zeros,
    compute_all_bessel_roots,
)

# mpmath (40 digits) reference values for J_n(x).  Covers x = 3.5, 8, 12
# where torch.special.bessel_j0/j1 are only ~5e-7 accurate in float64.
_JN_REFERENCE: list[tuple[int, float, float]] = [
    (0, 0.5, 0.9384698072408129),
    (0, 3.5, -0.38012773998726338),
    (0, 8.0, 0.17165080713755391),
    (0, 12.0, 0.047689310796833537),
    (0, 40.0, 0.0073668905842372896),
    (0, 150.0, -0.00077409037539429125),
    (1, 0.5, 0.24226845767487389),
    (1, 3.5, 0.13737752736232719),
    (1, 8.0, 0.23463634685391462),
    (1, 12.0, -0.22344710449062761),
    (1, 40.0, 0.126038318037585),
    (1, 150.0, -0.06514516365772736),
    (2, 0.5, 0.030604023458682641),
    (2, 3.5, 0.45862918419430748),
    (2, 8.0, -0.11299172042407525),
    (2, 12.0, -0.084930494878604805),
    (2, 40.0, -0.0010649746823580396),
    (2, 150.0, -9.4511806708740224e-5),
    (5, 0.5, 8.0536272413574741e-6),
    (5, 3.5, 0.080441986647991782),
    (5, 8.0, 0.18577477219056331),
    (5, 12.0, -0.073470963101658581),
    (5, 40.0, 0.12257346597711779),
    (5, 150.0, -0.064998631740725847),
    (20, 0.5, 3.7272019617047145e-31),
    (20, 3.5, 2.5768553102807591e-14),
    (20, 8.0, 2.0805829639717028e-7),
    (20, 12.0, 0.00025121327024539953),
    (20, 40.0, 0.1277939335508489),
    (20, 150.0, 0.063447240953861973),
]

# mpmath besseljzero(n, k) references.
_ROOT_REFERENCE: list[tuple[int, int, float]] = [
    (0, 1, 2.4048255576957728),
    (0, 5, 14.930917708487786),
    (0, 20, 62.04846919022717),
    (1, 1, 3.8317059702075123),
    (1, 7, 22.760084380592772),
    (2, 3, 11.619841172149059),
    (5, 4, 18.980133875179921),
    (10, 10, 45.231574103535045),
    (20, 8, 51.860019928074574),
    (40, 3, 56.658313455430063),
]


class TestBessel:
    @pytest.mark.parametrize('n, x, expected', _JN_REFERENCE)
    def test_jn_matches_mpmath(self, n: int, x: float, expected: float):
        """J_n must be accurate to double precision for every order, including n = 0, 1.

        ``torch.special.bessel_j0/j1`` are only ~5e-7 accurate on x in [2.8, 14.5]
        (torch 2.11 arm64), so they cannot be used as base cases.
        """
        got = bessel_jn(n, torch.tensor([x], dtype=torch.float64)).item()
        assert got == pytest.approx(expected, abs=1e-14, rel=1e-13)

    def test_jn_matches_mpmath_batched(self):
        """Mixed orders/arguments in one tensor evaluate identically to scalar calls."""
        for n in (0, 1, 2, 5, 20):
            rows = [(x, v) for nn, x, v in _JN_REFERENCE if nn == n]
            xs = torch.tensor([x for x, _ in rows], dtype=torch.float64)
            expected = torch.tensor([v for _, v in rows], dtype=torch.float64)
            torch.testing.assert_close(bessel_jn(n, xs), expected, atol=1e-14, rtol=1e-13)

    def test_j0_at_zero_is_one(self):
        x = torch.tensor([0.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(bessel_jn(0, x), torch.ones_like(x), atol=0, rtol=0)

    def test_jn_at_zero(self):
        assert bessel_jn(0, torch.tensor([0.0])).item() == pytest.approx(1.0)
        for n in [1, 2, 5, 10]:
            assert bessel_jn(n, torch.tensor([0.0])).item() == pytest.approx(0.0)

    def test_jn_float32_input(self):
        x = torch.tensor([0.5, 3.5, 12.0], dtype=torch.float32)
        got = bessel_jn(1, x)
        assert got.dtype == torch.float32
        expected = torch.tensor(
            [v for n, xx, v in _JN_REFERENCE if n == 1 and xx in (0.5, 3.5, 12.0)],
            dtype=torch.float32,
        )
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)

    def test_negative_order_raises(self):
        with pytest.raises(ValueError, match='Order n must be >= 0'):
            bessel_jn(-1, torch.tensor([1.0]))

    @pytest.mark.parametrize('n, k, expected_root', _ROOT_REFERENCE)
    def test_roots_match_mpmath(self, n: int, k: int, expected_root: float):
        roots = bessel_jn_zeros(n, k)
        assert roots[k - 1].item() == pytest.approx(expected_root, abs=1e-13)

    @pytest.mark.parametrize(
        'n, expected_first_root',
        [(0, 2.4048255576957728), (1, 3.8317059702075123), (2, 5.1356223018406826)],
    )
    def test_known_roots(self, n: int, expected_first_root: float):
        roots = bessel_jn_zeros(n, 3)
        assert roots[0].item() == pytest.approx(expected_first_root, abs=1e-13)

    @pytest.mark.parametrize(
        'n, x, expected',
        [
            # Reference values from mpmath (50 digits). The old forward-only
            # recurrence returned ~1e+11 for J_20(0.5).
            (10, 0.5, 2.6131773608228023e-13),
            (20, 0.5, 3.7272019617047145e-31),
            (20, 10.0, 1.1513369247813403e-05),
            (30, 1.0, 3.4828697942514834e-42),
            (40, 2.0, 1.1960774581136798e-48),
            (60, 30.0, 9.8075576431286213e-14),
        ],
    )
    def test_jn_small_argument_reference_values(self, n: int, x: float, expected: float):
        """Backward (Miller) recurrence is accurate in the x < n regime."""
        got = bessel_jn(n, torch.tensor([x], dtype=torch.float64)).item()
        assert got == pytest.approx(expected, rel=1e-12)

    def test_jn_continuous_across_regime_boundary(self):
        """Forward/backward switch at |x| = n must not create a jump."""
        for n in (5, 12, 25):
            x = torch.linspace(n - 0.5, n + 0.5, 101, dtype=torch.float64)
            vals = bessel_jn(n, x)
            jumps = (vals[1:] - vals[:-1]).abs()
            assert jumps.max().item() < 1e-2
            assert torch.isfinite(vals).all()

    def test_jn_odd_symmetry_negative_x(self):
        """J_n(-x) = (-1)^n J_n(x)."""
        x = torch.linspace(0.1, 10.0, 50, dtype=torch.float64)
        for n in (3, 8):
            sign = -1.0 if n % 2 == 1 else 1.0
            torch.testing.assert_close(bessel_jn(n, -x), sign * bessel_jn(n, x))

    @pytest.mark.parametrize(
        'n, k, expected_root',
        [
            # mpmath besseljzero references, including cases where the old
            # batch bisection walked away from an already-converged root
            # (e.g. j_{6,5} came back as 25.43 instead of 23.586).
            (6, 5, 23.586084435581391),
            (10, 9, 42.004190236671805),
            (20, 8, 51.860019928074567),
        ],
    )
    def test_high_order_roots_match_reference(self, n: int, k: int, expected_root: float):
        roots = bessel_jn_zeros(n, k)
        assert roots[k - 1].item() == pytest.approx(expected_root, abs=1e-12)

    def test_roots_interlace(self):
        """j_{n-1,k} < j_{n,k} < j_{n-1,k+1} for all consecutive orders.

        The old solver returned roots of J_{n-1} in J_n slots, which violates strict interlacing.
        """
        all_roots = compute_all_bessel_roots(20, 10)
        for n in range(1, 21):
            prev = all_roots[n - 1]
            curr = all_roots[n]
            for k in range(min(len(curr), len(prev) - 1)):
                assert prev[k] < curr[k] < prev[k + 1], f'Interlacing violated at n={n}, k={k + 1}'

    def test_root_residuals_high_orders(self):
        all_roots = compute_all_bessel_roots(20, 10)
        for n, roots in all_roots.items():
            vals = bessel_jn(n, torch.tensor(roots, dtype=torch.float64))
            assert vals.abs().max().item() < 1e-13, f'Large residual at order {n}'

    @pytest.mark.parametrize('n', [0, 1, 2, 5, 10])
    def test_jn_at_zeros_is_zero(self, n: int):
        roots = bessel_jn_zeros(n, 5)
        vals = bessel_jn(n, roots)
        assert vals.abs().max().item() < 1e-12

    def test_roots_monotonically_increasing(self):
        for n in [0, 1, 5, 10]:
            roots = bessel_jn_zeros(n, 10)
            diffs = roots[1:] - roots[:-1]
            assert (diffs > 0).all()

    def test_compute_all_roots_consistency(self):
        all_roots = compute_all_bessel_roots(10, 5)
        for n in [0, 3, 7, 10]:
            single = bessel_jn_zeros(n, 5)
            batch = torch.tensor(all_roots[n][:5], dtype=torch.float64)
            torch.testing.assert_close(single, batch)

    def test_empty_zeros(self):
        assert bessel_jn_zeros(0, 0).shape == (0,)

    def test_mcmahon_first_10_roots(self):
        """McMahon J_0 roots should be self-consistent: J_0(root) ≈ 0."""
        roots = _mcmahon_zeros_j0(10)
        assert len(roots) == 10
        for i, r in enumerate(roots):
            val = bessel_jn(0, torch.tensor([r], dtype=torch.float64)).item()
            assert abs(val) < 1e-12, f'J_0(root[{i}]={r}) = {val}, expected ≈ 0'
        for i in range(len(roots) - 1):
            assert roots[i] < roots[i + 1], f'Roots not monotonically increasing at {i}'

    def test_bisect_newton_batch_j0_known_root(self):
        """Order-0 brackets use J_0' = -J_1 for the Newton step."""
        a = torch.tensor([2.0, 5.0], dtype=torch.float64)
        b = torch.tensor([3.0, 6.0], dtype=torch.float64)
        roots = _bisect_newton_batch(0, a, b)
        torch.testing.assert_close(
            roots,
            torch.tensor([2.404825557695773, 5.520078110286311], dtype=torch.float64),
            atol=1e-14,
            rtol=0.0,
        )

    def test_bisect_newton_batch_j1_known_root(self):
        a = torch.tensor([3.5], dtype=torch.float64)
        b = torch.tensor([4.5], dtype=torch.float64)
        root = _bisect_newton_batch(1, a, b).item()
        val = bessel_jn(1, torch.tensor([root], dtype=torch.float64)).item()
        assert abs(val) < 1e-14, f'J_1({root}) = {val}'
        assert 3.831 < root < 3.832

    def test_bisect_newton_batch_degenerate_bracket(self):
        """When fa * fb > 0, should return midpoint."""
        a = torch.tensor([0.1], dtype=torch.float64)
        b = torch.tensor([0.5], dtype=torch.float64)
        root = _bisect_newton_batch(0, a, b).item()
        assert root == pytest.approx(0.3, abs=1e-6)


class TestDiskBasisConditioning:
    @pytest.mark.parametrize('L', [16, 32])
    def test_basis_matrix_full_rank(self, L: int):
        """The DHT basis must have full column rank.

        With the unstable forward-only Bessel recurrence the L=16 basis already lost rank (and L=32
        collapsed to rank 5 of 804).
        """
        bsp = SO2onDisk(L=L)
        rank = torch.linalg.matrix_rank(bsp._phi).item()
        n_cols = bsp._phi.shape[1]
        assert rank == n_cols, f'rank {rank} of {n_cols}'

    @pytest.mark.parametrize('L', [16, 32, 64])
    def test_default_basis_well_conditioned(self, L: int):
        """The default truncation must give a well-conditioned analysis operator.

        The former default (fle_2d count ``ne = floor(pi L^2 / 4)``, about one basis function per
        disk pixel) made Phi nearly square with cond ~1e16; ``pinv(rcond=1e-10)`` then silently
        discarded directions and every roundtrip was only accurate to O(1).  Sampling theory bounds
        the resolvable radial frequency on an L x L grid by the Nyquist rate pi * L / 2, where the
        basis is close to orthogonal on the pixel set.
        """
        bsp = SO2onDisk(L=L)
        cond = torch.linalg.cond(bsp._phi).item()
        assert cond < 10.0, f'L={L}: cond(Phi) = {cond:.3e}'


class TestRealComplexConversion:
    """Direct tests for _real_to_complex / _complex_to_real roundtrip."""

    def test_roundtrip_real_complex_real(self):
        bsp = SO2onDisk(L=8)
        torch.manual_seed(42)
        d_real = bsp._phi.shape[1]
        x = torch.randn(3, d_real, dtype=torch.float64)
        a = bsp._real_to_complex(x)
        x_rec = bsp._complex_to_real(a)
        torch.testing.assert_close(x_rec, x, atol=1e-14, rtol=0)

    def test_roundtrip_complex_real_complex(self):
        bsp = SO2onDisk(L=8)
        torch.manual_seed(43)
        a = torch.randn(2, bsp._m_nonneg, dtype=torch.complex128)
        for j, (n, _k) in enumerate(bsp._nonneg_indices):
            if n == 0:
                a[:, j] = a[:, j].real + 0j
        x = bsp._complex_to_real(a)
        a_rec = bsp._real_to_complex(x)
        torch.testing.assert_close(a_rec, a, atol=1e-14, rtol=0)

    def test_n0_entries_are_real(self):
        bsp = SO2onDisk(L=8)
        torch.manual_seed(44)
        d_real = bsp._phi.shape[1]
        x = torch.randn(2, d_real, dtype=torch.float64)
        a = bsp._real_to_complex(x)
        for j, (n, _k) in enumerate(bsp._nonneg_indices):
            if n == 0:
                assert a[:, j].imag.abs().max().item() == 0.0


class TestSO2onDiskConstruction:
    def test_instantiation(self):
        bsp = SO2onDisk(L=8)
        assert bsp.L == 8
        assert bsp.selective is True

    def test_no_trainable_parameters(self):
        bsp = SO2onDisk(L=8)
        assert isinstance(bsp, torch.nn.Module)
        assert sum(p.numel() for p in bsp.parameters()) == 0

    def test_extra_repr(self):
        bsp = SO2onDisk(L=8)
        r = repr(bsp)
        assert 'L=8' in r
        assert 'selective=True' in r

    def test_index_map_length(self):
        bsp = SO2onDisk(L=8)
        assert len(bsp.index_map) == bsp.output_size

    def test_index_map_types(self):
        bsp = SO2onDisk(L=8)
        for t, n, k in bsp.index_map:
            assert t in (0, 2)
            assert n >= 0
            assert k >= 1

    @pytest.mark.parametrize(
        'L, expected_m, expected_out',
        [(8, 34, 19), (16, 144, 76), (28, 462, 238)],
    )
    def test_coefficient_count_nyquist(self, L: int, expected_m: int, expected_out: int):
        """Default truncation keeps every (n, k) with Bessel root lambda_{n,k} <= pi * L / 2."""
        bsp = SO2onDisk(L=L)
        assert bsp._m == expected_m
        assert bsp.output_size == expected_out
        assert bsp._bessel_roots.max().item() <= math.pi * L / 2

    def test_explicit_bandlimit(self):
        bsp = SO2onDisk(L=8, bandlimit=10.0)
        assert bsp.output_size > 0
        assert bsp.output_size < SO2onDisk(L=8).output_size

    def test_full_not_implemented(self):
        bsp = SO2onDisk(L=8, selective=False)
        with pytest.raises(NotImplementedError):
            bsp(torch.randn(1, 8, 8, dtype=torch.float64))


class TestSO2onDiskForward:
    @pytest.mark.parametrize('L', [8, 16])
    def test_output_shape(self, L: int):
        bsp = SO2onDisk(L=L)
        f = torch.randn(3, L, L, dtype=torch.float64)
        out = bsp(f)
        assert out.shape == (3, bsp.output_size)

    def test_output_dtype(self):
        bsp = SO2onDisk(L=8)
        f = torch.randn(2, 8, 8, dtype=torch.float64)
        out = bsp(f)
        assert out.dtype == torch.complex128

    def test_deterministic(self):
        bsp = SO2onDisk(L=8)
        f = torch.randn(2, 8, 8, dtype=torch.float64)
        torch.testing.assert_close(bsp(f), bsp(f))

    def test_different_signals_differ(self):
        bsp = SO2onDisk(L=8)
        f1 = torch.randn(1, 8, 8, dtype=torch.float64)
        f2 = torch.randn(1, 8, 8, dtype=torch.float64)
        beta1 = bsp(f1)
        beta2 = bsp(f2)
        assert not torch.allclose(beta1, beta2)

    def test_batch_size_one(self):
        bsp = SO2onDisk(L=8)
        f = torch.randn(1, 8, 8, dtype=torch.float64)
        out = bsp(f)
        assert out.shape == (1, bsp.output_size)

    def test_analytical_rotation_invariance(self):
        """Verify bispectrum invariance by rotating DH coefficients directly.

        This tests the mathematical property without grid discretization effects:
        if a'_{n,k} = e^{inφ} a_{n,k}, then b'^f = b^f.
        """
        bsp = SO2onDisk(L=16)
        torch.manual_seed(42)
        a = torch.randn(2, bsp._m_nonneg, dtype=torch.complex128) * 0.5
        a = a.clone()
        for j, (n, _k) in enumerate(bsp._nonneg_indices):
            if n == 0:
                a[:, j] = a[:, j].real + 0j

        phi = 1.23

        # Compute bispectrum of original
        a_01 = a[:, bsp._nonneg_to_idx[(0, 1)]]
        a_0k = a[:, bsp._type0_a0k_idx]
        type0 = (a_01 * a_01).unsqueeze(-1) * a_0k.conj()
        a_11 = a[:, bsp._nonneg_to_idx[(1, 1)]]
        a_n1 = a[:, bsp._type2_an1_idx]
        a_np1k = a[:, bsp._type2_anp1k_idx]
        type2 = a_11.unsqueeze(-1) * a_n1 * a_np1k.conj()
        beta_orig = torch.cat([type0, type2], dim=-1)

        # Rotate coefficients
        a_rot = a.clone()
        for j, (n, _k) in enumerate(bsp._nonneg_indices):
            phase = torch.tensor(1j * n * phi, dtype=torch.complex128)
            a_rot[:, j] = torch.exp(phase) * a[:, j]

        a_01_r = a_rot[:, bsp._nonneg_to_idx[(0, 1)]]
        a_0k_r = a_rot[:, bsp._type0_a0k_idx]
        type0_r = (a_01_r * a_01_r).unsqueeze(-1) * a_0k_r.conj()
        a_11_r = a_rot[:, bsp._nonneg_to_idx[(1, 1)]]
        a_n1_r = a_rot[:, bsp._type2_an1_idx]
        a_np1k_r = a_rot[:, bsp._type2_anp1k_idx]
        type2_r = a_11_r.unsqueeze(-1) * a_n1_r * a_np1k_r.conj()
        beta_rot = torch.cat([type0_r, type2_r], dim=-1)

        torch.testing.assert_close(beta_orig, beta_rot, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize('L', [16, 28])
    def test_rot90_invariance(self, L: int):
        """A 90 degree rotation maps the pixel grid onto itself, so invariance must be exact."""
        bsp = SO2onDisk(L=L)
        torch.manual_seed(123)
        f = torch.randn(2, L, L, dtype=torch.float64)
        beta = bsp(f)
        for k in (1, 2, 3):
            beta_rot = bsp(torch.rot90(f, k, dims=(1, 2)))
            rel_err = (beta - beta_rot).abs().max().item() / beta.abs().max().item()
            assert rel_err < 1e-12, f'L={L}, k={k}: rot90 invariance error {rel_err:.2e}'

    @pytest.mark.parametrize('L', [16, 28])
    def test_spatial_rotation_invariance(self, L: int):
        """Invariance to an arbitrary-angle image rotation of a smooth, band-limited image.

        This is bounded by the bicubic interpolation error of ``grid_sample`` (measured ~2e-2 for
        content below a quarter of the Nyquist rate), not by the transform; white-noise images are
        not meaningfully rotatable by interpolation at all.
        """
        bsp = SO2onDisk(L=L)
        torch.manual_seed(7)
        a = torch.randn(1, bsp._m_nonneg, dtype=torch.complex128)
        for j, (n, _k) in enumerate(bsp._nonneg_indices):
            if n == 0:
                a[:, j] = a[:, j].real + 0j
        a[:, bsp._bessel_roots > math.pi * L / 8] = 0
        f = bsp._idht(a).reshape(1, L, L)
        beta = bsp(f)

        angle = 0.5
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        theta_mat = torch.tensor(
            [[[cos_a, sin_a, 0.0], [-sin_a, cos_a, 0.0]]], dtype=torch.float64
        )
        grid = F.affine_grid(theta_mat, [1, 1, L, L], align_corners=False)
        f_rot = F.grid_sample(
            f.unsqueeze(1), grid, mode='bicubic', align_corners=False, padding_mode='zeros'
        ).squeeze(1)
        beta_rot = bsp(f_rot)

        rel_err = (beta - beta_rot).abs().max().item() / beta.abs().max().item()
        assert rel_err < 0.1, f'Rotation invariance error too large: {rel_err:.4f}'


class TestSO2onDiskDHTRoundtrip:
    @pytest.mark.parametrize('L', [8, 16, 32])
    def test_dht_roundtrip_bandlimited(self, L: int):
        """DHT roundtrip for a signal synthesized from DH coefficients is exact.

        Every retained basis function is resolvable on the grid (the default truncation is the
        Nyquist rate), so analysis after synthesis must return the coefficients to machine
        precision.
        """
        bsp = SO2onDisk(L=L)
        torch.manual_seed(L)
        a = torch.randn(1, bsp._m_nonneg, dtype=torch.complex128) * 0.1
        a = a.clone()
        for j, (n, _k) in enumerate(bsp._nonneg_indices):
            if n == 0:
                a[:, j] = a[:, j].real + 0j

        f = bsp._idht(a)
        a_rec = bsp._dht(f)

        rel_err = (a - a_rec).abs().max().item() / a.abs().max().item()
        assert rel_err < 1e-12, f'DHT roundtrip error too large: {rel_err:.2e}'

    @pytest.mark.parametrize('L', [8, 16])
    def test_single_harmonic_roundtrip(self, L: int):
        """A single disk harmonic should survive the roundtrip exactly."""
        bsp = SO2onDisk(L=L)

        # Build ψ_{0,1} on the grid
        j = bsp._nonneg_to_idx[(0, 1)]
        a = torch.zeros(1, bsp._m_nonneg, dtype=torch.complex128)
        a[0, j] = 1.0

        f = bsp._idht(a)
        a_rec = bsp._dht(f)

        # The (0,1) coefficient should be ~1, others ~0
        assert a_rec[0, j].real.item() == pytest.approx(1.0, abs=1e-12)
        other_mask = torch.ones(bsp._m_nonneg, dtype=torch.bool)
        other_mask[j] = False
        assert a_rec[0, other_mask].abs().max().item() < 1e-12


class TestSO2onDiskInvert:
    @pytest.mark.parametrize('L', [16, 32])
    def test_invert_exact_coefficients(self, L: int):
        """Inversion from an exact bispectrum recovers every DH coefficient magnitude.

        Goes through the full pipeline (coefficients -> bispectrum -> invert -> image -> DHT), so
        it also exercises the conditioning of the analysis operator.
        """
        bsp = SO2onDisk(L=L)
        torch.manual_seed(42)
        a = torch.randn(1, bsp._m_nonneg, dtype=torch.complex128) + 0.5
        a = a.clone()
        for j, (n, _k) in enumerate(bsp._nonneg_indices):
            if n == 0:
                a[:, j] = a[:, j].real + 0j

        # Compute bispectrum directly from coefficients
        a_01 = a[:, bsp._nonneg_to_idx[(0, 1)]]
        a_0k = a[:, bsp._type0_a0k_idx]
        type0 = (a_01 * a_01).unsqueeze(-1) * a_0k.conj()
        a_11 = a[:, bsp._nonneg_to_idx[(1, 1)]]
        a_n1 = a[:, bsp._type2_an1_idx]
        a_np1k = a[:, bsp._type2_anp1k_idx]
        type2 = a_11.unsqueeze(-1) * a_n1 * a_np1k.conj()
        beta = torch.cat([type0, type2], dim=-1)

        f_rec = bsp.invert(beta)
        a_rec = bsp._dht(f_rec)

        # Inversion is up to a global rotation, so compare magnitudes.
        torch.testing.assert_close(a_rec.abs(), a.abs(), atol=1e-9, rtol=0)

    def test_invert_direct_bispectrum_roundtrip(self):
        """Inversion at the coefficient level is exact (no DHT involved)."""
        bsp = SO2onDisk(L=16)
        torch.manual_seed(42)
        a = torch.randn(2, bsp._m_nonneg, dtype=torch.complex128) + 0.5
        a = a.clone()
        for j, (n, _k) in enumerate(bsp._nonneg_indices):
            if n == 0:
                a[:, j] = a[:, j].real + 0j

        # Compute bispectrum from coefficients
        a_01 = a[:, bsp._nonneg_to_idx[(0, 1)]]
        a_0k = a[:, bsp._type0_a0k_idx]
        type0 = (a_01 * a_01).unsqueeze(-1) * a_0k.conj()
        a_11 = a[:, bsp._nonneg_to_idx[(1, 1)]]
        a_n1 = a[:, bsp._type2_an1_idx]
        a_np1k = a[:, bsp._type2_anp1k_idx]
        type2 = a_11.unsqueeze(-1) * a_n1 * a_np1k.conj()
        beta = torch.cat([type0, type2], dim=-1)

        # Run inversion internally (extract before idht)
        K = bsp._K
        N_m = bsp._N_m
        nti = bsp._nonneg_to_idx

        a_rec = torch.zeros_like(a)
        K_0 = K[0]
        offset = 0

        b_001 = beta[:, 0]
        a_01_r = torch.abs(b_001) ** (1.0 / 3.0) * torch.exp(1j * torch.angle(b_001))
        a_rec[:, nti[(0, 1)]] = a_01_r
        a_01_sq_r = a_01_r * a_01_r
        for k in range(2, K_0 + 1):
            a_rec[:, nti[(0, k)]] = (beta[:, offset + k - 1] / a_01_sq_r).conj()
        offset += K_0

        a_11_r = torch.sqrt(torch.abs(beta[:, offset] / a_01_r))
        a_rec[:, nti[(1, 1)]] = a_11_r
        K_1 = K.get(1, 0)
        for k in range(2, K_1 + 1):
            a_rec[:, nti[(1, k)]] = (beta[:, offset + k - 1] / (a_11_r * a_01_r)).conj()
        offset += K_1

        for n_val in range(1, N_m):
            K_np1 = K.get(n_val + 1, 0)
            a_n1_r = a_rec[:, nti[(n_val, 1)]]
            for k in range(1, K_np1 + 1):
                a_rec[:, nti[(n_val + 1, k)]] = (
                    beta[:, offset + k - 1] / (a_11_r * a_n1_r)
                ).conj()
            offset += K_np1

        # Compare magnitudes (inversion is up to global rotation)
        mag_err = (a.abs() - a_rec.abs()).abs().max().item()
        assert mag_err < 1e-12, f'Direct inversion magnitude error: {mag_err}'

    def test_invert_output_shape(self):
        bsp = SO2onDisk(L=8)
        beta = bsp(torch.randn(3, 8, 8, dtype=torch.float64))
        f_rec = bsp.invert(beta)
        assert f_rec.shape == (3, 8, 8)

    def test_invert_output_dtype_is_real(self):
        bsp = SO2onDisk(L=8)
        f = torch.randn(2, 8, 8, dtype=torch.float64)
        beta = bsp(f)
        f_rec = bsp.invert(beta)
        assert f_rec.dtype == torch.float64

    def test_invert_not_implemented_full(self):
        bsp = SO2onDisk(L=8, selective=False)
        with pytest.raises(NotImplementedError):
            bsp.invert(torch.randn(2, 27, dtype=torch.complex128))

    @pytest.mark.parametrize('L', [8, 16, 32])
    def test_invert_bispectrum_roundtrip(self, L: int):
        """Bsp(invert(bsp(f))) == bsp(f) to machine precision, even for white-noise images."""
        bsp = SO2onDisk(L=L)
        torch.manual_seed(L)
        f = torch.randn(2, L, L, dtype=torch.float64)
        beta = bsp(f)
        f_rec = bsp.invert(beta)
        beta_rec = bsp(f_rec)

        rel_err = (beta - beta_rec).abs().max().item() / beta.abs().max().item()
        assert rel_err < 1e-10, f'L={L}: bispectrum roundtrip rel err {rel_err:.2e}'
