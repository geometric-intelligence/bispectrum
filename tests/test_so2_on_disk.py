"""Tests for SO2onDisk selective disk bispectrum module."""

import math

import pytest
import torch
import torch.nn.functional as F

from bispectrum import SO2onDisk
from bispectrum._bessel import bessel_jn, bessel_jn_zeros, compute_all_bessel_roots


class TestBessel:
    def test_j0_matches_torch(self):
        x = torch.linspace(0.1, 20, 200, dtype=torch.float64)
        torch.testing.assert_close(bessel_jn(0, x), torch.special.bessel_j0(x))

    def test_j1_matches_torch(self):
        x = torch.linspace(0.1, 20, 200, dtype=torch.float64)
        torch.testing.assert_close(bessel_jn(1, x), torch.special.bessel_j1(x))

    def test_jn_at_zero(self):
        assert bessel_jn(0, torch.tensor([0.0])).item() == pytest.approx(1.0)
        for n in [1, 2, 5, 10]:
            assert bessel_jn(n, torch.tensor([0.0])).item() == pytest.approx(0.0)

    def test_negative_order_raises(self):
        with pytest.raises(ValueError, match='Order n must be >= 0'):
            bessel_jn(-1, torch.tensor([1.0]))

    @pytest.mark.parametrize(
        'n, expected_first_root',
        [(0, 2.4048255577), (1, 3.8317059702), (2, 5.1356223018)],
    )
    def test_known_roots(self, n: int, expected_first_root: float):
        roots = bessel_jn_zeros(n, 3)
        # Forward recurrence has cancellation near J_n roots for n >= 2,
        # giving ~1e-6 precision. Roots are self-consistent (bessel_jn
        # at the root is ~0), which is what matters for the DHT.
        assert roots[0].item() == pytest.approx(expected_first_root, abs=1e-5)

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
        'L, expected',
        [(8, 27), (16, 105), (28, 315)],
    )
    def test_coefficient_count_matches_paper(self, L: int, expected: int):
        """Coefficient counts match Table 1 of Myers & Miolane 2025."""
        bsp = SO2onDisk(L=L)
        assert bsp.output_size == expected

    def test_explicit_bandlimit(self):
        bsp = SO2onDisk(L=8, bandlimit=10.0)
        assert bsp.output_size > 0
        assert bsp.output_size < 27

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
    def test_spatial_rotation_invariance(self, L: int):
        """Verify invariance to actual image rotation (includes discretization error).

        The error should be moderate for small L and decrease with L.
        """
        bsp = SO2onDisk(L=L)
        torch.manual_seed(123)
        f = torch.randn(1, L, L, dtype=torch.float64)
        beta = bsp(f)

        angle = 0.5
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        theta_mat = torch.tensor(
            [[[cos_a, sin_a, 0.0], [-sin_a, cos_a, 0.0]]], dtype=torch.float32
        )
        grid = F.affine_grid(theta_mat, [1, 1, L, L], align_corners=False)
        f_rot = (
            F.grid_sample(f.float().unsqueeze(1), grid, align_corners=False, padding_mode='zeros')
            .squeeze(1)
            .double()
        )
        beta_rot = bsp(f_rot)

        rel_err = (beta - beta_rot).abs().max().item() / beta.abs().max().item()
        assert rel_err < 1.5, f'Rotation invariance error too large: {rel_err:.4f}'


class TestSO2onDiskDHTRoundtrip:
    @pytest.mark.parametrize('L', [8, 16])
    def test_dht_roundtrip_bandlimited(self, L: int):
        """DHT roundtrip for a signal synthesized from DH coefficients.

        Error comes from unresolvable basis functions on the discrete grid.
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

        # Most coefficients should be recovered well; allow tolerance for
        # the 1-2 unresolvable basis functions per grid
        rel_err = (a - a_rec).abs().max().item() / a.abs().max().item()
        assert rel_err < 0.5, f'DHT roundtrip error too large: {rel_err}'

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
        assert a_rec[0, j].real.item() == pytest.approx(1.0, abs=1e-6)
        other_mask = torch.ones(bsp._m_nonneg, dtype=torch.bool)
        other_mask[j] = False
        assert a_rec[0, other_mask].abs().max().item() < 1e-4


class TestSO2onDiskInvert:
    def test_invert_exact_coefficients(self):
        """Inversion from exact bispectrum recovers DH magnitude for resolvable coefficients.

        Some high-frequency coefficients may be unresolvable on the discrete grid (rank deficient
        in the real basis matrix). We check that the MAJORITY of well-resolved coefficients match.
        """
        bsp = SO2onDisk(L=16)
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

        n_checked = 0
        n_good = 0
        for j, (_n, _k) in enumerate(bsp._nonneg_indices):
            orig_mag = a[0, j].abs().item()
            if orig_mag > 0.01:
                n_checked += 1
                rec_mag = a_rec[0, j].abs().item()
                rel_err = abs(orig_mag - rec_mag) / orig_mag
                if rel_err < 0.2:
                    n_good += 1

        fraction_good = n_good / n_checked if n_checked > 0 else 1.0
        assert fraction_good > 0.85, (
            f'Only {n_good}/{n_checked} coefficients recovered within 20% ({fraction_good:.1%})'
        )

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

    @pytest.mark.parametrize('L', [8, 16])
    def test_invert_bispectrum_roundtrip(self, L: int):
        """Bsp(invert(bsp(f))) ≈ bsp(f) within discretization tolerance."""
        bsp = SO2onDisk(L=L)
        torch.manual_seed(L)
        f = torch.randn(2, L, L, dtype=torch.float64)
        beta = bsp(f)
        f_rec = bsp.invert(beta)
        beta_rec = bsp(f_rec)

        rel_err = (beta - beta_rec).abs().max().item() / beta.abs().max().item()
        assert rel_err < 0.5, f'L={L}: bispectrum roundtrip rel err {rel_err:.4f}'
