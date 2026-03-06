"""Tests for OonR3 (octahedral group) bispectrum module."""

import pytest
import torch

from bispectrum.o_on_r3 import _CAYLEY, _INVERSE, _KRON_TABLE, OonR3, _ELEMENTS_3x3

ATOL = 1e-4
RTOL = 1e-4


def _permute_signal(f: torch.Tensor, g: int) -> torch.Tensor:
    """Apply group action T_g: (T_g f)(h) = f(g^{-1} h)."""
    g_inv = _INVERSE[g].item()
    perm = _CAYLEY[g_inv]
    return f[:, perm]


class TestGroupData:
    def test_elements_count(self):
        assert len(_ELEMENTS_3x3) == 24

    def test_elements_orthogonal(self):
        for i, R in enumerate(_ELEMENTS_3x3):
            err = (R @ R.T - torch.eye(3)).abs().max()
            assert err < 1e-12, f'Element {i} not orthogonal: {err}'

    def test_elements_det_one(self):
        for i, R in enumerate(_ELEMENTS_3x3):
            d = torch.linalg.det(R)
            assert abs(d - 1.0) < 1e-12, f'Element {i} det={d}'

    def test_identity_exists(self):
        found = False
        for _i, R in enumerate(_ELEMENTS_3x3):
            if (R - torch.eye(3, dtype=torch.float64)).abs().max() < 1e-10:
                found = True
                break
        assert found, 'Identity element not found'

    def test_cayley_closure(self):
        for i in range(24):
            for j in range(24):
                product = _ELEMENTS_3x3[i] @ _ELEMENTS_3x3[j]
                k = _CAYLEY[i, j].item()
                err = (product - _ELEMENTS_3x3[k]).abs().max()
                assert err < 1e-12, f'{i}*{j}={k} err={err}'

    def test_cayley_identity_row_col(self):
        e = None
        for i in range(24):
            if (_ELEMENTS_3x3[i] - torch.eye(3, dtype=torch.float64)).abs().max() < 1e-10:
                e = i
                break
        assert e is not None
        for i in range(24):
            assert _CAYLEY[e, i].item() == i
            assert _CAYLEY[i, e].item() == i

    def test_inverse_map(self):
        e = None
        for i in range(24):
            if (_ELEMENTS_3x3[i] - torch.eye(3, dtype=torch.float64)).abs().max() < 1e-10:
                e = i
                break
        for i in range(24):
            inv_i = _INVERSE[i].item()
            assert _CAYLEY[i, inv_i].item() == e
            assert _CAYLEY[inv_i, i].item() == e

    def test_kronecker_table_dimensions(self):
        dims = [1, 3, 3, 2, 1]
        for i in range(5):
            for j in range(5):
                mult_str = _KRON_TABLE[i][j]
                total = sum(int(mult_str[k]) * dims[k] for k in range(5))
                assert total == dims[i] * dims[j], (
                    f'rho{i} x rho{j}: decomp dims {total} != {dims[i] * dims[j]}'
                )

    def test_kronecker_table_symmetric(self):
        for i in range(5):
            for j in range(5):
                assert _KRON_TABLE[i][j] == _KRON_TABLE[j][i]


class TestIrrepMatrices:
    def test_irrep_dims(self):
        bsp = OonR3()
        for k in range(5):
            mats = bsp._get_irrep_mats(k)
            expected_d = [1, 3, 3, 2, 1][k]
            assert mats.shape == (24, expected_d, expected_d)

    def test_irrep_identity(self):
        bsp = OonR3()
        e = None
        for i in range(24):
            if (_ELEMENTS_3x3[i] - torch.eye(3, dtype=torch.float64)).abs().max() < 1e-10:
                e = i
                break
        for k in range(5):
            mat_e = bsp._get_irrep_mats(k)[e]
            d = mat_e.shape[0]
            torch.testing.assert_close(
                mat_e,
                torch.eye(d, dtype=mat_e.dtype),
                atol=1e-12,
                rtol=0,
            )

    def test_irrep_homomorphism(self):
        """rho_k(g) rho_k(h) = rho_k(gh) for all g, h."""
        bsp = OonR3()
        for k in range(5):
            mats = bsp._get_irrep_mats(k).to(torch.float64)
            for i in range(24):
                for j in range(24):
                    product = mats[i] @ mats[j]
                    ij = _CAYLEY[i, j].item()
                    err = (product - mats[ij]).abs().max()
                    assert err < 1e-10, f'rho{k}({i})*rho{k}({j}) err={err}'

    def test_irrep_orthogonality(self):
        """Rows/columns of irrep matrices are orthogonal."""
        bsp = OonR3()
        for k in range(5):
            mats = bsp._get_irrep_mats(k).to(torch.float64)
            for i in range(24):
                err = (
                    (mats[i] @ mats[i].T - torch.eye(mats.shape[1], dtype=torch.float64))
                    .abs()
                    .max()
                )
                assert err < 1e-10, f'rho{k}({i}) not orthogonal: {err}'


class TestOonR3Construction:
    def test_instantiation(self):
        bsp = OonR3()
        assert bsp.selective is True

    def test_instantiation_full(self):
        bsp = OonR3(selective=False)
        assert bsp.selective is False

    def test_no_trainable_parameters(self):
        bsp = OonR3()
        assert isinstance(bsp, torch.nn.Module)
        assert sum(p.numel() for p in bsp.parameters()) == 0

    def test_selective_output_size(self):
        bsp = OonR3()
        assert bsp.output_size == 172

    def test_extra_repr(self):
        bsp = OonR3()
        assert 'selective=True' in repr(bsp)
        assert 'output_size=172' in repr(bsp)

    def test_index_map_length(self):
        bsp = OonR3()
        assert len(bsp.index_map) == bsp.output_size

    def test_group_order(self):
        assert OonR3.GROUP_ORDER == 24

    def test_n_irreps(self):
        assert OonR3.N_IRREPS == 5


class TestOonR3Forward:
    def test_output_shape(self):
        bsp = OonR3()
        f = torch.randn(3, 24)
        out = bsp(f)
        assert out.shape == (3, 172)

    def test_output_dtype_complex64(self):
        bsp = OonR3()
        out = bsp(torch.randn(2, 24))
        assert out.dtype == torch.complex64

    def test_output_is_real_valued(self):
        """O has only real irreps, so bispectrum values should be real."""
        bsp = OonR3()
        f = torch.randn(4, 24, dtype=torch.float64)
        out = bsp(f)
        assert out.imag.abs().max() < 1e-10

    def test_output_dtype_from_float64(self):
        bsp = OonR3()
        out = bsp(torch.randn(2, 24, dtype=torch.float64))
        assert out.dtype == torch.complex128

    def test_deterministic(self):
        bsp = OonR3()
        f = torch.randn(4, 24)
        torch.testing.assert_close(bsp(f), bsp(f))

    def test_batch_size_one(self):
        bsp = OonR3()
        f = torch.randn(1, 24)
        out = bsp(f)
        assert out.shape == (1, 172)

    @pytest.mark.parametrize('g', range(24))
    def test_invariance_under_group_action(self, g: int):
        torch.manual_seed(g + 42)
        bsp = OonR3()
        f = torch.randn(4, 24, dtype=torch.float64)
        beta = bsp(f)
        f_shifted = _permute_signal(f, g)
        beta_shifted = bsp(f_shifted)
        torch.testing.assert_close(
            beta_shifted.real,
            beta.real,
            atol=1e-10,
            rtol=1e-10,
        )

    def test_different_signals_differ(self):
        bsp = OonR3()
        f1 = torch.zeros(1, 24)
        f1[0, 0] = 1.0
        f2 = torch.zeros(1, 24)
        f2[0, 0] = 1.0
        f2[0, 1] = 1.0
        assert not torch.allclose(bsp(f1), bsp(f2))

    def test_forward_not_implemented_full(self):
        bsp = OonR3(selective=False)
        with pytest.raises(NotImplementedError):
            bsp(torch.randn(2, 24))


class TestOonR3DFT:
    def test_dft_roundtrip(self):
        bsp = OonR3()
        f = torch.randn(4, 24, dtype=torch.float64)
        fhat = bsp._group_dft(f)
        f_rec = bsp._inverse_dft(fhat)
        torch.testing.assert_close(f_rec, f, atol=1e-10, rtol=1e-10)

    def test_dft_shapes(self):
        bsp = OonR3()
        f = torch.randn(3, 24)
        fhat = bsp._group_dft(f)
        assert len(fhat) == 5
        expected_dims = [1, 3, 3, 2, 1]
        for k, d in enumerate(expected_dims):
            assert fhat[k].shape == (3, d, d)

    def test_dft_equivariance(self):
        """DFT of T_g f = rho_k(g) @ DFT(f)."""
        bsp = OonR3()
        f = torch.randn(2, 24, dtype=torch.float64)
        fhat = bsp._group_dft(f)
        for g in range(24):
            f_shifted = _permute_signal(f, g)
            fhat_shifted = bsp._group_dft(f_shifted)
            for k in range(5):
                rho_g = bsp._get_irrep_mats(k)[g].to(torch.float64)
                expected = rho_g @ fhat[k]
                torch.testing.assert_close(
                    fhat_shifted[k],
                    expected,
                    atol=1e-10,
                    rtol=1e-10,
                )

    def test_inverse_dft_of_zero(self):
        bsp = OonR3()
        fhat = [torch.zeros(1, d, d) for d in [1, 3, 3, 2, 1]]
        f = bsp._inverse_dft(fhat)
        torch.testing.assert_close(f, torch.zeros(1, 24))


class TestOonR3CG:
    def test_cg_11_orthogonal(self):
        bsp = OonR3()
        C = bsp._cg_11.to(torch.float64)
        err = (C @ C.T - torch.eye(9, dtype=torch.float64)).abs().max()
        assert err < 1e-10, f'CG_11 not orthogonal: {err}'

    def test_cg_12_orthogonal(self):
        bsp = OonR3()
        C = bsp._cg_12.to(torch.float64)
        err = (C @ C.T - torch.eye(9, dtype=torch.float64)).abs().max()
        assert err < 1e-10, f'CG_12 not orthogonal: {err}'

    def test_cg_11_block_diagonalizes(self):
        """C^T (rho1(g) ⊗ rho1(g)) C is block-diagonal for all g."""
        bsp = OonR3()
        C = bsp._cg_11.to(torch.float64)
        rho1 = bsp._get_irrep_mats(1).to(torch.float64)
        for g in range(24):
            kron = torch.kron(rho1[g], rho1[g])
            bd = C.T @ kron @ C
            for irrep_k, r0, r1 in bsp._block_info_11:
                for irrep_j, s0, s1 in bsp._block_info_11:
                    if irrep_k != irrep_j:
                        block = bd[r0:r1, s0:s1]
                        assert block.abs().max() < 1e-10, (
                            f'g={g}: cross-block ({irrep_k},{irrep_j}) max={block.abs().max()}'
                        )

    def test_cg_12_block_diagonalizes(self):
        """C^T (rho1(g) ⊗ rho2(g)) C is block-diagonal for all g."""
        bsp = OonR3()
        C = bsp._cg_12.to(torch.float64)
        rho1 = bsp._get_irrep_mats(1).to(torch.float64)
        rho2 = bsp._get_irrep_mats(2).to(torch.float64)
        for g in range(24):
            kron = torch.kron(rho1[g], rho2[g])
            bd = C.T @ kron @ C
            for irrep_k, r0, r1 in bsp._block_info_12:
                for irrep_j, s0, s1 in bsp._block_info_12:
                    if irrep_k != irrep_j:
                        block = bd[r0:r1, s0:s1]
                        assert block.abs().max() < 1e-10, (
                            f'g={g}: cross-block ({irrep_k},{irrep_j}) max={block.abs().max()}'
                        )


class TestOonR3Invert:
    def test_roundtrip_bispectrum(self):
        torch.manual_seed(42)
        bsp = OonR3()
        f = torch.randn(2, 24)
        beta = bsp(f)
        f_rec = bsp.invert(beta, n_corrections=10, n_restarts=6)
        beta_rec = bsp(f_rec)
        torch.testing.assert_close(
            beta_rec.real,
            beta.real,
            atol=1.0,
            rtol=0.1,
        )

    def test_roundtrip_signal_up_to_group_action(self):
        torch.manual_seed(42)
        bsp = OonR3()
        f = torch.randn(1, 24)
        beta = bsp(f)
        f_rec = bsp.invert(beta, n_corrections=10, n_restarts=6)

        min_err = float('inf')
        for g in range(24):
            f_shifted = _permute_signal(f, g)
            err = (f_rec - f_shifted).abs().max().item()
            min_err = min(min_err, err)
        assert min_err < 0.05, f'Signal not recovered up to group action: {min_err}'

    def test_invert_output_shape(self):
        bsp = OonR3()
        beta = bsp(torch.randn(3, 24))
        f_rec = bsp.invert(beta, n_corrections=3, n_restarts=1)
        assert f_rec.shape == (3, 24)

    def test_invert_output_is_real(self):
        bsp = OonR3()
        beta = bsp(torch.randn(2, 24))
        f_rec = bsp.invert(beta, n_corrections=3, n_restarts=1)
        assert not f_rec.is_complex()

    def test_invert_not_implemented_full(self):
        bsp = OonR3(selective=False)
        with pytest.raises(NotImplementedError):
            bsp.invert(torch.randn(2, 172))

    def test_bootstrap_init_f0_recovery(self):
        """Bootstrap should exactly recover F0."""
        torch.manual_seed(42)
        bsp = OonR3()
        f = torch.randn(1, 24, dtype=torch.float64)
        beta = bsp(f)
        f_init = bsp._bootstrap_init(beta)
        fhat_orig = bsp._group_dft(f)
        fhat_init = bsp._group_dft(f_init)
        torch.testing.assert_close(
            fhat_init[0][:, 0, 0].abs(),
            fhat_orig[0][:, 0, 0].abs(),
            atol=1e-8,
            rtol=1e-8,
        )

    def test_bootstrap_init_fourier_norms(self):
        """Bootstrap should recover ||F_k|| correctly for all k."""
        torch.manual_seed(42)
        bsp = OonR3()
        f = torch.randn(2, 24, dtype=torch.float64)
        beta = bsp(f)
        f_init = bsp._bootstrap_init(beta)
        fhat_orig = bsp._group_dft(f)
        fhat_init = bsp._group_dft(f_init)
        for k in range(5):
            norm_orig = torch.linalg.norm(fhat_orig[k], dim=(-2, -1))
            norm_init = torch.linalg.norm(fhat_init[k], dim=(-2, -1))
            torch.testing.assert_close(
                norm_orig,
                norm_init,
                atol=1e-4,
                rtol=1e-4,
            )

    def test_bootstrap_init_near_isotropic_signal(self):
        """Bootstrap should not crash on a near-zero-mean signal where S_kron is singular."""
        bsp = OonR3()
        f = torch.zeros(1, 24, dtype=torch.float64)
        f[0, 0] = 1e-14
        beta = bsp(f)
        f_init = bsp._bootstrap_init(beta)
        assert f_init.shape == (1, 24)
        assert torch.isfinite(f_init).all(), (
            'Bootstrap produced non-finite values for near-isotropic signal'
        )


class TestOonR3JacfwdCompatibility:
    def test_jacfwd_produces_valid_jacobian(self):
        """Jacfwd should trace through forward() and produce a (172, 24) Jacobian."""
        bsp = OonR3()
        f = torch.randn(24)

        def fwd(x: torch.Tensor) -> torch.Tensor:
            return bsp.forward(x.unsqueeze(0)).squeeze(0).real

        J = torch.func.jacfwd(fwd)(f)
        assert J.shape == (172, 24)
        assert torch.isfinite(J).all(), 'Jacobian contains non-finite values'

    def test_jacfwd_jacobian_nonzero(self):
        """Jacobian should be non-trivial for a generic signal."""
        torch.manual_seed(99)
        bsp = OonR3()
        f = torch.randn(24)

        def fwd(x: torch.Tensor) -> torch.Tensor:
            return bsp.forward(x.unsqueeze(0)).squeeze(0).real

        J = torch.func.jacfwd(fwd)(f)
        assert J.abs().max() > 1e-6, 'Jacobian is unexpectedly all-zero'

    def test_jacfwd_matches_finite_differences(self):
        """Forward-mode AD Jacobian should match numerical finite differences."""
        torch.manual_seed(7)
        bsp = OonR3()
        f = torch.randn(24, dtype=torch.float64)

        def fwd(x: torch.Tensor) -> torch.Tensor:
            return bsp.forward(x.unsqueeze(0)).squeeze(0).real

        J_ad = torch.func.jacfwd(fwd)(f)

        eps = 1e-6
        J_fd = torch.zeros_like(J_ad)
        f0 = fwd(f)
        for i in range(24):
            f_pert = f.clone()
            f_pert[i] += eps
            J_fd[:, i] = (fwd(f_pert) - f0) / eps

        torch.testing.assert_close(J_ad, J_fd, atol=1e-4, rtol=1e-4)


class TestOonR3LmStepRobustness:
    def test_lm_step_constant_signal(self):
        """LM step should not crash on a constant signal (rank-deficient Jacobian)."""
        bsp = OonR3()
        f = torch.ones(1, 24)
        beta = bsp(f)
        f_out = bsp._lm_step(f, beta)
        assert f_out.shape == (1, 24)
        assert torch.isfinite(f_out).all(), (
            'LM step produced non-finite values for constant signal'
        )

    def test_lm_step_zero_signal(self):
        """LM step should handle zero signal gracefully."""
        bsp = OonR3()
        f = torch.zeros(1, 24)
        beta = bsp(f)
        f_out = bsp._lm_step(f, beta)
        assert f_out.shape == (1, 24)
        assert torch.isfinite(f_out).all(), 'LM step produced non-finite values for zero signal'

    def test_lm_step_reduces_or_preserves_loss(self):
        """LM step should not increase the bispectral residual."""
        torch.manual_seed(42)
        bsp = OonR3()
        f = torch.randn(2, 24)
        f_target = torch.randn(2, 24)
        beta_target = bsp(f_target)

        loss_before = (bsp(f).real - beta_target.real).norm(dim=-1)
        f_after = bsp._lm_step(f, beta_target)
        loss_after = (bsp(f_after).real - beta_target.real).norm(dim=-1)

        assert (loss_after <= loss_before + 1e-6).all(), (
            f'LM step increased loss: {loss_before} -> {loss_after}'
        )
