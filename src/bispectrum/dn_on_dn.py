"""Dihedral group bispectrum on D_n.

Implements the G-Bispectrum for the dihedral group D_n acting on itself.

D_n = <a, x | a^n = x^2 = e, xax = a^{-1}> has |D_n| = 2n elements.
Signal encoding: f = [f(e), f(a), ..., f(a^{n-1}), f(x), f(ax), ..., f(a^{n-1}x)]

Irreps (Eq. 5 and Appendix A of Mataigne et al. 2024):
  2D: rho_k(a^l x^m) = R(2*pi*k*l/n) @ diag(1,-1)^m,  k = 1..floor((n-1)/2)
  1D: rho_0 (trivial), rho_01, and for even n: rho_02, rho_03

CG matrices computed analytically via eigendecomposition of the Kronecker
product representation evaluated at the generators (Appendix C).

All D_n irreps are real-valued, so the bispectral coefficients are real.

Reference: Mataigne et al. (2024) Algorithm 3, Theorem 3.1.
"""

import math
from typing import NamedTuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class _CGBlock(NamedTuple):
    """One irreducible block in a CG decomposition of rho_i x rho_j."""

    block_type: str  # '1d' or '2d'
    label: str | int  # 'rho0'/'rho01'/'rho02'/'rho03' for 1D; int k for 2D
    rows: tuple[int, ...]  # row/col indices in the 4x4 block-diagonal form


def _givens(theta: float) -> torch.Tensor:
    """2x2 rotation matrix R(theta) in float64."""
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)


_REFL_2x2 = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)


def _identify_1d_irrep(val_a: float, val_x: float) -> str:
    """Identify a 1D irrep from its values on generators a and x.

    rho_0:  (+1, +1)    rho_01: (+1, -1)
    rho_02: (-1, +1)    rho_03: (-1, -1)
    """
    if val_a > 0:
        return 'rho0' if val_x > 0 else 'rho01'
    return 'rho02' if val_x > 0 else 'rho03'


def _compute_cg(i: int, j: int, n: int) -> tuple[torch.Tensor, list[_CGBlock]]:
    """CG matrix for rho_i x rho_j on D_n via eigendecomposition.

    Pure PyTorch — no scipy.  Uses kron(rho_i(a), rho_j(a)) for the primary decomposition and
    kron(rho_i(x), rho_j(x)) to resolve degenerate 1D eigenspaces (Appendix C of the paper).

    Returns (C, blocks) where C is 4x4 orthogonal (float64) and blocks describes which irrep lives
    at which row/col indices of the block-diagonal form.
    """
    n2d = (n - 1) // 2

    Q_a = torch.kron(_givens(2 * math.pi * i / n), _givens(2 * math.pi * j / n))
    Q_x = torch.kron(_REFL_2x2, _REFL_2x2)  # diag(1, -1, -1, 1)

    eigvals_c, eigvecs_c = torch.linalg.eig(Q_a)

    eps = 1e-9
    used = [False] * 4
    conjugate_pairs: list[tuple[int, int]] = []
    real_indices: list[int] = []

    for idx in range(4):
        if used[idx]:
            continue
        if abs(eigvals_c[idx].imag.item()) < eps:
            real_indices.append(idx)
            used[idx] = True
        else:
            partner = -1
            for k in range(idx + 1, 4):
                if not used[k] and (eigvals_c[k] - eigvals_c[idx].conj()).abs().item() < eps * 100:
                    partner = k
                    break
            if partner == -1:
                real_indices.append(idx)
                used[idx] = True
                continue
            conjugate_pairs.append((idx, partner))
            used[idx] = True
            used[partner] = True

    V_cols: list[torch.Tensor] = []
    blocks: list[_CGBlock] = []
    col_idx = 0

    # --- conjugate pairs -> 2x2 rotation blocks ---
    for idx_a, _idx_b in conjugate_pairs:
        v = eigvecs_c[:, idx_a]
        u1 = v.real.clone()
        u2 = v.imag.clone()

        u1 = u1 / u1.norm()
        u2 = u2 - (u2 @ u1) * u1
        u2 = u2 / u2.norm()

        theta = math.atan2(eigvals_c[idx_a].imag.item(), eigvals_c[idx_a].real.item())
        # The eigenvector basis gives R(-θ) in the Schur block.
        # We want R(+|θ|) so the block matches the standard irrep
        # ρ_k(a) = R(2πk/n).  Negate u2 when θ > 0 to flip R(-θ)→R(θ);
        # when θ < 0, R(-θ)=R(|θ|) already has the correct sign.
        if theta < 0:
            theta = -theta
        else:
            u2 = -u2

        # Fix the basis so C^T Q_x C = diag(1,-1) in this 2D block.
        # The subspace representation of x is a reflection matrix
        # X_sub = [[cos2α, sin2α],[sin2α,-cos2α]]; rotate by α to
        # align it with [[1,0],[0,-1]].
        V_sub = torch.stack([u1, u2], dim=1)  # (4, 2)
        X_sub = V_sub.T @ Q_x @ V_sub  # (2, 2)
        alpha = math.atan2(X_sub[0, 1].item(), X_sub[0, 0].item()) / 2.0
        ca, sa = math.cos(alpha), math.sin(alpha)
        R_alpha = torch.tensor([[ca, -sa], [sa, ca]], dtype=torch.float64)
        V_sub = V_sub @ R_alpha
        u1, u2 = V_sub[:, 0], V_sub[:, 1]

        k_label = round(theta * n / (2 * math.pi))
        if k_label < 0:
            k_label += n
        if k_label > n2d:
            k_label = n - k_label
        k_label = max(k_label, 1)

        V_cols.extend([u1, u2])
        blocks.append(_CGBlock('2d', k_label, (col_idx, col_idx + 1)))
        col_idx += 2

    # --- real eigenvalues -> 1x1 blocks (with degeneracy refinement) ---
    if real_indices:
        real_groups: dict[int, list[int]] = {}
        for idx in real_indices:
            val = 1 if eigvals_c[idx].real.item() > 0 else -1
            real_groups.setdefault(val, []).append(idx)

        for val in sorted(real_groups, reverse=True):
            indices = real_groups[val]
            need = len(indices)

            # Gather candidate real vectors (both Re and Im of each eigvec)
            candidates: list[torch.Tensor] = []
            for idx in indices:
                v = eigvecs_c[:, idx]
                candidates.append(v.real.clone())
                if v.imag.norm() > eps:
                    candidates.append(v.imag.clone())

            # Orthogonalise candidates against existing V_cols and each other
            vecs: list[torch.Tensor] = []
            for cand in candidates:
                v = cand.clone()
                for prev in V_cols:
                    v = v - (v @ prev) * prev
                for prev in vecs:
                    v = v - (v @ prev) * prev
                if v.norm() > eps:
                    v = v / v.norm()
                    vecs.append(v)
                if len(vecs) == need:
                    break

            if need == 1:
                v_real = vecs[0]
                val_x = (v_real @ Q_x @ v_real).item()
                label = _identify_1d_irrep(float(val), val_x)
                V_cols.append(v_real)
                blocks.append(_CGBlock('1d', label, (col_idx,)))
                col_idx += 1
            else:
                V_sub = torch.stack(vecs, dim=1)
                Q_x_sub = V_sub.T @ Q_x @ V_sub
                evals_x, evecs_x = torch.linalg.eigh(Q_x_sub)
                V_sub = V_sub @ evecs_x

                for col in range(V_sub.shape[1]):
                    v_col = V_sub[:, col]
                    val_x = evals_x[col].item()
                    label = _identify_1d_irrep(float(val), val_x)
                    V_cols.append(v_col)
                    blocks.append(_CGBlock('1d', label, (col_idx,)))
                    col_idx += 1

    V = torch.stack(V_cols, dim=1)

    # ensure strict orthogonality
    VtV = V.T @ V
    if not torch.allclose(VtV, torch.eye(4, dtype=torch.float64), atol=1e-8):
        V, _ = torch.linalg.qr(V)

    return V, blocks


def _batched_kron_2x2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Batched Kronecker product of 2x2 matrices -> (batch, 4, 4)."""
    return torch.einsum('bij,bkl->bikjl', A, B).reshape(A.shape[0], 4, 4)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class DnonDn(nn.Module):
    """Bispectrum of D_n acting on D_n.

    Signal f: D_n -> R has length 2n.
    Encoding: f = [f(e), f(a), ..., f(a^{n-1}), f(x), f(ax), ..., f(a^{n-1}x)]

    The bispectrum is real-valued (all D_n irreps are real).

    Args:
        n: Polygon order (|D_n| = 2n).  Must be > 2.
        selective: If True (default), compute the selective bispectrum
            (Algorithm 3).  Full bispectrum raises NotImplementedError.
    """

    def __init__(self, n: int, selective: bool = True) -> None:
        super().__init__()
        if n <= 2:
            raise ValueError(f'n must be > 2 (D_1, D_2 are commutative, use CnonCn). Got n={n}.')

        self.n = n
        self.selective = selective

        n2d = (n - 1) // 2
        self._n2d = n2d

        # Fix for small odd n (g-invariance bug): n3 = max(n2-1, 1) for odd
        n3 = n2d if n % 2 == 0 else max(n2d - 1, 1)
        self._n3 = n3

        # --- CG matrices for beta_{rho1, rho_k}, k = 1..n3 ----------------
        cg_buf = torch.zeros(max(n3, 1), 4, 4, dtype=torch.float64)
        self._decompositions: list[list[_CGBlock]] = []

        for k in range(1, n3 + 1):
            C, decomp = _compute_cg(1, k, n)
            cg_buf[k - 1] = C
            self._decompositions.append(decomp)

        self.register_buffer('_cg_matrices', cg_buf)

        # --- DFT rotation tensors ------------------------------------------
        if n2d > 0:
            i_range = torch.arange(1, n2d + 1, dtype=torch.float64)
            j_range = torch.arange(n, dtype=torch.float64)
            omega = 2 * math.pi * i_range[:, None] * j_range[None, :] / n

            rho_rot = torch.zeros(n2d, 2, 2, n, dtype=torch.float64)
            rho_rot[:, 0, 0] = torch.cos(omega)
            rho_rot[:, 0, 1] = -torch.sin(omega)
            rho_rot[:, 1, 0] = torch.sin(omega)
            rho_rot[:, 1, 1] = torch.cos(omega)

            rho_ref = rho_rot.clone()
            rho_ref[:, :, 1, :] *= -1
        else:
            rho_rot = torch.zeros(0, 2, 2, n, dtype=torch.float64)
            rho_ref = rho_rot.clone()

        self.register_buffer('_rho_rot', rho_rot)
        self.register_buffer('_rho_ref', rho_ref)

        # --- index map ------------------------------------------------------
        idx_map: list[tuple[int, ...]] = [(0, 0)]
        for r in range(2):
            for c in range(2):
                idx_map.append((0, 1, r, c))
        for k in range(1, n3 + 1):
            for r in range(4):
                for c in range(4):
                    idx_map.append((1, k, r, c))
        self._index_map: list[tuple[int, ...]] = idx_map

    # -----------------------------------------------------------------------
    # Group DFT / inverse DFT
    # -----------------------------------------------------------------------

    def _group_dft(self, f: torch.Tensor) -> torch.Tensor:
        """Forward DFT on D_n.

        (batch, 2n) -> (batch, 2, 2, n2d+1).
        """
        n = self.n
        n2d = self._n2d
        batch = f.shape[0]
        device = f.device
        dtype = f.dtype

        f_rot = f[:, :n]
        f_ref = f[:, n:]

        fhat = torch.zeros(batch, 2, 2, n2d + 1, device=device, dtype=dtype)

        # 1D irreps
        fhat[:, 0, 0, 0] = f.sum(dim=-1)
        fhat[:, 1, 0, 0] = f_rot.sum(dim=-1) - f_ref.sum(dim=-1)
        if n % 2 == 0:
            signs = torch.pow(-1.0, torch.arange(n, device=device, dtype=dtype))
            f_rot_s = (f_rot * signs).sum(dim=-1)
            f_ref_s = (f_ref * signs).sum(dim=-1)
            fhat[:, 0, 1, 0] = f_rot_s + f_ref_s
            fhat[:, 1, 1, 0] = f_rot_s - f_ref_s

        # 2D irreps
        if n2d > 0:
            rho_r = self._rho_rot.to(dtype)  # (n2d, 2, 2, n)
            rho_x = self._rho_ref.to(dtype)
            fhat_2d = torch.einsum('bl,krcl->brck', f_rot, rho_r) + torch.einsum(
                'bl,krcl->brck', f_ref, rho_x
            )
            fhat[:, :, :, 1:] = fhat_2d

        return fhat

    def _inverse_dft(self, fhat: torch.Tensor) -> torch.Tensor:
        """Inverse DFT on D_n.

        (batch, 2, 2, n2d+1) -> (batch, 2n).
        """
        n = self.n
        n2d = self._n2d
        batch = fhat.shape[0]
        device = fhat.device
        dtype = fhat.dtype
        inv_2n = 1.0 / (2.0 * n)

        f_rot = torch.zeros(batch, n, device=device, dtype=dtype)
        f_ref = torch.zeros(batch, n, device=device, dtype=dtype)

        F0 = fhat[:, 0, 0, 0]
        F01 = fhat[:, 1, 0, 0]
        f_rot += (F0 + F01)[:, None] * inv_2n
        f_ref += (F0 - F01)[:, None] * inv_2n

        if n % 2 == 0:
            F02 = fhat[:, 0, 1, 0]
            F03 = fhat[:, 1, 1, 0]
            signs = torch.pow(-1.0, torch.arange(n, device=device, dtype=dtype))
            f_rot += ((F02 + F03)[:, None] * signs) * inv_2n
            f_ref += ((F02 - F03)[:, None] * signs) * inv_2n

        if n2d > 0:
            l_range = torch.arange(n, device=device, dtype=dtype)
            inv_n = 1.0 / n
            for k_idx in range(n2d):
                theta = 2 * math.pi * (k_idx + 1) * l_range / n
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
                Fk = fhat[:, :, :, k_idx + 1]
                F00 = Fk[:, 0, 0]
                F01_ = Fk[:, 0, 1]
                F10 = Fk[:, 1, 0]
                F11 = Fk[:, 1, 1]
                f_rot += (cos_t * (F00 + F11)[:, None] + sin_t * (F10 - F01_)[:, None]) * inv_n
                f_ref += (cos_t * (F00 - F11)[:, None] + sin_t * (F01_ + F10)[:, None]) * inv_n

        return torch.cat([f_rot, f_ref], dim=-1)

    # -----------------------------------------------------------------------
    # Fplus construction
    # -----------------------------------------------------------------------

    def _build_fplus(self, fhat: torch.Tensor, cg_idx: int) -> torch.Tensor:
        """Block-diagonal Fplus = oplus F(rho) (untransposed).

        The bispectrum formula uses Fplus^T = oplus F(rho)^T.
        """
        batch = fhat.shape[0]
        Fplus = torch.zeros(batch, 4, 4, device=fhat.device, dtype=fhat.dtype)
        decomp = self._decompositions[cg_idx]

        for block in decomp:
            if block.block_type == '2d':
                r0, r1 = block.rows
                k = block.label
                assert isinstance(k, int)
                Fplus[:, r0 : r1 + 1, r0 : r1 + 1] = fhat[:, :, :, k]
            else:
                r = block.rows[0]
                lbl = block.label
                assert isinstance(lbl, str)
                if lbl == 'rho0':
                    Fplus[:, r, r] = fhat[:, 0, 0, 0]
                elif lbl == 'rho01':
                    Fplus[:, r, r] = fhat[:, 1, 0, 0]
                elif lbl == 'rho02':
                    Fplus[:, r, r] = fhat[:, 0, 1, 0]
                elif lbl == 'rho03':
                    Fplus[:, r, r] = fhat[:, 1, 1, 0]
        return Fplus

    # -----------------------------------------------------------------------
    # forward / invert
    # -----------------------------------------------------------------------

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Compute the D_n selective bispectrum.

        Args:
            f: Real signal on D_n.  Shape ``(batch, 2n)``.

        Returns:
            Real bispectrum.  Shape ``(batch, output_size)``, dtype same as *f*.
        """
        if not self.selective:
            raise NotImplementedError('Full bispectrum not yet implemented for DnonDn.')

        n3 = self._n3
        batch = f.shape[0]
        dtype = f.dtype

        fhat = self._group_dft(f)

        parts: list[torch.Tensor] = []

        # beta_{rho0, rho0} = F(rho0)^3
        F_rho0 = fhat[:, 0, 0, 0]
        parts.append((F_rho0**3).unsqueeze(-1))

        # beta_{rho0, rho1} = F(rho0) * F(rho1)^T @ F(rho1)   (2x2)
        F_rho1 = fhat[:, :, :, 1]
        beta_01 = F_rho0[:, None, None] * torch.bmm(F_rho1.transpose(-1, -2), F_rho1)
        parts.append(beta_01.reshape(batch, 4))

        # beta_{rho1, rho_k}  for k = 1..n3   (4x4 each)
        for m in range(n3):
            k = m + 1
            F_k = fhat[:, :, :, k]
            fh_kron = _batched_kron_2x2(F_rho1, F_k)

            Fplus = self._build_fplus(fhat, m)
            C = self._cg_matrices[m].to(dtype)

            # beta = C (oplus F^T) C^T (F1 x Fk)
            beta_1k = C @ Fplus.transpose(-1, -2) @ C.T @ fh_kron
            parts.append(beta_1k.reshape(batch, 16))

        return torch.cat(parts, dim=-1)

    def invert(self, beta: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Recover a signal from its selective bispectrum (Algorithm 3).

        Reconstruction has O(2) indeterminacy (continuous rotations and
        reflections), so the recovered signal matches the original up to
        a D_n group action.

        Args:
            beta: Selective bispectrum, shape ``(batch, output_size)``.

        Returns:
            Reconstructed real signal, shape ``(batch, 2n)``.
        """
        if not self.selective:
            raise NotImplementedError('Inversion only implemented for selective bispectrum.')

        n2d = self._n2d
        n3 = self._n3
        batch = beta.shape[0]
        device = beta.device
        dtype = beta.dtype

        fhat = torch.zeros(batch, 2, 2, n2d + 1, device=device, dtype=dtype)

        # Step 1 — F(rho_0) from beta_{rho0,rho0} = F(rho0)^3
        b00 = beta[:, 0]
        fhat[:, 0, 0, 0] = torch.sign(b00) * torch.abs(b00) ** (1.0 / 3.0)

        # Step 2 — F(rho_1) via eigendecomposition of beta_{rho0,rho1}/F(rho0)
        b01 = beta[:, 1:5].reshape(batch, 2, 2)
        F0 = fhat[:, 0, 0, 0]
        M = b01 / F0[:, None, None]  # = F1^T @ F1,  positive semi-definite
        eigvals_M, eigvecs_M = torch.linalg.eigh(M)
        eigvals_M = torch.clamp(eigvals_M, min=0.0)
        S = eigvecs_M @ torch.diag_embed(torch.sqrt(eigvals_M)) @ eigvecs_M.transpose(-1, -2)

        fhat[:, :, :, 1] = S

        # Step 3 — sequential recovery from beta_{rho1, rho_k}.
        # The extraction gives Fourier coefficients that are "twisted"
        # by the O(2) ambiguity in F_1.  Fourier coefficient magnitudes
        # (Frobenius norms) are always exact; the bispectrum of the
        # recovered signal matches the original only up to O(2).
        offset = 5
        for m in range(n3):
            k_prev = m + 1
            b_1k = beta[:, offset : offset + 16].reshape(batch, 4, 4)
            offset += 16

            C = self._cg_matrices[m].to(dtype)
            F1 = fhat[:, :, :, 1]
            Fkp = fhat[:, :, :, k_prev]

            A = _batched_kron_2x2(F1, Fkp)
            A_inv = torch.linalg.inv(A)

            # beta = C (oplus F^T) C^T A  =>  oplus F = [C^T beta A^{-1} C]^T
            temp = torch.matmul(b_1k, A_inv)
            temp = torch.matmul(C.T, temp)
            temp = torch.matmul(temp, C)
            block_diag = temp.transpose(-1, -2)

            decomp = self._decompositions[m]
            for block in decomp:
                if block.block_type == '2d':
                    r0, r1 = block.rows
                    k_label = block.label
                    assert isinstance(k_label, int)
                    fhat[:, :, :, k_label] = block_diag[:, r0 : r1 + 1, r0 : r1 + 1]
                else:
                    r = block.rows[0]
                    lbl = block.label
                    assert isinstance(lbl, str)
                    val = block_diag[:, r, r]
                    if lbl == 'rho0':
                        fhat[:, 0, 0, 0] = val
                    elif lbl == 'rho01':
                        fhat[:, 1, 0, 0] = val
                    elif lbl == 'rho02':
                        fhat[:, 0, 1, 0] = val
                    elif lbl == 'rho03':
                        fhat[:, 1, 1, 0] = val

        return self._inverse_dft(fhat)

    # -----------------------------------------------------------------------
    # properties
    # -----------------------------------------------------------------------

    @property
    def output_size(self) -> int:
        """Number of scalar bispectral values in the output."""
        return len(self._index_map)

    @property
    def index_map(self) -> list[tuple[int, ...]]:
        """Maps flat output index -> (irrep pair, matrix entry) tuple."""
        return list(self._index_map)

    def extra_repr(self) -> str:
        return f'n={self.n}, selective={self.selective}, output_size={self.output_size}'
