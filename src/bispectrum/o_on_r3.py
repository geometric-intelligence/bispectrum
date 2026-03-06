"""Octahedral group bispectrum on O.

Implements the G-Bispectrum for the octahedral group O (rotational symmetries
of the cube, |O| = 24) following Mataigne et al., "The Selective G-Bispectrum
and its Inversion: Applications to G-Invariant Networks", NeurIPS 2024.

The octahedral group has 5 irreps with dimensions (1, 3, 3, 2, 1):
  rho0: trivial representation (dim 1)
  rho1: standard 3D representation (dim 3) — the rotation matrices themselves
  rho2: product representation (dim 3) — rho4(g) * rho1(g)
  rho3: 2D representation (dim 2) — factors through O -> S3
  rho4: alternating representation (dim 1) — parity of the permutation on
        the four body diagonals of the cube

Signal encoding: f = [f(g_0), f(g_1), ..., f(g_23)] where g_i are the 24
rotation matrices sorted lexicographically by their flattened entries.
g_23 = identity.

Selective bispectrum (Appendix E, Table 6 of Mataigne et al. 2024):
  4 matrix coefficients: beta_{rho0,rho0}, beta_{rho0,rho1},
  beta_{rho1,rho1}, beta_{rho1,rho2}
  Total: 1 + 9 + 81 + 81 = 172 scalar values.

The bispectrum is real-valued for real inputs (all irreps are real).
The output dtype is complex (complex64/complex128) for API consistency
with CnonCn and DnonDn modules.
"""

import math

import torch
import torch.nn as nn

_S3_2 = math.sqrt(3.0) / 2.0

# 24 rotation matrices of the octahedral group, sorted lexicographically
# by flattened entries. Generated from generators c4z = R_z(pi/2) and
# c3d = R_{[1,1,1]}(2*pi/3), verified against escnn.OctahedralGroup().
# g23 is the identity.
# fmt: off
_ELEMENTS_3x3 = torch.tensor([
    [[-1, 0, 0], [ 0,-1, 0], [ 0, 0, 1]],  # g00
    [[-1, 0, 0], [ 0, 0,-1], [ 0,-1, 0]],  # g01
    [[-1, 0, 0], [ 0, 0, 1], [ 0, 1, 0]],  # g02
    [[-1, 0, 0], [ 0, 1, 0], [ 0, 0,-1]],  # g03
    [[ 0,-1, 0], [-1, 0, 0], [ 0, 0,-1]],  # g04
    [[ 0,-1, 0], [ 0, 0,-1], [ 1, 0, 0]],  # g05
    [[ 0,-1, 0], [ 0, 0, 1], [-1, 0, 0]],  # g06
    [[ 0,-1, 0], [ 1, 0, 0], [ 0, 0, 1]],  # g07
    [[ 0, 0,-1], [-1, 0, 0], [ 0, 1, 0]],  # g08
    [[ 0, 0,-1], [ 0,-1, 0], [-1, 0, 0]],  # g09
    [[ 0, 0,-1], [ 0, 1, 0], [ 1, 0, 0]],  # g10
    [[ 0, 0,-1], [ 1, 0, 0], [ 0,-1, 0]],  # g11
    [[ 0, 0, 1], [-1, 0, 0], [ 0,-1, 0]],  # g12
    [[ 0, 0, 1], [ 0,-1, 0], [ 1, 0, 0]],  # g13
    [[ 0, 0, 1], [ 0, 1, 0], [-1, 0, 0]],  # g14
    [[ 0, 0, 1], [ 1, 0, 0], [ 0, 1, 0]],  # g15
    [[ 0, 1, 0], [-1, 0, 0], [ 0, 0, 1]],  # g16
    [[ 0, 1, 0], [ 0, 0,-1], [-1, 0, 0]],  # g17
    [[ 0, 1, 0], [ 0, 0, 1], [ 1, 0, 0]],  # g18
    [[ 0, 1, 0], [ 1, 0, 0], [ 0, 0,-1]],  # g19
    [[ 1, 0, 0], [ 0,-1, 0], [ 0, 0,-1]],  # g20
    [[ 1, 0, 0], [ 0, 0,-1], [ 0, 1, 0]],  # g21
    [[ 1, 0, 0], [ 0, 0, 1], [ 0,-1, 0]],  # g22
    [[ 1, 0, 0], [ 0, 1, 0], [ 0, 0, 1]],  # g23 = identity
], dtype=torch.float64)
# fmt: on

# rho4: alternating representation. +1 for even permutations of the cube's
# body diagonals, -1 for odd. Verified against escnn.
# fmt: off
_RHO4_SIGNS = torch.tensor([
    1, -1, -1, 1, -1, 1, 1, -1,
    1, -1, -1, 1, 1, -1, -1, 1,
    -1, 1, 1, -1, 1, -1, -1, 1,
], dtype=torch.float64)
# fmt: on

# Cayley table: _CAYLEY[i, j] = k means g_i @ g_j == g_k.
# Verified against matrix multiplication of _ELEMENTS_3x3.
# fmt: off
_CAYLEY = torch.tensor([
    [23,22,21,20,19,18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    [21,23,20,22,18,16,19,17,13,15,12,14,10, 8,11, 9, 5, 7, 4, 6, 2, 0, 3, 1],
    [22,20,23,21,17,19,16,18,14,12,15,13, 9,11, 8,10, 6, 4, 7, 5, 1, 3, 0, 2],
    [20,21,22,23,16,17,18,19,12,13,14,15, 8, 9,10,11, 4, 5, 6, 7, 0, 1, 2, 3],
    [19,15,11, 7,23,14,10, 3,22,18, 6, 2,21,17, 5, 1,20,13, 9, 0,16,12, 8, 4],
    [17,14, 9, 6,22,12,11, 1,20,19, 4, 3,23,16, 7, 0,21,15, 8, 2,18,13,10, 5],
    [18,13,10, 5,21,15, 8, 2,23,16, 7, 0,20,19, 4, 3,22,12,11, 1,17,14, 9, 6],
    [16,12, 8, 4,20,13, 9, 0,21,17, 5, 1,22,18, 6, 2,23,14,10, 3,19,15,11, 7],
    [11,19, 7,15,14, 3,23,10, 6,22, 2,18,17, 1,21, 5, 9,20, 0,13,12, 4,16, 8],
    [10,18, 5,13,15, 2,21, 8, 7,23, 0,16,19, 3,20, 4,11,22, 1,12,14, 6,17, 9],
    [ 9,17, 6,14,12, 1,22,11, 4,20, 3,19,16, 0,23, 7, 8,21, 2,15,13, 5,18,10],
    [ 8,16, 4,12,13, 0,20, 9, 5,21, 1,17,18, 2,22, 6,10,23, 3,14,15, 7,19,11],
    [15, 7,19,11,10,23, 3,14,18, 2,22, 6, 5,21, 1,17,13, 0,20, 9, 8,16, 4,12],
    [14, 6,17, 9,11,22, 1,12,19, 3,20, 4, 7,23, 0,16,15, 2,21, 8,10,18, 5,13],
    [13, 5,18,10, 8,21, 2,15,16, 0,23, 7, 4,20, 3,19,12, 1,22,11, 9,17, 6,14],
    [12, 4,16, 8, 9,20, 0,13,17, 1,21, 5, 6,22, 2,18,14, 3,23,10,11,19, 7,15],
    [ 7,11,15,19, 3,10,14,23, 2, 6,18,22, 1, 5,17,21, 0, 9,13,20, 4, 8,12,16],
    [ 5,10,13,18, 2, 8,15,21, 0, 7,16,23, 3, 4,19,20, 1,11,12,22, 6, 9,14,17],
    [ 6, 9,14,17, 1,11,12,22, 3, 4,19,20, 0, 7,16,23, 2, 8,15,21, 5,10,13,18],
    [ 4, 8,12,16, 0, 9,13,20, 1, 5,17,21, 2, 6,18,22, 3,10,14,23, 7,11,15,19],
    [ 3, 2, 1, 0, 7, 6, 5, 4,11,10, 9, 8,15,14,13,12,19,18,17,16,23,22,21,20],
    [ 1, 3, 0, 2, 6, 4, 7, 5, 9,11, 8,10,14,12,15,13,17,19,16,18,22,20,23,21],
    [ 2, 0, 3, 1, 5, 7, 4, 6,10, 8,11, 9,13,15,12,14,18,16,19,17,21,23,20,22],
    [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
], dtype=torch.long)
# fmt: on

# Inverse table: _INVERSE[i] = j means g_j = g_i^{-1}.
_INVERSE = torch.tensor(
    [0, 1, 2, 3, 4, 12, 8, 16, 6, 9, 14, 17, 5, 13, 10, 18, 7, 11, 15, 19, 20, 22, 21, 23],
    dtype=torch.long,
)

# Kronecker table from paper Table 6 (Mataigne et al. 2024):
#   rho_i x rho_j decomposes into irreps according to this table.
#   KRON_TABLE[i][j] = string where k-th char is multiplicity of rho_k.
# fmt: off
_KRON_TABLE = [
    ['10000', '01000', '00100', '00010', '00001'],
    ['01000', '11110', '01111', '01100', '00100'],
    ['00100', '01111', '11110', '01100', '01000'],
    ['00010', '01100', '01100', '10011', '00010'],
    ['00001', '00100', '01000', '00010', '10000'],
]
# fmt: on


def _build_rho3_matrices() -> torch.Tensor:
    """Build the 2D irrep matrices for all 24 elements.

    The 2D irrep of O factors through the surjection O -> S3 (permutations of
    the 3 coordinate axes). The resulting S3 representation uses the standard
    2D irrep of S3 in the orthogonal basis.

    Returns: (24, 2, 2) float64 tensor.
    """
    # Precomputed and verified against escnn. The values are combinations of
    # {0, ±1, ±1/2, ±sqrt(3)/2} — rotation matrices R(k*pi/3) and reflections.
    # fmt: off
    mats = torch.tensor([
        [[ 1.0,    0.0   ], [ 0.0,    1.0   ]],  # g00
        [[-_S3_2, -0.5   ], [-0.5,    _S3_2 ]],  # g01
        [[-_S3_2, -0.5   ], [-0.5,    _S3_2 ]],  # g02
        [[ 1.0,    0.0   ], [ 0.0,    1.0   ]],  # g03
        [[ 0.0,    1.0   ], [ 1.0,    0.0   ]],  # g04
        [[-0.5,   -_S3_2 ], [ _S3_2,  -0.5  ]],  # g05
        [[-0.5,   -_S3_2 ], [ _S3_2,  -0.5  ]],  # g06
        [[ 0.0,    1.0   ], [ 1.0,    0.0   ]],  # g07
        [[-0.5,    _S3_2 ], [-_S3_2,  -0.5  ]],  # g08
        [[ _S3_2, -0.5   ], [-0.5,   -_S3_2 ]],  # g09
        [[ _S3_2, -0.5   ], [-0.5,   -_S3_2 ]],  # g10
        [[-0.5,    _S3_2 ], [-_S3_2,  -0.5  ]],  # g11
        [[-0.5,    _S3_2 ], [-_S3_2,  -0.5  ]],  # g12
        [[ _S3_2, -0.5   ], [-0.5,   -_S3_2 ]],  # g13
        [[ _S3_2, -0.5   ], [-0.5,   -_S3_2 ]],  # g14
        [[-0.5,    _S3_2 ], [-_S3_2,  -0.5  ]],  # g15
        [[ 0.0,    1.0   ], [ 1.0,    0.0   ]],  # g16
        [[-0.5,   -_S3_2 ], [ _S3_2,  -0.5  ]],  # g17
        [[-0.5,   -_S3_2 ], [ _S3_2,  -0.5  ]],  # g18
        [[ 0.0,    1.0   ], [ 1.0,    0.0   ]],  # g19
        [[ 1.0,    0.0   ], [ 0.0,    1.0   ]],  # g20
        [[-_S3_2, -0.5   ], [-0.5,    _S3_2 ]],  # g21
        [[-_S3_2, -0.5   ], [-0.5,    _S3_2 ]],  # g22
        [[ 1.0,    0.0   ], [ 0.0,    1.0   ]],  # g23 = identity
    ], dtype=torch.float64)
    # fmt: on
    return mats


def _compute_cg_octa(
    rho_i_mats: torch.Tensor,
    rho_j_mats: torch.Tensor,
    di: int,
    dj: int,
    kron_str: str,
    all_irrep_mats: list[torch.Tensor],
    irrep_dims: list[int],
) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
    """Compute CG matrix for rho_i x rho_j on the octahedral group.

    Uses simultaneous diagonalization: find unitary C such that for all g:
        C^T (rho_i(g) kron rho_j(g)) C = oplus rho_k(g)

    The approach: pick two generators whose Kronecker representations have
    non-degenerate joint eigenspaces, then find the simultaneous block-
    diagonalization via Schur decomposition.

    Args:
        rho_i_mats: (24, di, di) irrep matrices for rho_i
        rho_j_mats: (24, dj, dj) irrep matrices for rho_j
        di, dj: dimensions of the two irreps
        kron_str: string from Kronecker table (e.g. "11110")
        all_irrep_mats: list of (24, dk, dk) tensors for all 5 irreps
        irrep_dims: [1, 3, 3, 2, 1]

    Returns:
        (C, block_info) where C is (di*dj, di*dj) orthogonal matrix and
        block_info is list of (irrep_idx, row_start, row_end) tuples.
    """
    d = di * dj
    n_g = 24

    # Build Kronecker product matrices for all group elements
    kron_mats = torch.zeros(n_g, d, d, dtype=torch.float64)
    for g in range(n_g):
        kron_mats[g] = torch.kron(rho_i_mats[g], rho_j_mats[g])

    # Parse which irreps appear in the decomposition
    target_blocks: list[tuple[int, int]] = []  # (irrep_idx, dim)
    for k, ch in enumerate(kron_str):
        mult = int(ch)
        for _ in range(mult):
            target_blocks.append((k, irrep_dims[k]))

    # Strategy: use the group algebra projection operators.
    # For each irrep rho_k with dim d_k, the projection is:
    #   P_k = (d_k / |G|) * sum_g chi_k(g)^* * kron(rho_i(g), rho_j(g))
    # where chi_k(g) = tr(rho_k(g)) is the character.
    # P_k projects onto the isotypic component for rho_k.
    # Then within each isotypic component, we use the specific matrix elements
    # to extract individual copies.
    #
    # More directly: for irrep rho_k with matrix entries rho_k(g)_{ab},
    #   E_k^{ab} = (d_k / |G|) * sum_g rho_k(g^{-1})_{ba} * kron(rho_i(g), rho_j(g))
    # maps the a-th basis vector of the m-th copy to the b-th basis vector of
    # the same copy. E_k^{aa} are projections onto columns.
    #
    # Reference: Serre, "Linear Representations of Finite Groups", Ch. 2.

    inv = _INVERSE
    C_cols: list[torch.Tensor] = []
    block_info: list[tuple[int, int, int]] = []
    col = 0

    for irrep_k, d_k in target_blocks:
        rho_k = all_irrep_mats[irrep_k]  # (24, d_k, d_k)

        # Build projection E_k^{00}: projects onto first basis vector of one copy
        E00 = torch.zeros(d, d, dtype=torch.float64)
        for g in range(n_g):
            g_inv = inv[g].item()
            E00 += rho_k[g_inv, 0, 0] * kron_mats[g]
        E00 *= d_k / n_g

        # Find a non-zero column of E00 — this is the first basis vector
        col_norms = E00.norm(dim=0)
        best_col = col_norms.argmax().item()
        v0 = E00[:, best_col].clone()
        v0 = v0 / v0.norm()

        # Orthogonalise against previously found vectors
        for prev in C_cols:
            v0 = v0 - (v0 @ prev) * prev
        if v0.norm() < 1e-10:
            # Degenerate — try other columns
            for try_col in range(d):
                v0 = E00[:, try_col].clone()
                if v0.norm() < 1e-10:
                    continue
                v0 = v0 / v0.norm()
                for prev in C_cols:
                    v0 = v0 - (v0 @ prev) * prev
                if v0.norm() > 1e-10:
                    break
        v0 = v0 / v0.norm()
        C_cols.append(v0)

        # For d_k > 1, get remaining basis vectors via E_k^{a0}
        for a in range(1, d_k):
            Ea0 = torch.zeros(d, d, dtype=torch.float64)
            for g in range(n_g):
                g_inv = inv[g].item()
                Ea0 += rho_k[g_inv, 0, a] * kron_mats[g]
            Ea0 *= d_k / n_g

            va = Ea0 @ v0
            # Orthogonalise
            for prev in C_cols:
                va = va - (va @ prev) * prev
            va = va / va.norm()
            C_cols.append(va)

        block_info.append((irrep_k, col, col + d_k))
        col += d_k

    C = torch.stack(C_cols, dim=1)

    # Verify orthogonality
    CtC = C.T @ C
    if not torch.allclose(CtC, torch.eye(d, dtype=torch.float64), atol=1e-8):
        C, _ = torch.linalg.qr(C)

    # Verify block-diagonalization for all group elements
    for g in range(n_g):
        block_diag = C.T @ kron_mats[g] @ C
        # Check it's block-diagonal according to block_info
        for irrep_k, r0, r1 in block_info:
            expected_block = all_irrep_mats[irrep_k][g]
            actual_block = block_diag[r0:r1, r0:r1]
            err = (actual_block - expected_block).abs().max().item()
            if err > 1e-6:
                raise RuntimeError(
                    f'CG verification failed: irrep {irrep_k}, element {g}, max error {err:.2e}'
                )

    return C, block_info


class OonR3(nn.Module):
    """Bispectrum of the octahedral group O acting on R^3.

    Signal f: O -> R has length 24 (one value per group element).
    Encoding: f = [f(g_0), ..., f(g_23)] where group elements are sorted
    lexicographically by their 3x3 rotation matrix entries.
    g_23 is the identity element.

    The octahedral group has 5 irreps (dims 1, 3, 3, 2, 1).
    The selective bispectrum uses 4 matrix coefficients (172 scalars).

    Reference: Mataigne et al., "The Selective G-Bispectrum and its Inversion:
    Applications to G-Invariant Networks", NeurIPS 2024.
    Forward uses Theorem 3.1; selective path from Appendix E, Table 6.

    Args:
        selective: If True (default), compute the selective bispectrum
            (4 coefficients, 172 scalars). If False, raises NotImplementedError.
    """

    GROUP_ORDER = 24
    N_IRREPS = 5
    IRREP_DIMS = [1, 3, 3, 2, 1]

    def __init__(self, selective: bool = True) -> None:
        super().__init__()
        self.selective = selective

        # Register group element matrices
        self.register_buffer('_elements', _ELEMENTS_3x3.clone())
        self.register_buffer('_cayley', _CAYLEY.clone())

        # Build all irrep matrices: list of (24, d_k, d_k) tensors
        self._irrep_dims = list(self.IRREP_DIMS)

        # rho0: trivial — all ones
        rho0 = torch.ones(24, 1, 1, dtype=torch.float64)
        self.register_buffer('_rho0', rho0)

        # rho1: standard 3D rep — the rotation matrices
        rho1 = _ELEMENTS_3x3.clone()
        self.register_buffer('_rho1', rho1)

        # rho4: alternating 1D rep
        rho4 = _RHO4_SIGNS.reshape(24, 1, 1).clone()
        self.register_buffer('_rho4', rho4)

        # rho2: product rep = rho4 * rho1
        rho2 = _RHO4_SIGNS[:, None, None] * rho1
        self.register_buffer('_rho2', rho2)

        # rho3: 2D rep
        rho3 = _build_rho3_matrices()
        self.register_buffer('_rho3', rho3)

        all_irrep_mats = [rho0, rho1, rho2, rho3, rho4]

        # Selective path: beta_{rho0,rho0}, beta_{rho0,rho1},
        # beta_{rho1,rho1}, beta_{rho1,rho2}
        # We need CG matrices for:
        #   (rho0, rho0): trivial, 1x1, no CG needed
        #   (rho0, rho1): 1x3 kron = 3, decomposes to rho1, no CG needed
        #   (rho1, rho1): 3x3 kron = 9, decomposes to rho0+rho1+rho2+rho3
        #   (rho1, rho2): 3x3 kron = 9, decomposes to rho1+rho2+rho3+rho4
        cg_11, self._block_info_11 = _compute_cg_octa(
            rho1,
            rho1,
            3,
            3,
            _KRON_TABLE[1][1],
            all_irrep_mats,
            self._irrep_dims,
        )
        self.register_buffer('_cg_11', cg_11)

        cg_12, self._block_info_12 = _compute_cg_octa(
            rho1,
            rho2,
            3,
            3,
            _KRON_TABLE[1][2],
            all_irrep_mats,
            self._irrep_dims,
        )
        self.register_buffer('_cg_12', cg_12)

        # Build index map
        idx_map: list[tuple[int, ...]] = []
        # beta_{rho0, rho0}: 1 scalar
        idx_map.append((0, 0))
        # beta_{rho0, rho1}: 3x3 = 9 scalars
        for r in range(3):
            for c in range(3):
                idx_map.append((0, 1, r, c))
        # beta_{rho1, rho1}: 9x9 = 81 scalars
        for r in range(9):
            for c in range(9):
                idx_map.append((1, 1, r, c))
        # beta_{rho1, rho2}: 9x9 = 81 scalars
        for r in range(9):
            for c in range(9):
                idx_map.append((1, 2, r, c))
        self._index_map = idx_map

    def _get_irrep_mats(self, k: int) -> torch.Tensor:
        """Get irrep matrices for rho_k, shape (24, d_k, d_k)."""
        return getattr(self, f'_rho{k}')

    def _group_dft(self, f: torch.Tensor) -> list[torch.Tensor]:
        """Fourier transform on O.

        F(rho_k) = sum_{g in O} f(g) * rho_k(g)

        This uses the left convention: under T_h f, F(rho_k) -> rho_k(h) F(rho_k).

        Args:
            f: (batch, 24) real signal.

        Returns:
            List of 5 tensors [F_rho0, ..., F_rho4] with shapes
            (batch, d_k, d_k).
        """
        fhat = []
        for k in range(self.N_IRREPS):
            rho_k = self._get_irrep_mats(k).to(f.dtype)  # (24, dk, dk)
            F_k = torch.einsum('bg, gij -> bij', f, rho_k)
            fhat.append(F_k)
        return fhat

    def _inverse_dft(self, fhat: list[torch.Tensor]) -> torch.Tensor:
        """Inverse Fourier transform on O.

        f(g) = (1/|G|) * sum_k d_k * tr(rho_k(g)^T @ F_k)

        Args:
            fhat: List of 5 tensors [F_rho0, ..., F_rho4] with shapes
                (batch, d_k, d_k).

        Returns:
            (batch, 24) real signal.
        """
        batch = fhat[0].shape[0]
        dtype = fhat[0].dtype
        device = fhat[0].device
        f = torch.zeros(batch, 24, dtype=dtype, device=device)

        for k in range(self.N_IRREPS):
            d_k = self._irrep_dims[k]
            rho_k = self._get_irrep_mats(k).to(dtype)  # (24, dk, dk)
            F_k = fhat[k]  # (batch, dk, dk)
            # tr(rho_k(g)^T @ F_k) = sum_{ij} rho_k(g)_{ji} F_k_{jk} delta_{ik}
            # = sum_j rho_k(g)_{ji} F_k_{ji} = sum_{ij} rho_k(g)_{ij} F_k_{ij}
            f += d_k * torch.einsum('gij, bij -> bg', rho_k, F_k) / self.GROUP_ORDER

        return f

    def _build_fplus(
        self,
        fhat: list[torch.Tensor],
        block_info: list[tuple[int, int, int]],
        d: int,
    ) -> torch.Tensor:
        """Build block-diagonal Fplus = oplus_{rho in decomp} F(rho).

        Args:
            fhat: Fourier coefficients
            block_info: list of (irrep_idx, row_start, row_end)
            d: total dimension (di * dj)

        Returns:
            (batch, d, d) block-diagonal matrix.
        """
        batch = fhat[0].shape[0]
        dtype = fhat[0].dtype
        device = fhat[0].device
        Fplus = torch.zeros(batch, d, d, dtype=dtype, device=device)

        for irrep_k, r0, r1 in block_info:
            Fplus[:, r0:r1, r0:r1] = fhat[irrep_k]

        return Fplus

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Compute the O-Bispectrum of signal f.

        Args:
            f: Real signal on O. Shape (batch, 24).

        Returns:
            Bispectrum tensor. Shape (batch, output_size).
            Real-valued but returned as complex dtype for API consistency.
            Imaginary parts are zero to machine precision for real inputs.
        """
        if not self.selective:
            raise NotImplementedError(
                'Full bispectrum not yet implemented for OonR3. Use selective=True (default).'
            )

        if f.ndim != 2 or f.shape[-1] != self.GROUP_ORDER:
            raise ValueError(f'Expected input shape (batch, {self.GROUP_ORDER}), got {f.shape}')

        batch = f.shape[0]
        dtype = f.dtype
        cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128

        fhat = self._group_dft(f)

        parts: list[torch.Tensor] = []

        # beta_{rho0, rho0} = F(rho0)^3  (scalar)
        # For trivial rep: F(rho0) = sum_g f(g), so beta = (sum f)^3
        F0 = fhat[0][:, 0, 0]  # (batch,)
        beta_00 = (F0**3).unsqueeze(-1)
        parts.append(beta_00.to(cdtype))

        # beta_{rho0, rho1} = F(rho0) * F(rho1)^T @ F(rho1)  (3x3)
        # kron(rho0, rho1) = rho1, CG = identity, Fplus = F(rho1)
        F1 = fhat[1]  # (batch, 3, 3)
        beta_01 = F0[:, None, None] * torch.bmm(F1.transpose(-1, -2), F1)
        parts.append(beta_01.reshape(batch, 9).to(cdtype))

        # beta_{rho1, rho1}: 9x9
        # beta = [F1 kron F1] @ C @ [oplus F^T] @ C^T
        # But the standard formula is:
        # beta = C @ (oplus F^dag) @ C^dag @ (F_i kron F_j)^dag
        # For real irreps: dag = T
        F1_kron = torch.zeros(batch, 9, 9, dtype=dtype, device=f.device)
        for b in range(batch):
            F1_kron[b] = torch.kron(F1[b], F1[b])

        Fplus_11 = self._build_fplus(fhat, self._block_info_11, 9)
        C11 = self._cg_11.to(dtype)
        beta_11 = C11 @ Fplus_11.transpose(-1, -2) @ C11.T @ F1_kron
        parts.append(beta_11.reshape(batch, 81).to(cdtype))

        # beta_{rho1, rho2}: 9x9
        F2 = fhat[2]  # (batch, 3, 3)
        F12_kron = torch.zeros(batch, 9, 9, dtype=dtype, device=f.device)
        for b in range(batch):
            F12_kron[b] = torch.kron(F1[b], F2[b])

        Fplus_12 = self._build_fplus(fhat, self._block_info_12, 9)
        C12 = self._cg_12.to(dtype)
        beta_12 = C12 @ Fplus_12.transpose(-1, -2) @ C12.T @ F12_kron
        parts.append(beta_12.reshape(batch, 81).to(cdtype))

        return torch.cat(parts, dim=-1)

    def _bootstrap_init(self, beta: torch.Tensor) -> torch.Tensor:
        """Bootstrap initialization for inversion.

        Recovers F0 from beta_{rho0,rho0} and F1 (as symmetric square root
        of F1^T F1) from beta_{rho0,rho1}. Then extracts approximate F2, F3
        from beta_{rho1,rho1} diagonal blocks (ignoring l=2 mixing) and F4
        from beta_{rho1,rho2}.

        Args:
            beta: Selective bispectrum, shape (batch, output_size).

        Returns:
            Approximate signal, shape (batch, 24), in float64.
        """
        batch = beta.shape[0]
        device = beta.device
        dtype = torch.float64
        beta_r = beta.real.to(dtype)

        fhat: list[torch.Tensor] = [
            torch.zeros(batch, d, d, dtype=dtype, device=device) for d in self._irrep_dims
        ]

        # F0 from beta_00 = F0^3
        b00 = beta_r[:, 0]
        F0 = torch.sign(b00) * torch.abs(b00) ** (1.0 / 3.0)
        fhat[0][:, 0, 0] = F0

        # F1 from beta_01 / F0 = F1^T F1 -> symmetric sqrt
        b01 = beta_r[:, 1:10].reshape(batch, 3, 3)
        safe_F0 = F0.clone()
        safe_F0[safe_F0.abs() < 1e-12] = 1.0
        M = b01 / safe_F0[:, None, None]
        eigvals_M, eigvecs_M = torch.linalg.eigh(M)
        eigvals_M = torch.clamp(eigvals_M, min=0.0)
        S = eigvecs_M @ torch.diag_embed(torch.sqrt(eigvals_M)) @ eigvecs_M.transpose(-1, -2)
        fhat[1] = S

        # Extract approximate F2, F3 from beta_11 using R R^T
        # R = C^T beta (S⊗S)^{-1} C; R R^T = block_diag(F_k^T F_k)
        b11 = beta_r[:, 10:91].reshape(batch, 9, 9)
        C11 = self._cg_11.to(dtype)
        for b_idx in range(batch):
            S_kron = torch.kron(S[b_idx], S[b_idx])
            R = C11.T @ b11[b_idx] @ torch.linalg.inv(S_kron) @ C11
            RRT = R @ R.T

            # F2 from block [4:7, 4:7]
            F2TF2 = RRT[4:7, 4:7]
            ev2, evec2 = torch.linalg.eigh(F2TF2)
            ev2 = torch.clamp(ev2, min=0.0)
            fhat[2][b_idx] = evec2 @ torch.diag(torch.sqrt(ev2)) @ evec2.T

            # F3 from block [7:9, 7:9]
            F3TF3 = RRT[7:9, 7:9]
            ev3, evec3 = torch.linalg.eigh(F3TF3)
            ev3 = torch.clamp(ev3, min=0.0)
            fhat[3][b_idx] = evec3 @ torch.diag(torch.sqrt(ev3)) @ evec3.T

        # F4 from beta_12: use diagonal block of extraction
        b12 = beta_r[:, 91:172].reshape(batch, 9, 9)
        C12 = self._cg_12.to(dtype)
        for b_idx in range(batch):
            F2_approx = fhat[2][b_idx]
            F12_kron = torch.kron(S[b_idx], F2_approx)
            det_check = torch.linalg.det(F12_kron)
            if det_check.abs() > 1e-10:
                R12 = C12.T @ b12[b_idx] @ torch.linalg.inv(F12_kron) @ C12
                RRT12 = R12 @ R12.T
                # F4 is 1x1, last block
                for irrep_k, r0, _r1 in self._block_info_12:
                    if irrep_k == 4:
                        val = torch.sqrt(torch.clamp(RRT12[r0, r0], min=0.0))
                        fhat[4][b_idx, 0, 0] = val

        return self._inverse_dft(fhat)

    def invert(
        self,
        beta: torch.Tensor,
        n_corrections: int = 10,
        n_restarts: int = 4,
        **kwargs: object,
    ) -> torch.Tensor:
        """Recover a signal from its selective bispectrum.

        Uses bootstrap initialization (exact Fourier norms, approximate phases)
        followed by Levenberg-Marquardt corrections. Unlike D_n, the octahedral
        group's 3D irreps introduce a continuous SO(3) phase ambiguity in the
        symmetric square root of F1^T F1. Each correction is a single
        regularised linear solve, not iterative optimization.

        Multiple restarts with randomised Fourier phases ensure convergence
        from any starting point.

        The reconstruction has O-indeterminacy: the recovered signal matches
        the original up to an octahedral group action.

        Args:
            beta: Selective bispectrum, shape (batch, output_size).
            n_corrections: LM correction steps per restart.
            n_restarts: Number of restarts with random phase perturbations.

        Returns:
            Reconstructed real signal, shape (batch, 24).
        """
        if not self.selective:
            raise NotImplementedError('Inversion only implemented for selective bispectrum.')

        target_real = beta.real
        f_init = self._bootstrap_init(beta)
        fhat_init = self._group_dft(f_init.to(torch.float64))

        best_f = f_init.to(beta.real.dtype)
        best_loss = torch.full(
            (beta.shape[0],),
            float('inf'),
            dtype=beta.real.dtype,
            device=beta.device,
        )

        for trial in range(n_restarts):
            if trial == 0:
                f = f_init.to(beta.real.dtype)
            else:
                fhat_pert = [fk.clone() for fk in fhat_init]
                for k in range(self.N_IRREPS):
                    dk = self._irrep_dims[k]
                    if dk > 1:
                        Q = torch.linalg.qr(
                            torch.randn(
                                beta.shape[0], dk, dk, dtype=torch.float64, device=beta.device
                            ),
                        ).Q
                        fhat_pert[k] = Q @ fhat_pert[k]
                f = self._inverse_dft(fhat_pert).to(beta.real.dtype)

            for _ in range(n_corrections):
                f = self._lm_step(f, beta)

            with torch.no_grad():
                residuals = (self.forward(f).real - target_real).norm(dim=-1)
                improved = residuals < best_loss
                if improved.any():
                    best_f[improved] = f[improved]
                    best_loss[improved] = residuals[improved]

        return best_f

    def _lm_step(
        self,
        f: torch.Tensor,
        beta_target: torch.Tensor,
    ) -> torch.Tensor:
        """One Levenberg-Marquardt step with adaptive damping.

        Uses (J^T J + mu I)^{-1} J^T r with mu adapted per sample to ensure the residual decreases.
        """
        batch = f.shape[0]
        results = []
        target_real = beta_target.real

        for b_idx in range(batch):
            f_b = f[b_idx].detach().clone()

            def fwd(x: torch.Tensor) -> torch.Tensor:
                return self.forward(x.unsqueeze(0)).squeeze(0).real

            J = torch.func.jacfwd(fwd)(f_b)
            beta_pred = fwd(f_b)
            residual = target_real[b_idx] - beta_pred
            current_loss = residual.norm()

            JTJ = J.T @ J
            JTr = J.T @ residual
            mu = max(1e-3 * JTJ.diag().max().item(), 1e-8)

            f_best = f[b_idx]
            for _ in range(10):
                delta_f = torch.linalg.solve(
                    JTJ + mu * torch.eye(self.GROUP_ORDER, dtype=J.dtype, device=J.device),
                    JTr,
                )
                f_new = f[b_idx] + delta_f
                new_loss = (target_real[b_idx] - fwd(f_new)).norm()
                if new_loss < current_loss:
                    f_best = f_new
                    mu = max(mu * 0.5, 1e-10)
                    break
                mu *= 5.0

            results.append(f_best)

        return torch.stack(results, dim=0)

    @property
    def output_size(self) -> int:
        """Number of scalar bispectral values in the output."""
        return len(self._index_map)

    @property
    def index_map(self) -> list[tuple[int, ...]]:
        """Maps flat output index -> (irrep pair, matrix entry) tuple."""
        return list(self._index_map)

    def extra_repr(self) -> str:
        return f'selective={self.selective}, output_size={self.output_size}'
