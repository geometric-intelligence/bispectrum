"""SO(3) bispectrum on S^2.

Implements the G-Bispectrum for SO(3) acting on the 2-sphere.

beta(f)_{l1,l2}^{(l)} = (F_l1 ⊗ F_l2) · C_{l1,l2} · F_l^†

where F_l are the degree-l SH coefficient vectors and C denotes the
Clebsch-Gordan matrices for SO(3).

Supports a **selective** mode that reduces the output from O(L³) to O(L²)
bispectral entries while preserving completeness for generic signals.
See ``_build_selective_index_map`` for the construction.

Reference: Kakarala (1992), Cohen et al.
"""

import torch
import torch.nn as nn
from torch_harmonics import RealSHT

from bispectrum._cg import load_cg_matrices


def _build_full_index_map(
    lmax: int, cg_data: dict[tuple[int, int], torch.Tensor]
) -> list[tuple[int, int, int]]:
    """Build the full O(L³) bispectrum index map."""
    index_map: list[tuple[int, int, int]] = []
    for l1 in range(lmax + 1):
        for l2 in range(l1, lmax + 1):
            if (l1, l2) not in cg_data:
                continue
            for l_val in range(abs(l1 - l2), min(l1 + l2, lmax) + 1):
                index_map.append((l1, l2, l_val))
    return index_map


def _build_selective_index_map(lmax: int) -> list[tuple[int, int, int]]:
    """Build the O(L²) selective bispectrum index map.

    For each target degree ``l_target = 0 .. lmax``, selects up to
    ``2 * l_target + 1`` bispectral triples that provide enough equations
    to recover the SH coefficients at that degree (given all lower degrees
    are already known).

    Entry types (in priority order):

    1. **Chain** ``(l1, l2, l_target)`` with ``l1 <= l2 < l_target`` —
       linear in ``F_{l_target}`` (appears via ``conj(F_l)``).
    2. **Cross** ``(l1, l_target, l)`` with ``l1 < l_target, l < l_target``
       — linear in ``F_{l_target}`` (appears in the middle ⊗ position).
    3. **Power** ``(0, l_target, l_target)`` — gives ``||F_{l_target}||²``
       (quadratic).
    4. **Self-coupling** ``(l_target, l_target, l)`` with ``l < l_target``
       — quadratic/cubic in ``F_{l_target}``.

    Total output size is approximately ``(lmax + 1)²``.
    """
    index_map: list[tuple[int, int, int]] = []

    for l_target in range(lmax + 1):
        budget = 2 * l_target + 1

        if l_target == 0:
            index_map.append((0, 0, 0))
            continue

        candidates: list[tuple[int, int, int]] = []

        # 1. Chain entries: (l1, l2, l_target) with l1 <= l2 < l_target,
        #    l1 + l2 >= l_target (triangle inequality).
        #    Iterate highest l2 first for better conditioning.
        for l2 in range(l_target - 1, -1, -1):
            for l1 in range(min(l2, l_target - 1), -1, -1):
                if l1 + l2 >= l_target and abs(l1 - l2) <= l_target:
                    candidates.append((l1, l2, l_target))

        # 2. Cross entries: (l1, l_target, l) with 1 <= l1 < l_target,
        #    l_target - l1 <= l < l_target (triangle + known).
        for l1 in range(l_target - 1, 0, -1):
            l_lo = l_target - l1  # triangle lower bound (since l1 < l_target)
            l_hi = min(l_target - 1, l1 + l_target)  # must be < l_target
            for l_val in range(l_hi, l_lo - 1, -1):
                candidates.append((l1, l_target, l_val))

        # 3. Power entry: (0, l_target, l_target).
        candidates.append((0, l_target, l_target))

        # 4. Self-coupling: (l_target, l_target, l) with 0 <= l < l_target.
        for l_val in range(l_target - 1, -1, -1):
            candidates.append((l_target, l_target, l_val))

        # Deduplicate preserving priority order, take up to budget.
        seen: set[tuple[int, int, int]] = set()
        selected: list[tuple[int, int, int]] = []
        for entry in candidates:
            if entry not in seen:
                seen.add(entry)
                selected.append(entry)
                if len(selected) == budget:
                    break

        index_map.extend(selected)

    return index_map


class SO3onS2(nn.Module):
    """Bispectrum of SO(3) acting on S^2.

    Takes a real-valued signal on the sphere f(theta, phi) sampled on an
    equiangular grid, computes the spherical harmonic transform internally,
    and returns the bispectrum coefficients.

    When ``selective=True``, outputs O(L²) bispectral entries instead of
    O(L³), using a degree-by-degree construction that preserves completeness
    for generic signals.

    Args:
        lmax: Maximum spherical harmonic degree.
        nlat: Number of latitude grid points.
        nlon: Number of longitude grid points.
        selective: If True, use the O(L²) selective bispectrum. If False,
            compute all O(L³) entries.
    """

    def __init__(
        self,
        lmax: int = 5,
        nlat: int = 64,
        nlon: int = 128,
        selective: bool = False,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.nlat = nlat
        self.nlon = nlon
        self.selective = selective

        # RealSHT with lmax=L outputs coefficients for l=0..L-1,
        # so we need sht_lmax = lmax + 1 to get l=0..lmax.
        sht_lmax = lmax + 1
        self._sht = RealSHT(
            nlat, nlon, lmax=sht_lmax, mmax=sht_lmax, grid='equiangular', norm='ortho'
        )

        cg_data = load_cg_matrices(lmax)

        if selective:
            self._index_map = _build_selective_index_map(lmax)
        else:
            self._index_map = _build_full_index_map(lmax, cg_data)

        # Register CG buffers for (l1, l2) pairs used in the index map.
        registered: set[tuple[int, int]] = set()
        for l1, l2, _l in self._index_map:
            key = (l1, l2)
            if key not in registered and key in cg_data:
                self.register_buffer(f'cg_{l1}_{l2}', cg_data[key])
                registered.add(key)

        self._build_batched_bispectrum_tensors(cg_data)

    def _build_batched_bispectrum_tensors(
        self, cg_data: dict[tuple[int, int], torch.Tensor]
    ) -> None:
        """Precompute padded CG matrices and index arrays for vectorized forward."""
        lmax = self.lmax
        num_entries = len(self._index_map)
        if num_entries == 0:
            return

        D_max = (2 * lmax + 1) ** 2
        2 * lmax + 1

        l1_arr = torch.zeros(num_entries, dtype=torch.long)
        l2_arr = torch.zeros(num_entries, dtype=torch.long)
        l_arr = torch.zeros(num_entries, dtype=torch.long)
        d_prod_arr = torch.zeros(num_entries, dtype=torch.long)
        n_p_arr = torch.zeros(num_entries, dtype=torch.long)
        size_l_arr = torch.zeros(num_entries, dtype=torch.long)

        cg_padded = torch.zeros(num_entries, D_max, D_max, dtype=torch.float64)

        for idx, (l1, l2, l_val) in enumerate(self._index_map):
            l1_arr[idx] = l1
            l2_arr[idx] = l2
            l_arr[idx] = l_val
            d = (2 * l1 + 1) * (2 * l2 + 1)
            d_prod_arr[idx] = d
            size_l_arr[idx] = 2 * l_val + 1
            n_p, _ = _compute_padding_indices(l1, l2, l_val)
            n_p_arr[idx] = n_p

            cg = cg_data[(l1, l2)]
            cg_padded[idx, :d, :d] = cg

        self.register_buffer('_l1_arr', l1_arr)
        self.register_buffer('_l2_arr', l2_arr)
        self.register_buffer('_l_arr', l_arr)
        self.register_buffer('_d_prod_arr', d_prod_arr)
        self.register_buffer('_n_p_arr', n_p_arr)
        self.register_buffer('_size_l_arr', size_l_arr)
        self.register_buffer('_cg_padded', cg_padded)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Compute the SO(3)-bispectrum of a signal on S^2.

        Args:
            f: Real-valued signal on S^2. Shape: (batch, nlat, nlon).

        Returns:
            Complex bispectrum tensor. Shape: (batch, output_size).
        """
        coeffs = self._sht(f)
        f_coeffs = _get_full_sh_coefficients(coeffs)

        batch_size = f.shape[0]
        num_entries = self.output_size
        if num_entries == 0:
            return torch.zeros(batch_size, 0, dtype=coeffs.dtype, device=f.device)

        lmax = self.lmax
        W = 2 * lmax + 1
        D_max = W * W

        f_padded = torch.zeros(batch_size, lmax + 1, W, dtype=coeffs.dtype, device=f.device)
        for l_val, fl in f_coeffs.items():
            sz = fl.shape[1]
            start = (W - sz) // 2
            f_padded[:, l_val, start : start + sz] = fl

        f_l1 = f_padded[:, self._l1_arr]
        f_l2 = f_padded[:, self._l2_arr]
        outer = torch.einsum('bni,bnj->bnij', f_l1, f_l2)
        outer_flat = outer.reshape(batch_size, num_entries, D_max)

        cg = self._cg_padded.to(dtype=outer_flat.dtype, device=outer_flat.device)
        transformed = torch.bmm(
            outer_flat.reshape(batch_size * num_entries, 1, D_max),
            cg.unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .reshape(batch_size * num_entries, D_max, D_max),
        ).reshape(batch_size, num_entries, D_max)

        f_l_vals = f_padded[:, self._l_arr]
        f_hat_padded = torch.zeros(
            batch_size, num_entries, D_max, dtype=coeffs.dtype, device=f.device
        )
        for idx in range(num_entries):
            n_p = self._n_p_arr[idx].item()
            sz = self._size_l_arr[idx].item()
            l_val = self._l_arr[idx].item()
            start = (W - (2 * l_val + 1)) // 2
            f_hat_padded[:, idx, n_p : n_p + sz] = f_l_vals[:, idx, start : start + sz]

        result = torch.sum(transformed * torch.conj(f_hat_padded), dim=-1)
        return result

    def invert(self, beta: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Inversion is an open problem for SO(3) on S^2.

        Raises:
            NotImplementedError: Selective bispectrum and inversion for
                continuous groups remain open mathematical problems.
        """
        raise NotImplementedError(
            'Inversion for SO(3) on S^2 is an open mathematical problem. '
            'See DESIGN.md TODO-M1 and TODO-M4.'
        )

    @property
    def output_size(self) -> int:
        """Number of bispectral coefficients in the output."""
        return len(self._index_map)

    @property
    def index_map(self) -> list[tuple[int, int, int]]:
        """Maps flat output index -> (l1, l2, l) triple."""
        return list(self._index_map)

    def extra_repr(self) -> str:
        return (
            f'lmax={self.lmax}, nlat={self.nlat}, nlon={self.nlon}, '
            f'selective={self.selective}, output_size={self.output_size}'
        )


def _get_full_sh_coefficients(
    coeffs_positive_m: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Extend SHT output (m >= 0) to full m range using F_l^{-m} = (-1)^m conj(F_l^m)."""
    batch_size, lmax, mmax = coeffs_positive_m.shape
    result: dict[int, torch.Tensor] = {}

    for l_val in range(lmax):
        m_max_for_l = min(l_val, mmax - 1)
        if m_max_for_l == 0:
            result[l_val] = coeffs_positive_m[:, l_val, 0:1]
            continue

        m_range = torch.arange(1, m_max_for_l + 1, device=coeffs_positive_m.device)
        pos_m = coeffs_positive_m[:, l_val, 1 : m_max_for_l + 1]
        signs = (-1.0) ** m_range.to(coeffs_positive_m.dtype)
        neg_m = signs.unsqueeze(0) * torch.conj(pos_m)

        result[l_val] = torch.cat(
            [neg_m.flip(-1), coeffs_positive_m[:, l_val, 0:1], pos_m], dim=-1
        )

    return result


def _compute_padding_indices(l1: int, l2: int, l_val: int) -> tuple[int, int]:
    """Compute (n_p, n_s) zero-padding sizes for F_l in the coupled basis."""
    l_min = abs(l1 - l2)
    l_max = l1 + l2
    n_p = sum(2 * q + 1 for q in range(l_min, l_val))
    n_s = sum(2 * q + 1 for q in range(l_val + 1, l_max + 1))
    return n_p, n_s


def _pad_sh_coefficients(f_l: torch.Tensor, l1: int, l2: int, l_val: int) -> torch.Tensor:
    """Zero-pad F_l to size (2l1+1)(2l2+1) in the coupled basis."""
    batch_size = f_l.shape[0]
    total_size = (2 * l1 + 1) * (2 * l2 + 1)
    n_p, _ = _compute_padding_indices(l1, l2, l_val)
    padded = torch.zeros(batch_size, total_size, dtype=f_l.dtype, device=f_l.device)
    padded[:, n_p : n_p + 2 * l_val + 1] = f_l
    return padded


def _bispectrum_entry(
    f_coeffs: dict[int, torch.Tensor],
    l1: int,
    l2: int,
    l_val: int,
    cg_matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute scalar bispectrum entry beta_{l1,l2}[l].

    beta = (F_l1 ⊗ F_l2) @ C_{l1,l2} @ F_hat_l^†
    """
    f_l1 = f_coeffs[l1]
    f_l2 = f_coeffs[l2]

    if l_val not in f_coeffs:
        return torch.zeros(f_l1.shape[0], dtype=f_l1.dtype, device=f_l1.device)

    batch_size = f_l1.shape[0]

    outer = torch.einsum('bi,bj->bij', f_l1, f_l2)
    tensor_product = outer.reshape(batch_size, -1)

    cg = cg_matrix.to(device=tensor_product.device, dtype=tensor_product.dtype)
    transformed = tensor_product @ cg

    f_hat_l = _pad_sh_coefficients(f_coeffs[l_val], l1, l2, l_val)
    return torch.sum(transformed * torch.conj(f_hat_l), dim=-1)
