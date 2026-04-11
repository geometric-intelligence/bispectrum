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

import hashlib
from collections import OrderedDict

import torch
import torch.nn as nn
from torch_harmonics import RealSHT

from bispectrum._cg import _CACHE_DIR, compute_reduced_cg_parallel, load_cg_matrices


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

    1. **Mixed chains** ``(l1, l2, l_target)`` with ``l1 < l2 < l_target``
       — linear in ``F_{l_target}`` (appears via ``conj(F_l)``).
    2. **Cross** ``(l1, l_target, l)`` with ``l1 < l_target, l < l_target``
       — linear in ``F_{l_target}`` (appears in the middle ⊗ position).
       Interleaved round-robin across ``l1`` for diverse coupling.
    3. **Self-pairing chains** ``(l, l, l_target)`` — deprioritized because
       symmetric tensor products can be linearly dependent on mixed chains.
    4. **Power** ``(0, l_target, l_target)`` — gives ``||F_{l_target}||²``
       (quadratic).
    5. **Self-coupling** ``(l_target, l_target, l)`` with ``l < l_target``
       — quadratic/cubic in ``F_{l_target}``.

    At ``l_target = 2`` the self-coupling entries are excluded:
    ``beta_{2,2,0}`` is redundant with the power entry, and
    ``beta_{2,2,1}`` vanishes identically (antisymmetry of the
    ``2 ⊗ 2 → 1`` Clebsch–Gordan channel). Degrees 2 and 3 are
    recovered jointly; see the completeness proof for details.

    Total output size is exactly ``(lmax + 1)² - 3``, matching the
    information-theoretic lower bound ``dim(R^{(L+1)²} / SO(3))``.
    """
    index_map: list[tuple[int, int, int]] = []

    for l_target in range(lmax + 1):
        budget = 2 * l_target + 1

        if l_target == 0:
            index_map.append((0, 0, 0))
            continue

        candidates: list[tuple[int, int, int]] = []

        # 1+2. Chain and cross entries, interleaved for generic full rank.
        #
        # Chain: (l1, l2, l_target) with l1 <= l2 < l_target,
        #        l1 + l2 >= l_target.  Mixed (l1 < l2) before self-pairing.
        # Cross: (l1, l_target, l) with 1 <= l1 < l_target,
        #        l_target - l1 <= l < l_target.  Round-robin across l1.
        chain_mixed: list[tuple[int, int, int]] = []
        chain_self: list[tuple[int, int, int]] = []
        for l2 in range(l_target - 1, -1, -1):
            for l1 in range(min(l2, l_target - 1), -1, -1):
                if l1 + l2 >= l_target and abs(l1 - l2) <= l_target:
                    if l1 < l2:
                        chain_mixed.append((l1, l2, l_target))
                    else:
                        chain_self.append((l1, l2, l_target))

        cross_all: list[tuple[int, int, int]] = []
        cross_by_l1: dict[int, list[tuple[int, int, int]]] = {}
        for l1 in range(l_target - 1, 0, -1):
            l_lo = l_target - l1
            l_hi = min(l_target - 1, l1 + l_target)
            entries = [(l1, l_target, lv) for lv in range(l_hi, l_lo - 1, -1)]
            if entries:
                cross_by_l1[l1] = entries
        round_idx = 0
        active_l1s = sorted(cross_by_l1.keys(), reverse=True)
        while active_l1s:
            next_active: list[int] = []
            for l1 in active_l1s:
                if round_idx < len(cross_by_l1[l1]):
                    cross_all.append(cross_by_l1[l1][round_idx])
                if round_idx + 1 < len(cross_by_l1[l1]):
                    next_active.append(l1)
            active_l1s = next_active
            round_idx += 1

        # Interleave: mixed chains, then cross entries, then self-pairing
        # chains. Cross entries provide independent linear constraints
        # that self-pairing chains cannot (due to CG symmetries).
        candidates.extend(chain_mixed)
        candidates.extend(cross_all)
        candidates.extend(chain_self)

        # 3. Power entry: (0, l_target, l_target).
        candidates.append((0, l_target, l_target))

        # 4. Self-coupling: (l_target, l_target, l) with 0 <= l < l_target.
        #    Skipped at l_target=2: beta_{2,2,0} ∝ ||F_2||² (redundant with
        #    power) and beta_{2,2,1} ≡ 0 (CG antisymmetry).
        if l_target != 2:
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

        if selective:
            self._index_map = _build_selective_index_map(lmax)
        else:
            cg_data = load_cg_matrices(lmax)
            self._index_map = _build_full_index_map(lmax, cg_data)

        self._build_group_tables(cg_data if not selective else None)

    def _build_group_tables(self, cg_data: dict[tuple[int, int], torch.Tensor] | None) -> None:
        """Precompute per-(l1, l2) group tables with reduced CG matrices.

        For each (l1, l2) group, builds a *reduced* CG matrix containing
        only the columns needed by the entries in the group.  This turns
        the ``(batch, d) @ (d, d)`` matmul into ``(batch, d) @ (d, c)``
        where ``c`` is the total number of coupled-basis elements actually
        used — often much smaller than ``d`` for the selective bispectrum.

        When *cg_data* is ``None`` (selective mode), computes only the
        needed columns directly using parallel workers — never building
        full CG matrices.  When *cg_data* is provided (full mode), slices
        columns from the precomputed full matrices.
        """
        groups: OrderedDict[tuple[int, int], list[tuple[int, int, int, int]]] = OrderedDict()
        for out_idx, (l1, l2, l_val) in enumerate(self._index_map):
            key = (l1, l2)
            n_p, _ = _compute_padding_indices(l1, l2, l_val)
            size_l = 2 * l_val + 1
            groups.setdefault(key, []).append((l_val, out_idx, n_p, size_l))

        # Build extraction metadata for each group.
        # Sort entries by l_val within each group so columns match the
        # natural ascending-l order produced by compute_cg_columns.
        group_meta: list[
            tuple[int, int, list[tuple[int, int, int, int]], list[int], list[int]]
        ] = []
        for (l1, l2), entries in groups.items():
            entries_sorted = sorted(entries, key=lambda e: e[0])  # sort by l_val
            col_indices: list[int] = []
            extract_entries: list[tuple[int, int, int, int]] = []
            l_vals_needed: list[int] = []
            offset = 0
            for l_val, out_idx, n_p, size_l in entries_sorted:
                col_indices.extend(range(n_p, n_p + size_l))
                extract_entries.append((out_idx, offset, size_l, l_val))
                l_vals_needed.append(l_val)
                offset += size_l
            group_meta.append((l1, l2, extract_entries, col_indices, l_vals_needed))

        # Compute reduced CG matrices.
        if cg_data is not None:
            # Full mode: slice from precomputed full matrices.
            reduced_cgs = {}
            for gid, (l1, l2, _, col_indices, _) in enumerate(group_meta):
                reduced_cgs[gid] = cg_data[(l1, l2)][:, col_indices]
        else:
            # Selective mode: try disk cache, else compute in parallel.
            reduced_cgs = self._load_or_compute_reduced_cg(group_meta)

        # Register buffers and build _group_data.
        self._group_data: list[tuple[int, int, int, list[tuple[int, int, int, int]]]] = []
        for gid, (l1, l2, extract_entries, _col_indices, _) in enumerate(group_meta):
            cg_red = reduced_cgs[gid]
            c = cg_red.shape[1]
            self.register_buffer(f'_cg_red_{gid}', cg_red)
            self._group_data.append((l1, l2, c, extract_entries))

    @staticmethod
    def _load_or_compute_reduced_cg(
        group_meta: list,
    ) -> dict[int, torch.Tensor]:
        """Load reduced CG from disk cache, or compute + save."""
        # Build a deterministic cache key from the group structure.
        key_parts = []
        for gid, (l1, l2, _, _, l_vals) in enumerate(group_meta):
            key_parts.append(f'{gid}:{l1},{l2}:{l_vals}')
        key_str = '|'.join(key_parts)
        cache_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
        cache_path = _CACHE_DIR / f'cg_reduced_{cache_hash}.pt'

        # Try loading from cache.
        if cache_path.exists():
            try:
                data = torch.load(cache_path, weights_only=True)
                return {int(k): v for k, v in data.items()}
            except (OSError, RuntimeError):
                pass

        # Compute in parallel.
        tasks = [(gid, l1, l2, l_vals) for gid, (l1, l2, _, _, l_vals) in enumerate(group_meta)]
        reduced_cgs = compute_reduced_cg_parallel(tasks)

        # Save to cache.
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({str(k): v for k, v in reduced_cgs.items()}, cache_path)
        except OSError:
            pass

        return reduced_cgs

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Compute the SO(3)-bispectrum of a signal on S^2.

        Uses block-sparse computation with reduced CG matrices: for each
        (l1, l2) group, multiplies the tensor product by a CG matrix
        containing only the needed columns, then extracts and contracts.

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

        result = torch.zeros(batch_size, num_entries, dtype=coeffs.dtype, device=f.device)

        for gid, (l1, l2, _c, extract_entries) in enumerate(self._group_data):
            fl1 = f_coeffs[l1]  # (batch, 2l1+1)
            fl2 = f_coeffs[l2]  # (batch, 2l2+1)

            # Outer product → flatten → reduced CG transform.
            tp = torch.einsum('bi,bj->bij', fl1, fl2).reshape(batch_size, -1)
            cg = getattr(self, f'_cg_red_{gid}')
            cg = cg.to(dtype=tp.dtype, device=tp.device)
            transformed = tp @ cg  # (batch, c)  — c ≪ d typically

            # Extract each l-block and contract with conj(F_l).
            for out_idx, offset, size_l, l_val in extract_entries:
                block = transformed[:, offset : offset + size_l]
                result[:, out_idx] = torch.sum(block * torch.conj(f_coeffs[l_val]), dim=-1)

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
