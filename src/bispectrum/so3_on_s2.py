"""SO(3) bispectrum on S^2.

Implements the G-Bispectrum for SO(3) acting on the 2-sphere.

beta(f)_{l1,l2}^{(l)} = (F_l1 ⊗ F_l2) · C_{l1,l2} · F_l^†

where F_l are the degree-l SH coefficient vectors and C denotes the
Clebsch-Gordan matrices for SO(3).

Supports a **selective** mode that reduces the output from O(L³) to O(L²)
entries while preserving completeness for generic signals, using an
augmented invariant that combines scalar bispectral entries with
CG power spectrum entries P_{l1,l2,l} = ||(F_{l1} ⊗ F_{l2})|_l||^2.
See ``_build_selective_index_map`` and ``_build_cg_power_index_map``.

**Completeness** (``proof_completeness.tex``): for band-limits L ≥ 4, the
augmented selective bispectrum is a complete SO(3)-invariant on generic
real-valued signals (those satisfying a_0^0 ≠ 0, ||F_1|| ≠ 0, and
a_2^1 ≠ 0 after gauge-fixing).  The proof uses:
  - Seed recovery at degrees 0–3 (finite fibre, Jacobian rank 11/11),
  - Degree-4 fibre reduction via parity-breaking entries β_{2,3,4} etc.,
  - Linear bootstrap at ℓ ≥ 4 (verified ℓ ≤ 100, closed-form for ℓ ≥ 8).
For L ≤ 3, the invariant separates O(3)-orbits but cannot resolve the
T_R (azimuthal reflection) ambiguity.

Reference: Kakarala (1992), Cohen et al.
"""

import hashlib
import logging
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from torch_harmonics import RealSHT

from bispectrum._cg import (
    _CACHE_DIR,
    compute_reduced_cg_parallel,
    compute_sparse_cg_parallel,
    load_cg_matrices,
)

logger = logging.getLogger(__name__)


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


def _small_linear_bootstrap_block(l_target: int) -> list[tuple[int, int, int]]:
    """Explicit low-degree linear blocks used in the completeness proof."""
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
    return explicit_blocks[l_target]


def _proved_linear_bootstrap_block(l_target: int) -> list[tuple[int, int, int]]:
    """Closed-form linear block used for l_target >= 8.

    For each target degree ell >= 8, use:

    - X0_a = (a, ell, ell-a),    1 <= a <= ell-1
    - X1_a = (a, ell, ell-a+1),  2 <= a <= ell-1, EXCLUDING a = (ell+1)//2
      when ell is odd. The excluded triple is (r, 2*r-1, r) at ell = 2r-1,
      which has a repeated index and parity 4r-1 (odd), so it is identically
      zero on real signals (Proposition prop:app-odd-vanishing in the paper).
    - C_a  = (a, ell-a, ell),    1 <= a <= 4 (chain rows).
    - Z_ell = {(2, ell-1, ell)} for odd ell, empty for even ell. This single
      shifted-chain row compensates for the dropped second-family entry,
      preserving the budget |T_ell| = 2*ell + 1. It has all-distinct
      indices (admissible, not contained in C since C only covers
      pairs (a, ell-a) with a in {1,2,3,4}).
    """
    block: list[tuple[int, int, int]] = []

    for a in range(1, l_target):
        block.append((a, l_target, l_target - a))

    skip_a = (l_target + 1) // 2 if l_target % 2 == 1 else None
    for a in range(2, l_target):
        if a == skip_a:
            continue
        block.append((a, l_target, l_target - a + 1))

    for a in range(1, 5):
        block.append((a, l_target - a, l_target))

    if l_target % 2 == 1:
        block.append((2, l_target - 1, l_target))

    return block


def _build_selective_index_map(lmax: int) -> list[tuple[int, int, int]]:
    """Build the O(L²) selective bispectrum index map.

    For each target degree ``l_target = 0 .. lmax``, selects up to
    ``2 * l_target + 1`` bispectral triples that provide enough equations
    to recover the SH coefficients at that degree (given all lower degrees
    are already known).

    Entry types (in priority order):

    1. **Seed block** for ``l_target <= 3``: the original low-degree
       construction, including the joint ``(2,3)`` seed.
    2. **Explicit low-degree linear blocks** for ``4 <= l_target <= 7``:
       hard-coded linear families proved separately in the appendix.
    3. **Closed-form linear block** for ``l_target >= 8``:
       all cross rows ``(a, l_target, l_target-a)`` and
       ``(a, l_target, l_target-a+1)``, plus four chain rows
       ``(a, l_target-a, l_target)`` for ``a = 1,2,3,4``.
    4. **Power** ``(0, l_target, l_target)`` — gives ``||F_{l_target}||²``
       (quadratic), used only when the linear block does not already fill
       the budget.
    5. **Self-coupling** ``(l_target, l_target, l)`` with ``l < l_target``
       — quadratic/cubic in ``F_{l_target}``, likewise only used when needed.

    Self-coupling ``(l, l, 0)`` is redundant with the power entry for
    all real signals: ``beta_{l,l,0} = (-1)^l beta_{0,l,l}/sqrt(2l+1)``.
    This is excluded at ``l_target = 1`` (where it is the only
    self-coupling candidate) and at ``l_target = 2`` (where
    ``beta_{2,2,1}`` also vanishes by CG antisymmetry). At
    ``l_target = 2`` the symmetric self-coupling ``beta_{2,2,2}`` is
    added instead—it lives in ``Sym^2(V_2)`` and is generically nonzero.

    At ``l_target = 2``, the cross entry ``(1,2,1)`` is also excluded:
    after gauge-fixing ``F_1 = (0,c,0)``, it collapses to a scalar
    multiple of the chain entry ``(1,1,2)`` (both proportional to
    ``c² a_2^0``). To compensate, ``l_target = 4`` keeps all 10
    chain+cross candidates instead of the usual 9, making that system
    overdetermined and providing a compatibility constraint that
    resolves the seed ambiguity.

    After the budget-limited selection, **all even self-coupling entries**
    ``(l_target, l_target, l)`` with ``l`` even, ``2 ≤ l ≤ l_target``,
    are appended unconditionally (they are needed for global injectivity,
    as shown by computational fiber analysis at ``L = 4``).  This adds
    ``O(L)`` entries per degree, preserving the ``Θ(L²)`` total.

    The total augmented output (bispectral + CG power) is ``Θ(L²)``; see
    ``_build_cg_power_index_map`` for the CG power complement.
    """
    index_map: list[tuple[int, int, int]] = []

    for l_target in range(lmax + 1):
        budget = 2 * l_target + 1
        if l_target == 4:
            budget = 10

        if l_target == 0:
            index_map.append((0, 0, 0))
            continue

        candidates: list[tuple[int, int, int]] = []

        if 4 <= l_target <= 7:
            candidates.extend(_small_linear_bootstrap_block(l_target))
        elif l_target >= 8:
            candidates.extend(_proved_linear_bootstrap_block(l_target))
        else:
            # Low-degree seed logic.
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
            if l_target != 2:
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

            candidates.extend(chain_mixed)
            candidates.extend(cross_all)
            candidates.extend(chain_self)

        # 3. Power entry: (0, l_target, l_target).
        candidates.append((0, l_target, l_target))

        # 4. Self-coupling: (l_target, l_target, l) with 0 <= l < l_target.
        #    Skipped at l_target in {1, 2}: beta_{l,l,0} is always redundant
        #    with the power entry; at l=2, beta_{2,2,1} also vanishes (CG
        #    antisymmetry).
        if l_target not in (1, 2):
            for l_val in range(l_target - 1, -1, -1):
                candidates.append((l_target, l_target, l_val))

        # 5. At l_target=2, add the symmetric self-coupling (2,2,2).
        if l_target == 2:
            candidates.append((2, 2, 2))

        # Deduplicate preserving priority order, take up to budget.
        seen: set[tuple[int, int, int]] = set()
        selected: list[tuple[int, int, int]] = []
        for entry in candidates:
            if entry not in seen:
                seen.add(entry)
                selected.append(entry)
                if len(selected) == budget:
                    break

        # Mandatory even self-coupling entries β(ℓ,ℓ,l) for global
        # injectivity.  Odd-l entries vanish by the exchange symmetry
        # of F_ℓ ⊗ F_ℓ; l=0 is redundant with the power entry.
        if l_target >= 3:
            for l_sc in range(2, l_target + 1, 2):
                sc_entry = (l_target, l_target, l_sc)
                if sc_entry not in seen:
                    seen.add(sc_entry)
                    selected.append(sc_entry)

        index_map.extend(selected)

    return index_map


def _build_cg_power_index_map(lmax: int) -> list[tuple[int, int, int]]:
    """Build the CG power spectrum augmentation entries.

    Returns triples (l1, l2, l_out) representing
    P_{l1,l2,l_out} = ||(F_{l1} ⊗ F_{l2})|_{l_out}||^2,
    the degree-4 SO(3)-invariant entries that complement the scalar
    bispectrum for real signals.

    The entries follow the pattern discovered by verified greedy
    augmentation (lmax 4--8) and extended systematically:

    - l1=1, l2>=2: 1 entry for l2<=3, 2 entries for l2>=4
    - l1=2, l2>=3: 2 entries per l2
    - l1=3, l2=3: self-coupling (3,3,2)
    - l1=3, l2>=4: 1 entry for even l2, 3 entries for odd l2>=5
    """
    if lmax < 2:
        return []

    entries: list[tuple[int, int, int]] = []

    for l2 in range(2, lmax + 1):
        lo_min = l2 - 1
        lo_max = min(l2 + 1, lmax)
        entries.append((1, l2, lo_min))
        if l2 >= 4 and lo_min + 1 <= lo_max:
            entries.append((1, l2, lo_min + 1))

    for l2 in range(3, lmax + 1):
        lo_min = l2 - 2
        lo_max = min(l2 + 2, lmax)
        entries.append((2, l2, lo_min))
        if lo_min + 1 <= lo_max:
            entries.append((2, l2, lo_min + 1))

    if lmax >= 3:
        entries.append((3, 3, 2))

    for l2 in range(4, lmax + 1):
        lo_min = l2 - 3
        lo_max = min(l2 + 3, lmax)
        entries.append((3, l2, lo_min))
        if l2 % 2 == 1 and l2 >= 5:
            if lo_min + 1 <= lo_max:
                entries.append((3, l2, lo_min + 1))
            if lo_min + 2 <= lo_max:
                entries.append((3, l2, lo_min + 2))

    return entries


class SO3onS2(nn.Module):
    """Bispectrum of SO(3) acting on S^2.

    Takes a real-valued signal on the sphere f(theta, phi) sampled on an
    equiangular grid, computes the spherical harmonic transform internally,
    and returns the bispectrum coefficients.

    When ``selective=True``, outputs O(L²) augmented selective entries
    (scalar bispectral + CG power spectrum) that form a complete
    SO(3)-invariant for generic real signals when lmax ≥ 4
    (see ``proof_completeness.tex``).  When ``selective=False``, computes
    all O(L³) scalar bispectral entries.

    Args:
        lmax: Maximum spherical harmonic degree.
        nlat: Number of latitude grid points.
        nlon: Number of longitude grid points.
        selective: If True, use the O(L²) augmented selective bispectrum.
            If False, compute all O(L³) entries.
    """

    def __init__(
        self,
        lmax: int = 5,
        nlat: int = 64,
        nlon: int = 128,
        selective: bool = True,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.nlat = nlat
        self.nlon = nlon
        self.selective = selective

        sht_lmax = lmax + 1
        self._sht = RealSHT(
            nlat, nlon, lmax=sht_lmax, mmax=sht_lmax, grid='equiangular', norm='ortho'
        )

        if selective:
            self._index_map = _build_selective_index_map(lmax)
            self._cg_power_map = _build_cg_power_index_map(lmax)
        else:
            cg_data = load_cg_matrices(lmax)
            self._index_map = _build_full_index_map(lmax, cg_data)
            self._cg_power_map = []

        all_triples = list(self._index_map) + list(self._cg_power_map)

        self._has_dense = False
        self._has_sparse = False

        if selective:
            self._build_sparse_tables(all_triples)
        else:
            self._build_group_tables(cg_data, all_triples)

    def _build_sparse_tables(self, all_triples: list[tuple[int, int, int]]) -> None:
        """Build sparse CG representation for the forward pass.

        Instead of dense (2l1+1)(2l2+1) × cg_cols matrices per (l1,l2) group,
        stores only the nonzero CG coefficients per entry, exploiting the
        selection rule m1 + m2 = m. This reduces storage from O(L^5) to O(L^3)
        and makes init feasible at high lmax.

        Packs everything into contiguous buffers:
        - _sparse_cg_vals: float64 flat array of all nonzero CG values
        - _sparse_m1_idx: int32 flat array of m1 offsets into f_coeffs[l1]
        - _sparse_m_idx: int32 flat array of coupled m offsets
        - _sparse_entry_offsets: int64 [n_entries+1] CSR-style offsets
        - _sparse_entry_meta: int32 [n_entries, 4] — (l1, l2, l_val, is_power)
        """
        n_bispec = len(self._index_map)
        entries_for_cg: list[tuple[int, int, int, int, bool]] = []
        for out_idx, (l1, l2, l_val) in enumerate(all_triples):
            is_power = out_idx >= n_bispec
            entries_for_cg.append((out_idx, l1, l2, l_val, is_power))

        cache_path = _CACHE_DIR / f'sparse_cg_lmax{self.lmax}_selective.pt'
        cached = self._load_sparse_cache(cache_path, entries_for_cg)
        if cached is not None:
            cg_vals_t, m1_idx_t, m_idx_t, offsets_t, entry_meta = cached
        else:
            sparse_results = compute_sparse_cg_parallel(entries_for_cg)

            all_m1: list[int] = []
            all_m: list[int] = []
            all_cg: list[float] = []
            offsets = [0]
            entry_meta: list[list[int]] = []

            for (_out_idx, l1, l2, l_val, is_power), (m1_idx, m_idx, cg_vals) in zip(
                entries_for_cg, sparse_results, strict=False
            ):
                all_m1.extend(m1_idx.tolist())
                all_m.extend(m_idx.tolist())
                all_cg.extend(cg_vals.tolist())
                offsets.append(len(all_cg))
                entry_meta.append([l1, l2, l_val, int(is_power)])

            cg_vals_t = torch.tensor(all_cg, dtype=torch.float64)
            m1_idx_t = torch.tensor(all_m1, dtype=torch.int32)
            m_idx_t = torch.tensor(all_m, dtype=torch.int32)
            offsets_t = torch.tensor(offsets, dtype=torch.int64)

            self._save_sparse_cache(
                cache_path, cg_vals_t, m1_idx_t, m_idx_t, offsets_t, entry_meta
            )

        self.register_buffer('_sparse_cg_vals', cg_vals_t)
        self.register_buffer('_sparse_m1_idx', m1_idx_t)
        self.register_buffer('_sparse_m_idx', m_idx_t)
        self.register_buffer('_sparse_entry_offsets', offsets_t)
        self._sparse_entry_meta = entry_meta
        self._has_sparse = True

        logger.info(
            'Built sparse CG: %d entries, %d nonzero coeffs (%.1f MB)',
            len(entry_meta),
            cg_vals_t.numel(),
            cg_vals_t.numel() * 8 / 1e6,
        )

    @staticmethod
    def _save_sparse_cache(
        path: Path,
        cg_vals: torch.Tensor,
        m1_idx: torch.Tensor,
        m_idx: torch.Tensor,
        offsets: torch.Tensor,
        entry_meta: list[list[int]],
    ) -> None:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    'cg_vals': cg_vals,
                    'm1_idx': m1_idx,
                    'm_idx': m_idx,
                    'offsets': offsets,
                    'entry_meta': entry_meta,
                },
                path,
            )
            logger.info('Saved sparse CG cache to %s', path)
        except OSError:
            pass

    @staticmethod
    def _load_sparse_cache(
        path: Path,
        entries_for_cg: list[tuple[int, int, int, int, bool]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]]] | None:
        if not path.exists():
            return None
        try:
            data = torch.load(path, weights_only=True)
            cached_meta = data['entry_meta']
            if len(cached_meta) != len(entries_for_cg):
                return None
            return data['cg_vals'], data['m1_idx'], data['m_idx'], data['offsets'], cached_meta
        except (OSError, RuntimeError, KeyError):
            return None

    def _build_group_tables(
        self,
        cg_data: dict[tuple[int, int], torch.Tensor] | None,
        all_triples: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Precompute per-(l1, l2) group tables with reduced CG matrices.

        Handles both bispectral entries and CG power entries. Each entry
        in ``all_triples`` corresponds to an output index; the first
        ``len(self._index_map)`` are bispectral, the rest are CG power.
        """
        if all_triples is None:
            all_triples = list(self._index_map)

        n_bispec = len(self._index_map)

        groups: OrderedDict[tuple[int, int], list[tuple[int, int, int, int, bool]]] = OrderedDict()
        for out_idx, (l1, l2, l_val) in enumerate(all_triples):
            key = (l1, l2)
            n_p, _ = _compute_padding_indices(l1, l2, l_val)
            size_l = 2 * l_val + 1
            is_power = out_idx >= n_bispec
            groups.setdefault(key, []).append((l_val, out_idx, n_p, size_l, is_power))

        group_meta: list[
            tuple[
                int,
                int,
                list[tuple[int, int, int, int, bool]],
                list[int],
                list[int],
            ]
        ] = []
        for (l1, l2), entries in groups.items():
            entries_sorted = sorted(entries, key=lambda e: e[0])

            unique_lvals: list[int] = []
            lval_offset: dict[int, tuple[int, int]] = {}
            col_indices: list[int] = []
            offset = 0
            for l_val, _, n_p, size_l, _ in entries_sorted:
                if l_val not in lval_offset:
                    col_indices.extend(range(n_p, n_p + size_l))
                    lval_offset[l_val] = (offset, size_l)
                    unique_lvals.append(l_val)
                    offset += size_l

            extract_entries: list[tuple[int, int, int, int, bool]] = []
            for l_val, out_idx, _n_p, _size_l, is_power in entries_sorted:
                off, sz = lval_offset[l_val]
                extract_entries.append((out_idx, off, sz, l_val, is_power))

            group_meta.append((l1, l2, extract_entries, col_indices, unique_lvals))

        if cg_data is not None:
            reduced_cgs = {}
            for gid, (l1, l2, _, col_indices, _) in enumerate(group_meta):
                reduced_cgs[gid] = cg_data[(l1, l2)][:, col_indices]
        else:
            reduced_cgs = self._load_or_compute_reduced_cg(group_meta)

        self._group_data: list[tuple[int, int, int, list[tuple[int, int, int, int, bool]]]] = []
        for gid, (l1, l2, extract_entries, _col_indices, _) in enumerate(group_meta):
            cg_red = reduced_cgs[gid]
            c = cg_red.shape[1]
            self.register_buffer(f'_cg_red_{gid}', cg_red)
            self._group_data.append((l1, l2, c, extract_entries))
        self._has_dense = True

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
        """Compute the augmented SO(3)-bispectrum of a signal on S^2.

        Dispatch priority (inference on CUDA, i.e. ``torch.no_grad()``):
          1. **CUDA Graph** — replays a captured graph of the sparse or
             Python-loop computation, eliminating kernel-launch overhead.
             Works at any lmax and for a fixed batch size.
          2. **Sparse** — vectorized gather-multiply-scatter using
             precomputed sparse CG indices. Default for selective=True.
          3. **Python loop** — always-available fallback for selective=False,
             also used when gradients are required (training).

        Args:
            f: Real-valued signal on S^2. Shape: (batch, nlat, nlon).

        Returns:
            Bispectrum tensor. Shape: (batch, output_size). Bispectral
            entries are complex, CG power entries are real (stored with
            zero imaginary part).
        """
        batch_size = f.shape[0]
        num_entries = self.output_size

        if num_entries == 0:
            coeffs = self._sht(f)
            return torch.zeros(batch_size, 0, dtype=coeffs.dtype, device=f.device)

        if f.is_cuda and not torch.is_grad_enabled():
            graph_result = self._forward_cuda_graph(f, batch_size, num_entries)
            if graph_result is not None:
                return graph_result

        coeffs = self._sht(f)

        if self._has_sparse:
            return self._forward_sparse_fast(coeffs, batch_size, num_entries)

        f_coeffs = _get_full_sh_coefficients(coeffs)
        return self._forward_python(f_coeffs, coeffs.dtype, f.device, batch_size, num_entries)

    def _forward_cuda_graph(
        self,
        f: torch.Tensor,
        batch_size: int,
        num_entries: int,
    ) -> torch.Tensor | None:
        """Try the CUDA-graph path.

        Returns None if capture fails or is skipped.
        """
        cache = getattr(self, '_cuda_graph_cache', None)
        if cache is not None and cache['batch_size'] == batch_size:
            cache['static_input'].copy_(f)
            cache['graph'].replay()
            return cache['static_output'].clone()

        if cache is not None and cache['batch_size'] != batch_size:
            return None

        try:
            static_input = torch.empty_like(f)
            static_input.copy_(f)

            if self._has_sparse:

                def _run_fwd(inp: torch.Tensor) -> torch.Tensor:
                    coeffs = self._sht(inp)
                    return self._forward_sparse_fast(coeffs, batch_size, num_entries)
            else:

                def _run_fwd(inp: torch.Tensor) -> torch.Tensor:
                    coeffs = self._sht(inp)
                    f_coeffs = _get_full_sh_coefficients(coeffs)
                    return self._forward_python(
                        f_coeffs, coeffs.dtype, inp.device, batch_size, num_entries
                    )

            # Warmup run (required before capture to initialize cuDNN/cuBLAS plans)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                _ = _run_fwd(static_input)
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_output = _run_fwd(static_input)

            self._cuda_graph_cache = {
                'graph': g,
                'static_input': static_input,
                'static_output': static_output,
                'batch_size': batch_size,
            }

            static_input.copy_(f)
            g.replay()
            return static_output.clone()

        except Exception:
            logger.debug('CUDA graph capture failed, disabling for this module', exc_info=True)
            self._cuda_graph_cache = {'batch_size': -1}
            return None

    def _forward_python(
        self,
        f_coeffs: dict[int, torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
        num_entries: int,
    ) -> torch.Tensor:
        """Python-loop fallback for the bispectrum forward pass."""
        result = torch.zeros(batch_size, num_entries, dtype=dtype, device=device)

        for gid, (l1, l2, _c, extract_entries) in enumerate(self._group_data):
            fl1 = f_coeffs[l1]
            fl2 = f_coeffs[l2]

            tp = torch.einsum('bi,bj->bij', fl1, fl2).reshape(batch_size, -1)
            cg = getattr(self, f'_cg_red_{gid}')
            cg = cg.to(dtype=tp.dtype, device=tp.device)
            transformed = tp @ cg

            for out_idx, offset, size_l, l_val, is_power in extract_entries:
                block = transformed[:, offset : offset + size_l]
                if is_power:
                    result[:, out_idx] = torch.sum(block.real**2 + block.imag**2, dim=-1).to(
                        result.dtype
                    )
                else:
                    result[:, out_idx] = torch.sum(block * torch.conj(f_coeffs[l_val]), dim=-1)

        return result

    def _build_sparse_global_indices(self) -> None:
        """Convert per-entry (l, m) indices to global flat-array indices.

        Packs all SH coefficients into a single flat array per batch element,
        then precomputes absolute indices for vectorized gather-multiply-sum.

        Also precomputes:
        - Bispec-only element indices (avoids recomputing mask every call)
        - SHT→flat coefficient mapping (avoids _get_full_sh_coefficients)
        """
        lmax = self.lmax
        coeff_offsets: dict[int, int] = {}
        off = 0
        for l_val in range(lmax + 1):
            coeff_offsets[l_val] = off
            off += 2 * l_val + 1
        self._sparse_total_coeffs = off

        offsets = self._sparse_entry_offsets
        meta = self._sparse_entry_meta
        m1_idx = self._sparse_m1_idx
        m_idx = self._sparse_m_idx
        N = int(offsets[-1].item())

        fl1_abs = torch.zeros(N, dtype=torch.int64)
        fl2_abs = torch.zeros(N, dtype=torch.int64)
        fl_abs = torch.zeros(N, dtype=torch.int64)
        entry_ids = torch.zeros(N, dtype=torch.int64)
        is_power_mask = torch.zeros(len(meta), dtype=torch.bool)

        for eidx, (l1, l2, l_val, is_power) in enumerate(meta):
            start = int(offsets[eidx].item())
            end = int(offsets[eidx + 1].item())
            if start == end:
                continue

            e_m1 = m1_idx[start:end].long()
            e_m = m_idx[start:end].long()

            fl1_abs[start:end] = coeff_offsets[l1] + e_m1
            m2 = e_m - l_val - e_m1 + l1 + l2
            fl2_abs[start:end] = coeff_offsets[l2] + m2
            fl_abs[start:end] = coeff_offsets[l_val] + e_m
            entry_ids[start:end] = eidx
            is_power_mask[eidx] = bool(is_power)

        self.register_buffer('_sparse_fl1_abs', fl1_abs)
        self.register_buffer('_sparse_fl2_abs', fl2_abs)
        self.register_buffer('_sparse_fl_abs', fl_abs)
        self.register_buffer('_sparse_entry_ids', entry_ids)
        self.register_buffer('_sparse_is_power', is_power_mask)

        bispec_elem_mask = ~is_power_mask[entry_ids]
        self.register_buffer('_sparse_bispec_idx', torch.where(bispec_elem_mask)[0])

        self._build_sht_to_flat_map(lmax, off)

    def _build_sht_to_flat_map(self, lmax: int, total_coeffs: int) -> None:
        """Precompute index/sign arrays to go from SHT half-spectrum to flat complex coefficients.

        SHT output shape: (batch, lmax+1, mmax+1) with m >= 0 only.
        Flat target shape: (batch, total_coeffs) with m in [-l..l] for each l.

        For each flat position, stores (l_idx_in_sht, m_abs_idx_in_sht) and a sign
        for the (-1)^m conjugation on negative-m entries.
        """
        src_l = []
        src_m = []
        conj_sign = []

        for l_val in range(lmax + 1):
            for m in range(-l_val, l_val + 1):
                src_l.append(l_val)
                src_m.append(abs(m))
                if m < 0:
                    conj_sign.append((-1.0) ** (-m))
                else:
                    conj_sign.append(0.0)

        self.register_buffer('_flat_src_l', torch.tensor(src_l, dtype=torch.long))
        self.register_buffer('_flat_src_m', torch.tensor(src_m, dtype=torch.long))
        self.register_buffer('_flat_conj_sign', torch.tensor(conj_sign, dtype=torch.float64))

    def _sht_to_flat(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Convert SHT half-spectrum to flat complex coefficients using precomputed maps.

        Replaces _get_full_sh_coefficients + torch.cat with a single gather + in-place conjugation.
        """
        if not hasattr(self, '_flat_src_l'):
            self._build_sparse_global_indices()

        dev = coeffs.device
        cache_key = (coeffs.dtype, dev)
        if getattr(self, '_sht_flat_cache_key', None) != cache_key:
            self._sht_src_l = self._flat_src_l.to(device=dev)
            self._sht_src_m = self._flat_src_m.to(device=dev)
            conj = self._flat_conj_sign.to(dtype=coeffs.dtype, device=dev)
            neg_idx = torch.where(conj != 0)[0]
            self._sht_neg_idx = neg_idx
            self._sht_conj_vals = conj[neg_idx]
            self._sht_flat_cache_key = cache_key

        flat = coeffs[:, self._sht_src_l, self._sht_src_m]
        idx = self._sht_neg_idx
        flat[:, idx] = self._sht_conj_vals.unsqueeze(0) * torch.conj(flat[:, idx])
        return flat

    def _forward_sparse(
        self,
        f_coeffs: dict[int, torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
        num_entries: int,
    ) -> torch.Tensor:
        """Sparse-CG forward pass — fully vectorized with global gather."""
        if not hasattr(self, '_sparse_fl1_abs'):
            self._build_sparse_global_indices()

        flat = torch.cat([f_coeffs[l] for l in range(self.lmax + 1)], dim=1).to(
            dtype=dtype, device=device
        )
        return self._forward_sparse_from_flat(flat, dtype, device, batch_size, num_entries)

    def _forward_sparse_fast(
        self,
        coeffs: torch.Tensor,
        batch_size: int,
        num_entries: int,
    ) -> torch.Tensor:
        """Fast sparse forward: SHT coefficients → bispectrum, no dict intermediary."""
        if not hasattr(self, '_sparse_fl1_abs'):
            self._build_sparse_global_indices()

        flat = self._sht_to_flat(coeffs)
        return self._forward_sparse_from_flat(
            flat, coeffs.dtype, coeffs.device, batch_size, num_entries
        )

    def _ensure_precomputed_on_device(self, device: torch.device, dtype: torch.dtype) -> None:
        """Lazily cache precomputed bispec and power index arrays in working dtype."""
        if not hasattr(self, '_sparse_fl1_abs'):
            self._build_sparse_global_indices()

        cache_key = (device, dtype)
        if getattr(self, '_sparse_device_cache_key', None) == cache_key:
            return

        bi = self._sparse_bispec_idx
        fl1_i = self._sparse_fl1_abs
        fl2_i = self._sparse_fl2_abs
        fl_i = self._sparse_fl_abs
        eid = self._sparse_entry_ids

        perm = torch.randperm(bi.numel())
        bi_perm = bi[perm]
        self._sc_bi_fl1 = fl1_i[bi_perm].to(device=device)
        self._sc_bi_fl2 = fl2_i[bi_perm].to(device=device)
        self._sc_bi_fl = fl_i[bi_perm].to(device=device)
        self._sc_bi_eid = eid[bi_perm].to(device=device)
        self._sc_bi_cg = self._sparse_cg_vals[bi_perm].to(dtype=dtype, device=device)

        is_power = self._sparse_is_power
        offsets = self._sparse_entry_offsets
        power_indices = torch.where(is_power)[0].tolist()
        self._sc_pw_n_rows = 0
        if power_indices:
            max_coupled = max(2 * self._sparse_entry_meta[i][2] + 1 for i in power_indices)
            pw_fl1_parts: list[torch.Tensor] = []
            pw_fl2_parts: list[torch.Tensor] = []
            pw_cg_parts: list[torch.Tensor] = []
            pw_cidx_parts: list[torch.Tensor] = []
            pw_eids: list[int] = []
            row = 0
            for eidx in power_indices:
                start = int(offsets[eidx].item())
                end = int(offsets[eidx + 1].item())
                if start == end:
                    continue
                pw_fl1_parts.append(fl1_i[start:end])
                pw_fl2_parts.append(fl2_i[start:end])
                pw_cg_parts.append(self._sparse_cg_vals[start:end])
                pw_cidx_parts.append(row * max_coupled + self._sparse_m_idx[start:end].long())
                pw_eids.append(eidx)
                row += 1

            if pw_fl1_parts:
                self._sc_pw_fl1 = torch.cat(pw_fl1_parts).to(device=device)
                self._sc_pw_fl2 = torch.cat(pw_fl2_parts).to(device=device)
                self._sc_pw_cg = torch.cat(pw_cg_parts).to(dtype=dtype, device=device)
                self._sc_pw_cidx = torch.cat(pw_cidx_parts).to(device=device)
                self._sc_pw_eids = torch.tensor(pw_eids, dtype=torch.long, device=device)
                self._sc_pw_n_rows = row
                self._sc_pw_max_coupled = max_coupled

        self._sparse_device_cache_key = cache_key

    def _forward_sparse_from_flat(
        self,
        flat: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
        num_entries: int,
    ) -> torch.Tensor:
        """Core sparse computation from flat coefficient array."""
        self._ensure_precomputed_on_device(device, dtype)

        result = torch.zeros(batch_size, num_entries, dtype=dtype, device=device)
        bi_fl1 = self._sc_bi_fl1
        if bi_fl1.numel() > 0:
            prods = (
                flat[:, bi_fl1]
                * flat[:, self._sc_bi_fl2]
                * self._sc_bi_cg.unsqueeze(0)
                * torch.conj(flat[:, self._sc_bi_fl])
            )
            result.scatter_add_(1, self._sc_bi_eid.unsqueeze(0).expand(batch_size, -1), prods)

        if self._sc_pw_n_rows > 0:
            n_pw = self._sc_pw_n_rows
            mc = self._sc_pw_max_coupled
            pw_prods = (
                flat[:, self._sc_pw_fl1] * flat[:, self._sc_pw_fl2] * self._sc_pw_cg.unsqueeze(0)
            )
            coupled_flat = torch.zeros(batch_size, n_pw * mc, dtype=dtype, device=device)
            coupled_flat.scatter_add_(
                1, self._sc_pw_cidx.unsqueeze(0).expand(batch_size, -1), pw_prods
            )
            coupled_2d = coupled_flat.reshape(batch_size, n_pw, mc)
            norms = torch.sum(coupled_2d.real**2 + coupled_2d.imag**2, dim=-1).to(dtype)
            result[:, self._sc_pw_eids] = norms

        return result

    def reset_cuda_graph_cache(self) -> None:
        """Invalidate the cached CUDA graph (e.g. after changing batch size)."""
        if hasattr(self, '_cuda_graph_cache'):
            del self._cuda_graph_cache

    def invert(self, beta: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Inversion is an open problem for SO(3) on S^2.

        Raises:
            NotImplementedError: Selective bispectrum and inversion for
                continuous groups remain open mathematical problems.
        """
        raise NotImplementedError('Inversion for SO(3) on S^2 is an open mathematical problem.')

    @property
    def output_size(self) -> int:
        """Total number of output entries (bispectral + CG power)."""
        return len(self._index_map) + len(self._cg_power_map)

    @property
    def n_bispec(self) -> int:
        """Number of scalar bispectral entries."""
        return len(self._index_map)

    @property
    def n_cg_power(self) -> int:
        """Number of CG power spectrum entries."""
        return len(self._cg_power_map)

    @property
    def index_map(self) -> list[tuple[int, int, int]]:
        """Maps flat output index -> (l1, l2, l) triple.

        First ``n_bispec`` entries are bispectral, remaining are CG power.
        """
        return list(self._index_map) + list(self._cg_power_map)

    @property
    def cg_power_map(self) -> list[tuple[int, int, int]]:
        """Maps CG power output index -> (l1, l2, l_out) triple."""
        return list(self._cg_power_map)

    def extra_repr(self) -> str:
        parts = [
            f'lmax={self.lmax}, nlat={self.nlat}, nlon={self.nlon}',
            f'selective={self.selective}',
            f'output_size={self.output_size}',
        ]
        if self.selective:
            parts.append(f'n_bispec={self.n_bispec}, n_cg_power={self.n_cg_power}')
        return ', '.join(parts)


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
