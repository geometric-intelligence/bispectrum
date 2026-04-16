"""Clebsch-Gordan coefficient utilities for SO(3).

Computes CG matrices analytically via the Wigner 3j symbol (Racah formula). Also supports loading
precomputed matrices from the bundled JSON file for validation.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

_DATA_PATH = Path(__file__).parent / 'data' / 'cg_lmax5.json'

# ---------------------------------------------------------------------------
# Log-factorial table (precomputed up to a generous ceiling; extended lazily)
# ---------------------------------------------------------------------------
_LOG_FACT: list[float] = [0.0]  # _LOG_FACT[n] = log(n!)


def _ensure_log_fact(n: int) -> None:
    """Extend the log-factorial table so that _LOG_FACT[n] is available."""
    while len(_LOG_FACT) <= n:
        _LOG_FACT.append(_LOG_FACT[-1] + math.log(len(_LOG_FACT)))


# ---------------------------------------------------------------------------
# Wigner 3j symbol  (j1 j2 j3 ; m1 m2 m3)
# ---------------------------------------------------------------------------


def wigner3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    r"""Compute the Wigner 3j symbol using the Racah formula.

    .. math::

        \begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix}

    Uses log-factorials for numerical stability at large j.

    Returns 0.0 when selection rules are violated.
    """
    # --- Selection rules ---
    if m1 + m2 + m3 != 0:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0
    # j1, j2, j3 must satisfy the triangle inequality (already implied above)
    # and j1+j2+j3 must be an integer (always true for integer j).

    # --- Racah formula ---
    # Pre-compute needed factorials.
    max_fact = max(
        j1 + j2 + j3 + 1,
        j1 + m1,
        j1 - m1,
        j2 + m2,
        j2 - m2,
        j3 + m3,
        j3 - m3,
        j1 + j2 - j3,
        j1 - j2 + j3,
        -j1 + j2 + j3,
    )
    _ensure_log_fact(max_fact + 1)

    # Triangle coefficient (log scale).
    log_tri = (
        _LOG_FACT[j1 + j2 - j3]
        + _LOG_FACT[j1 - j2 + j3]
        + _LOG_FACT[-j1 + j2 + j3]
        - _LOG_FACT[j1 + j2 + j3 + 1]
    )

    # Prefactor (log scale).
    log_pre = 0.5 * log_tri + 0.5 * (
        _LOG_FACT[j1 + m1]
        + _LOG_FACT[j1 - m1]
        + _LOG_FACT[j2 + m2]
        + _LOG_FACT[j2 - m2]
        + _LOG_FACT[j3 + m3]
        + _LOG_FACT[j3 - m3]
    )

    # Sum over t (Racah's sum).
    t_min = max(0, j2 - j3 - m1, j1 - j3 + m2)
    t_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)

    _ensure_log_fact(t_max + 1)
    for t in range(t_min, t_max + 1):
        n = max(
            j1 + j2 - j3 - t,
            j1 - m1 - t,
            j2 + m2 - t,
            j3 - j2 + m1 + t,
            j3 - j1 - m2 + t,
        )
        _ensure_log_fact(n + 1)

    total = 0.0
    for t in range(t_min, t_max + 1):
        log_denom = (
            _LOG_FACT[t]
            + _LOG_FACT[j1 + j2 - j3 - t]
            + _LOG_FACT[j1 - m1 - t]
            + _LOG_FACT[j2 + m2 - t]
            + _LOG_FACT[j3 - j2 + m1 + t]
            + _LOG_FACT[j3 - j1 - m2 + t]
        )
        sign = (-1) ** t
        total += sign * math.exp(log_pre - log_denom)

    phase = (-1) ** (j1 - j2 - m3)
    return phase * total


# ---------------------------------------------------------------------------
# CG coefficient  <l1 m1; l2 m2 | l m>
# ---------------------------------------------------------------------------


def clebsch_gordan(l1: int, m1: int, l2: int, m2: int, l: int, m: int) -> float:
    """Compute a single Clebsch-Gordan coefficient.

    Related to the Wigner 3j symbol by::

        <l1 m1; l2 m2 | l m> = (-1)^{l1-l2+m} sqrt(2l+1) * (l1 l2 l; m1 m2 -m)
    """
    return ((-1) ** (l1 - l2 + m)) * math.sqrt(2 * l + 1) * wigner3j(l1, l2, l, m1, m2, -m)


# ---------------------------------------------------------------------------
# Full CG matrix for a given (l1, l2) pair
# ---------------------------------------------------------------------------


def compute_cg_matrix(l1: int, l2: int) -> torch.Tensor:
    """Compute the unitary CG matrix for the tensor product ρ_{l1} ⊗ ρ_{l2}.

    The matrix transforms from the uncoupled basis |l1,m1⟩ ⊗ |l2,m2⟩
    (m1 varying slowest, m2 varying fastest) to the coupled basis |l,m⟩
    (blocks ordered by l = |l1-l2| .. l1+l2).

    Returns:
        Tensor of shape ((2l1+1)(2l2+1), (2l1+1)(2l2+1)), dtype float64.
    """
    d1 = 2 * l1 + 1
    d2 = 2 * l2 + 1
    D = d1 * d2

    # Ensure the log-factorial table is large enough for all 3j calls.
    _ensure_log_fact(l1 + l2 + max(l1, l2) + 2)

    C = torch.zeros(D, D, dtype=torch.float64)

    col = 0
    for l_val in range(abs(l1 - l2), l1 + l2 + 1):
        for m in range(-l_val, l_val + 1):
            for im1, m1 in enumerate(range(-l1, l1 + 1)):
                m2 = m - m1
                if abs(m2) > l2:
                    continue
                im2 = m2 + l2
                row = im1 * d2 + im2
                C[row, col] = clebsch_gordan(l1, m1, l2, m2, l_val, m)
            col += 1

    return C


def _compute_cg_matrix_fast(l1: int, l2: int) -> torch.Tensor:
    """Fast CG matrix computation using precomputed log-factorial table.

    Same result as ``compute_cg_matrix`` but avoids per-call overhead in
    the Racah summation by inlining the hot loop.
    """
    d1 = 2 * l1 + 1
    d2 = 2 * l2 + 1
    D = d1 * d2

    # Pre-grow the log-factorial table for the largest argument we'll need.
    max_n = 2 * (l1 + l2) + 2
    _ensure_log_fact(max_n)
    lf = _LOG_FACT  # local alias for speed

    C = torch.zeros(D, D, dtype=torch.float64)

    col = 0
    for l_val in range(abs(l1 - l2), l1 + l2 + 1):
        sqrt_2l1 = math.sqrt(2 * l_val + 1)
        # Triangle coefficient (log) for (l1, l2, l_val)
        log_tri = (
            lf[l1 + l2 - l_val]
            + lf[l1 - l2 + l_val]
            + lf[-l1 + l2 + l_val]
            - lf[l1 + l2 + l_val + 1]
        )
        for m in range(-l_val, l_val + 1):
            neg_m = -m
            for im1, m1 in enumerate(range(-l1, l1 + 1)):
                m2 = m - m1
                if abs(m2) > l2:
                    continue

                # --- Inline wigner3j(l1, l2, l_val, m1, m2, neg_m) ---
                # Selection rule m1+m2+neg_m = m1+m2-m = 0 is guaranteed.

                log_pre = 0.5 * log_tri + 0.5 * (
                    lf[l1 + m1]
                    + lf[l1 - m1]
                    + lf[l2 + m2]
                    + lf[l2 - m2]
                    + lf[l_val + neg_m]
                    + lf[l_val - neg_m]
                )

                t_min = max(0, l2 - l_val - m1, l1 - l_val + m2)
                t_max = min(l1 + l2 - l_val, l1 - m1, l2 + m2)

                total = 0.0
                for t in range(t_min, t_max + 1):
                    log_denom = (
                        lf[t]
                        + lf[l1 + l2 - l_val - t]
                        + lf[l1 - m1 - t]
                        + lf[l2 + m2 - t]
                        + lf[l_val - l2 + m1 + t]
                        + lf[l_val - l1 - m2 + t]
                    )
                    if t & 1:
                        total -= math.exp(log_pre - log_denom)
                    else:
                        total += math.exp(log_pre - log_denom)

                phase_3j = (-1) ** (l1 - l2 - neg_m)
                w3j = phase_3j * total

                # CG = (-1)^{l1-l2+m} * sqrt(2l+1) * w3j
                phase_cg = (-1) ** (l1 - l2 + m)
                im2 = m2 + l2
                row = im1 * d2 + im2
                C[row, col] = phase_cg * sqrt_2l1 * w3j
            col += 1

    return C


# ---------------------------------------------------------------------------
# Vectorized CG computation using numpy
# ---------------------------------------------------------------------------


def _compute_cg_columns_vectorized(l1: int, l2: int, l_vals: list[int]) -> torch.Tensor:
    """Compute reduced CG columns using vectorized numpy operations.

    For each requested l_val, computes all CG coefficients <l1,m1; l2,m2 | l,m> simultaneously
    using array operations on the log-factorial table, avoiding per-element Python loops.
    """
    d1 = 2 * l1 + 1
    d2 = 2 * l2 + 1
    d = d1 * d2
    c = sum(2 * lv + 1 for lv in l_vals)

    max_n = 2 * (l1 + l2) + 2
    _ensure_log_fact(max_n)
    lf = np.array(_LOG_FACT[: max_n + 1])

    C = np.zeros((d, c), dtype=np.float64)
    needed = set(l_vals)

    col = 0
    l_min = abs(l1 - l2)
    for l_val in range(l_min, l1 + l2 + 1):
        if l_val not in needed:
            continue

        sqrt_2l1 = math.sqrt(2 * l_val + 1)
        log_tri = (
            lf[l1 + l2 - l_val]
            + lf[l1 - l2 + l_val]
            + lf[-l1 + l2 + l_val]
            - lf[l1 + l2 + l_val + 1]
        )

        for m in range(-l_val, l_val + 1):
            neg_m = -m
            # Vectorize over m1.
            m1_arr = np.arange(-l1, l1 + 1)
            m2_arr = m - m1_arr
            valid = np.abs(m2_arr) <= l2
            if not np.any(valid):
                col += 1
                continue

            m1_v = m1_arr[valid]
            m2_v = m2_arr[valid]

            log_pre = 0.5 * log_tri + 0.5 * (
                lf[l1 + m1_v]
                + lf[l1 - m1_v]
                + lf[l2 + m2_v]
                + lf[l2 - m2_v]
                + lf[l_val + neg_m]
                + lf[l_val - neg_m]
            )

            t_min = np.maximum(0, np.maximum(l2 - l_val - m1_v, l1 - l_val + m2_v))
            t_max = np.minimum(l1 + l2 - l_val, np.minimum(l1 - m1_v, l2 + m2_v))

            # Racah sum: vectorize over m1, loop over t (typically 1-3 terms).
            max_t = int(t_max.max()) if t_max.size > 0 else 0
            min_t = int(t_min.min()) if t_min.size > 0 else 0
            total = np.zeros(len(m1_v))

            for t in range(min_t, max_t + 1):
                mask = (t >= t_min) & (t <= t_max)
                if not np.any(mask):
                    continue
                m1_t = m1_v[mask]
                m2_t = m2_v[mask]
                lp = log_pre[mask]

                log_denom = (
                    lf[t]
                    + lf[l1 + l2 - l_val - t]
                    + lf[l1 - m1_t - t]
                    + lf[l2 + m2_t - t]
                    + lf[l_val - l2 + m1_t + t]
                    + lf[l_val - l1 - m2_t + t]
                )
                vals = np.exp(lp - log_denom)
                if t & 1:
                    total[mask] -= vals
                else:
                    total[mask] += vals

            phase_3j = (-1.0) ** (l1 - l2 - neg_m)
            phase_cg = (-1.0) ** (l1 - l2 + m)
            cg_vals = phase_cg * sqrt_2l1 * phase_3j * total

            im1_v = (m1_v + l1).astype(int)
            im2_v = (m2_v + l2).astype(int)
            rows = im1_v * d2 + im2_v
            C[rows, col] = cg_vals
            col += 1

    return torch.from_numpy(C)


# ---------------------------------------------------------------------------
# Reduced CG: compute only the columns we need
# ---------------------------------------------------------------------------


def compute_cg_columns(l1: int, l2: int, l_vals: list[int]) -> torch.Tensor:
    """Compute only the columns of C_{l1,l2} corresponding to given l values.

    Instead of building the full (d, d) matrix, builds (d, c) where c is
    the total number of coupled-basis elements for the requested l values.

    Args:
        l1, l2: Tensor-product pair.
        l_vals: List of angular momentum values whose blocks are needed
                (each in range [|l1-l2|, l1+l2]).

    Returns:
        Tensor of shape ((2l1+1)(2l2+1), c), dtype float64.
    """
    d1 = 2 * l1 + 1
    d2 = 2 * l2 + 1
    d = d1 * d2

    c = sum(2 * lv + 1 for lv in l_vals)
    C = torch.zeros(d, c, dtype=torch.float64)

    max_n = 2 * (l1 + l2) + 2
    _ensure_log_fact(max_n)
    lf = _LOG_FACT

    l_min = abs(l1 - l2)
    needed = set(l_vals)

    col = 0
    for l_val in range(l_min, l1 + l2 + 1):
        if l_val not in needed:
            continue
        sqrt_2l1 = math.sqrt(2 * l_val + 1)
        log_tri = (
            lf[l1 + l2 - l_val]
            + lf[l1 - l2 + l_val]
            + lf[-l1 + l2 + l_val]
            - lf[l1 + l2 + l_val + 1]
        )
        for m in range(-l_val, l_val + 1):
            neg_m = -m
            for im1, m1 in enumerate(range(-l1, l1 + 1)):
                m2 = m - m1
                if abs(m2) > l2:
                    continue

                log_pre = 0.5 * log_tri + 0.5 * (
                    lf[l1 + m1]
                    + lf[l1 - m1]
                    + lf[l2 + m2]
                    + lf[l2 - m2]
                    + lf[l_val + neg_m]
                    + lf[l_val - neg_m]
                )

                t_min = max(0, l2 - l_val - m1, l1 - l_val + m2)
                t_max = min(l1 + l2 - l_val, l1 - m1, l2 + m2)

                total = 0.0
                for t in range(t_min, t_max + 1):
                    log_denom = (
                        lf[t]
                        + lf[l1 + l2 - l_val - t]
                        + lf[l1 - m1 - t]
                        + lf[l2 + m2 - t]
                        + lf[l_val - l2 + m1 + t]
                        + lf[l_val - l1 - m2 + t]
                    )
                    if t & 1:
                        total -= math.exp(log_pre - log_denom)
                    else:
                        total += math.exp(log_pre - log_denom)

                phase_3j = (-1) ** (l1 - l2 - neg_m)
                w3j = phase_3j * total
                phase_cg = (-1) ** (l1 - l2 + m)
                im2 = m2 + l2
                row = im1 * d2 + im2
                C[row, col] = phase_cg * sqrt_2l1 * w3j
            col += 1

    return C


# ---------------------------------------------------------------------------
# Worker for multiprocessing (must be at module level for pickling)
# ---------------------------------------------------------------------------


def _worker_compute_cg_columns(
    args: tuple[int, int, int, list[int]],
) -> tuple[int, torch.Tensor]:
    """Compute reduced CG columns for a single group.

    Returns (gid, tensor).
    """
    gid, l1, l2, l_vals = args
    _ensure_log_fact(2 * (l1 + l2) + 2)
    return gid, _compute_cg_columns_vectorized(l1, l2, l_vals)


# ---------------------------------------------------------------------------
# Public API: compute all CG matrices needed for a given lmax
# ---------------------------------------------------------------------------


def compute_cg_matrices(lmax: int) -> dict[tuple[int, int], torch.Tensor]:
    """Compute CG matrices for all (l1, l2) pairs with l1 <= l2 <= lmax.

    Returns:
        Dict mapping (l1, l2) -> CG matrix of shape ((2l1+1)(2l2+1), (2l1+1)(2l2+1)).
    """
    # Pre-grow log-factorial table for the largest argument we'll need.
    _ensure_log_fact(4 * lmax + 2)

    matrices: dict[tuple[int, int], torch.Tensor] = {}
    for l1 in range(lmax + 1):
        for l2 in range(l1, lmax + 1):
            matrices[(l1, l2)] = _compute_cg_matrix_fast(l1, l2)
    return matrices


def compute_sparse_cg_entry(
    l1: int, l2: int, l_val: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sparse CG coefficients for a single (l1, l2, l_val) triple.

    Exploits the selection rule m1 + m2 = m to store only nonzero entries.

    Returns:
        m1_indices: int32 array of m1 offsets (m1 + l1) into f_coeffs[l1].
        m_indices: int32 array of m offsets (m + l_val) indicating which
            coupled-basis column this coefficient belongs to.
        cg_values: float64 array of CG coefficient values.
        All three arrays have the same length (number of nonzero CG entries).
    """
    max_n = 2 * (l1 + l2) + 2
    _ensure_log_fact(max_n)
    lf = np.array(_LOG_FACT[: max_n + 1])

    2 * l2 + 1
    sqrt_2l1 = math.sqrt(2 * l_val + 1)
    log_tri = (
        lf[l1 + l2 - l_val] + lf[l1 - l2 + l_val] + lf[-l1 + l2 + l_val] - lf[l1 + l2 + l_val + 1]
    )

    all_m1_idx: list[int] = []
    all_m_idx: list[int] = []
    all_cg: list[float] = []

    for m in range(-l_val, l_val + 1):
        neg_m = -m
        m1_lo = max(-l1, m - l2)
        m1_hi = min(l1, m + l2)
        if m1_hi < m1_lo:
            continue

        m1_arr = np.arange(m1_lo, m1_hi + 1)
        m2_arr = m - m1_arr

        log_pre = 0.5 * log_tri + 0.5 * (
            lf[l1 + m1_arr]
            + lf[l1 - m1_arr]
            + lf[l2 + m2_arr]
            + lf[l2 - m2_arr]
            + lf[l_val + neg_m]
            + lf[l_val - neg_m]
        )

        t_min = np.maximum(0, np.maximum(l2 - l_val - m1_arr, l1 - l_val + m2_arr))
        t_max = np.minimum(l1 + l2 - l_val, np.minimum(l1 - m1_arr, l2 + m2_arr))

        max_t_val = int(t_max.max()) if t_max.size > 0 else 0
        min_t_val = int(t_min.min()) if t_min.size > 0 else 0
        total = np.zeros(len(m1_arr))

        for t in range(min_t_val, max_t_val + 1):
            mask = (t >= t_min) & (t <= t_max)
            if not np.any(mask):
                continue
            m1_t = m1_arr[mask]
            m2_t = m2_arr[mask]
            lp = log_pre[mask]
            log_denom = (
                lf[t]
                + lf[l1 + l2 - l_val - t]
                + lf[l1 - m1_t - t]
                + lf[l2 + m2_t - t]
                + lf[l_val - l2 + m1_t + t]
                + lf[l_val - l1 - m2_t + t]
            )
            vals = np.exp(lp - log_denom)
            if t & 1:
                total[mask] -= vals
            else:
                total[mask] += vals

        phase_3j = (-1.0) ** (l1 - l2 - neg_m)
        phase_cg = (-1.0) ** (l1 - l2 + m)
        cg_vals = phase_cg * sqrt_2l1 * phase_3j * total

        nonzero = np.abs(cg_vals) > 0
        if not np.any(nonzero):
            continue

        m1_nz = m1_arr[nonzero]
        cg_nz = cg_vals[nonzero]

        all_m1_idx.extend((m1_nz + l1).tolist())
        all_m_idx.extend([m + l_val] * int(nonzero.sum()))
        all_cg.extend(cg_nz.tolist())

    return (
        np.array(all_m1_idx, dtype=np.int32),
        np.array(all_m_idx, dtype=np.int32),
        np.array(all_cg, dtype=np.float64),
    )


def compute_sparse_cg_parallel(
    entries: list[tuple[int, int, int, int, bool]],
    max_workers: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute sparse CG for multiple entries in parallel.

    Args:
        entries: List of (out_idx, l1, l2, l_val, is_power) tuples.
        max_workers: Max parallel threads.

    Returns:
        List of (m1_indices, m_indices, cg_values) per entry, same order as input.
    """
    if not entries:
        return []

    max_l = max(l1 + l2 for _, l1, l2, _, _ in entries)
    _ensure_log_fact(2 * max_l + 2)

    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, 16, len(entries))

    def _compute(
        entry: tuple[int, int, int, int, bool],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, l1, l2, l_val, _ = entry
        return compute_sparse_cg_entry(l1, l2, l_val)

    if max_workers <= 1 or len(entries) <= 1:
        return [_compute(e) for e in entries]

    entries_with_idx = list(enumerate(entries))
    entries_with_idx.sort(key=lambda x: (2 * x[1][1] + 1) * (2 * x[1][2] + 1), reverse=True)

    results: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = [None] * len(entries)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_compute, e): idx for idx, e in entries_with_idx}
        for fut in futures:
            results[futures[fut]] = fut.result()

    return results  # type: ignore[return-value]


def compute_reduced_cg_parallel(
    groups: list[tuple[int, int, int, list[int]]],
    max_workers: int | None = None,
) -> dict[int, torch.Tensor]:
    """Compute reduced CG matrices for multiple groups in parallel.

    Uses ``ThreadPoolExecutor`` because the numpy-vectorized inner loop
    releases the GIL, giving true parallelism without the fork/spawn
    overhead of ``ProcessPoolExecutor``.

    Args:
        groups: List of (gid, l1, l2, l_vals) tuples.
        max_workers: Max parallel threads (defaults to CPU count, capped at 16).

    Returns:
        Dict mapping gid -> reduced CG tensor of shape (d, c).
    """
    if not groups:
        return {}

    # Pre-grow log-factorial table (shared across threads, append-only so safe).
    max_l = max(l1 + l2 for _, l1, l2, _ in groups)
    _ensure_log_fact(2 * max_l + 2)

    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, 16, len(groups))

    if max_workers <= 1 or len(groups) <= 1:
        return {gid: _compute_cg_columns_vectorized(l1, l2, lv) for gid, l1, l2, lv in groups}

    # Sort by descending workload (largest pairs first) for better load balancing.
    groups_sorted = sorted(groups, key=lambda g: (2 * g[1] + 1) * (2 * g[2] + 1), reverse=True)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_compute_cg_columns_vectorized, l1, l2, lv): gid
            for gid, l1, l2, lv in groups_sorted
        }
        result = {}
        for fut in futures:
            gid = futures[fut]
            result[gid] = fut.result()
    return result


# ---------------------------------------------------------------------------
# Disk cache for CG matrices
# ---------------------------------------------------------------------------

_CACHE_DIR = Path.home() / '.cache' / 'bispectrum'


def _cache_path(lmax: int) -> Path:
    return _CACHE_DIR / f'cg_lmax{lmax}.pt'


def _save_cache(lmax: int, matrices: dict[tuple[int, int], torch.Tensor]) -> None:
    """Persist CG matrices to disk."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Convert tuple keys to strings for torch.save compatibility.
        data = {f'{l1}_{l2}': m for (l1, l2), m in matrices.items()}
        data['__lmax__'] = torch.tensor(lmax)
        torch.save(data, _cache_path(lmax))
    except OSError:
        pass  # Non-fatal: cache write failure is silently ignored.


def _load_cache(lmax: int) -> dict[tuple[int, int], torch.Tensor] | None:
    """Load CG matrices from disk cache.

    Returns None on miss.
    """
    path = _cache_path(lmax)
    if not path.exists():
        return None
    try:
        data = torch.load(path, weights_only=True)
        if data.get('__lmax__', torch.tensor(-1)).item() != lmax:
            return None
        matrices: dict[tuple[int, int], torch.Tensor] = {}
        for key, val in data.items():
            if key == '__lmax__':
                continue
            l1, l2 = map(int, key.split('_'))
            matrices[(l1, l2)] = val
        return matrices
    except (OSError, RuntimeError, ValueError, KeyError, Exception):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_cg_matrices(lmax: int) -> dict[tuple[int, int], torch.Tensor]:
    """Compute (or load from cache) CG matrices for l1 <= l2 <= lmax.

    On first call for a given lmax, computes all matrices analytically and
    caches them to ``~/.cache/bispectrum/cg_lmax{N}.pt``.  Subsequent calls
    load directly from the cache file.

    Returns:
        Dict mapping (l1, l2) -> CG matrix of shape ((2l1+1)(2l2+1), (2l1+1)(2l2+1)).
    """
    cached = _load_cache(lmax)
    if cached is not None:
        return cached
    matrices = compute_cg_matrices(lmax)
    _save_cache(lmax, matrices)
    return matrices
