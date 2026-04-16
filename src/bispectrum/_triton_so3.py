"""Triton-fused SO(3) bispectrum forward pass.

Replaces the Python loop in SO3onS2.forward() with a fused kernel
that parallelizes over (batch, entry) pairs, eliminating thousands of
small CUDA kernel launches.

The kernel computes one bispectral or CG-power entry per program instance:
  beta_{l1,l2,l} = sum_{m} [sum_k fl1[i_k] * fl2[j_k] * CG[k, col_offset+m]] * conj(fl[m])
or for power entries:
  P_{l1,l2,l} = sum_m |sum_k fl1[i_k] * fl2[j_k] * CG[k, col_offset+m]|^2
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _bispectrum_entry_kernel(
    f_flat_ptr,
    cg_flat_ptr,
    output_ptr,
    entry_desc_ptr,
    coeff_offsets_ptr,
    f_flat_stride: tl.constexpr,
    out_stride: tl.constexpr,
    MAX_BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused bispectrum kernel. Grid: (batch, n_entries).

    Each program computes one output entry for one batch element. entry_desc layout per entry (8
    int32s):   [out_idx, l1, l2, l_val, cg_elem_off, cg_cols, block_offset, block_size, is_power]
    """
    bid = tl.program_id(0)
    eid = tl.program_id(1)

    ed_base = eid * 9
    out_idx = tl.load(entry_desc_ptr + ed_base + 0)
    l1 = tl.load(entry_desc_ptr + ed_base + 1)
    l2 = tl.load(entry_desc_ptr + ed_base + 2)
    l_val = tl.load(entry_desc_ptr + ed_base + 3)
    cg_elem_off = tl.load(entry_desc_ptr + ed_base + 4)
    cg_cols = tl.load(entry_desc_ptr + ed_base + 5)
    block_offset = tl.load(entry_desc_ptr + ed_base + 6)
    block_size = tl.load(entry_desc_ptr + ed_base + 7)
    is_power = tl.load(entry_desc_ptr + ed_base + 8)

    d1 = 2 * l1 + 1
    d2 = 2 * l2 + 1
    n_rows = d1 * d2

    fl1_off = tl.load(coeff_offsets_ptr + l1)
    fl2_off = tl.load(coeff_offsets_ptr + l2)
    fl_off = tl.load(coeff_offsets_ptr + l_val)

    f_base = bid * f_flat_stride

    idx0 = tl.arange(0, BLOCK_K)
    total_acc_r = tl.zeros([BLOCK_K], dtype=tl.float32)
    total_acc_i = tl.zeros([BLOCK_K], dtype=tl.float32)

    for col in tl.range(0, MAX_BLOCK_SIZE):
        col_idx = block_offset + col

        dot_r_acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        dot_i_acc = tl.zeros([BLOCK_K], dtype=tl.float32)

        for k_start in tl.range(0, n_rows, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < n_rows

            i_vals = k_offsets // d2
            j_vals = k_offsets % d2

            fl1_addr = f_base + (fl1_off + i_vals) * 2
            fl1_r = tl.load(f_flat_ptr + fl1_addr, mask=k_mask, other=0.0)
            fl1_i = tl.load(f_flat_ptr + fl1_addr + 1, mask=k_mask, other=0.0)
            fl2_addr = f_base + (fl2_off + j_vals) * 2
            fl2_r = tl.load(f_flat_ptr + fl2_addr, mask=k_mask, other=0.0)
            fl2_i = tl.load(f_flat_ptr + fl2_addr + 1, mask=k_mask, other=0.0)

            tp_r = fl1_r * fl2_r - fl1_i * fl2_i
            tp_i = fl1_r * fl2_i + fl1_i * fl2_r

            cg_addr = (cg_elem_off + k_offsets * cg_cols + col_idx) * 2
            cg_r = tl.load(cg_flat_ptr + cg_addr, mask=k_mask, other=0.0)
            cg_i = tl.load(cg_flat_ptr + cg_addr + 1, mask=k_mask, other=0.0)

            dot_r_acc += tp_r * cg_r - tp_i * cg_i
            dot_i_acc += tp_r * cg_i + tp_i * cg_r

        dot_r = tl.sum(dot_r_acc, axis=0)
        dot_i = tl.sum(dot_i_acc, axis=0)

        col_valid = col < block_size
        slot = tl.where(idx0 == 0, 1.0, 0.0)

        if is_power == 1:
            val = tl.where(col_valid, dot_r * dot_r + dot_i * dot_i, 0.0)
            total_acc_r += val * slot
        else:
            fl_addr = f_base + (fl_off + col) * 2
            fl_r = tl.load(f_flat_ptr + fl_addr)
            fl_i = tl.load(f_flat_ptr + fl_addr + 1)

            val_r = tl.where(col_valid, dot_r * fl_r + dot_i * fl_i, 0.0)
            val_i = tl.where(col_valid, dot_i * fl_r - dot_r * fl_i, 0.0)
            total_acc_r += val_r * slot
            total_acc_i += val_i * slot

    final_r = tl.sum(total_acc_r, axis=0)
    final_i = tl.sum(total_acc_i, axis=0)

    out_addr = bid * out_stride + out_idx
    tl.store(output_ptr + out_addr * 2, final_r)
    tl.store(output_ptr + out_addr * 2 + 1, final_i)


def build_fused_buffers(
    group_data: list[tuple[int, int, int, list[tuple[int, int, int, int, bool]]]],
    lmax: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pack per-group CG and entry metadata into flat contiguous tensors.

    Returns:
        entry_desc: int32 [n_entries_total, 9] — per-entry descriptor with all
            info needed by the kernel: (out_idx, l1, l2, l_val, cg_elem_offset,
            cg_cols, block_offset, block_size, is_power).
        coeff_offsets: int32 [lmax+2] — cumulative SH coefficient offsets.
        max_block_size: int — maximum block_size across all entries (for constexpr bound).
    """
    coeff_offsets = torch.zeros(lmax + 2, dtype=torch.int32)
    offset = 0
    for l_val in range(lmax + 1):
        coeff_offsets[l_val] = offset
        offset += 2 * l_val + 1
    coeff_offsets[lmax + 1] = offset

    entry_desc_list: list[list[int]] = []
    max_block_size = 0

    cg_elem_offset = 0
    for _gid, (l1, l2, cg_cols, extract_entries) in enumerate(group_data):
        d1 = 2 * l1 + 1
        d2 = 2 * l2 + 1
        n_rows = d1 * d2

        for out_idx, block_off, block_sz, l_val, is_power in extract_entries:
            entry_desc_list.append(
                [
                    out_idx,
                    l1,
                    l2,
                    l_val,
                    cg_elem_offset,
                    cg_cols,
                    block_off,
                    block_sz,
                    int(is_power),
                ]
            )
            max_block_size = max(max_block_size, block_sz)

        cg_elem_offset += n_rows * cg_cols

    if entry_desc_list:
        entry_desc = torch.tensor(entry_desc_list, dtype=torch.int32)
    else:
        entry_desc = torch.zeros(0, 9, dtype=torch.int32)

    return entry_desc, coeff_offsets, max(max_block_size, 1)


def flatten_cg_matrices(
    module: torch.nn.Module,
    group_data: list[tuple[int, int, int, list[tuple[int, int, int, int, bool]]]],
) -> torch.Tensor:
    """Concatenate all per-group reduced CG matrices into one flat tensor.

    Each CG matrix has shape (n_rows, cg_cols) where n_rows = (2*l1+1)*(2*l2+1). They are
    concatenated along the element dimension (row-major within each group).

    Returns a flat real tensor with interleaved real/imag pairs. CG matrices are typically real-
    valued; zero imaginary parts are inserted.
    """
    cg_parts: list[torch.Tensor] = []
    for gid in range(len(group_data)):
        cg = getattr(module, f'_cg_red_{gid}')
        if cg.is_complex():
            cg_real = torch.view_as_real(cg.resolve_conj().contiguous()).reshape(-1)
        else:
            cg_ri = torch.stack([cg, torch.zeros_like(cg)], dim=-1)
            cg_real = cg_ri.reshape(-1)
        cg_parts.append(cg_real)
    if cg_parts:
        return torch.cat(cg_parts)
    return torch.zeros(0)


def flatten_sh_coefficients(
    f_coeffs: dict[int, torch.Tensor],
    lmax: int,
    coeff_offsets: torch.Tensor,
) -> torch.Tensor:
    """Pack the dict of SH coefficient vectors into a single flat real tensor.

    Returns shape [batch, total_coeffs * 2] (interleaved real/imag).
    """
    total = int(coeff_offsets[-1].item())
    parts = [f_coeffs[l] for l in range(lmax + 1) if l in f_coeffs]
    if not parts:
        batch_size = next(iter(f_coeffs.values())).shape[0]
        device = next(iter(f_coeffs.values())).device
        return torch.zeros(batch_size, total * 2, device=device)
    flat = torch.cat(parts, dim=-1)
    return torch.view_as_real(flat).reshape(flat.shape[0], -1).contiguous()


def _next_power_of_2(n: int) -> int:
    """Round up to the next power of 2 (minimum 1)."""
    n = max(n, 1)
    p = 1
    while p < n:
        p *= 2
    return p


def triton_bispectrum_forward(
    f_coeffs: dict[int, torch.Tensor],
    module: torch.nn.Module,
    num_entries: int,
) -> torch.Tensor:
    """Launch the fused Triton bispectrum kernel."""
    lmax = module.lmax
    batch_size = next(iter(f_coeffs.values())).shape[0]
    device = next(iter(f_coeffs.values())).device
    dtype = next(iter(f_coeffs.values())).dtype

    f_flat = flatten_sh_coefficients(f_coeffs, lmax, module._fused_coeff_offsets)
    f_flat = f_flat.to(device=device)

    cg_flat = module._fused_cg_flat.to(device=device, dtype=f_flat.dtype)
    entry_desc = module._fused_entry_desc.to(device=device)
    coeff_offsets = module._fused_coeff_offsets.to(device=device)

    out_flat = torch.zeros(batch_size, num_entries * 2, dtype=f_flat.dtype, device=device)

    n_total_entries = entry_desc.shape[0]
    if n_total_entries == 0:
        return torch.zeros(batch_size, 0, dtype=dtype, device=device)

    MAX_BLOCK_SIZE = _next_power_of_2(module._fused_max_block_size)
    BLOCK_K = 64
    if lmax <= 8:
        BLOCK_K = 32

    grid = (batch_size, n_total_entries)
    _bispectrum_entry_kernel[grid](
        f_flat,
        cg_flat,
        out_flat,
        entry_desc.reshape(-1),
        coeff_offsets,
        f_flat_stride=f_flat.shape[1],
        out_stride=num_entries,
        MAX_BLOCK_SIZE=MAX_BLOCK_SIZE,
        BLOCK_K=BLOCK_K,
    )

    return torch.view_as_complex(out_flat.reshape(batch_size, num_entries, 2)).to(dtype)
