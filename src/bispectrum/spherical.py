"""Spherical harmonics utilities and bispectrum computation.

This module provides functions for computing the bispectrum of spherical functions
using complex spherical harmonics coefficients from torch-harmonics.

CRITICAL: RealSHT from torch-harmonics computes COMPLEX spherical harmonics
coefficients F_l^m of a REAL-valued function. The "Real" refers to the input
function being real, NOT the output coefficients. Do NOT confuse with "real
spherical harmonics" R_l^m.
"""

from typing import Callable

import torch

# Use RealSHT from torch-harmonics to compute complex SH coefficients (m >= 0)
# REMINDER: RealSHT outputs COMPLEX coefficients for a REAL function!
from torch_harmonics import RealSHT  # noqa: F401


def get_full_sh_coefficients(
    coeffs_positive_m: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Extend SHT output to include negative m values.

    From coeffs of shape (batch, lmax, mmax) for m >= 0,
    build full coefficients for m in [-l, l] for each l.

    Uses the symmetry relation for real functions:
        F_l^{-m} = (-1)^m * conj(F_l^m)

    Args:
        coeffs_positive_m: Complex tensor of shape (batch, lmax, mmax)
            containing SH coefficients for m >= 0 from RealSHT.

    Returns:
        Dict mapping l -> tensor of shape (batch, 2l+1) containing
        coefficients for m in [-l, -l+1, ..., l-1, l].
    """
    batch_size, lmax, mmax = coeffs_positive_m.shape
    result: dict[int, torch.Tensor] = {}

    for l in range(lmax):
        # Number of m values for this l: from -l to l, so 2l+1 values
        # We can only use m values up to min(l, mmax-1)
        m_max_for_l = min(l, mmax - 1)

        # Create tensor for full coefficients: m from -l to l
        # Index mapping: m -> index = m + l
        # So m=-l -> index 0, m=0 -> index l, m=l -> index 2l
        full_coeffs = torch.zeros(
            batch_size, 2 * l + 1, dtype=coeffs_positive_m.dtype, device=coeffs_positive_m.device
        )

        # Fill m = 0 coefficient (index l)
        full_coeffs[:, l] = coeffs_positive_m[:, l, 0]

        # Fill positive m coefficients and derive negative m
        for m in range(1, m_max_for_l + 1):
            # Positive m: direct from input
            # m -> index = l + m
            full_coeffs[:, l + m] = coeffs_positive_m[:, l, m]

            # Negative m: F_l^{-m} = (-1)^m * conj(F_l^m)
            # -m -> index = l - m
            sign = (-1) ** m
            full_coeffs[:, l - m] = sign * torch.conj(coeffs_positive_m[:, l, m])

        result[l] = full_coeffs

    return result


def compute_padding_indices(l1: int, l2: int, l: int) -> tuple[int, int]:
    """Compute n_p and n_s for zero-padding F_l.

    The CG matrix organizes the coupled basis in blocks for each l value
    from |l1-l2| to l1+l2. This function computes how many zeros to place
    before (n_p) and after (n_s) the F_l coefficients.

    Args:
        l1: First degree index.
        l2: Second degree index.
        l: Target degree index (must satisfy |l1-l2| <= l <= l1+l2).

    Returns:
        Tuple (n_p, n_s) where:
            n_p = sum_{q=|l1-l2|}^{l-1} (2q+1) -- zeros before F_l
            n_s = sum_{q=l+1}^{l1+l2} (2q+1)   -- zeros after F_l
    """
    l_min = abs(l1 - l2)
    l_max = l1 + l2

    # n_p: sum of (2q+1) for q from |l1-l2| to l-1
    n_p = sum(2 * q + 1 for q in range(l_min, l))

    # n_s: sum of (2q+1) for q from l+1 to l1+l2
    n_s = sum(2 * q + 1 for q in range(l + 1, l_max + 1))

    return n_p, n_s


def pad_sh_coefficients(
    f_l: torch.Tensor, l1: int, l2: int, l: int
) -> torch.Tensor:
    """Create zero-padded vector F_hat_l of size (2l1+1)(2l2+1).

    F_hat_l = [0, ..., 0, F_l, 0, ..., 0]
    with n_p zeros before and n_s zeros after.

    Args:
        f_l: Tensor of shape (batch, 2l+1) containing SH coefficients for degree l.
        l1: First degree index for the bispectrum.
        l2: Second degree index for the bispectrum.
        l: Degree of the input coefficients.

    Returns:
        Tensor of shape (batch, (2l1+1)*(2l2+1)) with F_l in the correct position.
    """
    batch_size = f_l.shape[0]
    total_size = (2 * l1 + 1) * (2 * l2 + 1)

    n_p, n_s = compute_padding_indices(l1, l2, l)

    # Create zero-padded tensor
    padded = torch.zeros(
        batch_size, total_size, dtype=f_l.dtype, device=f_l.device
    )

    # Place F_l at the correct position
    f_l_size = 2 * l + 1
    padded[:, n_p : n_p + f_l_size] = f_l

    return padded


def bispectrum(
    f_coeffs: dict[int, torch.Tensor],
    l1: int,
    l2: int,
    clebsch_gordan_fn: Callable[[int, int], torch.Tensor],
) -> torch.Tensor:
    """Compute bispectrum beta(f)_{l1,l2}[l] for all valid l values.

    Formula: beta_{l1,l2}[l] = (F_l1 ⊗ F_l2) @ C_{l1,l2} @ F_hat_l^†

    where:
        - F_l1, F_l2 are the SH coefficient vectors for degrees l1, l2
        - ⊗ denotes the tensor (Kronecker) product
        - C_{l1,l2} is the Clebsch-Gordan matrix
        - F_hat_l is the zero-padded F_l vector
        - † denotes conjugate transpose

    Args:
        f_coeffs: Dict mapping l -> (batch, 2l+1) complex tensor containing
            the full SH coefficients (for all m from -l to l).
        l1: First degree index.
        l2: Second degree index.
        clebsch_gordan_fn: Function(l1, l2) -> square matrix of shape (d, d)
            where d = (2l1+1)*(2l2+1). Returns the Clebsch-Gordan matrix.

    Returns:
        torch.Tensor of shape (batch, num_l) where num_l = l1 + l2 - |l1 - l2| + 1.
        Entry [..., i] corresponds to l = |l1 - l2| + i.
    """
    # Get the coefficient vectors for l1 and l2
    f_l1 = f_coeffs[l1]  # (batch, 2*l1+1)
    f_l2 = f_coeffs[l2]  # (batch, 2*l2+1)

    batch_size = f_l1.shape[0]

    # Compute batch-wise tensor product using einsum (NOT torch.kron!)
    # f_l1: (batch, 2*l1+1), f_l2: (batch, 2*l2+1)
    outer = torch.einsum('bi,bj->bij', f_l1, f_l2)
    # Flatten to (batch, (2*l1+1)*(2*l2+1))
    tensor_product = outer.reshape(batch_size, -1)

    # Get Clebsch-Gordan matrix
    cg_matrix = clebsch_gordan_fn(l1, l2)  # (d, d) where d = (2l1+1)*(2l2+1)

    # Move CG matrix to same device as coefficients
    cg_matrix = cg_matrix.to(device=tensor_product.device, dtype=tensor_product.dtype)

    # Apply CG matrix: (batch, d) @ (d, d) -> (batch, d)
    transformed = tensor_product @ cg_matrix

    # Compute bispectrum for each valid l
    l_min = abs(l1 - l2)
    l_max = l1 + l2
    num_l = l_max - l_min + 1

    # Result tensor
    result = torch.zeros(batch_size, num_l, dtype=tensor_product.dtype, device=tensor_product.device)

    for i, l in enumerate(range(l_min, l_max + 1)):
        if l not in f_coeffs:
            # Skip if we don't have coefficients for this l
            continue

        f_l = f_coeffs[l]  # (batch, 2l+1)

        # Create zero-padded version
        f_hat_l = pad_sh_coefficients(f_l, l1, l2, l)  # (batch, d)

        # Compute inner product: sum over d dimension
        # beta = transformed @ f_hat_l^†
        # For each batch element: (d,) @ (d,)^† = scalar
        # f_hat_l^† means conjugate transpose, but since we're doing dot product
        # with row vectors, we use conj and sum
        result[:, i] = torch.sum(transformed * torch.conj(f_hat_l), dim=-1)

    return result
