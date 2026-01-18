"""Clebsch-Gordan coefficients for spherical harmonics coupling.

This module provides a placeholder for the Clebsch-Gordan matrix computation.
The actual implementation should be provided externally.
"""

import torch


def clebsch_gordan(l1: int, l2: int) -> torch.Tensor:
    """Return full Clebsch-Gordan matrix C_{l1,l2} for coupling l1 x l2.

    This is the unitary change-of-basis matrix from tensor product basis
    to coupled basis, organized in blocks for l in [|l1-l2|, l1+l2].

    The tensor product of two irreducible representations of SO(3) with
    degrees l1 and l2 decomposes as:

        V_{l1} ⊗ V_{l2} = V_{|l1-l2|} ⊕ V_{|l1-l2|+1} ⊕ ... ⊕ V_{l1+l2}

    The Clebsch-Gordan matrix performs this change of basis.

    Args:
        l1: First angular momentum quantum number (non-negative integer).
        l2: Second angular momentum quantum number (non-negative integer).

    Returns:
        torch.Tensor: Square unitary matrix of shape ((2l1+1)*(2l2+1), (2l1+1)*(2l2+1)).
            The matrix is organized in blocks, where each block corresponds to
            a coupled angular momentum l in the range [|l1-l2|, l1+l2].

    Raises:
        NotImplementedError: This is a placeholder. Provide an external implementation.
    """
    raise NotImplementedError(
        'Clebsch-Gordan coefficients not yet implemented. '
        'Please provide external implementation. '
        f'Called with l1={l1}, l2={l2}. '
        f'Expected output shape: ({(2*l1+1)*(2*l2+1)}, {(2*l1+1)*(2*l2+1)})'
    )
