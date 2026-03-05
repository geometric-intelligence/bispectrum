"""Internal Clebsch-Gordan coefficient utilities for SO(3).

Loads precomputed CG matrices from the bundled JSON data file. Not part of the public API.
"""

import json
from pathlib import Path

import torch

_DATA_PATH = Path(__file__).parent / 'data' / 'cg_lmax5.json'


def load_cg_matrices(lmax: int) -> dict[tuple[int, int], torch.Tensor]:
    """Load CG matrices from bundled JSON for all (l1, l2) pairs with l1, l2 <= lmax.

    The CG matrix C_{l1,l2} is unitary and transforms from the uncoupled
    tensor-product basis |l1,m1> x |l2,m2> to the coupled basis |l,m>.

    Args:
        lmax: Maximum angular momentum degree to load.

    Returns:
        Dict mapping (l1, l2) -> CG matrix of shape ((2l1+1)(2l2+1), (2l1+1)(2l2+1)).

    Raises:
        ValueError: If lmax exceeds the precomputed JSON limits.
    """
    with open(_DATA_PATH) as f:
        data = json.load(f)

    metadata = data['metadata']
    max_l1 = metadata['l1_max']
    max_l2 = metadata['l2_max']

    if lmax > min(max_l1, max_l2):
        raise ValueError(
            f'lmax={lmax} exceeds JSON limits (l1_max={max_l1}, l2_max={max_l2}). '
            'Generate a larger CG file or reduce lmax.'
        )

    matrices: dict[tuple[int, int], torch.Tensor] = {}
    for entry in data['matrices'].values():
        l1, l2 = entry['l1'], entry['l2']
        if l1 <= lmax and l2 <= lmax:
            matrices[(l1, l2)] = torch.tensor(entry['matrix'], dtype=torch.float64)

    return matrices
