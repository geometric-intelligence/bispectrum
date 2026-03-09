"""SO(2) bispectrum on the circle S^1.

Implements the G-Bispectrum for the rotation group SO(2) acting on S^1,
discretized at n equally-spaced sample points. This is mathematically
equivalent to the cyclic group C_n acting on Z/nZ — the class is a thin
subclass of CnonCn that provides the continuous-group naming convention.

Reference: Kakarala (2009), Mataigne et al. (2024) Algorithms 1-2.
"""

from bispectrum.cn_on_cn import CnonCn


class SO2onS1(CnonCn):
    """Bispectrum of SO(2) acting on the circle S^1.

    SO(2) acts on S^1 by rotation: (T_phi f)(theta) = f(theta - phi).
    After discretization at n equally-spaced points, this reduces exactly
    to the cyclic group C_n acting on Z/nZ, so all computation is inherited
    from CnonCn.

    This wrapper exists so the API has a `{Group}on{Domain}` entry for the
    continuous circle case, consistent with SO3onS2, SO2onD2, etc.

    Args:
        n: Number of sample points on S^1 (discretization resolution).
        selective: If True, use selective O(n) bispectrum.
            If False, use full bispectrum (n(n+1)/2 upper-triangular coefficients).
    """

    def __init__(self, n: int, selective: bool = True) -> None:
        super().__init__(n=n, selective=selective)
