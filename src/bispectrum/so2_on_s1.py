"""Cyclic bispectrum for n equally spaced circle samples.

This naming wrapper implements exactly the finite subgroup C_n of SO(2):
rotations by integer multiples of 2*pi/n. It inherits modular frequency
addition from CnonCn, which is not invariant to arbitrary continuous SO(2)
rotations of a band-limited function. For example, n=3 and
f(theta)=1+cos(theta-alpha) give beta_{1,1}=(3/2)^3 exp(-3i alpha).

Reference: Kakarala (2009), Mataigne et al. (2024) Algorithms 1-2.
"""

from bispectrum.cn_on_cn import CnonCn


class SO2onS1(CnonCn):
    """Bispectrum of C_n acting on n equally spaced samples of S^1.

    The exact action is the cyclic sample permutation induced by rotations
    phi=2*pi*k/n. Arbitrary continuous rotations are outside this guarantee;
    modular bispectral triplets can alias even for band-limited real input.
    The historical SO2onS1 name is retained for compatibility.

    Args:
        n: Number of sample points on S^1 (discretization resolution).
        selective: If True, use selective O(n) bispectrum.
            If False, use full bispectrum (n(n+1)/2 upper-triangular coefficients).
    """

    def __init__(self, n: int, selective: bool = True) -> None:
        super().__init__(n=n, selective=selective)
