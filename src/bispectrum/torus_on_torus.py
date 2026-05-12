"""Discrete d-torus bispectrum on C_{n_1} × ··· × C_{n_d}.

Implements the G-Bispectrum for the discrete d-torus (product of cyclic groups)
acting on itself.

For signal f: C_{n_1} × ··· × C_{n_d} → R:
    β(f)_{k1,k2} = f_hat[k1] · f_hat[k2] · conj(f_hat[(k1+k2) mod n])

where f_hat is the d-dimensional DFT and k1, k2 are multi-indices
with componentwise modular addition.

Reference: Kakarala (2009), Mataigne et al. (2024) Theorem 4.3 + Algorithm 2.
"""

import math
import warnings

import torch
import torch.nn as nn


def _ravel(multi_idx: tuple[int, ...], ns: tuple[int, ...]) -> int:
    """Convert multi-index to flat (row-major) index."""
    flat = 0
    for k, n in zip(multi_idx, ns, strict=False):
        flat = flat * n + k
    return flat


def _unravel(flat: int, ns: tuple[int, ...]) -> tuple[int, ...]:
    """Convert flat (row-major) index to multi-index."""
    result: list[int] = []
    for n in reversed(ns):
        result.append(flat % n)
        flat //= n
    return tuple(reversed(result))


def _add_mod(a: tuple[int, ...], b: tuple[int, ...], ns: tuple[int, ...]) -> tuple[int, ...]:
    """Componentwise modular addition of multi-indices."""
    return tuple((ai + bi) % n for ai, bi, n in zip(a, b, ns, strict=False))


class TorusOnTorus(nn.Module):
    """Bispectrum of the discrete d-torus C_{n_1} × ··· × C_{n_d} acting on itself.

    The discrete d-torus is a finite abelian group. All irreps are 1D characters,
    so no Clebsch-Gordan matrices are needed — the bispectrum reduces to scalar
    triple products of multidimensional DFT coefficients.

    Reference: Mataigne et al., "The Selective G-Bispectrum and its Inversion",
    NeurIPS 2024. Forward uses Theorem A.12; selectivity from Theorem 4.3;
    inversion uses Algorithm 2 (Appendix D).

    Args:
        ns: Shape of the torus — tuple of group orders for each dimension.
            E.g. (8, 8) for C_8 × C_8, (64, 64, 64) for a 3D periodic volume.
        selective: If True, use selective O(|G|) bispectrum.
            If False, use full bispectrum (|G|*(|G|+1)/2 upper-triangular coefficients).
    """

    def __init__(self, ns: tuple[int, ...], selective: bool = True) -> None:
        super().__init__()
        if not ns:
            raise ValueError('ns must be non-empty.')
        if any(n <= 0 for n in ns):
            raise ValueError('All dimensions must be positive.')
        if selective and any(n < 2 for n in ns):
            raise ValueError('selective=True requires all dimensions >= 2.')

        self.ns = tuple(ns)
        self.selective = selective
        self._d = len(ns)
        self._group_order_val = math.prod(ns)

        if selective:
            idx_k1, idx_k2, idx_sum, index_map = self._build_selective_indices()
            self._generator_flat: list[int] = [
                _ravel(tuple(1 if j == l else 0 for j in range(self._d)), self.ns)
                for l in range(self._d)
            ]
        else:
            if self._group_order_val > 10_000:
                warnings.warn(
                    f'Full bispectrum for |G|={self._group_order_val} will produce '
                    f'{self._group_order_val * (self._group_order_val + 1) // 2} '
                    f'coefficients. Consider selective=True.',
                    stacklevel=2,
                )
            idx_k1, idx_k2, idx_sum, index_map = self._build_full_indices()

        self.register_buffer('_idx_k1', torch.tensor(idx_k1, dtype=torch.long))
        self.register_buffer('_idx_k2', torch.tensor(idx_k2, dtype=torch.long))
        self.register_buffer('_idx_k1pk2', torch.tensor(idx_sum, dtype=torch.long))
        self._index_map: list[tuple[tuple[int, ...], tuple[int, ...]]] = index_map

    def _build_selective_indices(
        self,
    ) -> tuple[list[int], list[int], list[int], list[tuple[tuple[int, ...], tuple[int, ...]]]]:
        """Build selective bispectrum indices via row-major predecessor traversal (Algorithm 2).

        Iterates through all |G| elements in row-major order. For each element k:
        - k = 0: pair is (0, 0)
        - k != 0: find first axis l with k_l > 0, pair is (e_l, k - e_l)

        Row-major ordering guarantees k - e_l is always processed before k.
        """
        ns = self.ns
        d = self._d
        G = self._group_order_val
        zero = (0,) * d
        generators = [tuple(1 if j == l else 0 for j in range(d)) for l in range(d)]

        idx_k1: list[int] = []
        idx_k2: list[int] = []
        idx_sum: list[int] = []
        index_map: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        for flat_idx in range(G):
            k = _unravel(flat_idx, ns)

            if k == zero:
                idx_k1.append(0)
                idx_k2.append(0)
                idx_sum.append(0)
                index_map.append((zero, zero))
            else:
                l = next(i for i in range(d) if k[i] > 0)
                e_l = generators[l]
                k_prev = tuple((k[j] - e_l[j]) % ns[j] for j in range(d))

                idx_k1.append(_ravel(e_l, ns))
                idx_k2.append(_ravel(k_prev, ns))
                idx_sum.append(flat_idx)
                index_map.append((e_l, k_prev))

        return idx_k1, idx_k2, idx_sum, index_map

    def _build_full_indices(
        self,
    ) -> tuple[list[int], list[int], list[int], list[tuple[tuple[int, ...], tuple[int, ...]]]]:
        """Build full (upper-triangular) bispectrum indices."""
        ns = self.ns
        G = self._group_order_val

        idx_k1: list[int] = []
        idx_k2: list[int] = []
        idx_sum: list[int] = []
        index_map: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        for i in range(G):
            ki = _unravel(i, ns)
            for j in range(i, G):
                kj = _unravel(j, ns)
                k_s = _add_mod(ki, kj, ns)
                idx_k1.append(i)
                idx_k2.append(j)
                idx_sum.append(_ravel(k_s, ns))
                index_map.append((ki, kj))

        return idx_k1, idx_k2, idx_sum, index_map

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Compute the bispectrum of a signal on the torus.

        Args:
            f: Real-valued signal. Shape: (batch, n_1, n_2, ..., n_d).

        Returns:
            Complex bispectrum tensor. Shape: (batch, output_size).
        """
        if f.is_complex():
            raise TypeError('f must be a real-valued tensor, got complex dtype.')
        if f.ndim != 1 + self._d or tuple(f.shape[1:]) != self.ns:
            raise ValueError(
                f'Expected shape (batch, {", ".join(str(n) for n in self.ns)}), '
                f'got {tuple(f.shape)}'
            )

        spatial_dims = tuple(range(1, 1 + self._d))
        fhat = torch.fft.fftn(f, dim=spatial_dims)
        fhat_flat = fhat.reshape(f.shape[0], -1)

        a = fhat_flat[:, self._idx_k1]
        b = fhat_flat[:, self._idx_k2]
        c = fhat_flat[:, self._idx_k1pk2]
        return a * b * c.conj()

    @staticmethod
    def _safe_denom(z: torch.Tensor, eps: float) -> torch.Tensor:
        """Clamp magnitude away from zero, preserving exact values when safe."""
        return torch.where(
            torch.abs(z) < eps,
            torch.full_like(z, eps),
            z,
        )

    def invert(self, beta: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Recover the signal from selective bispectrum coefficients.

        Implements Algorithm 2 from Mataigne et al. (2024).
        The output is a canonical representative of the translation-equivalence
        class, not the original signal. It is determined up to d independent
        translations (one per cyclic factor); generator phases are fixed to
        zero.

        Near-zero pivots are clamped rather than raising, so a single
        degenerate sample in a batch won't crash the whole computation;
        the affected sample will get a noisy reconstruction for those
        frequencies.

        Args:
            beta: Selective bispectrum. Shape: (batch, |G|), complex.

        Returns:
            Reconstructed signal (complex). Shape: (batch, n_1, ..., n_d).

        Raises:
            NotImplementedError: If selective=False.
            TypeError: If beta is not complex.
        """
        if not self.selective:
            raise NotImplementedError(
                'Inversion is only implemented for the selective bispectrum. Use selective=True.'
            )
        if not beta.is_complex():
            raise TypeError('beta must be a complex tensor.')

        G = self._group_order_val
        if beta.ndim != 2 or beta.shape[-1] != G:
            raise ValueError(f'Expected shape (batch, {G}), got {tuple(beta.shape)}')

        batch = beta.shape[0]
        fhat = torch.zeros(batch, G, dtype=beta.dtype, device=beta.device)

        eps = 1e-10

        # Phase 0: recover fhat[0] from β_{0,0} = |fhat[0]|^2 · fhat[0]
        fhat[:, 0] = torch.abs(beta[:, 0]) ** (1.0 / 3.0) * torch.exp(1j * torch.angle(beta[:, 0]))

        # Phase 1: recover |fhat[e_l]| for each generator, fix phase to 0
        gen_set = set(self._generator_flat)
        for _l, g_flat in enumerate(self._generator_flat):
            fhat[:, g_flat] = torch.sqrt(
                torch.abs(beta[:, g_flat] / self._safe_denom(fhat[:, 0], eps))
            )

        # Phases 2+3: sequential bootstrap
        # fhat[k] = conj(β_k / (fhat[e_l] · fhat[k - e_l]))
        idx_k1_cpu = self._idx_k1.tolist()
        idx_k2_cpu = self._idx_k2.tolist()
        for flat_k in range(1, G):
            if flat_k in gen_set:
                continue
            e_l_flat = idx_k1_cpu[flat_k]
            k_prev_flat = idx_k2_cpu[flat_k]
            denom = fhat[:, e_l_flat] * fhat[:, k_prev_flat]
            denom = self._safe_denom(denom, eps)
            fhat[:, flat_k] = torch.conj(beta[:, flat_k] / denom)

        fhat_nd = fhat.reshape(batch, *self.ns)
        spatial_dims = tuple(range(1, 1 + self._d))
        return torch.fft.ifftn(fhat_nd, dim=spatial_dims)

    @property
    def output_size(self) -> int:
        """Number of bispectral coefficients in the output."""
        return len(self._index_map)

    @property
    def group_order(self) -> int:
        """|G| = n_1 × n_2 × ··· × n_d."""
        return self._group_order_val

    @property
    def ndim(self) -> int:
        """Number of torus dimensions d."""
        return self._d

    @property
    def index_map(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Maps flat output index → (k1, k2) multi-index frequency pair."""
        return list(self._index_map)

    def extra_repr(self) -> str:
        return f'ns={self.ns}, selective={self.selective}, output_size={self.output_size}'
