"""SO(2) bispectrum on the unit disk.

Implements the selective disk bispectrum for SO(2) acting on the unit disk,
following Myers & Miolane, "The Selective Disk Bispectrum and Its Inversion,
with Application to Multi-Reference Alignment", arXiv:2511.19706, 2025.

The disk harmonic transform (DHT) decomposes a function f: D -> R into
disk harmonic coefficients a_{n,k} using the basis:
    psi_{nk}(r, theta) = c_{nk} * J_n(lambda_{nk} * r) * exp(i*n*theta)

The selective disk bispectrum (Definition 4.2) is:
    b_{0,0,k}  = a_{0,1}^2 * conj(a_{0,k})       for k = 1..K_0
    b_{2,n,k}  = a_{1,1} * a_{n,1} * conj(a_{n+1,k})  for n = 0..N_m-1, k = 1..K_{n+1}

Total selective coefficient count: N = sum_{n=0}^{N_m} K_n.
"""

import math

import torch
import torch.nn as nn

from bispectrum._bessel import bessel_jn, compute_all_bessel_roots


def _build_truncated_indices(
    L: int,
    bandlimit: float | None,
) -> tuple[list[tuple[int, int]], dict[int, int], int]:
    """Build the truncated DH coefficient index set.

    Following the convention of Marshall et al. (2023, fle_2d), the number
    of basis functions equals ne = floor(L^2 * pi / 4), approximately the
    number of pixels inside the unit disk. Coefficients are sorted by
    ascending Bessel root and the first ne are kept.

    If bandlimit is provided explicitly, it is used as a threshold on the
    Bessel root instead.

    Returns:
        indices: list of (n, k) pairs sorted by ascending Bessel root,
            including both positive and negative n.
        K: dict mapping angular order |n| -> max radial index K_n.
        N_m: maximum angular frequency (max |n|).
    """
    if bandlimit is not None:
        ne = None
        lam_max = bandlimit
    else:
        ne = int(L * L * math.pi / 4)
        lam_max = math.sqrt(math.pi * ne) * 1.5 + 10

    n_max_search = int(lam_max) + 2
    k_max_search = int(lam_max / math.pi) + 3

    all_roots = compute_all_bessel_roots(n_max_search, k_max_search)

    nk_lambda: list[tuple[float, int, int]] = []
    for abs_n, roots in all_roots.items():
        for k_idx, lam in enumerate(roots):
            if lam > lam_max:
                break
            k = k_idx + 1
            nk_lambda.append((lam, abs_n, k))
            if abs_n > 0:
                nk_lambda.append((lam, -abs_n, k))

    nk_lambda.sort(key=lambda t: t[0])

    if ne is not None and len(nk_lambda) >= ne:
        nk_lambda = nk_lambda[:ne]
        # Ensure complete conjugate pairs: if the last entry has n != 0
        # and its partner (-n, k) was cut off, remove it.
        if nk_lambda:
            _, last_n, last_k = nk_lambda[-1]
            if last_n != 0:
                partner = (-last_n, last_k)
                partner_present = any(n == partner[0] and k == partner[1] for _, n, k in nk_lambda)
                if not partner_present:
                    nk_lambda.pop()

    indices = [(n, k) for _, n, k in nk_lambda]

    K: dict[int, int] = {}
    for _, n, k in nk_lambda:
        abs_n = abs(n)
        K[abs_n] = max(K.get(abs_n, 0), k)

    N_m = max(K.keys()) if K else 0

    return indices, K, N_m


class SO2onDisk(nn.Module):
    """Bispectrum of SO(2) acting on the unit disk.

    Takes a square grayscale image f of shape (batch, L, L), computes the
    disk harmonic transform (DHT) internally, and returns the selective
    disk bispectrum coefficients.

    Reference: Myers & Miolane, "The Selective Disk Bispectrum and Its
    Inversion, with Application to Multi-Reference Alignment", arXiv:2511.19706, 2025.
    Forward uses Definition 4.2; inversion uses Theorem 4.4.

    Args:
        L: Image side length (input shape is (batch, L, L)).
        bandlimit: Maximum Bessel root frequency to include.
            Defaults to the fle_2d convention ne = floor(L^2 * pi / 4).
        selective: If True (default), compute O(m) selective bispectrum.
            If False, raises NotImplementedError.
    """

    def __init__(
        self,
        L: int,
        bandlimit: float | None = None,
        selective: bool = True,
    ) -> None:
        super().__init__()
        self.L = L
        self.selective = selective
        self._explicit_bandlimit = bandlimit

        indices, K, N_m = _build_truncated_indices(L, bandlimit)
        self._K = K
        self._N_m = N_m
        m = len(indices)

        nk_to_idx: dict[tuple[int, int], int] = {}
        for j, (n, k) in enumerate(indices):
            nk_to_idx[(n, k)] = j

        self._nk_to_idx = nk_to_idx
        self._indices = indices
        self._m = m

        # Build non-negative-n index set for the DH coefficients
        nonneg_indices: list[tuple[int, int]] = []
        for n, k in indices:
            if n >= 0 and (n, k) not in nonneg_indices:
                nonneg_indices.append((n, k))
        self._nonneg_indices = nonneg_indices
        m_nonneg = len(nonneg_indices)

        nonneg_to_idx: dict[tuple[int, int], int] = {}
        for j, (n, k) in enumerate(nonneg_indices):
            nonneg_to_idx[(n, k)] = j
        self._nonneg_to_idx = nonneg_to_idx
        self._m_nonneg = m_nonneg

        dx = 2.0 / L
        ys = torch.linspace(-1 + dx / 2, 1 - dx / 2, L, dtype=torch.float64)
        xs = torch.linspace(-1 + dx / 2, 1 - dx / 2, L, dtype=torch.float64)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        r_grid = torch.sqrt(grid_x**2 + grid_y**2)
        theta_grid = torch.atan2(grid_y, grid_x)
        disk_mask = r_grid < 1.0

        self.register_buffer('_disk_mask', disk_mask)

        pixel_r = r_grid[disk_mask]
        pixel_theta = theta_grid[disk_mask]
        p_disk = pixel_r.shape[0]
        self._p_disk = p_disk

        max_abs_n = max(abs(n) for n, _ in indices) if indices else 0
        max_k = max(k for _, k in indices) if indices else 0
        all_roots = compute_all_bessel_roots(max_abs_n + 1, max_k + 1)

        # Build REAL basis matrix Φ of shape (p_disk, d_real)
        # For real-valued signals, we decompose into:
        #   n=0: c * J_0(λ r)  (1 real column per (0,k))
        #   n>0: 2c * J_n(λ r) * cos(nθ)  and  -2c * J_n(λ r) * sin(nθ)
        #        (2 real columns per (n,k), capturing Re and Im of a_{n,k})
        K_0 = K.get(0, 0)
        d_real = K_0 + 2 * (m_nonneg - K_0)  # = m (total coefficients incl ±n)

        phi = torch.zeros(p_disk, d_real, dtype=torch.float64)
        bessel_roots_nonneg = torch.zeros(m_nonneg, dtype=torch.float64)

        # Map from non-negative (n,k) index to real-basis column indices
        # For n=0: real_col[j] = (col_re,) with a_{0,k} = x[col_re]
        # For n>0: real_col[j] = (col_re, col_im) with a_{n,k} = x[col_re] + i*x[col_im]
        real_col_map: list[tuple[int, ...]] = []
        col = 0
        for j, (n, k) in enumerate(nonneg_indices):
            abs_n = abs(n)
            lam = all_roots[abs_n][k - 1]
            bessel_roots_nonneg[j] = lam

            lam_t = torch.tensor(lam, dtype=torch.float64)
            jn_at_r = bessel_jn(abs_n, lam_t * pixel_r)

            jn_plus1_at_lam = bessel_jn(abs_n + 1, lam_t.unsqueeze(0))
            c_nk = 1.0 / (math.sqrt(math.pi) * jn_plus1_at_lam.abs().item())

            if n == 0:
                phi[:, col] = c_nk * jn_at_r
                real_col_map.append((col,))
                col += 1
            else:
                w = 2.0
                phi[:, col] = w * c_nk * jn_at_r * torch.cos(n * pixel_theta)
                phi[:, col + 1] = -w * c_nk * jn_at_r * torch.sin(n * pixel_theta)
                real_col_map.append((col, col + 1))
                col += 2

        self._real_col_map = real_col_map

        self.register_buffer('_bessel_roots', bessel_roots_nonneg)
        self.register_buffer('_phi', phi)

        # Analysis: x_real = pinv(Φ) @ f_disk (real least-squares)
        # Regularize by zeroing singular values below 1e-10 * max_sv
        # to handle basis functions that are unresolvable on the grid.
        phi_pinv = torch.linalg.pinv(phi, rcond=1e-10)  # (d_real, p_disk)
        self.register_buffer('_phi_pinv', phi_pinv)

        self._build_selective_indices()

    def _build_selective_indices(self) -> None:
        """Precompute vectorized index arrays for the selective bispectrum."""
        K = self._K
        N_m = self._N_m
        nonneg_to_idx = self._nonneg_to_idx

        idx_map: list[tuple[int, int, int]] = []

        # Type 0: b_{0,0,k} = a_{0,1}^2 * conj(a_{0,k})  for k=1..K_0
        type0_a0k_idx: list[int] = []
        for k in range(1, K.get(0, 0) + 1):
            type0_a0k_idx.append(nonneg_to_idx[(0, k)])
            idx_map.append((0, 0, k))

        # Type 2: b_{2,n,k} = a_{1,1} * a_{n,1} * conj(a_{n+1,k})
        # for n=0..N_m-1, k=1..K_{n+1}
        type2_an1_idx: list[int] = []
        type2_anp1k_idx: list[int] = []
        for n in range(0, N_m):
            Kn1 = K.get(n + 1, 0)
            for k in range(1, Kn1 + 1):
                type2_an1_idx.append(nonneg_to_idx[(n, 1)])
                type2_anp1k_idx.append(nonneg_to_idx[(n + 1, k)])
                idx_map.append((2, n, k))

        self._index_map_list = idx_map

        self.register_buffer(
            '_type0_a0k_idx',
            torch.tensor(type0_a0k_idx, dtype=torch.long),
        )
        self.register_buffer(
            '_type2_an1_idx',
            torch.tensor(type2_an1_idx, dtype=torch.long),
        )
        self.register_buffer(
            '_type2_anp1k_idx',
            torch.tensor(type2_anp1k_idx, dtype=torch.long),
        )

    def _real_to_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Convert real parameter vector to complex DH coefficients (n >= 0).

        Args:
            x: Real parameter vector, shape (batch, d_real).

        Returns:
            Complex DH coefficients for n >= 0, shape (batch, m_nonneg).
        """
        batch = x.shape[0]
        a = torch.zeros(batch, self._m_nonneg, dtype=torch.complex128, device=x.device)

        for j, cols in enumerate(self._real_col_map):
            if len(cols) == 1:
                a[:, j] = x[:, cols[0]].to(torch.complex128)
            else:
                a[:, j] = x[:, cols[0]] + 1j * x[:, cols[1]]

        return a

    def _complex_to_real(self, a: torch.Tensor) -> torch.Tensor:
        """Convert complex DH coefficients (n >= 0) to real parameter vector.

        Args:
            a: Complex DH coefficients for n >= 0, shape (batch, m_nonneg).

        Returns:
            Real parameter vector, shape (batch, d_real).
        """
        batch = a.shape[0]
        d_real = self._phi.shape[1]
        x = torch.zeros(batch, d_real, dtype=torch.float64, device=a.device)

        for j, cols in enumerate(self._real_col_map):
            if len(cols) == 1:
                x[:, cols[0]] = a[:, j].real
            else:
                x[:, cols[0]] = a[:, j].real
                x[:, cols[1]] = a[:, j].imag

        return x

    def _dht(self, f: torch.Tensor) -> torch.Tensor:
        """Compute disk harmonic coefficients from an image.

        Uses the real-basis pseudoinverse for well-conditioned analysis.

        Args:
            f: Image tensor, shape (batch, L, L).

        Returns:
            Complex DH coefficients (n >= 0), shape (batch, m_nonneg).
        """
        mask = self._disk_mask
        f_disk = f[:, mask].to(torch.float64)  # (batch, p_disk)
        x_real = f_disk @ self._phi_pinv.T  # (batch, d_real)
        return self._real_to_complex(x_real)

    def _idht(self, a: torch.Tensor) -> torch.Tensor:
        """Reconstruct image from disk harmonic coefficients (synthesis).

        Args:
            a: Complex DH coefficients (n >= 0), shape (batch, m_nonneg).

        Returns:
            Reconstructed image, shape (batch, L, L).
        """
        x_real = self._complex_to_real(a)
        f_disk = x_real @ self._phi.T  # (batch, p_disk)
        batch = f_disk.shape[0]
        f = torch.zeros(batch, self.L, self.L, device=f_disk.device, dtype=f_disk.dtype)
        f[:, self._disk_mask] = f_disk
        return f

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Compute the SO(2) selective disk bispectrum.

        Args:
            f: Real-valued image. Shape: (batch, L, L).

        Returns:
            Complex bispectrum tensor. Shape: (batch, output_size).
        """
        if not self.selective:
            raise NotImplementedError(
                'Full disk bispectrum not yet implemented. Use selective=True.'
            )

        a = self._dht(f)  # (batch, m_nonneg)

        a_01 = a[:, self._nonneg_to_idx[(0, 1)]]
        a_01_sq = a_01 * a_01

        # Type 0: b_{0,0,k} = a_{0,1}^2 * conj(a_{0,k})
        a_0k = a[:, self._type0_a0k_idx]
        type0 = a_01_sq.unsqueeze(-1) * a_0k.conj()

        a_11 = a[:, self._nonneg_to_idx[(1, 1)]]

        # Type 2: b_{2,n,k} = a_{1,1} * a_{n,1} * conj(a_{n+1,k})
        a_n1 = a[:, self._type2_an1_idx]
        a_np1k = a[:, self._type2_anp1k_idx]
        type2 = a_11.unsqueeze(-1) * a_n1 * a_np1k.conj()

        return torch.cat([type0, type2], dim=-1)

    def invert(self, beta: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Recover an image from selective disk bispectrum coefficients.

        Implements the bootstrap recovery from Theorem 4.4 of Myers &
        Miolane (arXiv:2511.19706). The recovered image is determined up to a global
        SO(2) rotation.

        Args:
            beta: Selective bispectrum. Shape: (batch, output_size), complex.

        Returns:
            Reconstructed image (real). Shape: (batch, L, L).

        Raises:
            NotImplementedError: If selective=False.
        """
        if not self.selective:
            raise NotImplementedError(
                'Inversion only implemented for selective bispectrum. Use selective=True.'
            )

        K = self._K
        N_m = self._N_m
        nonneg_to_idx = self._nonneg_to_idx
        batch = beta.shape[0]
        device = beta.device

        a = torch.zeros(batch, self._m_nonneg, dtype=torch.complex128, device=device)

        K_0 = K[0]
        offset = 0

        # Step 1: a_{0,1} from b_{0,0,1} = |a_{0,1}|^2 * a_{0,1}
        b_001 = beta[:, 0]
        a_01 = torch.abs(b_001) ** (1.0 / 3.0) * torch.exp(1j * torch.angle(b_001))
        a[:, nonneg_to_idx[(0, 1)]] = a_01

        # Step 2: a_{0,k} for k=2..K_0
        a_01_sq = a_01 * a_01
        for k in range(2, K_0 + 1):
            b_00k = beta[:, offset + k - 1]
            a[:, nonneg_to_idx[(0, k)]] = (b_00k / a_01_sq).conj()
        offset += K_0

        # Step 3: |a_{1,1}| from b_{2,0,1} = a_{0,1} * |a_{1,1}|^2
        b_201 = beta[:, offset]
        a_11 = torch.sqrt(torch.abs(b_201 / a_01))
        a[:, nonneg_to_idx[(1, 1)]] = a_11

        # Step 4: a_{1,k} for k=2..K_1
        K_1 = K.get(1, 0)
        for k in range(2, K_1 + 1):
            b_idx = offset + k - 1
            a[:, nonneg_to_idx[(1, k)]] = (beta[:, b_idx] / (a_11 * a_01)).conj()
        offset += K_1

        # Step 5: sequential for n=1..N_m-1
        for n in range(1, N_m):
            K_np1 = K.get(n + 1, 0)
            a_n1 = a[:, nonneg_to_idx[(n, 1)]]
            for k in range(1, K_np1 + 1):
                b_idx = offset + k - 1
                a[:, nonneg_to_idx[(n + 1, k)]] = (beta[:, b_idx] / (a_11 * a_n1)).conj()
            offset += K_np1

        return self._idht(a)

    @property
    def output_size(self) -> int:
        """Number of selective disk bispectrum coefficients."""
        return len(self._index_map_list)

    @property
    def index_map(self) -> list[tuple[int, int, int]]:
        """Maps flat output index -> (type, n, k) triple.

        type=0: b_{0,0,k} coefficient type=2: b_{2,n,k} coefficient
        """
        return list(self._index_map_list)

    def extra_repr(self) -> str:
        bl = f'{self._explicit_bandlimit:.2f}' if self._explicit_bandlimit is not None else 'auto'
        return f'L={self.L}, bandlimit={bl}, selective={self.selective}, output_size={self.output_size}'
