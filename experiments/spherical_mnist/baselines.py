"""Learned equivariant baselines for Spherical MNIST, built on e3nn.

    1. ``s2cnn``          - Cohen et al. (2018) spherical CNN: S2 conv -> SO(3) conv -> SO(3) integral
    2. ``tensor_product`` - Clebsch-Gordan tensor-product network (Kondor et al. 2018) with gated
                            nonlinearities; invariant scalars -> MLP

Both read the same equiangular (nlat x nlon) grid as the bispectrum model. Spherical and
SO(3) Fourier transforms are dense matmuls against e3nn real spherical harmonics / Wigner-D
matrices evaluated on quadrature grids, so every layer is exactly equivariant up to the
grid's band-limit (aliasing above it).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from e3nn import nn as enn
from e3nn import o3


def clenshaw_curtis_weights(n: int) -> torch.Tensor:
    """Weights for int_{-1}^{1} g(x) dx at x_j = cos(j pi / (n - 1)), j = 0..n-1."""
    order = n - 1
    theta = torch.arange(n, dtype=torch.float64) * math.pi / order
    w = torch.zeros(n, dtype=torch.float64)
    v = torch.ones(n - 2, dtype=torch.float64)
    inner = theta[1:-1]
    if order % 2 == 0:
        w[0] = w[-1] = 1.0 / (order**2 - 1)
        for k in range(1, order // 2):
            v -= 2.0 * torch.cos(2 * k * inner) / (4 * k**2 - 1)
        v -= torch.cos(order * inner) / (order**2 - 1)
    else:
        w[0] = w[-1] = 1.0 / order**2
        for k in range(1, (order - 1) // 2 + 1):
            v -= 2.0 * torch.cos(2 * k * inner) / (4 * k**2 - 1)
    w[1:-1] = 2.0 * v / order
    return w


def real_spherical_harmonics(lmax: int, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """(N, (lmax+1)^2) e3nn real SH, 'integral' normalization, at e3nn angles (alpha, beta).

    Built as Y_l(R(alpha, beta, 0) y) = D_l(alpha, beta, 0) Y_l(y): matches
    ``o3.spherical_harmonics`` where it exists (l <= 12) and extends it to any degree.
    """
    zero = torch.zeros_like(alpha)
    return torch.cat(
        [
            o3.wigner_D(lv, alpha, beta, zero)[..., :, lv]
            * math.sqrt((2 * lv + 1) / (4 * math.pi))
            for lv in range(lmax + 1)
        ],
        dim=-1,
    )


def equiangular_quadrature(
    nlat: int, nlon: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """e3nn angles (alpha, beta) of the ``data.py`` sphere grid and weights summing to 4 pi.

    The grid pole is e3nn's y-axis (beta = theta, alpha = phi), so a rotation by 2 pi k / nlon
    about the pole is a column shift of the image and an alpha shift of the SOFT grids.
    """
    theta = torch.linspace(0, math.pi, nlat, dtype=torch.float64)
    phi = torch.linspace(0, 2 * math.pi, nlon + 1, dtype=torch.float64)[:-1]
    t, p = torch.meshgrid(theta, phi, indexing='ij')
    w = clenshaw_curtis_weights(nlat)[:, None].expand(nlat, nlon) * (2 * math.pi / nlon)
    return p.reshape(-1), t.reshape(-1), w.reshape(-1)


def sphere_analysis_matrix(lmax: int, nlat: int, nlon: int) -> torch.Tensor:
    """(nlat*nlon, (lmax+1)^2) map from grid values to real SH coefficients, l <= lmax."""
    alpha, beta, w = equiangular_quadrature(nlat, nlon)
    return real_spherical_harmonics(lmax, alpha, beta) * w[:, None]


def so3_soft_grid(b: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """SOFT grid (2b)^3 for bandwidth b (degrees l < b) with Haar weights summing to 1."""
    k = torch.arange(2 * b, dtype=torch.float64)
    beta = math.pi * (2 * k + 1) / (4 * b)
    j = torch.arange(b, dtype=torch.float64)
    wb = (
        (2.0 / b)
        * beta.sin()
        * (torch.sin((2 * j[None] + 1) * beta[:, None]) / (2 * j + 1)).sum(-1)
    )
    angle = 2 * math.pi * k / (2 * b)
    a, bt, g = torch.meshgrid(angle, beta, angle, indexing='ij')
    w = wb[None, :, None].expand_as(a)
    return a.reshape(-1), bt.reshape(-1), g.reshape(-1), (w / w.sum()).reshape(-1)


def flat_wigner(
    lmax: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    """(N, sum_{l<=lmax} (2l+1)^2) real Wigner-D entries D^l_{mn}, row-major per degree."""
    return torch.cat(
        [o3.wigner_D(lv, alpha, beta, gamma).flatten(-2) for lv in range(lmax + 1)], dim=-1
    )


def s2_near_identity_grid(
    max_beta: float = math.pi / 8, n_alpha: int = 8, n_beta: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cohen et al. S2 kernel support: rings at beta in (0, max_beta] around the pole."""
    beta = torch.arange(1, n_beta + 1, dtype=torch.float64) * max_beta / n_beta
    alpha = torch.arange(n_alpha, dtype=torch.float64) * 2 * math.pi / n_alpha
    b, a = torch.meshgrid(beta, alpha, indexing='ij')
    return a.reshape(-1), b.reshape(-1)


def so3_near_identity_grid(
    max_beta: float = math.pi / 8,
    max_gamma: float = 2 * math.pi,
    n_alpha: int = 8,
    n_beta: int = 3,
    n_gamma: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cohen et al. SO(3) kernel support around the identity."""
    n_gamma = n_alpha if n_gamma is None else n_gamma
    beta = torch.arange(1, n_beta + 1, dtype=torch.float64) * max_beta / n_beta
    alpha = torch.arange(n_alpha, dtype=torch.float64) * 2 * math.pi / n_alpha
    pre_gamma = torch.linspace(-max_gamma, max_gamma, n_gamma, dtype=torch.float64)
    b, a, c = torch.meshgrid(beta, alpha, pre_gamma, indexing='ij')
    return a.reshape(-1), b.reshape(-1), (c - a).reshape(-1)


def _blocks(lmax: int) -> list[tuple[int, int, int]]:
    """(degree, start, stop) of each D^l block in a flat Wigner vector."""
    out, start = [], 0
    for lv in range(lmax + 1):
        size = (2 * lv + 1) ** 2
        out.append((lv, start, start + size))
        start += size
    return out


class S2CNN(nn.Module):
    """Cohen et al. (2018) Spherical MNIST architecture.

    S2 conv (1 -> f1, bandwidth b1) -> ReLU -> SO(3) conv (f1 -> f2, b1 -> b2) -> ReLU ->
    SO(3) integral -> linear. Kernels live on the paper's near-identity grids. With
    ``batch_norm`` a per-channel BatchNorm (statistics over batch and grid, hence still
    equivariant) follows each conv, and the invariant features are batch-normalized before
    the linear readout.
    """

    def __init__(
        self,
        nlat: int = 64,
        nlon: int = 128,
        features: tuple[int, int] = (20, 40),
        bandwidths: tuple[int, int] = (10, 6),
        num_classes: int = 10,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        f1, f2 = features
        b1, b2 = bandwidths
        self.b1, self.b2 = b1, b2
        self.invariant_features = f2

        self.register_buffer(
            'analysis', sphere_analysis_matrix(b1 - 1, nlat, nlon).float(), persistent=False
        )
        ka, kb = s2_near_identity_grid()
        y_kernel = real_spherical_harmonics(b1 - 1, ka, kb)
        self.register_buffer('s2_kernel_basis', y_kernel.float(), persistent=False)
        self.kernel1 = nn.Parameter(torch.empty(1, f1, len(ka)).uniform_(-1, 1))

        a1, be1, g1, w1 = so3_soft_grid(b1)
        d1 = flat_wigner(b1 - 1, a1, be1, g1)
        n_low = sum((2 * lv + 1) ** 2 for lv in range(b2))
        degree = torch.cat([torch.full(((2 * lv + 1) ** 2,), 2 * lv + 1.0) for lv in range(b2)])
        self.register_buffer('synthesis1', d1.T.contiguous().float(), persistent=False)
        self.register_buffer(
            'analysis1', (d1[:, :n_low] * w1[:, None] * degree[None]).float(), persistent=False
        )

        sa, sb, sg = so3_near_identity_grid()
        self.register_buffer(
            'so3_kernel_basis', flat_wigner(b2 - 1, sa, sb, sg).float(), persistent=False
        )
        self.kernel2 = nn.Parameter(torch.empty(f1, f2, len(sa)).uniform_(-1, 1))

        a2, be2, g2, w2 = so3_soft_grid(b2)
        self.register_buffer(
            'synthesis2', flat_wigner(b2 - 1, a2, be2, g2).T.contiguous().float(), persistent=False
        )
        self.register_buffer('haar2', w2.float(), persistent=False)

        self.scale1 = 1.0 / math.sqrt(len(ka))
        self.scale2 = 1.0 / math.sqrt(len(sa) * f1)
        self.bias1 = nn.Parameter(torch.zeros(1, f1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1, f2, 1))
        self.norm1: nn.Module = nn.BatchNorm1d(f1) if batch_norm else nn.Identity()
        self.norm2: nn.Module = nn.BatchNorm1d(f2) if batch_norm else nn.Identity()
        self.norm_out: nn.Module = nn.BatchNorm1d(f2) if batch_norm else nn.Identity()
        self.out = nn.Linear(f2, num_classes)

    def s2_conv(self, x: torch.Tensor) -> torch.Tensor:
        """(B, nlat, nlon) -> SO(3) coefficients (B, f1, sum_{l<b1} (2l+1)^2)."""
        coeffs = x.flatten(1) @ self.analysis
        psi = torch.einsum('iop,pk->iok', self.kernel1, self.s2_kernel_basis) * self.scale1
        blocks = []
        for lv in range(self.b1):
            sl = slice(lv * lv, (lv + 1) ** 2)
            blocks.append(torch.einsum('bm,ion->bomn', coeffs[:, sl], psi[:, :, sl]).flatten(2))
        return torch.cat(blocks, dim=-1)

    def so3_conv(self, g: torch.Tensor) -> torch.Tensor:
        """(B, f1, sum_{l<b2}) -> (B, f2, sum_{l<b2}): h_l = sum_i g_l k_l^T per degree."""
        k = torch.einsum('iop,pk->iok', self.kernel2, self.so3_kernel_basis) * self.scale2
        blocks = []
        for lv, start, stop in _blocks(self.b2 - 1):
            d = 2 * lv + 1
            gl = g[..., start:stop].unflatten(-1, (d, d))
            kl = k[..., start:stop].unflatten(-1, (d, d))
            blocks.append(torch.einsum('ziab,iocb->zoac', gl, kl).flatten(2))
        return torch.cat(blocks, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            h = self.s2_conv(x.float()) @ self.synthesis1
            h = torch.relu(self.norm1(h + self.bias1))
            h = self.so3_conv(h @ self.analysis1) @ self.synthesis2
            h = torch.relu(self.norm2(h + self.bias2))
            z = self.norm_out(h @ self.haar2)
        return self.out(z)


class TensorProductNet(nn.Module):
    """Clebsch-Gordan network on real SH coefficients with e3nn tensor products.

    Layer i maps h_{i-1} (the SH coefficients for i = 0) through a fully connected tensor
    product with the input coefficients, a gated nonlinearity and an equivariant BatchNorm.
    The 0e channels of every layer are rotation invariants; an MLP shaped like the bispectrum
    head classifies them. Irreps carry even parity only: the model is SO(3)-equivariant, so
    every CG path allowed by SO(3) is kept.
    """

    def __init__(
        self,
        lmax: int = 15,
        nlat: int = 64,
        nlon: int = 128,
        channels: int = 16,
        hidden_lmax: int = 6,
        n_layers: int = 3,
        hidden: int = 256,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.register_buffer(
            'analysis', sphere_analysis_matrix(lmax, nlat, nlon).float(), persistent=False
        )
        self.irreps_in = o3.Irreps([(1, (lv, 1)) for lv in range(lmax + 1)])
        self.norm_in = enn.BatchNorm(self.irreps_in)
        gated = o3.Irreps([(channels, (lv, 1)) for lv in range(1, hidden_lmax + 1)])
        scalars = o3.Irreps([(channels, (0, 1))])
        gates = o3.Irreps([(gated.num_irreps, (0, 1))])
        self.irreps_hidden = scalars + gated

        self.tps = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.norms = nn.ModuleList()
        irreps_prev = self.irreps_in
        for _ in range(n_layers):
            gate = enn.Gate(scalars, [torch.relu], gates, [torch.sigmoid], gated)
            self.tps.append(
                o3.FullyConnectedTensorProduct(irreps_prev, self.irreps_in, gate.irreps_in)
            )
            self.gates.append(gate)
            self.norms.append(enn.BatchNorm(gate.irreps_out))
            irreps_prev = gate.irreps_out
        self.invariant_features = n_layers * channels
        self.channels = channels
        self.mlp = nn.Sequential(
            nn.Linear(self.invariant_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            f = self.norm_in(x.float().flatten(1) @ self.analysis)
            h = f
            invariants = []
            for tp, gate, norm in zip(self.tps, self.gates, self.norms, strict=True):
                h = norm(gate(tp(h, f)))
                invariants.append(h[:, : self.channels])
            z = torch.cat(invariants, dim=-1)
        return self.mlp(z)
