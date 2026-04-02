"""Equivariant DenseNet models for PatchCamelyon with swappable nonlinearities.

No escnn dependency — group-equivariant convolutions are implemented from
scratch following Cohen & Welling (2016).

Five model variants share the same backbone, differing only in the
nonlinearity / invariant-map used:

    1. ``standard``   — vanilla DenseNet + data augmentation (no group structure)
    2. ``norm``       — equivariant DenseNet + NormReLU nonlinearity
    3. ``gate``       — equivariant DenseNet + gated nonlinearity
    4. ``fourier_elu``— equivariant DenseNet + FFT→ELU→IFFT nonlinearity
    5. ``bispectrum`` — equivariant DenseNet + bispectral invariant pooling
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from bispectrum import CnonCn, DnonDn

GroupName = Literal['c8', 'd4']


def _rotation_matrix(angle: float) -> torch.Tensor:
    """2×2 rotation matrix for *angle* (radians), float32."""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)


def _make_group(name: GroupName):
    """Return (elements, cayley, inverses) tensors for a discrete 2-D group.

    elements:  (|G|, 2, 2) — transformation matrices.
    cayley:    (|G|, |G|)  — cayley_table[g, h] = index of g·h.
    inverses:  (|G|,)      — inverses[g] = index of g⁻¹.
    """
    if name == 'c8':
        n = 8
        elements = torch.stack([_rotation_matrix(2 * math.pi * k / n) for k in range(n)])
    elif name == 'd4':
        # D4 = {r^k, r^k·s} for k=0..3, where r=90° rotation, s=reflection.
        rots = [_rotation_matrix(math.pi / 2 * k) for k in range(4)]
        refl = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
        refs = [r @ refl for r in rots]
        elements = torch.stack(rots + refs)  # (8, 2, 2)
        n = 8
    else:
        raise ValueError(f'Unknown group: {name}')

    # Build Cayley table by matching g·h to known elements.
    cayley = torch.zeros(n, n, dtype=torch.long)
    inverses = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        for j in range(n):
            prod = elements[i] @ elements[j]
            dists = (elements - prod.unsqueeze(0)).norm(dim=(1, 2))
            cayley[i, j] = dists.argmin()
        inv_mat = elements[i].inverse()
        dists = (elements - inv_mat.unsqueeze(0)).norm(dim=(1, 2))
        inverses[i] = dists.argmin()

    return elements, cayley, inverses


def _make_rotated_grids(elements: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Precompute sampling grids for rotating a (K, K) kernel by each group element.

    Returns: (|G|, K, K, 2) — grids for ``F.grid_sample``.
    """
    G = elements.shape[0]
    K = kernel_size
    coords = torch.linspace(-1, 1, K)
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    base = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (K*K, 2)

    grids = []
    for g in range(G):
        # Inverse rotation: to sample the rotated kernel we pull from the
        # inverse-rotated coordinate.
        R_inv = elements[g].T  # orthogonal → inverse = transpose
        rotated = (R_inv @ base.T).T  # (K*K, 2)
        grids.append(rotated.reshape(K, K, 2))
    return torch.stack(grids)  # (G, K, K, 2)


def _rotate_kernel(weight: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Rotate a (C_out, C_in, K, K) kernel using a (K, K, 2) sampling grid.

    Returns rotated kernel of the same shape.
    """
    C_out, C_in, K, _ = weight.shape
    # grid_sample expects (N, H, W, 2) and input (N, C, H, W).
    w = weight.reshape(C_out * C_in, 1, K, K)
    g = grid.unsqueeze(0).expand(C_out * C_in, -1, -1, -1)
    w_rot = F.grid_sample(w, g, mode='bilinear', padding_mode='zeros', align_corners=True)
    return w_rot.reshape(C_out, C_in, K, K)


class LiftingConv2d(nn.Module):
    """Lift spatial input to the group: (B, C_in, H, W) → (B, C_out, |G|, H, W).

    Vectorized: all |G| rotated kernels are computed in a single grid_sample
    call, then applied in a single conv2d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group: GroupName,
        padding: int | str = 'same',
    ):
        super().__init__()
        self.out_channels = out_channels
        elements, cayley, inverses = _make_group(group)
        self.register_buffer('elements', elements)
        self.register_buffer('cayley', cayley)
        self.register_buffer('inverses', inverses)
        self.group_order = elements.shape[0]

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.padding = padding

        grids = _make_rotated_grids(elements, kernel_size)
        self.register_buffer('grids', grids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C_out, C_in, K, _ = self.weight.shape
        G = self.group_order
        n = C_out * C_in

        w = self.weight.reshape(n, 1, K, K).expand(G, -1, -1, -1, -1).reshape(G * n, 1, K, K)
        g = self.grids.unsqueeze(1).expand(-1, n, -1, -1, -1).reshape(G * n, K, K, 2)
        w_rot = F.grid_sample(w, g, mode='bilinear', padding_mode='zeros', align_corners=True)
        w_rot = w_rot.reshape(G * C_out, C_in, K, K)

        out = F.conv2d(x, w_rot, padding=self.padding)
        B, _, H, W = out.shape
        return out.reshape(B, G, C_out, H, W).permute(0, 2, 1, 3, 4)


class GroupConv2d(nn.Module):
    """Group convolution: (B, C_in, |G|, H, W) → (B, C_out, |G|, H, W).

    Vectorized: all |G| reindexed+rotated kernels are computed in a single
    grid_sample call, then applied in a single conv2d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group: GroupName,
        padding: int | str = 'same',
    ):
        super().__init__()
        elements, cayley, inverses = _make_group(group)
        self.register_buffer('elements', elements)
        self.register_buffer('cayley', cayley)
        self.register_buffer('inverses', inverses)
        self.group_order = elements.shape[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, self.group_order, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.padding = padding

        grids = _make_rotated_grids(elements, kernel_size)
        self.register_buffer('grids', grids)

        # Precompute all group reindexing patterns: (G, G).
        all_reindex = cayley[inverses]
        self.register_buffer('all_reindex', all_reindex)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, G, H, W = x.shape
        C_out = self.out_channels
        K = self.kernel_size

        # Reindex weight for all group elements at once.
        # weight[:, :, all_reindex] → (C_out, C_in, G_out, G_in, K, K)
        w_reindexed = self.weight[:, :, self.all_reindex]
        # Rearrange to (G_out, C_out, C_in, G_in, K, K)
        w_reindexed = w_reindexed.permute(2, 0, 1, 3, 4, 5)

        if K == 1:
            w_rot = w_reindexed.reshape(G * C_out, C_in * G, 1, 1)
        else:
            n = C_out * C_in * G
            w_flat = w_reindexed.reshape(G * n, 1, K, K)
            g_expanded = self.grids.unsqueeze(1).expand(-1, n, -1, -1, -1).reshape(G * n, K, K, 2)
            w_rot = F.grid_sample(
                w_flat, g_expanded, mode='bilinear', padding_mode='zeros', align_corners=True,
            )
            w_rot = w_rot.reshape(G * C_out, C_in * G, K, K)

        x_flat = x.reshape(B, C_in * G, H, W)
        out = F.conv2d(x_flat, w_rot, padding=self.padding)
        return out.reshape(B, G, C_out, H, W).permute(0, 2, 1, 3, 4)


class EquivBatchNorm(nn.Module):
    """Batch norm that shares statistics across the group dimension.

    Input: (B, C, |G|, H, W).  Normalises over (B, |G|, H, W).
    Learnable affine per channel only.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, G, H, W = x.shape
        # Merge B, G, H, W into a single batch dim for BN.
        x = x.permute(0, 2, 3, 4, 1).reshape(B * G * H * W, C)
        x = self.bn(x)
        return x.reshape(B, G, H, W, C).permute(0, 4, 1, 2, 3)


class NormReLU(nn.Module):
    """Norm nonlinearity: ReLU(||x||_G - bias) * x / ||x||_G.

    Computes the norm over the group dimension, applies a biased ReLU to it,
    then rescales the original feature by the ratio. This preserves direction
    (equivariance) while applying a nonlinear gating on magnitude.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, |G|, H, W)
        norm = x.norm(dim=2, keepdim=True)  # (B, C, 1, H, W)
        bias = self.bias.reshape(1, -1, 1, 1, 1)
        gate = F.relu(norm - bias)
        return x * (gate / (norm + 1e-8))


class GatedNonlinearity(nn.Module):
    """Gated nonlinearity: splits channels into features + gates.

    The first ``out_channels`` channels are feature channels.
    The remaining ``out_channels`` channels are gate channels: their group-norm
    is passed through sigmoid and multiplied into the feature channels.

    Input channels must be 2 * out_channels.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2*C, |G|, H, W)
        feat = x[:, : self.out_channels]
        gate_input = x[:, self.out_channels :]
        gate = torch.sigmoid(gate_input.norm(dim=2, keepdim=True))
        return feat * gate


class FourierELU(nn.Module):
    """Approximate equivariant ELU via FFT → upsample → ELU → downsample → IFFT.

    Following Franzen & Wand (NeurIPS 2021).
    """

    def __init__(self, group_order: int, upsample_factor: int = 2):
        super().__init__()
        self.n = group_order
        self.n_up = group_order * upsample_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, |G|, H, W)
        # Disable autocast: ComplexHalf is broken and wastes memory.
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            X = torch.fft.rfft(x, dim=2)
            n_freq = X.shape[2]
            pad_size = (self.n_up // 2 + 1) - n_freq
            if pad_size > 0:
                X = F.pad(X, (0, 0, 0, 0, 0, pad_size))
            x_up = torch.fft.irfft(X, n=self.n_up, dim=2)
            x_up = F.elu(x_up)
            X2 = torch.fft.rfft(x_up, dim=2)
            X2 = X2[:, :, : self.n // 2 + 1]
            return torch.fft.irfft(X2, n=self.n, dim=2)


class BispectrumPool(nn.Module):
    """Invariant pooling via the selective bispectrum.

    Applies CnonCn (for cyclic groups) or DnonDn (for dihedral groups) along
    the group dimension of each channel at each spatial location.

    Input:  (B, C, |G|, H, W)
    Output: (B, C * n_bispec_features, H, W) — invariant (no group dim).
    """

    def __init__(self, group: GroupName, num_channels: int):
        super().__init__()
        self.group = group
        if group == 'c8':
            self.bispec = CnonCn(n=8, selective=True)
        elif group == 'd4':
            self.bispec = DnonDn(n=4, selective=True)
        else:
            raise ValueError(f'Unknown group: {group}')

        # Number of real-valued output features per input channel.
        if group == 'c8':
            # CnonCn(8) produces 8 complex values → 16 real.
            self.features_per_channel = self.bispec.output_size * 2
        else:
            # DnonDn output is already real.
            self.features_per_channel = self.bispec.output_size

        # 1×1 projection to keep channel count manageable.
        self.proj = nn.Conv2d(num_channels * self.features_per_channel, num_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, G, H, W = x.shape
        with torch.amp.autocast('cuda', enabled=False):
            x_flat = x.float().permute(0, 1, 3, 4, 2).reshape(-1, G)
            beta = self.bispec(x_flat)

            if beta.is_complex():
                beta_real = beta.real
                beta_imag = beta.imag
                beta_real = torch.sign(beta_real) * torch.log1p(beta_real.abs())
                beta_imag = torch.sign(beta_imag) * torch.log1p(beta_imag.abs())
                out = torch.cat([beta_real, beta_imag], dim=-1)
            else:
                out = torch.sign(beta) * torch.log1p(beta.abs())

            out = out.reshape(B, C, H, W, -1).permute(0, 1, 4, 2, 3)
            out = out.reshape(B, C * self.features_per_channel, H, W)
        return self.proj(F.relu(out))


class GroupMaxPool(nn.Module):
    """Max-pool over the group dimension: (B, C, |G|, H, W) → (B, C, H, W)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=2).values


class GroupNormPool(nn.Module):
    """L2-norm over the group dimension: (B, C, |G|, H, W) → (B, C, H, W)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.norm(dim=2)


class _DenseLayer(nn.Module):
    """BN → nonlin → 1×1 conv → BN → nonlin → 3×3 conv (bottleneck)."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        group: GroupName | None,
        nonlin_type: str,
    ):
        super().__init__()
        self.equivariant = group is not None
        bn_factor = 4
        inter = bn_factor * growth_rate

        if self.equivariant:
            group_order = 8  # both C8 and D4 have order 8
            # For gated nonlinearity, conv outputs 2× channels (half for gates).
            gate_factor = 2 if nonlin_type == 'gate' else 1

            self.bn1 = EquivBatchNorm(in_channels)
            self.conv1 = GroupConv2d(in_channels, inter * gate_factor, 1, group, padding=0)
            self.nonlin1 = _make_nonlinearity(nonlin_type, inter * gate_factor, group_order, inter)

            self.bn2 = EquivBatchNorm(inter)
            self.conv2 = GroupConv2d(inter, growth_rate * gate_factor, 3, group, padding='same')
            self.nonlin2 = _make_nonlinearity(
                nonlin_type, growth_rate * gate_factor, group_order, growth_rate
            )
        else:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.conv1 = nn.Conv2d(in_channels, inter, 1, bias=False)
            self.nonlin1 = nn.ReLU(inplace=True)

            self.bn2 = nn.BatchNorm2d(inter)
            self.conv2 = nn.Conv2d(inter, growth_rate, 3, padding=1, bias=False)
            self.nonlin2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.nonlin1(self.conv1(self.bn1(x)))
        out = self.nonlin2(self.conv2(self.bn2(out)))
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        group: GroupName | None,
        nonlin_type: str,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                _DenseLayer(in_channels + i * growth_rate, growth_rate, group, nonlin_type)
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nn.Module):
    """Channel compression + spatial downsampling between dense blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: GroupName | None,
        nonlin_type: str,
    ):
        super().__init__()
        self.equivariant = group is not None

        if self.equivariant:
            group_order = 8
            gate_factor = 2 if nonlin_type == 'gate' else 1
            self.bn = EquivBatchNorm(in_channels)
            self.conv = GroupConv2d(in_channels, out_channels * gate_factor, 1, group, padding=0)
            self.nonlin = _make_nonlinearity(
                nonlin_type, out_channels * gate_factor, group_order, out_channels
            )
        else:
            self.bn = nn.BatchNorm2d(in_channels)
            self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.nonlin = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.nonlin(self.conv(self.bn(x)))
        if self.equivariant:
            x = F.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        else:
            x = F.avg_pool2d(x, 2)
        return x


def _make_nonlinearity(
    nonlin_type: str,
    in_channels: int,
    group_order: int,
    out_channels: int | None = None,
) -> nn.Module:
    """Factory for equivariant nonlinearities."""
    if nonlin_type == 'norm':
        return NormReLU(in_channels)
    elif nonlin_type == 'gate':
        return GatedNonlinearity(out_channels or in_channels // 2)
    elif nonlin_type == 'fourier_elu':
        return FourierELU(group_order)
    elif nonlin_type == 'bispectrum':
        # For intermediate layers, use ReLU (equivariant for regular repr).
        # Bispectrum is applied only as the final invariant pool.
        return _RegularReLU()
    else:
        raise ValueError(f'Unknown nonlinearity: {nonlin_type}')


class _RegularReLU(nn.Module):
    """Pointwise ReLU on features in the regular representation.

    For finite groups with regular representation, pointwise ReLU is equivariant because the group
    action is a permutation of channels.
    """

    def forward(self, x):
        return F.relu(x)


NonlinType = Literal['standard', 'norm', 'gate', 'fourier_elu', 'bispectrum']


class PCamDenseNet(nn.Module):
    """DenseNet for PCam with configurable equivariance and nonlinearity.

    Args:
        nonlin_type: One of ``"standard"``, ``"norm"``, ``"gate"``,
            ``"fourier_elu"``, ``"bispectrum"``.
        group: ``"c8"`` or ``"d4"``.  Ignored when ``nonlin_type="standard"``.
        growth_rate: DenseNet growth rate *k*.
        block_config: Number of layers in each dense block.
        init_channels: Channels after the initial convolution.
        compression: Channel reduction factor at transitions.
    """

    def __init__(
        self,
        nonlin_type: NonlinType = 'bispectrum',
        group: GroupName = 'c8',
        growth_rate: int = 12,
        block_config: tuple[int, ...] = (4, 4, 4),
        init_channels: int = 24,
        compression: float = 0.5,
    ):
        super().__init__()
        self.nonlin_type = nonlin_type
        self.equivariant = nonlin_type != 'standard'
        self.group = group if self.equivariant else None
        g = self.group

        if self.equivariant:
            self.init_conv = LiftingConv2d(3, init_channels, 3, group, padding='same')
            self.init_bn = EquivBatchNorm(init_channels)
        else:
            self.init_conv = nn.Conv2d(3, init_channels, 3, padding=1, bias=False)
            self.init_bn = nn.BatchNorm2d(init_channels)

        channels = init_channels
        blocks = []
        for i, num_layers in enumerate(block_config):
            blocks.append(_DenseBlock(num_layers, channels, growth_rate, g, nonlin_type))
            channels += num_layers * growth_rate
            if i < len(block_config) - 1:
                out_ch = int(channels * compression)
                blocks.append(_Transition(channels, out_ch, g, nonlin_type))
                channels = out_ch

        self.blocks = nn.Sequential(*blocks)

        if self.equivariant:
            self.final_bn = EquivBatchNorm(channels)
        else:
            self.final_bn = nn.BatchNorm2d(channels)

        if nonlin_type == 'bispectrum':
            self.invariant_pool = BispectrumPool(group, channels)
        elif self.equivariant:
            self.invariant_pool = GroupMaxPool()
        else:
            self.invariant_pool = nn.Identity()

        self.classifier = nn.Linear(channels, 1)

        self._channels_before_fc = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = F.relu(x)
        x = self.blocks(x)
        x = self.final_bn(x)
        x = F.relu(x)

        # Invariant pooling: collapse group dim (if equivariant).
        x = self.invariant_pool(x)

        # Global average pool over spatial dims.
        if x.ndim == 5:
            # Still has group dim (shouldn't happen but just in case).
            x = x.mean(dim=(2, 3, 4))
        else:
            x = x.mean(dim=(2, 3))

        return self.classifier(x).squeeze(-1)


def build_model(
    nonlin_type: str,
    group: str = 'c8',
    growth_rate: int = 12,
    block_config: tuple[int, ...] = (4, 4, 4),
) -> PCamDenseNet:
    """Build a PCamDenseNet with the specified configuration."""
    return PCamDenseNet(
        nonlin_type=nonlin_type,
        group=group,
        growth_rate=growth_rate,
        block_config=block_config,
    )
