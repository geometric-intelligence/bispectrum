"""O-equivariant 3D ResNet models for OrganMNIST3D with swappable pooling.

No escnn dependency — group-equivariant 3D convolutions are implemented from
scratch using the octahedral group O (|O| = 24).

Four model variants share the same backbone, differing only in the
invariant-map used:

    1. ``standard``    — vanilla 3D ResNet + data augmentation (no group structure)
    2. ``max_pool``    — equivariant 3D ResNet + max pooling over group dim
    3. ``norm_pool``   — equivariant 3D ResNet + NormReLU + norm pooling
    4. ``bispectrum``  — equivariant 3D ResNet + bispectral invariant pooling
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from bispectrum import OctaonOcta
from bispectrum.octa_on_octa import _CAYLEY, _ELEMENTS_3x3

GROUP_ORDER = 24

NonlinType = Literal['standard', 'max_pool', 'norm_pool', 'bispectrum']


def _make_octahedral_group() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (elements, cayley, inverses) for the octahedral group O.

    elements:  (24, 3, 3) — rotation matrices (integer-valued).
    cayley:    (24, 24)   — cayley[g, h] = index of g·h.
    inverses:  (24,)      — inverses[g] = index of g⁻¹.
    """
    elements = _ELEMENTS_3x3.float()
    cayley = _CAYLEY.clone()

    inverses = torch.zeros(GROUP_ORDER, dtype=torch.long)
    identity_idx = GROUP_ORDER - 1  # g23 is identity
    for g in range(GROUP_ORDER):
        for h in range(GROUP_ORDER):
            if cayley[g, h] == identity_idx:
                inverses[g] = h
                break

    return elements, cayley, inverses


def _precompute_kernel_permutations(
    elements: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """Precompute voxel index permutations for rotating a K³ kernel by each group element.

    Octahedral rotations are signed coordinate permutations, so rotating a
    kernel centered at the origin maps integer coordinates to integer
    coordinates exactly — no interpolation needed.

    Returns: (|G|, K³) long tensor mapping source voxel index to destination.
    """
    K = kernel_size
    half = (K - 1) / 2.0

    coords = torch.stack(
        torch.meshgrid(
            torch.arange(K, dtype=torch.float32) - half,
            torch.arange(K, dtype=torch.float32) - half,
            torch.arange(K, dtype=torch.float32) - half,
            indexing='ij',
        ),
        dim=-1,
    ).reshape(-1, 3)

    G = elements.shape[0]
    perms = torch.zeros(G, K**3, dtype=torch.long)

    for g in range(G):
        R = elements[g].float()
        new_coords = (R @ coords.T).T
        ni = (new_coords[:, 0] + half).round().long()
        nj = (new_coords[:, 1] + half).round().long()
        nk = (new_coords[:, 2] + half).round().long()
        perms[g] = ni * K * K + nj * K + nk

    return perms


class LiftingConv3d(nn.Module):
    """Lift spatial input to the group: (B, C_in, D, H, W) -> (B, C_out, |O|, D, H, W)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int | str = 'same',
    ):
        super().__init__()
        self.out_channels = out_channels
        elements, cayley, inverses = _make_octahedral_group()
        self.register_buffer('elements', elements)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.padding = padding

        perms = _precompute_kernel_permutations(elements, kernel_size)
        inv_perms = perms[inverses]
        self.register_buffer('inv_perms', inv_perms)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C_out, C_in, K, _, _ = self.weight.shape
        G = GROUP_ORDER

        w_flat = self.weight.reshape(C_out * C_in, K**3)
        w_all = w_flat[:, self.inv_perms]      # (C_out*C_in, G, K³) — R_{g⁻¹} rotation
        w_all = w_all.permute(1, 0, 2)         # (G, C_out*C_in, K³)
        w_all = w_all.reshape(G * C_out, C_in, K, K, K)

        out = F.conv3d(x, w_all, padding=self.padding)
        B = x.shape[0]
        D, H, W = out.shape[2:]
        return out.reshape(B, G, C_out, D, H, W).permute(0, 2, 1, 3, 4, 5)


class GroupConv3d(nn.Module):
    """Group convolution: (B, C_in, |O|, D, H, W) -> (B, C_out, |O|, D, H, W)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int | str = 'same',
    ):
        super().__init__()
        elements, cayley, inverses = _make_octahedral_group()
        self.register_buffer('elements', elements)
        self.register_buffer('cayley', cayley)
        self.register_buffer('inverses', inverses)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, GROUP_ORDER, kernel_size, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.padding = padding

        perms = _precompute_kernel_permutations(elements, kernel_size)
        inv_perms = perms[inverses]
        self.register_buffer('inv_perms', inv_perms)

        all_reindex = cayley[inverses]
        self.register_buffer('all_reindex', all_reindex)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, G, D, H, W = x.shape
        C_out = self.out_channels
        K = self.kernel_size

        w_reindexed = self.weight[:, :, self.all_reindex]
        w_reindexed = w_reindexed.permute(2, 0, 1, 3, 4, 5, 6)

        if K == 1:
            w_rot = w_reindexed.reshape(G * C_out, C_in * G, 1, 1, 1)
        else:
            w_flat = w_reindexed.reshape(G, C_out, C_in * G, K**3)
            idx = self.inv_perms[:, None, None, :].expand_as(w_flat)
            w_rotated = w_flat.gather(-1, idx)
            w_rot = w_rotated.reshape(G * C_out, C_in * G, K, K, K)

        x_flat = x.reshape(B, C_in * G, D, H, W)
        out = F.conv3d(x_flat, w_rot, padding=self.padding)
        return out.reshape(B, G, C_out, D, H, W).permute(0, 2, 1, 3, 4, 5)


class EquivBatchNorm3d(nn.Module):
    """Batch norm that shares statistics across the group dimension.

    Input: (B, C, |O|, D, H, W).  Normalises over (B, |O|, D, H, W).
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, G, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 5, 1).reshape(B * G * D * H * W, C)
        x = self.bn(x)
        return x.reshape(B, G, D, H, W, C).permute(0, 5, 1, 2, 3, 4)


class NormReLU3d(nn.Module):
    """Norm nonlinearity: ReLU(||x||_G - bias) * x / ||x||_G."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=2, keepdim=True)
        bias = self.bias.reshape(1, -1, 1, 1, 1, 1)
        gate = F.relu(norm - bias)
        return x * (gate / (norm + 1e-8))


class BispectrumPool3d(nn.Module):
    """Invariant pooling via the selective bispectrum (OctaonOcta).

    Input:  (B, C, 24, D, H, W)
    Output: (B, head_dim, D, H, W) — invariant (no group dim).
    """

    def __init__(self, num_channels: int, head_dim: int = 64):
        super().__init__()
        self.bispec = OctaonOcta(selective=True)
        self.features_per_channel = self.bispec.output_size
        self.proj = nn.Conv3d(num_channels * self.features_per_channel, head_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, G, D, H, W = x.shape
        with torch.amp.autocast('cuda', enabled=False):
            x_flat = x.float().permute(0, 1, 3, 4, 5, 2).reshape(-1, G)
            beta = self.bispec(x_flat)

            out = beta.real
            out = torch.sign(out) * torch.log1p(out.abs())
            out = out.reshape(B, C, D, H, W, -1).permute(0, 1, 5, 2, 3, 4)
            out = out.reshape(B, C * self.features_per_channel, D, H, W)
        return self.proj(F.relu(out))


class GroupMaxPool3d(nn.Module):
    """Max-pool over the group dimension: (B, C, |O|, D, H, W) -> (B, C, D, H, W)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=2).values


class GroupNormPool3d(nn.Module):
    """L2-norm over the group dimension: (B, C, |O|, D, H, W) -> (B, C, D, H, W)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.norm(dim=2)


class _BasicBlock(nn.Module):
    """GroupConv3d + EquivBN + nonlin + GroupConv3d + EquivBN + skip + nonlin."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        equivariant: bool,
        nonlin_type: str,
    ):
        super().__init__()
        self.equivariant = equivariant

        if equivariant:
            self.conv1 = GroupConv3d(channels, channels, kernel_size, padding='same')
            self.bn1 = EquivBatchNorm3d(channels)
            self.conv2 = GroupConv3d(channels, channels, kernel_size, padding='same')
            self.bn2 = EquivBatchNorm3d(channels)
            self.nonlin1 = _make_nonlinearity(nonlin_type, channels)
            self.nonlin2 = _make_nonlinearity(nonlin_type, channels)
        else:
            self.conv1 = nn.Conv3d(channels, channels, kernel_size, padding='same')
            self.bn1 = nn.BatchNorm3d(channels)
            self.conv2 = nn.Conv3d(channels, channels, kernel_size, padding='same')
            self.bn2 = nn.BatchNorm3d(channels)
            self.nonlin1 = nn.ReLU(inplace=True)
            self.nonlin2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.nonlin1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.nonlin2(out)


def _make_nonlinearity(nonlin_type: str, channels: int) -> nn.Module:
    if nonlin_type == 'norm_pool':
        return NormReLU3d(channels)
    else:
        return _RegularReLU()


class _RegularReLU(nn.Module):
    """Pointwise ReLU — equivariant for regular representations (group acts by permutation)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


class Organ3DResNet(nn.Module):
    """3D ResNet for OrganMNIST3D with configurable equivariance and pooling.

    Args:
        nonlin_type: One of ``"standard"``, ``"max_pool"``, ``"norm_pool"``,
            ``"bispectrum"``.
        channels: Tuple of (C0, C1) channel widths for stages.
        head_dim: Output dim of bispectral pool projection.
    """

    def __init__(
        self,
        nonlin_type: NonlinType = 'bispectrum',
        channels: tuple[int, int] = (4, 8),
        head_dim: int = 64,
    ):
        super().__init__()
        self.nonlin_type = nonlin_type
        self.equivariant = nonlin_type != 'standard'
        C0, C1 = channels

        if self.equivariant:
            self.stem_conv = LiftingConv3d(1, C0, 3, padding='same')
            self.stem_bn = EquivBatchNorm3d(C0)
        else:
            self.stem_conv = nn.Conv3d(1, C0, 3, padding=1, bias=False)
            self.stem_bn = nn.BatchNorm3d(C0)

        self.stage1 = nn.Sequential(
            _BasicBlock(C0, 3, self.equivariant, nonlin_type),
            _BasicBlock(C0, 3, self.equivariant, nonlin_type),
        )

        if self.equivariant:
            self.proj = GroupConv3d(C0, C1, 1, padding=0)
            self.proj_bn = EquivBatchNorm3d(C1)
        else:
            self.proj = nn.Conv3d(C0, C1, 1, bias=False)
            self.proj_bn = nn.BatchNorm3d(C1)

        self.stage2 = nn.Sequential(
            _BasicBlock(C1, 3, self.equivariant, nonlin_type),
            _BasicBlock(C1, 3, self.equivariant, nonlin_type),
        )

        self.stage3 = nn.Sequential(
            _BasicBlock(C1, 3, self.equivariant, nonlin_type),
            _BasicBlock(C1, 3, self.equivariant, nonlin_type),
        )

        if self.equivariant:
            self.final_bn = EquivBatchNorm3d(C1)
        else:
            self.final_bn = nn.BatchNorm3d(C1)

        if nonlin_type == 'bispectrum':
            self.invariant_pool = BispectrumPool3d(C1, head_dim)
            self._fc_in = head_dim
        elif nonlin_type == 'max_pool':
            self.invariant_pool = GroupMaxPool3d()
            self._fc_in = C1
        elif nonlin_type == 'norm_pool':
            self.invariant_pool = GroupNormPool3d()
            self._fc_in = C1
        else:
            self.invariant_pool = nn.Identity()
            self._fc_in = C1

        self.classifier = nn.Linear(self._fc_in, 11)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem_bn(self.stem_conv(x)))

        x = self.stage1(x)
        if self.equivariant:
            x = F.avg_pool3d(
                x.reshape(x.shape[0], -1, *x.shape[3:]),
                kernel_size=2, stride=2,
            )
            x = x.reshape(x.shape[0], -1, GROUP_ORDER, *x.shape[2:])
        else:
            x = F.avg_pool3d(x, kernel_size=2, stride=2)

        x = F.relu(self.proj_bn(self.proj(x)))

        x = self.stage2(x)
        if self.equivariant:
            x = F.avg_pool3d(
                x.reshape(x.shape[0], -1, *x.shape[3:]),
                kernel_size=2, stride=2,
            )
            x = x.reshape(x.shape[0], -1, GROUP_ORDER, *x.shape[2:])
        else:
            x = F.avg_pool3d(x, kernel_size=2, stride=2)

        x = self.stage3(x)
        x = F.relu(self.final_bn(x))

        x = self.invariant_pool(x)

        if x.ndim == 6:
            x = x.mean(dim=(2, 3, 4, 5))
        elif x.ndim == 5:
            x = x.mean(dim=(2, 3, 4))
        else:
            x = x.mean(dim=tuple(range(2, x.ndim)))

        return self.classifier(x)


def build_model(
    nonlin_type: str,
    channels: tuple[int, int] = (4, 8),
    head_dim: int = 64,
) -> Organ3DResNet:
    """Build an Organ3DResNet with the specified configuration."""
    return Organ3DResNet(
        nonlin_type=nonlin_type,
        channels=channels,
        head_dim=head_dim,
    )
