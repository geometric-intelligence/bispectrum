"""Models for the Spherical MNIST experiment.

Three model variants for comparing rotation invariance on spherical data:

    1. ``bispectrum``      - SO3onS2 augmented invariant features -> MLP
    2. ``power_spectrum``  - SH power spectrum -> MLP (invariant, incomplete)
    3. ``standard``        - CNN on equirectangular image (not rotation-invariant)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch_harmonics import RealSHT

from bispectrum import SO3onS2

NonlinType = Literal['bispectrum', 'power_spectrum', 'standard']
BispectrumFeatureSet = Literal['bootstrap', 'bootstrap_self', 'bootstrap_self_cg', 'full']


class BispectrumClassifier(nn.Module):
    """SO(3) bispectrum features -> MLP classifier.

    Rotation-invariant by construction: the bispectrum is exactly SO(3)-invariant
    (up to discretization of the SHT).
    """

    def __init__(
        self,
        lmax: int = 15,
        nlat: int = 64,
        nlon: int = 128,
        selective: bool = True,
        hidden: int = 256,
        num_classes: int = 10,
        feature_set: BispectrumFeatureSet = 'full',
    ) -> None:
        super().__init__()
        if not selective and feature_set != 'full':
            raise ValueError('Feature-set ablations require selective=True')
        self.bispectrum = SO3onS2(
            lmax=lmax,
            nlat=nlat,
            nlon=nlon,
            selective=selective,
        )
        self.feature_set = feature_set
        feature_mask = self._build_feature_mask(feature_set)
        self.register_buffer('_feature_mask', feature_mask, persistent=False)
        self.active_feature_count = int(feature_mask.sum().item())
        n_features = self.bispectrum.output_size * 2
        self.mlp = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_classes),
        )

    def _build_feature_mask(self, feature_set: BispectrumFeatureSet) -> torch.Tensor:
        """Return a cumulative ablation mask while preserving the MLP width."""
        if feature_set == 'full':
            return torch.ones(self.bispectrum.output_size, dtype=torch.bool)

        mask = torch.zeros(self.bispectrum.output_size, dtype=torch.bool)
        allowed_tags = {'bootstrap'}
        if feature_set in {'bootstrap_self', 'bootstrap_self_cg'}:
            allowed_tags.add('mandatory_even_self')
        for idx, tag in enumerate(self.bispectrum.bispec_feature_tags):
            mask[idx] = tag in allowed_tags
        if feature_set == 'bootstrap_self_cg':
            mask[self.bispectrum.n_bispec :] = True
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            beta = self.bispectrum(x)
            beta = beta * self._feature_mask
            real = torch.sign(beta.real) * torch.log1p(beta.real.abs())
            imag = torch.sign(beta.imag) * torch.log1p(beta.imag.abs())
            features = torch.cat([real, imag], dim=-1)
        return self.mlp(features)


class PowerSpectrumClassifier(nn.Module):
    """SH power spectrum -> MLP classifier.

    Rotation-invariant but incomplete: captures only ||F_l||^2 per degree,
    losing all phase coupling between different l values.
    """

    def __init__(
        self,
        lmax: int = 15,
        nlat: int = 64,
        nlon: int = 128,
        hidden: int = 128,
        num_classes: int = 10,
    ):
        super().__init__()
        self.lmax = lmax
        sht_lmax = lmax + 1
        self._sht = RealSHT(
            nlat,
            nlon,
            lmax=sht_lmax,
            mmax=sht_lmax,
            grid='equiangular',
            norm='ortho',
        )
        self.mlp = nn.Sequential(
            nn.Linear(lmax + 1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            coeffs = self._sht(x)
            B = x.shape[0]
            power = torch.zeros(B, self.lmax + 1, device=x.device)

            for l_val in range(min(self.lmax + 1, coeffs.shape[1])):
                m_max = min(l_val, coeffs.shape[2] - 1)
                power[:, l_val] = coeffs[:, l_val, 0].abs().square()
                if m_max > 0:
                    power[:, l_val] += 2 * (
                        coeffs[:, l_val, 1 : m_max + 1].abs().square().sum(dim=-1)
                    )

            power = torch.log1p(power)

        return self.mlp(power)


class StandardCNN(nn.Module):
    """Simple CNN on the equirectangular (nlat x nlon) image.

    Not rotation-invariant: treats the spherical grid as a planar image
    with translational convolutions.
    """

    def __init__(
        self,
        nlat: int = 64,
        nlon: int = 128,
        num_classes: int = 10,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_model(
    model_type: str,
    lmax: int = 15,
    nlat: int = 64,
    nlon: int = 128,
    selective: bool = True,
    hidden: int = 256,
    bispectrum_features: BispectrumFeatureSet = 'full',
) -> nn.Module:
    """Build a model for the Spherical MNIST experiment."""
    if model_type == 'bispectrum':
        return BispectrumClassifier(
            lmax=lmax,
            nlat=nlat,
            nlon=nlon,
            selective=selective,
            hidden=hidden,
            feature_set=bispectrum_features,
        )
    elif model_type == 'power_spectrum':
        return PowerSpectrumClassifier(
            lmax=lmax,
            nlat=nlat,
            nlon=nlon,
            hidden=max(64, hidden // 2),
        )
    elif model_type == 'standard':
        return StandardCNN(nlat=nlat, nlon=nlon)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
