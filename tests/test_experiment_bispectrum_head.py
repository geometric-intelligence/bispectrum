"""Bispectral pooling heads of the PCam and OrganMNIST3D experiment models.

``BispectrumPool`` (C8 / D4, ``experiments/pcam/model.py``) and ``BispectrumPool3d``
(octahedral group, ``experiments/organ3d/model.py``) pool the group dimension of each
channel with the selective bispectrum, compress the result and project it with a 1x1
convolution. The head has three options (``BispectrumHead``):

* ``nonlin='pre'`` reproduces the original head, ReLU on the signed-log bispectrum before
  the projection; ``'post'`` (default) applies the ReLU after the projection so the sign
  and phase information of the bispectrum reaches the projection.
* ``power`` appends the group power spectrum of each channel.
* ``scale`` compresses with a signed ``log1p`` or a signed degree-root.

Every variant must stay exactly invariant under the group action on the fiber, and the
feature counts and the legacy head must match what the paper's runs used.

The modules are imported by path; the tests are skipped if their dependencies are absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

torch.manual_seed(0)

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / 'experiments'


def _load_model_module(name: str) -> ModuleType:
    path = EXPERIMENTS_DIR / name / 'model.py'
    if not path.exists():
        pytest.skip(f'{path} not present')
    spec = importlib.util.spec_from_file_location(f'_exp_{name}_model', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their defining module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def pcam() -> ModuleType:
    return _load_model_module('pcam')


@pytest.fixture(scope='module')
def organ3d() -> ModuleType:
    return _load_model_module('organ3d')


def _heads(module: ModuleType) -> list[object]:
    head = module.BispectrumHead
    return [
        head(nonlin='pre'),
        head(),
        head(power=True),
        head(power=True, scale='cbrt'),
        head(scale='cbrt'),
    ]


def _signed_log(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


# ---------------------------------------------------------------------------
# PCam: C8 / D4 pooling over (B, C, |G|, H, W)
# ---------------------------------------------------------------------------


def test_pcam_default_head_is_post_relu_without_power(pcam: ModuleType) -> None:
    head = pcam.BispectrumHead()
    assert (head.nonlin, head.power, head.scale) == ('post', False, 'log')


def test_pcam_feature_counts(pcam: ModuleType) -> None:
    assert pcam.BispectrumPool('c8', 3).features_per_channel == 16
    assert pcam.BispectrumPool('c8', 3, pcam.BispectrumHead(power=True)).features_per_channel == 21
    assert pcam.BispectrumPool('d4', 3).features_per_channel == pcam.DnonDn(4).output_size
    with pytest.raises(ValueError, match='c8'):
        pcam.BispectrumPool('d4', 3, pcam.BispectrumHead(power=True))


@pytest.mark.parametrize('shift', [1, 3, 5])
def test_pcam_c8_heads_are_rotation_invariant(pcam: ModuleType, shift: int) -> None:
    x = torch.randn(2, 3, 8, 5, 5)
    for head in _heads(pcam):
        pool = pcam.BispectrumPool('c8', 3, head)
        out = pool(x)
        assert out.shape == (2, 3, 5, 5)
        rotated = pool(torch.roll(x, shift, dims=2))
        torch.testing.assert_close(rotated, out, rtol=1e-4, atol=1e-4), head


def test_pcam_legacy_head_matches_original_formula(pcam: ModuleType) -> None:
    """``nonlin='pre'`` is proj(relu(signed_log([Re beta, Im beta])))."""
    pool = pcam.BispectrumPool('c8', 3, pcam.BispectrumHead(nonlin='pre'))
    x = torch.randn(2, 3, 8, 4, 4)
    beta = pool.bispec(x.permute(0, 1, 3, 4, 2).reshape(-1, 8))
    feats = torch.cat([_signed_log(beta.real), _signed_log(beta.imag)], dim=-1)
    feats = feats.reshape(2, 3, 4, 4, -1).permute(0, 1, 4, 2, 3).reshape(2, 3 * 16, 4, 4)
    expected = pool.proj(torch.relu(feats))
    torch.testing.assert_close(pool(x), expected)


def test_pcam_post_head_keeps_negative_features(pcam: ModuleType) -> None:
    """The post head projects the signed features; the pre head only sees relu(features)."""
    x = torch.randn(2, 3, 8, 4, 4)
    pre = pcam.BispectrumPool('c8', 3, pcam.BispectrumHead(nonlin='pre'))
    post = pcam.BispectrumPool('c8', 3, pcam.BispectrumHead(nonlin='post'))
    post.load_state_dict(pre.state_dict())
    beta = pre.bispec(x.permute(0, 1, 3, 4, 2).reshape(-1, 8))
    assert (beta.real < 0).any(), 'test input must produce negative bispectrum entries'
    assert not torch.allclose(pre(x), post(x))


def test_pcam_power_features_are_the_power_spectrum(pcam: ModuleType) -> None:
    pool = pcam.BispectrumPool('c8', 1, pcam.BispectrumHead(power=True))
    x = torch.randn(1, 1, 8, 1, 1)
    captured: dict[str, torch.Tensor] = {}
    pool.proj.register_forward_hook(lambda _m, inputs, _o: captured.setdefault('in', inputs[0]))
    pool(x)
    feats = captured['in'].reshape(-1)
    power = torch.fft.rfft(x.reshape(8)).abs().square()
    torch.testing.assert_close(feats[16:], torch.log1p(power))


# ---------------------------------------------------------------------------
# OrganMNIST3D: octahedral pooling over (B, C, 24, D, H, W)
# ---------------------------------------------------------------------------


def _octa_translate(x: torch.Tensor, g: int, cayley: torch.Tensor) -> torch.Tensor:
    """Left translation (T_g f)(h) = f(g^-1 h) on the fiber axis (dim 2)."""
    identity = next(e for e in range(24) if bool((cayley[e] == torch.arange(24)).all()))
    g_inv = int((cayley[g] == identity).nonzero()[0])
    return x[:, :, cayley[g_inv]]


def test_organ3d_default_head_is_post_relu_without_power(organ3d: ModuleType) -> None:
    head = organ3d.BispectrumHead()
    assert (head.nonlin, head.power, head.scale) == ('post', False, 'log')


def test_organ3d_feature_counts(organ3d: ModuleType) -> None:
    assert organ3d.BispectrumPool3d(3, 8).features_per_channel == 172
    power = organ3d.BispectrumPool3d(3, 8, organ3d.BispectrumHead(power=True))
    assert power.features_per_channel == 177
    assert power.proj.in_channels == 3 * 177 and power.proj.out_channels == 8


@pytest.mark.parametrize('g', [0, 5, 11, 17])
def test_organ3d_heads_are_rotation_invariant(organ3d: ModuleType, g: int) -> None:
    x = torch.randn(2, 3, 24, 3, 3, 3)
    for head in _heads(organ3d):
        pool = organ3d.BispectrumPool3d(3, 8, head)
        out = pool(x)
        assert out.shape == (2, 8, 3, 3, 3)
        rotated = pool(_octa_translate(x, g, organ3d._CAYLEY))
        torch.testing.assert_close(rotated, out, rtol=1e-4, atol=1e-4), head


def test_organ3d_legacy_head_matches_original_formula(organ3d: ModuleType) -> None:
    pool = organ3d.BispectrumPool3d(3, 8, organ3d.BispectrumHead(nonlin='pre'))
    x = torch.randn(2, 3, 24, 2, 2, 2)
    beta = pool.bispec(x.permute(0, 1, 3, 4, 5, 2).reshape(-1, 24)).real
    feats = _signed_log(beta).reshape(2, 3, 2, 2, 2, -1).permute(0, 1, 5, 2, 3, 4)
    expected = pool.proj(torch.relu(feats.reshape(2, 3 * 172, 2, 2, 2)))
    torch.testing.assert_close(pool(x), expected)


def test_organ3d_power_features_are_irrep_norms(organ3d: ModuleType) -> None:
    pool = organ3d.BispectrumPool3d(1, 8, organ3d.BispectrumHead(power=True))
    x = torch.randn(1, 1, 24, 1, 1, 1)
    captured: dict[str, torch.Tensor] = {}
    pool.proj.register_forward_hook(lambda _m, inputs, _o: captured.setdefault('in', inputs[0]))
    pool(x)
    feats = captured['in'].reshape(-1)
    fhat = pool.bispec._group_dft(x.reshape(1, 24))
    power = torch.stack([fk.square().sum(dim=(-2, -1)) for fk in fhat], dim=-1).reshape(-1)
    torch.testing.assert_close(feats[172:], torch.log1p(power))
    # Parseval on O: sum_k d_k ||F_k||^2 = |G| sum_g f(g)^2.
    dims = torch.tensor([1.0, 3.0, 3.0, 2.0, 1.0])
    torch.testing.assert_close((dims * power).sum(), 24 * x.square().sum(), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('name', ['pcam', 'organ3d'])
def test_compress_is_odd_monotone_and_differentiable_at_zero(name: str) -> None:
    module = _load_model_module(name)
    x = torch.linspace(-5, 5, 101, requires_grad=True)
    for scale, degree in (('log', 3), ('cbrt', 3), ('cbrt', 2)):
        y = module._compress(x, scale, degree)
        torch.testing.assert_close(y, -module._compress(-x, scale, degree))
        assert (y[1:] > y[:-1]).all(), f'{scale} degree {degree} not increasing'
        (grad,) = torch.autograd.grad(y.sum(), x)
        assert torch.isfinite(grad).all()
    zero = torch.zeros(4, requires_grad=True)
    (grad,) = torch.autograd.grad(module._compress(zero, 'cbrt', 3).sum(), zero)
    assert torch.isfinite(grad).all()


def test_cbrt_scale_is_linear_in_the_input(pcam: ModuleType) -> None:
    """Scaling the signal by s scales the cube-rooted bispectrum by ~s (log does not)."""
    x = torch.randn(1, 1, 8, 1, 1) * 3
    cbrt = pcam.BispectrumPool('c8', 1, pcam.BispectrumHead(scale='cbrt'))
    captured: list[torch.Tensor] = []
    cbrt.proj.register_forward_hook(lambda _m, inputs, _o: captured.append(inputs[0]))
    cbrt(x)
    cbrt(2 * x)
    ratio = captured[1] / captured[0]
    mask = captured[0].abs() > 0.5
    assert torch.allclose(ratio[mask], torch.full_like(ratio[mask], 2.0), rtol=0.05)
