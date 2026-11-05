"""Spherical MNIST e3nn baselines: quadrature exactness and rotation invariance."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

o3 = pytest.importorskip('e3nn.o3')

SMNIST_DIR = Path(__file__).resolve().parents[1] / 'experiments' / 'spherical_mnist'


def _baselines() -> ModuleType:
    spec = importlib.util.spec_from_file_location('smnist_baselines', SMNIST_DIR / 'baselines.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bl = _baselines()


def _block_wigner(rotation: torch.Tensor, lmax: int) -> torch.Tensor:
    angles = o3.matrix_to_angles(rotation)
    return torch.block_diag(*[o3.wigner_D(lv, *angles) for lv in range(lmax + 1)])


def test_real_spherical_harmonics_match_e3nn() -> None:
    alpha = torch.rand(40, dtype=torch.float64) * 2 * torch.pi
    beta = torch.rand(40, dtype=torch.float64) * torch.pi
    ours = bl.real_spherical_harmonics(12, alpha, beta)
    ref = o3.spherical_harmonics(list(range(13)), o3.angles_to_xyz(alpha, beta), True, 'integral')
    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=0)


@pytest.mark.parametrize('lmax', [9, 15])
def test_sphere_quadrature_is_orthonormal(lmax: int) -> None:
    alpha, beta, _ = bl.equiangular_quadrature(64, 128)
    gram = bl.sphere_analysis_matrix(lmax, 64, 128).T @ bl.real_spherical_harmonics(
        lmax, alpha, beta
    )
    torch.testing.assert_close(gram, torch.eye(gram.shape[0], dtype=gram.dtype), atol=1e-5, rtol=0)


@pytest.mark.parametrize('b', [6, 10])
def test_soft_quadrature_is_exact(b: int) -> None:
    alpha, beta, gamma, w = bl.so3_soft_grid(b)
    d = bl.flat_wigner(b - 1, alpha, beta, gamma)
    inv_dim = torch.cat([torch.full(((2 * lv + 1) ** 2,), 1.0 / (2 * lv + 1)) for lv in range(b)])
    torch.testing.assert_close(
        d.T @ (d * w[:, None]), torch.diag(inv_dim.double()), atol=1e-6, rtol=0
    )


@pytest.mark.parametrize('batch_norm', [False, True])
def test_s2cnn_invariant_to_grid_rotation(batch_norm: bool) -> None:
    torch.manual_seed(0)
    model = bl.S2CNN(batch_norm=batch_norm).eval()
    x = torch.rand(3, 64, 128)
    with torch.no_grad():
        ref = model(x)
        quarter_turn = model(torch.roll(x, 32, dims=-1))
        off_grid = model(torch.roll(x, 5, dims=-1))
    torch.testing.assert_close(quarter_turn, ref, atol=1e-5, rtol=0)
    assert (off_grid - ref).abs().max() < 1e-2 * ref.abs().max()


def test_tensor_product_net_exactly_invariant() -> None:
    torch.manual_seed(0)
    model = bl.TensorProductNet(lmax=6, channels=4, hidden_lmax=3, n_layers=2, hidden=32).eval()
    coeffs = torch.randn(5, 49)
    rotation = _block_wigner(o3.rand_matrix(), 6).float()

    def invariants(f: torch.Tensor) -> torch.Tensor:
        f = model.norm_in(f)
        h, out = f, []
        for tp, gate, norm in zip(model.tps, model.gates, model.norms, strict=True):
            h = norm(gate(tp(h, f)))
            out.append(h[:, : model.channels])
        return torch.cat(out, dim=-1)

    with torch.no_grad():
        torch.testing.assert_close(
            invariants(coeffs @ rotation.T), invariants(coeffs), atol=1e-4, rtol=1e-4
        )
