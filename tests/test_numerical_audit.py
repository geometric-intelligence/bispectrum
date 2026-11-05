"""Numerical audit regressions, including independent group-sum references."""

import math

import pytest
import torch

from bispectrum import CnonCn, DnonDn, OctaonOcta, SO2onS1, TorusOnTorus
from bispectrum._bessel import bessel_jn
from bispectrum._cg import wigner3j
from bispectrum.octa_on_octa import _CAYLEY, _INVERSE


@pytest.mark.parametrize('n', [0, 1, 2, 4])
def test_bessel_derivative_near_origin(n: int):
    """Derivative identity provides an independent oracle near the origin."""
    x = torch.tensor([-1e-10, 0.0, 1e-10], dtype=torch.float64, requires_grad=True)
    actual = torch.autograd.grad(bessel_jn(n, x).sum(), x)[0]
    if n == 0:
        expected = -x.detach() / 2
    elif n == 1:
        expected = torch.full_like(x, 0.5)
    elif n == 2:
        expected = x.detach() / 4
    else:
        expected = x.detach().pow(n - 1) / (2**n * math.factorial(n - 1))
    torch.testing.assert_close(actual, expected, atol=1e-35, rtol=1e-12)


def test_bessel_second_derivative_at_origin():
    x = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    derivative = torch.autograd.grad(bessel_jn(0, x), x, create_graph=True)[0]
    second = torch.autograd.grad(derivative, x)[0]
    torch.testing.assert_close(second, torch.tensor(-0.5, dtype=torch.float64))


@pytest.mark.parametrize('module', [CnonCn(5), SO2onS1(5), TorusOnTorus((3, 4))])
def test_abelian_inverse_gradcheck(module):
    """Check complex inverse derivatives against finite differences, not only finiteness."""
    generator = torch.Generator().manual_seed(285)
    shape = (2, *module.ns) if isinstance(module, TorusOnTorus) else (2, module.n)
    beta = module(torch.randn(shape, dtype=torch.float64, generator=generator) + 0.4)
    beta = beta.detach().requires_grad_()
    assert torch.autograd.gradcheck(module.invert, (beta,), fast_mode=True)


@pytest.mark.parametrize('n', [4, 5])
def test_dihedral_inverse_composition_gradient(n: int):
    """A finite group orbit preserves norm; inverse composition must have gradient 2f."""
    generator = torch.Generator().manual_seed(1283)
    f = torch.randn(1, 2 * n, dtype=torch.float64, generator=generator, requires_grad=True)
    module = DnonDn(n)
    recovered = module.invert(module(f))
    actual = torch.autograd.grad(recovered.square().sum(), f)[0]
    torch.testing.assert_close(actual, 2 * f, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize('scale', [1e-4, 1e-6, 1e-8])
def test_torus_inverse_does_not_clamp_valid_small_pivot_products(scale: float):
    module = TorusOnTorus((3, 4))
    generator = torch.Generator().manual_seed(491)
    f = scale * (torch.randn(2, 3, 4, dtype=torch.float64, generator=generator) + 0.4)
    recovered = module.invert(module(f))
    expected = torch.fft.fftn(f, dim=(1, 2)).abs()
    actual = torch.fft.fftn(recovered, dim=(1, 2)).abs()
    torch.testing.assert_close(actual / scale, expected / scale, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize('scale', [1.0, 1e-4, 1e4])
def test_octahedral_inverse_recovers_orbit_independent_of_scale(scale: float):
    module = OctaonOcta()
    generator = torch.Generator().manual_seed(1283)
    original = torch.randn(2, 24, dtype=torch.float64, generator=generator)
    # Preserve the surrounding test suite's RNG while fixing the solver restarts.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4096)
        recovered = module.invert(module(scale * original)) / scale
    errors = torch.stack(
        [(recovered - original[:, _CAYLEY[_INVERSE[g]]]).norm(dim=1) for g in range(24)], dim=1
    ).amin(dim=1)
    assert torch.all(errors / original.norm(dim=1) < 1e-9)


def _group_sum_beta(f: torch.Tensor, *, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Evaluate the defining Fourier tensor product directly, without any CG coefficients."""
    left_hat = torch.einsum('bg,gij->bij', f, left)
    right_hat = torch.einsum('bg,gij->bij', f, right)
    tensor_rep = torch.stack([torch.kron(a, b) for a, b in zip(left, right, strict=True)])
    product_hat = torch.einsum('bg,gij->bij', f, tensor_rep)
    hats_kron = torch.stack([torch.kron(a, b) for a, b in zip(left_hat, right_hat, strict=True)])
    return product_hat.transpose(-1, -2) @ hats_kron


@pytest.mark.parametrize('n', [3, 4, 5, 6, 7, 8])
def test_dihedral_matches_cg_free_group_sum(n: int):
    generator = torch.Generator().manual_seed(194)
    f = torch.randn(2, 2 * n, dtype=torch.float64, generator=generator)
    theta = 2 * torch.pi * torch.arange(n, dtype=torch.float64) / n
    reflection = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64))
    irreps = []
    for k in range(1, (n - 1) // 2 + 1):
        c, s = torch.cos(k * theta), torch.sin(k * theta)
        rotations = torch.stack([c, -s, s, c], dim=-1).reshape(n, 2, 2)
        irreps.append(torch.cat([rotations, rotations @ reflection]))
    trivial = torch.ones(2 * n, 1, 1, dtype=torch.float64)
    expected = torch.cat(
        [
            _group_sum_beta(f, left=trivial, right=trivial).flatten(1),
            _group_sum_beta(f, left=trivial, right=irreps[0]).flatten(1),
        ]
        + [_group_sum_beta(f, left=irreps[0], right=rho).flatten(1) for rho in irreps],
        dim=1,
    )
    torch.testing.assert_close(DnonDn(n)(f), expected, atol=1e-11, rtol=1e-11)


def test_high_degree_wigner3j_independent_reference():
    # SymPy 1.14 exact wigner_3j(96,96,96,0,0,0), evaluated at 80 digits.
    expected = 0.006282433505226595
    assert wigner3j(96, 96, 96, 0, 0, 0) == pytest.approx(expected, rel=1e-12)


def test_torus_continuous_translation_is_not_the_supported_discrete_action():
    """Embed the three-sample circle counterexample in a two-dimensional torus."""
    theta = 2 * torch.pi * torch.arange(3, dtype=torch.float64) / 3
    alpha = torch.tensor([0.0, 0.37, 2 * torch.pi / 3], dtype=torch.float64)
    first_axis = 1 + torch.cos(theta[None] - alpha[:, None])
    f = first_axis[:, :, None].expand(-1, -1, 2)
    module = TorusOnTorus((3, 2))
    entry = module.index_map.index(((1, 0), (1, 0)))
    beta = module(f)
    expected = 3**3 * torch.exp(-3j * alpha)
    torch.testing.assert_close(beta[:, entry], expected, atol=1e-12, rtol=1e-13)
    assert not torch.allclose(beta[0], beta[1])
    torch.testing.assert_close(beta[0], beta[2], atol=1e-12, rtol=1e-13)
