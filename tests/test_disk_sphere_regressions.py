"""SO2onDisk dtype/inverse-gradient and SO3onS2 checkpoint regressions."""

import pytest
import torch

from bispectrum import SO2onDisk, SO3onS2


def test_disk_float_module_forward_and_inverse():
    module = SO2onDisk(8).float()
    generator = torch.Generator().manual_seed(456)
    f = torch.randn(2, 8, 8, generator=generator)
    beta = module(f)
    assert beta.dtype == torch.complex64
    reconstructed = module.invert(beta)
    assert reconstructed.dtype == torch.float32
    torch.testing.assert_close(module(reconstructed), beta, atol=1e-5, rtol=1e-4)


def test_disk_inverse_composition_gradient():
    module = SO2onDisk(8)
    generator = torch.Generator().manual_seed(456)
    # Work in coefficient space, where rotation preserves harmonic coefficient norm.
    x = torch.randn(
        1, module._phi.shape[1], dtype=torch.float64, generator=generator, requires_grad=True
    )
    f = torch.zeros(1, 8, 8, dtype=torch.float64)
    f[:, module._disk_mask] = x @ module._phi.T
    beta = module(f)
    reconstructed = module.invert(beta)
    recovered_coefficients = module._dht(reconstructed)
    expected_coefficients = module._dht(f)
    actual = torch.autograd.grad(
        recovered_coefficients.abs().square().sum(), x, retain_graph=True
    )[0]
    expected = torch.autograd.grad(expected_coefficients.abs().square().sum(), x)[0]
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize('selective', [True, False])
def test_sphere_state_dict_loads_after_first_forward(selective: bool):
    module = SO3onS2(lmax=3, nlat=8, nlon=12, selective=selective)
    f = torch.randn(1, 8, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(811))
    expected = module(f)
    assert set(module.state_dict()) == set(
        SO3onS2(lmax=3, nlat=8, nlon=12, selective=selective).state_dict()
    )
    restored = SO3onS2(lmax=3, nlat=8, nlon=12, selective=selective)
    restored.load_state_dict(module.state_dict())
    torch.testing.assert_close(restored(f), expected, atol=1e-12, rtol=1e-12)


def test_sphere_strict_load_of_checkpoint_with_derived_index_buffers():
    """0.4.1 checkpoints saved after a forward pass also carry the derived index buffers."""
    module = SO3onS2(lmax=3, nlat=8, nlon=12, selective=True)
    f = torch.randn(1, 8, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(811))
    expected = module(f)
    legacy = {f'bsp.{k}': v for k, v in module.state_dict().items()}
    legacy |= {f'bsp.{name}': getattr(module, name) for name in SO3onS2._DERIVED_INDEX_BUFFERS}
    restored = torch.nn.ModuleDict({'bsp': SO3onS2(lmax=3, nlat=8, nlon=12, selective=True)})
    restored.load_state_dict(legacy, strict=True)
    torch.testing.assert_close(restored['bsp'](f), expected, atol=1e-12, rtol=1e-12)
