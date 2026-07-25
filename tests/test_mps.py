"""Apple MPS backend parity tests."""

from __future__ import annotations

import pytest
import torch

from bispectrum import DnonDn, OctaonOcta, SO2onDisk, SO3onS2

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason='MPS not available',
)


def _assert_mps_close(
    cpu_module: torch.nn.Module,
    mps_module: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    with torch.inference_mode():
        expected = cpu_module(inputs)
        actual = mps_module(inputs.to('mps')).cpu()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@requires_mps
def test_dn_on_dn_float32_forward() -> None:
    torch.manual_seed(101)
    inputs = torch.randn(2, 16, dtype=torch.float32)
    _assert_mps_close(
        DnonDn(n=8).to(dtype=torch.float32),
        DnonDn(n=8).to(device='mps', dtype=torch.float32),
        inputs,
    )


@requires_mps
def test_so2_on_disk_float32_forward() -> None:
    torch.manual_seed(102)
    inputs = torch.randn(2, 8, 8, dtype=torch.float32)
    _assert_mps_close(
        SO2onDisk(L=8).to(dtype=torch.float32),
        SO2onDisk(L=8).to(device='mps', dtype=torch.float32),
        inputs,
        atol=2e-4,
        rtol=2e-4,
    )


@requires_mps
def test_so3_on_s2_sparse_float32_forward() -> None:
    torch.manual_seed(103)
    inputs = torch.randn(2, 32, 64, dtype=torch.float32)
    _assert_mps_close(
        SO3onS2(lmax=3, nlat=32, nlon=64).to(dtype=torch.float32),
        SO3onS2(lmax=3, nlat=32, nlon=64).to(device='mps', dtype=torch.float32),
        inputs,
        atol=2e-4,
        rtol=2e-4,
    )


@requires_mps
def test_octa_on_octa_float32_forward() -> None:
    torch.manual_seed(104)
    inputs = torch.randn(2, 24, dtype=torch.float32)
    _assert_mps_close(
        OctaonOcta().to(dtype=torch.float32),
        OctaonOcta().to(device='mps', dtype=torch.float32),
        inputs,
    )
