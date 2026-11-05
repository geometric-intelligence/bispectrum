"""Tests for the octahedral inversion benchmark's recovery metric."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

from bispectrum.octa_on_octa import _CAYLEY, _INVERSE, OctaonOcta

BENCH_PATH = Path(__file__).resolve().parents[1] / 'benchmarks' / 'bench_inversion.py'


@pytest.fixture(scope='module')
def bench_inversion() -> ModuleType:
    spec = importlib.util.spec_from_file_location('bench_inversion', BENCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _translate(f: torch.Tensor, g: int) -> torch.Tensor:
    """(T_g f)(h) = f(g^{-1} h)."""
    return f[:, _CAYLEY[_INVERSE[g]]]


class TestRecoveryErrorUpToGroupAction:
    def test_zero_on_every_group_translate(self, bench_inversion: ModuleType) -> None:
        torch.manual_seed(0)
        f = torch.randn(8, 24, dtype=torch.float64)
        for g in range(24):
            err = bench_inversion.recovery_error_up_to_group_action(f, _translate(f, g), _CAYLEY)
            assert err.max() < 1e-12, f'g={g}: {err.max():.2e}'

    def test_cyclic_shift_is_not_a_group_translate(self, bench_inversion: ModuleType) -> None:
        torch.manual_seed(0)
        f = torch.randn(8, 24, dtype=torch.float64)
        err = bench_inversion.recovery_error_up_to_group_action(
            f, torch.roll(f, 1, dims=-1), _CAYLEY
        )
        assert err.min() > 0.1

    def test_translates_share_the_bispectrum(self) -> None:
        torch.manual_seed(0)
        bsp = OctaonOcta().double()
        f = torch.randn(4, 24, dtype=torch.float64)
        beta = bsp(f)
        for g in range(24):
            torch.testing.assert_close(bsp(_translate(f, g)), beta, atol=1e-9, rtol=1e-9)

    def test_matches_module_cayley_buffer(self, bench_inversion: ModuleType) -> None:
        torch.manual_seed(0)
        f = torch.randn(4, 24)
        f_rec = _translate(f, 7)
        err = bench_inversion.recovery_error_up_to_group_action(f, f_rec, OctaonOcta()._cayley)
        assert err.max() < 1e-6
