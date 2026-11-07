"""Spherical MNIST capacity table: grouping, statistics and formatting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip('matplotlib')
pytest.importorskip('scipy')

SMNIST_DIR = Path(__file__).resolve().parents[1] / 'experiments' / 'spherical_mnist'


def _analyze_capacity() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        '_exp_analyze_capacity', SMNIST_DIR / 'analyze_capacity.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cap = _analyze_capacity()


def _run(model: str, n_params: int, accuracy: float, train_mode: str = 'C') -> dict:
    return {
        'base_model': model,
        'n_params': n_params,
        'train_mode': train_mode,
        'test_r': {'accuracy': accuracy},
    }


@pytest.mark.parametrize(
    ('n_params', 'label'), [(3690, '3.7K'), (11466, '11K'), (106698, '107K'), (550410, '550K')]
)
def test_format_params(n_params: int, label: str) -> None:
    assert cap.format_params(n_params) == label


def test_capacity_rows_groups_by_size_with_sample_std() -> None:
    runs = [
        _run('bispectrum', 51306, 0.95),
        _run('bispectrum', 25146, 0.94),
        _run('bispectrum', 25146, 0.96),
        _run('power_spectrum', 3690, 0.76),
        _run('power_spectrum', 3690, 0.50, train_mode='R'),
        _run('standard', 3690, 0.10),
    ]
    rows = cap.capacity_rows(runs)
    assert [r[0] for r in rows['bispectrum']] == [25146, 51306]
    n_params, mean, std, n_seeds = rows['bispectrum'][0]
    assert (n_params, n_seeds) == (25146, 2)
    assert mean == pytest.approx(95.0)
    assert std == pytest.approx(2**0.5)
    assert rows['power_spectrum'] == [(3690, pytest.approx(76.0), 0.0, 1)]


def test_latex_rows_pad_the_shorter_column() -> None:
    rows = {
        'power_spectrum': [(3690, 75.79, 0.19, 5)],
        'bispectrum': [(25146, 94.05, 0.23, 5), (51306, 94.9, 0.29, 5)],
    }
    assert cap.latex_rows(rows).splitlines() == [
        r'    3.7K & $75.79 \pm 0.19$ & 25K & $94.05 \pm 0.23$ \\',
        r'     &  & 51K & $94.90 \pm 0.29$ \\',
    ]
