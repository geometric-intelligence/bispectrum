"""Test that bispectrum package can be imported."""

import bispectrum
from bispectrum import (
    CnonCn,
    DnonDn,
    OctaonOcta,
    SO2onDisk,
    SO2onS1,
    SO3onS2,
    TorusOnTorus,
)


def test_import() -> None:
    """Test that bispectrum can be imported."""
    assert bispectrum is not None


def test_public_api() -> None:
    for name in (
        'CnonCn',
        'DnonDn',
        'OctaonOcta',
        'SO2onDisk',
        'SO2onS1',
        'SO3onS2',
        'TorusOnTorus',
        'random_rotation_matrix',
        'rotate_spherical_function',
    ):
        assert hasattr(bispectrum, name), f'missing public API: {name}'


def test_supports_inversion_attribute() -> None:
    """Each module exposes a class-level ``supports_inversion`` flag."""
    expected: dict[type, bool] = {
        CnonCn: True,
        DnonDn: True,
        OctaonOcta: True,
        SO2onDisk: True,
        SO2onS1: True,
        SO3onS2: False,
        TorusOnTorus: True,
    }
    for cls, expected_flag in expected.items():
        assert cls.supports_inversion is expected_flag, (
            f'{cls.__name__}.supports_inversion = {cls.supports_inversion}, expected {expected_flag}'
        )
