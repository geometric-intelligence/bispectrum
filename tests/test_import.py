"""Test that bispectrum package can be imported."""

import bispectrum


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
