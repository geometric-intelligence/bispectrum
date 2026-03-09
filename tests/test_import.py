"""Test that bispectrum package can be imported."""

import bispectrum


def test_import() -> None:
    """Test that bispectrum can be imported."""
    assert bispectrum is not None


def test_version() -> None:
    assert bispectrum.__version__ == '0.2.0'


def test_public_api() -> None:
    assert hasattr(bispectrum, 'CnonCn')
    assert hasattr(bispectrum, 'DnonDn')
    assert hasattr(bispectrum, 'SO2onS1')
    assert hasattr(bispectrum, 'SO3onS2')
    assert hasattr(bispectrum, 'random_rotation_matrix')
    assert hasattr(bispectrum, 'rotate_spherical_function')
