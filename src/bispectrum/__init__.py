"""Bispectrum analysis for machine learning."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version('bispectrum')
except PackageNotFoundError:
    __version__ = '0.0.0+unknown'

from bispectrum.cn_on_cn import CnonCn
from bispectrum.dn_on_dn import DnonDn
from bispectrum.octa_on_octa import OctaonOcta
from bispectrum.rotation import random_rotation_matrix, rotate_spherical_function
from bispectrum.so2_on_disk import SO2onDisk
from bispectrum.so2_on_s1 import SO2onS1
from bispectrum.so3_on_s2 import SO3onS2
from bispectrum.torus_on_torus import TorusOnTorus

__all__ = [
    'CnonCn',
    'DnonDn',
    'OctaonOcta',
    'SO2onDisk',
    'SO2onS1',
    'SO3onS2',
    'TorusOnTorus',
    'random_rotation_matrix',
    'rotate_spherical_function',
]
