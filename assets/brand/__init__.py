"""Design tokens and matplotlib style for every figure, slide, and web asset in this repo.

Scripts outside this directory put the repo root on ``sys.path`` and call ``use_style``::

    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT))
    from assets import brand  # noqa: E402

    brand.use_style()  # white "paper" surface; BISPECTRUM_SURFACE=screen for beige
    ...
    brand.finish(fig)  # tick labels in Plex Mono
    fig.savefig(...)

Colours follow ``README.md`` in this directory: ink, beige, and one orange accent. The orange is
reserved for the bispectrum ("ours"); baselines get warm grays from ``RAMP``. The logo SVGs
next to this file are the only logo; do not regenerate or redraw them.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Final, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure

type Surface = Literal['paper', 'screen']
type GridAxis = Literal['x', 'y', 'both']

BRAND_DIR: Final = Path(__file__).resolve().parent
FONT_DIR: Final = BRAND_DIR / 'fonts'

# --------------------------------------------------------------------------
# Core tokens (README.md)
# --------------------------------------------------------------------------

INK: Final = '#1C1C1A'
BEIGE: Final = '#F7F3EB'
SAND: Final = '#EAE6DB'
MUTED: Final = '#5D5852'
ORANGE: Final = '#F57050'
WHITE: Final = '#FFFFFF'

# Orange is 2.9:1 on white and 2.6:1 on beige: area fills only. Lines, markers and text that
# carry the accent use ORANGE_TEXT, as assets/brand prescribes for highlighted plot lines.
ORANGE_TEXT: Final = '#B53E20'  # 5.7:1 on white, 5.2:1 on beige
RULE: Final = '#DDD6CA'  # gridlines, legend frames, separators
NEUTRAL_LIGHT: Final = '#D6CFC3'

# --------------------------------------------------------------------------
# Categorical ramp: the sheet has one hue, so everything that is not ours is a warm gray on
# its ink -> muted -> sand line (CIELAB), 14-18 L* apart. Ink vs. orange is the logo's
# own pairing. Grays survive any CVD and grayscale print, but adjacent steps blur on thin
# lines: always pair with distinct markers, and avoid alpha (it compresses the L* range).
# --------------------------------------------------------------------------

GRAPHITE: Final = '#3D3834'  # L* 24
ASH: Final = '#807B74'  # L* 52
STONE: Final = '#A8A298'  # L* 67; 2.3:1 on beige, marks only

RAMP: Final = (INK, GRAPHITE, MUTED, ASH, STONE)
CYCLE: Final = RAMP

METHOD_COLORS: Final[dict[str, str]] = {
    # Complete invariant (ours).
    'bispectrum': ORANGE_TEXT,
    # Incomplete second-order invariants: the direct foil, so the strongest mark.
    'norm_pool': INK,
    'power_spectrum': INK,
    # Non-equivariant CNN baselines: weakest prior, lightest mark.
    'standard': STONE,
    'standard_3d': STONE,
    'standard_s2': STONE,
    # Equivariant pooling baselines.
    'norm': GRAPHITE,
    'gate': MUTED,
    'max_pool': MUTED,
    'fourier_elu': ASH,
    # Spherical baselines.
    's2cnn_bn': GRAPHITE,
    's2cnn': MUTED,
    'tensor_product': ASH,
}

# Library modules; SO3onS2 carries the accent as the paper's new construction.
GROUP_COLORS: Final[dict[str, str]] = {
    'CnonCn': INK,
    'TorusOnTorus': MUTED,
    'DnonDn': ASH,
    'SO2onDisk': STONE,
    'SO3onS2': ORANGE_TEXT,
    'OctaonOcta': GRAPHITE,
}

SURFACES: Final[dict[Surface, str]] = {'paper': WHITE, 'screen': BEIGE}

SANS: Final = 'IBM Plex Sans'
MONO: Final = 'IBM Plex Mono'

# Orange alone reaches only L*=63 against ink's 10; deepening the positive end keeps both
# halves legible for mostly-positive fields.
DIVERGING: Final = LinearSegmentedColormap.from_list(
    'bispectrum_diverging',
    [(0.0, INK), (0.5, BEIGE), (0.8, ORANGE), (1.0, ORANGE_TEXT)],
)

_MATHTEXT_NOISE: Final = re.compile(r'\\mathdefault|\\times|\\cdot|[${}\s]')
_NUMERIC_LABEL: Final = re.compile(
    r'[−\-+]?(\d[\d.,]*(e[−\-+]?\d+)?)?(\d+\^[−\-+]?\d+)?%?[kKM]?|full|all'
)

_FONTS_REGISTERED = False


def register_fonts() -> None:
    """Make the vendored IBM Plex TTFs visible to matplotlib (idempotent)."""
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    for path in sorted(FONT_DIR.glob('*.ttf')):
        font_manager.fontManager.addfont(str(path))
    if DIVERGING.name not in mpl.colormaps:
        mpl.colormaps.register(DIVERGING)
    _FONTS_REGISTERED = True


def resolve_surface(surface: Surface | None = None) -> Surface:
    resolved = surface or os.environ.get('BISPECTRUM_SURFACE', 'paper')
    match resolved:
        case 'paper' | 'screen':
            return resolved
    raise ValueError(f'unknown surface {resolved!r}; expected one of {sorted(SURFACES)}')


def rc_params(surface: Surface | None = None) -> dict[str, object]:
    """Fonts, colors, and export settings; sizes are left to each figure."""
    background = SURFACES[resolve_surface(surface)]
    return {
        'font.family': 'sans-serif',
        'font.sans-serif': [SANS, 'DejaVu Sans'],
        'font.monospace': [MONO, 'DejaVu Sans Mono'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': SANS,
        'mathtext.sf': SANS,
        'mathtext.it': f'{SANS}:italic',
        'mathtext.bf': f'{SANS}:bold',
        'mathtext.tt': MONO,
        'mathtext.cal': f'{SANS}:italic',
        'mathtext.fallback': 'stixsans',
        'text.usetex': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'text.color': INK,
        'axes.labelcolor': INK,
        'axes.titlecolor': INK,
        'axes.edgecolor': MUTED,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelcolor': INK,
        'ytick.labelcolor': INK,
        'axes.prop_cycle': cycler(color=list(CYCLE)),
        'grid.color': RULE,
        'grid.alpha': 1.0,
        'legend.edgecolor': RULE,
        'legend.facecolor': background,
        'figure.facecolor': background,
        'axes.facecolor': background,
        'savefig.facecolor': background,
        'savefig.edgecolor': background,
    }


def use_style(surface: Surface | None = None) -> None:
    register_fonts()
    plt.rcParams.update(rc_params(surface))


def surface_color() -> str:
    """Background of the active style; use for marker/bar edges instead of 'white'."""
    return str(plt.rcParams['axes.facecolor'])


def tint(color: str, strength: float) -> tuple[float, float, float]:
    """Opaque mix of *color* into the active surface (1.0 = color, 0.0 = surface).

    Use instead of alpha for lighter variants: PDF viewers draw hatches and overlaps of
    translucent patches inconsistently, and legend swatches then stop matching the plot.
    """
    fg = np.asarray(to_rgb(color))
    bg = np.asarray(to_rgb(surface_color()))
    r, g, b = strength * fg + (1.0 - strength) * bg
    return float(r), float(g), float(b)


def style_axes(ax: Axes, *, grid: GridAxis | None = 'y') -> None:
    """Open frame (no top/right spines) with light rules behind the data."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    if grid is not None:
        ax.grid(True, axis=grid, which='major', color=RULE, linewidth=0.4)
        ax.grid(False, which='minor')


def _is_numeric_label(label: str) -> bool:
    return _NUMERIC_LABEL.fullmatch(_MATHTEXT_NOISE.sub('', label).strip()) is not None


def _is_categorical(axis: Axis) -> bool:
    labels = [text.get_text() for text in axis.get_ticklabels() if text.get_text()]
    return sum(not _is_numeric_label(label) for label in labels) > len(labels) / 2


def finish(fig: Figure) -> None:
    """Set numeric tick labels in Plex Mono (category names stay in Plex Sans).

    Call after ticks and tick labels are final, before saving.
    """
    register_fonts()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if not _is_categorical(axis):
                axis.set_tick_params(which='both', labelfontfamily=MONO)
