"""Build the lettermark, wordmark, favicons, and social card into ``brand/logo/``.

Geometry is measured from the Lettermark sheet and expressed in lettermark heights. The wordmark is
the lettermark as its "b" followed by Plex Mono SemiBold "ıspectrum", with a terracotta dot over
the dotless i. Rebuild with::

    .venv/bin/python brand/make_logo.py
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final, Literal

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fontTools.pens.basePen import AbstractPen, BasePen
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch, Patch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import brand  # noqa: E402

type Role = Literal['fg', 'accent', 'bg']
type Point = tuple[float, float]
type Box = tuple[float, float, float, float]

MARK_WIDTH: Final = 0.806
STROKE: Final = 0.113
STROKE_SMALL: Final = 0.14
STROKE_TINY: Final = 0.167
RING_R: Final = 0.364
RING_CX: Final = 0.442
TRIANGLE: Final[tuple[Point, ...]] = ((0.1009, -0.0256), (-0.0808, -0.0647), (-0.0173, 0.0914))

WORDMARK_TEXT: Final = '\u0131spectrum'
WORDMARK_FONT: Final = brand.FONT_DIR / 'IBMPlexMono-SemiBold.ttf'
WORDMARK_MARK_X: Final = -648.0
WORDMARK_MARK_HEIGHT: Final = 765.0
WORDMARK_DOT: Final = (290.0, 750.0, 80.0)

LIGHT: Final[dict[Role, str]] = {'fg': brand.INK, 'accent': brand.TERRACOTTA, 'bg': brand.CREAM}
DARK: Final[dict[Role, str]] = {'fg': brand.CREAM, 'accent': brand.TERRACOTTA, 'bg': brand.INK}


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float
    radius: float = 0.0
    role: Role = 'fg'


@dataclass(frozen=True)
class Ring:
    cx: float
    cy: float
    r_out: float
    r_in: float
    role: Role = 'fg'


@dataclass(frozen=True)
class Poly:
    points: tuple[Point, ...]
    role: Role = 'accent'


@dataclass(frozen=True)
class Disc:
    cx: float
    cy: float
    r: float
    role: Role = 'accent'


@dataclass(frozen=True)
class Glyphs:
    text: str
    x: float
    y: float
    size: float
    role: Role = 'fg'


type Shape = Rect | Ring | Poly | Disc | Glyphs


def lettermark(
    x: float = 0.0, y: float = 0.0, height: float = 1000.0, stroke: float = STROKE
) -> list[Shape]:
    r_in = RING_R - stroke
    k = r_in / (RING_R - STROKE)
    cx, cy = x + RING_CX * height, y + RING_R * height
    triangle = tuple((cx + dx * k * height, cy + dy * k * height) for dx, dy in TRIANGLE)
    return [
        Rect(x, y, stroke * height, height),
        Ring(cx, cy, RING_R * height, r_in * height),
        Poly(triangle),
    ]


def wordmark() -> list[Shape]:
    """Font units (1000/em), baseline at y=0, origin at the dotless-i cell."""
    return [
        *lettermark(WORDMARK_MARK_X, 0.0, WORDMARK_MARK_HEIGHT),
        Glyphs(WORDMARK_TEXT, 0.0, 0.0, 1000.0),
        Disc(*WORDMARK_DOT),
    ]


@cache
def _font() -> TTFont:
    return TTFont(WORDMARK_FONT)


def _draw_glyphs(glyphs: Glyphs, pen: AbstractPen, *, flip_top: float | None = None) -> None:
    font = _font()
    glyph_set, cmap, hmtx = font.getGlyphSet(), font.getBestCmap(), font['hmtx']
    scale = glyphs.size / font['head'].unitsPerEm
    x = glyphs.x
    for char in glyphs.text:
        name = cmap[ord(char)]
        if flip_top is None:
            matrix = (scale, 0, 0, scale, x, glyphs.y)
        else:
            matrix = (scale, 0, 0, -scale, x, flip_top - glyphs.y)
        glyph_set[name].draw(TransformPen(pen, matrix))
        x += hmtx[name][0] * scale


def _shape_bounds(shape: Shape) -> Box:
    match shape:
        case Rect(x, y, w, h):
            return x, y, x + w, y + h
        case Ring(cx, cy, r, _) | Disc(cx, cy, r):
            return cx - r, cy - r, cx + r, cy + r
        case Poly(points):
            xs, ys = zip(*points, strict=True)
            return min(xs), min(ys), max(xs), max(ys)
        case Glyphs():
            pen = BoundsPen(_font().getGlyphSet())
            _draw_glyphs(shape, pen)
            assert pen.bounds is not None
            return pen.bounds


def bounds(shapes: Iterable[Shape], pad: float = 0.0) -> Box:
    boxes = [_shape_bounds(shape) for shape in shapes]
    return (
        min(b[0] for b in boxes) - pad,
        min(b[1] for b in boxes) - pad,
        max(b[2] for b in boxes) + pad,
        max(b[3] for b in boxes) + pad,
    )


def _num(value: float) -> str:
    text = f'{value:.2f}'.rstrip('0').rstrip('.')
    return '0' if text == '-0' else text


def to_svg(
    shapes: Sequence[Shape],
    box: Box,
    colors: dict[Role, str],
    *,
    title: str,
    style: str | None = None,
) -> str:
    x0, y0, x1, y1 = box

    def pt(x: float, y: float) -> str:
        return f'{_num(x - x0)},{_num(y1 - y)}'

    body: list[str] = []
    for shape in shapes:
        attrs = f'class="{shape.role}" fill="{colors[shape.role]}"'
        match shape:
            case Rect(x, y, w, h, radius):
                corner = f' rx="{_num(radius)}"' if radius else ''
                body.append(
                    f'<rect {attrs} x="{_num(x - x0)}" y="{_num(y1 - y - h)}" '
                    f'width="{_num(w)}" height="{_num(h)}"{corner}/>'
                )
            case Ring(cx, cy, r_out, r_in):
                d = (
                    f'M{pt(cx - r_out, cy)}A{_num(r_out)},{_num(r_out)} 0 1 0 {pt(cx + r_out, cy)}'
                    f'A{_num(r_out)},{_num(r_out)} 0 1 0 {pt(cx - r_out, cy)}Z'
                    f'M{pt(cx - r_in, cy)}A{_num(r_in)},{_num(r_in)} 0 1 1 {pt(cx + r_in, cy)}'
                    f'A{_num(r_in)},{_num(r_in)} 0 1 1 {pt(cx - r_in, cy)}Z'
                )
                body.append(f'<path {attrs} fill-rule="evenodd" d="{d}"/>')
            case Poly(points):
                body.append(f'<polygon {attrs} points="{" ".join(pt(x, y) for x, y in points)}"/>')
            case Disc(cx, cy, r):
                body.append(
                    f'<circle {attrs} cx="{_num(cx - x0)}" cy="{_num(y1 - cy)}" r="{_num(r)}"/>'
                )
            case Glyphs():
                pen = SVGPathPen(_font().getGlyphSet(), ntos=_num)
                _draw_glyphs(shape, TransformPen(pen, (1, 0, 0, 1, -x0, 0)), flip_top=y1)
                body.append(f'<path {attrs} d="{pen.getCommands()}"/>')
    width, height = x1 - x0, y1 - y0
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_num(width)} {_num(height)}" '
        f'role="img" aria-label="{title}">'
    )
    style_block = [f'<style>{style}</style>'] if style else []
    return '\n'.join([head, f'<title>{title}</title>', *style_block, *body, '</svg>']) + '\n'


class _MplPen(BasePen):
    def __init__(self) -> None:
        super().__init__(None)
        self.vertices: list[Point] = []
        self.codes: list[int] = []

    def _add(self, points: Iterable[Point], code: int) -> None:
        for point in points:
            self.vertices.append(point)
            self.codes.append(code)

    def _moveTo(self, pt: Point) -> None:
        self._add([pt], MplPath.MOVETO)

    def _lineTo(self, pt: Point) -> None:
        self._add([pt], MplPath.LINETO)

    def _curveToOne(self, pt1: Point, pt2: Point, pt3: Point) -> None:
        self._add([pt1, pt2, pt3], MplPath.CURVE4)

    def _qCurveToOne(self, pt1: Point, pt2: Point) -> None:
        self._add([pt1, pt2], MplPath.CURVE3)

    def _closePath(self) -> None:
        self._add([(0.0, 0.0)], MplPath.CLOSEPOLY)


def _to_patch(shape: Shape, color: str) -> Patch:
    kwargs = {'facecolor': color, 'edgecolor': 'none', 'linewidth': 0}
    match shape:
        case Rect(x, y, w, h, radius) if radius:
            return FancyBboxPatch(
                (x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}', **kwargs
            )
        case Rect(x, y, w, h):
            return Rectangle((x, y), w, h, **kwargs)
        case Ring(cx, cy, r_out, r_in):
            outer = MplPath.unit_circle().transformed(Affine2D().scale(r_out).translate(cx, cy))
            inner = MplPath.unit_circle().transformed(
                Affine2D().scale(-r_in, r_in).translate(cx, cy)
            )
            return PathPatch(MplPath.make_compound_path(outer, inner), **kwargs)
        case Poly(points):
            return Polygon(points, closed=True, **kwargs)
        case Disc(cx, cy, r):
            return Circle((cx, cy), r, **kwargs)
        case Glyphs():
            pen = _MplPen()
            _draw_glyphs(shape, pen)
            return PathPatch(MplPath(pen.vertices, pen.codes), **kwargs)


def draw(ax: Axes, shapes: Iterable[Shape], colors: dict[Role, str]) -> None:
    for shape in shapes:
        ax.add_patch(_to_patch(shape, colors[shape.role]))


def _canvas(box: Box, width_in: float, background: str | None) -> tuple[Figure, Axes]:
    x0, y0, x1, y1 = box
    fig = plt.figure(figsize=(width_in, width_in * (y1 - y0) / (x1 - x0)))
    fig.patch.set_facecolor(background or 'none')
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax


def render(
    shapes: Sequence[Shape],
    box: Box,
    colors: dict[Role, str],
    path: Path,
    *,
    width_px: int | None = None,
    width_in: float = 2.0,
    background: str | None = None,
) -> None:
    dpi = 100
    fig, ax = _canvas(box, width_px / dpi if width_px else width_in, background)
    draw(ax, shapes, colors)
    fig.savefig(path, dpi=dpi, transparent=background is None, metadata=_metadata(path))
    plt.close(fig)


def _metadata(path: Path) -> dict[str, str | None]:
    if path.suffix == '.pdf':
        return {'Creator': None, 'Producer': None, 'CreationDate': None}
    if path.suffix == '.svg':
        return {'Date': None}
    return {}


def app_icon(
    size: float = 1000.0, *, stroke: float = STROKE, scale: float = 0.62, radius: float = 0.22
) -> list[Shape]:
    """Lettermark centred on a cream (rounded) square."""
    height = scale * size
    x = (size - MARK_WIDTH * height) / 2
    y = (size - height) / 2
    return [
        Rect(0.0, 0.0, size, size, radius * size, role='bg'),
        *lettermark(x, y, height, stroke),
    ]


FAVICON_STYLE: Final = (
    f'.fg{{fill:{brand.INK}}}@media (prefers-color-scheme:dark){{.fg{{fill:{brand.CREAM}}}}}'
)


def write_svg(path: Path, svg: str) -> None:
    path.write_text(svg)
    print(f'wrote {path.relative_to(brand.BRAND_DIR.parent)}')


def build_marks(out: Path) -> None:
    mark = lettermark()
    mark_box = bounds(mark)
    word = wordmark()
    word_box = bounds(word)
    for suffix, colors in (('', LIGHT), ('-dark', DARK)):
        write_svg(
            out / f'lettermark{suffix}.svg', to_svg(mark, mark_box, colors, title='bispectrum')
        )
        write_svg(
            out / f'wordmark{suffix}.svg', to_svg(word, word_box, colors, title='bispectrum')
        )
        render(mark, mark_box, colors, out / f'lettermark{suffix}.pdf', width_in=1.0)
        render(word, word_box, colors, out / f'wordmark{suffix}.pdf', width_in=3.0)
        render(word, word_box, colors, out / f'wordmark{suffix}.png', width_px=1600)
        render(mark, mark_box, colors, out / f'lettermark{suffix}.png', width_px=512)


def build_favicons(out: Path) -> None:
    tiny = lettermark(stroke=STROKE_SMALL)
    write_svg(
        out / 'favicon.svg',
        to_svg(tiny, bounds(tiny, pad=60.0), LIGHT, title='bispectrum', style=FAVICON_STYLE),
    )
    icons: list[Path] = []
    for size, stroke in (
        (16, STROKE_TINY),
        (32, STROKE_SMALL),
        (48, STROKE_SMALL),
        (180, STROKE),
        (512, STROKE),
    ):
        rounded = 0.0 if size == 180 else 0.22
        icon = app_icon(stroke=stroke, scale=0.72 if size <= 48 else 0.62, radius=rounded)
        path = out / f'icon-{size}.png'
        render(icon, (0.0, 0.0, 1000.0, 1000.0), LIGHT, path, width_px=size)
        icons.append(path)
    (out / 'icon-180.png').rename(out / 'apple-touch-icon.png')
    sizes = [(16, 16), (32, 32), (48, 48)]
    Image.open(out / 'icon-48.png').save(
        out / 'favicon.ico',
        sizes=sizes,
        append_images=[Image.open(out / f'icon-{n}.png') for n in (16, 32)],
    )
    for n in (16, 32, 48):
        (out / f'icon-{n}.png').unlink()
    print(f'wrote favicons to {out.relative_to(brand.BRAND_DIR.parent)}')


def build_social_card(out: Path, *, surface: str = brand.CREAM) -> None:
    """1280x640 GitHub/OpenGraph card."""
    brand.register_fonts()
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(surface)
    word = wordmark()
    x0, y0, x1, y1 = bounds(word)
    width = 0.62
    height = width * (y1 - y0) / (x1 - x0) * 12.8 / 6.4
    ax = fig.add_axes(((1 - width) / 2, 0.50, width, height))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.axis('off')
    draw(ax, word, LIGHT)
    fig.text(
        0.5,
        0.40,
        'Selective G-bispectra as differentiable PyTorch modules',
        ha='center',
        va='center',
        fontsize=24,
        family=brand.SANS,
        color=brand.MUTED,
    )
    fig.text(
        0.5,
        0.24,
        'pip install bispectrum',
        ha='center',
        fontsize=18,
        family=brand.MONO,
        color=brand.INK,
    )
    fig.savefig(out / 'social-card.png', dpi=100, facecolor=surface)
    plt.close(fig)
    print(f'wrote {(out / "social-card.png").relative_to(brand.BRAND_DIR.parent)}')


def main() -> None:
    brand.LOGO_DIR.mkdir(exist_ok=True)
    build_marks(brand.LOGO_DIR)
    build_favicons(brand.LOGO_DIR)
    build_social_card(brand.LOGO_DIR)


if __name__ == '__main__':
    main()
