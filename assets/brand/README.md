# Brand assets

The `bispectrum` lettermark is a lowercase b whose round counter holds a small equilateral triangle pointing right: the closed triple of the bispectrum inside the circle of a group orbit.

| File                                                              | Use                                                                 |
| ----------------------------------------------------------------- | ------------------------------------------------------------------- |
| `bispectrum-wordmark.svg`                                         | Primary logo on light grounds                                       |
| `bispectrum-wordmark-reversed.svg`                                | Primary logo on dark grounds                                        |
| `bispectrum-mark.svg`, `bispectrum-mark-reversed.svg`             | Lettermark alone, 24 px and up                                      |
| `bispectrum-mark-small.svg`, `bispectrum-mark-small-reversed.svg` | Lettermark at 16 to 24 px, with heavier strokes                     |
| `bispectrum-favicon.svg`                                          | Browser tabs and app icons                                          |
| `glyphs/`                                                         | One glyph per module, in ink and reversed versions, for 24 to 48 px |
| `fonts/`                                                          | IBM Plex Sans and Mono (OFL) for figures                            |
| `__init__.py`                                                     | Colour tokens and matplotlib style: `from assets import brand`      |

## Colours

| Name            | Hex       | Use                                                              |
| --------------- | --------- | ---------------------------------------------------------------- |
| Ink             | `#1C1C1A` | The mark and text on light grounds; the dark ground              |
| Beige           | `#F7F3EB` | The light ground; the mark on dark grounds                       |
| Orange          | `#F57050` | The triangle and the dot on the i, and nothing else in the logo  |
| Orange for text | `#B53E20` | Links, accented text and highlighted plot lines on light grounds |

## Please don't

- Turn the triangle, enlarge it, or recolour it.
- Retype the wordmark or set it in capitals. The SVGs are outlined and need no font.
- Add outlines, shadows or gradients, or place the mark on an orange ground.
