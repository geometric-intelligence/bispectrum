<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/bispectrum-wordmark-reversed.svg">
    <img alt="bispectrum" src="assets/brand/bispectrum-wordmark.svg" width="360">
  </picture>
</p>

<p align="center">Selective G-bispectra for PyTorch.</p>

<p align="center">
  <a href="https://github.com/geometric-intelligence/bispectrum/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/geometric-intelligence/bispectrum/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://github.com/geometric-intelligence/bispectrum/actions/workflows/pre-commit.yml"><img alt="Pre-commit" src="https://github.com/geometric-intelligence/bispectrum/actions/workflows/pre-commit.yml/badge.svg"></a>
  <a href="https://codecov.io/github/geometric-intelligence/bispectrum"><img alt="codecov" src="https://codecov.io/github/geometric-intelligence/bispectrum/graph/badge.svg?token=J6GGY4VK1E"></a>
  <a href="https://arxiv.org/abs/2605.07270"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2605.07270-1C1C1A"></a>
</p>

Many learning problems come with a group of transformations the answer should not depend on: shifts of a signal, rotations of an image, rotations of the sphere. The G-bispectrum is an invariant to such a group that is also *complete*. It discards the transformation and nothing else, so a signal can be recovered from it up to the group action.

`bispectrum` is an open-source, fully unit-tested PyTorch library that implements *selective* G-bispectra for seven group actions as differentiable `torch.nn.Module`s, ready to plug into ML pipelines and to sit in deep networks as invariant pooling layers. Selectivity reduces the cost from O(|G|²) to O(|G|) coefficients for finite groups, and from O(L³) to Θ(L²) for SO(3) on S².

## Installation

```bash
pip install bispectrum
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install bispectrum
```

### Apple silicon (MPS)

`torch-harmonics>=0.9` publishes Linux x86_64 wheels only, so on macOS build it from source before installing `bispectrum`:

```bash
pip install "torch>=2.10,<2.12" numpy setuptools setuptools-scm
SETUPTOOLS_SCM_PRETEND_VERSION=0.9.2 pip install --no-build-isolation \
  "torch-harmonics @ https://github.com/NVIDIA/torch-harmonics/archive/refs/tags/v0.9.2.tar.gz"
pip install bispectrum
```

MPS has no float64, and the precomputed tables are float64, so move modules with `module.to(device='mps', dtype=torch.float32)`.

## Usage

### Cyclic group on itself

```python
import torch
from bispectrum import CnonCn

bsp = CnonCn(n=8)
f = torch.randn(4, 8)       # signal on Z/8Z
beta = bsp(f)                # shape (4, 8), complex64
f_rec = bsp.invert(beta)     # reconstructed up to cyclic shift
```

### SO(3) on the 2-sphere

```python
import torch
from bispectrum import SO3onS2

# Selective bispectrum: O(L²) entries with generic completeness
bsp = SO3onS2(lmax=5, nlat=64, nlon=128, selective=True)
f = torch.randn(1, 64, 128)  # signal on S²
beta = bsp(f)                 # shape (1, 35), complex64

# Full bispectrum: O(L³) entries
bsp_full = SO3onS2(lmax=5, nlat=64, nlon=128, selective=False)
beta_full = bsp_full(f)       # shape (1, 69), complex64
```

### Octahedral group

```python
import torch
from bispectrum import OctaonOcta

bsp = OctaonOcta()
f = torch.randn(4, 24)       # signal on O (|O| = 24)
beta = bsp(f)                 # shape (4, 172), complex64
f_rec = bsp.invert(beta)      # reconstructed up to group action
```

## Supported groups

|                                                                                                                                                                                           | Module         | Group / domain                    | Output mode      | Complexity (selective) |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | --------------------------------- | ---------------- | ---------------------- |
| <picture><source media="(prefers-color-scheme: dark)" srcset="assets/brand/glyphs/CnonCn-reversed.svg"><img alt="" src="assets/brand/glyphs/CnonCn.svg" width="28"></picture>             | `CnonCn`       | Cₙ on Cₙ                          | selective + full | O(n)                   |
| <picture><source media="(prefers-color-scheme: dark)" srcset="assets/brand/glyphs/SO2onS1-reversed.svg"><img alt="" src="assets/brand/glyphs/SO2onS1.svg" width="28"></picture>           | `SO2onS1`      | SO(2) on S¹                       | selective + full | O(n)                   |
| <picture><source media="(prefers-color-scheme: dark)" srcset="assets/brand/glyphs/TorusOnTorus-reversed.svg"><img alt="" src="assets/brand/glyphs/TorusOnTorus.svg" width="28"></picture> | `TorusOnTorus` | 𝕋ᵈ                                | selective + full | O(n)                   |
| <picture><source media="(prefers-color-scheme: dark)" srcset="assets/brand/glyphs/DnonDn-reversed.svg"><img alt="" src="assets/brand/glyphs/DnonDn.svg" width="28"></picture>             | `DnonDn`       | Dₙ on Dₙ                          | selective        | O(n)                   |
| <picture><source media="(prefers-color-scheme: dark)" srcset="assets/brand/glyphs/SO2onDisk-reversed.svg"><img alt="" src="assets/brand/glyphs/SO2onDisk.svg" width="28"></picture>       | `SO2onDisk`    | SO(2) on the disk                 | selective        | O(L)                   |
| <picture><source media="(prefers-color-scheme: dark)" srcset="assets/brand/glyphs/SO3onS2-reversed.svg"><img alt="" src="assets/brand/glyphs/SO3onS2.svg" width="28"></picture>           | `SO3onS2`      | SO(3) on S²                       | selective + full | Θ(L²)                  |
| <picture><source media="(prefers-color-scheme: dark)" srcset="assets/brand/glyphs/OctaonOcta-reversed.svg"><img alt="" src="assets/brand/glyphs/OctaonOcta.svg" width="28"></picture>     | `OctaonOcta`   | chiral octahedral O (24 elements) | selective        | 172 coefficients       |

`SO2onS1` is the continuous-n limit of `CnonCn` and shares its implementation.

## API

Every module exposes a uniform interface:

- **`forward(f)`** — selective (default) or full bispectral invariants
- **`fourier(f)`** — group Fourier coefficients
- **`invert(beta)`** — signal reconstruction up to group-action indeterminacy (where available)

Modules default to O(|G|) selective coefficients; pass `selective=False` for the full O(|G|²) set. CG matrices, DFT kernels, and Bessel roots are precomputed as non-learnable buffers. Dependencies: PyTorch, NumPy, and `torch_harmonics` (for `SO3onS2`).

## Benchmarks

Median wall-clock on a single NVIDIA H100 80 GB GPU (batch=16, `torch.utils.benchmark`):

| Module         | Group       | \|G\| / L_max | Coefs (sel.) | Coefs (full) | Fwd sel. (ms) | Fwd full (ms) |
| -------------- | ----------- | ------------- | ------------ | ------------ | ------------- | ------------- |
| `CnonCn`       | C_128       | 128           | 128          | 8,256        | 0.14          | 8.53          |
| `TorusOnTorus` | C_32 × C_32 | 1,024         | 1,024        | 524,800      | 0.07          | 0.31          |
| `DnonDn`       | D_32        | 64            | 245          | —            | 0.57          | —             |
| `SO2onDisk`    | SO(2)       | L=16          | 76           | —            | 0.22          | —             |
| `SO3onS2`      | SO(3)       | L=16          | 426          | —            | 0.48          | —             |
| `OctaonOcta`   | O           | 24            | 172          | —            | 0.68          | —             |

## Development

```bash
git clone https://github.com/geometric-intelligence/bispectrum.git
cd bispectrum
uv pip install -e ".[dev]"
pre-commit install
```

### Pre-commit hooks

This project uses [pre-commit](https://pre-commit.com/) for code quality checks. After installing dev dependencies, the hooks run automatically on each commit.

```bash
# Run hooks on all files (useful for first-time setup or CI)
pre-commit run --all-files

# Run a specific hook
pre-commit run ruff --all-files

# Update hooks to latest versions
pre-commit autoupdate
```

## Citation

If you use `bispectrum` in your research, please cite the paper:

```bibtex
@article{mathe2026bispectrum,
  title   = {bispectrum: Selective $G$-Bispectra Made Practical},
  author  = {Mathe, Johan and Myers, Adele and Mataigne, Simon and Miolane, Nina},
  journal = {arXiv preprint arXiv:2605.07270},
  year    = {2026}
}
```

## License

MIT
