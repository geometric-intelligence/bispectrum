# bispectrum

Bispectrum analysis for machine learning.

## Installation

```bash
pip install bispectrum
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install bispectrum
```

### Development

```bash
uv pip install -e ".[dev]"
pre-commit install
```

### Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) for code quality checks. After installing dev dependencies, the hooks run automatically on each commit.

```bash
# Run hooks on all files (useful for first-time setup or CI)
pre-commit run --all-files

# Run a specific hook
pre-commit run ruff --all-files

# Update hooks to latest versions
pre-commit autoupdate
```

## Usage

### Cyclic group on itself

```python
from bispectrum import CnonCn

bsp = CnonCn(n=8)
f = torch.randn(4, 8)       # signal on Z/8Z
beta = bsp(f)                # shape (4, 8), complex64
f_rec = bsp.invert(beta)     # reconstructed up to cyclic shift
```

### SO(3) on the 2-sphere

```python
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
from bispectrum import OctaonOcta

bsp = OctaonOcta()
f = torch.randn(4, 24)       # signal on O (|O| = 24)
beta = bsp(f)                 # shape (4, 172), complex64
f_rec = bsp.invert(beta)      # reconstructed up to group action
```

## License

MIT
