# bispectrum

An open-source, fully unit-tested PyTorch library that implements *selective* G-bispectra for seven group actions as differentiable `torch.nn.Module`s, ready to plug into ML pipelines and deep learning architectures.

The G-bispectrum is a principled *complete* invariant of a signal — it retains all information up to the group action. Selectivity reduces cost from O(|G|²) to O(|G|) for finite groups, and from O(L³) to Θ(L²) coefficients for SO(3) on S².

## Supported Groups

| Module | Group / Domain | Output mode | Complexity (selective) |
|--------|---------------|-------------|----------------------|
| `CnonCn` | C_n on C_n | selective + full | O(n) |
| `SO2onS1` | SO(2) on S¹ | selective + full | O(n) |
| `TorusOnTorus` | T^d | selective + full | O(n) |
| `DnonDn` | D_n on D_n | selective | O(n) |
| `SO2onDisk` | SO(2) on disk | selective | O(L) |
| `SO3onS2` | SO(3) on S² | selective + full | Θ(L²) |
| `OctaonOcta` | chiral octahedral O | selective | O(|G|) |

`SO2onS1` is the continuous-n limit of `CnonCn` and shares its implementation.

## API

Every module exposes a uniform interface:

- **`forward(f)`** — selective (default) or full bispectral invariants
- **`fourier(f)`** — group Fourier coefficients
- **`invert(beta)`** — signal reconstruction up to group-action indeterminacy (where available)

Modules default to O(|G|) selective coefficients; pass `selective=False` for the full O(|G|²) set. CG matrices, DFT kernels, and Bessel roots are precomputed as non-learnable buffers. Dependencies: PyTorch, NumPy, and `torch_harmonics` (for `SO3onS2`).

## Benchmarks

Median wall-clock on a single NVIDIA H100 80 GB GPU (batch=16, `torch.utils.benchmark`):

| Module | Group | \|G\| / L_max | Coefs (sel.) | Coefs (full) | Fwd sel. (ms) | Fwd full (ms) |
|--------|-------|-----------|------------|-------------|--------------|----------------|
| `CnonCn` | C_128 | 128 | 128 | 8,256 | 0.14 | 8.53 |
| `TorusOnTorus` | C_32 × C_32 | 1,024 | 1,024 | 524,800 | 0.07 | 0.31 |
| `DnonDn` | D_32 | 64 | 245 | — | 0.57 | — |
| `SO2onDisk` | SO(2) | L=16 | 105 | — | 0.22 | — |
| `SO3onS2` | SO(3) | L=16 | 430 | — | 0.48 | — |
| `OctaonOcta` | O | 24 | 172 | — | 0.68 | — |

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
