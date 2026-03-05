# Bispectrum Module — API Design Document

**Status**: Accepted
**Author**: Johan Mathe
**Date**: 2026-02-23

______________________________________________________________________

## Overview

This document specifies the API design for the `bispectrum` Python module. The goal is to provide a clean, mathematically faithful, and composable interface for computing G-Bispectra across different groups and domains — while keeping the library as simple as possible, but not simpler.

______________________________________________________________________

## Core Design Principles

01. **Simple is better than clever.** If there are two ways to do something, pick the one that needs less explanation.

02. **`nn.Module` only.** No parallel functional APIs. One way to compute a bispectrum.

03. **Naming encodes the math.** Class names follow `{Group}on{Domain}` — a reader unfamiliar with the codebase can immediately identify the group and the domain it acts on.

04. **Raw signals in.** Modules take real-valued signals in the natural domain (spatial, discrete cycle, etc.) and handle the Fourier transform internally. This keeps the interface clean and the math self-contained. `SO3onS2` accepts a raw spatial signal `f` of shape `(batch, nlat, nlon)` and handles the SHT internally; `nlat` and `nlon` are constructor arguments so the `RealSHT` transform is pre-initialized at construction time for efficiency.

05. **Selective by default.** All modules default to `selective=True`, computing the minimal subset of bispectral coefficients sufficient for complete signal reconstruction. `selective=False` is available for debugging and comparison.

06. **Complex output.** `forward()` returns `torch.complex64`. Users who need real features can call `.abs()`, `.real`, or `torch.view_as_real()` — this is one line of code and avoids losing phase information.

07. **Inversion on-module.** Every module exposes `invert(beta, **kwargs)`. For modules where inversion is mathematically available, it returns the reconstructed signal (up to group-action indeterminacy). For modules where inversion is not yet available, it raises a precise `NotImplementedError` with guidance.

08. **float32 throughout.** For compatibility with GPU training pipelines.

09. **Minimal dependencies.** Don't add a dependency if standard PyTorch/numpy can do the job.

10. **Code is the math documentation.** Every module docstring references the exact paper theorem it implements. Every non-obvious operation cites an equation number.

______________________________________________________________________

## Naming Convention: `{Group}on{Domain}`

Groups act on domains. Both matter. The class name encodes both, in mathematical notation:

| Class     | Group                  | Domain          | Description                                    |
| --------- | ---------------------- | --------------- | ---------------------------------------------- |
| `CnonCn`  | $C_n$                  | $C_n$           | Cyclic group acting on itself (discrete cycle) |
| `DnonDn`  | $D_n$                  | $D_n$           | Dihedral group acting on itself                |
| `SO3onS2` | $\\mathrm{SO}(3)$      | $S^2$           | 3D rotations on the 2-sphere                   |
| `OonR3`   | $O$ (octahedral group) | $\\mathbb{R}^3$ | Octahedral symmetries on 3D space *(future)*   |

This convention is deliberately mathematical rather than verbal (`CyclicBispectrum`, etc.) because the mathematical name carries precise meaning and avoids ambiguity as the library grows.

______________________________________________________________________

## Module Interface Contract

Every bispectrum module is a `torch.nn.Module` satisfying this contract:

```python
class {Group}on{Domain}(nn.Module):

    def __init__(self, ...):
        """Initialize with group parameters.

        CG matrices and other precomputed quantities registered as buffers
        (not parameters — they are not learnable).
        """

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Compute the G-Bispectrum of signal f.

        Args:
            f: Real-valued signal on the domain. Shape: (batch, *domain_shape).

        Returns:
            Complex bispectrum tensor. Shape: (batch, output_size).
        """

    def invert(self, beta: torch.Tensor, **kwargs) -> torch.Tensor:
        """Attempt inversion from bispectrum coefficients.

        Default behavior can raise NotImplementedError for modules
        where inversion is not yet available.
        """

    @property
    def output_size(self) -> int:
        """Number of bispectral coefficients in the output."""

    @property
    def index_map(self) -> list[tuple]:
        """Maps flat output index → (irrep indices) tuple.
        Allows interpreting which (rho_1, rho_2, rho) each output corresponds to.
        """
```

**Key invariants:**

- No trainable parameters (`sum(p.numel() for p in bsp.parameters()) == 0`)
- Output is `torch.complex64` (complex float32)
- Module moves correctly with `.to(device)` and `.to(dtype)`
- Deterministic: same input always gives same output

______________________________________________________________________

## Implemented Modules

### `CnonCn(n: int)` — Cyclic group $C_n$ on $\\mathbb{Z}/n\\mathbb{Z}$

**Mathematical setting:**
Signal $f: \\mathbb{Z}/n\\mathbb{Z} \\to \\mathbb{R}$. Group $C_n$ acts by cyclic shift: $(T_g f)(x) = f(x - g \\bmod n)$.

**Bispectrum formula** (commutative case, [Kakarala 2009]):

$$\\beta(f)_{k_1, k_2} = \\hat{f}_{k_1} \\cdot \\hat{f}_{k_2} \\cdot \\hat{f}^\*_{k_1 + k_2 \\bmod n}$$

where:
$$ \\hat{f}_k = \\sum_{x=0}^{n-1} f(x), e^{-i \\frac{2\\pi k x}{n}} $$

is the DFT.

**Selective coefficients** (Algorithm 1, [Mataigne et al. 2024]):
Full bispectrum has $n^2$ coefficients. Selective version needs only $n$:

$${ \\beta\_{0,0},\\ \\beta\_{0,1},\\ \\beta\_{1,1},\\ \\beta\_{1,2},\\ \\ldots,\\ \\beta\_{1,n-2} }$$

These suffice for complete inversion (recovering $f$ up to cyclic shift).

**Usage:**

```python
bsp = CnonCn(n=8)
f = torch.randn(batch_size, 8)          # signal on Z/8Z, shape (batch, n)
output = bsp(f)                          # shape (batch, output_size), complex64

print(bsp.output_size)                   # n = 8 (selective) or n² (full)
print(bsp.index_map)                     # [(0,0), (0,1), (1,1), (1,2), ...]
```

**Constructor parameters:**

```python
CnonCn(
    n: int,                # Group order / signal length
    selective: bool = True # Use selective O(n) or full O(n²) bispectrum
)
```

______________________________________________________________________

### `DnonDn(n: int)` — Dihedral group $D_n$ on $D_n$

**Mathematical setting:**
Signal $f: G \\to \\mathbb{R}$ where $G = D_n = \\langle a, x \\mid a^n = x^2 = e,\\ xax = a^{-1} \\rangle$.
$D_n$ acts on $\\mathbb{R}^2$ via rotations ($a$: rotation by $2\\pi/n$) and reflections ($x$: flip).

**Irreducible representations:**

- 2D irreps $\\rho_k$, $k = 1, \\ldots, \\lfloor(n-1)/2\\rfloor$:

$$\\rho_k(a^l x^m) = \\begin{pmatrix} \\cos(\\tfrac{2\\pi lk}{n}) & -\\sin(\\tfrac{2\\pi lk}{n}) \\ \\sin(\\tfrac{2\\pi lk}{n}) & \\cos(\\tfrac{2\\pi lk}{n}) \\end{pmatrix} \\begin{pmatrix} 1 & 0 \\ 0 & -1 \\end{pmatrix}^m$$

- 1D irreps: $\\rho_0$ (trivial), $\\rho\_{01}$, and (for $n$ even) $\\rho\_{02}$, $\\rho\_{03}$

**Selective coefficients** (Algorithm 3, [Mataigne et al. 2024]):
Only $\\lfloor(n-1)/2\\rfloor + 2$ matrix-valued bispectral coefficients needed, corresponding to approximately $4\\lvert D_n \\rvert$ scalar values.

**Usage:**

```python
bsp = DnonDn(n=4)
f = torch.randn(batch_size, 8)          # signal on D_4 (|D_4| = 2n = 8), shape (batch, 2n)
output = bsp(f)                          # shape (batch, output_size), float32
```

Note: unlike `CnonCn`, `DnonDn.forward()` returns `float32` (not `complex64`) because all $D_n$ irreps are real-valued, so bispectral coefficients are real.

**Constructor parameters:**

```python
DnonDn(
    n: int,                # Polygon order (|D_n| = 2n)
    selective: bool = True
)
```

______________________________________________________________________

### `SO3onS2(lmax: int)` — $\\mathrm{SO}(3)$ on $S^2$

**Mathematical setting:**
Signal $f: S^2 \\to \\mathbb{R}$. Group $\\mathrm{SO}(3)$ acts by 3D rotation.

**Bispectrum formula** \[Kakarala 1992, Cohen et al.\]:

$$\\beta(f)_{l_1, l_2}^{(l)} = \\bigl(\\mathcal{F}_{l_1} \\otimes \\mathcal{F}_{l_2}\\bigr) \\cdot C_{l_1, l_2}^{(l)} \\cdot \\mathcal{F}\_l^\\dagger$$

where $\\mathcal{F}\_l$ are the degree-$l$ SH coefficient matrices and $C$ denotes the Clebsch-Gordan matrices for $\\mathrm{SO}(3)$.

**Usage:**

```python
bsp = SO3onS2(lmax=5)
f = torch.randn(batch_size, 64, 128)    # signal on S², shape (batch, nlat, nlon)
output = bsp(f)                          # shape (batch, output_size), complex64
```

**Constructor parameters:**

```python
SO3onS2(
    lmax: int = 5,          # Maximum spherical harmonic degree
    nlat: int = 64,         # Latitude grid points
    nlon: int = 128,        # Longitude grid points
    selective: bool = True  # Selective or full bispectrum
)
```

> **Breaking change from v0.1.0**: v0.1.0 accepted pre-computed SH coefficients. v0.2.0 accepts raw spatial signals and handles SHT internally.

______________________________________________________________________

## Selectivity Roadmap

The central value proposition of the library is the *selective* G-bispectrum: a minimal subset of bispectral pairs $(\\rho, \\sigma)$ that suffices for complete signal reconstruction, reducing coefficient count from $O(\\lvert G\\rvert^2)$ to $O(\\lvert G\\rvert)$.

This is proven for finite groups only. Here, `O` denotes the finite octahedral rotation group (not the full orthogonal group). The table below tracks the current state across all group/domain combinations of interest.

| Class     | Group                | Domain                             | Selective?                                      | Inversion?     | Status           |
| --------- | -------------------- | ---------------------------------- | ----------------------------------------------- | -------------- | ---------------- |
| `CnonCn`  | $C_n$                | $C_n$                              | ✅ $n$ coefficients                             | ✅ Algorithm 1 | ✅ Done          |
| `DnonDn`  | $D_n$                | $D_n$                              | ✅ $\\lfloor(n{-}1)/2\\rfloor{+}2$ matrix coefs | ✅ Algorithm 3 | ✅ Done          |
| `OonR3`   | $O$                  | $\\mathbb{R}^3$                    | ✅ 4 matrix coefs (paper App. B)                | ✅             | Planned          |
| —         | All commutative $G$  | $G$                                | ✅ $\\lvert G \\rvert$ coefs                    | ✅ Algorithm 2 | —                |
| `SO3onS2` | $\\mathrm{SO}(3)$    | $S^2$                              | ❌ Full only                                    | ❌             | **Open problem** |
| —         | $\\mathrm{SO}(2)$    | $S^1 \\times \\mathbb{R}^+$ (disk) | ❌ Full only                                    | ❌             | Open problem     |
| —         | $\\mathrm{SO}(3)$    | $S^2 \\times \\mathbb{R}^+$ (ball) | ❌ Full only                                    | ❌             | Open problem     |
| —         | Compact $G$          | $G$                                | ❌ Full only                                    | ❌             | Open problem     |
| —         | Homogeneous $(H, G)$ | $H = G/G_0$                        | ❌ Full only                                    | ❌             | Open problem     |

Sources: Mataigne et al. 2024 for all ✅ entries; "Bispectral Signatures of Data" (internal draft) for the full-bispectrum formulas of the remaining cases.

### Mathematical TODOs

The following are open mathematical problems whose solutions would directly extend the library:

**TODO-M1: Selective bispectrum for $\\mathrm{SO}(3)$ on $S^2$**

The full bispectrum at band-limit $L$ has $O(L^3)$ coefficients, governed by the Kronecker product rule for $\\mathrm{SO}(3)$:

$$\\rho\_{l_1} \\otimes \\rho\_{l_2} = \\bigoplus\_{l=|l_1-l_2|}^{l_1+l_2} \\rho_l$$

A selective version would identify the minimal set of index pairs needed for inversion. The challenge: $\\mathrm{SO}(3)$ has infinitely many irreps $\\rho_l$ (one per $l \\geq 0$), and the BFS on the Kronecker table (used for finite groups) does not terminate. A truncated selective version with guaranteed reconstruction error bounds would be a meaningful contribution.

**TODO-M2: Selective bispectrum for $\\mathrm{SO}(2)$ on the disk $S^1 \\times \\mathbb{R}^+$**

Fourier coefficients are indexed by $(m, n)$ (angular frequency $m$, radial zero $n$) via the Fourier-Bessel transform:

$$F(f)_{nm} = \\int f(r,\\theta), J_m(2\\pi l_{nm} r), e^{-im\\theta}, r, dr, d\\theta$$

The bispectrum is:

$$\\beta(f)_{(m_1, n_1),(m_2, n_2)} = F(f)_{m_1 n_1} \\cdot F(f)_{m_2 n_2} \\cdot F(f)^\*_{(m_1+m_2), n\_{12}}$$

($\\mathrm{SO}(2)$ is commutative, so no CG needed.) A selective version analogous to $C_n$ — exploiting the same cyclic structure in the angular index $m$ — likely exists.

**TODO-M3: Selective bispectrum for $\\mathrm{SO}(3)$ on the ball $S^2 \\times \\mathbb{R}^+$**

Same challenge as TODO-M1 plus the radial dimension. The bispectrum formula is identical to the $S^2$ case (same equivariance); selectivity is the open question.

**TODO-M4: Inversion algorithms for continuous groups**

For finite groups, inversion follows from recovering each $F(f)_\\rho$ via a bootstrap from the trivial representation. For compact/continuous groups (where irreps are infinite-dimensional in the $l \\to \\infty$ limit), the analogous reconstruction strategy needs to contend with truncation and stability. Conditioning of the bootstrap in the presence of $l_\\mathrm{max}$ truncation is the key question.

______________________________________________________________________

## Public API Surface

The top-level `bispectrum` namespace exposes only what a user needs:

```python
# Main modules
from bispectrum import CnonCn, DnonDn, SO3onS2

# Rotation utilities (useful for testing/data augmentation)
from bispectrum import random_rotation_matrix, rotate_spherical_function
```

**Explicitly not exported:**

- `clebsch_gordan` (internal implementation detail; current placeholder removed)
- `compute_padding_indices`, `pad_sh_coefficients`, `get_full_sh_coefficients` (SO(3) internals)
- Any `_private` helper functions

We do not depend on `escnn` or any external library for Clebsch-Gordan coefficients. Instead:

- **$C_n$**: No CG matrices needed. The bispectrum for commutative groups reduces to scalar products — no change-of-basis required.
- **$D_n$**: CG matrices are computed analytically from the explicit 2D irrep formulas (given in the paper). These are closed-form rotation matrices; no library is needed.
- **$\\mathrm{SO}(3)$**: CG matrices are pre-computed and stored as a bundled JSON file (`data/cg_lmax5.json`). A generation script is provided so users can extend to higher $l\_\\mathrm{max}$ if needed.

This keeps the dependency footprint minimal and makes the math transparent — CG coefficients are computed from first principles, not treated as a black box.

______________________________________________________________________

## What Goes in Each File

```
src/bispectrum/
├── __init__.py          # Public exports only
├── cn_on_cn.py          # CnonCn
├── dn_on_dn.py          # DnonDn
├── so3_on_s2.py         # SO3onS2 (refactored)
├── rotation.py          # random_rotation_matrix, rotate_spherical_function
└── _cg.py               # Internal CG utilities (not exported)
```

Files removed in v0.2.0:

- `clebsch_gordan.py` → merged into `_cg.py` (internal)
- `spherical.py` → folded into `so3_on_s2.py`
- `so3.py` → replaced by `so3_on_s2.py`

______________________________________________________________________

## Testing Philosophy

Every bispectrum module must have tests for:

1. **Output shape** — `(batch, output_size)` for various batch sizes
2. **G-invariance** — `bsp(T_g(f)) ≈ bsp(f)` for random $g$, random $f$
   This is the *most important test*. If this fails, nothing else matters.
3. **Determinism** — same input → same output
4. **Device/dtype compatibility** — works on CPU, moves to GPU correctly
5. **No trainable parameters** — `sum(p.numel() for p in bsp.parameters()) == 0`
6. **Numerical precision** — invariance holds to `atol=1e-4` in float32

For every module where inversion is mathematically available:
7\. **Inversion test** — call `bsp.invert(beta)` and check reconstruction error up to the known group-action indeterminacy.

For modules where inversion is not yet available (e.g., current SO(3) selective roadmap):
8\. **Explicit NotImplemented test** — ensure `bsp.invert(...)` raises a clear, documented `NotImplementedError`.

```python
# Example: canonical invariance test pattern
def test_invariance(self):
    bsp = CnonCn(n=8)
    f = torch.randn(4, 8)
    shift = 3  # arbitrary cyclic shift
    f_shifted = torch.roll(f, shift, dims=-1)

    torch.testing.assert_close(bsp(f), bsp(f_shifted), atol=1e-4, rtol=1e-4)
```

______________________________________________________________________

## Migration from v0.1.0

| v0.1.0                                | v0.2.0                                          | Notes                                 |
| ------------------------------------- | ----------------------------------------------- | ------------------------------------- |
| `SO3onS2(lmax=5)(coeffs)`             | `SO3onS2(lmax=5, nlat=64, nlon=128)(f_spatial)` | Now takes spatial signal              |
| `bispectrum(f_coeffs, l1, l2, cg_fn)` | Removed                                         | Use `SO3onS2` module                  |
| `clebsch_gordan(l1, l2)`              | Removed                                         | Internal; use bundled `cg_lmax5.json` |
| `compute_padding_indices(...)`        | Removed (internal)                              |                                       |
| `pad_sh_coefficients(...)`            | Removed (internal)                              |                                       |
| `get_full_sh_coefficients(...)`       | Removed (internal)                              |                                       |

______________________________________________________________________

## References

- Mataigne et al. (2024). *The Selective G-Bispectrum and its Inversion: Applications to G-Invariant Networks*. NeurIPS 2024. [arXiv:2407.07655](https://arxiv.org/abs/2407.07655)
- Kakarala (1992). *Triple Correlation on Groups*. PhD thesis, UC Irvine.
- Kakarala (2009). *Bispectrum on Finite Groups*. ICASSP 2009.
- Cohen & Welling (2016). *Group Equivariant Convolutional Networks*. ICML 2016.
- Weiler & Cesa (2019). *General E(2)-Equivariant Steerable CNNs*. NeurIPS 2019.
