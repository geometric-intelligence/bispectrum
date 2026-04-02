# bispectrum

[![Tests](https://github.com/geometric-intelligence/bispectrum/actions/workflows/tests.yml/badge.svg)](https://github.com/geometric-intelligence/bispectrum/actions/workflows/tests.yml)
[![Pre-commit](https://github.com/geometric-intelligence/bispectrum/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/geometric-intelligence/bispectrum/actions/workflows/pre-commit.yml)
[![codecov](https://codecov.io/github/geometric-intelligence/bispectrum/graph/badge.svg?token=J6GGY4VK1E)](https://codecov.io/github/geometric-intelligence/bispectrum)

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
git clone https://github.com/geometric-intelligence/bispectrum.git
cd bispectrum
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

```python
import bispectrum
```

## License

MIT
