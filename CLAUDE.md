# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Generalized Contrastive PCA (gcPCA) — a hyperparameter-free dimensionality reduction method for comparing high-dimensional datasets across experimental conditions. Published in PLOS Computational Biology. Implementations exist in Python (primary), R, and MATLAB.

## Build and Development Commands

```bash
# Install in development mode
pip install -e .

# Install dependencies
pip install numpy scipy numba

# Run tutorials (marimo interactive notebooks)
pip install marimo
marimo edit tutorial/gcpca_tutorial1.py

# Build distributions for PyPI
python -m build
```

There is no test suite, linter, or formatter configured. PyPI publishing is automated via GitHub Actions on release creation (`.github/workflows/python-publish.yml`).

## Architecture

### Python Package (`generalized_contrastive_PCA/contrastive_methods.py`)

Single file containing the full implementation. Two main classes with a scikit-learn-style API (`fit`/`transform`/`fit_transform`):

- **`gcPCA`** — Core class. Takes two data matrices `Ra` (foreground) and `Rb` (background), both shaped (samples × features). Supports multiple method variants:
  - `v1`: Contrastive (Ra - α·Rb)
  - `v2`/`v2.1`: Ratio (Ra / Rb)
  - `v3`/`v3.1`: Normalized difference ((Ra-Rb) / Rb)
  - `v4`/`v4.1` (default): Index normalized ((Ra-Rb) / (Ra+Rb))
  - `.1` suffixes return orthogonal dimensions via iterative deflation

- **`sparse_gcPCA`** — Wraps `gcPCA` internally, then applies elastic net (L1+L2) penalty via variable projection optimization. Uses Numba `@njit` for the inner loop.

Key output attributes (trailing underscore convention): `loadings_`, `gcPCA_values_`, `Ra_scores_`, `Rb_scores_`, `Ra_values_`, `Rb_values_`.

### R (`R/gcPCA.R`, `R/sparse_gcPCA.R`) and MATLAB (`matlab/`) implementations

Parallel implementations maintained for cross-language compatibility. R uses S3-style classes; MATLAB uses class methods.

### Numerical Considerations

- Ill-conditioned denominator matrices are regularized when condition number exceeds `cond_number` threshold (default 10^13)
- Eigendecomposition uses `np.linalg.eigh()` for symmetric matrices
- Normalization: optional z-score + L2-norm via `normalize_flag`

## Conventions

- Data matrices: samples in rows, features in columns
- `Ra` = foreground/condition A, `Rb` = background/condition B
- Covariance notation: `RaRa`, `RbRb`, `JRaRaJ` (matrix product names)
- Branches: `main` (production), `lukedev` (development)
