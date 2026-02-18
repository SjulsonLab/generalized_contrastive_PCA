# Bug Findings Report (Python, MATLAB, R)

Date: 2026-02-18
Repository: `generalized_contrastive_PCA`
Branch: `codex/luke-codex`

## Scope

I reviewed and smoke-tested:

- Python: `generalized_contrastive_PCA/contrastive_methods.py`
- MATLAB: `matlab/gcPCA.m`, `matlab/sparse_gcPCA.m`, and helper files
- R: `R/gcPCA.R`, `R/sparse_gcPCA.R`

The findings below are focused on correctness and runtime behavior (not style).

## High-Severity Findings

### 1) Python `sparse_gcPCA(method='v1')` is broken by variable-name bugs

- File: `generalized_contrastive_PCA/contrastive_methods.py`
- Lines: around 440, 442-443, 446, 449
- Issues:
  - Uses `self.lambdas` (undefined) instead of `self.lasso_penalty`
  - Uses `final_pos_loadings_`/`final_neg_loadings_` names that were never populated
- Result:
  - Runtime failures (`AttributeError`, then `UnboundLocalError`) when running sparse v1

### 2) R `sparse_gcPCA(method='v1')` calls an undefined function

- File: `R/sparse_gcPCA.R`
- Lines: 52-54 and TODO at 180
- Issue:
  - Calls `J_variable_projection(...)`, but no implementation exists
- Result:
  - Hard runtime error: `could not find function "J_variable_projection"`

### 3) MATLAB `sparse_gcPCA(v1)` can index out of bounds

- File: `matlab/sparse_gcPCA.m`
- Lines: 194-201
- Issue:
  - Uses `Nsparse` to slice per-lambda blocks even when positive/negative component counts are smaller
- Result:
  - Frequent runtime failures under random seeds:
    - `Index in position 2 exceeds array bounds. Index must not exceed ...`

### 4) Python `gcPCA.fit()` can fail on singular denominator cases

- File: `generalized_contrastive_PCA/contrastive_methods.py`
- Lines: around 182-183 and 191
- Issue:
  - Denominator-conditioning fix can still leave a singular matrix
- Result:
  - `LinAlgError: Singular matrix` in edge cases (e.g., zero-variance/background denominator)

### 5) R `gcPCA` can fail on singular denominator cases

- File: `R/gcPCA.R`
- Lines: around 141-155
- Issue:
  - Same pattern as Python: regularization path can remain singular before `solve(M)`
- Result:
  - Hard LAPACK failure:
    - `system is exactly singular: U[1,1] = 0`

## Medium-Severity Findings

### 6) R `gcPCA` returns incorrect score/value scaling shapes

- File: `R/gcPCA.R`
- Lines: 224-227
- Issue:
  - Uses one global matrix norm for scores/values instead of per-component norms
- Result:
  - `Ra_values`/`Rb_values` become scalars (length 1), inconsistent with expected per-component vectors
  - Downstream scaling semantics differ from Python/MATLAB behavior

### 7) Python docs/API mismatch (`gcPCA_values_` missing)

- Files:
  - `README.md` (documents `gcPCA_values_`)
  - `generalized_contrastive_PCA/contrastive_methods.py` (sets `objective_values_`, not `gcPCA_values_`)
- Result:
  - User code following docs can fail due to missing attribute

## Low-Severity Findings

### 8) Python transform methods hide real errors via bare `except`

- File: `generalized_contrastive_PCA/contrastive_methods.py`
- Lines: around 295-296 and 362-363
- Issue:
  - Catches all exceptions and prints “Loadings not defined...”
- Result:
  - Masks real shape/type/runtime errors and complicates debugging

### 9) R normalization differs from Python/MATLAB semantics

- File: `R/gcPCA.R`
- Lines: 19-22
- Issue:
  - Normalizes by single matrix 2-norm after scaling, not per-feature norming used in Python/MATLAB
- Result:
  - Cross-language output inconsistency

## Notes from Runtime Checks

- MATLAB `gcPCA.m` basic path ran successfully in this environment.
- MATLAB `gcPCA.m` singular denominator case did not hard-crash in the tested case, but emitted singular warnings.
- Python and R both showed hard failures in singular denominator edge cases.

## Suggested Fix Order

1. Fix all sparse v1 hard failures (Python, R, MATLAB) first.
2. Harden denominator regularization/inversion paths (Python + R).
3. Correct R score/value normalization outputs to per-component vectors.
4. Align Python API/docs (`gcPCA_values_` alias or docs update).
5. Replace bare `except` blocks with targeted exceptions and clear error propagation.
