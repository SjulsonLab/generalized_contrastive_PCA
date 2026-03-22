# gcpca 0.0.0.9000

- Created initial S3 package scaffold for dense and sparse gcPCA.
- Added `gcPCA()` and `sparse_gcPCA()` fit constructors with validation helpers.
- Added S3 methods for `predict()`, `print()`, `summary()`, `coef()`, `fitted()`, and `plot()`.
- Added utility accessors `scores()` and `loadings()`.
- Added `testthat` tests for S3 dispatch and core return types.
