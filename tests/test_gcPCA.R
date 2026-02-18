# Tests for generalized contrastive PCA (R implementation)
#
# Run with: Rscript tests/test_gcPCA.R
# from the repository root directory.

# Source the R implementations
source("R/gcPCA.R")
source("R/sparse_gcPCA.R")

# ---- Test helpers ----

test_count <- 0
pass_count <- 0
fail_count <- 0
failures <- list()

assert <- function(condition, msg) {
  test_count <<- test_count + 1
  if (isTRUE(condition)) {
    pass_count <<- pass_count + 1
    cat(sprintf("  PASS: %s\n", msg))
  } else {
    fail_count <<- fail_count + 1
    failures[[length(failures) + 1]] <<- msg
    cat(sprintf("  FAIL: %s\n", msg))
  }
}

assert_close <- function(x, y, tol = 1e-6, msg = "") {
  assert(all(abs(x - y) < tol), msg)
}

# ---- Generate synthetic data ----

set.seed(42)
n_a <- 80
n_b <- 60
p <- 20

Rb <- matrix(rnorm(n_b * p), nrow = n_b, ncol = p)
Ra <- matrix(rnorm(n_a * p), nrow = n_a, ncol = p)
Ra[, 1] <- Ra[, 1] + rnorm(n_a) * 5
Ra[, 2] <- Ra[, 2] + rnorm(n_a) * 3

# Small data for quick tests
set.seed(42)
n_a_sm <- 30
n_b_sm <- 25
p_sm <- 8
Rb_sm <- matrix(rnorm(n_b_sm * p_sm), nrow = n_b_sm, ncol = p_sm)
Ra_sm <- matrix(rnorm(n_a_sm * p_sm), nrow = n_a_sm, ncol = p_sm)
Ra_sm[, 1] <- Ra_sm[, 1] + rnorm(n_a_sm) * 4

cat("\n========================================\n")
cat("  R gcPCA Tests\n")
cat("========================================\n\n")

# ---- Test gcPCA: non-orthogonal methods ----

cat("--- gcPCA non-orthogonal methods ---\n")

for (method in c("v1", "v2", "v3", "v4")) {
  result <- tryCatch({
    mdl <- gcPCA(Ra, Rb, method = method)
    TRUE
  }, error = function(e) {
    cat(sprintf("  ERROR in %s: %s\n", method, e$message))
    FALSE
  })
  assert(result, sprintf("gcPCA %s fits without error", method))

  if (result) {
    mdl <- gcPCA(Ra, Rb, method = method)
    assert(nrow(mdl$loadings) == p, sprintf("gcPCA %s: loadings has %d rows (features)", method, p))
    assert(!is.null(mdl$Ra_scores), sprintf("gcPCA %s: Ra_scores exists", method))
    assert(!is.null(mdl$Rb_scores), sprintf("gcPCA %s: Rb_scores exists", method))
    assert(!is.null(mdl$objective_values), sprintf("gcPCA %s: objective_values exists", method))
  }
}

# ---- Test gcPCA: orthogonal methods ----

cat("\n--- gcPCA orthogonal methods ---\n")

for (method in c("v2.1", "v3.1", "v4.1")) {
  result <- tryCatch({
    mdl <- gcPCA(Ra_sm, Rb_sm, method = method, Ncalc = 4)
    TRUE
  }, error = function(e) {
    cat(sprintf("  ERROR in %s: %s\n", method, e$message))
    FALSE
  })
  assert(result, sprintf("gcPCA %s fits without error", method))

  if (result) {
    mdl <- gcPCA(Ra_sm, Rb_sm, method = method, Ncalc = 4)
    X <- mdl$loadings
    gram <- t(X) %*% X
    off_diag_max <- max(abs(gram - diag(diag(gram))))
    assert(off_diag_max < 1e-5, sprintf("gcPCA %s: loadings are orthogonal (max off-diag = %.2e)", method, off_diag_max))
  }
}

# ---- Test gcPCA: v4 objective values bounded ----

cat("\n--- gcPCA v4 value bounds ---\n")

mdl <- gcPCA(Ra, Rb, method = "v4")
assert(all(mdl$objective_values >= -1 - 1e-10), "v4 objective values >= -1")
assert(all(mdl$objective_values <= 1 + 1e-10), "v4 objective values <= 1")

# ---- Test gcPCA: v2 objective values positive ----

cat("\n--- gcPCA v2 value bounds ---\n")

mdl <- gcPCA(Ra, Rb, method = "v2")
assert(all(mdl$objective_values > 0), "v2 objective values > 0")

# ---- Test gcPCA: different feature counts should error ----

cat("\n--- gcPCA input validation ---\n")

result <- tryCatch({
  gcPCA(matrix(rnorm(50), 10, 5), matrix(rnorm(60), 10, 6), method = "v4")
  FALSE
}, error = function(e) TRUE)
assert(result, "Different feature counts raises error")

# ---- Test gcPCA: invalid method should error ----

result <- tryCatch({
  gcPCA(Ra, Rb, method = "v5")
  FALSE
}, error = function(e) TRUE)
assert(result, "Invalid method raises error")

# ---- Test gcPCA: rank-deficient data ----

cat("\n--- gcPCA rank-deficient data ---\n")

set.seed(42)
Ra_rd <- matrix(rnorm(5 * 20), nrow = 5, ncol = 20)
Rb_rd <- matrix(rnorm(5 * 20), nrow = 5, ncol = 20)
result <- tryCatch({
  mdl <- suppressWarnings(gcPCA(Ra_rd, Rb_rd, method = "v4"))
  !is.null(mdl$loadings)
}, error = function(e) {
  cat(sprintf("  ERROR: %s\n", e$message))
  FALSE
})
assert(result, "Rank-deficient data handled")

# ---- Test gcPCA: equal data gives zero v4 values ----

cat("\n--- gcPCA equal data ---\n")

set.seed(42)
data_eq <- matrix(rnorm(30 * 10), nrow = 30, ncol = 10)
mdl <- gcPCA(data_eq, data_eq, method = "v4", normalize_flag = FALSE)
assert_close(mdl$objective_values, rep(0, length(mdl$objective_values)), tol = 1e-10,
             msg = "Equal data gives zero v4 objective values")

# ---- Test gcPCA: predict method ----

cat("\n--- gcPCA predict ---\n")

mdl <- gcPCA(Ra_sm, Rb_sm, method = "v4")

# predict with no new data returns training scores
pred_train <- predict(mdl)
assert(!is.null(pred_train$Ra_scores), "predict() with no args returns Ra_scores")
assert(!is.null(pred_train$Rb_scores), "predict() with no args returns Rb_scores")

# predict with new data
set.seed(99)
Ra_new <- matrix(rnorm(10 * p_sm), nrow = 10, ncol = p_sm)
pred_new <- predict(mdl, Ra = Ra_new)
assert(nrow(pred_new$Ra_scores) == 10, "predict() with new Ra has correct rows")

# ---- Test gcPCA: reproducibility ----

cat("\n--- gcPCA reproducibility ---\n")

mdl1 <- gcPCA(Ra, Rb, method = "v4")
mdl2 <- gcPCA(Ra, Rb, method = "v4")
assert_close(abs(mdl1$loadings), abs(mdl2$loadings), tol = 1e-10,
             msg = "Same data gives same loadings")

# ---- Test sparse_gcPCA: basic fit ----

cat("\n--- sparse_gcPCA ---\n")

lambdas <- c(0.1, 0.5, 1.0)

for (method in c("v2", "v3", "v4")) {
  result <- tryCatch({
    mdl <- sparse_gcPCA(Ra_sm, Rb_sm, method = method, Nsparse = 2, lasso_penalty = lambdas)
    TRUE
  }, error = function(e) {
    cat(sprintf("  ERROR in sparse %s: %s\n", method, e$message))
    FALSE
  })
  assert(result, sprintf("sparse_gcPCA %s fits without error", method))

  if (result) {
    mdl <- sparse_gcPCA(Ra_sm, Rb_sm, method = method, Nsparse = 2, lasso_penalty = lambdas)
    assert(length(mdl$sparse_loadings) == length(lambdas),
           sprintf("sparse_gcPCA %s: correct number of loading sets", method))
    assert(nrow(mdl$sparse_loadings[[1]]) == p_sm,
           sprintf("sparse_gcPCA %s: loadings have correct number of rows", method))
    assert(ncol(mdl$sparse_loadings[[1]]) == 2,
           sprintf("sparse_gcPCA %s: loadings have Nsparse columns", method))
  }
}

# ---- Test sparse_gcPCA v1: basic fit ----

cat("\n--- sparse_gcPCA v1 ---\n")

result <- tryCatch({
  mdl <- sparse_gcPCA(Ra_sm, Rb_sm, method = "v1", Nsparse = 2, lasso_penalty = lambdas)
  TRUE
}, error = function(e) {
  cat(sprintf("  ERROR in sparse v1: %s\n", e$message))
  FALSE
})
assert(result, "sparse_gcPCA v1 fits without error")

if (result) {
  mdl <- sparse_gcPCA(Ra_sm, Rb_sm, method = "v1", Nsparse = 2, lasso_penalty = lambdas)
  assert(length(mdl$sparse_loadings) == length(lambdas),
         "sparse_gcPCA v1: correct number of loading sets")
  assert(nrow(mdl$sparse_loadings[[1]]) == p_sm,
         "sparse_gcPCA v1: loadings have correct number of rows")
}

# ---- Test sparse_gcPCA v1: scores output ----

cat("\n--- sparse_gcPCA v1 scores ---\n")

mdl <- sparse_gcPCA(Ra_sm, Rb_sm, method = "v1", Nsparse = 2, lasso_penalty = lambdas)
assert(length(mdl$Ra_scores) == length(lambdas), "v1 Ra_scores list length matches lambdas")
assert(length(mdl$Rb_scores) == length(lambdas), "v1 Rb_scores list length matches lambdas")
assert(length(mdl$Ra_values) == length(lambdas), "v1 Ra_values list length matches lambdas")
assert(length(mdl$Rb_values) == length(lambdas), "v1 Rb_values list length matches lambdas")

# ---- Test sparse_gcPCA: scores output ----

cat("\n--- sparse_gcPCA scores ---\n")

mdl <- sparse_gcPCA(Ra_sm, Rb_sm, method = "v4", Nsparse = 2, lasso_penalty = lambdas)
assert(length(mdl$Ra_scores) == length(lambdas), "Ra_scores list length matches lambdas")
assert(length(mdl$Rb_scores) == length(lambdas), "Rb_scores list length matches lambdas")
assert(length(mdl$Ra_values) == length(lambdas), "Ra_values list length matches lambdas")
assert(length(mdl$Rb_values) == length(lambdas), "Rb_values list length matches lambdas")

# ---- Test sparse_gcPCA: predict method ----

cat("\n--- sparse_gcPCA predict ---\n")

mdl <- sparse_gcPCA(Ra_sm, Rb_sm, method = "v4", Nsparse = 2, lasso_penalty = c(0.1))

# predict with no new data
pred_train <- predict(mdl)
assert(!is.null(pred_train$Ra_transform), "sparse predict() with no args returns Ra_transform")
assert(!is.null(pred_train$Rb_transform), "sparse predict() with no args returns Rb_transform")

# predict with new data
set.seed(99)
Ra_new <- matrix(rnorm(10 * p_sm), nrow = 10, ncol = p_sm)
pred_new <- predict(mdl, Ra = Ra_new)
assert(!is.null(pred_new$Ra_transform), "sparse predict() with new Ra works")

# ---- Summary ----

cat("\n========================================\n")
cat(sprintf("  Results: %d passed, %d failed out of %d tests\n", pass_count, fail_count, test_count))
cat("========================================\n")

if (fail_count > 0) {
  cat("\nFailed tests:\n")
  for (f in failures) {
    cat(sprintf("  - %s\n", f))
  }
  quit(status = 1)
} else {
  cat("\nAll tests passed!\n")
  quit(status = 0)
}
