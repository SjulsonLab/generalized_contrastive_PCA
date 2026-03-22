set.seed(123)
Ra <- matrix(rnorm(30 * 4), ncol = 4)
Rb <- matrix(rnorm(28 * 4), ncol = 4)

dense_fit <- gcPCA(Ra, Rb, method = "v4", Ncalc = 2, normalize_flag = TRUE)
sparse_fit <- sparse_gcPCA(
  Ra, Rb,
  method = "v4",
  Ncalc = 2,
  Nsparse = 2,
  lasso_penalty = c(0.05, 0.1),
  max_steps = 20
)

test_that("S3 classes are set correctly", {
  expect_s3_class(dense_fit, "gcPCA")
  expect_s3_class(sparse_fit, "sparse_gcPCA")
  expect_s3_class(sparse_fit, "gcPCA")
})

test_that("print returns object invisibly", {
  expect_identical(print(dense_fit), dense_fit)
})

test_that("summary returns summary.gcPCA", {
  smry <- summary(dense_fit)
  expect_s3_class(smry, "summary.gcPCA")
})

test_that("predict returns expected score containers", {
  dense_pred <- predict(dense_fit, Ra = Ra, Rb = Rb)
  expect_true(is.matrix(dense_pred$Ra_scores))
  expect_true(is.matrix(dense_pred$Rb_scores))

  sparse_pred <- predict(sparse_fit, Ra = Ra, Rb = Rb)
  expect_named(sparse_pred, c("Ra_scores", "Rb_scores"))
  expect_true(is.list(sparse_pred$Ra_scores))
  expect_true(is.list(sparse_pred$Rb_scores))
  expect_true(all(vapply(sparse_pred$Ra_scores, is.matrix, logical(1))))
  expect_true(all(vapply(sparse_pred$Rb_scores, is.matrix, logical(1))))
})
