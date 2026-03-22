`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

pc_names <- function(k) {
  if (k <= 0) {
    character(0)
  } else {
    paste0("gcPC", seq_len(k))
  }
}

#' Get gcPCA scores
#'
#' Convenience wrapper around `predict()` for `"gcPCA"` and `"sparse_gcPCA"`
#' objects.
#'
#' @param object A fitted model object.
#' @param newdata Optional matrix (treated as `Ra`) or list with `Ra`/`Rb`.
#' @param ... Extra arguments passed to `predict()`.
#'
#' @return A list of projected scores.
#' @export
scores <- function(object, newdata = NULL, ...) {
  if (is.null(newdata)) {
    return(predict(object, ...))
  }
  if (is.list(newdata)) {
    return(do.call(predict, c(list(object = object), newdata, list(...))))
  }
  predict(object, Ra = newdata, ...)
}

#' Get gcPCA loadings
#'
#' Convenience wrapper around `coef()` for `"gcPCA"` and `"sparse_gcPCA"`
#' objects.
#'
#' @param object A fitted model object.
#' @param ... Extra arguments passed to `coef()`.
#'
#' @return A loading matrix or a named list of loading matrices.
#' @export
loadings <- function(object, ...) {
  coef(object, ...)
}
