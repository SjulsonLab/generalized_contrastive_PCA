#' Print a gcPCA model
#'
#' @param x A fitted `"gcPCA"` or `"sparse_gcPCA"` object.
#' @param ... Unused.
#'
#' @return `x`, invisibly.
#' @method print gcPCA
#' @export
print.gcPCA <- function(x, ...) {
  sparse_model <- inherits(x, "sparse_gcPCA")
  base_obj <- if (sparse_model && !is.null(x$gcPCA_object)) x$gcPCA_object else x

  n_a <- if (!is.null(base_obj$Ra)) nrow(base_obj$Ra) else NA_integer_
  n_b <- if (!is.null(base_obj$Rb)) nrow(base_obj$Rb) else NA_integer_
  p <- if (!is.null(base_obj$loadings)) nrow(base_obj$loadings) else NA_integer_
  k <- if (sparse_model) {
    if (!is.null(x$sparse_loadings) && length(x$sparse_loadings) > 0) {
      ncol(x$sparse_loadings[[1]])
    } else {
      NA_integer_
    }
  } else if (!is.null(base_obj$loadings)) {
    ncol(base_obj$loadings)
  } else {
    NA_integer_
  }

  model_label <- if (sparse_model) "sparse_gcPCA" else "gcPCA"
  method <- base_obj$method %||% NA_character_
  preprocessing <- if (isTRUE(base_obj$normalize_flag)) "zscore + l2" else "none"

  cat("<", model_label, " model>\n", sep = "")
  cat("n (Ra/Rb): ", n_a, " / ", n_b, "\n", sep = "")
  cat("p: ", p, " | k: ", k, "\n", sep = "")
  cat("method: ", method, "\n", sep = "")
  cat("preprocessing: ", preprocessing, "\n", sep = "")
  if (sparse_model && !is.null(x$lasso_penalty)) {
    cat("n_lambda: ", length(x$lasso_penalty), "\n", sep = "")
  }

  invisible(x)
}

#' Summarize a gcPCA model
#'
#' @param object A fitted `"gcPCA"` or `"sparse_gcPCA"` object.
#' @param ... Unused.
#'
#' @return A `"summary.gcPCA"` list with eigenvalues, standard deviations,
#' explained variance summaries, and tuning settings.
#' @method summary gcPCA
#' @export
summary.gcPCA <- function(object, ...) {
  sparse_model <- inherits(object, "sparse_gcPCA")
  base_obj <- if (sparse_model && !is.null(object$gcPCA_object)) object$gcPCA_object else object

  eigenvalues <- base_obj$objective_values
  if (is.null(eigenvalues)) {
    eigenvalues <- numeric(0)
  } else {
    eigenvalues <- as.numeric(eigenvalues)
  }

  sdev <- sqrt(abs(eigenvalues))
  denom <- sum(abs(eigenvalues))
  prop_var <- if (length(eigenvalues) == 0 || denom == 0) numeric(0) else abs(eigenvalues) / denom
  cum_var <- if (length(prop_var) == 0) numeric(0) else cumsum(prop_var)

  tuning <- list(
    method = base_obj$method %||% NA_character_,
    normalize_flag = isTRUE(base_obj$normalize_flag),
    Ncalc = if (!is.null(base_obj$loadings)) ncol(base_obj$loadings) else NA_integer_
  )

  if (sparse_model) {
    tuning$Nsparse <- object$Nsparse %||% NA_integer_
    tuning$lasso_penalty <- object$lasso_penalty %||% numeric(0)
    tuning$ridge_penalty <- object$ridge_penalty %||% NA_real_
  }

  out <- list(
    eigenvalues = eigenvalues,
    sdev = sdev,
    prop_var = prop_var,
    cum_var = cum_var,
    tuning = tuning,
    sparse = sparse_model
  )
  class(out) <- "summary.gcPCA"
  out
}

#' Print a gcPCA summary
#'
#' @param x A `"summary.gcPCA"` object from [summary.gcPCA()].
#' @param ... Unused.
#'
#' @return `x`, invisibly.
#' @method print summary.gcPCA
#' @export
print.summary.gcPCA <- function(x, ...) {
  cat("<summary.gcPCA>\n")
  if (length(x$eigenvalues) == 0) {
    cat("No eigenvalues available.\n")
  } else {
    tab <- data.frame(
      Component = pc_names(length(x$eigenvalues)),
      Eigenvalue = signif(x$eigenvalues, 4),
      SD = signif(x$sdev, 4),
      PVE = signif(x$prop_var, 4),
      Cumulative = signif(x$cum_var, 4),
      check.names = FALSE
    )
    print(tab, row.names = FALSE)
  }

  cat("\nKey settings:\n")
  cat("  method: ", x$tuning$method, "\n", sep = "")
  cat("  normalize_flag: ", x$tuning$normalize_flag, "\n", sep = "")
  cat("  Ncalc: ", x$tuning$Ncalc, "\n", sep = "")
  if (!is.null(x$tuning$Nsparse)) {
    cat("  Nsparse: ", x$tuning$Nsparse, "\n", sep = "")
  }
  if (!is.null(x$tuning$lasso_penalty)) {
    cat("  n_lambda: ", length(x$tuning$lasso_penalty), "\n", sep = "")
  }

  invisible(x)
}

#' Extract gcPCA loadings
#'
#' @param object A fitted `"gcPCA"` or `"sparse_gcPCA"` object.
#' @param ... Unused.
#'
#' @return For dense models, a loadings matrix. For sparse models, a named list
#' of loading matrices (one per lasso penalty).
#' @method coef gcPCA
#' @export
coef.gcPCA <- function(object, ...) {
  if (inherits(object, "sparse_gcPCA")) {
    if (is.null(object$sparse_loadings)) {
      stop("No sparse loadings available in this object.", call. = FALSE)
    }
    out <- object$sparse_loadings
    if (is.null(names(out))) {
      lambda <- object$lasso_penalty
      if (is.null(lambda) || length(lambda) != length(out)) {
        lambda <- seq_along(out)
      }
      names(out) <- paste0("lambda_", as.character(lambda))
    }
    return(out)
  }

  if (is.null(object$loadings)) {
    stop("No loadings available in this object.", call. = FALSE)
  }
  object$loadings
}

#' @rdname coef.gcPCA
#' @method coef sparse_gcPCA
#' @export
coef.sparse_gcPCA <- function(object, ...) {
  coef.gcPCA(object, ...)
}

#' Return fitted training scores
#'
#' @param object A fitted `"gcPCA"` or `"sparse_gcPCA"` object.
#' @param ... Passed to `predict()`.
#'
#' @return A list with `Ra_scores` and `Rb_scores`.
#' @method fitted gcPCA
#' @export
fitted.gcPCA <- function(object, ...) {
  fit_scores <- predict(object, ...)
  if (!is.list(fit_scores) || is.null(fit_scores$Ra_scores) || is.null(fit_scores$Rb_scores)) {
    stop("Training scores are not available for this object.", call. = FALSE)
  }
  fit_scores
}

#' Plot gcPCA scores
#'
#' @param x A fitted `"gcPCA"` or `"sparse_gcPCA"` object.
#' @param which Integer vector of length 2 indicating components to plot.
#' @param ... Extra graphical parameters passed to `plot()`.
#'
#' @return `x`, invisibly.
#' @method plot gcPCA
#' @export
plot.gcPCA <- function(x, which = c(1, 2), ...) {
  which <- as.integer(which)
  if (length(which) != 2L || any(is.na(which)) || any(which < 1L)) {
    stop("`which` must be an integer vector of length 2 with positive values.", call. = FALSE)
  }

  fit_scores <- fitted(x)
  ra_scores <- fit_scores$Ra_scores
  rb_scores <- fit_scores$Rb_scores

  if (inherits(x, "sparse_gcPCA")) {
    if (!is.list(ra_scores) || !is.list(rb_scores) || length(ra_scores) == 0 || length(rb_scores) == 0) {
      stop("Sparse model scores are not available for plotting.", call. = FALSE)
    }
    ra_scores <- ra_scores[[1]]
    rb_scores <- rb_scores[[1]]
  }

  if (!is.matrix(ra_scores) || !is.matrix(rb_scores)) {
    stop("Scores must be matrices for plotting.", call. = FALSE)
  }
  if (max(which) > ncol(ra_scores) || max(which) > ncol(rb_scores)) {
    stop("Requested components exceed available score dimensions.", call. = FALSE)
  }

  xlab <- paste0("gcPC", which[1])
  ylab <- paste0("gcPC", which[2])
  plot(ra_scores[, which[1]], ra_scores[, which[2]],
       xlab = xlab, ylab = ylab, col = "steelblue", pch = 16, ...)
  points(rb_scores[, which[1]], rb_scores[, which[2]], col = "firebrick", pch = 17)
  legend("topright", legend = c("Ra", "Rb"), col = c("steelblue", "firebrick"),
         pch = c(16, 17), bty = "n")

  invisible(x)
}
