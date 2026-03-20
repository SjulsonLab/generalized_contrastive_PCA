#' Project data with a fitted gcPCA model
#'
#' @param object A fitted `"gcPCA"` or `"sparse_gcPCA"` object.
#' @param Ra Optional matrix for condition A.
#' @param Rb Optional matrix for condition B.
#' @param ... Unused.
#'
#' @return For `gcPCA`, a list with `Ra_scores` and `Rb_scores` matrices. For
#' `sparse_gcPCA`, a list with `Ra_scores` and `Rb_scores` as lists of matrices
#' (one per penalty value).
#'
#' @method predict gcPCA
#' @importFrom stats predict
#' @export
predict.gcPCA <- function(object, Ra = NULL, Rb = NULL, ...) {
  # object: result of gcPCA
  # newdata: matrix with same columns as Ra/Rb used in fitting

  if (is.null(Ra) && is.null(Rb)) {
    # Return training scores if newdata not supplied
    return(list(Ra_scores = object$Ra_scores * object$Ra_values,
                Rb_scores = object$Rb_scores * object$Rb_values))
  }

  loadings <- object$loadings
  Ra_scores <- NULL
  Rb_scores <- NULL

  # Process Ra if provided
  if (!is.null(Ra)) {
    Ra <- as.matrix(Ra)
    # Normalize new data if gcPCA was fitted with normalization
    if (object$normalize_flag) {
      Ra <- object$normalize(Ra)
    }
    # Project new data onto gcPCA loadings
    Ra_scores <- Ra %*% loadings
  } else {
    # Apply Ra_values to the scores so it matches magnitude with new data
    Ra_scores <- object$Ra_scores * object$Ra_values
  }

  # Process Rb if provided
  if (!is.null(Rb)) {
    Rb <- as.matrix(Rb)
    # Normalize new data if gcPCA was fitted with normalization
    if (object$normalize_flag) {
      Rb <- object$normalize(Rb)
    }
    # Project new data onto gcPCA loadings
    Rb_scores <- Rb %*% loadings
  } else {
    # Apply Rb_values to the scores so it matches magnitude with new data
    Rb_scores <- object$Rb_scores * object$Rb_values
  }

  return(list(Ra_scores = Ra_scores, Rb_scores = Rb_scores))
}

#' @rdname predict.gcPCA
#' @method predict sparse_gcPCA
#' @export
predict.sparse_gcPCA <- function(object, Ra = NULL, Rb = NULL, ...) {
  # object: result of sparse_gcPCA
  # Ra/Rb: matrices with same columns as the Ra/Rb used in fitting
  #
  # Returns:
  #   - If Ra and Rb are NULL: the training scores/values stored in the object
  #   - Otherwise: updated scores/values for any supplied Ra and/or Rb
  #     (still as lists, one entry per lasso penalty)

  # If no new data, just return what was fitted
  if (is.null(Ra) && is.null(Rb)) {
    Ra_scores <- lapply(seq_along(object$Ra_scores), function(i) {
        scores <- object$Ra_scores[[i]]  # n x k_i
        vals   <- object$Ra_values[[i]]  # length k_i
        sweep(scores, 2, vals, "*")      # n x k_i, raw transform
    })
    
    Rb_scores <- lapply(seq_along(object$Rb_scores), function(i) {
        scores <- object$Rb_scores[[i]]  # n x k_i
        vals   <- object$Rb_values[[i]]  # length k_i
        sweep(scores, 2, vals, "*")      # n x k_i, raw transform
      })

    return(list(
      Ra_scores = Ra_scores,
      Rb_scores = Rb_scores
    ))
  }

  # Initialize
  Ra_scores <- NULL
  Rb_scores <- NULL

  # ancillary function to project data onto a list of sparse loading matrices
  project_sparse <- function(X, loadings_list) {
    X <- as.matrix(X)

    scores_list <- vector("list", length(loadings_list))
    # values_list <- vector("list", length(loadings_list))

    for (i in seq_along(loadings_list)) {
      L <- loadings_list[[i]]              # p x k_i
      temp_scores <- X %*% L                      # n x k_i

      scores_list[[i]] <- temp_scores
      # Norm-1 scores (same as in sparse_gcPCA fitting)
      # scores_list[[i]] <- sweep(temp, 2, norms, "/")
      # values_list[[i]] <- norms
    }

    list(scores = scores_list)
  }

  sparse_loadings <- object$sparse_loadings
  # normalize_flag <- isTRUE(object$normalize_flag)

  # Try to pull the same normalization function used in gcPCA()
  # normalize_fun <- NULL
  # if (!is.null(object$gcPCA_object) &&
  #     !is.null(object$gcPCA_object$normalize)) {
  #   normalize_fun <- object$gcPCA_object$normalize
  # }

  # Process Ra if provided
  if (!is.null(Ra)) {
    Ra_proc <- as.matrix(Ra)


    # if (normalize_flag && !is.null(normalize_fun)) {
      # Ra_proc <- normalize_fun(Ra_proc)
    # }

    proj_Ra <- project_sparse(Ra_proc, sparse_loadings)
    Ra_scores <- proj_Ra$scores
  }

  # Process Rb if provided
  if (!is.null(Rb)) {
    Rb_proc <- as.matrix(Rb)

    # if (normalize_flag && !is.null(normalize_fun)) {
    #   Rb_proc <- normalize_fun(Rb_proc)
    # }

    proj_Rb <- project_sparse(Rb_proc, sparse_loadings)
    Rb_scores <- proj_Rb$scores
  }

  return(list(
    Ra_scores = Ra_scores,
    Rb_scores = Rb_scores
  ))
}
