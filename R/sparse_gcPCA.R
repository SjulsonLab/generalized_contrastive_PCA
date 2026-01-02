# sparse_gcPCA.R

# Implementation of sparse gcPCA in R, it is not yet guaranteed that this implementation is correct and works, still need to run some tests
sparse_gcPCA <- function(Ra, Rb, method = 'v4', Ncalc = NULL, normalize_flag = TRUE, Nsparse = NULL, Nshuffle = 0,
                         lasso_penalty = exp(seq(log(1e-2), log(1), length.out = 10)), ridge_penalty = 0,
                         alpha = 1, alpha_null = 0.975, tol = 1e-5, max_steps = 1000, cond_number = 1e13) {
  
  stopifnot(is.matrix(Ra), is.matrix(Rb))
  stopifnot(ncol(Ra) == ncol(Rb))
  stopifnot(all(lasso_penalty >= 0))
  stopifnot(method %in% c("v1","v2","v2.1","v3","v3.1","v4","v4.1"))
  
  if (is.null(Ncalc)) Ncalc <- ncol(Ra)
  if (is.null(Nsparse)) Nsparse <- ncol(Ra)

  # fit base gcPCA model
  gc_model <- gcPCA(Ra, Rb, method, Ncalc, Nshuffle, normalize_flag, alpha, alpha_null, cond_number)
  
  # extract needed components
  Ra <- gc_model$Ra
  Rb <- gc_model$Rb
  Jorig <- gc_model$J
  p <- ncol(Ra)
  
  # fitting function
  sparse_fit <- function() {

    # projecting data to shared components
    RaJ <- Ra %*% Jorig
    RbJ <- Rb %*% Jorig

    # covariance matrices
    JRaRaJ <- (t(RaJ) %*% RaJ) / (nrow(RaJ) - 1)
    JRbRbJ <- (t(RbJ) %*% RbJ) / (nrow(RbJ) - 1)
    
    # v1 is equivalent to sparse contrastive PCA
    if (method == 'v1') {

      # Compute theta matrix
      theta <- JRaRaJ - alpha * JRbRbJ
      eig <- eigen(theta)
      w <- eig$values
      v <- eig$vectors
      
      # Split positive/negative eigenvalues
      new_w_pos <- pmax(w, 0)
      new_w_neg <- -pmin(w, 0)
      
      # Calculating only the number of dimensions requested by user
      # TODO: J_variable_projection needs to be implemented
      final_loadings <- lapply(lasso_penalty, function(lmbda) {
        pos_load <- J_variable_projection(v %*% diag(sqrt(new_w_pos)) %*% t(v), Jorig, Nsparse, lmbda, ridge_penalty, max_steps, tol)
        neg_load <- J_variable_projection(v %*% diag(sqrt(new_w_neg)) %*% t(v), Jorig, Nsparse, lmbda, ridge_penalty, max_steps, tol)
        cbind(pos_load, neg_load)
      })
      
    } else {
      # v2-v4 methods
      if (method %in% c('v2', 'v2.1')) {
        numerator <- JRaRaJ
        denominator <- JRbRbJ
      } else if (method %in% c('v3', 'v3.1')) {
        numerator <- JRaRaJ - JRbRbJ
        denominator <- JRbRbJ
      } else if (method %in% c('v4', 'v4.1')) {
        numerator <- JRaRaJ - JRbRbJ
        denominator <- JRaRaJ + JRbRbJ
      }
      
      # find M matrix
      eig_denom <- eigen(denominator)
      M <- eig_denom$vectors %*% diag(sqrt(eig_denom$values)) %*% t(eig_denom$vectors)
      Minv <- solve(M)
      sigma <- t(Minv) %*% numerator %*% Minv
      
      # sparse loading
      final_loadings <- lapply(lasso_penalty, function(lmbda) {
        J_M_variable_projection(sigma, Jorig, M, Nsparse, lmbda, ridge_penalty, max_steps, tol)
      })
    }
    
    # normalize loading
    sparse_loadings <- lapply(final_loadings, function(load) {
      sweep(load, 2, apply(load, 2, function(x) norm(x, "2")), "/")
    })
    
    Ra_scores <- lapply(final_loadings, function(L) {
      temp <- Ra %*% L                 # projecting
      temp_norm <- apply(temp, 2, norm, type = "2") # getting the norm
      temp_norm[temp_norm == 0] <- 1   # avoid division by zero
      sweep(temp, 2, temp_norm, "/")   # norm 1 scores
    })
    
    Ra_values <- lapply(final_loadings, function(L) {
      temp <- Ra %*% L
      apply(temp, 2, norm, type = "2") # same as LA.norm(temp, axis=0)
    })

    Rb_scores <- lapply(final_loadings, function(L) {
      temp <- Rb %*% L                 # projecting
      temp_norm <- apply(temp, 2, norm, type = "2") # getting the norm
      temp_norm[temp_norm == 0] <- 1   # avoid division by zero
      sweep(temp, 2, temp_norm, "/")   # norm 1 scores
    })

    Rb_values <- lapply(final_loadings, function(L) {
      temp <- Rb %*% L
      apply(temp, 2, norm, type = "2") # same as LA.norm(temp, axis=0)
    })

    return(list(sparse_loadings = sparse_loadings, original_loadings = gc_model$loadings, Ra_scores = Ra_scores, Ra_values = Ra_values,
                Rb_scores = Rb_scores, Rb_values = Rb_values, Ra = Ra,
                Rb = Rb, J = Jorig, normalize_flag = normalize_flag, gcPCA_object = gc_model))
  }
  
  l2_norm_vec <- function(x) {
    sqrt(sum(x^2))
  }
  
  # J and M variable projection
  J_M_variable_projection <- function(theta_input, J, M, k, alpha, beta, max_iter, tol) {
    svd_theta <- svd(theta_input)
    Dmax <- svd_theta$d[1]
    B <- svd_theta$v[, 1:k, drop = FALSE]
    
    VD <- sweep(svd_theta$v, 2, svd_theta$d, "*")
    VD2 = sweep(svd_theta$v, 2, svd_theta$d^2, "*")
    
    # tuning parameters
    alpha_scaled <- alpha * Dmax^2
    beta_scaled <- beta * Dmax^2
    nu <- 1 / (Dmax^2 + beta_scaled)
    kappa <- nu * alpha_scaled
    
    obj <- numeric(0)
    improvement <- Inf
    
    # reducing computations in the loop
    VD2_Vt = VD2 %*% t(svd_theta$v)
    Minv <- solve(M)
    JMinv <- J %*% Minv
    MJt <- M %*% t(J)
    
    for (iter in 1:max_iter) {
      Z <- VD2_Vt %*% B
      svd_Z <- svd(Z)
      A <- svd_Z$u %*% t(svd_Z$v)
      
      # gradient update
      grad <- (VD2_Vt %*% (A - B)) - beta_scaled * B
      B_temp <- JMinv %*% B + nu * JMinv %*% grad
      
      # L1 soft thresholding
      Bf <- ifelse(B_temp > kappa, B_temp - kappa,
                   ifelse(B_temp < -kappa, B_temp + kappa, 0))
      
      # returning it to the Y space from the feature space
      B <- MJt %*% Bf
      
      R = t(VD) - t(VD) %*% B %*% t(A)
      obj_value = 0.5 * sum(R^2) + alpha_scaled * sum(abs(B)) + 0.5 * beta_scaled * sum(B^2)
      obj <- c(obj, obj_value)
      
      # check convergence
      if (iter > 1) {
        improvement <- (obj[iter-1] - obj[iter]) / obj[iter]
        if (improvement < tol) break
      }
    }
    # l2 norm for normalization
    l2_norms <- apply(Bf, 2, l2_norm_vec)
    
    # loadings normalization
    loadings_ = sweep(Bf, 2, l2_norms, FUN = "/")
    
    return(loadings_)
  }
  
  #J variable projection
  #TODO: WRITE THE J_variable_projection FUNCTION
  
  # execute and return results
  result <- sparse_fit()
  class(result) <- "sparse_gcPCA"
  return(result)
}

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
    Ra_transform <- lapply(seq_along(object$Ra_scores), function(i) {
        scores <- object$Ra_scores[[i]]  # n x k_i
        vals   <- object$Ra_values[[i]]  # length k_i
        sweep(scores, 2, vals, "*")      # n x k_i, raw transform
    })
    
    Rb_transform <- lapply(seq_along(object$Rb_scores), function(i) {
        scores <- object$Rb_scores[[i]]  # n x k_i
        vals   <- object$Rb_values[[i]]  # length k_i
        sweep(scores, 2, vals, "*")      # n x k_i, raw transform
      })

    return(list(
      Ra_transform = Ra_transform,
      Rb_transform = Rb_transform
    ))
  }

  # Initialize
  Ra_transform <- NULL
  Rb_transform <- NULL

  # ancillary function to project data onto a list of sparse loading matrices
  project_sparse <- function(X, loadings_list) {
    X <- as.matrix(X)

    transform_list <- vector("list", length(loadings_list))
    # values_list <- vector("list", length(loadings_list))

    for (i in seq_along(loadings_list)) {
      L <- loadings_list[[i]]              # p x k_i
      temp_transform <- X %*% L                      # n x k_i

      transform_list[[i]] <- temp_transform
      # Norm-1 scores (same as in sparse_gcPCA fitting)
      # scores_list[[i]] <- sweep(temp, 2, norms, "/")
      # values_list[[i]] <- norms
    }

    list(transform = transform_list)
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
    Ra_transform <- proj_Ra$transform
  }

  # Process Rb if provided
  if (!is.null(Rb)) {
    Rb_proc <- as.matrix(Rb)

    # if (normalize_flag && !is.null(normalize_fun)) {
    #   Rb_proc <- normalize_fun(Rb_proc)
    # }

    proj_Rb <- project_sparse(Rb_proc, sparse_loadings)
    Rb_transform <- proj_Rb$transform
  }

  return(list(
    Ra_transform = Ra_transform,
    Rb_transform = Rb_transform
  ))
}
