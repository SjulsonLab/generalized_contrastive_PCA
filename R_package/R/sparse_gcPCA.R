validate_sparse_gcpca_inputs <- function(Ra, Rb, method, lasso_penalty) {
  validate_gcpca_inputs(Ra, Rb, method)
  if (!all(lasso_penalty >= 0)) {
    stop("`lasso_penalty` values must be >= 0.", call. = FALSE)
  }
  invisible(TRUE)
}

#' Fit a sparse generalized contrastive PCA model
#'
#' `sparse_gcPCA()` fits sparse gcPCA loadings over a path of lasso penalties.
#'
#' @param Ra Matrix for condition A (`n_a x p`).
#' @param Rb Matrix for condition B (`n_b x p`).
#' @param method Character scalar selecting optimization variant.
#' @param Ncalc Optional number of gcPCs for orthogonal variants (`*.1`).
#' @param normalize_flag Logical; if `TRUE`, z-score and L2-normalize columns.
#' @param Nsparse Number of sparse gcPCs to estimate.
#' @param Nshuffle Number of shuffles passed to the dense gcPCA stage.
#' @param lasso_penalty Numeric vector of lasso penalties.
#' @param ridge_penalty Numeric ridge penalty.
#' @param alpha Contrastive coefficient used by method `"v1"`.
#' @param alpha_null Quantile for null thresholding (reserved in current code).
#' @param tol Convergence tolerance for sparse optimization.
#' @param max_steps Maximum number of sparse optimization iterations.
#' @param cond_number Maximum allowed denominator condition number.
#'
#' @return An object of class `c("sparse_gcPCA", "gcPCA")` containing sparse
#' loadings and training projections across penalties.
#'
#' @examples
#' set.seed(1)
#' Ra <- matrix(rnorm(50 * 6), ncol = 6)
#' Rb <- matrix(rnorm(45 * 6), ncol = 6)
#' fit <- sparse_gcPCA(Ra, Rb, method = "v4", Nsparse = 2,
#'                     lasso_penalty = c(0.05, 0.1))
#' pred <- predict(fit, Ra = Ra)
#' str(pred$Ra_scores)
#'
#' @export
sparse_gcPCA <- function(Ra, Rb, method = 'v4', Ncalc = NULL, normalize_flag = TRUE, Nsparse = NULL, Nshuffle = 0,
                         lasso_penalty = exp(seq(log(1e-2), log(1), length.out = 10)), ridge_penalty = 0,
                         alpha = 1, alpha_null = 0.975, tol = 1e-5, max_steps = 1000, cond_number = 1e13) {
  
  validate_sparse_gcpca_inputs(Ra, Rb, method, lasso_penalty)
  
  if (is.null(Ncalc)) Ncalc <- ncol(Ra)
  if (is.null(Nsparse)) Nsparse <- ncol(Ra)

  # fit base gcPCA model
  gc_model <- gcPCA(Ra, Rb, method, Ncalc, Nshuffle, normalize_flag, alpha, alpha_null, cond_number)
  
  # extract needed components
  Ra <- gc_model$Ra
  Rb <- gc_model$Rb
  Jorig <- gc_model$J

  # Legacy safeguard: some gcPCA versions returned a shrunken J (orthogonal complement
  # after the first loading) instead of the original SVD basis.
  target_gcpcs <- ncol(gc_model$loadings)
  if (is.null(Jorig) || ncol(Jorig) < target_gcpcs) {
    svd_result <- svd(rbind(Ra, Rb))
    Jorig <- svd_result$v[, 1:target_gcpcs, drop = FALSE]
    warning("Recovered original J basis from combined SVD because gc_model$J was reduced.")
  }
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
      eig <- eigen(theta, symmetric = TRUE)
      w <- eig$values
      v <- eig$vectors
      
      # Split positive/negative eigenvalues and allocate sparse dimensions
      new_w_pos <- pmax(w, 0)
      new_w_neg <- -pmin(w, 0)

      n_target <- if (is.infinite(Nsparse)) length(w) else min(as.integer(Nsparse), length(w))
      n_gcpcs_pos <- min(sum(w > 0), n_target)
      n_gcpcs_neg <- min(sum(w < 0), n_target - n_gcpcs_pos)

      theta_pos <- v %*% diag(sqrt(new_w_pos)) %*% t(v)
      theta_neg <- v %*% diag(sqrt(new_w_neg)) %*% t(v)

      final_loadings <- lapply(lasso_penalty, function(lmbda) {
        pos_load <- if (n_gcpcs_pos > 0) {
          J_variable_projection(theta_pos, Jorig, n_gcpcs_pos, lmbda, ridge_penalty, max_steps, tol)
        } else {
          matrix(numeric(0), nrow = nrow(Jorig), ncol = 0)
        }

        neg_load <- if (n_gcpcs_neg > 0) {
          J_variable_projection(theta_neg, Jorig, n_gcpcs_neg, lmbda, ridge_penalty, max_steps, tol)
        } else {
          matrix(numeric(0), nrow = nrow(Jorig), ncol = 0)
        }

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
      eig_denom <- eigen(denominator, symmetric = TRUE)
      M <- eig_denom$vectors %*% diag(sqrt(eig_denom$values)) %*% t(eig_denom$vectors)
      Minv <- solve(M)
      sigma <- t(Minv) %*% numerator %*% Minv

      eig_sigma <- eigen(sigma, symmetric = TRUE)
      w <- eig_sigma$values
      v <- eig_sigma$vectors

      # all positive eigenvalues
      new_w <- w + 2

      theta_pos <- v %*% diag(sqrt(new_w)) %*% t(v)
      
      # sparse loading
      final_loadings <- lapply(lasso_penalty, function(lmbda) {
        J_M_variable_projection(theta_pos, Jorig, M, Nsparse, lmbda, ridge_penalty, max_steps, tol)
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
  
  # J variable projection (used for method v1)
  J_variable_projection <- function(theta_input, J, k, alpha, beta, max_iter, tol) {
    svd_theta <- svd(theta_input)
    Dmax <- svd_theta$d[1]
    k_eff <- min(k, ncol(svd_theta$v))
    B <- svd_theta$v[, 1:k_eff, drop = FALSE]

    VD <- sweep(svd_theta$v, 2, svd_theta$d, "*")
    VD2 <- sweep(svd_theta$v, 2, svd_theta$d^2, "*")

    # tuning parameters
    alpha_scaled <- alpha * Dmax^2
    beta_scaled <- beta * Dmax^2
    nu <- 1 / (Dmax^2 + beta_scaled)
    kappa <- nu * alpha_scaled

    obj <- numeric(0)
    improvement <- Inf
    VD2_Vt <- VD2 %*% t(svd_theta$v)
    Bf <- B

    for (iter in 1:max_iter) {
      # update A
      Z <- VD2_Vt %*% B
      svd_Z <- svd(Z)
      A <- svd_Z$u %*% t(svd_Z$v)

      # gradient update in Y-space and map to feature-space through J
      grad <- (VD2_Vt %*% (A - B)) - beta_scaled * B
      B_temp <- J %*% B + nu * J %*% grad

      # l1 soft thresholding
      Bf <- ifelse(B_temp > kappa, B_temp - kappa,
                   ifelse(B_temp < -kappa, B_temp + kappa, 0))

      # map back to Y-space
      B <- t(J) %*% Bf

      R <- t(VD) - t(VD) %*% B %*% t(A)
      obj_value <- 0.5 * sum(R^2) + alpha_scaled * sum(abs(B)) + 0.5 * beta_scaled * sum(B^2)
      obj <- c(obj, obj_value)

      if (iter > 1) {
        improvement <- (obj[iter - 1] - obj[iter]) / obj[iter]
        if (improvement < tol) break
      }
    }

    # normalize loadings per column
    l2_norms <- apply(Bf, 2, l2_norm_vec)
    l2_norms[l2_norms == 0] <- 1
    loadings_ <- sweep(Bf, 2, l2_norms, FUN = "/")
    return(loadings_)
  }
  
  # execute and return results
  result <- sparse_fit()
  result$lasso_penalty <- lasso_penalty
  result$ridge_penalty <- ridge_penalty
  result$Nsparse <- Nsparse
  result$method <- method
  class(result) <- c("sparse_gcPCA", "gcPCA")
  return(result)
}
