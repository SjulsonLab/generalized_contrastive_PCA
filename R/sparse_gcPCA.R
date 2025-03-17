# sparse_gcPCA.R

# Implementation of sparse gcPCA in R, it is not yet guaranteed that this implementation is correct and works, still need to run some tests
sparse_gcPCA <- function(Ra, Rb, method = 'v4', Ncalc = Inf, normalize_flag = TRUE, Nsparse = Inf, Nshuffle = 0,
                         lasso_penalty = exp(seq(log(1e-2), log(1), length.out = 10)), ridge_penalty = 0,
                         alpha = 1, alpha_null = 0.975, tol = 1e-5, max_steps = 1000, cond_number = 1e13) {
  
  # fit base gcPCA model
  gc_model <- gcPCA(Ra, Rb, method, Ncalc, Nshuffle, normalize_flag, alpha, alpha_null, cond_number)
  
  # extract needed components
  Ra <- gc_model$Ra
  Rb <- gc_model$Rb
  Jorig <- gc_model$J
  p <- ncol(Ra)
  
  # sparse fitting function
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
      theta <- RaRa - alpha * RbRb
      eig <- eigen(theta)
      w <- eig$values
      v <- eig$vectors
      
      # Split positive/negative eigenvalues
      new_w_pos <- pmax(w, 0)
      new_w_neg <- -pmin(w, 0)
      
      # Calculating only the number of dimensions requested by user
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
      
      # sparse loadings
      final_loadings <- lapply(lasso_penalty, function(lmbda) {
        J_M_variable_projection(sigma, Jorig, M, Nsparse, lmbda, ridge_penalty, max_steps, tol)
      })
    }
    
    # normalize loadings
    normalized_loadings <- lapply(final_loadings, function(load) {
      sweep(load, 2, apply(load, 2, function(x) norm(x, "2")), "/")
    })
    
    return(normalized_loadings)
  }
  
  # J and M variable projection, as described in the paper
  J_M_variable_projection <- function(theta_input, J, M, k, alpha, beta, max_iter, tol) {
    svd_theta <- svd(theta_input)
    Dmax <- svd_theta$d[1]
    B <- svd_theta$v[, 1:k, drop = FALSE]
    
    # tuning parameters
    alpha_scaled <- alpha * Dmax^2
    beta_scaled <- beta * Dmax^2
    nu <- 1 / (Dmax^2 + beta_scaled)
    kappa <- nu * alpha_scaled
    
    Minv <- solve(M)
    JMinv <- J %*% Minv
    MJt <- M %*% t(J)
    
    for (iter in 1:max_iter) {
      Z <- theta_input %*% B
      svd_Z <- svd(Z)
      A <- svd_Z$u %*% t(svd_Z$v)
      
      # gradient update
      grad <- (theta_input %*% (A - B)) - beta_scaled * B
      B_temp <- JMinv %*% B + nu * JMinv %*% grad
      
      # soft thresholding
      Bf <- ifelse(B_temp > kappa, B_temp - kappa,
                   ifelse(B_temp < -kappa, B_temp + kappa, 0))
      
      B <- MJt %*% Bf
      
      # check convergence
      if (iter > 1 && abs(obj_prev - obj_value)/obj_value < tol) break
      obj_prev <- sum((theta_input - B %*% t(A))^2)/2 + alpha_scaled*sum(abs(B)) + beta_scaled*sum(B^2)/2
    }
    
    return(Bf)
  }
  
  # execute and return results
  sparse_loadings <- sparse_fit()
  
  return(list(
    sparse_loadings = sparse_loadings,
    Ra = Ra,
    Rb = Rb,
    original_loadings = gc_model$loadings,
    objective_function = gc_model$objective_function
  ))
}