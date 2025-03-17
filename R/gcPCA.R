# gcPCA.R

# Implementation of gcPCA in R, it is not yet guaranteed that this implementation is correct and works, still need to run some tests
# The Generalized Contrastive PCA (gcPCA) is a method to find the most contrastive dimensions between data collected under different conditions
# 
gcPCA <- function(Ra, Rb, method = 'v4', Ncalc = Inf, Nshuffle = 0, normalize_flag = TRUE, alpha = 1, alpha_null = 0.975, cond_number = 10^13) {
  
  # initializing variables
  null_gcpca_values <- NULL
  # function to normalize the data to zscore and norm
  normalize <- function(data) {
    data <- scale(data)
    data <- data / norm(data, type = "2")
    return(data)
  }

  # inspect the data size
  inspect_inputs <- function(Ra, Rb) {
    if (ncol(Ra) != ncol(Rb)) {
      stop("Ra and Rb have different numbers of features")
    }

    # normalize the data
    if (normalize_flag) {
      Ra <- normalize(Ra)
      Rb <- normalize(Rb)
    }

    # discard dimensions if nececssary, whichever is smaller sets the number of gcPCs
    n_gcpcs <- min(nrow(Ra), nrow(Rb))

    # SVD of the combined data
    RaRb <- rbind(Ra, Rb)
    svd_result <- svd(RaRb)
    Sab <- svd_result$d
    tol <- max(dim(RaRb)) * .Machine$double.eps * max(Sab)

    # check if the data is rank-deficient and select appropriate number of dimensions
    if (sum(Sab > tol) < n_gcpcs) {
      warning('Input data is rank-deficient! Discarding dimensions.')
      n_gcpcs <- sum(Sab > tol)
    }

    # set the number of gcPCs to return for different methods
    if (method %in% c('v1', 'v2', 'v3', 'v4') && !is.infinite(Ncalc)) {
      warning('Ncalc is only relevant if using orthogonal gcPCA. The full set of gcPCs will be returned.')
      print(paste(n_gcpcs, 'gcPCs will be returned.'))
    } else if (method %in% c('v2.1', 'v3.1', 'v4.1')) {
      n_gcpcs <- min(Ncalc, n_gcpcs)
      print(paste(n_gcpcs, 'gcPCs will be returned.'))
    }

    J <- svd_result$v
    return(list(n_gcpcs = n_gcpcs, J = J[, 1:n_gcpcs], Ra = Ra, Rb = Rb))
  }

  # Fit the gcPCA model
  fit <- function(Ra, Rb) {

    # inspecting the inputs
    inspected <- inspect_inputs(Ra, Rb)

    # unpacking data
    Ra <- inspected$Ra
    Rb <- inspected$Rb
    J <- inspected$J
    n_gcpcs <- inspected$n_gcpcs

    # covariance matrices
    RaRa <- (t(Ra) %*% Ra) / (nrow(Ra) - 1)
    RbRb <- (t(Rb) %*% Rb) / (nrow(Rb) - 1)

    # method v1 is the original contrastive PCA
    if (method == 'v1') {
      alpha <- alpha
      JRaRaJ <- t(J) %*% RaRa %*% J
      JRbRbJ <- t(J) %*% RbRb %*% J
      sigma <- JRaRaJ - alpha * JRbRbJ

      eig <- eigen(sigma)
      w <- eig$values
      v <- eig$vectors
      eig_idx <- order(w, decreasing = TRUE)
      x <- J %*% v[, eig_idx]
      s_total <- w[eig_idx]
      obj_info <- 'Ra - alpha * Rb'
    } else {
      # method v2-v4 as described in the manuscript
      denom_well_conditioned <- FALSE

      # for the ordering and keep tracking of the orthogonal gcPCA
      ortho_column_order <- c()
      count_dim <- 0
      x <- NULL
      x_orth <- NULL

      # loop over the number of gcPCs (for orthogonal gcPCA only)
      for (idx in 1:n_gcpcs) {
        # covariance matrices projected in the shared J space
        JRaRaJ <- t(J) %*% RaRa %*% J
        JRbRbJ <- t(J) %*% RbRb %*% J

        # define numerator and denominator according to the method requested
        if (method %in% c('v2', 'v2.1')) {
          numerator <- JRaRaJ
          denominator <- JRbRbJ
          obj_info <- 'Ra / Rb'
        } else if (method %in% c('v3', 'v3.1')) {
          numerator <- JRaRaJ - JRbRbJ
          denominator <- JRbRbJ
          obj_info <- '(Ra-Rb) / Rb'
        } else if (method %in% c('v4', 'v4.1')) {
          numerator <- JRaRaJ - JRbRbJ
          denominator <- JRaRaJ + JRbRbJ
          obj_info <- '(Ra-Rb) / (Ra+Rb)'
        } else {
          stop('Version input not recognized, please pick between v1-v4')
        }

        # check if the denominator is well-conditioned
        if (!denom_well_conditioned) {
          if (kappa(denominator) > cond_number) {
            warning('Denominator is ill-conditioned, fixing it.')
            w <- eigen(denominator)$values
            w <- w[order(w, decreasing = TRUE)]
            alpha <- w[1] / cond_number - w[length(w)]
            denominator <- denominator + diag(nrow(denominator)) * alpha
            denom_well_conditioned <- TRUE
          } else {
            denom_well_conditioned <- TRUE
          }
        }

        # solving gcPCA
        d <- eigen(denominator)$values
        e <- eigen(denominator)$vectors
        M <- e %*% diag(sqrt(d)) %*% t(e)  # square root of the denominator
        Minv <- solve(M)  # inverse of the M matrix
        sigma <- t(Minv) %*% numerator %*% Minv

        # getting eigenvectors
        eig <- eigen(sigma)
        w <- eig$values
        v <- eig$vectors
        eig_idx <- order(w, decreasing = TRUE)
        v <- v[, eig_idx]
        x_temp <- J %*% Minv %*% v
        x_temp <- x_temp / norm(x_temp, type = "2")

        #copying results to X and X_orth
        if (idx == 1) {
          x <- x_temp
          x_orth <- matrix(x_temp[, 1], ncol = 1)
          ortho_column_order <- c(ortho_column_order, count_dim)
          count_dim <- count_dim + 1
        } else {
          if (idx %% 2 == 1) {
            x_add <- matrix(x_temp[, ncol(x_temp)], ncol = 1)
            ortho_column_order <- c(ortho_column_order, ncol(x_temp) + count_dim - 1)
          } else {
            x_add <- matrix(x_temp[, 1], ncol = 1)
            ortho_column_order <- c(ortho_column_order, count_dim)
            count_dim <- count_dim + 1
          }
          x_orth <- cbind(x_orth, x_add)
        }

        # shrinking J (find an orthonormal basis for the subspace of J orthogonal to x_orth)
        j <- svd(J - x_orth %*% t(x_orth) %*% J)$u
        J <- j[, 1:(n_gcpcs - idx)]
      }

      # getting the orthogonal gcPCA loadings if it was requested
      if (method %in% c('v2.1', 'v3.1', 'v4.1')) {
        new_column_order <- order(ortho_column_order)
        x <- x_orth[, new_column_order]
      }

      # getting the eingevalues of gcPCA
      RaX <- Ra %*% x
      RbX <- Rb %*% x
      XRaRaX <- t(RaX) %*% RaX
      XRbRbX <- t(RbX) %*% RbX

      if (method %in% c('v2', 'v2.1')) {
        numerator_orig <- XRaRaX
        denominator_orig <- XRbRbX
      } else if (method %in% c('v3', 'v3.1')) {
        numerator_orig <- XRaRaX - XRbRbX
        denominator_orig <- XRbRbX
      } else if (method %in% c('v4', 'v4.1')) {
        numerator_orig <- XRaRaX - XRbRbX
        denominator_orig <- XRaRaX + XRbRbX
      }

      s_total <- diag(numerator_orig) / diag(denominator_orig)
    }

    # preparing results to send back
    loadings_ <- x
    Ra_scores_ <- Ra %*% x / norm(Ra %*% x, type = "2")
    Ra_values_ <- norm(Ra %*% x, type = "2")
    Rb_scores_ <- Rb %*% x / norm(Rb %*% x, type = "2")
    Rb_values_ <- norm(Rb %*% x, type = "2")
    objective_function_ <- obj_info
    objective_values_ <- s_total

    if (Nshuffle > 0) {
      null_gcpca_values <- null_distribution(Ra, Rb)
    }

    return(list(loadings = loadings_, Ra_scores = Ra_scores_, Ra_values = Ra_values_,
                Rb_scores = Rb_scores_, Rb_values = Rb_values_, objective_function = objective_function_,
                objective_values = objective_values_, null_gcpca_values = null_gcpca_values))
  }

  # Null distribution for shuffling, not completely tested yet
  null_distribution <- function(Ra, Rb) {
    null_gcpca_values <- c()
    for (ns in 1:Nshuffle) {
      na <- nrow(Ra)
      nb <- nrow(Rb)
      p <- ncol(Rb)
      Ra_shuffled <- Ra
      Rb_shuffled <- Rb

      for (b in 1:p) {
        Ra_shuffled[, b] <- Ra[sample(na), b]
        Rb_shuffled[, b] <- Rb[sample(nb), b]
      }

      fit_result <- fit(Ra_shuffled, Rb_shuffled)
      null_gcpca_values <- rbind(null_gcpca_values, fit_result$objective_values)
    }
    return(null_gcpca_values)
  }

  return(fit(Ra, Rb))
}