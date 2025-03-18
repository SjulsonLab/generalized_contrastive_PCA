# -*- coding: utf-8 -*-

"""
Set of classes to do different contrastive methods in foreground (Ra) and
background (Rb) that you want to compare, it's implemented here:
    v1 : contrastive PCA (Ra - alpha*Rb),
    v2 : ratio contrastive PCA (Ra/Rb), 
    v3 : normalized contrastive PCA ((Ra-Rb)/Rb),
    v4 : index normalized contrastive PCA ((Ra-Rb)/(Ra+Rb)).

The original versions do not return orthogonal dimensions, for that you have to use 
v2.1, v3.1 and v4.1 for each method respectively. Be aware this method is much slower

The algorithm assumes the samples are in the rows and features in the columns,
as in n x p where n is the sample number and p is the feature number

The method fit returns the following fields:
loadings_ : loadings of the gcPCs
gcPCA_values_ : similar to eigenvalue, what is the gcPCA value according to the method you picked.
Ra_scores_ : Ra projected into the gcPCA vectors (loadings_)
Rb_scores_ : Rb projected into the gcPCA vectors (loadings_)
objetive_function_ : Objective function based on the method you picked.
@author: Eliezyer de Oliveira
"""

import warnings
import numpy as np
import numpy.linalg as LA
import time
from numba import njit
from scipy import stats

# optimized functions to speed up computation

# %% class for generalized contrastive PCA
class gcPCA():

    def __init__(self, method='v4',
                 Ncalc=np.inf,
                 Nshuffle=0,
                 normalize_flag=True,
                 alpha=1,
                 alpha_null=0.975,
                 cond_number=10**13):

        self.method = method
        self.Nshuffle = Nshuffle
        self.normalize_flag = normalize_flag
        self.alpha = alpha
        self.alpha_null = alpha_null
        self.cond_number = cond_number
        self.Ncalc = Ncalc
        self.Ra = None
        self.Rb = None

    def normalize(self):
        """ Normalize the data to zscore and norm. """

        Ra_temp = np.divide(stats.zscore(self.Ra),
                            LA.norm(stats.zscore(self.Ra), axis=0))
        Rb_temp = np.divide(stats.zscore(self.Rb),
                            LA.norm(stats.zscore(self.Rb), axis=0))

        if np.sum(np.sum(np.square(Ra_temp - self.Ra)) >
                  (0.01*np.square(Ra_temp))):
            warnings.warn("Ra was not normalized properly - normalizing now")
            self.Ra = Ra_temp

        if np.sum(np.sum(np.square(Rb_temp - self.Rb)) >
                  (0.01*np.square(Rb_temp))):
            warnings.warn("Rb was not normalized properly - normalizing now")
            self.Rb = Rb_temp

    def inspect_inputs(self):
        """ Inspect the input data for multiple criteria, as number of
        features, normalization and number of gcPCs possible to get vs
        number of gcPCs requested by the user."""

        # Test that inputs have the same number of features
        if self.Ra.shape[1] != self.Rb.shape[1]:
            raise ValueError("Ra and Rb have different numbers of features")
        
        # Normalization step
        if self.normalize_flag:
            self.normalize()

        # discard dimensions if necessary
        # whichever dataset has less datapoints dictate the amount of gcPCs
        n_gcpcs = int(np.min((self.Ra.shape, self.Rb.shape)))
        
        # SVD on the combined dataset to discard dimensions with near-zero variance
        RaRb = np.concatenate((self.Ra, self.Rb), axis=0)
        _, Sab, v = LA.svd(RaRb, full_matrices=False)
        tol = max(RaRb.shape) * np.finfo(Sab.max()).eps
        
        if sum(Sab > tol) < n_gcpcs:
            warnings.warn('Input data is rank-deficient! Discarding dimensions; cannot shuffle.')
            n_gcpcs = sum(Sab > tol)  # the number of gcPCs we can return
        
        # Set number of gcPCs to return
        if sum(np.char.equal(self.method, ['v1', 'v2', 'v3', 'v4'])) & ~np.isinf(self.Ncalc):
            warnings.warn('Ncalc is only relevant if using orthogonal gcPCA. Will calculate full set of gcPCs.')
            print(str(n_gcpcs) + ' gcPCs will be returned.')
        elif sum(np.char.equal(self.method, ['v2.1', 'v3.1', 'v4.1'])):
            n_gcpcs = int(np.min((self.Ncalc, n_gcpcs)))  # either the requested by user or maximum rank there can be
            print(str(n_gcpcs) + ' gcPCs will be returned.')
        
        J = v.T.copy()
        self.N_gcPCs = n_gcpcs
        self.Jorig = J[:, :n_gcpcs]


    def fit(self, Ra, Rb):

        # Assigning data to class
        self.Ra = Ra
        self.Rb = Rb
        
        # inspecting whether the inputs are normalized and the dimensionality to use
        self.inspect_inputs()
        J = self.Jorig.copy()  # J shrinks every round, but Jorig is the first-round's J

        # covariance matrices
        RaRa = (self.Ra.T @ self.Ra) / (self.Ra.shape[0] - 1)
        RbRb = (self.Rb.T @ self.Rb) / (self.Rb.shape[0] - 1)

        start_time = time.time()
        # in orthogonal gcPCA, we iterate multiple times
        if sum(np.char.equal(self.method, ['v2.1', 'v3.1', 'v4.1'])):
            n_iter = self.N_gcPCs
        else:
            n_iter = 1

        # solving gcPCA
        if self.method == 'v1':  # original cPCA
            alpha = self.alpha
            JRaRaJ = LA.multi_dot((J.T, RaRa, J))
            JRbRbJ = LA.multi_dot((J.T, RbRb, J))
            sigma = JRaRaJ - alpha*JRbRbJ

            # getting eigenvalues and eigenvectors
            w, v = LA.eigh(sigma)
            eig_idx = np.argsort(w)[::-1]
            
            x = J@v[:, eig_idx]
            s_total = w[eig_idx]
            obj_info = 'Ra - alpha * Rb'
        else:

            denom_well_conditioned = False
            # for the ordering and keep tracking of the orthogonal gcPCA columns
            ortho_column_order = []
            count_dim = 0
            for idx in np.arange(n_iter):
                JRaRaJ = LA.multi_dot((J.T, RaRa, J))
                JRbRbJ = LA.multi_dot((J.T, RbRb, J))

                # define numerator and denominator according to the method requested
                if sum(np.char.equal(self.method, ['v2', 'v2.1'])):
                    numerator = JRaRaJ
                    denominator = JRbRbJ
                    obj_info = 'Ra / Rb'
                elif sum(np.char.equal(self.method, ['v3', 'v3.1'])):
                    numerator = JRaRaJ - JRbRbJ
                    denominator = JRbRbJ
                    obj_info = '(Ra-Rb) / Rb'
                elif sum(np.char.equal(self.method, ['v4', 'v4.1'])):
                    numerator = JRaRaJ - JRbRbJ
                    denominator = JRaRaJ + JRbRbJ
                    obj_info = '(Ra-Rb) / (Ra+Rb)'
                else:
                    raise ValueError('Version input not recognized, please pick between v1-v4')

                if not denom_well_conditioned:
                    if LA.cond(denominator) > self.cond_number:
                        warnings.warn('Denominator is ill-conditioned, fixing it. ' +
                                      'Be aware that gcPCA values will be' +
                                      'slightly smaller')

                        w = LA.eigvalsh(denominator)
                        w = w[np.argsort(w)[::-1]]
                        alpha = w[0] / self.cond_number - w[-1]
                        denominator = denominator + np.eye(denominator.shape[0]) * alpha
                        denom_well_conditioned = True
                    else:
                        denom_well_conditioned = True
                
                # Solving gcPCA
                d, e = LA.eigh(denominator)
                M = e * np.sqrt(d) @ e.T  # getting square root matrix M
                Minv = LA.inv(M)  # inverse of the M matrix
                
                sigma = LA.multi_dot((Minv.T, numerator, Minv))
                # Getting eigenvectors
                w, v = LA.eigh(sigma)
                eig_idx = np.argsort(w)[::-1]
                v = v[:, eig_idx]
                
                x_temp = LA.multi_dot((J, Minv, v))
                x_temp = np.divide(x_temp, LA.norm(x_temp, axis=0))

                # Copy results to X
                if idx == 0:
                    x = x_temp
                    x_orth = np.expand_dims(x_temp[:, 0], axis=1)
                    ortho_column_order.append(count_dim)
                    count_dim += 1
                else:
                    # alternating between the first and last column of x_temp
                    if (idx % 2) == 1:
                        x_add = np.expand_dims(x_temp[:, -1], axis=1)
                        ortho_column_order.append(x_temp.shape[1]+count_dim-1)
                    elif (idx % 2) == 0:
                        x_add = np.expand_dims(x_temp[:, 0], axis=1)
                        ortho_column_order.append(count_dim)
                        count_dim += 1
                    x_orth = np.hstack((x_orth, x_add))
                    
                # shrinking J (find an orthonormal basis for the subspace of J orthogonal
                # to the X vectors we have already collected)
                j, _, _ = LA.svd(self.Jorig - LA.multi_dot((x_orth, x_orth.T, self.Jorig)), full_matrices=False)
                J = j[:, :n_iter-(idx+1)]

            self.elapsed_time_ = time.time() - start_time
            # getting orthogonal loadings if it was method requested
            if sum(np.char.equal(self.method, ['v2.1', 'v3.1', 'v4.1'])):
                new_column_order = np.argsort(np.array(ortho_column_order))
                x = x_orth[:, new_column_order]  # rearranging the columns to the original order

            # getting the eigenvalue of gcPCA
            RaX = Ra@x
            RbX = Rb@x
            XRaRaX = RaX.T@RaX
            XRbRbX = RbX.T@RbX
            if sum(np.char.equal(self.method, ['v2', 'v2.1'])):
                numerator_orig = XRaRaX
                denominator_orig = XRbRbX
            elif sum(np.char.equal(self.method, ['v3', 'v3.1'])):
                numerator_orig = XRaRaX - XRbRbX
                denominator_orig = XRbRbX
            elif sum(np.char.equal(self.method, ['v4', 'v4.1'])):
                numerator_orig = XRaRaX - XRbRbX
                denominator_orig = XRaRaX + XRbRbX
            s_total = np.divide(np.diagonal(numerator_orig), np.diagonal(denominator_orig))
                
        self.loadings_ = x
        temp = np.dot(Ra, x)
        self.Ra_scores_ = np.divide(temp, LA.norm(temp, axis=0))
        self.Ra_values_ = LA.norm(temp, axis=0)
        temp = np.dot(Rb, x)
        self.Rb_scores_ = np.divide(temp, LA.norm(temp, axis=0))
        self.Rb_values_ = LA.norm(temp, axis=0)
        self.objective_function_ = obj_info
        self.objective_values_ = s_total
        
                
        # Shuffling to define a null distribution
        if self.Nshuffle > 0:
            self.null_distribution()


    def null_distribution(self):
        import copy
        """The current null method is by shuffling the samples within a
        feature of both B and A dataset"""
        
        # variable for the null
        null_gcpca_values = []
        # copying the object to do null
        copy_obj = copy.deepcopy(self)
        copy_obj.Nshuffle = 0  # Removing the shuffle so it doesnt stuck
        # loop of multiple different nshuffle
        for ns in np.arange(self.Nshuffle):
            na = self.Ra.shape[0]
            nb = self.Rb.shape[0]
            p = self.Rb.shape[1]
            Ra = self.Ra.copy()
            Rb = self.Rb.copy()
            
            for b in np.arange(p):
                Ra[:, b] = self.Ra[np.random.permutation(np.arange(na)), b]
                Rb[:, b] = self.Rb[np.random.permutation(np.arange(nb)), b]
            # running the fit model again to get the null objective
            copy_obj.fit(Ra, Rb)
            null_gcpca_values.append(copy_obj.objective_values_)
        self.null_objective_values_ = np.vstack(null_gcpca_values)

    def transform(self, Ra, Rb):
        try:
            x = self.loadings_
            Ra_transf = Ra@x
            Rb_transf = Rb@x
            self.Ra_transformed_ = Ra_transf
            self.Rb_transformed_ = Rb_transf
        except:
            print('Loadings not defined, you have to first fit the model')   
    def fit_transform(self, Ra, Rb):
        self.fit(Ra, Rb)
        self.transform(Ra, Rb)


# %% class for sparse generalized contrastive PCA
class sparse_gcPCA():

    def __init__(self, method='v4',
                 Ncalc=np.inf,
                 normalize_flag=True,
                 Nsparse=np.inf,
                 Nshuffle=0,
                 lasso_penalty=np.exp(np.linspace(np.log(1e-2), np.log(1), num=10)),
                 ridge_penalty=0,
                 alpha=1,
                 alpha_null=0.975,
                 tol=1e-5,
                 max_steps=1000,
                 cond_number=10 ** 13):

        self.method = method
        self.Nshuffle = Nshuffle
        self.normalize_flag = normalize_flag
        self.alpha = alpha
        self.alpha_null = alpha_null
        self.lasso_penalty = lasso_penalty
        self.ridge_penalty = ridge_penalty
        self.cond_number = cond_number
        self.Ncalc = Ncalc
        self.Nsparse = Nsparse
        self.tol = tol
        self.max_steps = max_steps
        self.Ra = None
        self.Rb = None

    
    def fit(self, Ra, Rb):
        gcPCA_mdl = gcPCA(method=self.method,
                          Ncalc=self.Ncalc,
                          normalize_flag=self.normalize_flag,
                          Nshuffle=self.Nshuffle,
                          alpha=self.alpha,
                          alpha_null=self.alpha_null,
                          cond_number=self.cond_number)
        gcPCA_mdl.fit(Ra, Rb)
        
        # pass results from gcPCA object to sparse gcPCA
        self.Ra = gcPCA_mdl.Ra
        self.Rb = gcPCA_mdl.Rb
        self.Jorig = gcPCA_mdl.Jorig
        self.original_loadings_ = gcPCA_mdl.loadings_
        self.original_gcPCA = gcPCA_mdl
        self.objective_function_ = gcPCA_mdl.objective_function_
        
        # solving sparse gcPCA
        self.sparse_fitting()

    def transform(self, Ra, Rb):
        try:
            x = self.sparse_loadings_
            ra_transf = Ra@x
            rb_transf = Rb@x
            self.Ra_transformed_ = ra_transf
            self.Rb_transformed_ = rb_transf
        except:
            print('Loadings not defined, you have to first fit the model')

    # ancillary method
    def sparse_fitting(self):
        """Method to find sparse loadings of gcPCA, based on Zou et al 2006
        sparse PCA method. It uses elastic net to identify the set of sparse
        loadings"""

        # covariance matrices
        RaJ = self.Ra @ self.Jorig
        RbJ = self.Rb @ self.Jorig

        JRaRaJ = (RaJ.T@RaJ) / (RaJ.shape[0] - 1)
        JRbRbJ = (RbJ.T@RbJ) / (RbJ.shape[0] - 1)

        start_time = time.time()
        # solving for sparse gcPCA
        if self.method == 'v1':  # original sparse cPCA
            # fitting again
            theta = JRaRaJ - self.alpha * JRbRbJ

            # Getting eigenvectors
            w, v = LA.eigh(theta)
            new_w_pos, new_w_neg = w.copy(), w.copy()

            # Calculating only the number of dimensions requested by user
            n_gcpcs_pos = np.sum(new_w_pos > 0)
            if (n_gcpcs_pos - self.Nsparse) >= 0:
                n_gcpcs_pos = self.Nsparse
                n_gcpcs_neg = 0
            else:
                temp = self.Nsparse - n_gcpcs_pos
                n_gcpcs_neg = temp

            # separating positive and negative eigenvalues, flipping negative
            new_w_pos[new_w_pos < 0] = 0
            new_w_neg[new_w_neg > 0] = 0
            new_w_neg[new_w_neg < 0] = new_w_neg[new_w_neg < 0] * -1  # Flipping the sign of negative eigenvalues

            # Square root matrix of sigma plus
            alpha_pos = new_w_pos.max() / self.cond_number  # fixing it to be positive definite
            theta_pos = v * np.sqrt(new_w_pos + alpha_pos) @ v.T

            alpha_neg = new_w_neg.max() / self.cond_number  # fixing it to be positive definite
            theta_neg = v * np.sqrt(new_w_neg + alpha_neg) @ v.T

            # if there is any positive eigenvalue
            if n_gcpcs_pos > 0:
                final_pos_loadings = []
                for lmbda in self.lasso_penalty:
                    feature_space_loadings = self.J_variable_projection(theta_pos, self.Jorig, k=self.Nsparse,
                                                                               alpha=lmbda, beta=self.ridge_penalty,
                                                                               max_iter=self.max_steps, tol=self.tol)
                    temp_load_norm = LA.norm(feature_space_loadings, axis=0)  # getting the norm of each dimensions
                    temp_load_norm[temp_load_norm == 0] = 1  # to not divide by 0
                    final_pos_loadings.append(
                        np.divide(feature_space_loadings, temp_load_norm))  # normalizing the dimensions and saving it

            else:
                final_pos_loadings_ = []

            # if there is any negative eigenvalue
            if n_gcpcs_neg > 0:
                final_neg_loadings = []
                for lmbda in self.lasso_penalty:
                    feature_space_loadings = self.J_variable_projection(theta_neg, self.Jorig, k=self.Nsparse,
                                                                               alpha=lmbda, beta=self.ridge_penalty,
                                                                               max_iter=self.max_steps, tol=self.tol)
                    temp_load_norm = LA.norm(feature_space_loadings, axis=0)  # getting the norm of each dimensions
                    temp_load_norm[temp_load_norm == 0] = 1  # to not divide by 0
                    final_neg_loadings.append(
                        np.divide(feature_space_loadings, temp_load_norm))  # normalizing the dimensions and saving it
            else:
                final_neg_loadings_ = []

            # Rearranging the PCs by vectors
            final_loadings = []
            for a in np.arange(self.lambdas.shape[0]):
                if n_gcpcs_pos > 0 and n_gcpcs_neg > 0:
                    sigma_pos_loadings_ = final_pos_loadings_[a]
                    sigma_neg_loadings_ = final_neg_loadings_[a]
                    final_loadings.append(np.concatenate((sigma_pos_loadings_, sigma_neg_loadings_), axis=1))
                elif n_gcpcs_pos == 0 and n_gcpcs_neg > 0:
                    sigma_neg_loadings_ = final_neg_loadings_[a]
                    final_loadings.append(sigma_neg_loadings_)
                else:
                    sigma_pos_loadings_ = final_pos_loadings_[a]
                    final_loadings.append(sigma_pos_loadings_)
        else:
            #  Define numerator and denominator according to the method requested
            if sum(np.char.equal(self.method, ['v2', 'v2.1'])):
                numerator = JRaRaJ
                denominator = JRbRbJ
                obj_info = 'Ra / Rb'
            elif sum(np.char.equal(self.method, ['v3', 'v3.1'])):
                numerator = JRaRaJ - JRbRbJ
                denominator = JRbRbJ
                obj_info = '(Ra-Rb) / Rb'
            elif sum(np.char.equal(self.method, ['v4', 'v4.1'])):
                numerator = JRaRaJ - JRbRbJ
                denominator = JRaRaJ + JRbRbJ
                obj_info = '(Ra-Rb) / (Ra+Rb)'
            else:
                raise ValueError('Version input not recognized, please pick between v1-v4')

            # getting the square root matrix of denominator
            d, e = LA.eigh(denominator)
            M = e * np.sqrt(d) @ e.T  # getting square root matrix M
            Minv = LA.inv(M)

            sigma = LA.multi_dot((Minv.T, numerator, Minv))

            ###
            # Getting square root matrix of sigma
            w, v = LA.eigh(sigma)

            ##########
            # off setting the eigenvalues to be positive definite
            new_w = w.copy() + 2  # adding 2 to make it positive definite
            theta_pos = (v * np.sqrt(new_w)) @ v.T

            final_loadings = []
            for lmbda in self.lasso_penalty:
                feature_space_loadings = J_M_variable_projection(theta_pos, J=self.Jorig, M=M, k=self.Nsparse, alpha=lmbda, beta=self.ridge_penalty,
                                                 max_iter=self.max_steps, tol=self.tol)
                temp_load_norm = LA.norm(feature_space_loadings, axis=0)  # getting the norm of each dimensions
                temp_load_norm[temp_load_norm == 0] = 1  # to not divide by 0
                final_loadings.append(np.divide(feature_space_loadings, temp_load_norm))  # normalizing the dimensions and saving it

        self.elapsed_time_ = time.time() - start_time
        self.sparse_loadings_ = final_loadings
        
        temp_ra_scores = []
        temp_ra_values = []
        for sload in final_loadings:
            temp = self.Ra@sload
            temp_norm = LA.norm(temp,axis=0)
            temp_norm[temp_norm == 0] = 1
            temp_ra_scores.append(np.divide(temp,temp_norm))
            temp_ra_values.append(LA.norm(temp, axis=0))

        self.Ra_scores_ = temp_ra_scores
        self.Ra_values_ = temp_ra_values

        temp_rb_scores = []
        temp_rb_values = []
        for sload in final_loadings:
            temp = self.Rb@sload
            temp_norm = LA.norm(temp, axis=0)
            temp_norm[temp_norm == 0] = 1
            temp_rb_scores.append(np.divide(temp,temp_norm))
            temp_rb_values.append(LA.norm(temp, axis=0))

        self.Rb_scores_ = temp_rb_scores
        self.Rb_values_ = temp_rb_values

    # using variable projection as an optimization strategy to the lasso lars, used in sPCA by benjamin erichson
    def J_variable_projection(self, theta_input, J, k=None, alpha=1e-4, beta=1e-4, max_iter=1000, tol=1e-5, verbose=True):
        # solves the sparse gcPCA problem using variable projection, used for v1 (cPCA)
        # alpha is the lasso penalty and beta is the ridge penalty

        _, S, Vt = np.linalg.svd(theta_input, full_matrices=False)
        Dmax = S[0]
        B = Vt.T[:, :k]

        VD = Vt.T * S
        VD2 = Vt.T * (S ** 2)

        # Set tuning parameters
        alpha *= Dmax ** 2
        beta *= Dmax ** 2

        nu = 1.0 / (Dmax ** 2 + beta)
        kappa = nu * alpha

        obj = []
        improvement = np.inf

        # Apply Variable Projection Solver
        VD2_Vt = VD2@Vt
        for noi in range(1, max_iter+1):
            # Update A: X'XB = UDV'
            Z = VD2_Vt@B
            U_Z, _, Vt_Z = np.linalg.svd(Z, full_matrices=False)
            A = U_Z @ Vt_Z
            ######

            grad = (VD2 @ (Vt @ (A - B))) - beta * B  # Gradient step in the original space (PCA)
            B_temp = J@B + nu * J@grad  # passing it to the feature space
            B_temp_f = B_temp

            # l1 soft_threshold
            Bf = np.zeros_like(B_temp_f)
            Bf[B_temp_f > kappa] = B_temp_f[B_temp_f > kappa] - kappa
            Bf[B_temp_f <= -kappa] = B_temp_f[B_temp_f <= -kappa] + kappa

            B = J.T@Bf  # returning it to the Y space from the feature space

            R = VD.T - np.linalg.multi_dot((VD.T, B, A.T))  # residuals
            obj_value = 0.5 * np.sum(R ** 2) + alpha * np.sum(np.abs(B)) + 0.5 * beta * np.sum(B ** 2)

            obj.append(obj_value)

            # Break if objective is not improving
            if noi > 1:
                improvement = (obj[noi - 2] - obj[noi - 1]) / obj[noi - 1]

            if improvement < tol:
                print("Improvement is smaller than the tolerance, stopping the optimization")
                break

            # Verbose output
            if verbose and (noi % 10 == 0):
                print(f"Iteration: {noi}, Objective: {obj_value:.5e}, Relative improvement: {improvement:.5e}")

        loadings_ = Bf/LA.norm(Bf, axis=0)  # returning the feature space loadings
        return loadings_

    def M_variable_projection(self, theta_input, M=None, k=None, alpha=1e-4, beta=1e-4, max_iter=1000, tol=1e-5, verbose=True):
        # solves the sparse gcPCA problem using variable projection
        # alpha is the lasso penalty and beta is the ridge penalty
        _, S, Vt = np.linalg.svd(theta_input, full_matrices=False)
        Dmax = S[0]
        B = Vt.T[:, :k]

        VD = Vt.T * S
        VD2 = Vt.T * (S ** 2)

        # Set tuning parameters
        alpha *= Dmax ** 2
        beta *= Dmax ** 2

        nu = 1.0 / (Dmax ** 2 + beta)
        kappa = nu * alpha

        obj = []
        improvement = np.inf

        # Apply Variable Projection Solver
        VD2_Vt = VD2@Vt
        Minv = LA.inv(M)
        for noi in range(1, max_iter+1):
            # Update A: X'XB = UDV'
            Z = VD2_Vt@B
            U_Z, _, Vt_Z = np.linalg.svd(Z, full_matrices=False)
            A = U_Z @ Vt_Z
            ######

            ######
            grad = (VD2_Vt @ (A - B)) - beta * B  # Gradient step in the original space
            B_temp_f = (Minv@B) + nu * Minv@grad

            # l1 soft_threshold
            Bf = np.zeros_like(B_temp_f)
            Bf[B_temp_f > kappa] = B_temp_f[B_temp_f > kappa] - kappa
            Bf[B_temp_f <= -kappa] = B_temp_f[B_temp_f <= -kappa] + kappa

            B = M @ Bf  # returning it to the Y space from the feature space
            R = VD.T - LA.multi_dot((VD.T, B, A.T))
            obj_value = 0.5 * np.sum(R ** 2) + alpha * np.sum(np.abs(B)) + 0.5 * beta * np.sum(B ** 2)

            ######

            obj.append(obj_value)

            # Break if objective is not improving
            if noi > 1:
                improvement = (obj[noi - 2] - obj[noi - 1]) / obj[noi - 1]

            if improvement < tol:
                print("Improvement is smaller than the tolerance, stopping the optimization")
                break

            # Verbose output
            if verbose and (noi % 10 == 0):
                print(f"Iteration: {noi}, Objective: {obj_value:.5e}, Relative improvement: {improvement:.5e}")

        loadings_ = Bf/LA.norm(B, axis=0)
        return loadings_

    def J_M_variable_projection_old(self, theta_input, J, M, k=None, alpha=1e-4, beta=1e-4, max_iter=1000, tol=1e-5, verbose=True):
        # solves the sparse gcPCA (v2-4) problem using variable projection in the gcPCA space and penalizing in the feature space
        # alpha is the lasso penalty and beta is the ridge penalty

        _, S, Vt = np.linalg.svd(theta_input, full_matrices=False)
        Dmax = S[0]
        B = Vt.T[:, :k]

        VD = Vt.T * S
        VD2 = Vt.T * (S ** 2)

        # Set tuning parameters
        alpha *= Dmax ** 2
        beta *= Dmax ** 2

        nu = 1.0 / (Dmax ** 2 + beta)
        kappa = nu * alpha

        obj = []
        improvement = np.inf

        # Apply Variable Projection Solver
        VD2_Vt = VD2@Vt
        Minv = LA.inv(M)
        JMinv = J@Minv
        MJt = M@J.T
        for noi in range(1, max_iter+1):
            # Update A: X'XB = UDV'
            Z = VD2_Vt@B
            U_Z, _, Vt_Z = np.linalg.svd(Z, full_matrices=False)
            A = U_Z @ Vt_Z
            ######

            ######
            # Version passing from Y space to feature space
            grad = (VD2_Vt @ (A - B)) - beta * B  # Gradient step in the original space
            B_temp = JMinv@B + nu * JMinv@grad  # passing it to the feature space
            B_temp_f = B_temp

            # l1 soft_threshold
            Bf = np.zeros_like(B_temp_f)
            Bf[B_temp_f > kappa] = B_temp_f[B_temp_f > kappa] - kappa
            Bf[B_temp_f <= -kappa] = B_temp_f[B_temp_f <= -kappa] + kappa

            B = MJt @ Bf  # returning it to the Y space from the feature space

            R = VD.T - LA.multi_dot((VD.T, B, A.T))
            obj_value = 0.5 * np.sum(R ** 2) + alpha * np.sum(np.abs(B)) + 0.5 * beta * np.sum(B ** 2)

            ######

            obj.append(obj_value)
            # Break if objective is not improving
            if noi > 1:
                improvement = (obj[noi - 2] - obj[noi - 1]) / obj[noi - 1]

            if improvement < tol:
                # print("Improvement is smaller than the tolerance, stopping the optimization")
                break

            # Verbose output
            if verbose and (noi % 10 == 0):
                print(f"Iteration: {noi}, Objective: {obj_value:.5e}, Relative improvement: {improvement:.5e}")

        loadings_ = Bf/LA.norm(Bf, axis=0)
        return loadings_

@njit 
def soft_threshold(x, kappa):
    """ apply soft thresholding to x with threshold kappa """
    if x > kappa:
        return x - kappa
    elif x < -kappa:
        return x + kappa
    else:
        return 0.0

@njit
def l2_norm(x,axis=0):
    return np.sqrt(np.sum(x**2,axis=axis))
    
@njit
def J_M_variable_projection(theta_input, J, M, k=None, alpha=1e-4, beta=1e-4, max_iter=1000, tol=1e-5, verbose=True):
    # solves the sparse gcPCA (v2-4) problem using variable projection in the gcPCA space and penalizing in the feature space
    # alpha is the lasso penalty and beta is the ridge penalty

    _, S, Vt = np.linalg.svd(theta_input, full_matrices=False)
    Dmax = S[0]
    B = np.ascontiguousarray(Vt.T[:, :k])

    VD = Vt.T * S
    VD2 = Vt.T * (S ** 2)

    # Set tuning parameters
    alpha *= Dmax ** 2
    beta *= Dmax ** 2

    nu = 1.0 / (Dmax ** 2 + beta)
    kappa = nu * alpha

    obj = []
    improvement = np.inf

    # Apply Variable Projection Solver
    VD2_Vt = VD2@Vt
    Minv = LA.inv(M)
    JMinv = J@Minv
    MJt = M@J.T
    for noi in range(1, max_iter+1):
        # Update A: X'XB = UDV'
        Z = VD2_Vt@B
        U_Z, _, Vt_Z = np.linalg.svd(Z, full_matrices=False)
        A = U_Z @ Vt_Z
        ######

        ######
        # Version passing from Y space to feature space
        grad = (VD2_Vt @ (A - B)) - beta * B  # Gradient step in the original space
        B_temp = JMinv@B + nu * JMinv@grad  # passing it to the feature space
        B_temp_f = B_temp

        # l1 soft_threshold
        Bf = np.zeros_like(B_temp_f)
        for i in range(B_temp.shape[0]):
            for j in range(B_temp.shape[1]):
                Bf[i, j] = soft_threshold(B_temp_f[i, j], kappa)
        # Bf[B_temp_f > kappa] = B_temp_f[B_temp_f > kappa] - kappa
        # Bf[B_temp_f <= -kappa] = B_temp_f[B_temp_f <= -kappa] + kappa

        B = MJt @ Bf  # returning it to the Y space from the feature space

        R = VD.T - VD.T.dot(B).dot(A.T)  #LA.multi_dot((VD.T, B, A.T))
        obj_value = 0.5 * np.sum(R ** 2) + alpha * np.sum(np.abs(B)) + 0.5 * beta * np.sum(B ** 2)

        ######

        obj.append(obj_value)
        # Break if objective is not improving
        if noi > 1:
            improvement = (obj[noi - 2] - obj[noi - 1]) / obj[noi - 1]

        if improvement < tol:
            # print("Improvement is smaller than the tolerance, stopping the optimization")
            break

        # Verbose output
        # if verbose and (noi % 10 == 0):
        #     print(f"Iteration: {noi}, Objective: {obj_value:.5e}, Relative improvement: {improvement:.5e}")

    loadings_ = Bf/l2_norm(Bf, axis=0)
    return loadings_