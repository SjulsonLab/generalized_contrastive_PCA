# -*- coding: utf-8 -*-

"""
Created on Thu May  4 17:52:12 2023

Set of classes to do different contrastive methods in foreground (Ra) and
background (Rb) that you want to compare, it's implemented here:
    v1 : contrastive PCA (Ra - alpha*Rb),
    v2 : ratio contrastive PCA (Ra/Rb), 
    v3 : normalized contrastive PCA ((Ra-Rb)/Rb),
    v4 : index normalized contrastive PCA ((Ra-Rb)/(Ra+Rb)).

The original versions do not return orthogonal dimensions, for that you have to use 
v2.1, v3.1 and v4.1 for each method respectively. Be aware this method is much slower

The algorthim assumes you put samples in the rows and features in the columns,
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

# %% class for generalized contrastive PCA
class gcPCA():

    """TO DO
    [ ]add a method for doing fit for multiple alphas and returning multiple models """

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
        from scipy import stats

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
        """ Inspect the input data for multiple criterias, as number of
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
        
        # inspecting whether the inputs are normalized and dimensionality to use
        self.inspect_inputs()
        J = self.Jorig.copy()  # J shrinks every round, but Jorig is the first-round's J
        
        # in orthogonal gcPCA, we iterate multiple times
        if sum(np.char.equal(self.method, ['v2.1', 'v3.1', 'v4.1'])):
            n_iter = self.N_gcPCs
        else:
            n_iter = 1

        # solving gcPCA
        if self.method == 'v1':  # original cPCA
            alpha = self.alpha
            JRaRaJ = LA.multi_dot((J.T, self.Ra.T, self.Ra, J))
            JRbRbJ = LA.multi_dot((J.T, self.Rb.T, self.Rb, J))
            sigma = JRaRaJ - alpha*JRbRbJ

            # getting eigenvalues and eigenvectors
            w, v = LA.eigh(sigma)
            eig_idx = np.argsort(w)[::-1]
            
            x = J.dot(v[:, eig_idx])
            s_total = w[eig_idx]
            obj_info = 'Ra - alpha * Rb'
        else:

            denom_well_conditioned = False
            for idx in np.arange(n_iter):
                # define numerator and denominator according to the method requested
                JRaRaJ = LA.multi_dot((J.T, self.Ra.T, self.Ra, J))
                JRbRbJ = LA.multi_dot((J.T, self.Rb.T, self.Rb, J))
                
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
                        alpha = w[0]/self.cond_number - w[-1]
                        denominator = denominator + np.eye(denominator.shape[0])*alpha
                        denom_well_conditioned = True
                    else:
                        denom_well_conditioned = True
                
                # Solving gcPCA
                d, e = LA.eigh(denominator)
                M = e * np.sqrt(d) @ e.T  # getting square root matrix M
                
                sigma = LA.multi_dot((LA.inv(M).T, numerator, LA.inv(M)))
                # Getting eigenvectors
                w, v = LA.eigh(sigma)
                eig_idx = np.argsort(w)[::-1]
                v = v[:, eig_idx]
                
                x_temp = LA.multi_dot((J, LA.inv(M), v))
                x_temp = np.divide(x_temp, LA.norm(x_temp, axis=0))

                # Copy results to X
                if idx == 0:
                    x = x_temp
                    x_orth = np.expand_dims(x_temp[:, 0], axis=1)
                else:
                    x_add = np.expand_dims(x_temp[:, 0], axis=1)
                    x_orth = np.hstack((x_orth, x_add))
                    
                # shrinking J (find an orthonormal basis for the subspace of J orthogonal
                # to the X vectors we have already collected)
                j, _, _ = LA.svd(self.Jorig - LA.multi_dot((x_orth, x_orth.T, self.Jorig)), full_matrices=False)
                J = j[:, :n_iter-(idx+1)]
                
            # getting orthogonal loadings if it was method requested
            if sum(np.char.equal(self.method, ['v2.1', 'v3.1', 'v4.1'])):
                x = x_orth

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
        temp = np.dot(Ra, x);
        self.Ra_scores_ = np.divide(temp,LA.norm(temp,axis=0))
        self.Ra_values_ = LA.norm(temp,axis=0)
        temp = np.dot(Rb, x);
        self.Rb_scores_ = np.divide(temp,LA.norm(temp,axis=0))
        self.Rb_values_ = LA.norm(temp,axis=0)
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
            null_gcpca_values.append(copy_obj.gcPCA_values_)
        self.null_gcPCA_values_ = np.vstack(null_gcpca_values)

    def transform(self, Ra, Rb):
        try:
            x = self.loadings_
            Ra_transf = np.dot(Ra, x)
            Rb_transf = np.dot(Rb, x)
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
                 lambdas = np.exp(np.linspace(np.log(1e-2), np.log(1), num=10)),
                 alpha=1,
                 alpha_null=0.975,
                 tol=1e-8,
                 max_steps=1000,
                 cond_number=10 ** 13):

        self.method = method
        self.Nshuffle = Nshuffle
        self.normalize_flag = normalize_flag
        self.alpha = alpha
        self.alpha_null = alpha_null
        self.lambdas = lambdas
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
            ra_transf = np.dot(Ra, x)
            rb_transf = np.dot(Rb, x)
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
        RaRa = self.Ra.T.dot(self.Ra)
        RbRb = self.Rb.T.dot(self.Rb)

        # solving for sparse gcPCA
        if self.method == 'v1':  # original cPCA
            # fitting again
            theta = RaRa - self.alpha * RbRb

            # Getting eigenvectors
            w, v = LA.eigh(theta)
            new_w_pos, new_w_neg = w.copy(), w.copy()

            # Calculating only the requested amount by user
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

            y_gcpc = self.original_loadings_
            if n_gcpcs_pos > 0:
                theta_pos_loadings_ = self.spca_algorithm(theta_pos,
                                                          y_gcpc[:, :n_gcpcs_pos],
                                                          self.lambdas)
            else:
                theta_pos_loadings_ = []

            if n_gcpcs_neg > 0:
                theta_neg_loadings_ = self.spca_algorithm(theta_neg,
                                                          y_gcpc[:, n_gcpcs_pos:n_gcpcs_pos + n_gcpcs_neg],
                                                          self.lambdas)
            else:
                theta_neg_loadings_ = []

            # Rearranging the PCs by vectors
            final_loadings = []
            for a in np.arange(self.lambdas.shape[0]):
                if n_gcpcs_pos > 0 and n_gcpcs_neg > 0:
                    sigma_pos_loadings_ = theta_pos_loadings_[a]
                    sigma_neg_loadings_ = theta_neg_loadings_[a]
                    final_loadings.append(np.concatenate((sigma_pos_loadings_, sigma_neg_loadings_), axis=1))
                elif n_gcpcs_pos == 0 and n_gcpcs_neg > 0:
                    sigma_neg_loadings_ = theta_neg_loadings_[a]
                    final_loadings.append(sigma_neg_loadings_)
                else:
                    sigma_pos_loadings_ = theta_pos_loadings_[a]
                    final_loadings.append(sigma_pos_loadings_)
        else:
            denom_well_conditioned = False
            #  Define numerator and denominator according to the method requested
            if sum(np.char.equal(self.method, ['v2', 'v2.1'])):
                numerator = RaRa
                denominator = RbRb
                obj_info = 'Ra / Rb'
            elif sum(np.char.equal(self.method, ['v3', 'v3.1'])):
                numerator = RaRa - RbRb
                denominator = RbRb
                obj_info = '(Ra-Rb) / Rb'
            elif sum(np.char.equal(self.method, ['v4', 'v4.1'])):
                numerator = RaRa - RbRb
                denominator = RaRa + RbRb
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

            # getting the square root matrix of denominator
            d, e = LA.eigh(denominator)
            M = e * np.sqrt(d) @ e.T
            
            y_gcpc = M@self.original_loadings_
            theta = LA.multi_dot((LA.inv(M).T, numerator, LA.inv(M)))

            ###
            # Getting eigenvectors
            w, v = LA.eigh(theta)
            new_w_pos, new_w_neg = w.copy(), w.copy()

            # Calculating only the requested amount by user
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
            theta_pos = v * (np.sqrt(new_w_pos)+alpha_pos) @ v.T

            alpha_neg = new_w_neg.max() / self.cond_number  # fixing it to be positive definite
            theta_neg = v * (np.sqrt(new_w_neg)+alpha_neg) @ v.T

            if n_gcpcs_pos > 0:
                theta_pos_loadings_ = self.spca_algorithm(theta_pos,
                                                     y_gcpc[:, :n_gcpcs_pos],
                                                     self.lambdas)
            else:
                theta_pos_loadings_ = []

            if n_gcpcs_neg > 0:
                theta_neg_loadings_ = self.spca_algorithm(theta_neg,
                                                     y_gcpc[:, n_gcpcs_pos:n_gcpcs_pos+n_gcpcs_neg],
                                                     self.lambdas)
            else:
                theta_neg_loadings_ = []

            # Rearranging the PCs by vectors
            final_loadings = []
            for a in np.arange(self.lambdas.shape[0]):
                if n_gcpcs_pos > 0 and n_gcpcs_neg > 0:
                    sigma_pos_loadings_ = LA.inv(M) @ theta_pos_loadings_[a]
                    sigma_neg_loadings_ = LA.inv(M) @ theta_neg_loadings_[a]
                    final_loadings.append(np.concatenate((sigma_pos_loadings_, sigma_neg_loadings_), axis=1))
                elif n_gcpcs_pos == 0 and n_gcpcs_neg > 0:
                    sigma_neg_loadings_ = LA.inv(M) @ theta_neg_loadings_[a]
                    final_loadings.append(sigma_neg_loadings_)
                else:
                    sigma_pos_loadings_ = LA.inv(M) @ theta_pos_loadings_[a]
                    final_loadings.append(sigma_pos_loadings_)
        self.sparse_loadings_ = final_loadings
        
        temp_ra_scores = []
        temp_ra_values = []
        for sload in final_loadings:
            temp = np.dot(self.Ra,sload)
            temp_ra_scores.append(np.divide(temp, LA.norm(temp,axis=0)))
            temp_ra_values.append(LA.norm(temp, axis=0))

        self.Ra_scores_ = temp_ra_scores
        self.Ra_values_ = temp_ra_values

        temp_rb_scores = []
        temp_rb_values = []
        for sload in final_loadings:
            temp = np.dot(self.Rb,sload)
            temp_rb_scores.append(np.divide(temp, LA.norm(temp, axis=0)))
            temp_rb_values.append(LA.norm(temp, axis=0))

        self.Rb_scores_ = temp_rb_scores
        self.Rb_values_ = temp_rb_values


    def spca_algorithm(self, theta_input, V, lambda_array):
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LassoLars
        """
        Algorithm that finds the sparse PCs, as described in Zou, Hastie and Tibshirani, 2006
        :param lambda_array: array with values of lambda to fit the model
        :param v: arrays of initial PCs or gcPCs, size n x PCs, colum-wise PCs
        :param n_pcs: number of n_pcs to return
        :return: fitted sparse PCs for each lambda
        """

        betas_by_lambda = []
        # version sparse PCA
        for lmbd in lambda_array:  # looping through lambdas vector
            if lmbd<1e-3:  # this number is empirical
                warnings.warn('Small lambda might return incorrect loadings')

            # print a message of which lambda is being fit
            print('fitting sparse gcPCA lambda=' + str(lmbd))
            Y = V.copy()
            step, diff_criterion, criterion_past = 0, 1, 1e6
            while (diff_criterion > self.tol and step < self.max_steps):
                # ####
                # Elastic net (lasso) step
                beta = []
                for a in np.arange(V.shape[1]):  # Looping through PCs
                    y = theta_input @ Y[:, a]

                    # Solving L1 constrain with the least angle regression
                    lasso_mdl = LassoLars(alpha=lmbd,
                                          fit_intercept=False)
                    lasso_mdl.fit(theta_input, y)
                    beta.append(lasso_mdl.coef_)
                betas = np.vstack(beta).T
                # ###
                # Reduced Rank Procrustes rotation
                u, _, v = LA.svd((theta_input.dot(theta_input)) @ betas, full_matrices=False)
                Y = u@v

                # Checking convergence of criterion
                criterion = np.sum(LA.norm(theta_input - (Y @ betas.T @ theta_input), axis=0)) + np.sum(lmbd * LA.norm(betas, ord=1, axis=0))
                diff_criterion = np.abs(criterion_past - criterion)
                criterion_past = criterion.copy()  # update criterion
                step += 1

            if step >= self.max_steps:
                warnings.warn('sparse gcPCA did not converge to tol., returning last iteration gcPCs')
            temp_betas = np.divide(betas, LA.norm(betas, axis=0))
            temp_betas[:,LA.norm(betas, axis=0)==0] = 0  # when norm is zero, returning nan
            betas_by_lambda.append(temp_betas)
        return betas_by_lambda  # Returning sparse PCs for each lambda
