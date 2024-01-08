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
from scipy.linalg import sqrtm

# %% class for generalized contrastive PCA
class gcPCA():

    """TO DO
    [ ]add a method for doing fit for multiple alphas and returning multiple models """

    def __init__(self, method='v4.1',
                 Ncalc=np.inf,
                 Nshuffle=0,
                 normalize_flag=True,
                 alpha=1,
                 alpha_null=0.975,
                 tol=1e-8,
                 max_steps=1000,
                 cond_number=10**13):

        self.method = method
        self.Nshuffle = Nshuffle
        self.normalize_flag = normalize_flag
        self.alpha = alpha
        self.alpha_null = alpha_null
        self.cond_number = cond_number
        self.Ncalc = Ncalc
        self.tol = tol
        self.max_steps = max_steps
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

        # Covariance matrices
        RaRa = self.Ra.T.dot(self.Ra)
        RbRb = self.Rb.T.dot(self.Rb)

        # solving gcPCA
        if self.method == 'v1':  # original cPCA
            alpha = self.alpha
            JRaRaJ = LA.multi_dot((J.T, RaRa, J))
            JRbRbJ = LA.multi_dot((J.T, RbRb, J))
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
                JRaRaJ = LA.multi_dot((J.T, RaRa, J))
                JRbRbJ = LA.multi_dot((J.T, RbRb, J))
                
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
                M = sqrtm(denominator)
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
            # Stop = LA.multi_dot((X.T, numerator_orig, X))
            # Sbot = LA.multi_dot((X.T, denominator_orig, X))
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
            
        # solving sparse gcPCA if user requested
        # if self.sparse_gcPCA:
        #     self.sparse_fitting()
        
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

    def __init__(self, method='v4.1',
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
        gcPCA_mdl.fit(Ra,Rb)
        
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
        J = self.Jorig
        # solving for sparse gcPCA
        if self.method == 'v1':  # original cPCA
            # fitting again
            JRaRaJ = LA.multi_dot((J.T, RaRa, J))
            JRbRbJ = LA.multi_dot((J.T, RbRb, J))
            sigma = JRaRaJ - self.alpha * JRbRbJ

            # splitting cov matrix in pos and neg parts
            d, e = LA.eigh(sigma)
            eig_idx = np.argsort(d)[::-1]
            e = e[eig_idx]
            d = d[:, eig_idx]
            new_d_pos, new_d_neg = d.copy(), d.copy()

            # Calculating only the requested amount
            n_gcpcs_pos = np.sum(new_d_pos > 0)
            # n_gcpcs_pos = np.sum(self.gcPCA_values_> 0)
            # n_gcpcs_neg = np.sum(new_d_neg < 0)
            if (n_gcpcs_pos - self.Nsparse) >= 0:
                n_gcpcs_pos = self.Nsparse
                n_gcpcs_neg = 0
            else:
                temp = self.Nsparse - n_gcpcs_pos
                n_gcpcs_neg = temp

            # separating positive and negative eigenvalues, flipping negative
            new_d_pos[new_d_pos < 0] = 0
            new_d_neg[new_d_neg > 0] = 0
            new_d_neg = new_d_neg * -1  # Flipping the sign of negative eigenvalues

            alpha_pos = new_d_pos.max() / self.cond_number  # fixing it to be positive definite
            Mpos = e @ np.sqrt(np.diag(new_d_pos+alpha_pos)) @ e.T
            
            alpha_neg = new_d_neg.max() / self.cond_number  # fixing it to be positive definite
            Mneg = e @ np.sqrt(np.diag(new_d_neg+alpha_neg)) @ e.T
            
            # if the user didn't input anything, provide a default lambda vector
            lambda_array = np.exp(np.linspace(np.log(1e-2), np.log(1), num=10))

            if n_gcpcs_pos > 0:
                Mpos_loadings_ = self.spca_algorithm(Mpos,
                                                self.original_loadings_[:, :n_gcpcs_pos],
                                                self.lambdas)
            else:
                Mpos_loadings_ = []

            if n_gcpcs_neg > 0:
                Mneg_loadings_ = self.spca_algorithm(Mneg,
                                                self.original_loadings_[:, n_gcpcs_pos:n_gcpcs_pos+n_gcpcs_neg],
                                                self.lambdas)  # """THINK OF BETTER NAME THAN ORIGINAL, MAYBE NONSPARSE, OR TEMP"""
            else:
                Mneg_loadings_ = []

            # Rearranging the PCs by vectors
            final_loadings = []
            for a in np.arange(self.lambdas.shape[0]):
                if n_gcpcs_pos > 0 and n_gcpcs_neg > 0:
                    final_loadings.append(np.concatenate((Mpos_loadings_[a], Mneg_loadings_[a]), axis=1))
                elif n_gcpcs_pos==0 and n_gcpcs_neg > 0:
                    final_loadings.append(np.array(Mneg_loadings_[a]))
                else:
                    final_loadings.append(np.array(Mpos_loadings_[a]))
        else:
            denom_well_conditioned = False
            #  Define numerator and denominator according to the method requested
            JRaRaJ = LA.multi_dot((J.T, RaRa, J))
            JRbRbJ = LA.multi_dot((J.T, RbRb, J))

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
            """solve sparse Y, then find x = J' M^-1 Y"""
            M = sqrtm(J@denominator@J.T)  # EFO: projecting back to neural space
            
            y = M@self.original_loadings_
            sigma = LA.multi_dot((LA.inv(M).T,J,numerator,J.T,LA.inv(M))) # EFO: projecting back to neural space

            # Getting eigenvectors
            d, e = LA.eigh(sigma)
            eig_idx = np.argsort(d)[::-1]
            e = e[:, eig_idx]
            d = d[eig_idx]
            new_d_pos, new_d_neg = d.copy(), d.copy()

            # Calculating only the requested amount
            n_gcpcs_pos = np.sum(new_d_pos > 0)
            if (n_gcpcs_pos - self.Nsparse) >= 0:
                n_gcpcs_pos = self.Nsparse
                n_gcpcs_neg = 0
            else:
                temp = self.Nsparse - n_gcpcs_pos
                n_gcpcs_neg = temp

            # separating positive and negative eigenvalues, flipping negative
            new_d_pos[new_d_pos < 0] = 0
            new_d_neg[new_d_neg > 0] = 0
            new_d_neg = new_d_neg * -1  # Flipping the sign of negative eigenvalues

            # Square root matrix of sigma plus
            alpha_pos = new_d_pos.max() / self.cond_number  # fixing it to be positive definite
            Mpos = e @ np.sqrt(np.diag(new_d_pos+alpha_pos)) @ e.T
            
            alpha_neg = new_d_neg.max() / self.cond_number  # fixing it to be positive definite
            Mneg = e @ np.sqrt(np.diag(new_d_neg+alpha_neg)) @ e.T

            if n_gcpcs_pos > 0:
                Mpos_loadings_ = self.spca_algorithm(Mpos,
                                                     y[:, :n_gcpcs_pos],
                                                     self.lambdas)
            else:
                Mpos_loadings_ = []

            if n_gcpcs_neg > 0:
                Mneg_loadings_ = self.spca_algorithm(Mneg,
                                                     y[:, n_gcpcs_pos:n_gcpcs_pos+n_gcpcs_neg],
                                                     self.lambdas)
            else:
                Mneg_loadings_ = []

            # Rearranging the PCs by vectors
            final_loadings = []
            for a in np.arange(self.lambdas.shape[0]):
                if  n_gcpcs_pos > 0 and n_gcpcs_neg > 0:
                    sigma_pos_loadings_ = LA.inv(M) @ Mpos_loadings_[a]
                    sigma_neg_loadings_ = LA.inv(M) @ Mneg_loadings_[a]
                    final_loadings.append(np.concatenate((sigma_pos_loadings_, sigma_neg_loadings_), axis=1))
                elif n_gcpcs_pos == 0 and n_gcpcs_neg > 0:
                    sigma_neg_loadings_ = LA.inv(M) @ Mneg_loadings_[a]
                    final_loadings.append(sigma_neg_loadings_)
                else:
                    sigma_pos_loadings_ = LA.inv(M) @ Mpos_loadings_[a]
                    """LASSO IS PICKING THE BEST J TO PICK! THAT'S WHY ITS DENSE, 
                    Y IS ALREADY VERY SPARSE SO LASSO IS NOT CHANGING IT MUCH
                    when lambda_lasso is low it mixes more PCs and the end result lookssparse
                    but that's because we are not looking in the right space"""
                    # sigma_pos_loadings_ = Mpos_loadings_[a]
                    final_loadings.append(sigma_pos_loadings_)
            # x_temp = LA.multi_dot((J, LA.inv(m), v))
        self.sparse_loadings_ = final_loadings
        
        temp_ra_scores = []
        temp_ra_values = []
        for sload in final_loadings:
            temp = np.dot(self.Ra,sload)
            temp_ra_scores.append(np.divide(temp,LA.norm(temp,axis=0)))
            temp_ra_values.append(LA.norm(temp,axis=0))

        self.Ra_scores_ = temp_ra_scores
        self.Ra_values_ = temp_ra_values

        temp_rb_scores = []
        temp_rb_values = []
        for sload in final_loadings:
            temp = np.dot(self.Rb,sload)
            temp_rb_scores.append(np.divide(temp,LA.norm(temp,axis=0)))
            temp_rb_values.append(LA.norm(temp,axis=0))

        self.Rb_scores_ = temp_rb_scores
        self.Rb_values_ = temp_rb_values
        # temp = np.dot(Ra, x);
        
        # self.Ra_values_ = LA.norm(temp,axis=0)
        # temp = np.dot(Rb, x);
        # self.Rb_scores_ = np.divide(temp,LA.norm(temp,axis=0))
        # self.Rb_values_ = LA.norm(temp,axis=0)
        # self.objective_function_ = obj_info
        # self.objective_values_ = s_total

    def spca_algorithm(self, M, V, lambda_array):
        from sklearn.linear_model import LassoLars
        """
        Algorithm that finds the sparse PCs, as described in Zou, Hastie and Tibshirani, 2006
        :param lambda_array: array with values of lambda to fit the model
        :param v: arrays of initial PCs or gcPCs, size n x PCs, colum-wise PCs
        :param n_pcs: number of n_pcs to return
        :return: fitted sparse PCs for each lambda
        """
        betas_by_lambda = []
        for lmbd in lambda_array:  # looping through lambdas vector
            A = V.copy()
            step, diff_criterion, criterion_past = 0, 1, 1e6
            while (diff_criterion > self.tol and step < self.max_steps):
                # ####
                # Elastic net (lasso) step
                beta = []
                for a in np.arange(V.shape[1]):  # Looping through PCs
                    y = M @ A[:, a]

                    # Solving L1 constrain with least angle regression
                    lasso_mdl = LassoLars(alpha=lmbd,
                                          fit_intercept=False,
                                          normalize=False)
                    lasso_mdl.fit(M, y)
                    beta.append(lasso_mdl.coef_)
                betas = np.vstack(beta).T

                # ###
                # Reduced Rank Procrustes rotation
                u, _, v = LA.svd(M @ M.T @ betas, full_matrices=False)
                A = u @ v
                # Checking convergence of criterion
                criterion = np.sum(LA.norm(M - A @ betas.T @ M, axis=0)) + np.sum(lmbd * LA.norm(betas, ord=1, axis=0))
                diff_criterion = np.abs(criterion_past - criterion)
                criterion_past = criterion.copy()  # update criterion
                step += 1

            if step >= self.max_steps:
                warnings.warn('sparse gcPCA did not converge to tol., returning last iteration gcPCs')
            temp_betas = np.divide(betas, LA.norm(betas, axis=0))
            temp_betas[:,LA.norm(betas, axis=0)==0] = 0  # when norm is zero, returning nan
            betas_by_lambda.append(temp_betas)
        return betas_by_lambda  # Returning sparse PCs for each lambda