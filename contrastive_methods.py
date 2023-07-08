# -*- coding: utf-8 -*-

"""
Created on Thu May  4 17:52:12 2023

Set of classes to do different contrastive methods in foreground (FG) and
background (BG), it's implemented here 1) contrastive PCA (FG - alpha*BG),
2) ratio contrastive PCA (FG/BG), 3) normalized contrastive PCA ((FG-BG)/BG),
4) index normalized contrastive PCA ((FG-BG)/(FG+BG)). The methods are called
1) cPCA, 2) ratio_cPCA, 3) ncPCA, 4) index_ncPCA

@author: Eliezyer de Oliveira
"""

#%% should prepare this code as a class with methods and etc.
# Return an object with data projected on to the ncPCs
# loadings
# other stuff
# put option to normalize the data or not

import warnings
import numpy as np
import numpy.linalg as LA
#%% defining classes for each method

#%% class for generalized contrastive PCA

class gcPCA():
    
    """TO DO
    [ ]add a method for doing fit for multiple alphas and returning multiple models """
    def __init__(self, method='v4.1', Ncalc = np.inf, Nshuffle=0, normalize_flag=True, alpha=1, alpha_null=0.975,cond_number = 10**13):
        self.method         = method
        self.Nshuffle       = Nshuffle
        self.normalize_flag = normalize_flag
        self.alpha          = alpha
        self.alpha_null     = alpha_null
        self.cond_number    = cond_number
        self.Ncalc          = Ncalc
    
    def normalize(self):
        """method to normalize data to have same norm"""
        from scipy import stats
        
        Ra_temp = np.divide(stats.zscore(self.Ra),LA.norm(stats.zscore(self.Ra),axis=0))
        Rb_temp = np.divide(stats.zscore(self.Rb),LA.norm(stats.zscore(self.Rb),axis=0))
        

        if np.sum(np.sum(np.square(Ra_temp - self.Ra)) > (0.01*np.square(Ra_temp))):
            warnings.warn("Ra was not normalized properly - normalizing now")
            Ra = Ra_temp
            self.Ra = Ra
            
        if np.sum(np.sum(np.square(Rb_temp - self.Rb)) > (0.01*np.square(Rb_temp))):
            warnings.warn("Rb was not normalized properly - normalizing now")
            Rb = Rb_temp
            self.Rb = Rb
    
    def inspect_inputs(self):
        """Method to inspect the data for multiple criterias, as number of 
        features, normalization and number of gcPCs possible to get vs
        number of gcPCs requested by the user """
        
        #test that inputs have the same number of inputs
        if self.Ra.shape[1] != self.Rb.shape[1]:
            raise ValueError("Ra and Rb have different numbers of features")
        
        #normalizing as to have same variance and norm
        if self.normalize_flag:
            self.normalize()

        # discard dimensions if necessary
        # whichever dataset has less datapoints dictate the amount of gcPCs
        N_gcPCs = int(np.min((self.Ra.shape, self.Rb.shape)))
        
        # SVD on the combined dataset to discard dimensions with near-zero variance
        RaRb = np.concatenate((self.Ra,self.Rb),axis=0)
        _, Sab, V = LA.svd(RaRb,full_matrices=False)
        tol  = max(RaRb.shape) * np.finfo(Sab.max()).eps
        
        if sum(Sab>tol) < N_gcPCs:
            warnings.warn('Input data is rank-deficient! Discarding dimensions; cannot shuffle.')
            N_gcPCs = sum(Sab>tol); # the new number of gcPCs
            Nshuffle = 0;
        
        #setting number of gcPCs to return
        if sum(np.char.equal(self.method,['v1','v2','v3','v4'])) & ~np.isinf(self.Ncalc):
            warnings.warn('Ncalc is only relevant if using orthogonal gcPCA. Will calculate full set of gcPCs.')
            print(str(N_gcPCs) + ' gcPCs will be returned.')
        elif sum(np.char.equal(self.method,['v2.1','v3.1','v4.1'])):
            N_gcPCs = int(np.min((self.Ncalc,N_gcPCs))) #either the requested by user or maximum rank there can be
            print(str(N_gcPCs) + ' gcPCs will be returned.')
        
        J = V.T.copy()
        self.N_gcPCs = N_gcPCs
        self.Jorig   = J[:,:N_gcPCs]
        
    def fit(self,Ra,Rb): 
        from scipy.linalg import sqrtm
        
        #parameters
        method         = self.method
        Ncalc          = self.Ncalc
        Nshuffle       = self.Nshuffle
        normalize_flag = self.normalize_flag        
        
        #putting data into class
        self.Ra = Ra
        self.Rb = Rb
        
        #inspecting whether the inputs are normalized and dimensionality to use
        self.inspect_inputs()
        Jorig = self.Jorig 
        J = Jorig.copy() # J shrinks every round, but Jorig is the first-round's J
        
        #in orthogonal gcPCA, we interate multiple time
        if sum(np.char.equal(method,['v2.1','v3.1','v4.1'])):
            Niter = self.N_gcPCs
        else:
            Niter = 1    

        #covariance matrices
        RaRa = self.Ra.T.dot(self.Ra)
        RbRb = self.Rb.T.dot(self.Rb)

        # solving gcPCA
        if method == 'v1': #original cPCA
            alpha = self.alpha
            JRaRaJ = LA.multi_dot((J.T,RaRa,J))
            JRbRbJ = LA.multi_dot((J.T,RbRb,J))
            sigma = JRaRaJ - alpha*JRbRbJ #doing pseudo inverse to throw out eigenvalues that would blow this ratio to infinity
        
            # getting eigenvalues and eigenvectors
            w, v = LA.eigh(sigma)
            eig_idx = np.argsort(w)[::-1]
            
            X = J.dot(v[:,eig_idx])
            S_total = w[eig_idx]
            obj_info    = 'Ra - alpha * Rb'
        else:

            denom_well_conditioned = False
            for idx in np.arange(Niter):
                #find the gcPCA loadings accord to method requested
                JRaRaJ = LA.multi_dot((J.T,RaRa,J))
                JRbRbJ = LA.multi_dot((J.T,RbRb,J))
                
                if sum(np.char.equal(method,['v2','v2.1'])):
                       numerator        = JRaRaJ
                       denominator      = JRbRbJ
                       obj_info    = 'Ra / Rb'
                elif sum(np.char.equal(method,['v3','v3.1'])):
                       numerator        = JRaRaJ - JRbRbJ
                       denominator      = JRbRbJ
                       obj_info    = '(Ra-Rb) / Rb'
                elif sum(np.char.equal(method,['v4','v4.1'])):
                       numerator        = JRaRaJ - JRbRbJ
                       denominator      = JRaRaJ + JRbRbJ
                       obj_info    = '(Ra-Rb) / (Ra+Rb)'  

                if denom_well_conditioned == False:
                    if LA.cond(denominator) > self.cond_number:
                        warnings.warn("Denominator is ill-conditioned, fixing it. " +
                                      "Be aware that gcPCA values will be" +
                                      "slightly smaller")
                        
                        w = LA.eigvals(denominator)
                        w = w[np.argsort(w)[::-1]]
                        alpha = w[0]/self.cond_number - w[-1]
                        denominator = denominator + np.eye(denominator.shape[0])*alpha
                        denom_well_conditioned = True;
                    else:
                        denom_well_conditioned = True;
                
                #mounting the equation
                M = sqrtm(denominator)
                sigma = LA.multi_dot((LA.inv(M).T,numerator,LA.inv(M)))
                #getting eigenvectors
                w, v = LA.eigh(sigma)
                eig_idx = np.argsort(w)[::-1]
                v = v[:,eig_idx]
                
                X_temp = LA.multi_dot((J,LA.inv(M),v))
                X_temp = np.divide(X_temp,LA.norm(X_temp,axis=0))

                #copy results to X
                if idx == 0:
                    X = X_temp;
                    X_orth = np.expand_dims(X_temp[:,0],axis=1)
                else:
                    X_add = np.expand_dims(X_temp[:,0],axis=1)
                    X_orth = np.hstack((X_orth,X_add))
                    
                # shrinking J (find an orthonormal basis for the subspace of J orthogonal
                # to the X vectors we have already collected)
                J,_,_ = LA.svd(Jorig - LA.multi_dot((X_orth,X_orth.T,Jorig)),full_matrices=False)
                J = J[:,:Niter-(idx+1)]
                
            #getting orthogonal loadings if it was method requested
            if sum(np.char.equal(method, ['v2.1', 'v3.1', 'v4.1'])):
                X = X_orth

            # getting the eigenvalue of gcPCA
            if sum(np.char.equal(method,['v2','v2.1'])):
                numerator_orig   = RaRa
                denominator_orig = RbRb
            elif sum(np.char.equal(method,['v3','v3.1'])):
                numerator_orig   = RaRa - RbRb
                denominator_orig = RbRb
            elif sum(np.char.equal(method,['v4','v4.1'])):
                numerator_orig   = RaRa - RbRb
                denominator_orig = RaRa + RbRb
            Stop = LA.multi_dot((X.T, numerator_orig, X))
            Sbot = LA.multi_dot((X.T, denominator_orig, X))
            S_total = np.divide(np.diagonal(Stop), np.diagonal(Sbot))
                
        self.loadings_ = X
        self.gcPCA_values_ = S_total
        self.Ra_scores_ = np.dot(Ra ,X)
        self.Rb_scores_ = np.dot(Rb, X)
        self.objetive_function_ = obj_info
                
        # shuffling to define a null distribution
        if Nshuffle>0:
            self.null_distribution()
        """add null method"""
        def transform(self,Ra,Rb):
            import numpy as np
            try:
                X = self.loadings_
                Ra_transf = np.dot(Ra,X)
                Rb_transf = np.dot(Rb,X)
                self.Ra_transformed_ = Ra_transf
                self.Rb_transformed_ = Rb_transf
            except:
                print('Loadings not defined, you have to first fit the model')        