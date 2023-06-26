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
#%% defining classes for each method

#%% class for generalized contrastive PCA

class gcPCA():
    
    """TO DO
    [ ]add a method for doing fit for multiple alphas and returning multiple models """
    def __init__(self, method='v4', Nshuffle=0, normalize_flag=True, alpha=1, alpha_null=0.975,cond_number = 10**13):
        self.method         = method
        self.Nshuffle       = Nshuffle
        self.normalize_flag = normalize_flag
        self.alpha          = alpha
        self.alpha_null     = alpha_null
        self.cond_number    = cond_number
    
    def normalize(self):
        """method to normalize data to have same norm"""
        from scipy import stats
        import numpy as np
        import numpy.linalg as LA
        
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
        
        
    def fit(self,Ra,Rb): 
        import numpy as np
        import numpy.linalg as LA
        
        #parameters
        method_strings = np.array(['v1','v2','v3','v4'])
        method         = self.method
        Nshuffle       = self.Nshuffle
        normalize_flag = self.normalize_flag
        
        #putting data into class
        self.Ra = Ra
        self.Rb = Rb
        
        #test that inputs have the same number of inputs
        if Ra.shape[1] != Rb.shape[1]:
            raise ValueError("Ra and Rb have different numbers of features")
        
        #normalizing as to have same norm
        if normalize_flag:
            self.normalize()
            Ra = self.Ra
            Rb = self.Rb
        
        #covariance matrices
        RaRa = Ra.T.dot(Ra)
        RbRb = Rb.T.dot(Rb)
        
        #find the gcPCA loadings accord to method requested
        if method == 'v1': #original cPCA
            alpha = self.alpha
            sigma = RaRa - alpha*RbRb #doing pseudo inverse to throw out eigenvalues that would blow this ratio to infinity
        
            # getting eigenvalues and eigenvectors
            w, v = LA.eig(sigma)
            eig_idx = np.argsort(w)[::-1]
            
            X = v[:,eig_idx]
            S_total = w[eig_idx]
            
        elif method == 'v2': #this is similar to cPCA++
        
            #check if RbRb is well-conditioned, if not we adjust the matrix
            if LA.cond(RbRb) > self.cond_number:
                warnings.warn("Rb is ill-conditioned, fixing it. " +
                              "Be aware that gcPCA values will be" +
                              "slightly smaller")
                w = LA.eigvals(RbRb)
                w = w[np.argsort(w)[::-1]]
                
                alpha = w[0]/self.cond_number - w[-1]
                
                RbRb = RbRb + np.eye(RbRb.shape[0])*alpha
            
            sigma = np.dot(LA.inv(RbRb),RaRa)
            # getting eigenvalues and eigenvectors
            w, v = LA.eig(sigma)
            eig_idx = np.argsort(w)[::-1]
            
            X = v[:,eig_idx]
            S_total = w[eig_idx]
            
        elif method == 'v2.1': #this is similar to cPCA++, but orthogonal
            from scipy.linalg import sqrtm
            #check if RbRb is well-conditioned, if not we adjust the matrix
            if LA.cond(RbRb) > self.cond_number:
                warnings.warn("Rb is ill-conditioned, fixing it. " +
                              "Be aware that gcPCA values will be" +
                              "slightly smaller")
                w = LA.eigvals(RbRb)
                w = w[np.argsort(w)[::-1]]
                
                alpha = w[0]/self.cond_number - w[-1]
                
                RbRb = RbRb + np.eye(RbRb.shape[0])*alpha
            
            M = sqrtm(RbRb)
            sigma = LA.multi_dot((LA.inv(M).T,RaRa,LA.inv(M)))
            
            # getting eigenvalues and eigenvectors
            w, v = LA.eig(sigma)
            eig_idx = np.argsort(w)[::-1]
            
            X = v[:,eig_idx]
            S_total = w[eig_idx]

        elif method == 'v3': #this is (A-B)/B
            from scipy.linalg import sqrtm
            if LA.cond(RbRb) > self.cond_number:
                warnings.warn("Rb is ill-conditioned, fixing it. " +
                              "Be aware that gcPCA values will be" +
                              "slightly smaller")
                w = LA.eigvals(RbRb)
                w = w[np.argsort(w)[::-1]]
                
                alpha = w[0]/self.cond_number - w[-1]
                
                RbRb = RbRb + np.eye(RbRb.shape[0])*alpha
                
            # Calculating gcPCA below
            B = sqrtm(RbRb)
            BtRaRbB = LA.multi_dot((LA.inv(B).T,RaRa-RbRb,LA.inv(B)))
            
            D, y = LA.eig(BtRaRbB)
            Y = y[:, np.flip(np.argsort(D))]
            X_temp = np.dot(LA.inv(B),Y)
            X = X_temp/LA.norm(X_temp, axis=0)

            # getting top and bottom eigenvalues
            Stop = LA.multi_dot((X.T,RaRa-RbRb,X))
            Sbot = LA.multi_dot((X.T,RbRb,X))
            S_total = np.divide(np.diagonal(Stop),np.diagonal(Sbot))
            
        elif method == 'v3.1': #this is (A-B)/B but with orthogonality constrain
            from scipy.linalg import sqrtm,orth
            
            if LA.cond(RbRb) > self.cond_number:
                warnings.warn("Rb is ill-conditioned, fixing it. " +
                              "Be aware that gcPCA values will be" +
                              "slightly smaller")
                w = LA.eigvals(RbRb)
                w = w[np.argsort(w)[::-1]]
                
                alpha = w[0]/self.cond_number - w[-1]
                
                RbRb = RbRb + np.eye(RbRb.shape[0])*alpha
                
            # Calculating gcPCA below
            # Iteratively take out the ncPCs by deflating basis
            n_basis = RbRb.shape[0]

            _, _, V = LA.svd(RbRb, full_matrices=False)
            
            Jnew = V.T
            for aa in np.arange(n_basis):
                B = sqrtm(LA.multi_dot((Jnew.T, RbRb, Jnew)))

                # find the eigenvalues and sort according to magnitude
                JBinv = LA.lstsq(Jnew.T, LA.pinv(B))[0]
                D, y = LA.eig(LA.multi_dot((JBinv.T, RaRa-RbRb, JBinv)))
                Y = y[:, np.flip(np.argsort(D))]

                X_temp = np.dot(JBinv, Y[:,0]);

                if aa == 0:
                    X = X_temp/LA.norm(X_temp, axis=0)
                else:
                    temp_norm_ldgs = X_temp/LA.norm(X_temp, axis=0)
                    X = np.column_stack((X, temp_norm_ldgs))

                # deflating J
                gamma = np.dot(LA.pinv(B), Y[:, 0])
                gamma = gamma/LA.norm(gamma)  # normalizing by the norm
                gamma_outer = np.outer(gamma, gamma.T)
                
                J_reduced = Jnew - np.dot(Jnew,gamma_outer)
                # J_reduced = J_reduced/LA.norm(J_reduced,axis=0)
                # _, _, V = LA.svd(LA.multi_dot((J_reduced.T,RbRb,J_reduced)), full_matrices=False)
                # new_rank = LA.matrix_rank(LA.multi_dot((J_reduced.T,RbRb,J_reduced)))
                # Jnew = J_reduced.dot(V[:new_rank,:].T)
                
                Jnew = orth(J_reduced)
            # getting top and bottom eigenvalues
            Stop = LA.multi_dot((X.T, RaRa-RbRb, X))
            Sbot = LA.multi_dot((X.T, RbRb, X))
            S_total = np.divide(np.diagonal(Stop), np.diagonal(Sbot))
              
        elif method == 'v3.2':
            if LA.cond(RbRb) > self.cond_number:
                warnings.warn("Rb is ill-conditioned, fixing it. " +
                              "Be aware that gcPCA values will be" +
                              "slightly smaller")
                w = LA.eigvals(RbRb)
                w = w[np.argsort(w)[::-1]]
                
                alpha = w[0]/self.cond_number - w[-1]
                
                RbRb = RbRb + np.eye(RbRb.shape[0])*alpha
                
            # Calculating gcPCA below
            iRbRaRb = np.dot(LA.inv(RbRb),RaRa-RbRb)
            
            D, x = LA.eig(iRbRaRb)
            X = x[:, np.flip(np.argsort(D))]

            # getting top and bottom eigenvalues
            Stop = LA.multi_dot((X.T,RaRa-RbRb,X))
            Sbot = LA.multi_dot((X.T,RbRb,X))
            S_total = np.divide(np.diagonal(Stop),np.diagonal(Sbot))

        elif method == 'v3.3': # the orthogonal version
            from scipy.linalg import sqrtm
            if LA.cond(RbRb) > self.cond_number:
                warnings.warn("Rb is ill-conditioned, fixing it. " +
                              "Be aware that gcPCA values will be" +
                              "slightly smaller")
                w = LA.eigvals(RbRb)
                w = w[np.argsort(w)[::-1]]
                
                alpha = w[0]/self.cond_number - w[-1]
                
                RbRb = RbRb + np.eye(RbRb.shape[0])*alpha
                
            # Calculating gcPCA below
            M = sqrtm(RbRb)
            iRbRaRb = LA.multi_dot((LA.inv(M).T,RaRa-RbRb,LA.inv(M)))
            D, x = LA.eig(iRbRaRb)
            X = x[:, np.flip(np.argsort(D))]

            # getting top and bottom eigenvalues
            Stop = LA.multi_dot((X.T,RaRa-RbRb,X))
            Sbot = LA.multi_dot((X.T,RbRb,X))
            S_total = np.divide(np.diagonal(Stop),np.diagonal(Sbot))
            
        elif method == 'v4': #this is ncPCA index
            from scipy.linalg import sqrtm
            
            RaRaRbRb = RaRa+RbRb
            if LA.cond(RaRaRbRb) > self.cond_number:
                warnings.warn("RaRa+RbRb is ill-conditioned, fixing it. " +
                              "Be aware that gcPCA values will be" +
                              "slightly smaller")
                w = LA.eigvals(RaRaRbRb)
                w = w[np.argsort(w)[::-1]]
                
                alpha = w[0]/self.cond_number - w[-1]
                
                RaRaRbRb = RaRaRbRb + np.eye(RaRaRbRb.shape[0])*alpha
                
            ## Calculating gcPCA below
            B = sqrtm(RaRaRbRb)
            BtRaRbB = LA.multi_dot((LA.inv(B).T,RaRa-RbRb,LA.inv(B)))
            
            D, y = LA.eig(BtRaRbB)
            Y = y[:, np.flip(np.argsort(D))]
            X_temp = np.dot(LA.inv(B),Y)
            X = X_temp/LA.norm(X_temp, axis=0)
    
            # getting top and bottom eigenvalues
            Stop = LA.multi_dot((X.T,RaRa-RbRb,X))
            Sbot = LA.multi_dot((X.T,RaRaRbRb,X))
            S_total = np.divide(np.diagonal(Stop),np.diagonal(Sbot))
            
        elif method == 'v4.1': #this is ncPCA index with orthogonality constrain
            from scipy.linalg import sqrtm,orth
            
            RaRaRbRb = RaRa+RbRb
            if LA.cond(RaRaRbRb) > self.cond_number:
                warnings.warn("RaRa+RbRb is ill-conditioned, fixing it. " +
                              "Be aware that gcPCA values will be" +
                              "slightly smaller")
                w = LA.eigvals(RaRaRbRb)
                w = w[np.argsort(w)[::-1]]
                
                alpha = w[0]/self.cond_number - w[-1]
                
                RaRaRbRb = RaRaRbRb + np.eye(RaRaRbRb.shape[0])*alpha
                
            # Calculating gcPCA below
            # Iteratively take out the ncPCs by deflating basis
            RaRaRbRb = RaRa+RbRb
            n_basis = RaRaRbRb.shape[0]

            _, _, V = LA.svd(RaRaRbRb, full_matrices=False)
            
            Jnew = V.T
            for aa in np.arange(n_basis):
                B = sqrtm(LA.multi_dot((Jnew.T, RaRaRbRb, Jnew)))

                # find the eigenvalues and sort according to magnitude
                JBinv = LA.lstsq(Jnew.T, LA.pinv(B))[0]
                D, y = LA.eig(LA.multi_dot((JBinv.T, RaRa-RbRb, JBinv)))
                Y = y[:, np.flip(np.argsort(D))]

                X_temp = np.dot(JBinv, Y[:,0]);

                if aa == 0:
                    X = X_temp/LA.norm(X_temp, axis=0)
                else:
                    temp_norm_ldgs = X_temp/LA.norm(X_temp, axis=0)
                    X = np.column_stack((X, temp_norm_ldgs))

                # deflating J
                gamma = np.dot(LA.pinv(B), Y[:, 0])
                gamma = gamma/LA.norm(gamma)  # normalizing by the norm
                gamma_outer = np.outer(gamma, gamma.T)
                
                J_reduced = Jnew - np.dot(Jnew,gamma_outer)
                Jnew = orth(J_reduced)

            # getting top and bottom eigenvalues
            Stop = LA.multi_dot((X.T, RaRa-RbRb, X))
            Sbot = LA.multi_dot((X.T, RaRa+RbRb, X))
            S_total = np.divide(np.diagonal(Stop), np.diagonal(Sbot))

        elif method == 'v4.2': #solving as inv((A+B))*(A-B)
            
            RaRaRbRb = RaRa+RbRb
            if LA.cond(RaRaRbRb) > self.cond_number:
                warnings.warn("RaRa+RbRb is ill-conditioned, fixing it. " +
                              "Be aware that gcPCA values will be" +
                              "slightly smaller")
                w = LA.eigvals(RaRaRbRb)
                w = w[np.argsort(w)[::-1]]
                
                alpha = w[0]/self.cond_number - w[-1]
                
                RaRaRbRb = RaRaRbRb + np.eye(RaRaRbRb.shape[0])*alpha
                
            # Calculating gcPCA below
            iRbRaRb = np.dot(LA.inv(RaRaRbRb),RaRa-RbRb)
            
            D, x = LA.eig(iRbRaRb)
            X = x[:, np.flip(np.argsort(D))]

            # getting top and bottom eigenvalues
            Stop = LA.multi_dot((X.T,RaRa-RbRb,X))
            Sbot = LA.multi_dot((X.T,RaRa+RbRb,X))
            S_total = np.divide(np.diagonal(Stop),np.diagonal(Sbot))

        elif method == 'v4.3': # the orthogonal version inv(M)*(A-B)*inv(M), M=sqrtm(A+B)
            from scipy.linalg import sqrtm
            RaRaRbRb = RaRa+RbRb
            if LA.cond(RaRaRbRb) > self.cond_number:
                warnings.warn("RaRa+RbRb is ill-conditioned, fixing it. " +
                              "Be aware that gcPCA values will be" +
                              "slightly smaller")
                w = LA.eigvals(RaRaRbRb)
                w = w[np.argsort(w)[::-1]]
                
                alpha = w[0]/self.cond_number - w[-1]
                
                RaRaRbRb = RaRaRbRb + np.eye(RaRaRbRb.shape[0])*alpha
                
            # Calculating gcPCA below
            M = sqrtm(RaRaRbRb)
            iRbRaRb = LA.multi_dot((LA.inv(M).T,RaRa-RbRb,LA.inv(M)))
            D, x = LA.eig(iRbRaRb)
            X = x[:, np.flip(np.argsort(D))]

            # getting top and bottom eigenvalues
            Stop = LA.multi_dot((X.T,RaRa-RbRb,X))
            Sbot = LA.multi_dot((X.T,RaRa+RbRb,X))
            S_total = np.divide(np.diagonal(Stop),np.diagonal(Sbot))
                
        self.loadings_ = X
        self.gcPCA_values_ = S_total
        self.Ra_scores_ = np.dot(Ra,X)
        self.Rb_scores_ = np.dot(Rb,X)
        self.Ra = Ra
        self.Rb = Rb
                
        # shuffling to define a null distribution
        if Nshuffle>0:
            self.null_distribution()
            
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