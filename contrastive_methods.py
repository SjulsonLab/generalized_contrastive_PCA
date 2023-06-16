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
    def __init__(self, method='v4', Nshuffle=0, normalize_flag=True, alpha_null=0.975):
        self.method         = method
        self.Nshuffle       = Nshuffle
        self.normalize_flag = normalize_flag
        self.alpha_null     = alpha_null
    
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
        
        
    def fit(self,Ra,Rb,alpha=1): 
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
        
            sigma = RaRa - alpha*RbRb #doing pseudo inverse to throw out eigenvalues that would blow this ratio to infinity
        
            # getting eigenvalues and eigenvectors
            w, v = LA.eig(sigma)
            eig_idx = np.argsort(w)[::-1]
            
            X = v[:,eig_idx]
            S_total = w[eig_idx]
            
            """I NEED TO UPDATE FROM HERE DOWN!!!!"""
        elif method == 'v2': 
        
            sigma = np.dot(LA.pinv(RbRb),RaRa) #doing pseudo inverse to throw out eigenvalues that would blow this ratio to infinity
            
            # getting eigenvalues and eigenvectors
            w, v = LA.eig(sigma)
            eig_idx = np.argsort(w)[::-1]
            
            X = v[:,eig_idx]
            
            S_total = w[eig_idx]
            
        elif method == 'v3': #this is ncPCA ratio
            #SVD (or PCA) on N1
            _, S1, V1 = np.linalg.svd(RbRb, full_matrices = False)
            
            # Picking based on rank (eps)
            rank = np.linalg.matrix_rank(N1N1)

            V1_hat = V1[:rank,:]
            J = V1_hat.T

            if J.size==0:
                warnings.warn("No basis found in N1 and N2")
                self.number_of_shared_basis = 0
                self.loadings_ = np.nan
                self.ncPCs_values_ = np.nan
                self.N1_scores_ = np.nan
                self.N2_scores_ = np.nan
                self.N1 = np.nan
                self.N2 = np.nan
            else:
                
                k = J.shape[1]
                
                ## Calculating ncPCA below
                ######### Iteratively take out the ncPCs by deflating J
                n_basis = J.shape[1]
                Jnew = J
                for aa in np.arange(n_basis):
                    B = LA.sqrtm(np.linalg.multi_dot((Jnew.T,RbRb,Jnew)))
                    
                    JBinv =  np.linalg.lstsq(Jnew.T, np.linalg.pinv(B))[0]
                    
                    # sort according to decreasing eigenvalue
                    D,y = np.linalg.eig(np.linalg.multi_dot((JBinv.T,RaRa-RbRb,JBinv)))
                    
                    Y = y[:,np.flip(np.argsort(D))]
                    
                    X_temp = np.dot(JBinv,Y[:,0]);
                    
                    if aa == 0:
                        X = X_temp/np.linalg.norm(X_temp,axis=0)
                    else:
                        temp_norm_ldgs = X_temp/np.linalg.norm(X_temp,axis=0)
                        X = np.column_stack((X,temp_norm_ldgs))
                    
                    #deflating J
                    gamma = np.dot(np.linalg.pinv(B),Y[:,0])
                    gamma = gamma/np.linalg.norm(gamma) #normalizing by the norm
                    gamma_outer = np.outer(gamma,gamma.T)
                    J_reduced = Jnew - np.dot(Jnew,gamma_outer)
                    Jnew = LA.orth(J_reduced)
                    #Jnew = J_reduced
                ##########
                
                # getting top and bottom eigenvalues
                Stop = np.linalg.multi_dot((X.T,RaRa-RbRb,X))
                Sbot = np.linalg.multi_dot((X.T,RbRb,X))
                S_total = np.divide(np.diagonal(Stop),np.diagonal(Sbot))
                
                
        elif method == 'v4': #this is ncPCA index
            #SVD (or PCA) on N1 and N2
            _, S1, V1 = np.linalg.svd(RaRa+RbRb, full_matrices = False)
            
            # Picking based on rank (eps)
            rank = np.linalg.matrix_rank(RaRa+RbRb)

            V1_hat = V1[:rank,:]
            J = V1_hat.T

            if J.size==0:
                warnings.warn("No basis found in N1 and N2")
                self.number_of_shared_basis = 0
                self.loadings_ = np.nan
                self.ncPCs_values_ = np.nan
                self.N1_scores_ = np.nan
                self.N2_scores_ = np.nan
                self.N1 = np.nan
                self.N2 = np.nan
            else:
                
                k = J.shape[1]
                
                ## Calculating ncPCA below
                
                """I think the loop below can be discarded with this new method
                because the matrix are symetric then the eigenvectors should all be
                orthogonal? I'm not sure about that"""
                ######### Iteratively take out the ncPCs by deflating J
                n_basis = J.shape[1]
                Jnew = J
                for aa in np.arange(n_basis):
                    B = LA.sqrtm(np.linalg.multi_dot((Jnew.T,RaRa+RbRb,Jnew)))
                    
                    JBinv =  np.linalg.lstsq(Jnew.T, np.linalg.pinv(B))[0]
                    
                    # sort according to decreasing eigenvalue
                    D,y = np.linalg.eig(np.linalg.multi_dot((JBinv.T,RaRa-RbRb,JBinv)))
                    
                    Y = y[:,np.flip(np.argsort(D))]
                    
                    X_temp = np.dot(JBinv,Y[:,0]);
                    
                    if aa == 0:
                        X = X_temp/np.linalg.norm(X_temp,axis=0)
                    else:
                        temp_norm_ldgs = X_temp/np.linalg.norm(X_temp,axis=0)
                        X = np.column_stack((X,temp_norm_ldgs))
                    
                    #deflating J
                    gamma = np.dot(np.linalg.pinv(B),Y[:,0])
                    gamma = gamma/np.linalg.norm(gamma) #normalizing by the norm
                    gamma_outer = np.outer(gamma,gamma.T)
                    J_reduced = Jnew - np.dot(Jnew,gamma_outer)
                    Jnew = LA.orth(J_reduced)
                    #Jnew = J_reduced
                ##########
                
                # getting top and bottom eigenvalues
                Stop = np.linalg.multi_dot((X.T,RaRa-RbRb,X))
                Sbot = np.linalg.multi_dot((X.T,RaRa+RbRb,X))
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