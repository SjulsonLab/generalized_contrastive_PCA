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


#%% defining classes for each method

#%% class for classical cPCA

class cPCA():
    """Class for cPCA with alpha as in the original implementation, FG - alpha*BG"""
    def __init__(self,Nshuffle=0,normalize_flag=True,alpha=1,alpha_null=0.975):
        self.Nshuffle       = Nshuffle
        self.normalize_flag = normalize_flag
        self.alpha_null     = alpha_null
        self.alpha          = alpha
        
    def fit(self,N1,N2): 
 
        """ method to perform ratio contrastive PCA FG/BG
        """
        import numpy as np
        import numpy.linalg as LA
        
        #parameters
        Nshuffle       = self.Nshuffle
        normalize_flag = self.normalize_flag
        alpha          = self.alpha
        #test that inputs are normalized
        if N2.shape[1] != N1.shape[1]:
            raise ValueError("N1 and N2 have different numbers of features")
        
        if normalize_flag:
            N1_temp = np.divide(stats.zscore(N1),LA.norm(stats.zscore(N1),axis=0))
            N2_temp = np.divide(stats.zscore(N2),LA.norm(stats.zscore(N2),axis=0))
            
            if np.sum(np.sum(np.square(N1_temp - N1)) > (0.01*np.square(N1_temp))):
                warnings.warn("N1 was not normalized properly - normalizing now")
                N1 = N1_temp
            
            if np.sum(np.sum(np.square(N2_temp - N2)) > (0.01*np.square(N2_temp))):
                warnings.warn("N2 was not normalized properly - normalizing now")
                N2 = N2_temp
        
        #covariance matrices
        N1N1 = np.dot(N1.T,N1)
        N2N2 = np.dot(N2.T,N2)
        
        sigma = N2N2 - alpha*N1N1 #doing pseudo inverse to throw out eigenvalues that would blow this ratio to infinity
        
        # getting eigenvalues and eigenvectors
        w, v = LA.eig(sigma)
        eig_idx = np.argsort(w)[::-1]
        
        X = v[:,eig_idx]
        
    
        S_total = w[eig_idx]
    
        self.number_of_shared_basis = J.shape[1]
        self.loadings_ = X
        self.eigenvalues_ = S_total
        self.N1_scores_ = np.dot(N1,X)
        self.N2_scores_ = np.dot(N2,X)
        self.N1 = N1
        self.N2 = N2
                
        # shuffling to define a null distribution
        if Nshuffle>0:
            self.null_distribution()
            
        def transform(self,N1,N2):
            import numpy as np
            try:
                X = self.loadings_
                N1_transf = np.dot(N1,X)
                N2_transf = np.dot(N2,X)
                self.N1_transformed_ = N1_transf
                self.N2_transformed_ = N2_transf
            except:
                print('Loadings not defined, you have to first fit the model')

#%% class for ratio cPCA
class ratio_cPCA():
    def __init__(self,Nshuffle=0,normalize_flag=True,alpha_null=0.975):
        
        self.Nshuffle = Nshuffle
        self.normalize_flag = normalize_flag
        self.alpha_null = alpha_null

    #%% new fit for ncPCA without using zassenhaus and just using union of basis
    def fit(self,N1,N2): 
 
    """method to perform ratio contrastive PCA FG/BG
    """
    import numpy as np
    import numpy.linalg as LA
    
    #parameters
    Nshuffle       = self.Nshuffle
    normalize_flag = self.normalize_flag
    
    #test that inputs are normalized
    if N2.shape[1] != N1.shape[1]:
        raise ValueError("N1 and N2 have different numbers of features")
    
    if normalize_flag:
        N1_temp = np.divide(stats.zscore(N1),LA.norm(stats.zscore(N1),axis=0))
        N2_temp = np.divide(stats.zscore(N2),LA.norm(stats.zscore(N2),axis=0))
        
        if np.sum(np.sum(np.square(N1_temp - N1)) > (0.01*np.square(N1_temp))):
            warnings.warn("N1 was not normalized properly - normalizing now")
            N1 = N1_temp
        
        if np.sum(np.sum(np.square(N2_temp - N2)) > (0.01*np.square(N2_temp))):
            warnings.warn("N2 was not normalized properly - normalizing now")
            N2 = N2_temp
    
    #covariance matrices
    N1N1 = np.dot(N1.T,N1)
    N2N2 = np.dot(N2.T,N2)
    
    sigma = np.dot(LA.pinv(N2N2),N1N1) #doing pseudo inverse to throw out eigenvalues that would blow this ratio to infinity
    
    # getting eigenvalues and eigenvectors
    w, v = LA.eig(sigma)
    eig_idx = np.argsort(w)[::-1]
    
    X = v[:,eig_idx]
    
    S_total = w[eig_idx]

    self.number_of_shared_basis = J.shape[1]
    self.loadings_ = X
    self.eigenvalues_ = S_total
    self.N1_scores_ = np.dot(N1,X)
    self.N2_scores_ = np.dot(N2,X)
    self.N1 = N1
    self.N2 = N2
            
    # shuffling to define a null distribution
    if Nshuffle>0:
        self.null_distribution()
        
    def transform(self,N1,N2):
        import numpy as np
        try:
            X = self.loadings_
            N1_transf = np.dot(N1,X)
            N2_transf = np.dot(N2,X)
            self.N1_transformed_ = N1_transf
            self.N2_transformed_ = N2_transf
        except:
            print('Loadings not defined, you have to first fit the model')
    

#%% making the class for normalized ncPCA - (FG-BG)/(BG)
class ncPCA():
    def __init__(self,Nshuffle = 0,normalize_flag = True,alpha_null=0.975):
        
        self.Nshuffle = Nshuffle
        self.normalize_flag = normalize_flag
        self.alpha_null = alpha_null

    #%% new fit for ncPCA without using zassenhaus and just using union of basis
    def fit(self,N1,N2): 
        
        """method fitting ncPCA
           (FG-BG)/(BG)
           Add more details here
        """
        
        #importing libraries
        import numpy as np
        from scipy import stats
        from scipy import linalg as LA
        import warnings
    
        #parameters
        Nshuffle       = self.Nshuffle
        normalize_flag = self.normalize_flag
        
        #test that inputs are normalized
        if N2.shape[1] != N1.shape[1]:
            raise ValueError("N1 and N2 have different numbers of features")
        
        if normalize_flag:
            N1_temp = np.divide(stats.zscore(N1),np.linalg.norm(stats.zscore(N1),axis=0))
            N2_temp = np.divide(stats.zscore(N2),np.linalg.norm(stats.zscore(N2),axis=0))
            
            if np.sum(np.sum(np.square(N1_temp - N1)) > (0.01*np.square(N1_temp))):
                warnings.warn("N1 was not normalized properly - normalizing now")
                N1 = N1_temp
            
            if np.sum(np.sum(np.square(N2_temp - N2)) > (0.01*np.square(N2_temp))):
                warnings.warn("N2 was not normalized properly - normalizing now")
                N2 = N2_temp
        
        #covariance matrices
        N1N1 = np.dot(N1.T,N1)
        N2N2 = np.dot(N2.T,N2)
        
        
        #SVD (or PCA) on N1
        _, S1, V1 = np.linalg.svd(N1N1, full_matrices = False)
        
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
            
            """I think the loop below can be discarded with this new method
            because the matrix are symetric then the eigenvectors should all be
            orthogonal? I'm not sure about that"""
            ######### Iteratively take out the ncPCs by deflating J
            n_basis = J.shape[1]
            Jnew = J
            for aa in np.arange(n_basis):
                B = LA.sqrtm(np.linalg.multi_dot((Jnew.T,N1N1,Jnew)))
                
                JBinv =  np.linalg.lstsq(Jnew.T, np.linalg.pinv(B))[0]
                
                # sort according to decreasing eigenvalue
                D,y = np.linalg.eig(np.linalg.multi_dot((JBinv.T,N2N2-N1N1,JBinv)))
                
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
            Stop = np.linalg.multi_dot((X.T,N2N2-N1N1,X))
            Sbot = np.linalg.multi_dot((X.T,N1N1,X))
            S_total = np.divide(np.diagonal(Stop),np.diagonal(Sbot))
            
            self.number_of_shared_basis = J.shape[1]
            self.loadings_ = X
            self.eigenvalues_ = S_total
            self.N1_scores_ = np.dot(N1,X)
            self.N2_scores_ = np.dot(N2,X)
            self.N1 = N1
            self.N2 = N2
            
            # shuffling to define a null distribution
            if Nshuffle>0:
                self.null_distribution()
        
    def transform(self,N1,N2):
        import numpy as np
        try:
            X = self.loadings_
            N1_transf = np.dot(N1,X)
            N2_transf = np.dot(N2,X)
            self.N1_transformed_ = N1_transf
            self.N2_transformed_ = N2_transf
        except:
            print('Loadings not defined, you have to first fit the model')

#%% making the class for normalized ncPCA - (FG-BG)/(FG+BG)
class index_ncPCA():
    
    def __init__(self,Nshuffle = 0,normalize_flag = True,alpha_null=0.975):
        
        self.Nshuffle = Nshuffle
        self.normalize_flag = normalize_flag
        self.alpha_null = alpha_null

    #%% new fit for ncPCA without using zassenhaus and just using union of basis
    def fit(self,N1,N2): #old method that was called ncPCA_orth
        
        """method fitting ncPCA
        Different from before, we get the union basis from PCA on FG+BG covariance, use the top PCs to continue with method
        """
        
        #importing libraries
        import numpy as np
        from scipy import stats
        from scipy import linalg as LA
        import warnings
    
        #parameters
        Nshuffle = self.Nshuffle
        normalize_flag = self.normalize_flag
        
        #test that inputs are normalized
        if N2.shape[1] != N1.shape[1]:
            raise ValueError("N1 and N2 have different numbers of features")
        
        if normalize_flag:
            N1_temp = np.divide(stats.zscore(N1),np.linalg.norm(stats.zscore(N1),axis=0))
            N2_temp = np.divide(stats.zscore(N2),np.linalg.norm(stats.zscore(N2),axis=0))
            
            if np.sum(np.sum(np.square(N1_temp - N1)) > (0.01*np.square(N1_temp))):
                warnings.warn("N1 was not normalized properly - normalizing now")
                N1 = N1_temp
            
            if np.sum(np.sum(np.square(N2_temp - N2)) > (0.01*np.square(N2_temp))):
                warnings.warn("N2 was not normalized properly - normalizing now")
                N2 = N2_temp
        
        #covariance matrices
        N1N1 = np.dot(N1.T,N1)
        N2N2 = np.dot(N2.T,N2)
        
        
        #SVD (or PCA) on N1 and N2
        _, S1, V1 = np.linalg.svd(N2N2+N1N1, full_matrices = False)
        
        # Picking based on rank (eps)
        rank = np.linalg.matrix_rank(N2N2+N1N1)

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
                B = LA.sqrtm(np.linalg.multi_dot((Jnew.T,N2N2+N1N1,Jnew)))
                
                JBinv =  np.linalg.lstsq(Jnew.T, np.linalg.pinv(B))[0]
                
                # sort according to decreasing eigenvalue
                D,y = np.linalg.eig(np.linalg.multi_dot((JBinv.T,N2N2-N1N1,JBinv)))
                
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
            Stop = np.linalg.multi_dot((X.T,N2N2-N1N1,X))
            Sbot = np.linalg.multi_dot((X.T,N2N2+N1N1,X))
            S_total = np.divide(np.diagonal(Stop),np.diagonal(Sbot))
            
            self.number_of_shared_basis = J.shape[1]
            self.loadings_ = X
            self.ncPCs_values_ = S_total
            self.N1_scores_ = np.dot(N1,X)
            self.N2_scores_ = np.dot(N2,X)
            self.N1 = N1
            self.N2 = N2
            
            # shuffling to define a null distribution
            if Nshuffle>0:
                self.null_distribution()
        
    def transform(self,N1,N2):
        import numpy as np
        try:
            X = self.loadings_
            N1_transf = np.dot(N1,X)
            N2_transf = np.dot(N2,X)
            self.N1_transformed_ = N1_transf
            self.N2_transformed_ = N2_transf
        except:
            print('Loadings not defined, you have to first fit the model')
            
    def null_distribution(self):
        import numpy as np
           
        #getting parameters
        Nshuffle = self.Nshuffle
        X = self.loadings_
        N1_org = self.N1 #original data
        N2_org = self.N2
        alpha_null = self.alpha_null
    
        S_null = [] #variable to save the null results
    
        for n in np.arange(Nshuffle):
    
            N1 = np.random.permutation(N1_org.T).T
            N2 = np.random.permutation(N2_org.T).T

            #covariance matrices
            N1N1 = np.dot(N1.T,N1)
            N2N2 = np.dot(N2.T,N2)

            # getting top and bottom eigenvalues
            Stop = np.linalg.multi_dot((X.T,N2N2-N1N1,X))
            Sbot = np.linalg.multi_dot((X.T,N2N2+N1N1,X))
            S_total = np.divide(np.diagonal(Stop),np.diagonal(Sbot))
            S_null.append(S_total)
            
        S_null = np.vstack(S_null).T
        S_null_sorted = np.sort(S_null,axis=1)
        
        #getting upper and bottom CI
        bot_CI = S_null_sorted[:,int(Nshuffle*(1-alpha_null))]
        top_CI = S_null_sorted[:,int(Nshuffle*alpha_null)]
        
        
        #finding significant top and bot ncPCs
        top_ncPCs = self.ncPCs_values_ > top_CI
        bot_ncPCs = self.ncPCs_values_ < bot_CI
        
        
        self.ncPCA_values_null_ = S_null_sorted
        
        self.top_ncPCs_num_ = np.sum(top_ncPCs)
        self.bot_ncPCs_num_ = np.sum(bot_ncPCs)
        
        self.top_ncPCs_idx = np.arange(self.top_ncPCs_num_)
        self.bot_ncPCs_idx = np.arange(start = X.shape[1]-self.bot_ncPCs_num_,stop = X.shape[1]+1)
        