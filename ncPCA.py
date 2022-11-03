# -*- coding: utf-8 -*-
"""
Created on Sat May  7 09:57:15 2022

@author: fermi
script to implement ncPCA and run some tests on it
"""

#%% should prepare this code as a class with methods and etc.
# Return an object with data projected on to the ncPCs
# loadings
# other stuff
# put option to normalize the data or not
#%% defining important functions

#function for fast reduced row echelon form
def frref(A, TOL=None, TYPE=''):
    '''
    %FRREF   Fast reduced row echelon form.
    %   R = FRREF(A) produces the reduced row echelon form of A.
    %   [R,jb] = FRREF(A,TOL) uses the given tolerance in the rank tests.
    %   [R,jb] = FRREF(A,TOL,TYPE) forces frref calculation using the algorithm
    %   for full (TYPE='f') or sparse (TYPE='s') matrices.
    %
    %
    %   Description:
    %   For full matrices, the algorithm is based on the vectorization of MATLAB's
    %   RREF function. A typical speed-up range is about 2-4 times of
    %   the MATLAB's RREF function. However, the actual speed-up depends on the
    %   size of A. The speed-up is quite considerable if the number of columns in
    %   A is considerably larger than the number of its rows or when A is not dense.
    %
    %   For sparse matrices, the algorithm ignores the TOL value and uses sparse
    %   QR to compute the rref form, improving the speed by a few orders of
    %   magnitude.
    %
    %   Authors: Armin Ataei-Esfahani (2008)
    %            Ashish Myles (2012)
    #            Snehesh Shrestha (2020)
    %
    %   Revisions:
    %   25-Sep-2008   Created Function
    %   21-Nov-2012   Added faster algorithm for sparse matrices
    #   30-June-2020  Ported to python. TODO: Only do_full implemented. The remaining of the function. See frref_orig below.
    '''
    
    import numpy as np
    from scipy.sparse import isspmatrix  # ,csr_matrix

    m = np.shape(A)[0]
    n = np.shape(A)[1]

    # Process Arguments
    # ----------------------------------------------------------
    # TYPE -- Sparce (s) or non-Sparce (Full, f)
    if TYPE == '':   # set TYPE if sparse or not
        if isspmatrix(A):
            TYPE = 's'
        else:
            TYPE = 'f'
    else:   # Set type
        if not type(TYPE) is str or len(TYPE) > 1:  # Check valid type
            print('Unknown matrix TYPE! Use "f" for full and "s" for sparse matrices.')
            exit()

        TYPE = TYPE.lower()
        if not TYPE == 'f' and not TYPE == 's':
            print(
                'Unknown matrix TYPE! Use ''f'' for full and ''s'' for sparse matrices.')
            exit()

    if TYPE=='f':
        # TOLERENCE
        # % Compute the default tolerance if none was provided.
        if TOL is None:
            # Prior commit had TOL default to 1e-6
            # TOL = max(m,n)*eps(class(A))*norm(A,'inf')
            TOL = max(m, n)*np.spacing(type(A)(1))*np.linalg.norm(A, np.inf)

    # Non-Sparse
    # ----------------------------------------------------------
    if not isspmatrix(A) or TYPE == 'f':
        # % Loop over the entire matrix.
        i = 0
        j = 0
        jb = []

        while (i < m) and (j < n):
            # % Find value (p) and index (k) of largest element in the remainder of column j.
            abscol = np.array(np.abs(A[i:m, j]))
            p = np.max(abscol)
            k = np.argmax(abscol, axis=0)
            if np.ndim(k) > 1:
                k = k[0]
            else:
                k = int(k)

            k = k+i  # -1 #python zero index, not needed

            if p <= TOL:
                # % The column is negligible, zero it out.
                A[i:m, j] = 0  # %(faster for sparse) %zeros(m-i+1,1);
                j += 1
            else:
                # % Remember column index
                jb.append(j)

                # % Swap i-th and k-th rows.
                A = np.array(A)
                A[[i, k], j:n] = A[[k, i], j:n]

                # % Divide the pivot row by the pivot element.
                Ai = np.nan_to_num(A[i, j:n] / A[i, j])
                Ai = np.matrix(Ai).T.T

                # % Subtract multiples of the pivot row from all the other rows.
                A[:, j:n] = A[:, j:n] - np.dot(A[:, [j]], Ai)
                A[i, j:n] = Ai
                i += 1
                j += 1

        return A, jb

    # Sparse
    # ----------------------------------------------------------
    else:
        A = np.array(A.toarray())
        return frref(A, TYPE='f')

        # # TODO: QR-decomposition of a Sparse matrix is not so simple in Python -- still need to figure out a solution
        # # % Non-pivoted Q-less QR decomposition computed by Matlab actually
        # # % produces the right structure (similar to rref) to identify independent
        # # % columns.
        # R = numpy.linalg.qr(A)

        # # % i_dep = pivot columns = dependent variables
        # # %       = left-most non-zero column (if any) in each row
        # # % indep_rows (binary vector) = non-zero rows of R
        # [indep_rows, i_dep] = np.max(R ~= 0, [], 2)     # TODO
        # indep_rows = full[indep_rows]; # % probably more efficient
        # i_dep = i_dep[indep_rows]
        # i_indep = setdiff[1:n, i_dep]

        # # % solve R(indep_rows, i_dep) x = R(indep_rows, i_indep)
        # # %   to eliminate all the i_dep columns
        # # %   (i.e. we want F(indep_rows, i_dep) = Identity)
        # F = sparse([],[],[], m, n)
        # F[indep_rows, i_indep] = R[indep_rows, i_dep] \ R[indep_rows, i_indep]
        # F[indep_rows, i_dep] = speye(length(i_dep))

        # # % result
        # A = F
        # jb = i_dep

        # return A, jb

#code for cPCA, useful for comparison and test
def cPCA(background,foreground,alpha=1):
    #code to do cPCA for comparison, this return loadings
    import numpy as np
    import numpy.linalg as LA
    
    #calculating covariance of the data, it's assumed the data is centered and scaled
    bg_cov = background.T.dot(background)
    fg_cov = foreground.T.dot(foreground)
    
    sigma = fg_cov - alpha*bg_cov
    w, v = LA.eig(sigma)
    eig_idx = np.argsort(w)
    
    cPCs = v[:,eig_idx]
    return cPCs

# old ncPCA code    
def ncPCA_old(self,N1,N2):
    Nshuffle = self.Nshuffle
    normalize_flag = self.normalize_flag
    """function [X,S_total] = ncPCA(Ns, Nw, Nshuffle)
    %
    % This function does normalized contrastive PCA (nvPCA) on binned spike trains to
    % get the WS index (W-S)/(W+S) <--- this extracts dimensions that are
    % maximally present in wakefulness but not sleep.
    %
    % INPUTS
    % Ns        -- (t1 x n) matrix of binned spike trains during sleep
    % Nw        -- (t2 x n) matrix of binned spike trains during awake behavior
    % Nshuffle  -- (optional) number of times to shuffle for null distribution
    %
    % OUTPUTS
    % B        -- maxSPV scores (temporal eigenvectors)
    % S        -- amount of variance captured for sleep, awake, sleep_shuf, awake_shuf
    % X        -- maxSPV loadings for both awake and asleep data
    % info     -- extra info
    %
    % Note: Ns and Nw should both be normalized (as in normalize(zscore(Ns), 'norm'))
    %
    % Analogous to SVD, which decomposes data matrix N into (U * S * T'), this
    % function decomposes N into (B * S * X') where B is the temporal
    % eigenvectors (or PC scores), S is a diagonal matrix representing amount
    % of variance captured, and X is the PC loadings across cells. Instead of
    % maximizing var(N) in each dimension, this ncPCA finds the spPCs X that maximize
    %
    %            (x' * (Nw'*Nw - Ns'*Ns) * x) / (x' * (Nw'*Nw + Ns'*Ns) * x)
    %
    % where Ns is asleep data and Nw is awake data. Because Ns and Nw may not
    % be full rank, X is calculated in a subspace spanned by PCs accounting for
    % 99.5% of the variance.
    %
    % Luke Sjulson, 2021-11-04 
    """
    
    #importing libraries
    import numpy as np
    from scipy import stats
    from scipy import linalg as LA
    import warnings

    #parameters
    cutoff = 0.995 #keeping this much variance with PCA
    
    #test that inputs are normalized
    if N2.shape[1] != N1.shape[1]:
        raise ValueError("N1 and N2 have different numbers of features")
    
    if normalize:
        N1_temp = np.divide(stats.zscore(N1),np.linalg.norm(stats.zscore(N1),axis=0))
        N2_temp = np.divide(stats.zscore(N2),np.linalg.norm(stats.zscore(N2),axis=0))
        
        if np.sum(np.sum(np.square(N1_temp - N1)) > (0.01*np.square(N1_temp))):
            warnings.warn("N1 was not normalized properly - normalizing now")
            N1 = N1_temp
        
        if np.sum(np.sum(np.square(N2_temp - N2)) > (0.01*np.square(N2_temp))):
            warnings.warn("N2 was not normalized properly - normalizing now")
            N2 = N2_temp
    
    #SVD (or PCA) on N1 and N2
    _,S1,V1 = np.linalg.svd(N1,full_matrices = False)
    _,S2,V2 = np.linalg.svd(N2,full_matrices = False)
    
    # discard PCs that cumulatively account for less than 1% of variance, i.e.
    # rank-deficient dimensions
    S1_diagonal = S1
    S2_diagonal = S2
    
    #cumulative variance
    cumvar_1 = np.divide(np.cumsum(S1_diagonal),np.sum(S1_diagonal))
    cumvar_2 = np.divide(np.cumsum(S2_diagonal),np.sum(S2_diagonal))
    
    #picking how many PCs to keep
    max_1 = np.where(cumvar_1 < cutoff)
    max_2 = np.where(cumvar_2 < cutoff)
    
    # Zassenhaus algorithm to find shared basis for N1 and N2
    # https://en.wikipedia.org/wiki/Zassenhaus_algorithm
    V1_hat = V1[max_1[0],:];
    V2_hat = V2[max_2[0],:];
    
    N_dim = V1.shape[1]
    V1_cat = np.concatenate((V1_hat,V1_hat),axis=1)
    V2_cat = np.concatenate((V2_hat,np.zeros(V2_hat.shape)),axis=1)
    zassmat = np.concatenate((V1_cat,V2_cat),axis=0)
    #it might be necessary to check the type of array (float vs int)!!
    zassmat_rref,_ = frref(zassmat)
    
    basis_idx = 0
    basis_mat = [];
    for idx in np.arange(np.shape(zassmat_rref)[0]):
        if np.all(zassmat_rref[idx,:N_dim] == 0): #this is a basis vector of intersection
            #basis_mat = np.concatenate((basis_mat,zassmat_rref[idx,N_dim:].T),axis=1)
            basis_mat.append(zassmat_rref[idx,N_dim:].T)
            basis_idx += 1
        else:
            basis_mat.append(zassmat_rref[idx,:N_dim].T)
            basis_idx += 1
    
    basis_mat2 = np.array(basis_mat).T
    
    J = LA.orth(basis_mat2) #orthonormal shared basis for Ns and Nw
    k = J.shape[1]
    
    ## Calculating ncPCA below
    
    #covariance matrices
    N1N1 = np.dot(N1.T,N1)
    N2N2 = np.dot(N2.T,N2)
    
    ######### Try iteratively reduce
    B = LA.sqrtm(np.linalg.multi_dot((J.T,N2N2+N1N1,J)))
    
    #JBinv = np.linalg.lstsq(np.linalg.inv(J),np.linalg.inv(B))
    #JBinv,_,_,_ = np.linalg.lstsq(J.T,np.linalg.inv(B))
    #JBinv = np.linalg.solve(J.T,B.T)
    #JBinv = np.dot(J,np.linalg.pinv(B))
    
    JBinv =  np.linalg.lstsq(J.T, np.linalg.pinv(B))[0]
    
    # sort according to decreasing eigenvalue
    D,y = np.linalg.eig(np.linalg.multi_dot((JBinv.T,N2N2-N1N1,JBinv)))
    
    Y = y[:,np.flip(np.argsort(D))]
    
    X_temp = np.dot(JBinv,Y);
    X = X_temp/np.linalg.norm(X_temp,axis=0)
    
    ##########
    
    # getting top and bottom eigenvalues
    Stop = np.linalg.multi_dot((X.T,N2N2-N1N1,X))
    Sbot = np.linalg.multi_dot((X.T,N2N2+N1N1,X))
    S_total = np.divide(np.diagonal(Stop),np.diagonal(Sbot))
    
    
    #write shuffling later
    
    return X,S_total


#%% starting the main class here
class ncPCA():
    
    def __init__(self,Nshuffle = 0,normalize_flag = True,basis_type = 'all',alpha_null=0.975):
        """ 
        Basis type can be 'all', 'union','intersect'
        
        """
        from numpy import char as ch
        
        #checking if user gave the correct basis type
        if ( (ch.lower(basis_type) == 'all') | (ch.lower(basis_type) == 'union') |
            (ch.lower(basis_type) == 'intersect') ):
            basis_type = ch.lower(basis_type)
        else:
            raise ValueError("Basis type entered is invalid, it can only be: 'all','union','intersect'")
        
        self.Nshuffle = Nshuffle
        self.normalize_flag = normalize_flag
        self.basis_type = basis_type
        self.alpha_null = alpha_null
        self.cutoff=0.995
        
    #%% write a function to generate data that can only be captured by ncPCA
    def generate_dataset(self,nsamples=10000,ntargets=100,high_var_rank=2,low_var_rank=2):
        high_rank_sigma = 10
        noise_sigma = 1
        low_rank_sigma = 2
        import numpy as np
        
        W_hv = np.random.randn(high_var_rank,ntargets);
        W_lv = np.random.randn(low_var_rank,ntargets);
        
        N1 = np.dot(np.random.randn(nsamples,high_var_rank),W_hv)*high_rank_sigma #high var activiy
        + np.random.randn(nsamples,ntargets)*noise_sigma # noise
        + np.dot(np.random.randn(nsamples,low_var_rank),W_lv)*low_rank_sigma #low var activity
        
        N2 = np.dot(np.random.randn(nsamples,high_var_rank),W_hv)*high_rank_sigma*0.5 #increase in high var activiy
        + np.random.randn(nsamples,ntargets)*noise_sigma # noise
        + np.dot(np.random.randn(nsamples,low_var_rank),W_lv)*low_rank_sigma*5 #increase in low var activity
        
        return N1,N2,W_hv,W_lv
    
    #%% new and orthogonal ncPCA
    def fit(self,N1,N2): #old method that was called ncPCA_orth
        
        """function [X,S_total] = ncPCA(Ns, Nw, Nshuffle)
        %
        % This function does normalized contrastive PCA (nvPCA) on binned spike trains to
        % get the WS index (W-S)/(W+S) <--- this extracts dimensions that are
        % maximally present in wakefulness but not sleep. This function will guarantee orthogonal
        % dimensions
        %
        % INPUTS
        % Ns        -- (t1 x n) matrix of binned spike trains during sleep
        % Nw        -- (t2 x n) matrix of binned spike trains during awake behavior
        % Nshuffle  -- (optional) number of times to shuffle for null distribution
        %
        % OUTPUTS
        % B        -- maxSPV scores (temporal eigenvectors)
        % S        -- amount of variance captured for sleep, awake, sleep_shuf, awake_shuf
        % X        -- maxSPV loadings for both awake and asleep data
        % info     -- extra info
        %
        % Note: Ns and Nw should both be normalized (as in normalize(zscore(Ns), 'norm'))
        %
        % Analogous to SVD, which decomposes data matrix N into (U * S * T'), this
        % function decomposes N into (B * S * X') where B is the temporal
        % eigenvectors (or PC scores), S is a diagonal matrix representing amount
        % of variance captured, and X is the PC loadings across cells. Instead of
        % maximizing var(N) in each dimension, this ncPCA finds the spPCs X that maximize
        %
        %            (x' * (Nw'*Nw - Ns'*Ns) * x) / (x' * (Nw'*Nw + Ns'*Ns) * x)
        %
        % where Ns is asleep data and Nw is awake data. Because Ns and Nw may not
        % be full rank, X is calculated in a subspace spanned by PCs accounting for
        % 99.5% of the variance.
        %
        % Luke Sjulson, 2021-11-04 
        """
        
        #importing libraries
        import numpy as np
        from scipy import stats
        from scipy import linalg as LA
        import warnings
    
        #parameters
        Nshuffle = self.Nshuffle
        normalize_flag = self.normalize_flag
        cutoff = self.cutoff #keeping this much variance with PCA
        basis_type = self.basis_type
        
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
        
        #SVD (or PCA) on N1 and N2
        _,S1,V1 = np.linalg.svd(N1,full_matrices = False)
        _,S2,V2 = np.linalg.svd(N2,full_matrices = False)
        
        # discard PCs that cumulatively account for less than 1% of variance, i.e.
        # rank-deficient dimensions
        S1_diagonal = S1
        S2_diagonal = S2
        
        #cumulative variance
        cumvar_1 = np.divide(np.cumsum(S1_diagonal),np.sum(S1_diagonal))
        cumvar_2 = np.divide(np.cumsum(S2_diagonal),np.sum(S2_diagonal))
        
        #picking how many PCs to keep
        max_1 = np.where(cumvar_1 < cutoff)
        max_2 = np.where(cumvar_2 < cutoff)
        
        # Zassenhaus algorithm to find shared basis for N1 and N2
        # https://en.wikipedia.org/wiki/Zassenhaus_algorithm
        V1_hat = V1[max_1[0],:];
        V2_hat = V2[max_2[0],:];
        
        #HAVE TO MAKE SURE THIS IS GETTING SUM AND INTERCEPT (JUST GETTING INTERCEPT)
        N_dim = V1.shape[1]
        V1_cat = np.concatenate((V1_hat,V1_hat),axis=1)
        V2_cat = np.concatenate((V2_hat,np.zeros(V2_hat.shape)),axis=1)
        zassmat = np.concatenate((V1_cat,V2_cat),axis=0)
        #it might be necessary to check the type of array (float vs int)!!
        zassmat_rref,_ = frref(zassmat)
        
        basis_idx = 0
        basis_mat = [];
        
        if basis_type == 'all':
                for idx in np.arange(np.shape(zassmat_rref)[0]):
                    if np.all(zassmat_rref[idx,:N_dim] == 0): #this is a basis vector of intersection
                        basis_mat.append(zassmat_rref[idx,N_dim:].T)
                        basis_idx += 1
                    else: #otherwise is a basis function from union
                        basis_mat.append(zassmat_rref[idx,:N_dim].T)
                        basis_idx += 1
        elif basis_type == 'union':
                for idx in np.arange(np.shape(zassmat_rref)[0]):
                    if np.logical_not(np.all(zassmat_rref[idx,:N_dim] == 0)): #this is a basis vector of union
                        basis_mat.append(zassmat_rref[idx,:N_dim].T)
                        basis_idx += 1
        elif basis_type == 'intersect':
                for idx in np.arange(np.shape(zassmat_rref)[0]):
                    if np.all(zassmat_rref[idx,:N_dim] == 0): #this is a basis vector of intersection
                        basis_mat.append(zassmat_rref[idx,N_dim:].T)
                        basis_idx += 1

                        
        basis_mat2 = np.array(basis_mat).T
        
        if basis_mat2.size==0:
            raise ValueError("No shared basis found between N1 and N2")
        
        J = LA.orth(basis_mat2) #orthonormal shared basis for Ns and Nw
        k = J.shape[1]
        
        ## Calculating ncPCA below
        
        #covariance matrices
        N1N1 = np.dot(N1.T,N1)
        N2N2 = np.dot(N2.T,N2)
        
        
        ######### Iteratively take out the ncPCs by deflating J
        n_basis = J.shape[1]
        Jnew = J
        for aa in np.arange(n_basis):
            B = LA.sqrtm(np.linalg.multi_dot((Jnew.T,N2N2+N1N1,Jnew)))
            
            #JBinv = np.linalg.lstsq(np.linalg.inv(J),np.linalg.inv(B))
            #JBinv,_,_,_ = np.linalg.lstsq(J.T,np.linalg.inv(B))
            #JBinv = np.linalg.solve(J.T,B.T)
            #JBinv = np.dot(J,np.linalg.pinv(B))
            
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
        
    # def null_distribution(self):
    #     import numpy as np
    #     from scipy import linalg as LA
        
    #     #getting parameters
    #     Nshuffle = self.Nshuffle
    #     basis_type = self.basis_type
    #     N1_org = self.N1 #original data
    #     N2_org = self.N2
        
    #     S_null = [] #variable to save the null results
        
    #     for n in np.arange(Nshuffle):
            
    #         N1 = np.random.permutation(N1_org.T).T
    #         N2 = np.random.permutation(N2_org.T).T
    #         #SVD (or PCA) on N1 and N2
    #         _,_,V1_hat = np.linalg.svd(N1,full_matrices = False)
    #         _,_,V2_hat = np.linalg.svd(N2,full_matrices = False)
    
            
    #         # Zassenhaus algorithm to find shared basis for N1 and N2
    #         #HAVE TO MAKE SURE THIS IS GETTING SUM AND INTERCEPT (JUST GETTING INTERCEPT)
    #         N_dim = V1_hat.shape[1]
    #         V1_cat = np.concatenate((V1_hat,V1_hat),axis=1)
    #         V2_cat = np.concatenate((V2_hat,np.zeros(V2_hat.shape)),axis=1)
    #         zassmat = np.concatenate((V1_cat,V2_cat),axis=0)
    #         #it might be necessary to check the type of array (float vs int)!!
    #         zassmat_rref,_ = frref(zassmat)
            
    #         basis_idx = 0
    #         basis_mat = [];
            
    #         if basis_type == 'all':
    #                 for idx in np.arange(np.shape(zassmat_rref)[0]):
    #                     if np.all(zassmat_rref[idx,:N_dim] == 0): #this is a basis vector of intersection
    #                         basis_mat.append(zassmat_rref[idx,N_dim:].T)
    #                         basis_idx += 1
    #                     else: #otherwise is a basis function from union
    #                         basis_mat.append(zassmat_rref[idx,:N_dim].T)
    #                         basis_idx += 1
    #         elif basis_type == 'union':
    #                 for idx in np.arange(np.shape(zassmat_rref)[0]):
    #                     if np.logical_not(np.all(zassmat_rref[idx,:N_dim] == 0)): #this is a basis vector of union
    #                         basis_mat.append(zassmat_rref[idx,:N_dim].T)
    #                         basis_idx += 1
    #         elif basis_type == 'intersect':
    #                 for idx in np.arange(np.shape(zassmat_rref)[0]):
    #                     if np.all(zassmat_rref[idx,:N_dim] == 0): #this is a basis vector of intersection
    #                         basis_mat.append(zassmat_rref[idx,N_dim:].T)
    #                         basis_idx += 1
    
                            
    #         basis_mat2 = np.array(basis_mat).T
            
    #         if basis_mat2.size==0:
    #             raise ValueError("No shared basis found between N1 and N2")
            
    #         J = LA.orth(basis_mat2) #orthonormal shared basis for Ns and Nw
    #         k = J.shape[1]
            
    #         ## Calculating ncPCA below
            
    #         #covariance matrices
    #         N1N1 = np.dot(N1.T,N1)
    #         N2N2 = np.dot(N2.T,N2)
            
            
    #         ######### Iteratively take out the ncPCs by deflating J
    #         n_basis = J.shape[1]
    #         Jnew = J
    #         for aa in np.arange(n_basis):
    #             B = LA.sqrtm(np.linalg.multi_dot((Jnew.T,N2N2+N1N1,Jnew)))
                
    #             #JBinv = np.linalg.lstsq(np.linalg.inv(J),np.linalg.inv(B))
    #             #JBinv,_,_,_ = np.linalg.lstsq(J.T,np.linalg.inv(B))
    #             #JBinv = np.linalg.solve(J.T,B.T)
    #             #JBinv = np.dot(J,np.linalg.pinv(B))
                
    #             JBinv =  np.linalg.lstsq(Jnew.T, np.linalg.pinv(B))[0]
                
    #             # sort according to decreasing eigenvalue
    #             D,y = np.linalg.eig(np.linalg.multi_dot((JBinv.T,N2N2-N1N1,JBinv)))
                
    #             Y = y[:,np.flip(np.argsort(D))]
                
    #             X_temp = np.dot(JBinv,Y[:,0]);
                
    #             if aa == 0:
    #                 X = X_temp/np.linalg.norm(X_temp,axis=0)
    #             else:
    #                 temp_norm_ldgs = X_temp/np.linalg.norm(X_temp,axis=0)
    #                 X = np.column_stack((X,temp_norm_ldgs))
                
    #             #deflating J
    #             gamma = np.dot(np.linalg.pinv(B),Y[:,0])
    #             gamma = gamma/np.linalg.norm(gamma) #normalizing by the norm
    #             gamma_outer = np.outer(gamma,gamma.T)
    #             J_reduced = Jnew - np.dot(Jnew,gamma_outer)
    #             Jnew = LA.orth(J_reduced)
    #             #Jnew = J_reduced
    #         ##########
            
    #         # getting top and bottom eigenvalues
    #         Stop = np.linalg.multi_dot((X.T,N2N2-N1N1,X))
    #         Sbot = np.linalg.multi_dot((X.T,N2N2+N1N1,X))
    #         S_total = np.divide(np.diagonal(Stop),np.diagonal(Sbot))
    #         S_null.append(S_total)
        
    #     S_null = np.vstack(S_null).T
    #     S_null_sorted = np.sort(S_null,axis=1)
    #     self.ncPCA_values_null_ = S_null_sorted
        
        
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