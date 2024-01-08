# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:36:29 2023

@author: eliezyer

script to make the toy model to explain what are we solving by using ncPCA, that is
the relative incremental of a dimensions compared to a background dataset

Loadings of each is exactly the same, i.e., U and V are identical
"""

# %% importing libraries and defining main variables

#libraries
import numpy as np
from matplotlib.pyplot import *
from scipy.linalg import orth
from scipy.stats import zscore
import sys

repo_dir = "C:\\Users\\fermi\\Documents\\GitHub\\normalized_contrastive_PCA" #repository dir
sys.path.append(repo_dir)
from ncPCA import ncPCA
from ncPCA import cPCA


#variables
N_features = 100 #number of features (and PCs, data is full rank)
pc_num = 80 #pc that is going to be altered more than the rest

#%% generating toy data

#background data
temp_S = np.linspace(1,stop=N_features,num=N_features) #variance of background activity, decays in 1/f
S_bg   = 1/temp_S

#foreground data, where we want to compare the change to background
delta_var = np.random.randn(N_features)/100 #how much variance to vary by default, we are doing a normali distribution of 1% change in the SD
S_fg      = S_bg*(1+delta_var)

S_fg[pc_num] = S_fg[pc_num]*(1+0.1)

# generating random orthogonal loadings
U = orth(np.random.randn(N_features,N_features))
V = orth(np.random.randn(N_features,N_features))

#plot(S_bg);plot(S_fg)
#figure;plot((S_bg-S_fg)/(S_bg+S_fg))
#%% reconstruct data

data_bg = np.linalg.multi_dot((V,np.diag(S_bg),V.T));
data_fg = np.linalg.multi_dot((V,np.diag(S_fg),V.T));


#%% now run cPCA and ncPCA

cPCs = cPCA(data_bg,data_fg,alpha=1)[:,0]

cPCs_all = cPCA(data_bg,data_fg,n_components=len(V))

mdl = ncPCA(basis_type='intersect',normalize_flag=False)
mdl.fit(data_bg, data_fg)
ncPCs = mdl.loadings_[:,0]
ncPCs_all = mdl.loadings_

#plot(np.corrcoef(cPCs_all.T,V[:,pc_num])[-1,:-1])
#plot(np.corrcoef(ncPCs_all.T,V[:,pc_num])[-1,:-1])
#%% get the correlation of cPCs 1 to the modeled

cPCs_corr = np.corrcoef(V.T,cPCs)[-1,:len(cPCs)]
ncPCs_corr = np.corrcoef(V.T,ncPCs)[-1,:len(ncPCs)]

figure;plot(cPCs_corr);plot(ncPCs_corr,'--')