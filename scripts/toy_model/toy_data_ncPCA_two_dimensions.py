# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 22:29:29 2023


@author: eliezyer

script to make the toy model for ncPCA with two big PCs

Loadings of each is exactly the same, i.e., U and V are identical
"""

# %% importing libraries and defining main variables

#libraries
import numpy as np
from matplotlib.pyplot import *
from scipy.linalg import orth
from scipy.stats import zscore
import sys
import seaborn as sns

repo_dir = "C:\\Users\\fermi\\Documents\\GitHub\\normalized_contrastive_PCA" #repository dir
sys.path.append(repo_dir)
from ncPCA import ncPCA
from ncPCA import cPCA

sns.set_style("whitegrid")
sns.set_context("talk")
#variables
N_times    = 1000 #number of observations
N_features = 100 #number of features (and PCs, data is full rank)
pc_num = 80 #pc that is going to be altered more than the rest

#%% generating toy data

#background data
temp_S = np.linspace(1,stop=N_features,num=N_features) #variance of background activity, decays in 1/f
S_bg   = 1/temp_S

#foreground data, where we want to compare the change to background
#delta_var = np.random.randn(N_features)/100 #how much variance to vary by default, we are doing a normali distribution of 1% change in the SD
#S_fg      = S_bg*(1+delta_var)
S_fg = S_bg.copy()
S_fg[0] = S_bg[0]*1.09;
S_fg[1] = S_bg[1]*1.05;

S_fg[pc_num] = S_fg[pc_num]*(1+0.1)
#S_bg[pc_num] = 0

# generating random orthogonal loadings
U = orth(np.random.randn(N_times,N_features))
V = orth(np.random.randn(N_features,N_features))

#figure(figsize=(10,10))
loglog(np.arange(1,101),S_bg,label='background dataset')
loglog(np.arange(1,101),S_fg,label='foreground dataset')
legend()
xlabel('PCs')
ylabel('Eigenvalues')
tight_layout()
#figure;plot((S_bg-S_fg)/(S_bg+S_fg))
#%% reconstruct data

data_bg = np.linalg.multi_dot((U,np.diag(S_bg),V.T));
data_fg = np.linalg.multi_dot((U,np.diag(S_fg),V.T));


#%% now run cPCA and ncPCA

cPCs = cPCA(data_bg,data_fg,alpha=1.2)[:,0]

cPCs_all = cPCA(data_bg,data_fg,n_components=len(V))
#"""
mdl = ncPCA(basis_type='union',normalize_flag=False)
mdl.fit(data_bg, data_fg)
ncPCs = mdl.loadings_[:,0]
ncPCs_all = mdl.loadings_
#"""
#plot(np.corrcoef(cPCs_all.T,V[:,pc_num])[-1,:-1])
#plot(np.corrcoef(ncPCs_all.T,V[:,pc_num])[-1,:-1])
#% get the correlation of cPCs 1 to the modeled

cPCs_corr = np.corrcoef(V.T,cPCs)[-1,:len(cPCs)]
ncPCs_corr = np.corrcoef(V.T,ncPCs)[-1,:len(ncPCs)]

figure()
plot(cPCs_corr,label='cPCs')
plot(ncPCs_corr,'--',label='ncPCs')
legend()
xlabel('PCs')
ylabel('Correlation with modeled vectors')
tight_layout()