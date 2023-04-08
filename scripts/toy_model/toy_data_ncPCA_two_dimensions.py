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
import matplotlib.pyplot as plt
from scipy.linalg import orth
from scipy.stats import zscore
import sys
import seaborn as sns

repo_dir = "C:\\Users\\fermi\\Documents\\GitHub\\normalized_contrastive_PCA" #repository dir
sys.path.append(repo_dir)
from ncPCA import ncPCA
from ncPCA import cPCA
from project_utils import cosine_similarity_multiple_vectors

sns.set_style("whitegrid")
sns.set_context("talk")
#variables
# N_times    = 1000 #number of observations
N_features = 30 #number of features (and PCs, data is full rank)
pc_change  = [0,27] #pcs that are going to be changed in variance

#%% generating toy data with linear decay

#background data
temp_S = np.linspace(1,stop=N_features,num=N_features) #variance of background activity, decays in 1/f
# S_bg   = 1/temp_S
S_bg = np.linspace(N_features,stop=1,num=N_features)/N_features

#foreground data, where we want to compare the change to background
#delta_var = np.random.randn(N_features)/100 #how much variance to vary by default, we are doing a normali distribution of 1% change in the SD
#S_fg      = S_bg*(1+delta_var)
S_fg = S_bg.copy()+0.02

#injecting variance in the data
S_fg[pc_change[0]] = S_fg[pc_change[0]]*1.05;
S_fg[pc_change[1]] = S_fg[pc_change[1]]*1.35;

#S_bg[pc_num] = 0

# generating random orthogonal loadings
V = orth(np.random.randn(N_features,N_features))

plt.figure()
plt.plot(np.arange(1,N_features+1),S_bg,':',label='background dataset')
plt.plot(np.arange(1,N_features+1),S_fg,'--',label='foreground dataset')
plt.legend()
plt.xlabel('PCs')
plt.ylabel('Eigenvalues')
plt.tight_layout()
#figure;plot((S_bg-S_fg)/(S_bg+S_fg))
#%% reconstruct data

data_bg = np.linalg.multi_dot((V,np.diag(S_bg),V.T));
data_fg = np.linalg.multi_dot((V,np.diag(S_fg),V.T));


#%% now run cPCA and ncPCA

# cPCs = cPCA(data_bg,data_fg,alpha=1)[:,0]

cPCs_all,_,_ = cPCA(data_bg,data_fg,n_components=len(V),alpha=1.1)
"""
mdl = ncPCA(basis_type='union',normalize_flag=False)
mdl.fit(data_bg, data_fg)
ncPCs = mdl.loadings_[:,0]
ncPCs_all = mdl.loadings_
"""
#plot(np.corrcoef(cPCs_all.T,V[:,pc_num])[-1,:-1])
#plot(np.corrcoef(ncPCs_all.T,V[:,pc_num])[-1,:-1])
#% get the correlation of cPCs 1 to the modeled

# cPC1_corr = np.corrcoef(V.T,cPCs_all)[-1,:len(cPCs_all)]
# cPC2_corr = np.corrcoef(V.T,cPCs_all)[-2,:len(cPCs_all)]
# ncPCs_corr = np.corrcoef(V.T,ncPCs)[-1,:len(ncPCs)]

cPC1_cossim = np.abs(cosine_similarity_multiple_vectors(V, cPCs_all[:,0]))
cPC2_cossim = np.abs(cosine_similarity_multiple_vectors(V, cPCs_all[:,1]))

plt.figure()
plt.plot(cPC1_cossim,color='red',label='1st cPC')
plt.plot(cPC2_cossim,'--',color='green',label='2nd cPC')
plt.legend()
plt.xlabel('PCs')
plt.ylabel('Cos. similarity')
plt.title('Modeled PCs captured by cPCA')
plt.tight_layout()

#%% make plot of multiple alpha and the distortion


alphas_vec = np.linspace(0.8,1.3,num=25)
cPC1st = []
cPC2nd = []
for alpha in alphas_vec:
    cPCs_all,w,eigidx = cPCA(data_bg,data_fg,alpha=alpha,n_components=len(V))
    cPCs_cossim = np.abs(cosine_similarity_multiple_vectors(V, cPCs_all[:,0]))
    cPC1st.append(np.argmax(cPCs_cossim))
    cPCs_cossim = np.abs(cosine_similarity_multiple_vectors(V, cPCs_all[:,1]))
    cPC2nd.append(np.argmax(cPCs_cossim))
    
figure()
scatter(alphas_vec,cPC1st,color='red',alpha=0.5,label='1st cPC')
scatter(alphas_vec,cPC2nd,color='green',alpha=0.5,label='2nd cPC')
plot(alphas_vec,np.ones(len(alphas_vec))*pc_change[0],'k--',alpha=0.4,label='dimensions of interest')
plot(alphas_vec,np.ones(len(alphas_vec))*pc_change[1],'k--',alpha=0.4)
xlabel('alpha values')
ylabel('Recovered dim. from model')
legend()
tight_layout()

#%% plot subtracting the eigenspectrum



sns.set_style("ticks")
SFG = np.tile(S_fg,len(alphas_vec))
SFG.resize((len(alphas_vec),N_features))

SBG = np.tile(S_bg,len(alphas_vec))
SBG.resize((len(alphas_vec),N_features))

newalpha = np.tile(alphas_vec,N_features)
newalpha.resize((N_features,len(alphas_vec)))
figure()
cpc_eq = SFG-np.multiply(newalpha.T,SBG)
plot(cpc_eq[np.arange(0,25,5),:].T,color='k')
