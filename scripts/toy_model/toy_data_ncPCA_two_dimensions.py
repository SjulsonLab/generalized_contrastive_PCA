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

repo_dir = "C:\\Users\\fermi\\Documents\\GitHub\\normalized_contrastive_PCA\\" #repository dir
sys.path.append(repo_dir)
from contrastive_methods import cPCA
from ncPCA_project_utils import cosine_similarity_multiple_vectors

sns.set_style("whitegrid")
sns.set_context("talk")
#variables
N_samples  = 100 #number of observations
N_features = 30 #number of features (and PCs, data is full rank)
pc_change  = [0,28] #pcs that are going to be changed in variance

np.random.default_rng(20237)
#%% generating toy data with linear decay

#background data
temp_S = np.linspace(1,stop=N_features,num=N_features) #variance of background activity, decays in 1/f
# S_bg   = 1/temp_S
S_bg = np.linspace(0,stop=4,num=N_features)[::-1]

#foreground data, where we want to compare the change to background
#delta_var = np.random.randn(N_features)/100 #how much variance to vary by default, we are doing a normali distribution of 1% change in the SD
#S_fg      = S_bg*(1+delta_var)
S_fg = 1.1*S_bg.copy()
# S_fg = S_bg.copy()

#injecting variance in the data
S_fg[pc_change[0]] = S_fg[pc_change[0]]*1.05;
S_fg[pc_change[1]] = S_fg[pc_change[1]]*1.35;

#S_bg[pc_num] = 0

# generating random orthogonal loadings
U = orth(np.random.randn(N_samples,N_features))
V = orth(np.random.randn(N_features,N_features))

plt.figure()
plt.plot(np.arange(1,N_features+1),S_bg,':',label='background dataset')
plt.plot(np.arange(1,N_features+1),S_fg,'--',label='foreground dataset')
plt.legend()
plt.xlabel('PCs')
plt.ylabel('Eigenvalues')
plt.tight_layout()

#plot to check the cPCA
plt.figure();plt.plot(np.arange(1,N_features+1),(S_fg-S_bg)/(S_bg))
#%% reconstruct data

data_bg = np.linalg.multi_dot((U,np.diag(S_bg),V.T));
data_fg = np.linalg.multi_dot((U,np.diag(S_fg),V.T));

cov_bg = data_bg.T.dot(data_bg)
cov_fg = data_fg.T.dot(data_fg)

#plot of covariance matrices
plt.figure();
plt.imshow(cov_bg,cmap='bwr')
plt.title('background cov')
plt.xlabel('features')
plt.ylabel('features')

plt.figure();
plt.imshow(cov_fg,cmap='bwr')
plt.title('foreground cov')
plt.xlabel('features')
plt.ylabel('features')

#plot of
# data_bg = np.linalg.multi_dot((U[:,pc_change],np.diag(S_bg[pc_change]),V.T[pc_change,:]));
data_fg_increased = np.linalg.multi_dot((U[:,pc_change],np.diag(S_fg[pc_change]),V.T[pc_change,:]));
cpca_cov = cov_fg - 1.25*cov_bg
plt.figure();
plt.imshow(cpca_cov,cmap='bwr')
plt.title('cPCA cov')
plt.xlabel('features')
plt.ylabel('features')

plt.figure();
plt.imshow(data_fg_increased.T.dot(data_fg_increased),cmap='bwr')
plt.title('increased dimensions cov')
plt.xlabel('features')
plt.ylabel('features')

cPCA_mdl = cPCA(normalize_flag=False);
cPCA_mdl.fit(data_bg,data_fg,alpha=1.25)
cPCs_all = cPCA_mdl.loadings_*cPCA_mdl.eigenvalues_
plt.figure();
plt.imshow(cPCs_all[:,:2].dot(cPCs_all[:,:2].T),cmap='bwr')
plt.title('top 2 cPCs cov')
plt.xlabel('features')
plt.ylabel('features')
#%% now run cPCA and ncPCA

# cPCs = cPCA(data_bg,data_fg,alpha=1)[:,0]
cPCA_mdl = cPCA(normalize_flag=False);
cPCA_mdl.fit(data_bg,data_fg,alpha=1.25)

cPCs_all = cPCA_mdl.loadings_

"""
mdl = ncPCA(normalize_flag=False)
mdl.fit(data_bg, data_fg)
ncPCs = mdl.loadings_[:,0]
ncPCs_all = mdl.loadings_
#"""
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

cPCA_mdl = cPCA(normalize_flag=False);
alphas_vec = np.linspace(0.8,1.6,num=25)
cPC1st = []
cPC2nd = []
cPC1st_allV = []
for a in alphas_vec:
    cPCA_mdl.fit(data_bg,data_fg,alpha=a)
    cPCs_all = cPCA_mdl.loadings_
    
    cPCs_cossim = np.abs(cosine_similarity_multiple_vectors(V, cPCs_all[:,0]))
    cPC1st_allV.append(cPCs_cossim)
    cPC1st.append(np.argmax(cPCs_cossim))
    
    cPCs_cossim = np.abs(cosine_similarity_multiple_vectors(V, cPCs_all[:,1]))
    cPC2nd.append(np.argmax(cPCs_cossim))
    
plt.figure()
plt.scatter(alphas_vec,cPC1st,color='red',alpha=0.5,label='1st cPC')
plt.scatter(alphas_vec,cPC2nd,color='green',alpha=0.5,label='2nd cPC')
plt.plot(alphas_vec,np.ones(len(alphas_vec))*pc_change[0],'k--',alpha=0.4,label='dimensions of interest')
plt.plot(alphas_vec,np.ones(len(alphas_vec))*pc_change[1],'k--',alpha=0.4)
plt.xlabel('alpha values')
plt.ylabel('Recovered dim. from model')
plt.legend()
plt.tight_layout()

#%% plot subtracting the eigenspectrum


alphas2use = alphas_vec[np.arange(0,25,5)]
sns.set_style("ticks")
SFG = np.tile(S_fg,len(alphas_vec))
SFG.resize((len(alphas_vec),N_features))

SBG = np.tile(S_bg,len(alphas_vec))
SBG.resize((len(alphas_vec),N_features))

newalpha = np.tile(alphas_vec,N_features)
newalpha.resize((N_features,len(alphas_vec)))
plt.figure()
cpc_eq = SFG-np.multiply(newalpha.T,SBG)
plt.plot(cpc_eq[np.arange(0,25,5),:].T)
plt.grid()
plt.xlabel('PCs')
plt.ylabel('FG - alpha*BG')
plt.text(7,0.7,'alpha = '+ str(alphas2use[0])[:4],color='tab:blue',rotation=340)
plt.text(7,0.3,'alpha = '+ str(alphas2use[1])[:4],color='tab:orange',rotation=-11)
plt.text(7,0.0,'alpha = '+ str(alphas2use[2])[:4],color='tab:green',rotation=1)
plt.text(7,-0.5,'alpha = '+ str(alphas2use[3])[:4],color='tab:red',rotation=11)
plt.text(7,-1,'alpha = '+ str(alphas2use[4])[:4],color='tab:purple',rotation=20)