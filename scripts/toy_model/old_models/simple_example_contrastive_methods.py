# -*- coding: utf-8 -*-
"""
Created on Sun May  7 11:31:40 2023

@author: fermi

script with simple example of how contrastive methods can help you find unsupervised
discrimination in a dataset
"""


#%% importing essentials

import numpy as np
import matplotlib.pyplot as plt
import sys

repo_dir = "C:\\Users\\fermi\\Documents\\GitHub\\normalized_contrastive_PCA\\" #repository dir
sys.path.append(repo_dir)
from contrastive_methods import cPCA
#%%
Nsamples = 250

temp = np.sort(np.random.randn(Nsamples)*2)
data_bg = np.concatenate((temp[:,np.newaxis],-1*temp[:,np.newaxis]+np.random.randn(Nsamples,1)),axis=1)

offset_rnd = 1.8
temp2 = np.random.permutation(np.concatenate((np.random.randn(int(Nsamples/2),1)-offset_rnd,np.random.randn(int(Nsamples/2),1)+offset_rnd),axis=0))

data_fg = np.concatenate((temp[:,np.newaxis],-1*temp[:,np.newaxis]+temp2),axis=1)

#%% plot of the data
plt.figure()
plt.scatter(data_bg[:,0],data_bg[:,1],c='gray')
plt.scatter(data_fg[:,0],data_fg[:,1],c='red')
#add plot of the directions each method is getting

u,sfg,v = np.linalg.svd(data_fg,full_matrices=False)
_,sbg,_ = np.linalg.svd(data_bg,full_matrices=False)

wbg,vbg = np.linalg.eig(data_bg.T.dot(data_bg))
wfg,vfg = np.linalg.eig(data_fg.T.dot(data_fg))

# aux = np.concatenate((np.ones((len(temp),1)),temp[:,np.newaxis]),axis=1)
aux = np.concatenate((temp[:,np.newaxis],-1*temp[:,np.newaxis]),axis=1)
vectors = data_fg.dot(v.T)
plt.plot(vectors[:,0],vectors[:,1],linewidth=3,c='black')
# plt.plot(temp,vectors[:,1],linewidth=3,c='red')
#plt.plot(temp,vectors[:,1])

#cpca
cpca_mdl = cPCA(normalize_flag=False)
cpca_mdl.fit(data_bg,data_fg,alpha=3)
cpc_v = cpca_mdl.loadings_

# aux = np.concatenate((np.ones((len(temp),1)),temp[:,np.newaxis]),axis=1)
# cpc_vectors = aux.dot(cpc_v)
# plt.plot(temp,cpc_vectors[:,0],linewidth=3,c='blue')

#%% plot of eigenvalues bg and fg
plt.figure()
plt.plot(sbg,c='gray')
plt.plot(sfg,c='red')
plt.plot(cpca_mdl.eigenvalues_)

