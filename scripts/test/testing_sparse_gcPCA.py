# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:22:48 2023

@author: fermi

script to test sparse gcPCA
"""


#%% first testing sparse gcPCA with toy data from Abid et al

import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import ListedColormap
import sys

# for plotting
cmap2 = ListedColormap(['r', 'k'])
cmap4 = ListedColormap([])

np.random.seed(0) # for reproducibility

N = 400; D = 30; gap=3
# In B, all the data pts are from the same distribution, which has different variances in three subspaces.
B = np.zeros((N, D))
B[:,0:10] = np.random.normal(0,10,(N,10))
B[:,10:20] = np.random.normal(0,3,(N,10))
B[:,20:30] = np.random.normal(0,1,(N,10))


# In A there are four clusters.
A = np.zeros((N, D))
A[:,0:10] = np.random.normal(0,10,(N,10))
# group 1
A[0:100, 10:20] = np.random.normal(0,1,(100,10))
A[0:100, 20:30] = np.random.normal(0,1,(100,10))
# group 2
A[100:200, 10:20] = np.random.normal(0,1,(100,10))
A[100:200, 20:30] = np.random.normal(gap,1,(100,10))
# group 3
A[200:300, 10:20] = np.random.normal(2*gap,1,(100,10))
A[200:300, 20:30] = np.random.normal(0,1,(100,10))
# group 4
A[300:400, 10:20] = np.random.normal(2*gap,1,(100,10))
A[300:400, 20:30] = np.random.normal(gap,1,(100,10))
sub_group_labels_ = [0]*100+[1]*100+[2]*100+[3]*100



# getting ncPCA loadings
# sys.path.append(r"C:\Users\fermi\Documents\GitHub\generalized_contrastive_PCA") # laptopt
sys.path.append("/home/eliezyer/Documents/github/generalized_contrastive_PCA/") # linux
from contrastive_methods import gcPCA, sparse_gcPCA
# fg = target
# bg = background
fg = A
bg = B
gcPCA_mdl = gcPCA(method='v4', alpha=1.5, normalize_flag=False)
gcPCA_mdl.fit(fg, bg)




# plotting gcPCA loadings
target_projected = gcPCA_mdl.Ra_scores_
colors = ['k', 'r', 'g', 'b']
plt.figure(num=1)
plt.scatter(target_projected[0:100, 0], target_projected[0:100, 1], color=colors[0], alpha=0.5)
plt.scatter(target_projected[100:200, 0], target_projected[100:200, 1], color=colors[1], alpha=0.5)
plt.scatter(target_projected[200:300, 0], target_projected[200:300, 1], color=colors[2], alpha=0.5)
plt.scatter(target_projected[300:400, 0], target_projected[300:400, 1], color=colors[3], alpha=0.5)
plt.title("gcPCA")
# plt.show()

#%%
lambdas_array = np.exp(np.linspace(np.log(2e-5), np.log(0.1), num=10))
sparse_gcPCA_mdl = sparse_gcPCA(method='v4', normalize_flag=False,Nsparse=2,lambdas=lambdas_array)
sparse_gcPCA_mdl.fit(fg, bg)
#%%
# target_projected = sparse_gcPCA_mdl.Ra@sparse_gcPCA_mdl.sparse_loadings_[7][:, :2]
target_projected = sparse_gcPCA_mdl.Ra_scores_[4]
colors = ['k', 'r', 'g', 'b']
plt.figure(num=2)
plt.scatter(target_projected[0:100,0], target_projected[0:100,1], color=colors[0], alpha=0.5)
plt.scatter(target_projected[100:200,0], target_projected[100:200,1], color=colors[1], alpha=0.5)
plt.scatter(target_projected[200:300,0], target_projected[200:300,1], color=colors[2], alpha=0.5)
plt.scatter(target_projected[300:400,0], target_projected[300:400,1], color=colors[3], alpha=0.5)
plt.title("gcPCA")
plt.show()

#%%

grid1 = plt.GridSpec(11, 3,left=0.1,right=0.95,top=0.95,bottom=0.1,wspace=0.1, hspace=0.1)
plt.figure(figsize=(6,15),num=3)
for wdim in np.arange(2):
    plt.subplot(grid1[0,wdim])
    plt.stem(sparse_gcPCA_mdl.original_loadings_[:, wdim])
    plt.title('gcpc'+str(wdim+1))
    for a in np.arange(10):
        plt.subplot(grid1[a+1,wdim])
        plt.stem(sparse_gcPCA_mdl.sparse_loadings_[a][:, wdim])

colors = ['k', 'r', 'g', 'b']
target_projected = gcPCA_mdl.Ra_scores_
plt.subplot(grid1[0,2])
plt.scatter(target_projected[0:100, 0], target_projected[0:100, 1], color=colors[0], alpha=0.5)
plt.scatter(target_projected[100:200, 0], target_projected[100:200, 1], color=colors[1], alpha=0.5)
plt.scatter(target_projected[200:300, 0], target_projected[200:300, 1], color=colors[2], alpha=0.5)
plt.scatter(target_projected[300:400, 0], target_projected[300:400, 1], color=colors[3], alpha=0.5)
plt.title("gcPCA")
for a in np.arange(10):
    # target_projected = sparse_gcPCA_mdl.Ra@sparse_gcPCA_mdl.sparse_loadings_[a][:, :2]
    target_projected = sparse_gcPCA_mdl.Ra_scores_[a]
    plt.subplot(grid1[a+1,2])
    plt.scatter(target_projected[0:100,0], target_projected[0:100,1], color=colors[0], alpha=0.5)
    plt.scatter(target_projected[100:200,0], target_projected[100:200,1], color=colors[1], alpha=0.5)
    plt.scatter(target_projected[200:300,0], target_projected[200:300,1], color=colors[2], alpha=0.5)
    plt.scatter(target_projected[300:400,0], target_projected[300:400,1], color=colors[3], alpha=0.5)
    
#%%
# test_loadings = np.zeros((30,2))
# test_loadings[19,0] = sparse_gcPCA_mdl.sparse_loadings_[4][19, 0]
# test_loadings[29,0] = sparse_gcPCA_mdl.sparse_loadings_[4][29, 0]
# test_loadings[10,1] = sparse_gcPCA_mdl.sparse_loadings_[4][10, 1]
# test_loadings[20,1] = sparse_gcPCA_mdl.sparse_loadings_[4][20, 1]

# target_projected = sparse_gcPCA_mdl.Ra@test_loadings
# colors = ['k', 'r', 'g', 'b']
# plt.figure()
# plt.scatter(target_projected[0:100,0], target_projected[0:100,1], color=colors[0], alpha=0.5)
# plt.scatter(target_projected[100:200,0], target_projected[100:200,1], color=colors[1], alpha=0.5)
# plt.scatter(target_projected[200:300,0], target_projected[200:300,1], color=colors[2], alpha=0.5)
# plt.scatter(target_projected[300:400,0], target_projected[300:400,1], color=colors[3], alpha=0.5)
# plt.title("gcPCA")
# plt.show()