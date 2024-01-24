#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:33:25 2024

@author: eliezyer
 testing a different toy model where we have changing vectors on both sides
"""

# %% importing libraries and defining main variables

#libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orth
from scipy.stats import zscore
import sys
import seaborn as sns
import random

# repo_dir = "C:\\Users\\fermi\\Documents\\GitHub\\generalized_contrastive_PCA\\" #repository dir on laptop
repo_dir = "/home/eliezyer/Documents/github/generalized_contrastive_PCA/" #repository dir on linux machine
sys.path.append(repo_dir)
from contrastive_methods import gcPCA
from ncPCA_project_utils import cosine_similarity_multiple_vectors

sns.set_style("whitegrid")
sns.set_context("talk")
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 150
plt.rc('axes',edgecolor='k')
# folder_save_plot = "C:\\Users\\fermi\\Dropbox\\figures_ncPCA\\toy_data\\" # on laptop
folder_save_plot = "/mnt/extraSSD4TB/CloudStorage/Dropbox/figures_gcPCA/toy_data/" #on linux machine
#variables
N_samples  = 100 #number of observations

N_features = 30 #number of features (and PCs, data is full rank)
pc_change  = [10,28] #pcs that are going to be changed in variance
pc_change2  = [15,25] #pcs that are going to be changed in variance
ratio_change = [1.05,2]
ratio_change2 = [1.05,2]
np.random.seed(1)
# generating toy data with linear decay

# B dataset or background data
S_B = np.linspace(0,stop=4,num=N_features)[::-1]+10**-4
# A dataset, or foreground data, what we want to identify the highly exposed
S_A = 1.0*S_B.copy()  # LATER ON TRY TO DO RANDOM VARIABILITY ON EACH COMP.

#injecting variance in specific components

S_A[pc_change[0]] = S_A[pc_change[0]]*ratio_change[0]
S_A[pc_change[1]] = S_A[pc_change[1]]*ratio_change[1]
S_B[pc_change2[0]] = S_B[pc_change2[0]]*ratio_change2[0]
S_B[pc_change2[1]] = S_B[pc_change2[1]]*ratio_change2[1]

#S_bg[pc_num] = 0

# generating random orthogonal loadings
U = orth(np.random.randn(N_samples,N_features))
V = orth(np.random.randn(N_features,N_features))

grid1 = plt.GridSpec(4, 1, left=0.2 , wspace = 0.2, hspace=0.05)

plt.figure(num=1,figsize=(5,20))
ax1=plt.subplot(grid1[0])
plt.plot(np.arange(1,N_features+1),S_B,':',label='Condition B')
plt.plot(np.arange(1,N_features+1),S_A,'--',label='Condition A')
plt.legend()
sns.despine()
plt.xlabel('Components')
plt.ylabel('Eigenvalues')


#plot to check the gcPCA
#plot the cPCA and show that is not symmetrical
plt.subplot(grid1[1],sharex=ax1)
plt.plot(np.arange(1,N_features+1),(S_A)/(S_B),'--',color='k',label='A/B')
plt.plot(np.arange(1,N_features+1),(S_B)/(S_A),color='r',label='B/A')
sns.despine()
plt.xticks(np.arange(5,N_features+1,step=5))
plt.xlabel('Dimensions')
plt.ylabel('Obj. function magnitude')
plt.legend()


#plot to check the gcPCA
plt.subplot(grid1[2])
plt.plot(np.arange(1,N_features+1),(S_A-S_B)/(S_B),'--',color='k',label='(A-B)/B')
plt.plot(np.arange(1,N_features+1),(S_B-S_A)/(S_A),color='r',label='(B-A)/A')
sns.despine()
plt.xticks(np.arange(5,N_features+1,step=5))
plt.xlabel('Dimensions')
plt.ylabel('Obj. function magnitude')
plt.legend()


#plot to check the gcPCA
plt.subplot(grid1[3])
plt.plot(np.arange(1,N_features+1),(S_A-S_B)/(S_A+S_B),'--',color='k',label='(A-B)/(A+B)')
plt.plot(np.arange(1,N_features+1),(S_B-S_A)/(S_B+S_A),color='r',label='(B-A)/(B+A)')
sns.despine()
plt.xticks(np.arange(5,N_features+1,step=5))
plt.xlabel('Dimensions')
plt.ylabel('Obj. function magnitude')
plt.legend()