# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 10:36:03 2022
Script for analyzing dataset of protein expression using ncPCA and comparing to cPCA
@author: Eliezyer de Oliveira
"""

#%% importing packages
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from matplotlib.colors import ListedColormap
import contrastive

#%% setting up constant variables for the rest of the code
repo_dir = "C:\\Users\\fermi\\Documents\\GitHub\\normalized_contrastive_PCA\\" #repository dir
data_dir = "C:\\Users\\fermi\\Documents\\GitHub\\normalized_contrastive_PCA\\datasets\\from_cPCA_paper\\"
#%% loading data

#going to the repo directory
os.chdir(repo_dir)
from ncPCA import ncPCA

data = np.genfromtxt(os.path.join(data_dir,'Data_Cortex_Nuclear.csv'),
                     delimiter=',', skip_header=1,usecols=range(1,78),filling_values=0)
classes = np.genfromtxt(os.path.join(data_dir,'Data_Cortex_Nuclear.csv'),delimiter=',',
                        skip_header =1,usecols=range(78,81),dtype=None,invalid_raise=False)

#%% preparing data for analysis

# identifying which mice to use
target_idx_A = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))[0]
target_idx_B = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Ts65Dn'))[0]


sub_group_labels = len(target_idx_A)*[0] + len(target_idx_B)*[1] # for identification on plots

target_idx = np.concatenate((target_idx_A,target_idx_B))

target = data[target_idx]
target = (target-np.mean(target,axis=0)) / np.std(target,axis=0) # standardize the dataset

background_idx = np.where((classes[:,-1]==b'C/S') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))
background = data[background_idx]
background = (background-np.mean(background,axis=0)) / np.std(background,axis=0) # standardize the dataset

#%% running contrastive PCA

mdl = contrastive.CPCA()
projected_data = mdl.fit_transform(target, background, plot=True, active_labels=sub_group_labels)

#%% running and plotting ncPCA
X, S = ncPCA(background, target)

#plotting ncPCA
target_projected = np.dot(target,X[:,:2])
sub_group_labels= np.array(sub_group_labels)
a = np.where(sub_group_labels == 0)
b = np.where(sub_group_labels == 1)
figure()
scatter (target_projected[a,0], target_projected[a,1], label = 'control', color = 'k',alpha=0.7)
scatter (target_projected[b,0], target_projected[b,1], label = 'DS', color = 'r', alpha=0.7)
xlabel('ncPC1')
ylabel('ncPC2')
plt.legend()
plt.show()