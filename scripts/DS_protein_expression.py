# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 10:36:03 2022
Script for analyzing dataset of protein expression using ncPCA and comparing to cPCA
@author: Eliezyer de Oliveira
"""

#%% importing packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import contrastive
import seaborn as sns


#%% setting up constant variables for the rest of the code
# repo_dir = "/home/eliezyer/Documents/github/normalized_contrastive_PCA/" #repository dir
# data_dir = "/home/eliezyer/Documents/github/normalized_contrastive_PCA/datasets/from_cPCA_paper/"
repo_dir = r'C:\Users\fermi\Documents\GitHub\generalized_contrastive_PCA'
data_dir = r'C:\Users\fermi\Documents\GitHub\generalized_contrastive_PCA\datasets\from_cPCA_paper'
folder_save_plot = r'C:\Users\fermi\Dropbox\figures_gcPCA\cPCA_paper_results'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 200
plt.rcParams.update({'font.size': 22})
#%% importing ncPCA
sys.path.append(repo_dir)
from contrastive_methods import gcPCA
#%% loading data

#going to the repo directory
os.chdir(repo_dir)


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
# sns.set_style("ticks")
# sns.set_context("notebook")

# style.use('default')
mdl = contrastive.CPCA()
projected_data = mdl.fit_transform(target, background, plot=True,colors=['k','r'], active_labels=sub_group_labels)

#%%
sub_group_labels= np.array(sub_group_labels)
a = np.where(sub_group_labels == 0)
b = np.where(sub_group_labels == 1)

# sns.set_style("ticks")
# sns.set_context("talk")
# style.use('default')
gcPCA_mdl = gcPCA(method='v1',alpha=4,normalize_flag=False)
gcPCA_mdl.fit(target,background)

gcPCs_all = gcPCA_mdl.loadings_
plt.figure()
target_projected = gcPCA_mdl.Ra_scores_
plt.scatter(target_projected[a,0], target_projected[a,1],s=80, label = 'control', color = 'k',alpha=0.9,edgecolors=None)
plt.scatter(target_projected[b,0], target_projected[b,1],s=80, label = 'DS', color = 'r', alpha=0.9,edgecolors=None)
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
plt.legend()
#%% running gcPCA v4

sub_group_labels= np.array(sub_group_labels)
a = np.where(sub_group_labels == 0)
b = np.where(sub_group_labels == 1)

# sns.set_style("ticks")
# sns.set_context("talk")
# style.use('default')
gcPCA_mdl = gcPCA(method='v4',normalize_flag=False)
gcPCA_mdl.fit(target,background)

gcPCs_all = gcPCA_mdl.loadings_
plt.figure()
target_projected = gcPCA_mdl.Ra_scores_
plt.scatter(target_projected[a,0], target_projected[a,1],s=80, label = 'control', color = 'k',alpha=0.9,edgecolors=None)
plt.scatter(target_projected[b,0], target_projected[b,1],s=80, label = 'DS', color = 'r', alpha=0.9,edgecolors=None)
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
plt.legend()
plt.savefig(folder_save_plot+"DS_data_gcPCAv4.pdf", transparent=True)
#%% running PCA to test
U,S,V = np.linalg.svd(target,full_matrices=False)
plt.figure()
target_projected = U
plt.scatter(target_projected[a,0], target_projected[a,1],s=80, label = 'control', color = 'k',alpha=0.9,edgecolors=None)
plt.scatter(target_projected[b,0], target_projected[b,1],s=80, label = 'DS', color = 'r', alpha=0.9,edgecolors=None)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

plt.savefig(folder_save_plot+"DS_data_PCA.pdf", transparent=True)
#%% running and plotting ncPCA

# sub_group_labels= np.array(sub_group_labels)
# a = np.where(sub_group_labels == 0)
# b = np.where(sub_group_labels == 1)

# sns.set_style("ticks")
# sns.set_context("notebook")
# style.use('dark_background')

# #X,S = ncPCA_mdl.ncPCA_orth(background,target)
# #X, S = ncPCA(background, target)

# #plotting ncPCA
# #target_projected = np.dot(target,X[:,:2])


# figure()
# ncPCA_mdl = ncPCA(basis_type='all')
# ncPCA_mdl.fit(background,target)
# target_projected = ncPCA_mdl.N2_scores_

# subplot(1,3,1,aspect='equal')
# scatter (target_projected[a,0], target_projected[a,1], label = 'control', color = 'w',alpha=0.7)
# scatter (target_projected[b,0], target_projected[b,1], label = 'DS', color = 'r', alpha=0.7)
# xlabel('ncPC1')
# ylabel('ncPC2')
# title('all basis')
# plt.legend()

# ncPCA_mdl = ncPCA(basis_type='union')
# ncPCA_mdl.fit(background,target)
# target_projected = ncPCA_mdl.N2_scores_

# subplot(1,3,2,aspect='equal')
# scatter (target_projected[a,0], target_projected[a,1], label = 'control', color = 'w',alpha=0.7)
# scatter (target_projected[b,0], target_projected[b,1], label = 'DS', color = 'r', alpha=0.7)
# xlabel('ncPC1')
# title('union')

# ncPCA_mdl = ncPCA(Nshuffle=10000,basis_type='intersect')
# ncPCA_mdl.fit(background,target)
# target_projected = ncPCA_mdl.N2_scores_

# subplot(1,3,3,aspect='equal')
# scatter (target_projected[a,0], target_projected[a,1], label = 'control', color = 'w',alpha=0.7)
# scatter (target_projected[b,0], target_projected[b,1], label = 'DS', color = 'r', alpha=0.7)
# xlabel('ncPC1')
# title('intersect')

# #%% testing the sorting
# test = ncPCA_mdl.ncPCA_values_null_
# # test2 = np.vstack(test).T
# # test3 = np.sort(test2,axis=1)
# color2use = cm.bwr(np.linspace(0,1,num=test.shape[1]))

# figure()
# for a in np.arange(test.shape[1]):
#      plot(test[:,a],color=color2use[a,:])
# plot(ncPCA_mdl.ncPCs_values_,color='k',label='real data')
# legend()
# xlabel('ncPCs')
# ylabel('ncPCA values')
# #xlim((-1,10))
# #ylim((0.9,1.1))
#%% 
