#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 21:32:48 2022

Script to analyze dataset of scRNA-seq from jens
@author: eliezyer
"""
#%% importing essentials
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import sys

repo_dir = "/home/eliezyer/Documents/github/generalized_contrastive_PCA/" #repository dir in linux machine
# repo_dir = "C:\\Users\\fermi\\Documents\\GitHub\\generalized_contrastive_PCA" #repository dir in win laptop
# repo_dir =  #repo dir in HPC

sys.path.append(repo_dir)
from contrastive_methods import gcPCA, sparse_gcPCA

# from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
# from scipy import stats

#%% important paramaters
data_path = "/mnt/extraSSD4TB/CloudStorage/Dropbox/preprocessing_data/gcPCA_files/Jens_data/scRNA_seq/"
save_path = "/mnt/extraSSD4TB/CloudStorage/Dropbox/figures_gcPCA/diabetes_scrna_seq/"
# data_path = "C:\\Users\\fermi\\Dropbox\\preprocessing_data\\gcPCA_files\\Jens_data\\scRNA_seq"
# save_path = r'C:\Users\fermi\Dropbox\figures_gcPCA\diabetes_scrna_seq'
os.chdir(data_path)
#%% loading data in as pandas dataframe
data = pd.read_table(os.path.join(data_path,'GSE153855_Expression_RPKM_HQ_allsamples.txt'),sep='\t',
                     index_col=0,header=None).T

annotation = pd.read_csv(os.path.join(data_path,'GSE153855_Cell_annotation.txt'),sep='\t')

#%% Separating beta cells from the rest

#The following command is a plot to confirm that beta cells express the INS gene
# i.e., the insulin gene
#data.INS.groupby(annotation.CellType.values=='Beta').hist(legend=['no Beta','Beta'])

#separating data into normal_df
diabetes_beta_df = data[(annotation.CellType.values=='Beta') & (annotation.Disease.values=='type II diabetes')]
normal_beta_df = data[(annotation.CellType.values=='Beta') & (annotation.Disease.values=='normal')]
diabetes_alpha_df = data[(annotation.CellType.values=='Alpha') & (annotation.Disease.values=='type II diabetes')]
normal_alpha_df = data[(annotation.CellType.values=='Alpha') & (annotation.Disease.values=='normal')]
diabetes_delta_df = data[(annotation.CellType.values=='Delta') & (annotation.Disease.values=='type II diabetes')]
normal_delta_df = data[(annotation.CellType.values=='Delta') & (annotation.Disease.values=='normal')]
diabetes_gamma_df = data[(annotation.CellType.values=='Gamma') & (annotation.Disease.values=='type II diabetes')]
normal_gamma_df = data[(annotation.CellType.values=='Gamma') & (annotation.Disease.values=='normal')]

subject_beta_t2d = annotation.Donor[(annotation.CellType.values=='Beta') & (annotation.Disease.values=='type II diabetes')]
subject_beta_normal = annotation.Donor[(annotation.CellType.values=='Beta') & (annotation.Disease.values=='normal')]

#%%
#log transforming the data
tempN1 = np.log(diabetes_beta_df.values+1)
tempN2 = np.log(normal_beta_df.values+1)

#throwing out the features that are empty in both datasets
features_to_keep = (np.sum(tempN1==0,axis=0)!=tempN1.shape[0]) + (np.sum(tempN2==0,axis=0)!=tempN2.shape[0])
N1_red = tempN1[:,features_to_keep]
N2_red = tempN2[:,features_to_keep]

# zscoring data

# N1 = np.divide(stats.zscore(N1_red),np.linalg.norm(stats.zscore(N1_red),axis=0))
# N2 = np.divide(stats.zscore(N2_red),np.linalg.norm(stats.zscore(N2_red),axis=0))


N1 = N1_red - np.mean(N1_red,axis=0)
N2 = N2_red - np.mean(N2_red,axis=0)

# running gcPCA and plotting the cells in scores

gcpca_mdl = gcPCA(method='v4',normalize_flag=False)
gcpca_mdl.fit(N1,N2) #N1 is diabetes and N2 is normal, all beta cells


#%% plotting
plt.figure(num=10,figsize=(30,10))
grid1 = plt.GridSpec(1, 4,left=0.01,right=0.99,top=0.90,bottom=0.1,wspace=0.2, hspace=0.15)
plt.subplot(grid1[0,1])
plt.scatter(gcpca_mdl.Rb_scores_[:,0],
            gcpca_mdl.Rb_scores_[:,1],s=150)
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
plt.xlim((-0.2,0.2))
plt.ylim((-0.3,0.3))
plt.title('Beta Cells - normal')

#clustering
mdl = KMeans(n_clusters=2)
mdl.fit(gcpca_mdl.Ra_scores_[:,:2])
plt.rcParams.update({'figure.dpi':150, 'font.size':24})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.figure(num=1,figsize=(30,10))
grid1 = plt.GridSpec(1, 4,left=0.01,right=0.99,top=0.90,bottom=0.1,wspace=0.2, hspace=0.15)
plt.subplot(grid1[0,1])
plt.scatter(gcpca_mdl.Ra_scores_[mdl.labels_==1,0],
            gcpca_mdl.Ra_scores_[mdl.labels_==1,1],s=150,c='red')
plt.scatter(gcpca_mdl.Ra_scores_[mdl.labels_==0,0],
            gcpca_mdl.Ra_scores_[mdl.labels_==0,1],s=150,c='blue')
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
plt.xlim((-0.2,0.2))
plt.ylim((-0.3,0.3))
plt.title('Beta Cells - type II diabetes')
# plt.figure(num=2);plt.scatter(gcpca_mdl.Rb_scores_[:,0],gcpca_mdl.Rb_scores_[:,1])


idx_score_sort = np.argsort(gcpca_mdl.Ra_scores_[:,0])
idx_sorting = np.argsort(gcpca_mdl.loadings_[:,0])
# sigma = 4.0
# low_thresh,high_thresh = -1*sigma*gcPCA_gene_space_sorted.std(),sigma*gcPCA_gene_space_sorted.std()
# idx = np.argwhere(np.logical_or(gcPCA_gene_space_sorted < low_thresh, gcPCA_gene_space_sorted > high_thresh))

# test = np.outer(gcpca_mdl.Ra_scores_[:,0],gcpca_mdl.loadings_[:,0].T)
temp = N1_red[idx_score_sort,:].copy()

N1_red_sorted = temp[:,idx_sorting]
N1_picked = np.concatenate((N1_red_sorted[:,:100],N1_red_sorted[:,-100:-1]),axis=1)

# N1_picked = np.squeeze(N1_red_sorted[:,idx])
# plt.figure(figsize=(10,17),num=4)
plt.subplot(grid1[0,2])
plt.imshow(N1_picked.T,clim=(0,6),aspect=2)
plt.xlabel('Beta cells')

plt.ylabel('genes')
plt.title('Type 2 diabetes')

plt.figtext(0.21, 0.93, 'B', fontsize=40, fontweight='bold')
plt.figtext(0.49, 0.93, 'C', fontsize=40, fontweight='bold')
plt.savefig(os.path.join(save_path,"pancreas_scRNAseq_betacells_figure5.pdf"), format="pdf")

#%%
# temp_t2d = np.concatenate((diabetes_beta_df.values,
                           # diabetes_alpha_df.values,
                           # diabetes_delta_df.values,
                           # diabetes_gamma_df.values),axis=0)

# temp_n = np.concatenate((normal_beta_df.values,
                           # normal_alpha_df.values,
                           # normal_delta_df.values,
                           # normal_gamma_df.values),axis=0)

# tempN1 = np.log(temp_t2d+1)
# tempN2 = np.log(temp_n+1)
plt.figure(num=2,figsize=(12,8))

for name in subject_beta_t2d.unique():
    plt.scatter(gcpca_mdl.Ra_scores_[subject_beta_t2d==name,0],
                gcpca_mdl.Ra_scores_[subject_beta_t2d==name,1],s=80)
    
plt.xlim((-0.2,0.2))
plt.ylim((-0.3,0.3))
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
plt.title('Beta cell T2D - per patient')

plt.figure(num=7,figsize=(12,8))

for name in subject_beta_normal.unique():
    plt.scatter(gcpca_mdl.Rb_scores_[subject_beta_normal==name,0],
                gcpca_mdl.Rb_scores_[subject_beta_normal==name,1],s=80)

plt.xlim((-0.2,0.2))
plt.ylim((-0.3,0.3))
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
plt.title('Beta cell normal - per patient')

#%% plotting gcPCA

gcpca_gene_space = gcpca_mdl.loadings_[:,0]
genes_names = data.columns;
genes_names_kept = genes_names[features_to_keep]
num = np.arange(len(gcpca_gene_space))

#sorting gene expression
idx_sorting = np.argsort(gcpca_gene_space)
gcPCA_gene_space_sorted = gcpca_gene_space[idx_sorting]
genes_names_sorted = genes_names_kept[idx_sorting]
indx_INS = np.argwhere(genes_names_sorted=='INS')
plt.figure(num=3)
plt.stem(gcPCA_gene_space_sorted)
plt.xlabel('genes')
plt.ylabel('expression norm.')
plt.title('gcPC1')
plt.text(num[indx_INS[0,0]],gcPCA_gene_space_sorted[indx_INS[0,0]],'INS',
      rotation='vertical',fontsize=7)

#%% writing gene names sorted
with open('top_genes_names.txt', 'w') as f:
    for line in genes_names_sorted[-100:-1]:
        f.write(f"{line}\n")
        
with open('bottom_genes_names.txt', 'w') as f:
    for line in genes_names_sorted[:100]:
        f.write(f"{line}\n")

#%% picking genes to make a heatmap, plotting the heatmap and names

idx_score_sort = np.argsort(gcpca_mdl.Ra_scores_[:,0])
idx_sorting = np.argsort(gcpca_mdl.loadings_[:,0])
# sigma = 4.0
# low_thresh,high_thresh = -1*sigma*gcPCA_gene_space_sorted.std(),sigma*gcPCA_gene_space_sorted.std()
# idx = np.argwhere(np.logical_or(gcPCA_gene_space_sorted < low_thresh, gcPCA_gene_space_sorted > high_thresh))

# test = np.outer(gcpca_mdl.Ra_scores_[:,0],gcpca_mdl.loadings_[:,0].T)
temp = N1_red[idx_score_sort,:].copy()

N1_red_sorted = temp[:,idx_sorting]
N1_picked = np.concatenate((N1_red_sorted[:,:100],N1_red_sorted[:,-100:-1]),axis=1)

# N1_picked = np.squeeze(N1_red_sorted[:,idx])
plt.figure(figsize=(10,17),num=4)

plt.imshow(N1_picked.T,clim=(0,6),aspect=7)
plt.xlabel('Beta cells ordered',fontsize=20)
# plt.xticks(fontsize=22)
n_genes = N1_picked.shape[1]
gene_n_list = np.concatenate((genes_names_sorted[:100],genes_names_sorted[-100:-1]),axis=0)
# plt.yticks(np.arange(0,n_genes,1),gene_n_list,fontsize=6)
plt.ylabel('genes')
plt.title('log count')

# with open('thresholded_genes_names.txt', 'w') as f:
#     for line in genes_names_sorted[idx]:
#         f.write(f"{line}\n")

# N2_red_sorted = N2_red[:,idx_sorting]
# plot N1_red and N2_red

