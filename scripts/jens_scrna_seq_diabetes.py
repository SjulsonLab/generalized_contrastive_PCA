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
import pynapple as nap
import ncPCA
import contrastive

from sklearn.model_selection import cross_val_score
from matplotlib.pyplot import *
from scipy import stats

#%% important paramaters
data_path = "/mnt/probox/Jens_data/scRNA_seq/"
#%% loading data in as pandas dataframe
data = pd.read_table(os.path.join(data_path,'GSE153855_Expression_RPKM_HQ_allsamples.txt'),sep='\t',
                     index_col=0,header=None).T

annotation = pd.read_csv(os.path.join(data_path,'GSE153855_Cell_annotation.txt'),sep='\t')

#%% Separating beta cells from the rest

#The following command is a plot to confirm that beta cells express the INS gene
# i.e., the insulin gene
#data.INS.groupby(annotation.CellType.values=='Beta').hist(legend=['no Beta','Beta'])

#separating data into normal_df
non_beta_df = data[(annotation.CellType.values!='Beta') & (annotation.Disease.values=='normal')]

beta_df = data[(annotation.CellType.values=='Beta') & (annotation.Disease.values=='normal')]

# zscoring data
N1 = non_beta_df.values
N1 = np.divide(stats.zscore(N1),np.linalg.norm(stats.zscore(N1),axis=0))

N2 = beta_df.values
N2 = np.divide(stats.zscore(N2),np.linalg.norm(stats.zscore(N2),axis=0))

# removing nan from the dataset (likely empty vectors etc, THIS NEEDS TO BE LOOKED UPON)
#indices_to_zero = (np.sum(np.isnan(N1),axis=0)>0) (np.sum(np.isnan(N2),axis=0)>0)
N1[np.isnan(N1)]=0
N2[np.isnan(N2)]=0

#%% cleaning data to have a smaller subspace to calculate

#SVD (or PCA) on N1 and N2
# N_full = np.concatenate((N1,N2),axis=0)
# index_n1 = np.zeros(N_full.shape[0])
# index_n1[:N1.shape[0]]=1
# index_n1 = index_n1>0


# U,S,V = np.linalg.svd(N_full,full_matrices=False)
# U1,S1,V1 = np.linalg.svd(N1,full_matrices = False)
# U2,S2,V2 = np.linalg.svd(N2,full_matrices = False)
    
# # discard PCs that cumulatively account for less than 1% of variance, i.e.
# # rank-deficient dimensions
# S1_diagonal = S1
# S2_diagonal = S2

# #cumulative variance
# cumvar_1 = np.divide(np.cumsum(S1_diagonal),np.sum(S1_diagonal))
# cumvar_2 = np.divide(np.cumsum(S2_diagonal),np.sum(S2_diagonal))

# #picking how many PCs to keep
# cutoff = 0.995
# max_1 = np.where(cumvar_1 < cutoff)
# max_2 = np.where(cumvar_2 < cutoff)

# #picking same number of feature, the problem with this is that some of them will miss some features
# n_pcs = np.min([max_1[0][-1],max_2[0][-1]])

#trimming down PCs
# V1_hat = V1[np.arange(n_pcs),:];
# V2_hat = V2[np.arange(n_pcs),:];

#testing truncating the data
#top_genes_index = np.argsort(np.sum(np.abs(V2_hat[:50,:]),axis=0)[-500:];
#N1_trimmed = N1[:,top_genes_index]
#N2_trimmed = N2[:,top_genes_index]

#projecting data on PCs to have a smaller space
#N1_trimmed = np.linalg.multi_dot((N1,V1_hat.T,V1_hat))
#N2_trimmed = np.linalg.multi_dot((N2,V2_hat.T,V2_hat))
# Smat = np.diag(S)
# N_reconstructed = np.linalg.multi_dot((U[:,:200],Smat[:200,:200],V[:200,:]))
# N1_trimmed = N_reconstructed[index_n1,:]
# N2_trimmed = N_reconstructed[np.logical_not(index_n1),:]

#%% trying to trim the data by using L1 regularized multinomial logistic regression

N_full = np.concatenate((N1,N2),axis=0)
index_n1 = np.zeros(N_full.shape[0])
index_n1[:N1.shape[0]]=1
index_n1 = index_n1>0

idx = (annotation.CellType.values!='Beta') & (annotation.Disease.values=='normal')
reduced_values_n1 = annotation.CellType.values[idx]
betalabels = annotation.CellType.values[(annotation.CellType.values=='Beta') & (annotation.Disease.values=='normal')]
concat_ann = np.concatenate((reduced_values_n1,betalabels))

c = 0
labels_id = np.zeros(concat_ann.shape)
str2legend = []
for name in np.unique(concat_ann):
    labels_id[concat_ann==name]=c
    str2legend.append(name)
    c += 1

from sklearn.linear_model import LogisticRegressionCV

l1_log_reg = LogisticRegressionCV(Cs=np.logspace(-4,6,20),penalty='l1',solver='liblinear')
#fitting
l1_log_reg.fit(N_full,labels_id)

#picking genes to use
genes_to_keep = np.argwhere(np.abs(l1_log_reg.coef_.sum(axis=0))>0)

#%% running ncPCA
#N1_trimmed = np.random.randn(10000,3000)
#N2_trimmed = np.random.randn(10000,3000)
N1_trimmed = stats.zscore(N_full[index_n1,genes_to_keep]).T
N2_trimmed = stats.zscore(N_full[np.logical_not(index_n1),genes_to_keep]).T

X,S_total = ncPCA.ncPCA_orth(N1_trimmed,N2_trimmed,skip_normalization=False)
cPCs = ncPCA.cPCA(N1_trimmed,N2_trimmed)


#%% making plots

#plotting the N1 and N2 trimmed data to the top ranks
figure()
imshow(N1_trimmed,cmap='magma',clim=(-0.5,0.5));title('non-beta');xlabel('features');ylabel('cells')
figure()
imshow(N2_trimmed,cmap='magma',clim=(-0.5,0.5));title('beta');xlabel('features');ylabel('cells')

#plotting ncPCA results, loadings and 'var'
figure()
plot(S_total);title('ncPCA ');xlabel('ncPCs')
imshow(X,cmap='magma',clim=((-0.1,0.1)))

# plot of ncPCA loadings, top 3
figure()
subplot(3,1,1);stem(X[:,0]);ylabel('loadgs ncPC1');
subplot(3,1,2);stem(X[:,1]);ylabel('loadgs ncPC2');
subplot(3,1,3);stem(X[:,2]);ylabel('loadgs ncPC2');
xlabel('features')


##############################
""" making a plot of spPCA projecting back on genes space to get the genes 
 overexpressed or underexpressed
"""
##############################
#%%
#DOING FOR THE FIRST PC ONLY
#projecting it back to gene space
#ncPCA_gene_space = np.dot(X[:,-1].T,V[:,:200].T)
ncPCA_gene_space = X[:,0]
genes_names = data.columns;
genes_names_kept = genes_names[genes_to_keep]
num = np.arange(len(ncPCA_gene_space))

#sorting gene expression
idx_sorting = np.argsort(ncPCA_gene_space);
ncPCA_gene_space_sorted = ncPCA_gene_space[idx_sorting]
genes_names_sorted = genes_names_kept[idx_sorting]
stem(ncPCA_gene_space_sorted);xlabel('genes');ylabel('expression norm.');title('ncPC1')
text(num[genes_names_sorted=='INS'],ncPCA_gene_space_sorted[genes_names_sorted=='INS'],'INS',
      rotation='vertical',fontsize=7)
#%%
figure()
stem(ncPCA_gene_space_sorted);xlabel('genes');ylabel('expression norm.');xlim((-1,40))
for a in np.arange(40):
    text(a,0.005,genes_names_sorted[a],rotation='vertical',fontsize=7)

figure()
stem(ncPCA_gene_space_sorted);xlabel('genes');ylabel('expression norm.');xlim((num[-1]-40,num[-1]))
for a in np.arange(num[-1]-40,num[-1]+1):
        text(a,-0.025,genes_names_sorted[a],rotation='vertical',fontsize=7)

#%%
#DOING FOR THE FIRST PC ONLY
ncPCA_gene_space = X[:,0]
genes_names = data.columns
#sorting gene expression
num = np.arange(len(genes_names_sorted))
temp_gn = genes_names[top_genes_index]

idx_sorting = np.argsort(ncPCA_gene_space)
ncPCA_gene_space_sorted = ncPCA_gene_space[idx_sorting]
genes_names_sorted = temp_gn[idx_sorting]

stem(ncPCA_gene_space_sorted);xlabel('genes');ylabel('expression norm.');title('ncPC1')
text(num[genes_names_sorted=='INS'],ncPCA_gene_space[genes_names_sorted=='INS'],'INS',
     rotation='vertical',fontsize=7)
#%%
figure()
stem(ncPCA_gene_space_sorted);xlabel('genes');ylabel('expression norm.');xlim((-1,40))
for a in np.arange(20):
    text(a,0.005,genes_names_sorted[a],rotation='vertical',fontsize=7)

figure()
stem(ncPCA_gene_space_sorted);xlabel('genes');ylabel('expression norm.');xlim((num[-1]-40,num[-1]))
for a in np.arange(num[-1]-40,num[-1]+1):
        text(a,-0.025,genes_names_sorted[a],rotation='vertical',fontsize=7)


#%%
beta_cells_PC = V2[0,:].T
idx_sorting = np.argsort(beta_cells_PC);
beta_cells_PC_sorted= beta_cells_PC[idx_sorting]
genes_names_sorted = genes_names[idx_sorting]
stem(beta_cells_PC_sorted);xlabel('genes');ylabel('expression norm.');title('Beta_cells_PC')
text(num[genes_names_sorted=='INS'],beta_cells_PC_sorted[genes_names_sorted=='INS'],'INS',
      rotation='vertical',fontsize=7)

#%% same plot as above but cPC
selected_cPCs = np.real(cPCs[:,1])

idx_sorting = np.argsort(selected_cPCs);
selected_cPCs_sorted = selected_cPCs[idx_sorting]
genes_names_sorted = genes_names[idx_sorting]
num = np.arange(len(genes_names_sorted))

stem(selected_cPCs_sorted);xlabel('genes');ylabel('expression norm.');title('cPCs')
text(num[genes_names_sorted=='INS'],selected_cPCs_sorted[genes_names_sorted=='INS'],'INS',
      rotation='vertical',fontsize=7)

#%% make plots on projections and class

#getting index of each cell type to label with different plot
#improve for readability later
idx = (annotation.CellType.values!='Beta') & (annotation.Disease.values=='normal')
reduced_values_n1 = annotation.CellType.values[idx]
betalabels = annotation.CellType.values[(annotation.CellType.values=='Beta') & (annotation.Disease.values=='normal')]
concat_ann = np.concatenate((reduced_values_n1,betalabels))
colormap = cm.tab20


c = 0
labels_id = np.zeros(concat_ann.shape)
str2legend = []
for name in np.unique(concat_ann):
    labels_id[concat_ann==name]=c
    str2legend.append(name)
    c += 1

# plot the scores on projections from beta cells PC, cPCA and ncPCA
#subplot(1,3,1)
#proj_beta_PC = np.dot(N_reconstructed,V2_hat[(0,1),:].T)
#aux_scat = scatter(proj_beta_PC[:,0],proj_beta_PC[:,1],c=labels_id,cmap=colormap,alpha=0.5)
#handles, labels = aux_scat.legend_elements(prop="colors", alpha=0.6)
#legend(handles,str2legend)
N2use = np.squeeze(N2use)
subplot(1,2,1)
proj_cPCs = np.dot(N2use,np.real(cPCs[:,(0,1)]))
scatter(proj_cPCs[:,0],proj_cPCs[:,1],c=labels_id,cmap=colormap,alpha=0.5)

subplot(1,2,2)
proj_ncPCs = np.dot(N2use,X[:,(936,937)])
scatter(proj_ncPCs[:,0],proj_ncPCs[:,1],c=labels_id,cmap=colormap,alpha=0.5)
xlim((-0.5,0.5))
ylim((-0.5,0.5))








