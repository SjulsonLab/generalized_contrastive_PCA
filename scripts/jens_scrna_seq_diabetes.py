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

# repo_dir = "/home/eliezyer/Documents/github/normalized_contrastive_PCA/" #repository dir in linux machine
repo_dir = "C:\\Users\\fermi\\Documents\\GitHub\\generalized_contrastive_PCA" #repository dir in win laptop
# repo_dir =  #repo dir in HPC

sys.path.append(repo_dir)
from contrastive_methods import gcPCA, sparse_gcPCA

# from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
# from scipy import stats

#%% important paramaters
# data_path = "/mnt/probox/Jens_data/scRNA_seq/"
data_path = "C:\\Users\\fermi\\Dropbox\\preprocessing_data\\gcPCA_files\\Jens_data\\scRNA_seq"
save_path = r'C:\Users\fermi\Dropbox\figures_gcPCA\diabetes_scrna_seq'
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

subject_beta = annotation.Donor[(annotation.CellType.values=='Beta') & (annotation.Disease.values=='type II diabetes')]

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
plt.title('Beta Cells')
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

for name in subject_beta.unique():
    plt.scatter(gcpca_mdl.Ra_scores_[subject_beta==name,0],
                gcpca_mdl.Ra_scores_[subject_beta==name,1],s=80)
    
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
plt.title('Beta cell per patient')
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

# N_full = np.concatenate((N1,N2),axis=0)
# index_n1 = np.zeros(N_full.shape[0])
# index_n1[:N1.shape[0]]=1
# index_n1 = index_n1>0

# idx = (annotation.CellType.values=='Beta') & (annotation.Disease.values=='type II diabetes')
# reduced_values_n1 = annotation.Disease.values[idx]
# betalabels = annotation.Disease.values[(annotation.CellType.values=='Beta') & (annotation.Disease.values=='normal')]
# concat_ann = np.concatenate((reduced_values_n1,betalabels)) #concatenated annotation

# c = 0
# labels_id = np.zeros(concat_ann.shape)
# str2legend = []
# for name in np.unique(concat_ann):
#     labels_id[concat_ann==name]=c
#     str2legend.append(name)
#     c += 1

# from sklearn.linear_model import LogisticRegressionCV

# l1_log_reg = LogisticRegressionCV(Cs=np.logspace(-4,6,20),penalty='l1',solver='liblinear')
# #fitting
# l1_log_reg.fit(N_full,labels_id)

# #picking genes to use
# genes_to_keep = np.argwhere(np.abs(l1_log_reg.coef_.sum(axis=0))>0)


#%% repeat the above plot per t2d subject

#%% running sparse gcPCA and plotting the cells in scores

# lambdas_array = np.exp(np.linspace(np.log(2e-5), np.log(5e-3), num=3))
# lambdas_array = np.exp(np.log(5e-3))
# sparse_gcPCA_mdl = sparse_gcPCA(method='v4', normalize_flag=False,Nsparse=2,lambdas=lambdas_array)
# sparse_gcPCA_mdl.fit(N1, N2)

# plt.figure(num=5);plt.scatter(gcpca_mdl.Ra_scores_[0][:,0],gcpca_mdl.Ra_scores_[0][:,1])

#%% plotting the clusters based on different lambdas
# grid1 = plt.GridSpec(11, 3,left=0.1,right=0.95,top=0.95,bottom=0.1,wspace=0.1, hspace=0.1)
# for a in np.arange(10):
#     # target_projected = sparse_gcPCA_mdl.Ra@sparse_gcPCA_mdl.sparse_loadings_[a][:, :2]
#     target_projected = sparse_gcPCA_mdl.Ra_scores_[a]
#     plt.subplot(grid1[a+1,2])
#     plt.scatter(target_projected[:,0], target_projected[:,1])
#%% plotting gcPCA and tagging insulin
# X=gcpca_mdl.loadings_
# genes_names = data.columns[features_to_keep]
# plt.scatter(X[:,0],X[:,1])
# indx_INS = np.argwhere(genes_names=='INS')
# plt.text(X[indx_INS[0,0],0],X[indx_INS[0,0],1],'INS',
#       rotation='horizontal',fontsize=12)
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

#%% plotting the top and bottom 40
# figure()
# stem(gcPCA_gene_space_sorted);xlabel('genes');ylabel('expression norm.');xlim((-1,40))
# for a in np.arange(20):
#     text(a,0.005,genes_names_sorted[a],rotation='vertical',fontsize=7)

# figure()
# stem(gcPCA_gene_space_sorted);xlabel('genes');ylabel('expression norm.');xlim((num[-1]-40,num[-1]))
# for a in np.arange(num[-1]-40,num[-1]+1):
#         text(a,-0.025,genes_names_sorted[a],rotation='vertical',fontsize=7)

# #%% running ncPCA
# #N1_trimmed = np.random.randn(10000,3000)
# #N2_trimmed = np.random.randn(10000,3000)
# N1_trimmed = stats.zscore(N_full[index_n1,genes_to_keep]).T
# N2_trimmed = stats.zscore(N_full[np.logical_not(index_n1),genes_to_keep]).T

# ncPCA_mdl = ncPCA(basis_type='intersect')
# ncPCA_mdl.fit(N1_trimmed,N2_trimmed)

# #X,S_total = ncPCA.ncPCA_orth(N1_trimmed,N2_trimmed,skip_normalization=False)

# cPCs = cPCA(N1_trimmed,N2_trimmed)
# cPCs_projected = np.dot(np.vstack((N1_trimmed,N2_trimmed)),cPCs)

# #%% making plots

# #plotting the N1 and N2 trimmed data to the top ranks
# figure()
# imshow(N1_trimmed,cmap='magma',clim=(-0.5,0.5));title('non-beta');xlabel('features');ylabel('cells')
# figure()
# imshow(N2_trimmed,cmap='magma',clim=(-0.5,0.5));title('beta');xlabel('features');ylabel('cells')

# #plotting ncPCA results, loadings and 'var'
# figure()
# plot(S_total);title('ncPCA ');xlabel('ncPCs')
# imshow(X,cmap='magma',clim=((-0.1,0.1)))

# # plot of ncPCA loadings, top 3
# figure()
# subplot(3,1,1);stem(X[:,0]);ylabel('loadgs ncPC1');
# subplot(3,1,2);stem(X[:,1]);ylabel('loadgs ncPC2');
# subplot(3,1,3);stem(X[:,2]);ylabel('loadgs ncPC2');
# xlabel('features')


# ##############################
# """ making a plot of spPCA projecting back on genes space to get the genes 
#  overexpressed or underexpressed
# """
# ##############################
# #%%
# #DOING FOR THE FIRST PC ONLY
# #projecting it back to gene space
# #ncPCA_gene_space = np.dot(X[:,-1].T,V[:,:200].T)
# ncPCA_gene_space = ncPCA_mdl.loadings_[:,6]
# genes_names = data.columns;
# genes_names_kept = genes_names[genes_to_keep]
# num = np.arange(len(ncPCA_gene_space))

# #sorting gene expression
# idx_sorting = np.argsort(ncPCA_gene_space);
# ncPCA_gene_space_sorted = ncPCA_gene_space[idx_sorting]
# genes_names_sorted = genes_names_kept[idx_sorting]
# indx_INS = np.argwhere(genes_names_sorted=='INS')

# stem(ncPCA_gene_space_sorted);xlabel('genes');ylabel('expression norm.');title('ncPC1')
# text(num[indx_INS[0,0]],ncPCA_gene_space_sorted[indx_INS[0,0]],'INS',
#       rotation='vertical',fontsize=7)
# #%%
# figure()
# stem(ncPCA_gene_space_sorted);xlabel('genes');ylabel('expression norm.');xlim((-1,40))
# for a in np.arange(40):
#     text(a,0.005,genes_names_sorted[a],rotation='vertical',fontsize=7)

# figure()
# stem(ncPCA_gene_space_sorted);xlabel('genes');ylabel('expression norm.');xlim((num[-1]-40,num[-1]))
# for a in np.arange(num[-1]-40,num[-1]+1):
#         text(a,-0.025,genes_names_sorted[a],rotation='vertical',fontsize=7)

# #%%
# #DOING FOR THE FIRST PC ONLY
# ncPCA_gene_space = X[:,0]
# genes_names = data.columns
# #sorting gene expression
# num = np.arange(len(genes_names_sorted))
# temp_gn = genes_names[top_genes_index]

# idx_sorting = np.argsort(ncPCA_gene_space)
# ncPCA_gene_space_sorted = ncPCA_gene_space[idx_sorting]
# genes_names_sorted = temp_gn[idx_sorting]

# stem(ncPCA_gene_space_sorted);xlabel('genes');ylabel('expression norm.');title('ncPC1')
# text(num[genes_names_sorted=='INS'],ncPCA_gene_space[genes_names_sorted=='INS'],'INS',
#      rotation='vertical',fontsize=7)
# #%%
# figure()
# stem(ncPCA_gene_space_sorted);xlabel('genes');ylabel('expression norm.');xlim((-1,40))
# for a in np.arange(20):
#     text(a,0.005,genes_names_sorted[a],rotation='vertical',fontsize=7)

# figure()
# stem(ncPCA_gene_space_sorted);xlabel('genes');ylabel('expression norm.');xlim((num[-1]-40,num[-1]))
# for a in np.arange(num[-1]-40,num[-1]+1):
#         text(a,-0.025,genes_names_sorted[a],rotation='vertical',fontsize=7)


# #%%
# beta_cells_PC = V2[0,:].T
# idx_sorting = np.argsort(beta_cells_PC);
# beta_cells_PC_sorted= beta_cells_PC[idx_sorting]
# genes_names_sorted = genes_names[idx_sorting]
# stem(beta_cells_PC_sorted);xlabel('genes');ylabel('expression norm.');title('Beta_cells_PC')
# text(num[genes_names_sorted=='INS'],beta_cells_PC_sorted[genes_names_sorted=='INS'],'INS',
#       rotation='vertical',fontsize=7)

# #%% same plot as above but cPC
# selected_cPCs = np.real(cPCs[:,1])

# idx_sorting = np.argsort(selected_cPCs);
# selected_cPCs_sorted = selected_cPCs[idx_sorting]
# genes_names_sorted = genes_names[idx_sorting]
# num = np.arange(len(genes_names_sorted))

# stem(selected_cPCs_sorted);xlabel('genes');ylabel('expression norm.');title('cPCs')
# text(num[genes_names_sorted=='INS'],selected_cPCs_sorted[genes_names_sorted=='INS'],'INS',
#       rotation='vertical',fontsize=7)

# #%% make plots on projections and class

# #getting index of each cell type to label with different plot
# #improve for readability later
# idx = (annotation.CellType.values!='Beta') & (annotation.Disease.values=='normal')
# reduced_values_n1 = annotation.CellType.values[idx]
# betalabels = annotation.CellType.values[(annotation.CellType.values=='Beta') & (annotation.Disease.values=='normal')]
# concat_ann = np.concatenate((reduced_values_n1,betalabels))
# colormap = cm.tab20


# c = 0
# labels_id = np.zeros(concat_ann.shape)
# str2legend = []
# for name in np.unique(concat_ann):
#     labels_id[concat_ann==name]=c
#     str2legend.append(name)
#     c += 1

# # plot the scores on projections from beta cells PC, cPCA and ncPCA
# #subplot(1,3,1)
# #proj_beta_PC = np.dot(N_reconstructed,V2_hat[(0,1),:].T)
# #aux_scat = scatter(proj_beta_PC[:,0],proj_beta_PC[:,1],c=labels_id,cmap=colormap,alpha=0.5)
# #handles, labels = aux_scat.legend_elements(prop="colors", alpha=0.6)
# #legend(handles,str2legend)
# #N2use = np.squeeze(N2use)
# subplot(1,2,1)
# #proj_cPCs = np.dot(N2use,np.real(cPCs[:,(0,1)]))
# scatter(cPCs_projected[:,-1],cPCs_projected[:,-2],c=labels_id,cmap=colormap,alpha=0.5)
# legend()
# title('cPCA')

# subplot(1,2,2)
# proj_ncPCs = np.vstack((ncPCA_mdl.N1_scores_,ncPCA_mdl.N2_scores_))
# #proj_ncPCs = np.dot(N2use,X[:,(936,937)])
# scatter(proj_ncPCs[:,-1],proj_ncPCs[:,-2],c=labels_id,cmap=colormap,alpha=0.5)
# #xlim((-0.5,0.5))
# #ylim((-0.5,0.5))
# title('ncPCs Last_dimensions')



# #%% make plot of variance in each dimension for each cell type

# data2plot = []
# for name in np.unique(concat_ann):
#     idx2use = concat_ann == name
#     #data2plot.append(np.var(proj_ncPCs[idx2use,:],axis=0)/np.sum(np.var(proj_ncPCs[idx2use,:],axis=0)))
#     data2plot.append(np.var(cPCs_projected[idx2use,:],axis=0))
#     print([str(np.sum(idx2use)) + name])

# data2plot = np.array(data2plot).T

# figure()
# for a in np.arange(data2plot.shape[1]):
#     subplot(13,1,a+1)
#     plot(data2plot[:,a],color=colormap(a),label=np.unique(concat_ann)[a])
#     legend(bbox_to_anchor=(1.04, 1))
#     if a == 0:
#         title('Beta vs Rest')
#         ylabel('Var')
# xlabel('cPCA')

