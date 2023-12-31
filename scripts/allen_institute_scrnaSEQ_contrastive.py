#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 20:56:20 2023
 script to load rna seq and 
@author: eliezyer
"""

#importing essentials
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns

repo_dir = "/home/eliezyer/Documents/github/normalized_contrastive_PCA/" #repository dir in linux machine
#repo_dir = "C:\\Users\\fermi\\Documents\\GitHub\\normalized_contrastive_PCA" #repository dir in win laptop
# repo_dir =  #repo dir in HPC

sys.path.append(repo_dir)
from contrastive_methods import gcPCA

#%% important functions to preprocess the scrnaSeq

# def feature_reduction(data_matrix,n_reduction):
#     """This function will remove all the genes that are zeros and then only keep
#     the top genes (based on dispersion), where the number of genes kept are 
#     given by n_reduction"""
    
#     original_index = np.arange(data_matrix.shape[1])
#     indexes = original_index.copy()
#     #first getting all the non-zeros genes
#     index_non_zero = data_matrix.sum(axis=0)>0 
#     data_matrix    = data_matrix[:,index_non_zero]
#     indexes        = indexes[index_non_zero]
    
#     #calculating the dispersion
#     dispersion = data_matrix.std(axis=0)/data_matrix.mean(axis=0)
#     index_reduced = np.argpartition(dispersion, -n_reduction)[-n_reduction:].flatten()
    
#     #final arrangements
#     reduced_data = data_matrix[:,index_reduced]
#     indexes      = indexes[index_reduced]
#     return reduced_data,indexes


# def feature_reduction2(data_matrix):
#     from scipy.stats import zscore
#     import numpy as np
#     """This function will remove all the genes that are zeros and then only keep
#     the top PCs (based on rank of the matrix)."""
    
#     original_index = np.arange(data_matrix.shape[1])
#     indexes = original_index.copy()
#     #first getting all the non-zeros genes
#     index_non_zero = data_matrix.sum(axis=0)>0 
#     data_matrix    = data_matrix[:,index_non_zero]
#     indexes        = indexes[index_non_zero]
    
#     zsc_data = zscore(data_matrix)
#     U,S,V = np.linalg.svd(zsc_data,full_matrices=False)
    
#     reduced_data = np.linalg.multi_dot((zsc_data,V.T,V))
#     return reduced_data,indexes
#%% loading data

data_dir = "/mnt/SSD4TB/ncPCA_files/allen_RNA_Seq/" #data dir in linux machine
# data_dir = "C:\\Users\\fermi\\Data\\allen_RNA_Seq\\"  #data dir in win laptop
# data_dir = #data dir in HPC

temp_visp = pd.read_csv(data_dir+'mouse_VISp_2018-06-14_exon-matrix.csv')
visp_count = temp_visp.T.values

temp_alm = pd.read_csv(data_dir+'mouse_ALM_2018-06-14_exon-matrix.csv')
alm_count = temp_alm.T.values

# temp_count = np.concatenate((visp_count,alm_count),axis=0)
#reducing the number of feature to the 10k top variance features. This might be problematic
# reduced_count,indexes = feature_reduction(temp_count, 10000)
# reduced_count,indexes = feature_reduction2(temp_count)

# new_visp_count = reduced_count[:visp_count.shape[0],:]
# new_alm_count  = reduced_count[visp_count.shape[0]:,:]

gcpca_mdl = gcPCA(method='v4.1',Ncalc = 10)
gcpca_mdl.fit(visp_count,alm_count)

#%% post analysis of index ncPCA
#get the name of the 