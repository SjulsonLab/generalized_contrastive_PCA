# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 11:35:36 2023

script to run gcPCA on the behavioral dataset for analysis

@author: eliezyer
"""


#importing essentials
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.stats import zscore
import pickle
import seaborn as sns
from collections import OrderedDict
# repo_dir = "/home/eliezyer/Documents/github/normalized_contrastive_PCA/" #repository dir in linux machine
repo_dir = "C:\\Users\\fermi\\Documents\\GitHub\\normalized_contrastive_PCA" #repository dir in win laptop
# repo_dir =  #repo dir in HPC

sys.path.append(repo_dir)
from contrastive_methods import gcPCA


#%% loading data

# data_dir = "/mnt/SSD4TB/ncPCA_files/allen_RNA_Seq/" #data dir in linux machine
data_dir = 'C:\\Users\\fermi\\Dropbox\\preprocessing_data\\gcPCA_files\\behavioral\\pharmacobehavioral\\'  #data dir in win laptop
fid = open(data_dir + 'fingerprints.pkl','rb')
fingerprints, fingerprint_labels = pickle.load(fid,encoding='latin1')
x = fingerprints['moseq']
x1 = fingerprints['moseq']
#loading syllables labels
fid = open(data_dir + 'syllablelabels.pkl','rb')
syllables = pickle.load(fid,encoding='latin1')

drug_labels = fingerprint_labels['y_drug']
dose_labels = fingerprint_labels['dose']

# getting labels of unique class and drug

drug_class = fingerprint_labels['drug_class']
unique_class = [b for a,b in zip([""]+drug_class,drug_class) if b!=a]
drug = fingerprint_labels['drug']
unique_drug = [b for a,b in zip([""]+drug,drug) if b!=a]

# separating control and drugs data
control_data = x[drug_labels==15,:]
drugs_data   = x[drug_labels!=15,:]
#picking only high doses
temp_dose_labels = np.array(dose_labels)
temp_hd = np.zeros(drugs_data.shape[0])#high dose
for y in np.unique(drug_labels):
    if y<15:
        idx = np.argwhere(drug_labels == y)
        # temp_dl = temp_dose_labels[idx] == np.max(temp_dose_labels[idx])
        temp_dl = temp_dose_labels[idx] > 0.24
        temp_hd[idx[temp_dl]] = 1
#%% preparing data for gcPCA

# drugs_data = drugs_data[temp_hd.astype(bool)]
# fitting gcpca
gcpca_mdl = gcPCA(method='v4')
gcpca_mdl.fit(drugs_data,control_data)

# lda = LinearDiscriminantAnalysis(n_components=2).fit(fingerprints['moseq'], fingerprint_labels['y_drug'])


#%% preparing for plot

short_name_map = OrderedDict([
    ('alprazolam','ALPR'),
    ('atomoxetine','ATOM'),
    ('bupropion','BUPR'),
    ('chlorpromazine','CHLO'),
    ('citalopram','CITA'),
    ('clozapine','CLOZ'),
    ('control','CTRL'),
    ('diazepam','DIAZ'),
    ('fluoxetine','FLUO'),
    ('haloperidol','HALO'),
    ('methamphetamine','METH'),
    ('methylphenidate','MTPH'),
    ('modafinil','MODA'),
    ('phenelzine','PHEN'),
    ('risperidone','RISP'),
    ('venlafaxine','VENL'),
])
#getting drug as number labels, removing the last one that corresponds to control
y_drug = fingerprint_labels['y_drug']
y_drug = y_drug[y_drug!=15]
# y_drug = y_drug[temp_hd.astype(bool)]

color_map = dict(zip(unique_class, sns.color_palette(palette='Paired', n_colors=len(unique_class))))
class_per_drug = []

colors_ = []
for drug_ in unique_drug:
    class_ = np.array(drug_class)[np.array(drug)==drug_][0]
    colors_.append(color_map[class_])

plt.figure()
p = gcpca_mdl.Ra_scores_
for i, treatment in enumerate(unique_drug[:-1]):
    idx = y_drug == i
    plt.plot(p[idx,0], p[idx, 1], 'o', alpha=0.2, color=colors_[i])
    x, y = np.mean(p[idx,0:2], axis=0)
    plt.plot(x, y, 'o',markersize=15, color=colors_[i], label=short_name_map[treatment])
    plt.text(x + 0.001, y, short_name_map[treatment], fontsize=10, verticalalignment='center', )
    
sns.despine()
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
 # short_name_map[treatment], fontsize=10, verticalalignment='center', )
# ylim(-5, 5)
# xlim(-5, 9)
# xlabel('LDA 1')
# ylabel('LDA 2');

#%% plotting the weights
# plt.figure()
# p1 =np.dot( gcpca_mdl.Ra_scores_[:,:2],gcpca_mdl.loadings_[:,:2].T)
# y = np.mean(p1[idx,:], axis=0)
# plt.plot(y)
#%% plot of drugs to gcPCA


# get average their projection on gcPC
avg_proj = []
for i, treatment in enumerate(unique_drug[:-1]):
    idx = y_drug == i
    avg_proj.append(np.mean(gcpca_mdl.Ra_scores_[idx,:3],axis=0))
avg_proj= np.array(avg_proj)

# plotting gcPC1
drugs_data_proj = np.outer(gcpca_mdl.Ra_scores_[:,0],gcpca_mdl.loadings_[:,0].T)
temp = np.argsort(gcpca_mdl.loadings_[:,0])
temp = temp[:,np.newaxis]
drug_sort = np.argsort(avg_proj[:,0])
plt.figure(figsize=(2,7))
ax = plt.subplot(16,1,1)
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.stem(gcpca_mdl.loadings_[temp,0])

for i, treatment in enumerate(unique_drug[:-1]):
    plot_i = np.argwhere(drug_sort==i)[0][0]
    ax = plt.subplot(16,1,plot_i+2)
    
    idx = y_drug == i
    plt.plot(np.mean(drugs_data[idx,temp].T,axis=0),color=colors_[i])
    plt.ylabel(short_name_map[treatment],fontsize=8)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # plt.ylim((-0.022,0.022))
    sns.despine()
    
# plotting gcPC2
temp = np.argsort(gcpca_mdl.loadings_[:,1])
temp = temp[:,np.newaxis]
drug_sort = np.argsort(avg_proj[:,1])
plt.figure(figsize=(2,7))
ax = plt.subplot(16,1,1)
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.stem(gcpca_mdl.loadings_[temp,1])

drugs_data_proj = np.outer(gcpca_mdl.Ra_scores_[:,1],gcpca_mdl.loadings_[:,1].T)
for i, treatment in enumerate(unique_drug[:-1]):
    plot_i = np.argwhere(drug_sort==i)[0][0]
    ax=plt.subplot(16,1,plot_i+2)
    
    idx = y_drug == i
    plt.plot(np.mean(drugs_data[idx,temp].T,axis=0),color=colors_[i])
    plt.ylabel(short_name_map[treatment],fontsize=8)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # plt.ylim((-0.027,0.027))
    sns.despine()
    
# plotting gcPC3
temp = np.argsort(gcpca_mdl.loadings_[:,2])
temp = temp[:,np.newaxis]
drug_sort = np.argsort(avg_proj[:,2])
plt.figure(figsize=(2,7))
ax = plt.subplot(16,1,1)
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.stem(gcpca_mdl.loadings_[temp,2])

drugs_data_proj = np.outer(gcpca_mdl.Ra_scores_[:,2],gcpca_mdl.loadings_[:,2].T)
for i, treatment in enumerate(unique_drug[:-1]):
    plot_i = np.argwhere(drug_sort==i)[0][0]
    ax=plt.subplot(16,1,plot_i+2)
    
    idx = y_drug == i
    plt.plot(np.mean(drugs_data[idx,temp].T,axis=0),color=colors_[i])
    plt.ylabel(short_name_map[treatment],fontsize=8)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # plt.ylim((-0.027,0.027))
    sns.despine()
    

#%% running tsne on projected data
drugs_data_red = np.outer(gcpca_mdl.Ra_scores_[:,0],gcpca_mdl.loadings_[:,0].T) + np.outer(gcpca_mdl.Ra_scores_[:,1],gcpca_mdl.loadings_[:,1].T)
X_embedded = TSNE(n_components=2, learning_rate='auto',
                   init='random', perplexity=10).fit_transform(drugs_data_red)

p1=X_embedded
plt.figure()
for i, treatment in enumerate(unique_drug[:-1]):
    idx = y_drug == i
    plt.plot(p1[idx,0], p1[idx, 1], 'o', alpha=0.2, color=colors_[i])
    x, y = np.mean(p1[idx,0:2], axis=0)
    plt.plot(x, y, 'o',markersize=15, color=colors_[i], label=short_name_map[treatment])
    plt.text(x + 0.001, y, short_name_map[treatment], fontsize=10, verticalalignment='center', )
    