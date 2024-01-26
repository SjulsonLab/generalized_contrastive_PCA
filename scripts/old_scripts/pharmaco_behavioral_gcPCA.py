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
# repo_dir = "/home/eliezyer/Documents/github/generalized_contrastive_PCA/" #repository dir in linux machine
repo_dir = "C:\\Users\\fermi\\Documents\\GitHub\\generalized_contrastive_PCA" #repository dir in win laptop
# repo_dir =  #repo dir in HPC

sys.path.append(repo_dir)
from contrastive_methods import gcPCA

plt.rcParams.update({'figure.dpi':150, 'font.size':24})
#%% loading data


# data_dir = "/mnt/SSD4TB/ncPCA_files/behavioral/pharmacobehavioral/" #data dir in linux machine
data_dir = 'C:\\Users\\fermi\\Dropbox\\preprocessing_data\\gcPCA_files\\behavioral\\pharmacobehavioral\\'  #data dir in win laptop
fid = open(data_dir + 'fingerprints.pkl','rb')
fingerprints, fingerprint_labels = pickle.load(fid,encoding='latin1')
x = fingerprints['moseq']
behavior_data = fingerprints['moseq']
#loading syllables labels
fid = open(data_dir + 'syllablelabels.pkl','rb')
syllables = pickle.load(fid,encoding='latin1')

drug_labels = fingerprint_labels['y_drug']
highlow_label = fingerprint_labels['highorlow']
dose_labels = fingerprint_labels['dose']

# getting labels of unique class and drug

drug_class = fingerprint_labels['drug_class']
unique_class = [b for a,b in zip([""]+drug_class,drug_class) if b!=a]
drug = fingerprint_labels['drug']
unique_drug = [b for a,b in zip([""]+drug,drug) if b!=a]

# separating control and drugs data
control_data = x[drug_labels==15,:]
drugs_data   = x[drug_labels!=15,:]
# drugs_data   = x[drug_labels==11,:]
#picking only high doses
temp_dose_labels = np.array(dose_labels)
temp_hd = np.zeros(drugs_data.shape[0])
for y in np.unique(drug_labels):
    if y<15:
        idx = np.argwhere(drug_labels == y)
        indexes = idx.astype(int).flatten()
        temp_dl = np.char.equal([highlow_label[index] for index in indexes],'High')
        # temp_dl = temp_dose_labels[idx] == np.min(temp_dose_labels[idx])
        temp_hd[idx[temp_dl]] = 1
#%% preparing data for gcPCA

# drugs_data = drugs_data[temp_hd.astype(bool)]
# fitting gcpca
gcpca_mdl = gcPCA(method='v4')
gcpca_mdl.fit(drugs_data,control_data)

# lda = LinearDiscriminantAnalysis(n_components=2).fit(fingerprints['moseq'], fingerprint_labels['y_drug'])

#%%
sort_id = np.argsort(gcpca_mdl.Ra_scores_[:,0])
sort_feat = np.argsort(gcpca_mdl.loadings_[:,0])
# test = np.outer((gcpca_mdl.Ra_scores_[:,0]*gcpca_mdl.Ra_values_[0]),gcpca_mdl.loadings_[:,0].T)
test = zscore(drugs_data)
temp = test[sort_id,:]
sort_test = temp[:,sort_feat]
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
y_drug0 = fingerprint_labels['y_drug']
y_drug = y_drug0[y_drug0!=15]
# y_drug = y_drug[temp_hd.astype(bool)]

color_map = dict(zip(unique_class, sns.color_palette(palette='Paired', n_colors=len(unique_class))))
class_per_drug = []

colors_ = []
for drug_ in unique_drug:
    class_ = np.array(drug_class)[np.array(drug)==drug_][0]
    colors_.append(color_map[class_])

plt.figure(num=1,figsize=(15,7))
# U,S,V = np.linalg.svd(zscore(drugs_data),full_matrices=False)
p = gcpca_mdl.Ra_scores_
# p = U
for i, treatment in enumerate(unique_drug[:-1]):
    # plt.figure()
    idx = y_drug == i
    plt.plot(p[idx,0], p[idx, 1], 'o', alpha=0.4, color=colors_[i])
    x, y = np.median(p[idx,0:2], axis=0)
    plt.plot(x, y, 'o',markersize=15, color=colors_[i], label=short_name_map[treatment])
    plt.text(x + 0.001, y, short_name_map[treatment], fontsize=14, verticalalignment='center')

sns.despine()
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
 # short_name_map[treatment], fontsize=10, verticalalignment='center', )
# ylim(-5, 5)
# xlim(-5, 9)
# xlabel('LDA 1')
# ylabel('LDA 2')

#%%
# plt.figure(num=11,figsize=(15,7))
p = gcpca_mdl.Ra_scores_
np_highlow_label = np.array(highlow_label)[y_drug0!=15]

# p = U
markers = ['o','v','^','>','<','s'] 
marker_size = [3,6,9,12,15,18]
for i, treatment in enumerate(unique_drug[:-1]):
    plt.figure()
    idx = y_drug == i
    for j,a in enumerate(np.unique(np_highlow_label[idx])):
        idx_dose = np_highlow_label[idx] == a
        x0, y0 = p[idx,0], p[idx, 1]
        plt.plot(x0[idx_dose], y0[idx_dose], 'o', alpha=0.7, color=colors_[i],
                 markersize=marker_size[j])
    x, y = np.median(p[idx,0:2], axis=0)
    plt.plot(x, y, 's',markersize=15, color='grey', label=short_name_map[treatment])
    plt.text(x + 0.005, y, short_name_map[treatment], fontsize=14, verticalalignment='center')

sns.despine()
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')

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

plt.figure(num=2,figsize=(5,14))
ax = plt.subplot(16,1,1)
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.stem(gcpca_mdl.loadings_[temp,0])
plt.title('1st gcPC')

for i, treatment in enumerate(unique_drug[:-1]):
    plot_i = np.argwhere(drug_sort==i)[0][0]
    ax = plt.subplot(16,1,plot_i+2)

    idx = y_drug == i
    plt.plot(np.mean(drugs_data[idx,temp].T,axis=0),color=colors_[i])
    plt.ylabel(short_name_map[treatment],fontsize=14)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # plt.ylim((-0.022,0.022))
    sns.despine()

# plotting gcPC2
temp = np.argsort(gcpca_mdl.loadings_[:,1])
temp = temp[:,np.newaxis]
drug_sort = np.argsort(avg_proj[:,1])
plt.figure(num=3,figsize=(5,14))
ax = plt.subplot(16,1,1)
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.stem(gcpca_mdl.loadings_[temp,1])
plt.title('2nd gcPC')

drugs_data_proj = np.outer(gcpca_mdl.Ra_scores_[:,1],gcpca_mdl.loadings_[:,1].T)
for i, treatment in enumerate(unique_drug[:-1]):
    plot_i = np.argwhere(drug_sort==i)[0][0]
    ax=plt.subplot(16,1,plot_i+2)

    idx = y_drug == i
    plt.plot(np.mean(drugs_data[idx,temp].T,axis=0),color=colors_[i])
    plt.ylabel(short_name_map[treatment],fontsize=14)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # plt.ylim((-0.027,0.027))
    sns.despine()

# plotting gcPC3
temp = np.argsort(gcpca_mdl.loadings_[:,2])
temp = temp[:,np.newaxis]
drug_sort = np.argsort(avg_proj[:,2])

# plt.figure(num=4,figsize=(2,7))
# ax = plt.subplot(16,1,1)
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# plt.stem(gcpca_mdl.loadings_[temp,2])
# plt.title('3rd gcPC')

# drugs_data_proj = np.outer(gcpca_mdl.Ra_scores_[:,2],gcpca_mdl.loadings_[:,2].T)
# for i, treatment in enumerate(unique_drug[:-1]):
#     plot_i = np.argwhere(drug_sort==i)[0][0]
#     ax=plt.subplot(16,1,plot_i+2)

#     idx = y_drug == i
#     plt.plot(np.mean(drugs_data[idx,temp].T,axis=0),color=colors_[i])
#     plt.ylabel(short_name_map[treatment],fontsize=8)
#     ax.set_yticklabels([])
#     ax.set_xticklabels([])
#     # plt.ylim((-0.027,0.027))
#     sns.despine()

#%% running gcPCA all doses haloperidol vs all doses risperidone

drug = fingerprint_labels['drug']
unique_drug = [b for a,b in zip([""]+drug,drug) if b!=a]
behavior_data = fingerprints['moseq']

# separating control and drugs data
halo_idx = np.logical_and(np.char.equal(drug,'haloperidol'),np.isin(highlow_label,['Low','Medium','High','Very High']))
halo_data = behavior_data[halo_idx,:]
risp_idx = np.logical_and(np.char.equal(drug,'risperidone'),np.isin(highlow_label,['Low','Medium','High','Very High']))
risp_data = behavior_data[risp_idx,:]

gcpca_mdl = gcPCA(method='v4')
gcpca_mdl.fit(risp_data,halo_data)

Ra_scores = gcpca_mdl.Ra_scores_
Rb_scores = gcpca_mdl.Rb_scores_

#%% make plot by dose top gcPCs

halo_dose_label = np.array(highlow_label)[halo_idx]
risp_dose_label = np.array(highlow_label)[risp_idx]
dose_order = ['Low','Medium','High','Very High']
# first plotting haloperidol
# marker_color_halo = ['lightgreen','limegreen','mediumseagreen','darkgreen']
# plt.figure()
# x0, y0 = Rb_scores[:,0], Rb_scores[:, 1]
# for j,a in enumerate(dose_order):
#     idx_dose = halo_dose_label == a
#     plt.plot((x0[idx_dose]), (y0[idx_dose]), 'o', alpha=0.8, color=marker_color_halo[j],
#                   markersize=10,label=a)
# x, y = np.median(Rb_scores[:,0]), np.median(Rb_scores[:,1])
# plt.plot(x, y, 'o',markersize=15, color='grey')
# plt.text(x + 0.005, y, 'HALO', fontsize=14, verticalalignment='center',weight='bold')

marker_color_risp = ['lightblue','dodgerblue','royalblue','darkblue']
# plotting risperidone now
x0, y0 = Ra_scores[:,0], Ra_scores[:, 1]
for j,a in enumerate(dose_order):
    idx_dose = risp_dose_label == a
    plt.plot((x0[idx_dose]), (y0[idx_dose]), 's', alpha=0.8, color=marker_color_risp[j],
                 markersize=10,label=a)
x, y = np.median(Ra_scores[:,0]), np.median(Ra_scores[:,1])
plt.plot(x, y, 's',markersize=15, color='grey')
plt.text(x + 0.005, y, 'RISP', fontsize=14, verticalalignment='center',weight='bold')

sns.despine()
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
plt.title('top gcPCS - RISP gcPCs')
# plt.legend()

#%% make plot by dose of bottom gcPCs
halo_dose_label = np.array(highlow_label)[halo_idx]
risp_dose_label = np.array(highlow_label)[risp_idx]
dose_order = ['Low','Medium','High','Very High']
# first plotting haloperidol
# marker_size = [3,6,9,12]
marker_color_halo = ['palegreen','springgreen','mediumseagreen','darkgreen']
plt.figure()
# idx = y_drug == i
x0, y0 = Rb_scores[:,-1], Rb_scores[:, -2]
for j,a in enumerate(dose_order):
    idx_dose = halo_dose_label == a
    plt.plot((x0[idx_dose]), (y0[idx_dose]), 'o', alpha=0.8, color=marker_color_halo[j],
                 markersize=10,label=a)
x, y = np.median(Rb_scores[:,-1]), np.median(Rb_scores[:,-2])
plt.plot(x, y, 'o',markersize=15, color='grey')
plt.text(x + 0.005, y, 'HALO', fontsize=14, verticalalignment='center',weight='bold')

# marker_color_risp = ['lightblue','dodgerblue','royalblue','darkblue']
# # plotting risperidone now
# x0, y0 = Ra_scores[:,-1], Ra_scores[:, -2]
# for j,a in enumerate(dose_order):
#     idx_dose = risp_dose_label == a
#     plt.plot((x0[idx_dose]), (y0[idx_dose]), 's', alpha=0.8, color=marker_color_risp[j],
#                   markersize=10,label=a)
# x, y = np.median(Ra_scores[:,-1]), np.median(Ra_scores[:,-2])
# plt.plot(x, y, 's',markersize=15, color='grey')
# plt.text(x + 0.005, y, 'RISP', fontsize=14, verticalalignment='center',weight='bold')

sns.despine()
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
plt.title('bot gcPCS - HALO gcPCs')
# plt.legend()

#%% plotting PCA - risp

data_risp = zscore(risp_data)
data_risp[np.isnan(data_risp)] = 0
U_risp,S,V_risp = np.linalg.svd(data_risp,full_matrices=False)
plt.figure()
marker_color_risp = ['lightblue','dodgerblue','royalblue','darkblue']
# plotting risperidone now
score_risp = data_risp@V_risp.T
x0, y0 = score_risp[:,0], score_risp[:, 1]
for j,a in enumerate(dose_order):
    idx_dose = risp_dose_label == a
    plt.plot((x0[idx_dose]), (y0[idx_dose]), 's', alpha=0.8, color=marker_color_risp[j],
                 markersize=10,label=a)
x, y = np.median(score_risp[:,0]), np.median(score_risp[:,1])
plt.plot(x, y, 's',markersize=15, color='grey')
plt.text(x + 0.005, y, 'RISP', fontsize=14, verticalalignment='center',weight='bold')

sns.despine()
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('top RISP PCs')

#%% plotting PCA halo
data_halo =zscore(halo_data)
data_halo[np.isnan(data_halo)] = 0
U_halo,S,V_halo = np.linalg.svd(data_halo,full_matrices=False)
score_halo_on_risp = data_halo@V_halo.T
plt.figure()
marker_color_risp = ['lightblue','dodgerblue','royalblue','darkblue']

x0, y0 = score_halo_on_risp[:,0], score_halo_on_risp[:, 1]
for j,a in enumerate(dose_order):
    idx_dose = halo_dose_label == a
    plt.plot((x0[idx_dose]), (y0[idx_dose]), 'o', alpha=0.8, color=marker_color_halo[j],
                 markersize=10,label=a)
x, y = np.median(score_halo_on_risp[:,0]), np.median(score_halo_on_risp[:,1])
plt.plot(x, y, 'o',markersize=15, color='grey')
plt.text(x + 0.005, y, 'HALO', fontsize=14, verticalalignment='center',weight='bold')

sns.despine()
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('top HALO PCs')