# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:26:15 2024

@author: fermi

comparing most changing vectors of risperidone and haloperidol against each other
with gcPCA
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
import os
# repo_dir = "/home/eliezyer/Documents/github/generalized_contrastive_PCA/" #repository dir in linux machine
# repository dir
repo_dir = r'C:\Users\fermi\Documents\GitHub\generalized_contrastive_PCA'
sys.path.append(repo_dir)
from contrastive_methods import gcPCA

save_fig_path = r'C:\Users\fermi\Dropbox\figures_gcPCA\pharmacobehavioral'

plt.rcParams.update({'figure.dpi':150, 'font.size':16,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
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

plt.figure(figsize=(7,7))
marker_color_risp = ['lightblue','dodgerblue','royalblue','darkblue']
# plotting risperidone now
x0, y0 = Ra_scores[:,0], Ra_scores[:, 1]
for j,a in enumerate(dose_order):
    idx_dose = risp_dose_label == a
    plt.plot((x0[idx_dose]), (y0[idx_dose]), 's', alpha=0.8, color=marker_color_risp[j],
                 markersize=15,label=a)
x, y = np.median(Ra_scores[:,0]), np.median(Ra_scores[:,1])
plt.plot(x, y, 's',markersize=25, color='grey')
plt.text(x + 0.005, y, 'RISP', fontsize=14, verticalalignment='center',weight='bold')

sns.despine()
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
plt.title('top gcPCS - RISP gcPCs')
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(save_fig_path,"risperidone_gcpcs.pdf"), format="pdf")

#%% make plot by dose of bottom gcPCs
halo_dose_label = np.array(highlow_label)[halo_idx]
risp_dose_label = np.array(highlow_label)[risp_idx]
dose_order = ['Low','Medium','High','Very High']
# first plotting haloperidol
# marker_size = [3,6,9,12]
marker_color_halo = ['palegreen','springgreen','mediumseagreen','darkgreen']
plt.figure(figsize=(7,7))
# idx = y_drug == i
x0, y0 = Rb_scores[:,-1], Rb_scores[:, -2]
for j,a in enumerate(dose_order):
    idx_dose = halo_dose_label == a
    plt.plot((x0[idx_dose]), (y0[idx_dose]), 'o', alpha=0.8, color=marker_color_halo[j],
                 markersize=15,label=a)
x, y = np.median(Rb_scores[:,-1]), np.median(Rb_scores[:,-2])
plt.plot(x, y, 'o',markersize=25, color='grey')
plt.text(x + 0.005, y, 'HALO', fontsize=14, verticalalignment='center',weight='bold')

sns.despine()
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
plt.title('bot gcPCS - HALO gcPCs')
plt.tight_layout()
plt.legend()

plt.savefig(os.path.join(save_fig_path,"haloperidol_gcpcs.pdf"), format="pdf")

#%% plotting PCA - risp

# data_risp = zscore(risp_data)
# data_risp[np.isnan(data_risp)] = 0
# U_risp,S,V_risp = np.linalg.svd(data_risp,full_matrices=False)
# plt.figure()
# marker_color_risp = ['lightblue','dodgerblue','royalblue','darkblue']
# # plotting risperidone now
# score_risp = data_risp@V_risp.T
# x0, y0 = score_risp[:,0], score_risp[:, 1]
# for j,a in enumerate(dose_order):
#     idx_dose = risp_dose_label == a
#     plt.plot((x0[idx_dose]), (y0[idx_dose]), 's', alpha=0.8, color=marker_color_risp[j],
#                  markersize=10,label=a)
# x, y = np.median(score_risp[:,0]), np.median(score_risp[:,1])
# plt.plot(x, y, 's',markersize=15, color='grey')
# plt.text(x + 0.005, y, 'RISP', fontsize=14, verticalalignment='center',weight='bold')

# sns.despine()
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('top RISP PCs')


#%% plotting PCA halo
# data_halo =zscore(halo_data)
# data_halo[np.isnan(data_halo)] = 0
# U_halo,S,V_halo = np.linalg.svd(data_halo,full_matrices=False)
# score_halo_on_risp = data_halo@V_halo.T
# plt.figure()
# marker_color_risp = ['lightblue','dodgerblue','royalblue','darkblue']

# x0, y0 = score_halo_on_risp[:,0], score_halo_on_risp[:, 1]
# for j,a in enumerate(dose_order):
#     idx_dose = halo_dose_label == a
#     plt.plot((x0[idx_dose]), (y0[idx_dose]), 'o', alpha=0.8, color=marker_color_halo[j],
#                  markersize=10,label=a)
# x, y = np.median(score_halo_on_risp[:,0]), np.median(score_halo_on_risp[:,1])
# plt.plot(x, y, 'o',markersize=15, color='grey')
# plt.text(x + 0.005, y, 'HALO', fontsize=14, verticalalignment='center',weight='bold')

# sns.despine()
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('top HALO PCs')


#%% running gcPCA reversed

# drug = fingerprint_labels['drug']
# unique_drug = [b for a,b in zip([""]+drug,drug) if b!=a]
# behavior_data = fingerprints['moseq']

# # separating control and drugs data
# halo_idx = np.logical_and(np.char.equal(drug,'haloperidol'),np.isin(highlow_label,['Low','Medium','High','Very High']))
# halo_data = behavior_data[halo_idx,:]
# risp_idx = np.logical_and(np.char.equal(drug,'risperidone'),np.isin(highlow_label,['Low','Medium','High','Very High']))
# risp_data = behavior_data[risp_idx,:]

# gcpca_mdl = gcPCA(method='v4')
# gcpca_mdl.fit(halo_data,risp_data)


# Ra_scores = gcpca_mdl.Ra_scores_
# Rb_scores = gcpca_mdl.Rb_scores_

# #% make plot by dose top gcPCs

# halo_dose_label = np.array(highlow_label)[halo_idx]
# risp_dose_label = np.array(highlow_label)[risp_idx]
# dose_order = ['Low','Medium','High','Very High']
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

# plt.figure()
# marker_color_risp = ['lightblue','dodgerblue','royalblue','darkblue']
# # plotting risperidone now
# x0, y0 = Rb_scores[:,-1], Rb_scores[:, -2]
# for j,a in enumerate(dose_order):
#     idx_dose = risp_dose_label == a
#     plt.plot((x0[idx_dose]), (y0[idx_dose]), 's', alpha=0.8, color=marker_color_risp[j],
#                  markersize=10,label=a)
# x, y = np.median(Rb_scores[:,-1]), np.median(Rb_scores[:,-2])
# plt.plot(x, y, 's',markersize=15, color='grey')
# plt.text(x + 0.005, y, 'RISP', fontsize=14, verticalalignment='center',weight='bold')

# sns.despine()
# plt.xlabel('gcPC1')
# plt.ylabel('gcPC2')
# plt.title('bottom gcPCS - RISP gcPCs')
# plt.legend()

# # make plot by dose of bottom gcPCs
# halo_dose_label = np.array(highlow_label)[halo_idx]
# risp_dose_label = np.array(highlow_label)[risp_idx]
# dose_order = ['Low','Medium','High','Very High']
# # first plotting haloperidol
# # marker_size = [3,6,9,12]
# marker_color_halo = ['palegreen','springgreen','mediumseagreen','darkgreen']
# plt.figure()
# # idx = y_drug == i
# x0, y0 = Ra_scores[:,0], Ra_scores[:, 1]
# for j,a in enumerate(dose_order):
#     idx_dose = halo_dose_label == a
#     plt.plot((x0[idx_dose]), (y0[idx_dose]), 'o', alpha=0.8, color=marker_color_halo[j],
#                  markersize=10,label=a)
# x, y = np.median(Ra_scores[:,0]), np.median(Ra_scores[:,1])
# plt.plot(x, y, 'o',markersize=15, color='grey')
# plt.text(x + 0.005, y, 'HALO', fontsize=14, verticalalignment='center',weight='bold')

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

# sns.despine()
# plt.xlabel('gcPC1')
# plt.ylabel('gcPC2')
# plt.title('top gcPCS - HALO gcPCs')
# # plt.legend()
