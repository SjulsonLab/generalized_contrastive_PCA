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

plt.rcParams.update({'figure.dpi':150, 'font.size':18,
                     'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.linewidth': 3})
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

plt.figure(figsize=(6,6))
plt.plot([-1,1],[0,0],'-',c='k',linewidth=2)
plt.plot([0,0],[-1,1],'-',c='k',linewidth=2)
marker_color_risp = ['lightblue','dodgerblue','royalblue','darkblue']
# plotting risperidone now
x0, y0 = Ra_scores[:,0], Ra_scores[:, 1]
for j,a in enumerate(dose_order):
    idx_dose = risp_dose_label == a
    plt.plot((x0[idx_dose]), (y0[idx_dose]), 's', alpha=0.8, color=marker_color_risp[j],
                 markersize=15,label=a)
plt.xlim((-0.5,0.5))
plt.ylim((-0.5,0.5))
plt.xticks([-0.3,0,0.3])
plt.yticks([-0.3,0,0.3])
plt.grid()
# x, y = np.median(Ra_scores[:,0]), np.median(Ra_scores[:,1])
# plt.plot(x, y, 's',markersize=25, color='grey')
# plt.text(x + 0.005, y, 'RISP', fontsize=14, verticalalignment='center',weight='bold')

sns.despine()
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
plt.title('Risperidone gcPCs')
plt.tight_layout()
plt.legend(title='Dose')
plt.savefig(os.path.join(save_fig_path,"risperidone_gcpcs.pdf"), format="pdf")

#%% make plot by dose of bottom gcPCs
halo_dose_label = np.array(highlow_label)[halo_idx]
risp_dose_label = np.array(highlow_label)[risp_idx]
dose_order = ['Low','Medium','High','Very High']
# first plotting haloperidol
# marker_size = [3,6,9,12]
marker_color_halo = ['palegreen','springgreen','mediumseagreen','darkgreen']
plt.figure(figsize=(6,6))
plt.plot([-1,1],[0,0],'-',c='k',linewidth=2)
plt.plot([0,0],[-1,1],'-',c='k',linewidth=2)
# idx = y_drug == i
x0, y0 = Rb_scores[:,-1], Rb_scores[:, -2]
for j,a in enumerate(dose_order):
    idx_dose = halo_dose_label == a
    plt.plot((x0[idx_dose]), (y0[idx_dose]), 'o', alpha=0.8, color=marker_color_halo[j],
                 markersize=15,label=a)
# x, y = np.median(Rb_scores[:,-1]), np.median(Rb_scores[:,-2])
# plt.plot(x, y, 'o',markersize=25, color='grey')
# plt.text(x + 0.005, y, 'HALO', fontsize=14, verticalalignment='center',weight='bold')

sns.despine()
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
plt.title('Haloperidol gcPCs')
plt.tight_layout()
plt.legend(title='Dose')
plt.xlim((-0.4,0.4))
plt.ylim((-0.4,0.4))
plt.xticks([-0.3,0,0.3])
plt.yticks([-0.3,0,0.3])
plt.grid()

plt.savefig(os.path.join(save_fig_path,"haloperidol_gcpcs.pdf"), format="pdf")

#%% plot of HALO gcPCs and highest doses of RISP

halo_dose_label = np.array(highlow_label)[halo_idx]
risp_dose_label = np.array(highlow_label)[risp_idx]
dose_order = ['Low','Medium','High','Very High']
# first plotting haloperidol
# marker_size = [3,6,9,12]
marker_color_halo = ['palegreen','springgreen','mediumseagreen','darkgreen']
plt.figure(figsize=(6,6))
plt.plot([-1,1],[0,0],'-',c='k',linewidth=2)
plt.plot([0,0],[-1,1],'-',c='k',linewidth=2)
# idx = y_drug == i
x0, y0 = Rb_scores[:,-1], Rb_scores[:, -2]
j=3
idx_dose = halo_dose_label == ['Very High']
plt.plot((x0[idx_dose]), (y0[idx_dose]), 'o', alpha=0.8, color=marker_color_halo[j],
                 markersize=15,label='HALO - '+a)
x, y = np.median(Rb_scores[:,-1]), np.median(Rb_scores[:,-2])


x0, y0 = Ra_scores[:,-1], Ra_scores[:, -2]
j=3
idx_dose = risp_dose_label == ['Very High']
plt.plot((x0[idx_dose]), (y0[idx_dose]), 's', alpha=0.8, color=marker_color_risp[j],
             markersize=15,label='RISP - '+a)

plt.xlim((-0.5,0.5))
plt.ylim((-0.5,0.5))
plt.xticks([-0.3,0,0.3])
plt.yticks([-0.3,0,0.3])
plt.grid()

sns.despine()
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
plt.title('Haloperidol gcPCs')
plt.tight_layout()
plt.legend(title='Drug - Dose')

plt.savefig(os.path.join(save_fig_path,"risp_on_haloperidol_gcpcs.pdf"), format="pdf")

#%% PLOTTING HALO ON RISP GCPCS AND VICE VERSA
halo_dose_label = np.array(highlow_label)[halo_idx]
risp_dose_label = np.array(highlow_label)[risp_idx]
dose_order = ['Low','Medium','High','Very High']
# first plotting haloperidol

plt.figure(figsize=(6,6))
plt.plot([-1,1],[0,0],'-',c='k',linewidth=2)
plt.plot([0,0],[-1,1],'-',c='k',linewidth=2)
marker_color_risp = ['lightblue','dodgerblue','royalblue','darkblue']
# plotting risperidone now

x0, y0 = Rb_scores[:,-1], Rb_scores[:, -2]
for j,a in enumerate(dose_order):
    idx_dose = halo_dose_label == a
    plt.plot((x0[idx_dose]), (y0[idx_dose]), 'o', alpha=0.8, color=marker_color_halo[j],
                 markersize=15,label=a)

x0, y0 = Ra_scores[:,-1], Ra_scores[:, -2]
for j,a in enumerate(dose_order):
    idx_dose = risp_dose_label == a
    plt.plot((x0[idx_dose]), (y0[idx_dose]), 's', alpha=0.8, color=marker_color_risp[j],
                 markersize=15,label=a)

plt.xlim((-0.55,0.55))
plt.ylim((-0.55,0.55))
plt.xticks([-0.3,0,0.3])
plt.yticks([-0.3,0,0.3])
plt.grid()
sns.despine()
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
plt.title('Haloperidol gcPCs')
plt.tight_layout()
# plt.legend(title='RISP Dose')

plt.savefig(os.path.join(save_fig_path,"risp_on_halo_gcPCs.pdf"), format="pdf")

# plot of halo on risp gcPCs

halo_dose_label = np.array(highlow_label)[halo_idx]
risp_dose_label = np.array(highlow_label)[risp_idx]
dose_order = ['Low','Medium','High','Very High']
# first plotting haloperidol
# marker_size = [3,6,9,12]
marker_color_halo = ['palegreen','springgreen','mediumseagreen','darkgreen']
plt.figure(figsize=(6,6))
plt.plot([-1,1],[0,0],'-',c='k',linewidth=2)
plt.plot([0,0],[-1,1],'-',c='k',linewidth=2)
# idx = y_drug == i

x0, y0 = Ra_scores[:,0], Ra_scores[:, 1]
for j,a in enumerate(dose_order):
    idx_dose = risp_dose_label == a
    plt.plot((x0[idx_dose]), (y0[idx_dose]), 's', alpha=0.8, color=marker_color_risp[j],
                 markersize=15,label='Risp '+a)

x0, y0 = Rb_scores[:,0], Rb_scores[:, 1]
for j,a in enumerate(dose_order):
    idx_dose = halo_dose_label == a
    plt.plot((x0[idx_dose]), (y0[idx_dose]), 'o', alpha=0.8, color=marker_color_halo[j],
                 markersize=15,label='Halo '+a)


sns.despine()
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
plt.title('Risperidone gcPCs')
plt.tight_layout()
# plt.legend(title='Dose')
plt.xlim((-0.55,0.55))
plt.ylim((-0.55,0.55))
plt.xticks([-0.3,0,0.3])
plt.yticks([-0.3,0,0.3])
plt.grid()
plt.savefig(os.path.join(save_fig_path,"halo_on_risp_gcPCs.pdf"), format="pdf")
#%% calculating average euclidean distance to origin

distance_risp_risp_gcPCs = []
x0, y0 = Ra_scores[:,0], Ra_scores[:, 1]
for j,a in enumerate(dose_order):
    idx_dose = risp_dose_label == a
    # distance_risp_risp_gcPCs.append(np.linalg.norm([x0[idx_dose],y0[idx_dose]],axis=0).mean())
    distance_risp_risp_gcPCs.append(np.linalg.norm([x0[idx_dose].mean(),y0[idx_dose].mean()]))
    
distance_risp_halo_gcPCs = []
x0, y0 = Ra_scores[:,-1], Ra_scores[:, -2]
for j,a in enumerate(dose_order):
    idx_dose = risp_dose_label == a
    # distance_risp_halo_gcPCs.append(np.linalg.norm([x0[idx_dose],y0[idx_dose]],axis=0).mean())
    distance_risp_halo_gcPCs.append(np.linalg.norm([x0[idx_dose].mean(),y0[idx_dose].mean()]))
    
distance_halo_halo_gcPCs = []
x0, y0 = Rb_scores[:,-1], Rb_scores[:, -2]
for j,a in enumerate(dose_order):
    idx_dose = halo_dose_label == a
    # distance_halo_halo_gcPCs.append(np.linalg.norm([x0[idx_dose],y0[idx_dose]],axis=0).mean())
    distance_halo_halo_gcPCs.append(np.linalg.norm([x0[idx_dose].mean(),y0[idx_dose].mean()]))

distance_halo_risp_gcPCs = []
x0, y0 = Rb_scores[:,0], Rb_scores[:, 1]
for j,a in enumerate(dose_order):
    idx_dose = halo_dose_label == a
    # distance_halo_risp_gcPCs.append(np.linalg.norm([x0[idx_dose],y0[idx_dose]],axis=0).mean())
    distance_halo_risp_gcPCs.append(np.linalg.norm([x0[idx_dose].mean(),y0[idx_dose].mean()]))


#%% plots of distance of doses from origin

markersize = 15
plt.figure(figsize=(6,6))
plt.plot(distance_risp_risp_gcPCs,'-s',
         c='black',
         linewidth=3,
         markersize=markersize,
         label='RISP gcPCs')
plt.plot(distance_risp_halo_gcPCs,'--s',
         c='royalblue',
         linewidth=3,
         markersize=markersize,
         label='HALO gcPCs')
plt.xticks(ticks=[0,1,2,3],labels=dose_order)
plt.xlabel('Dose')
plt.ylabel('Distance to origin')
plt.title('Risperidone data')
plt.legend(title='gcPC space')
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(save_fig_path,'risp_distance_on_gcpcs.pdf'), format='pdf')


markersize = 15
plt.figure(figsize=(6,6))
plt.plot(distance_halo_halo_gcPCs,'-s',
         c='black',
         linewidth=3,
         markersize=markersize,
         label='HALO gcPCs')
plt.plot(distance_halo_risp_gcPCs,'--s',
         c='mediumseagreen',
         linewidth=3,
         markersize=markersize,
         label='RISP gcPCs')
plt.xticks(ticks=[0,1,2,3],labels=dose_order)
plt.xlabel('Dose')
plt.ylabel('Distance to origin')
plt.title('Haloperidol data')
plt.legend(title='gcPC space')
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(save_fig_path,"halo_distance_on_gcpcs.pdf"), format="pdf")

# make plot of data with sorted features from gcPCA
new_risp_data = risp_data[:,np.argsort(gcpca_mdl.loadings_[:,-1])]
new_halo_data = halo_data[:,np.argsort(gcpca_mdl.loadings_[:,-1])]
plt.figure(figsize=(6,6))
temp2plot = zscore(new_risp_data)
temp2plot[np.isnan(temp2plot)] = 0
plt.imshow(temp2plot.T,aspect='auto',cmap='viridis')
plt.title('RISP ordered halo gcPC1')
plt.xlabel('Animals')
plt.ylabel('Features')
plt.savefig(os.path.join(save_fig_path,"risp_ordered_halo_gcPC1.pdf"), format="pdf",transparent=True)
plt.show()

plt.figure(figsize=(6,6))
temp2plot = zscore(new_halo_data)
temp2plot[np.isnan(temp2plot)] = 0
plt.imshow(temp2plot.T,aspect='auto',cmap='viridis')
plt.title('Halo ordered halo gcPC1')
plt.xlabel('Animals')
plt.ylabel('Features')
plt.savefig(os.path.join(save_fig_path,"halo_ordered_halo_gcPC1.pdf"), format="pdf",transparent=True)
plt.show()