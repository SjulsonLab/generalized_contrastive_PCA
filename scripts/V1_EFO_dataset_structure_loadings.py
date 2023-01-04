#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 13:40:55 2022
script to get structure that PCA and ncPCA captures, also save the loadings for plotting and analysis
@author: eliezyer
"""

#%% importing essentials
import os
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import time
from scipy.stats import zscore
from scipy.io import loadmat
from sklearn import svm
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from matplotlib.pyplot import *
import seaborn as sns

from scipy.stats import zscore
import pickle
from numpy import linalg as LA

#%% parameters
min_acc = 0.3; #overall full rank accuracy has to be higher than this
min_fr = 0.01; #min firing rate of cells to be used in the analysis
bin_size = 0.1;
kcv = KFold() #cross-validation method
sessions = ('/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_F2763/210710', \
            '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_F2763/210711', \
            '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_F2763/210712', \
            '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_F2763/210713', \
            '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_F2763/210714', \
            '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_F2762/210723', \
            '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_F2762/210724', \
            '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_F2762/210725', \
            '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_F2762/210726', \
            '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_F2762/210727')
    
savepath = '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions';
#%% import custom modules
repo_dir = "/home/eliezyer/Documents/github/normalized_contrastive_PCA/" #repository dir
sys.path.append(repo_dir)
from ncPCA import ncPCA
from ncPCA import cPCA
#import ncPCA_project_utils as utils #this is going to be 

#%% load the overall accuracy file, pick only sessions where the results are higher than the min threshold
temp = loadmat(os.path.join(savepath,'natural_images_on_non_off.mat'))
aux_fr_acc = np.mean(temp['natural_images']['fullrank_acc'][0][0],axis=1)

ses2use = np.argwhere(aux_fr_acc>min_acc)

sessions2use = []
for i in ses2use:
    sessions2use.append(sessions[i[0]])

#%% main loop of analysis
all_basenames = [] #to save the basenames
structure_session = {} #saving final results
structure_session['ncPCA_variance'] = []
structure_session['PCsws_variance'] = []
structure_session['PCns_variance']  = []

loadings_session = {}
loadings_session['ncPCA_loadings'] = []
loadings_session['PCsws_loadings'] = []
loadings_session['PCns_loadings']  = []

struct_session = {}
struct_session['variance'] = []
struct_session['PCnum']    = []
struct_session['method']   = []
struct_session['folds']    = [] 
struct_session['session']  = []

for name_ses in sessions2use:
    
    _,basename=os.path.split(name_ses)
    all_basenames.append(basename)
    
    #load spikes
    temp = loadmat(os.path.join(name_ses,basename+'.spikes.cellinfo.mat'))
    spikes_cellinfo = temp['spikes']
    
    #load mua
    temp = loadmat(os.path.join(name_ses,basename+'.mua.cellinfo.mat'))
    mua_cellinfo = temp['mua']
    
    #load sleep states
    temp = loadmat(os.path.join(name_ses,basename+'.SleepState.states.mat'))
    SleepState_info = temp['SleepState']
    
    #load visual stimulation information
    temp = loadmat(os.path.join(name_ses,basename+'.visual_stimulation.mat'))
    vis_stim_info = temp['visual_stimulation']
    
    #load time frames (this help us know when the animal was head fixed)
    temp = loadmat(os.path.join(name_ses,basename+'.t_frames.mat'))
    t_frames = temp['t_frames']
    
    #%% binning spikes and excluding neurons with low firing rate
    
    temp_spk_ts = {}
    nunits = spikes_cellinfo['times'][0][0][0].shape[0];
    for aa in np.arange(nunits):
        temp_spk_ts[aa] = spikes_cellinfo['times'][0][0][0][aa]
    nunits = mua_cellinfo['times'][0][0][0].shape[0];
    for a in np.arange(nunits):
        temp_spk_ts[aa+a+1] = mua_cellinfo['times'][0][0][0][a]
    
    # passing to pynapple ts group
    spikes_nap = nap.TsGroup(temp_spk_ts)
    
    spikes_binned = spikes_nap.count(10*60) #10 min bins
    
    #picking cells that fire during the whole recordings, doing this by testing against 5 blocks
    #cells2keep=np.logical_not(np.sum(spikes_binned.values==0,axis=0)>=(spikes_binned.values.shape[1]/5))
    
    """I'll have to deal with this later, do this using pynapple"""
    
    #%% binning the spikes to the images images
    
    # passing natural scenes to intervalsets
    ns_intervals = nap.IntervalSet(start=vis_stim_info['stimOnset'][0][0],end=vis_stim_info['stimOffset'][0][0])
    label_id = vis_stim_info['ID'][0][0][0]

    # binning through using a loop in the cells until I can find a better way
    #start timer
    spikes_binned_ns = np.empty((len(ns_intervals.values),len(spikes_nap.data)))
    bins_ns = ns_intervals.values.flatten()
    for aa in np.arange(len(spikes_nap.data)):
        tmp = np.array(np.histogram(spikes_nap.data[aa].index.values,bins_ns))
        spikes_binned_ns[:,aa] = tmp[0][np.arange(0,tmp[0].shape[0],2)]
    
    #%% getting activity during SWS
    
    sws_intervals = nap.IntervalSet(start=SleepState_info[0][0][0]['NREMstate'][0][0][:,0], \
                                    end=SleepState_info[0][0][0]['NREMstate'][0][0][:,1])
    tempsws = spikes_nap.restrict(sws_intervals).count(bin_size)
    spikes_binned_sws = tempsws.values
    
    #%% splitting data into train and test set
    swscv = kcv.split(spikes_binned_sws)
    ncPCA_structure = []
    PCAsws_structure= []
    PCAns_structure = []
    ncPCA_loadings  = []
    PCAsws_loadings = []
    PCAns_loadings  = []
    folds_track  = []
    PC_number    = []
    ft = 0 #counter for fold track
    for train_ind_ns,test_ind_ns in kcv.split(spikes_binned_ns):
        train_ind_ns = train_ind_ns.reshape(train_ind_ns.shape[0],1)
        test_ind_ns  = test_ind_ns.reshape(test_ind_ns.shape[0],1)
        
        #getting spike and labels of natural scenes train/test
        train_ns,test_ns = zscore(spikes_binned_ns[train_ind_ns,:]),zscore(spikes_binned_ns[test_ind_ns,:])
        train_ns = np.squeeze(train_ns)
        test_ns  = np.squeeze(test_ns)
        labels_ns_train,labels_ns_test = label_id[train_ind_ns],label_id[test_ind_ns]
        
        #getting spike sws train/test
        train_ind_sws,test_ind_sws = next(swscv)
        train_sws,test_sws = zscore(spikes_binned_sws[train_ind_sws,:]),zscore(spikes_binned_sws[test_ind_sws,:])
        
        #zeroing any cell that was nan (i.e. no activity)
        train_ns[np.isnan(train_ns)] = 0
        test_ns[np.isnan(test_ns)]   = 0
        train_sws[np.isnan(train_sws)]  = 0
        test_sws[np.isnan(test_sws)]   = 0
        
        
        # perform PCA in train sleep and trian natural scenes
        _,_,Vsws = LA.svd(train_sws,full_matrices=False)
        _,_,Vns = LA.svd(train_ns,full_matrices=False)
        
        # perform ncPCA
        ncPCA_mdl = ncPCA(basis_type='intersect',Nshuffle=0)
        ncPCA_mdl.fit(train_sws,train_ns)
        
        X = ncPCA_mdl.loadings_
        
        #getting the structure and saving
        ncPCA_structure.append(np.var(np.dot(test_ns,X),axis=0))
        PCAsws_structure.append(np.var(np.dot(test_ns,Vsws[:X.shape[1],:].T),axis=0))
        PCAns_structure.append(np.var(np.dot(test_ns,Vns[:X.shape[1],:].T),axis=0))
        
        #saving the loadings of each for later analysis
        ncPCA_loadings.append(X)
        PCAsws_loadings.append(Vsws.T)
        PCAns_loadings.append(Vns.T)
        
        PCnum = np.linspace(1,X.shape[1],X.shape[1]) #tracking number of PCs
        PC_number.append(PCnum)
        PC_number2     = np.hstack(PC_number)
        
        #tracking the folds for plotting
        folds_track.append(np.repeat(ft,X.shape[1]))
        ft+=1
        
        
    ncPCA_structure2   = np.hstack(ncPCA_structure)
    PCAsws_structure2  = np.hstack(PCAsws_structure)
    PCAns_structure2   = np.hstack(PCAns_structure)    
    
    loadings_session['ncPCA_loadings'].append(ncPCA_loadings)
    loadings_session['PCsws_loadings'].append(PCAsws_loadings)
    loadings_session['PCns_loadings'].append(PCAns_loadings)
    
    temp_var = np.hstack([ncPCA_structure2,PCAsws_structure2,PCAns_structure2])
    struct_session['variance'].append(temp_var)
    struct_session['PCnum'].append(np.tile(PC_number2,3))
    temp_method = np.concatenate((np.tile('ncPCA',len(PC_number2)),np.tile('PCsws',len(PC_number2)),np.tile('PCns',len(PC_number2))));
    struct_session['method'].append(temp_method)
    struct_session['folds'].append(np.tile(np.hstack(folds_track),3))
    struct_session['session'].append(np.tile(basename,len(temp_var)))

#%% save the results in a pickle file

with open(savepath+'/ncPCA_structure_loadings.pickle','wb') as handle:
    pickle.dump(structure_session,handle,protocol=pickle.HIGHEST_PROTOCOL)
    
#%% plotting the results
#passing dictionary to a dataframe 
"there's definitely a better way to do this"

df = pd.DataFrame([
                  np.hstack(struct_session['variance']),\
                  np.hstack(struct_session['PCnum']),\
                  np.hstack(struct_session['method']),\
                  np.hstack(struct_session['folds']),\
                  np.hstack(struct_session['session'])],\
                  index=list(struct_session))
df = df.T 
#making the plot
sns.set_style("ticks")
sns.set_context("talk")
style.use('dark_background')

"""First restrict the df for only a session"""
#df2 = df[df.T.query('session == `210711`')]

aux = sns.relplot(data=df,x="PCnum",y="variance",hue="method",col="session",kind="line",col_wrap=3)
aux.set(xscale='log',yscale='log')

#%% plotting example of loadings

figure()
fig, axes = subplots(1, 2, figsize=(10, 5), sharey=True)
subplot(1,2,1)
stem(loadings_session['ncPCA_loadings'][3][0][:,1])
xlabel('neurons')
ylabel('loading magnitude')
title('ncPC2')
subplot(1,2,2)
stem(loadings_session['PCns_loadings'][3][0][:,1])
title('PCsws2')
xlabel('neurons')

#%% trying to make summary of sparsity

def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)
    import numpy as np
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g

ncPCA_g = []
PCA_g   = []
for ses in np.arange(6):
    for cv in np.arange(5):
        ncPCS = loadings_session['ncPCA_loadings'][ses][cv]
        PCS = loadings_session['PCns_loadings'][ses][cv]
        for pc in np.arange(10):
            ncPCA_g.append(gini(np.abs(ncPCS[:,pc])))
            PCA_g.append(gini(np.abs(PCS[:,pc])))

gini = np.hstack((ncPCA_g,PCA_g));
method = np.hstack((np.tile('ncPCA',int(gini.shape[0]/2)),np.tile('PCA',int(gini.shape[0]/2))))

df = pd.DataFrame(data=[gini,method],index=('gini coefficient','method'))
df = df.T

sns.boxplot(data=df,x="method",y="gini coefficient")