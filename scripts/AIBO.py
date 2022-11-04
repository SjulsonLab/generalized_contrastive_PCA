#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 16:54:10 2021
Script to analyze data from allen institute part of the functional connectivity
dataset. We want to see if lower rank PCs are able to predict drifting gratings 
and natural images. The analysis steps will be:
    
    1) Get all the sessions from functional_connectivity dataset with VIsp
    recordings
    2) Perform PCA during the spontaneous activity recording
    3) Predict the activity of drifting gratings using windowed PCs and random
    projection

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

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from sklearn import svm
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from matplotlib.pyplot import *
from scipy.stats import zscore
from contrastive import CPCA
import pickle
from numpy import linalg as LA

#%% parameters
min_n_cell = 50 #min number of cells in the brain area to be used
kcv = KFold() #cross-validation method

#%% import custom modules
repo_dir = "/home/eliezyer/Documents/github/normalized_contrastive_PCA/" #repository dir
sys.path.append(repo_dir)
from ncPCA import ncPCA
from ncPCA import cPCA
import ncPCA_project_utils as utils #this is going to be 
#%% preparing data to be loaded

data_directory = '/mnt/probox/allen_institute_data/ecephys/' # must be a valid directory in your filesystem
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# getting brain observatory dataset
sessions = cache.get_session_table()
selected_sessions = sessions[(sessions.session_type == 'brain_observatory_1.1')]
    

#%% get units size
# getting neurons for each session
array_of_ba = ['LGd','VISp','VISrl','VISpm']
dict_units = {}

for i in range(len(selected_sessions.index.tolist())):
    tot_session = selected_sessions.index.tolist()[i]
    loaded_session = cache.get_session_data(tot_session)
    x = loaded_session.units
    x = x[x["ecephys_structure_acronym"].str.contains('V|LG')]

    for j in range(len(array_of_ba)):
        units_ba_wise = x["ecephys_structure_acronym"]==array_of_ba [j]
        key = str(tot_session)+"_"+ str(array_of_ba[j])
        dict_units[key] = units_ba_wise

# saving the array_unit file
units_file = str("array_units")
with open(r'/mnt/probox/allen_institute_data/pkl_sessions/array_units', 'wb') as handle:
     pickle.dump(dict_units, handle, protocol=pickle.HIGHEST_PROTOCOL)



#%% performing analysis, session loop starts here
for session_id in selected_sessions.index.values:

    loaded_session = cache.get_session_data(session_id)

    # getting natural scenes and static gratings info
    stimuli_info = loaded_session.get_stimulus_table(["natural_scenes","static_gratings"])

    # getting spikes times and information
    temp_spk_ts = loaded_session.spike_times
    temp_spk_info  = loaded_session.units

    # here I'm picking all the brain areas that are related to visual system, i.e.,
    # starting with V or LG
    spikes_info = temp_spk_info[temp_spk_info["ecephys_structure_acronym"].str.contains("V|LG")]
    units2use = spikes_info.index.values

    temp_spk_ts_2 = {}
    for aa in np.arange(0,len(units2use)):
        temp_spk_ts_2[aa] = temp_spk_ts[units2use[aa]]

    # passing to pynapple ts group
    spikes_times = nap.TsGroup(temp_spk_ts_2)

    # adding structure info into the spike times TSgroup
    structs = pd.DataFrame(index=np.arange(len(units2use)),data = spikes_info.ecephys_structure_acronym.values,columns=['struct'])
    spikes_times.set_info(structs)

    # passing natural scenes to intervalsets
    df_stim_ns = stimuli_info.query("stimulus_name=='natural_scenes'")
    ns_intervals = nap.IntervalSet(start=df_stim_ns.start_time.values,end=df_stim_ns.stop_time.values)

    df_stim_sg = stimuli_info.query("stimulus_name=='static_gratings'")
    sg_intervals = nap.IntervalSet(start=df_stim_sg.start_time.values,end=df_stim_sg.stop_time.values)

    # binning through using a loop in the cells until I can find a better way
    #start timer
    t_start = time.time()
    spikes_binned_ns = np.empty((len(ns_intervals.values),len(spikes_times.data)))
    bins_ns = ns_intervals.values.flatten()
    for aa in np.arange(len(spikes_times.data)):
        tmp = np.array(np.histogram(spikes_times.data[aa].index.values,bins_ns))
        spikes_binned_ns[:,aa] = tmp[0][np.arange(0,tmp[0].shape[0],2)]
    t_stop = time.time()
    
    # same method of binning through a loop in the cells, but for static gratings
    spikes_binned_sg = np.empty((len(sg_intervals.values),len(spikes_times.data)))
    bins_sg = sg_intervals.values.flatten()
    for aa in np.arange(len(spikes_times.data)):
        tmp = np.array(np.histogram(spikes_times.data[aa].index.values,bins_sg))
        spikes_binned_sg[:,aa] = tmp[0][np.arange(0,tmp[0].shape[0],2)]

    # getting labels
    sg_labels = df_stim_sg.stimulus_condition_id.values
    ns_labels = df_stim_ns.stimulus_condition_id.values

    #%% performing prediction
    array_of_ba = spikes_info["ecephys_structure_acronym"].unique();
    scores_ns = np.empty((5,len(array_of_ba)))
    scores_sg = np.empty((5,len(array_of_ba)))

    spikes_zsc_ns = zscore(spikes_binned_ns) #remove this later and update the variables name accordingly
    spikes_zsc_sg = zscore(spikes_binned_sg)

    #%% getting ncPCA and cPCA loadings

    # fw ns and bw ns cPCA loadings, ns PCA loadings and fw ns cPCA loadings
    brain_area_dict = {};
    #array_of_ba = ['LGd','VISp','VISrl','VISpm'] #we are supposed to be using the array_of_ba identified on the cell before

    for ba_name in array_of_ba:
        units_idx = spikes_info["ecephys_structure_acronym"]==ba_name

        if sum(units_idx.values) >= min_n_cell:
            
            #declaring variables before
            scores_PCA_ns = []
            scores_PCA_sg = []
            scores_ncPCA_fw_ns = []
            scores_ncPCA_bw_ns = []
            scores_ncPCA_fw_sg = []
            scores_ncPCA_bw_sg = []
            scores_cPCA_fw_ns = []
            scores_cPCA_bw_ns = []
            scores_cPCA_fw_sg = []
            scores_cPCA_bw_sg = []
            alpha2save = []
            
            ### This is where the cross validation starts
            #first we set up the train and test set for both datasets
            for train_ind,test_ind in kcv.split(spikes_zsc_ns):
                
                train_ind = train_ind.reshape(train_ind.shape[0],1)
                test_ind  = test_ind.reshape(test_ind.shape[0],1)
                train_ns,test_ns = zscore(spikes_zsc_ns[train_ind,units_idx]),zscore(spikes_zsc_ns[test_ind,units_idx])
                train_sg,test_sg = zscore(spikes_zsc_sg[train_ind,units_idx]),zscore(spikes_zsc_sg[test_ind,units_idx])
                labels_ns_train,labels_ns_test = ns_labels[train_ind],ns_labels[test_ind]
                labels_sg_train,labels_sg_test = sg_labels[train_ind],sg_labels[test_ind]
                
                #zeroing any cell that was nan (i.e. no activity)
                train_ns[np.isnan(train_ns)] = 0
                test_ns[np.isnan(test_ns)]   = 0
                train_sg[np.isnan(train_sg)]  = 0
                test_sg[np.isnan(test_sg)]   = 0
            
                """ This block will do ncPCA """
                # fitting ncPCA to train set
                ncPCA_mdl = ncPCA(basis_type='intersect',Nshuffle=10000)
                ncPCA_mdl.fit(train_sg,train_ns)
                
                X_fw_ns = ncPCA_mdl.loadings_
                
                X_bw_ns = np.flip(X_fw_ns, axis=1)
                X_ = {'X_fw_ns':X_fw_ns, 'X_bw_ns':X_bw_ns}
                
                for aaa in np.arange(X_fw_ns.shape[1]):
                    for Xstr in X_:
                        if str(Xstr) == 'X_fw_ns':
                            X = X_[Xstr]
                            #projecting sets into the proper dimensions
                            ncPCA_fw_ns_train = np.dot(train_ns,X[:,:aaa+1])
                            ncPCA_fw_sg_train = np.dot(train_sg,X[:,:aaa+1])
                            ncPCA_fw_ns_test  = np.dot(test_ns,X[:,:aaa+1])
                            ncPCA_fw_sg_test  = np.dot(test_sg,X[:,:aaa+1])
                            
                            #fitting models in training set - both NS and SG
                            clf_ns = svm.SVC().fit(ncPCA_fw_ns_train,labels_ns_train)
                            clf_sg = svm.SVC().fit(ncPCA_fw_sg_train,labels_sg_train)
                            
                            #saving cross validated scores
                            scores_ncPCA_fw_ns.append(clf_ns.score(ncPCA_fw_ns_test,labels_ns_test))
                            scores_ncPCA_fw_sg.append(clf_sg.score(ncPCA_fw_sg_test,labels_sg_test))
                            
                        elif str(Xstr) == 'X_bw_ns':
                            X = X_[Xstr]
                            #projecting sets into the proper dimensions
                            ncPCA_bw_ns_train = np.dot(train_ns,X[:,-1*(aaa+1):])
                            ncPCA_bw_sg_train = np.dot(train_sg,X[:,-1*(aaa+1):])
                            ncPCA_bw_ns_test  = np.dot(test_ns,X[:,-1*(aaa+1):])
                            ncPCA_bw_sg_test  = np.dot(test_sg,X[:,-1*(aaa+1):])
                            
                            #fitting models in training set - both NS and SG
                            clf_ns = svm.SVC().fit(ncPCA_bw_ns_train,labels_ns_train)
                            clf_sg = svm.SVC().fit(ncPCA_bw_sg_train,labels_sg_train)
                            
                            #saving cross validated scores
                            scores_ncPCA_bw_ns.append(clf_ns.score(ncPCA_bw_ns_test,labels_ns_test))
                            scores_ncPCA_bw_sg.append(clf_sg.score(ncPCA_bw_sg_test,labels_sg_test))
                
                
                """ This block will calculate for cPCA """
                
                #first getting alphas optimized (?) for the dataset <- this needs to be checked if it's true
                n_components = X_fw_ns.shape[1]
                mdl = CPCA(n_components)
                _, alpha_values = mdl.fit_transform(train_ns,train_sg, return_alphas= True)
                alpha2save.append(alpha_values)
    
    
                for alpha in alpha_values:
                    
                    cPCs = cPCA(train_sg,train_ns,alpha=alpha)[:,:n_components]
                    
                    for aaa in np.arange(cPCs.shape[1]):
                        #projection data sets into the proper cPCs
                        cPCA_fw_ns_train = np.dot(train_ns,cPCs[:,:aaa+1])
                        cPCA_fw_sg_train = np.dot(train_sg,cPCs[:,:aaa+1])
                        cPCA_fw_ns_test  = np.dot(test_ns,cPCs[:,:aaa+1])
                        cPCA_fw_sg_test  = np.dot(test_sg,cPCs[:,:aaa+1])
                        
                        #fitting models in training set - both NS and SG
                        clf_ns = svm.SVC().fit(cPCA_fw_ns_train,labels_ns_train)
                        clf_sg = svm.SVC().fit(cPCA_fw_sg_train,labels_sg_train)
                        
                        #saving cross validated scores
                        scores_cPCA_fw_ns.append(clf_ns.score(cPCA_fw_ns_test,labels_ns_test))
                        scores_cPCA_fw_sg.append(clf_sg.score(cPCA_fw_sg_test,labels_sg_test))
                        
                        #BACKWARDS CPCA dimensions
                        cPCA_bw_ns_train = np.dot(train_ns,cPCs[:,-1*(aaa+1):])
                        cPCA_bw_sg_train = np.dot(train_sg,cPCs[:,-1*(aaa+1):])
                        cPCA_bw_ns_test  = np.dot(test_ns,cPCs[:,-1*(aaa+1):])
                        cPCA_bw_sg_test  = np.dot(test_sg,cPCs[:,-1*(aaa+1):])
                        
                        #fitting models in training set
                        clf_ns = svm.SVC().fit(cPCA_bw_ns_train,labels_ns_train)
                        clf_sg = svm.SVC().fit(cPCA_bw_sg_train,labels_sg_train)
                        
                        #saving cross_validated scores
                        scores_cPCA_bw_ns.append(clf_ns.score(cPCA_bw_ns_test,labels_ns_test))
                        scores_cPCA_bw_sg.append(clf_sg.score(cPCA_bw_sg_test,labels_sg_test))
    

                
                """ This block will calculate the scores for regular PCA"""
                #the PCs for prediction also need to be cross validated
                _,_,Vns = np.linalg.svd(train_ns,full_matrices=False)
                _,_,Vsg = np.linalg.svd(train_sg,full_matrices=False)
                
                for aaa in np.arange(Vns.shape[1]):
                    
                    #projecting data into PCs
                    PCA_ns_train = np.dot(train_ns,Vns[:,:aaa+1])
                    PCA_ns_test = np.dot(test_ns,Vns[:,:aaa+1])
                    PCA_sg_train = np.dot(train_sg,Vsg[:,:aaa+1])
                    PCA_sg_test = np.dot(test_sg,Vsg[:,:aaa+1])
                    
                    #fitting models on training set
                    clf_ns = svm.SVC().fit(PCA_ns_train,labels_ns_train)
                    clf_sg = svm.SVC().fit(PCA_sg_train,labels_sg_train)
                    
                    #saving cross_validated scores
                    scores_PCA_ns.append(clf_ns.score(PCA_ns_test,labels_ns_test))
                    scores_PCA_sg.append(clf_sg.score(PCA_sg_test,labels_sg_test))
    
            #saving ncPCA        
            brain_area_dict['scores_ncPCA_fw_ns_'+ba_name] = scores_ncPCA_fw_ns
            brain_area_dict['scores_ncPCA_fw_sg_'+ba_name] = scores_ncPCA_fw_sg
            brain_area_dict['scores_ncPCA_bw_ns_'+ba_name] = scores_ncPCA_bw_ns
            brain_area_dict['scores_ncPCA_bw_sg_'+ba_name] = scores_ncPCA_bw_sg
            
            #saving cPCA
            brain_area_dict['scores_cPCA_fw_ns_'+ba_name] = scores_cPCA_fw_ns
            brain_area_dict['scores_cPCA_fw_sg_'+ba_name] = scores_cPCA_fw_sg
            brain_area_dict['scores_cPCA_bw_ns_'+ba_name] = scores_cPCA_bw_ns
            brain_area_dict['scores_cPCA_bw_sg_'+ba_name] = scores_cPCA_bw_sg
            #saving alpha
            brain_area_dict['alpha_value_of_cPCA'+ba_name] = alpha2save
            
            #saving PCA
            brain_area_dict['scores_PCA_ns_'+ba_name] = scores_PCA_ns
            brain_area_dict['scores_PCA_sg_'+ba_name] = scores_PCA_sg
            
# TODO: add lines of code to save the brain_area_dict for each session, or should we save each session as a pickle??

    #%% add dictionary to pickle file
    #file_path = r"\home\pranjal\Documents\pkl_sessions" + "\\" + str(session_id)

    # saving the brain_area_dict file
    #with open(file_path, 'wb') as handle:
    #     pickle.dump(brain_area_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# TODO: add plotting code
#%% plotting
#dir_list = os.listdir(r"/home/pranjal/Documents/pkl_sessions")
#session_id = []
# for i in dir_list:
#     try:
#         x = list(i)[0:9]
#         c = ''.join(map(str, x))
#         session_id.append((int(c)))
#
#     except:
#         continue
#
# session_id = session_id
#
# units_file = r"/home/pranjal/Documents/pkl_sessions/array_units"
# with open(file, 'rb') as array_units:
#     array_units = pickle.load(array_units)
