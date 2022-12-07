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
import seaborn as sns

from scipy.stats import zscore
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
    
    #TODO: this will have to leave later
    """THIS WILL NEED TO BE DELETED LATER, BECAUSE OTHERWISE FOR EVERY SESSION 
    WE WILL CLEAN THE THING, UNLESS WE WANT TO KEEP IT LIKE THAT AND SAVE A
    A FILE FOR EVERY SESSION
    """
    
    for ba_name in array_of_ba:
        brain_area_dict['scores_ncPCA_fw_ns_'+ba_name] = []
        brain_area_dict['scores_ncPCA_fw_sg_'+ba_name] = []
        brain_area_dict['scores_ncPCA_bw_ns_'+ba_name] = []
        brain_area_dict['scores_ncPCA_bw_sg_'+ba_name] = []
        
        #for saving cPCA
        brain_area_dict['scores_cPCA_fw_ns_'+ba_name] = []
        brain_area_dict['scores_cPCA_fw_sg_'+ba_name] = []
        brain_area_dict['scores_cPCA_bw_ns_'+ba_name] = []
        brain_area_dict['scores_cPCA_bw_sg_'+ba_name] = []
        #for saving alpha
        brain_area_dict['alpha_value_of_cPCA'+ba_name] = []
        
        #for saving PCA
        brain_area_dict['scores_PCA_ns_'+ba_name] = []
        brain_area_dict['scores_PCA_sg_'+ba_name] = []
        
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
            #
            
            #getting CV indices for SG separately because it has different size than NS
            sgcv = kcv.split(spikes_zsc_sg)
            for train_ind_ns,test_ind_ns in kcv.split(spikes_zsc_ns):
                
                train_ind_ns = train_ind_ns.reshape(train_ind_ns.shape[0],1)
                test_ind_ns  = test_ind_ns.reshape(test_ind_ns.shape[0],1)
                
                train_ns,test_ns = zscore(spikes_zsc_ns[train_ind_ns,units_idx]),zscore(spikes_zsc_ns[test_ind_ns,units_idx])
                labels_ns_train,labels_ns_test = ns_labels[train_ind_ns],ns_labels[test_ind_ns]
                
                train_ind_sg,test_ind_sg = next(sgcv)
                train_ind_sg = train_ind_sg.reshape(train_ind_sg.shape[0],1)
                test_ind_sg  = test_ind_sg.reshape(test_ind_sg.shape[0],1)
                
                train_sg,test_sg = zscore(spikes_zsc_sg[train_ind_sg,units_idx]),zscore(spikes_zsc_sg[test_ind_sg,units_idx])
                
                labels_sg_train,labels_sg_test = sg_labels[train_ind_sg],sg_labels[test_ind_sg]
                
                #zeroing any cell that was nan (i.e. no activity)
                train_ns[np.isnan(train_ns)] = 0
                test_ns[np.isnan(test_ns)]   = 0
                train_sg[np.isnan(train_sg)]  = 0
                test_sg[np.isnan(test_sg)]   = 0
            
                """ This block will do ncPCA """
                # fitting ncPCA to train set
                ncPCA_mdl = ncPCA(basis_type='intersect',Nshuffle=0)
                ncPCA_mdl.fit(train_sg,train_ns)
                loadings_ncpca = ncPCA_mdl.loadings_
                temp_fw_ns, temp_bw_ns =  utils.cumul_accuracy_projected(train_ns, labels_ns_train, test_ns, labels_ns_test,
                                             loadings_ncpca, analysis='both')

                temp_fw_sg, temp_bw_sg =  utils.cumul_accuracy_projected(train_sg, labels_sg_train, test_sg, labels_sg_test,
                                                 loadings_ncpca, analysis='both')
                
                scores_ncPCA_fw_ns.append(temp_fw_ns)
                scores_ncPCA_bw_ns.append(temp_bw_ns)
                scores_ncPCA_fw_sg.append(temp_fw_sg)
                scores_ncPCA_bw_sg.append(temp_bw_sg)
                """ This block will calculate for cPCA """
                
                #first getting alphas optimized (?) for the dataset <- this needs to be checked if it's true
                n_components = loadings_ncpca.shape[1]
                #mdl = CPCA(n_components)
                #_, alpha_values = mdl.fit_transform(train_ns,train_sg, return_alphas= True)
                #alpha2save.append(alpha_values)
    
                #alpha_values = 1
                #for alpha in alpha_values:
                alpha = 1   
                cPCs = cPCA(train_sg,train_ns,alpha=alpha)[:,:n_components]

                temp_fw_ns, temp_bw_ns = utils.cumul_accuracy_projected(train_ns, labels_ns_train, test_ns,
                                                                        labels_ns_test,
                                                                        cPCs, analysis='both')

                temp_fw_sg, temp_bw_sg = utils.cumul_accuracy_projected(train_sg, labels_sg_train, test_sg,
                                                                        labels_sg_test,
                                                                        cPCs, analysis='both')
                
                scores_cPCA_fw_ns.append(temp_fw_ns)
                scores_cPCA_bw_ns.append(temp_bw_ns)
                scores_cPCA_fw_sg.append(temp_fw_sg)
                scores_cPCA_bw_sg.append(temp_bw_sg)

                
                """ This block will calculate the scores for regular PCA"""
                #the PCs for prediction also need to be cross validated
                _,_,Vns = np.linalg.svd(train_ns,full_matrices=False)
                _,_,Vsg = np.linalg.svd(train_sg,full_matrices=False)


                temp_ns = utils.cumul_accuracy_projected(train_ns, labels_ns_train, test_ns,
                                                                        labels_ns_test, Vns.T)
                temp_sg = utils.cumul_accuracy_projected(train_sg, labels_sg_train, test_sg,
                                                            labels_sg_test, Vsg.T)

                scores_PCA_ns.append(temp_ns)
                scores_PCA_sg.append(temp_sg)

            #saving ncPCA
            
            brain_area_dict['scores_ncPCA_fw_ns_'+ba_name].append(scores_ncPCA_fw_ns)
            brain_area_dict['scores_ncPCA_fw_sg_'+ba_name].append(scores_ncPCA_fw_sg)
            brain_area_dict['scores_ncPCA_bw_ns_'+ba_name].append(scores_ncPCA_bw_ns)
            brain_area_dict['scores_ncPCA_bw_sg_'+ba_name].append(scores_ncPCA_bw_sg)
            
            #saving cPCA
            brain_area_dict['scores_cPCA_fw_ns_'+ba_name].append(scores_cPCA_fw_ns)
            brain_area_dict['scores_cPCA_fw_sg_'+ba_name].append(scores_cPCA_fw_sg)
            brain_area_dict['scores_cPCA_bw_ns_'+ba_name].append(scores_cPCA_bw_ns)
            brain_area_dict['scores_cPCA_bw_sg_'+ba_name].append(scores_cPCA_bw_sg)
            #saving alpha
            brain_area_dict['alpha_value_of_cPCA'+ba_name].append(alpha2save)
            
            #saving PCA
            brain_area_dict['scores_PCA_ns_'+ba_name].append(scores_PCA_ns)
            brain_area_dict['scores_PCA_sg_'+ba_name].append(scores_PCA_sg)
            
# TODO: add lines of code to save the brain_area_dict for each session, or should we save each session as a pickle??

    #%% add dictionary to pickle file
    #file_path = r"\home\pranjal\Documents\pkl_sessions" + "\\" + str(session_id)

    # saving the brain_area_dict file
    with open('/mnt/SSD4TB/ncPCA_files/test_AIBO.pickle', 'wb') as handle:
         pickle.dump(brain_area_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# TODO: add plotting code

#%% reading file
with open('/mnt/SSD4TB/ncPCA_files/test_AIBO.pickle', 'rb') as handle:
     x = pickle.load(handle)


#%% parameters for plotting
rcParams['figure.dpi'] = 500
rcParams['lines.linewidth'] = 2.5
rcParams['axes.linewidth']  = 1.5
rcParams['font.size'] = 12

#%% plotting performance curves

sns.set_style("ticks")
sns.set_context("talk")
#this is temporary
array_of_ba2 = array_of_ba[[1,2,3,5]]
n = len(array_of_ba2)
c = 0
figure(figsize=(25,5))
for ba_name in array_of_ba2:
    c +=1
    subplot(1,n,c)
            
    n1 = np.shape(brain_area_dict['scores_cPCA_fw_ns_' + ba_name][0])[0]
    temp = np.hstack(brain_area_dict['scores_cPCA_fw_ns_' + ba_name][0])
    pc = np.tile(np.arange(temp.size/n1),n1);
    method = np.tile('cPCA',pc.size)
    
    df_cPCA = pd.DataFrame({'pc':pc,'R^2':temp,'Method':method})
    
    n1 = np.shape(brain_area_dict['scores_ncPCA_fw_ns_' + ba_name][0])[0]
    temp = np.hstack(brain_area_dict['scores_ncPCA_fw_ns_' + ba_name][0])
    pc = np.tile(np.arange(temp.size/n1),n1);
    method = np.tile('ncPCA',pc.size)
    
    df_ncPCA = pd.DataFrame({'pc':pc,'R^2':temp,'Method':method})
    
    n1 = np.shape(brain_area_dict['scores_PCA_ns_' + ba_name][0])[0]
    temp = np.hstack(brain_area_dict['scores_PCA_ns_' + ba_name][0])
    pc = np.tile(np.arange(temp.size/n1),n1);
    method = np.tile('PCA',pc.size)
    
    df_PCA = pd.DataFrame({'pc':pc,'R^2':temp,'Method':method})
    
    df = pd.concat([df_PCA,df_ncPCA,df_cPCA],ignore_index=True)
    if c == 1:
        sns.lineplot(data=df,x='pc',y='R^2',hue='Method',legend=True)
    else:
        sns.lineplot(data=df,x='pc',y='R^2',hue='Method',legend = False)
    title(ba_name)
    #plot(np.mean(brain_area_dict['scores_ncPCA_fw_ns' + ba_name][0],axis=0))
    #plot(np.mean(brain_area_dict['scores_cPCA_fw_ns_VISp'+ ba_name][0],axis=0))

suptitle('Cumulative components prediction')
#%% performance curve PCA vs bw cPCA and ncPCA ns
#this is temporary
array_of_ba2 = array_of_ba[[1,2,3,5]]
n = len(array_of_ba2)
c = 0
figure(figsize=(25,5))
for ba_name in array_of_ba2:
    c +=1
    subplot(1,n,c)
            
    n1 = np.shape(brain_area_dict['scores_cPCA_bw_ns_' + ba_name][0])[0]
    temp = np.hstack(brain_area_dict['scores_cPCA_bw_ns_' + ba_name][0])
    pc = np.tile(np.arange(temp.size/n1),n1);
    method = np.tile('cPCA',pc.size)
    
    df_cPCA = pd.DataFrame({'pc':pc,'R^2':temp,'Method':method})
    
    n1 = np.shape(brain_area_dict['scores_ncPCA_bw_ns_' + ba_name][0])[0]
    temp = np.hstack(brain_area_dict['scores_ncPCA_bw_ns_' + ba_name][0])
    pc = np.tile(np.arange(temp.size/n1),n1);
    method = np.tile('ncPCA',pc.size)
    
    df_ncPCA = pd.DataFrame({'pc':pc,'R^2':temp,'Method':method})
    
    n1 = np.shape(brain_area_dict['scores_PCA_ns_' + ba_name][0])[0]
    temp = np.hstack(brain_area_dict['scores_PCA_ns_' + ba_name][0])
    pc = np.tile(np.arange(temp.size/n1),n1);
    method = np.tile('PCA',pc.size)
    
    df_PCA = pd.DataFrame({'pc':pc,'R^2':temp,'Method':method})
    
    df = pd.concat([df_PCA,df_ncPCA,df_cPCA],ignore_index=True)
    if c == 1:
        sns.lineplot(data=df,x='pc',y='R^2',hue='Method',legend=True)
    else:
        sns.lineplot(data=df,x='pc',y='R^2',hue='Method',legend = False)
    title(ba_name)
    #plot(np.mean(brain_area_dict['scores_ncPCA_fw_ns' + ba_name][0],axis=0))
    #plot(np.mean(brain_area_dict['scores_cPCA_fw_ns_VISp'+ ba_name][0],axis=0))


#%% plot fw PC prediction for SG

sns.set_style("ticks")
sns.set_context("talk")
#this is temporary
array_of_ba2 = array_of_ba[[1,2,3,5]]
n = len(array_of_ba2)
c = 0
figure(figsize=(25,5))
for ba_name in array_of_ba2:
    c +=1
    subplot(1,n,c)
            
    n1 = np.shape(brain_area_dict['scores_cPCA_fw_sg_' + ba_name][0])[0]
    temp = np.hstack(brain_area_dict['scores_cPCA_fw_sg_' + ba_name][0])
    pc = np.tile(np.arange(temp.size/n1),n1);
    method = np.tile('cPCA',pc.size)
    
    df_cPCA = pd.DataFrame({'pc':pc,'R^2':temp,'Method':method})
    
    n1 = np.shape(brain_area_dict['scores_ncPCA_fw_sg_' + ba_name][0])[0]
    temp = np.hstack(brain_area_dict['scores_ncPCA_fw_sg_' + ba_name][0])
    pc = np.tile(np.arange(temp.size/n1),n1);
    method = np.tile('ncPCA',pc.size)
    
    df_ncPCA = pd.DataFrame({'pc':pc,'R^2':temp,'Method':method})
    
    n1 = np.shape(brain_area_dict['scores_PCA_sg_' + ba_name][0])[0]
    temp = np.hstack(brain_area_dict['scores_PCA_sg_' + ba_name][0])
    pc = np.tile(np.arange(temp.size/n1),n1);
    method = np.tile('PCA',pc.size)
    
    df_PCA = pd.DataFrame({'pc':pc,'R^2':temp,'Method':method})
    
    df = pd.concat([df_PCA,df_ncPCA,df_cPCA],ignore_index=True)
    if c == 1:
        sns.lineplot(data=df,x='pc',y='R^2',hue='Method',legend=True)
    else:
        sns.lineplot(data=df,x='pc',y='R^2',hue='Method',legend = False)
    title(ba_name)
    #plot(np.mean(brain_area_dict['scores_ncPCA_fw_ns' + ba_name][0],axis=0))
    #plot(np.mean(brain_area_dict['scores_cPCA_fw_ns_VISp'+ ba_name][0],axis=0))

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
