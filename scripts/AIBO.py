#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Jan  4 08:40:25 2023

Script to analyze the Allen Institute Brain Observatory dataset using ncPCA. Here we used the
extracellular ephys recordings during visual stimulation.

The goal is to:
    1) Get spikes during static grating and natural scenes
    2) In a cross-validated manner get PCs of natural scenes and static gratings and ncPCs using both
    3) Show that ncPCs perform as well or better than PCs of that specific state

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
#repo_dir = "/gs/gsfs0/users/edeolive/github/normalized_contrastive_PCA/" #repository dir
repo_dir = "/home/eliezyer/Documents/github/normalized_contrastive_PCA/"
sys.path.append(repo_dir)
from ncPCA import ncPCA
from ncPCA import cPCA
import ncPCA_project_utils as utils #this is going to be our package of reusable functions
#%% preparing data to be loaded

data_directory = '/mnt/probox/allen_institute_data/ecephys/' # must be a valid directory in your filesystem
#data_directory = '/gs/gsfs0/users/edeolive/allen_institute_data/ecephys/'
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
scores_total         = []
track_fold           = []
component_num        = []
track_method         = []
direction_cumulative = []
stim_type            = []
number_units         = []
brain_area_name      = []
session_name         = []

loadings_dict  = {}
loadings_dict['ncPCA'] = []
loadings_dict['PCns'] = []
loadings_dict['PCsg'] = []
loadings_dict['brain_area'] = []
loadings_dict['session'] = []

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
    
    """Here's how store the results, save a dictionary with a column of 
        scores (acc/error) | fold | PCnumber | method | fw/bw | variable (ns/sg) | brain area name | session | 
        
    """
        
    for ba_name in array_of_ba:
        units_idx = spikes_info["ecephys_structure_acronym"]==ba_name

        """Code commented below is to check firing rate, not fully implemented yet!"""
        #checking firing rate to include the cells or not! natural scenes
        #total_ns_dur = np.diff(ns_intervals.values).sum()
        #ns_fr = spikes_binned_ns[:,units_idx].sum(axis=0)/total_ns_dur
        #checking for static gratings
        #total_sg_dur = np.diff(sg_intervals.values).sum()
        #sg_fr = spikes_binned_sg[:,units_idx].sum(axis=0)/total_sg_dur
        
        #checking for amount of bins where no cell fires, if more than 20% then throw the brain area out 
        test_sg = spikes_binned_sg[:,units_idx].sum(axis=1)
        ratio_dur_sg = np.sum(test_sg==0)/test_sg.shape[0]
        if ratio_dur_sg>0.20:
            continue
        
        test_ns = spikes_binned_ns[:,units_idx].sum(axis=1)
        ratio_dur_ns = np.sum(test_ns==0)/test_ns.shape[0]
        if ratio_dur_ns>0.20:
            continue
        
        """ALSO ADD RETURNING NULL IF NCPCA DOESN'T WORK!!!!"""
        if units_idx.values.sum() >= min_n_cell:
            
            ### This is where the cross validation starts
            #first we set up the train and test set for both datasets
            #
            
            #getting CV indices for SG separately because it has different size than NS
            fold = 0
            sgcv = kcv.split(spikes_zsc_sg)
            for train_ind_ns,test_ind_ns in kcv.split(spikes_zsc_ns):
                fold+=1
                
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
                train_sg[np.isnan(train_sg)] = 0
                test_sg[np.isnan(test_sg)]   = 0
            
                """ This block will do ncPCA """
                # fitting ncPCA to train set
                ncPCA_mdl = ncPCA(basis_type='intersect',Nshuffle=0)
                ncPCA_mdl.fit(train_sg,train_ns)
                if ncPCA_mdl.number_of_shared_basis==0:
                    continue #this could be a cause of some brain areas having less than 5 folds
                
                loadings_ncpca = ncPCA_mdl.loadings_
                _,temp_fw_ns, temp_bw_ns =  utils.cumul_accuracy_projected(train_ns, labels_ns_train, test_ns, labels_ns_test,
                                             loadings_ncpca, analysis='both',step_size=1)

                
                ncPCs_num,temp_fw_sg, temp_bw_sg =  utils.cumul_error_projected(train_sg, labels_sg_train, test_sg, labels_sg_test,
                                                 loadings_ncpca, analysis='both',step_size=1)
                
                
                """ This block will calculate the scores for regular PCA """
                #the PCs for prediction also need to be cross validated
                _,_,Vns = np.linalg.svd(train_ns,full_matrices=False)
                _,_,Vsg = np.linalg.svd(train_sg,full_matrices=False)


                PCns_num,temp_ns = utils.cumul_accuracy_projected(train_ns, labels_ns_train, test_ns,
                                                                        labels_ns_test, Vns.T,step_size=1)
                PCsg_num,temp_sg = utils.cumul_error_projected(train_sg, labels_sg_train, test_sg,
                                                            labels_sg_test, Vsg.T,step_size=1)
                
                scores_total.append(np.concatenate((temp_fw_ns,
                                                    temp_bw_ns,
                                                    temp_fw_sg,
                                                    temp_bw_sg,
                                                    temp_ns,
                                                    temp_sg)))
                #array of PC and ncPC folds
                track_fold.append(np.concatenate((np.tile(fold,len(temp_fw_ns)),
                                                 np.tile(fold,len(temp_bw_ns)),
                                                 np.tile(fold,len(temp_fw_sg)),
                                                 np.tile(fold,len(temp_bw_sg)),
                                                 np.tile(fold,len(temp_ns)),
                                                 np.tile(fold,len(temp_sg)))))
                
                #array of PC and ncPC numbers and order
                component_num.append(np.concatenate((ncPCs_num,
                                                 ncPCs_num,
                                                 ncPCs_num,
                                                 ncPCs_num,
                                                 PCns_num,
                                                 PCsg_num)))
                
                #array of method string
                track_method.append(np.concatenate((np.tile('ncPCA',len(temp_fw_ns)),
                                                 np.tile('ncPCA',len(temp_bw_ns)),
                                                 np.tile('ncPCA',len(temp_fw_sg)),
                                                 np.tile('ncPCA',len(temp_bw_sg)),
                                                 np.tile('PCAns',len(temp_ns)),
                                                 np.tile('PCAsg',len(temp_sg)))))
                
                #array of fw/bw
                direction_cumulative.append(np.concatenate((np.tile('fw',len(temp_fw_ns)),
                                                 np.tile('bw',len(temp_bw_ns)),
                                                 np.tile('fw',len(temp_fw_sg)),
                                                 np.tile('bw',len(temp_bw_sg)),
                                                 np.tile('fw',len(temp_ns)),
                                                 np.tile('fw',len(temp_sg)))))
                
                #array of variable decoded (ns/sg)
                stim_type.append(np.concatenate((np.tile('NS',len(temp_fw_ns)),
                                                 np.tile('NS',len(temp_bw_ns)),
                                                 np.tile('SG',len(temp_fw_sg)),
                                                 np.tile('SG',len(temp_bw_sg)),
                                                 np.tile('NS',len(temp_ns)),
                                                 np.tile('SG',len(temp_sg)))))
                
                #number of unitts in this session/brain area
                number_units.append(np.tile(sum(units_idx.values),len(temp_fw_ns)+len(temp_bw_ns)+
                                                      len(temp_fw_sg)+len(temp_bw_sg)+
                                                      len(temp_ns)+len(temp_sg)))
                #brain area name
                brain_area_name.append(np.tile(ba_name,len(temp_fw_ns)+len(temp_bw_ns)+
                                                      len(temp_fw_sg)+len(temp_bw_sg)+
                                                      len(temp_ns)+len(temp_sg)))
                
                #session name
                session_name.append(np.tile(session_id,len(temp_fw_ns)+len(temp_bw_ns)+
                                                      len(temp_fw_sg)+len(temp_bw_sg)+
                                                      len(temp_ns)+len(temp_sg)))
                
            # save another dict with the PC and ncPCA loadings
                loadings_dict['ncPCA'].append(loadings_ncpca)
                loadings_dict['PCns'].append(Vns.T)
                loadings_dict['PCsg'].append(Vsg.T)
                loadings_dict['brain_area'].append(ba_name)
                loadings_dict['session'].append(session_id)\


"saving the main results"
main_results = {}
main_results['scores']            = np.hstack(scores_total)
main_results['folds']             = np.hstack(track_fold) 
main_results['PC_number']         = np.hstack(component_num)
main_results['Method']            = np.hstack(track_method)
main_results['direction']         = np.hstack(direction_cumulative)
main_results['stim_type']         = np.hstack(stim_type)
main_results['units_number']      = np.hstack(number_units)
main_results['brain_region_name'] = np.hstack(brain_area_name)
main_results['session_name']      = np.hstack(session_name)

#%% add dictionary to pickle file
#file_path = r"\home\pranjal\Documents\pkl_sessions" + "\\" + str(session_id)
# saving the brain_area_dict file

df= pd.DataFrame(data=main_results)
with open('/mnt/SSD4TB/ncPCA_files/dataframe_AIBO_cumul_accuracy.pickle', 'wb') as handle:
         pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)


# TODO: add plotting code

#%% reading file
#with open('/mnt/SSD4TB/ncPCA_files/dataframe_AIBO_cumul_accuracy.pickle', 'rb') as handle:
#     x = pickle.load(handle)


#%% parameters for plotting
rcParams['figure.dpi'] = 500
rcParams['lines.linewidth'] = 2.5
rcParams['axes.linewidth']  = 1.5
rcParams['font.size'] = 12

