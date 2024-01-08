#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 18:41:07 2023

Script to analyze the Allen Institute Brain Observatory dataset using gcPCA.

I want to compare the representation of SG and NS in different brain areas using 
gcPCA, the higher the magnitude of top and bottom the larger the differences 
in coding there are. I want to include hippocampus and etc to use as a control

I might have to add a null distribution later on to pick top gcPCs

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

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore
import pickle
from numpy import linalg as LA

#%% parameters
min_n_cell = 30 #min number of cells in the brain area to be used

#%% import custom modules
#repo_dir = "/gs/gsfs0/users/edeolive/github/normalized_contrastive_PCA/" #repository dir
repo_dir = "/home/eliezyer/Documents/github/normalized_contrastive_PCA/"
sys.path.append(repo_dir)
from contrastive_methods import gcPCA
#%% preparing data to be loaded

data_directory = '/mnt/probox/allen_institute_data/ecephys/' # must be a valid directory in your filesystem
#data_directory = '/gs/gsfs0/users/edeolive/allen_institute_data/ecephys/'
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# getting brain observatory dataset
sessions = cache.get_session_table()
selected_sessions = sessions[(sessions.session_type == 'brain_observatory_1.1')]

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

results_dict  = {}
results_dict['top_gcPCA'] = []
results_dict['2nd_gcPCA'] = []
results_dict['bot_gcPCA'] = []
results_dict['brain_area'] = []
results_dict['session'] = []

for session_id in selected_sessions.index.values:

    loaded_session = cache.get_session_data(session_id)

    # getting natural scenes and static gratings info
    stimuli_info = loaded_session.get_stimulus_table(["drifting_gratings","static_gratings"])

    # getting spikes times and information
    temp_spk_ts = loaded_session.spike_times
    temp_spk_info  = loaded_session.units

    # here I'm picking all the brain areas recorded
    # spikes_info = temp_spk_info[temp_spk_info["ecephys_structure_acronym"].str.contains("V|LG")]
    spikes_info = temp_spk_info;
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
    df_stim_ns = stimuli_info.query("stimulus_name=='drifting_gratings'")
    ns_intervals = nap.IntervalSet(start=df_stim_ns.start_time.values,end=df_stim_ns.stop_time.values)
    ns_intervals_final = ns_intervals.merge_close_intervals(threshold=1)

    # df_stim_sg = stimuli_info.query("stimulus_name=='static_gratings' & orientation!='null'")
    df_stim_sg = stimuli_info.query("stimulus_name=='static_gratings'")
    sg_intervals = nap.IntervalSet(start=df_stim_sg.start_time.values,end=df_stim_sg.stop_time.values)
    sg_intervals_final = sg_intervals.merge_close_intervals(threshold=1)

    # binning through using a loop in the cells until I can find a better way
    #start timer
    # spikes_binned_ns = spikes_times.restrict(ns_intervals_final).count(0.100)
    # spikes_binned_ns = spikes_times.count(ep=ns_intervals,bin_size=0.250)
    spikes_binned_ns = np.empty((len(ns_intervals.values),len(spikes_times.data)))
    bins_ns = ns_intervals.values.flatten()
    for aa in np.arange(len(spikes_times.data)):
        tmp = np.array(np.histogram(spikes_times.data[aa].index.values,bins_ns))
        spikes_binned_ns[:,aa] = tmp[0][np.arange(0,tmp[0].shape[0],2)]
    
    # same method of binning through a loop in the cells, but for static gratings
    # spikes_binned_sg = spikes_times.restrict(sg_intervals_final).count(0.100)
    # spikes_binned_sg = spikes_times.count(ep=sg_intervals,bin_size=0.250)
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

    spikes_zsc_ns = zscore(spikes_binned_ns) #remove this later and update the variables name accordingly
    spikes_zsc_sg = zscore(spikes_binned_sg)

    spikes_zsc_ns[np.isnan(spikes_zsc_ns)] = 0
    spikes_zsc_sg[np.isnan(spikes_zsc_sg)] = 0
    #%% getting ncPCA and cPCA loadings

    # fw ns and bw ns cPCA loadings, ns PCA loadings and fw ns cPCA loadings
    
    """Here's how store the results, save a dictionary with a column of 
        scores (acc/error) | fold | PCnumber | method | fw/bw | variable (ns/sg) | brain area name | session | 
        
    """
        
    for ba_name in array_of_ba:
        units_idx = spikes_info["ecephys_structure_acronym"]=='VISp'

        """Code commented below is to check firing rate, not fully implemented yet!"""
        #checking firing rate to include the cells or not! natural scenes
        #total_ns_dur = np.diff(ns_intervals.values).sum()
        #ns_fr = spikes_binned_ns[:,units_idx].sum(axis=0)/total_ns_dur
        #checking for static gratings
        #total_sg_dur = np.diff(sg_intervals.values).sum()
        #sg_fr = spikes_binned_sg[:,units_idx].sum(axis=0)/total_sg_dur
        
        #checking for amount of bins where no cell fires, if more than 20% then throw the brain area out 
        #this is not properly implemented
        # test_sg = spikes_binned_sg[:,units_idx].sum(axis=1)
        # ratio_dur_sg = np.sum(test_sg==0)/test_sg.shape[0]
        # if ratio_dur_sg>0.20:
        #     continue
        
        # test_ns = spikes_binned_ns[:,units_idx].sum(axis=1)
        # ratio_dur_ns = np.sum(test_ns==0)/test_ns.shape[0]
        # if ratio_dur_ns>0.20:
        #     continue
        
        """ALSO ADD RETURNING NULL IF GCPCA DOESN'T WORK!!!!"""
        if units_idx.values.sum() >= min_n_cell:
            print('session:')
            print(session_id)
            print('brain area:')
            print(ba_name)
            #creating gcPCA model
            gcpca_mdl = gcPCA(method='v4');
            
            #run gcPCA between NS and SG and save the gcPCA, it is not cross validated
            gcpca_mdl.fit(spikes_zsc_ns[:,units_idx],spikes_zsc_sg[:,units_idx])
            
            temp_values = gcpca_mdl.gcPCA_values_
            
            # save another dict with the PC and ncPCA loadings
            results_dict['top_gcPCA'].append(temp_values[0])
            results_dict['2nd_gcPCA'].append(temp_values[1])
            results_dict['bot_gcPCA'].append(temp_values[-1])
            results_dict['brain_area'].append(ba_name)
            results_dict['session'].append(session_id)

#%% add dictionary to pickle file
#file_path = r"\home\pranjal\Documents\pkl_sessions" + "\\" + str(session_id)
# saving the brain_area_dict file

df= pd.DataFrame(data=results_dict)
with open('/mnt/SSD4TB/ncPCA_files/gcPCA_values_AIBO.pickle', 'wb') as handle:
         pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)


# TODO: add plotting code

#%% reading file
#with open('/mnt/SSD4TB/ncPCA_files/dataframe_AIBO_cumul_accuracy.pickle', 'rb') as handle:
#     x = pickle.load(handle)


#%% parameters for plotting
plt.rcParams['figure.dpi'] = 500
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['axes.linewidth']  = 1.5
plt.rcParams['font.size'] = 12

data_gcPCA = pd.DataFrame.from_dict(results_dict)

# string_list = ['PO','APN','VPM','MGv','LGd','LP','VISp','VISl','VISrl','VISal','VISpm','VISam','CA1','grey']
# string_list = ['PO','APN','VPM','MGv','LGd','LP','VISp','VISl','VISrl','VISal','VISpm','VISam','CA1','CA3','DG','ProS']
string_list = ['LGd','LP','VISp','VISl','VISrl','VISal','VISpm','VISam','CA1','CA3','DG','ProS']
sns.boxplot(data=data_gcPCA,y = 'brain_area',x='top_gcPCA',order=string_list)
plt.title('Shuffled Movie x Original Movie')
# plt.xlim((0,0.9))
plt.grid()
sns.boxplot(data=data_gcPCA,y = 'brain_area',x='bot_gcPCA',order=string_list)
plt.title('Shuffled Movie x Original Movie')
# plt.xlim((-0.9,0))
plt.grid()
#%%
g = sns.scatterplot(data=data_gcPCA,y = 'top_gcPCA',x='bot_gcPCA',hue='brain_area',hue_order=string_list)
sns.move_legend(g, 'best')