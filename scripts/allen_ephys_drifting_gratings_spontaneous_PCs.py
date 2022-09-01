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
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from ncPCA import ncPCA

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from sklearn import svm
from sklearn.model_selection import cross_val_score
from matplotlib.pyplot import *
from scipy import stats

#%% parameters
min_n_cell = 50


#%% preparing data to be loaded

data_directory = '/mnt/probox/allen_institute_data/ecephys/' # must be a valid directory in your filesystem
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# getting only the sessions from functonal connectivity dataset and that contain cells from VISp
sessions = cache.get_session_table()
#selected_sessions = sessions[(sessions.session_type == 'functional_connectivity') & \
#                            (['VISp' in acronyms for acronyms in
#                               sessions.ecephys_structure_acronyms])]
    
    
selected_sessions = sessions[(sessions.session_type == 'brain_observatory_1.1')]
    
#%% getting how many VISp units there are there
#number_of_neurons = np.zeros(selected_sessions.shape[0]);
#c = 0;
#for session_id, row in selected_sessions.iterrows():
#    
#    loaded_session = cache.get_session_data(session_id)
#    number_of_neurons[c] = loaded_session.structurewise_unit_counts.VISp
#    c = c+1;
#    print('Session ' + str(session_id) + ' completed. Found ' + str(number_of_neurons[c-1]) + ' VISp cells')
    
    
#%% performing analysis, session loop starts here

#ses2use = np.where(number_of_neurons==np.max(number_of_neurons));
#ses2use = number_of_neurons==np.max(number_of_neurons);
sessions2load = selected_sessions.index.values

"this for is for later, once I have the rest of the loop analysis figured out"
#for a in np.arange(0,len(sessions2load)):
#   session_id = sessions2load[a]
session_id = 715093703;
loaded_session = cache.get_session_data(session_id)

#getting natural scenes and static gratings info
stimuli_info = loaded_session.get_stimulus_table(["natural_scenes","static_gratings"])

#getting spikes times and information
temp_spk_ts = loaded_session.spike_times
temp_spk_info  = loaded_session.units

#here I'm picking all the brain areas that are related to visual system, i.e.,
#starting with V or LG
spikes_info = temp_spk_info[temp_spk_info["ecephys_structure_acronym"].str.contains("V|LG")]
units2use = spikes_info.index.values

temp_spk_ts_2 = {}
for aa in np.arange(0,len(units2use)):
    temp_spk_ts_2[aa] = temp_spk_ts[units2use[aa]]

#passing to pynapple ts group
spikes_times = nap.TsGroup(temp_spk_ts_2)

#adding structure info into the spike times TSgroup 
structs = pd.DataFrame(index=np.arange(len(units2use)),data = spikes_info.ecephys_structure_acronym.values,columns=['struct'])
spikes_times.set_info(structs)

#passing natural scenes to intervalsets
df_stim_ns = stimuli_info.query("stimulus_name=='natural_scenes'");
ns_intervals = nap.IntervalSet(start=df_stim_ns.start_time.values,end=df_stim_ns.stop_time.values)

df_stim_sg = stimuli_info.query("stimulus_name=='static_gratings'");
sg_intervals = nap.IntervalSet(start=df_stim_sg.start_time.values,end=df_stim_sg.stop_time.values)

#binning through using a loop in the cells until I can find a better way
spikes_binned_ns = np.empty((len(ns_intervals.values),len(spikes_times.data)))
bins_ns = ns_intervals.values.flatten()
for aa in np.arange(len(spikes_times.data)):
    tmp = np.array(np.histogram(spikes_times.data[aa].index.values,bins_ns))
    spikes_binned_ns[:,aa] = tmp[0][np.arange(0,tmp[0].shape[0],2)]

#same method of binning through a loop in the cells, but for static gratings
spikes_binned_sg = np.empty((len(sg_intervals.values),len(spikes_times.data)))
bins_sg = sg_intervals.values.flatten()
for aa in np.arange(len(spikes_times.data)):
    tmp = np.array(np.histogram(spikes_times.data[aa].index.values,bins_sg))
    spikes_binned_sg[:,aa] = tmp[0][np.arange(0,tmp[0].shape[0],2)]
    
#getting labels
sg_labels = df_stim_sg.stimulus_condition_id.values
ns_labels = df_stim_ns.stimulus_condition_id.values
#%% performing prediction

array_of_ba = spikes_info["ecephys_structure_acronym"].unique();
scores_ns = np.empty((5,len(array_of_ba)))
scores_sg = np.empty((5,len(array_of_ba)))

spikes_zsc_ns = stats.zscore(spikes_binned_ns)
spikes_zsc_sg = stats.zscore(spikes_binned_sg)
'''for aa in np.arange(len(array_of_ba)):
    clf_ns = svm.SVC()
    clf_sg = svm.SVC()
    units_idx = spikes_info["ecephys_structure_acronym"]==array_of_ba[aa]
    scores_ns[:,aa] = cross_val_score(clf_ns,spikes_zsc_ns[:,units_idx],ns_labels,cv=5)
    scores_sg[:,aa] = cross_val_score(clf_sg,spikes_zsc_sg[:,units_idx],sg_labels,cv=5)
'''
#%% getting ncPCA loadings
brain_area_dict = {};
for aa in np.arange(len(array_of_ba)):
    units_idx = spikes_info["ecephys_structure_acronym"]==array_of_ba[aa]
    if sum(units_idx.values) >= min_n_cell:
        X,S = ncPCA(spikes_binned_sg[:,units_idx],spikes_binned_ns[:,units_idx])
        _,_,Vns = np.linalg.svd(spikes_zsc_ns[:,units_idx],full_matrices=False)  
        scores_vns = np.empty((5,len(Vns)))
        scores_x = np.empty((5,len(Vns)))
        for aaa in np.arange(len(Vns)):
            clf_vns = svm.SVC()
            clf_x = svm.SVC()
            scores_x[:,aaa] = cross_val_score(clf_x,np.dot(spikes_zsc_ns[:,units_idx],X[:,:aaa+1]),ns_labels,cv=5)
            scores_vns[:,aaa] = cross_val_score(clf_vns,np.dot(spikes_zsc_ns[:,units_idx],Vns[:aaa+1,:].T),ns_labels,cv=5)
        brain_area_dict['scores_x_'+array_of_ba[aa]] = scores_x
        brain_area_dict['scores_vns_'+array_of_ba[aa]] = scores_vns
        
#%% making plots of the score prediction of natural images of ncPCA vs regular PCA

figure(figsize=(30, 6), dpi=150)
array_of_ba2 = ['LGd','VISp','VISrl','VISpm']; #DO THIS AUTOMATICALLY NEXT TIME
for aa in np.arange(len(array_of_ba2)):
    acc_pca_ns = brain_area_dict['scores_x_'+array_of_ba2[aa]]
    acc_ncPCA_ns = brain_area_dict['scores_vns_'+array_of_ba2[aa]]
    
    x = np.arange(start=1,stop=acc_pca_ns.shape[1]+1)
    
    pca_mu = np.mean(acc_pca_ns,axis=0);
    pca_err = np.std(acc_pca_ns,axis=0)/np.sqrt(5);
    
    ncpca_mu = np.mean(acc_ncPCA_ns,axis=0);
    ncpca_err = np.std(acc_ncPCA_ns,axis=0)/np.sqrt(5);
    
    subplot(1,5,aa+1)
    errorbar(x,pca_mu,yerr=pca_err,label = 'PCA')
    errorbar(x,ncpca_mu,yerr=ncpca_err,label='ncPCA')
    title(array_of_ba2[aa])
    xlabel('cumulative PCs')

legend(loc = 'lower right')
subplot(1,5,1)
ylabel('Accuracy')



#%% testing prediction over X and over Vns



#%% getting stimulus epoch, stimulus identity and spikes so we can conduct analysis

#this gets the stimulus table specifics test = loaded_session.get_stimulus_table("natural_movie_one_more_repeats")
#this gets the start and stop times. test.start_time test.stop_time
#this gets the spikes spike_times = loaded_session.spike_times
#this gets unit information unit_info=loaded_session.units
#this gets the brain area acronym unit_info["ecephys_structure_acronym"]