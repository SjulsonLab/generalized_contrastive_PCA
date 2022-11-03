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
import sys
from ncPCA import ncPCA

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from sklearn import svm
from sklearn.model_selection import cross_val_score
from matplotlib.pyplot import *
from scipy import stats
from contrastive import CPCA
import pickle
from sklearn.model_selection import train_test_split
from numpy import linalg as LA

#%% parameters
min_n_cell = 50



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
with open(r'/home/pranjal/Documents/pkl_sessions/array_units', 'wb') as handle:
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
    spikes_binned_ns = np.empty((len(ns_intervals.values),len(spikes_times.data)))
    bins_ns = ns_intervals.values.flatten()
    for aa in np.arange(len(spikes_times.data)):
        tmp = np.array(np.histogram(spikes_times.data[aa].index.values,bins_ns))
        spikes_binned_ns[:,aa] = tmp[0][np.arange(0,tmp[0].shape[0],2)]

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

    spikes_zsc_ns = stats.zscore(spikes_binned_ns)
    spikes_zsc_sg = stats.zscore(spikes_binned_sg)

    #%% getting ncPCA and cPCA loadings

    # fw ns and bw ns cPCA loadings, ns PCA loadings and fw ns cPCA loadings
    brain_area_dict = {};
    array_of_ba = ['LGd','VISp','VISrl','VISpm']

    for aa in np.arange(len(array_of_ba)):
        units_idx = spikes_info["ecephys_structure_acronym"]==array_of_ba[aa]

        if sum(units_idx.values) >= min_n_cell:
            X_fw_ns,S = ncPCA(spikes_binned_sg[:,units_idx],spikes_binned_ns[:,units_idx])
            X_bw_ns = np.flip(X_fw_ns, axis=1)
            X_ = {'X_fw_ns':X_fw_ns, 'X_bw_ns':X_bw_ns}

            _,_,Vns = np.linalg.svd(spikes_zsc_ns[:,units_idx],full_matrices=False)

            n_components = X_fw_ns.shape[0]
            mdl = CPCA(n_components)
            projected_data, alpha_values = mdl.fit_transform(spikes_binned_ns[:,units_idx], spikes_binned_sg[:,units_idx], return_alphas= True)
            brain_area_dict['alpha_value_of_fw_ns'+array_of_ba[aa]] = alpha_values

            scores_vns_ns = np.empty((5,len(Vns)))
            scores_x_fw_ns = np.empty((5, len(Vns)))
            scores_x_bw_ns = np.empty((5,len(Vns)))
            scores_cpca_ns = np.empty((len(alpha_values),5,len(Vns)))


            for aaa in np.arange(len(Vns)):
                PCA_ = np.dot(spikes_zsc_ns[:,units_idx],Vns[:aaa+1,:].T)
                PCA_x_train, PCA_x_test, PCA_y_train, PCA_y_test = train_test_split(
                PCA_, ns_labels, test_size=0.3, random_state=0)
                clf_vns = svm.SVC().fit(PCA_x_train, PCA_y_train) # PCA
                scores_vns_ns[:, aaa] = cross_val_score(clf_vns, PCA_x_test, PCA_y_test, cv=5)

                for X in X_:
                    ncPCA_= np.dot(spikes_zsc_ns[:,units_idx],X[:,:aaa+1])
                    ncPCA_x_train, ncPCA_x_test, ncPCA_y_train, ncPCA_y_test = train_test_split(
                    ncPCA_, ns_labels, test_size=0.3, random_state=0)
                    clf_x = svm.SVC().fit(ncPCA_x_train, ncPCA_y_train) # ncPCA

                    if str(X)=='X_fw_ns':
                        scores_x_fw_ns[:,aaa] = cross_val_score(clf_x,ncPCA_x_test, ncPCA_y_test,cv=5)

                    if str(X) == 'X_bw_ns':
                        scores_x_bw_ns[:, aaa] = cross_val_score(clf_x, ncPCA_x_test, ncPCA_y_test, cv=5)


            for alpha in np.arange(len(alpha_values)):

                fg_cov = spikes_zsc_ns[:,units_idx].T.dot(spikes_zsc_ns[:,units_idx])/(spikes_zsc_ns[:,units_idx].shape[0]-1)
                bg_cov = spikes_zsc_sg[:,units_idx].T.dot(spikes_zsc_sg[:,units_idx])/(spikes_zsc_sg[:,units_idx].shape[0]-1)
                sigma = fg_cov - alpha_values[alpha]*bg_cov
                w, v = LA.eig(sigma)
                eig_idx = np.argpartition(w, -n_components)[-n_components:]
                eig_idx = eig_idx[np.argsort(-w[eig_idx])]
                v_top = v[:,eig_idx]

                for aaa in np.arange(len(Vns)):

                    cPCA_= np.dot(spikes_zsc_ns[:,units_idx],v_top[:,:aaa+1])
                    cPCA_x_train, cPCA_x_test, cPCA_y_train, cPCA_y_test = train_test_split(
                    cPCA_, ns_labels, test_size=0.3, random_state=0)
                    clf_cpca = svm.SVC().fit(cPCA_x_train, cPCA_y_train) # cPCA
                    scores_cpca_ns[alpha,:,aaa] = cross_val_score(clf_cpca,cPCA_x_test, cPCA_y_test,cv=5)

            brain_area_dict['scores_x_fw_ns_' + array_of_ba[aa]] = scores_x_fw_ns
            brain_area_dict['scores_x_bw_ns_'+array_of_ba[aa]] = scores_x_bw_ns
            brain_area_dict['scores_vns_ns_'+array_of_ba[aa]] = scores_vns_ns
            brain_area_dict['scores_cpca_ns_'+array_of_ba[aa]] = scores_cpca_ns


    #############

    # fw sg and bw sg cPCA loadings, sg PCA loadings and sg cPCA loadings

    for aa in np.arange(len(array_of_ba)):
        # units_idx = spikes_info["ecephys_structure_acronym"] == array_of_ba[aa]

        if sum(units_idx.values) >= min_n_cell:
            X_fw_sg, _ = ncPCA(spikes_binned_ns[:, units_idx], spikes_binned_sg[:, units_idx])
            X_bw_sg = np.flip(X_fw_sg, axis=1)
            X_ = {'X_fw_sg':X_fw_sg, 'X_bw_sg':X_bw_sg}

            _, _, Vns = np.linalg.svd(spikes_zsc_sg[:, units_idx], full_matrices=False)

            n_components = X_fw_sg.shape[0]
            mdl = CPCA(n_components)
            projected_data, alpha_values = mdl.fit_transform(spikes_binned_sg[:, units_idx],
                                                             spikes_binned_ns[:, units_idx], return_alphas=True)
            brain_area_dict['alpha_value_of_fw_sg' + array_of_ba[aa]] = alpha_values

            scores_vns_sg = np.empty((5, len(Vns)))
            scores_x_fw_sg = np.empty((5, len(Vns)))
            scores_x_bw_sg = np.empty((5, len(Vns)))
            scores_cpca_sg = np.empty((len(alpha_values), 5, len(Vns)))

            for aaa in np.arange(len(Vns)):
                PCA_ = np.dot(spikes_zsc_sg[:, units_idx], Vns[:aaa+1,:].T)
                PCA_x_train, PCA_x_test, PCA_y_train, PCA_y_test = train_test_split(
                    PCA_, sg_labels, test_size=0.3, random_state=0)
                clf_vns = svm.SVC().fit(PCA_x_train, PCA_y_train)  # PCA
                scores_vns_sg[:, aaa] = cross_val_score(clf_vns, PCA_x_test, PCA_y_test, cv=5)

                for X_keys, X_items in X_.items():
                    ncPCA_ = np.dot(spikes_zsc_sg[:, units_idx], X_items[:,:aaa+1])
                    ncPCA_x_train, ncPCA_x_test, ncPCA_y_train, ncPCA_y_test = train_test_split(
                        ncPCA_, sg_labels, test_size=0.3, random_state=0)
                    clf_x = svm.SVC().fit(ncPCA_x_train, ncPCA_y_train)  # ncPCA

                    if X_keys == 'X_fw_sg':
                        scores_x_fw_sg[:, aaa] = cross_val_score(clf_x, ncPCA_x_test, ncPCA_y_test, cv=5)

                    if X_keys == 'X_bw_sg':
                        scores_x_bw_sg[:, aaa] = cross_val_score(clf_x, ncPCA_x_test, ncPCA_y_test, cv=5)

            for alpha in np.arange(len(alpha_values)):

                fg_cov = spikes_zsc_sg[:, units_idx].T.dot(spikes_zsc_ns[:, units_idx]) / (
                            spikes_zsc_sg[:, units_idx].shape[0] - 1)
                bg_cov = spikes_zsc_ns[:, units_idx].T.dot(spikes_zsc_sg[:, units_idx]) / (
                            spikes_zsc_ns[:, units_idx].shape[0] - 1)
                sigma = fg_cov - alpha_values[alpha] * bg_cov
                w, v = LA.eig(sigma)
                eig_idx = np.argpartition(w, -n_components)[-n_components:]
                eig_idx = eig_idx[np.argsort(-w[eig_idx])]
                v_top = v[:, eig_idx]

                for aaa in np.arange(len(Vns)):
                    cPCA_ = np.dot(spikes_zsc_sg[:, units_idx], v_top[:, :aaa + 1])
                    cPCA_x_train, cPCA_x_test, cPCA_y_train, cPCA_y_test = train_test_split(
                        cPCA_, sg_labels, test_size=0.3, random_state=0)
                    clf_cpca = svm.SVC().fit(cPCA_x_train, cPCA_y_train)  # cPCA
                    scores_cpca_sg[alpha, :, aaa] = cross_val_score(clf_cpca, cPCA_x_test, cPCA_y_test, cv=5)

            brain_area_dict['scores_x_fw_sg_' + array_of_ba[aa]] = scores_x_fw_sg
            brain_area_dict['scores_x_bw_sg_' + array_of_ba[aa]] = scores_x_bw_sg
            brain_area_dict['scores_vns_sg_' + array_of_ba[aa]] = scores_vns_sg
            brain_area_dict['scores_cpca_sg_' + array_of_ba[aa]] = scores_cpca_sg


    #%% add dictionary to pickle file
    file_path = r"\home\pranjal\Documents\pkl_sessions" + "\\" + str(session_id)

    # saving the brain_area_dict file
    with open(file_path, 'wb') as handle:
         pickle.dump(brain_area_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# TODO: add plotting code
#%% plotting
dir_list = os.listdir(r"/home/pranjal/Documents/pkl_sessions")
session_id = []
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
