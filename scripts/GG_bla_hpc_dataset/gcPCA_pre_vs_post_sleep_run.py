#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:03:34 2024

@author: eliezyer
script for gcPCA analysis on HPC-BLA dataset, testing post - pre sleep dimensions
"""


#%% importing essentials
import os
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore
from scipy.signal import savgol_filter
import mat73 #to load matlab v7.3 files
import pickle
from numpy import linalg as LA

#%% parameters
min_n_cell = 30 #min number of cells in the brain area to be used
min_fr = 0.01 #minimum firing rate to be included in the analysis
bin_size = 0.01 # 0.01
bin_size_task = 0.05; # 0.05
plt.rcParams['figure.dpi'] = 300
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 12
std_conv = 2  # standard deviation for convolution (in num bins units)
wind_cov = 5  # window size for convolution (in num bins units)
#%% import custom modules
#repo_dir = "/gs/gsfs0/users/edeolive/github/generalized_contrastive_PCA/" #repository dir
repo_dir = "/home/eliezyer/Documents/github/generalized_contrastive_PCA/"
sys.path.append(repo_dir)
from contrastive_methods import gcPCA

#%% defining functions
def extract_trials(temp,temp_spd):
    """Function to extract trials from GG dataset where animals
    run in a linear track with airpuff
    
    todo:
        [ ] add end of trial to reach the end of the maze 
    instead of time constrain < IMPORTANT. Add to start around 150 and to end after 450, and vice versa
    [ ] throw out teleportations
    """
    
    max_len = 10  # max length of trial, in seconds
    min_len = 0.5  # min length of trial, in seconds
    min_clip, max_clip = 150, 450  # this is for clipping the data
    min_dist = 150  # in cm
    min_speed = 3  # minimal speed is 3 cm/s
    # speed = nap.tsd(np.abs(np.diff(temp.clip(lower=min_clip,upper=max_clip).values))
    speed = nap.Tsd(np.array(temp.index[:-1]),np.abs(np.diff(temp.values))/np.diff(temp.index))
    logical_left  = np.diff(temp.as_series().clip(lower=min_clip,upper=max_clip).values)<0
    logical_right = np.diff(temp.as_series().clip(lower=min_clip,upper=max_clip).values)>0
    
    logical_left  = np.append(logical_left,False)
    logical_right = np.append(logical_right,False)
    logical_left  = np.insert(logical_left,0,False)
    logical_right = np.insert(logical_right,0,False)
    ts = temp.as_series().clip(lower=min_clip,upper=max_clip).index
    
    #finding start and stop of left runs (here we lose an index, correct later)
    temp_st = np.argwhere(np.diff(logical_left.astype(int))==1)+1
    temp_sp = np.argwhere(np.diff(logical_left.astype(int))==-1)
    
    #picking only the intervals that lasted > min_len s and < max_len s
    start1 = ts[temp_st]
    stop1  = ts[temp_sp]
    int2keep = ((stop1 - start1)>min_len) * ((stop1 - start1)<max_len)
    start2 = start1[int2keep].copy()
    stop2  = stop1[int2keep].copy()
    trials2keep = []
    for a in np.arange(len(start2)):
        interval = nap.IntervalSet(start=start2[a],end=stop2[a])
        """ REWRITING THE TRIAL LENGTH BASED ON SPEED"""
        tempt = speed.restrict(interval).index[speed.restrict(interval).values > min_speed]
        if tempt.size > 0:
            new_interval = nap.IntervalSet(start=tempt[0],end=tempt[-1])
            if (temp.restrict(new_interval).as_series().max() - temp.restrict(new_interval).as_series().min())>min_dist:  # min distance in cm that the trial is getting
                trials2keep.append(True)
                start2[a],stop2[a]=tempt[0],tempt[-1]
            else:
                trials2keep.append(False)
        else:
            trials2keep.append(False)
    
    left_runs_interval = nap.IntervalSet(start = start2[trials2keep],end = stop2[trials2keep])
    
    #finding start and stop of right runs (here we lose an index, correct later)
    temp_st = np.argwhere(np.diff(logical_right.astype(int))==1)+1
    temp_sp = np.argwhere(np.diff(logical_right.astype(int))==-1)
    
    #picking only the intervals that lasted > 1 s and <3s
    start1 = ts[temp_st]
    stop1  = ts[temp_sp]
    int2keep = ((stop1 - start1)>min_len) * ((stop1 - start1)<max_len)
    start2 = start1[int2keep].copy()
    stop2  = stop1[int2keep].copy()
    
    trials2keep = []
    for a in np.arange(len(start2)):
        interval = nap.IntervalSet(start=start2[a],end=stop2[a])
        tempt = speed.restrict(interval).index[speed.restrict(interval).values > min_speed]
        if tempt.size > 0:
            new_interval = nap.IntervalSet(start=tempt[0],end=tempt[-1])
            if (temp.restrict(new_interval).as_series().max() - temp.restrict(new_interval).as_series().min())>min_dist:  # min distance in cm that the trial is getting
                trials2keep.append(True)
                start2[a],stop2[a]=tempt[0],tempt[-1]
            else:
                trials2keep.append(False)
        else:
            trials2keep.append(False)
    
    right_runs_interval = nap.IntervalSet(start = start2[trials2keep],end = stop2[trials2keep])
    
    return left_runs_interval,right_runs_interval

def trials_projection(proj_df,intervals):
    """Method to define the activity in trials of the projected 
    data in specific dimensions"""
    #gcPCA projection on every left trial run
    c = 0
    for a in intervals.values:
        c+=1
        temp_is = nap.IntervalSet(a[0],end=a[1])
        tmpr = proj_df.restrict(temp_is)
        tempdf = tmpr.as_dataframe()
        # tempdf = tempdf.rolling(10).mean()
        tempdf.columns = {'dim1','dim2'}
        
        #append a new column to dataframe with trial information
        tempdf.insert(loc=2,
                      column='trial',
                      value = c*np.ones((tmpr.shape[0],1)) )
        if c==1:
            trials_proj = tempdf
        else:
            trials_proj = trials_proj.append(tempdf)
        
    return trials_proj

#%%
basepath = '/mnt/probox/buzsakilab.nyumc.org/datasets/GirardeauG/'
save_fig_path = '/mnt/probox/buzsakilab.nyumc.org/datasets/GirardeauG/gcPCA_plots/'

#%% load the structure
data_dict = mat73.loadmat(basepath+'hpc_bla_gg_dataset.mat')

#%%
safe_run    = []
safe_prun   = []
danger_run  = []
danger_prun = []
loadings    = []
right_ap    = []
left_ap     = []
ap_total    = []
subject     = []
n_cell      = []

#saving info
participation_pre_swr = []  # participation index in SWR pre run SWS state
participation_post_swr = [] # participation index in SWR post run SWS state
cellidx_gcpc1 = []  # boolean of indexes, true the cell crossed a threshold of loadings, false it didn't cross the threshold - gcpc1
cellidx_gcpc2 = []  # boolean of indexes, true the cell crossed a threshold of loadings, false it didn't cross the threshold - gcpc2
SWS_fr_pre    = []  # firing rate of the cell on SWS before running on the maze with airpuff
SWS_fr_post   = []  # firing rate of the cell on SWS after running on the maze with airpuff
SI_pre    = []  # spatial information the cell had on the pre run, before the maze with airpuff
SI_run    = []  # spatial information the cell had on the run, during the run with the airpuff
SI_post   = []  # spatial information the cell had on the post run, after the run in the maze with airpuff
cell_loadings_gcpc1 = [] #loadings of gcPC1
cell_loadings_gcpc2 = [] #loadings of gcPC2

for ses in np.arange(len(data_dict['hpc_bla_gg_dataset']['linspd'])):
    air_puff_times = data_dict['hpc_bla_gg_dataset']['air_puff_times'][ses]
    if len(air_puff_times)>10:  # if more than 10 trials had airpuff
        pos = data_dict['hpc_bla_gg_dataset']['location'][ses]
        linspd = data_dict['hpc_bla_gg_dataset']['linspd'][ses]
        pos_t = data_dict['hpc_bla_gg_dataset']['tracking_t'][ses]
        spikes = data_dict['hpc_bla_gg_dataset']['spikes_ts'][ses]
        region = data_dict['hpc_bla_gg_dataset']['spikes_region'][ses]
        if np.char.equal(region,'hpc').sum()>min_n_cell:
            
            pre_run_temp = data_dict['hpc_bla_gg_dataset']['task'][ses]['prerun_intervals']
            run_temp = data_dict['hpc_bla_gg_dataset']['task'][ses]['run_intervals']
            post_run_temp = data_dict['hpc_bla_gg_dataset']['task'][ses]['postrun_intervals']
            pre_sws_temp = data_dict['hpc_bla_gg_dataset']['task'][ses]['pre_sws_intervals']
            post_sws_temp = data_dict['hpc_bla_gg_dataset']['task'][ses]['post_sws_intervals']
            
            #getting riplpes start and stop points
            rip_start = data_dict['hpc_bla_gg_dataset']['ripple'][ses]['start']-50
            rip_stop = data_dict['hpc_bla_gg_dataset']['ripple'][ses]['stop']+50
            
            nap_pos = nap.Tsd(pos_t, d=pos[:,0],time_units="s")
            nap_spd = nap.Tsd(pos_t, d=linspd,time_units="s")
            nap_air_puff = nap.Ts(air_puff_times,time_units="ms")
            pre_run_intervals = nap.IntervalSet(start=pre_run_temp[0],end = pre_run_temp[1])
            run_intervals     = nap.IntervalSet(start=run_temp[0],end = run_temp[1])
            post_run_intervals = nap.IntervalSet(start=post_run_temp[0],end = post_run_temp[1])
            pre_sws_intervals = nap.IntervalSet(start=pre_sws_temp[0],end = pre_sws_temp[1])
            post_sws_intervals = nap.IntervalSet(start=post_sws_temp[0],end = post_sws_temp[1])
            
            #getting ripples and separating by sws intervals
            rip_intervals = nap.IntervalSet(start=rip_start,end=rip_stop,time_units="ms")
            pre_rip = pre_sws_intervals.intersect(rip_intervals)
            post_rip = post_sws_intervals.intersect(rip_intervals)
            
            #preparing spikes
            
            # this is to look at  BLA
            # cells_bla = np.argwhere([np.char.equal(region,'bla')])
            # cells_cea = np.argwhere([np.char.equal(region,'cea')])
            # cells_bmp = np.argwhere([np.char.equal(region,'bmp')])
            # cells_ladl = np.argwhere([np.char.equal(region,'ladl')])
            # cells_hpc = np.concatenate((cells_bla, cells_cea, cells_bmp, cells_ladl),axis=0)
            
            cells_hpc = np.argwhere([np.char.equal(region,'hpc')])
            if cells_hpc.size>0:
                spks_times = {}
                c=0
                for a in cells_hpc[:,1]:
                    c+=1
                    spks_times[c] = spikes[a]
    
                # passing to pynapple ts group
                spikes_times = nap.TsGroup(spks_times)
                
                #separating in pre/run
                temp_pr = spikes_times.restrict(pre_run_intervals)  # pre run
                temp_r  = spikes_times.restrict(run_intervals)      # run
                temp_psr = spikes_times.restrict(post_run_intervals)  # post run
                temp_presws = spikes_times.restrict(pre_sws_intervals)  # pre run SWS
                temp_postsws = spikes_times.restrict(post_sws_intervals)  # post run SWS
                temp_prerip = spikes_times.restrict(pre_rip)
                temp_postrip = spikes_times.restrict(post_rip)
                
                # picking cells based on firing rate
                # cells2keep = (temp_pr.rates.values>min_fr) * (temp_psr.rates.values>min_fr)
                cells2keep = (temp_r.rates.values>min_fr)
                if sum(cells2keep)>min_n_cell:
                    #normalizing and smoothing data - pre run periods
                    temp_data = zscore(temp_pr.count(bin_size_task).as_dataframe().rolling(window=wind_cov,
                            win_type='gaussian',center=True,min_periods=1, 
                            axis = 0).mean(std=std_conv).values)
                    prerun_time = np.array(temp_pr.count(bin_size_task).index)
                    prerun_data = temp_data[:,cells2keep].copy()
                    
                    #normalizing and smoothing data - run periods
                    temp_data = zscore(temp_r.count(bin_size_task).as_dataframe().rolling(window=wind_cov,
                            win_type='gaussian',center=True,min_periods=1, 
                            axis = 0).mean(std=std_conv).values)
                    run_time = np.array(temp_r.count(bin_size_task).index)
                    run_data = temp_data[:,cells2keep].copy()
                    
                    #normalizing and smoothing data - post run periods
                    temp_data = zscore(temp_psr.count(bin_size_task).as_dataframe().rolling(window=wind_cov,
                            win_type='gaussian',center=True,min_periods=1, 
                            axis = 0).mean(std=std_conv).values)
                    postrun_time = np.array(temp_psr.count(bin_size_task).index)
                    postrun_data = temp_data[:,cells2keep].copy()
                    #normalizing and smoothing data - sleep periods
                    temp_data = zscore(temp_presws.count(bin_size).as_dataframe().rolling(window=wind_cov,
                            win_type='gaussian',center=True,min_periods=1, 
                            axis = 0).mean(std=std_conv).values)
                    presws_time = np.array(temp_presws.count(bin_size).index)
                    presws_data = temp_data[:,cells2keep].copy()
                    
                    #normalizing and smoothing data - sleep periods
                    temp_data = zscore(temp_postsws.count(bin_size).as_dataframe().rolling(window=wind_cov,
                            win_type='gaussian',center=True,min_periods=1, 
                            axis = 0).mean(std=std_conv).values)
                    postsws_time = np.array(temp_postsws.count(bin_size).index)
                    postsws_data = temp_data[:,cells2keep].copy()
                    
                    #getting left and right trials intervals
                    pre_run_pos = nap_pos.restrict(pre_run_intervals).as_series()
                    pre_run_spd = nap_spd.restrict(pre_run_intervals).as_series()
                    run_pos     = nap_pos.restrict(run_intervals).as_series()
                    run_spd     = nap_spd.restrict(run_intervals).as_series()
                    post_run_pos     = nap_pos.restrict(post_run_intervals).as_series()
                    post_run_spd     = nap_spd.restrict(post_run_intervals).as_series()
                    
                    #for visualization
                    fig0, (tst) = plt.subplots(1, 2)
                    tst[0].plot(pos[:,1],pos[:,0])
                    tst[1].plot(run_pos.values)
                    plt.title('session: ' + str(ses))
                    
                    
                    #identifying trials of running left or right 
                    #pre run
                    temp = nap.Tsd(np.array(pre_run_pos.index),savgol_filter(np.array(pre_run_pos.values),300,3))
                    temp_spd = pre_run_spd
                    left_pr_int2,right_pr_int2 = extract_trials(temp,nap_spd)
                    left_pr_int = left_pr_int2.merge_close_intervals(threshold=1)
                    right_pr_int = right_pr_int2.merge_close_intervals(threshold=1)
                    
                    #post run
                    temp = nap.Tsd(np.array(post_run_pos.index),savgol_filter(np.array(post_run_pos.values),300,3))
                    temp_spd = pre_run_spd
                    left_psr_int2,right_psr_int2 = extract_trials(temp,nap_spd)
                    left_psr_int = left_psr_int2.merge_close_intervals(threshold=1)
                    right_psr_int = right_psr_int2.merge_close_intervals(threshold=1)
                    
                    #run
                    temp = nap.Tsd(np.array(run_pos.index),savgol_filter(np.array(run_pos.values),300,3))
                    # temp_spd = run_spd
                    left_runs_int2,right_runs_int2 = extract_trials(temp,nap_spd)
                    left_runs_int = left_runs_int2.merge_close_intervals(threshold=1)
                    right_runs_int = right_runs_int2.merge_close_intervals(threshold=1)
                    
                    #%% computing single cell features on the pre run and post run
                    
                    tc_left_run = nap.compute_1d_tuning_curves(spikes_times, nap_pos.restrict(left_runs_int), nb_bins=80)
                    tc_left_prerun = nap.compute_1d_tuning_curves(spikes_times, nap_pos.restrict(left_pr_int), nb_bins=80)
                    tc_left_postrun = nap.compute_1d_tuning_curves(spikes_times, nap_pos.restrict(left_psr_int), nb_bins=80)
                    
                    tc_right_run = nap.compute_1d_tuning_curves(spikes_times, nap_pos.restrict(right_runs_int), nb_bins=80)
                    tc_right_prerun = nap.compute_1d_tuning_curves(spikes_times, nap_pos.restrict(right_pr_int), nb_bins=80)
                    tc_right_postrun = nap.compute_1d_tuning_curves(spikes_times, nap_pos.restrict(right_pr_int), nb_bins=80)
                    # the input to mutual info is the output of 1d tuning curves
                    mi_left_prerun = nap.compute_1d_mutual_info(tc_left_prerun,  nap_pos.restrict(left_pr_int),bitssec=True)
                    mi_right_prerun = nap.compute_1d_mutual_info(tc_right_prerun,  nap_pos.restrict(right_pr_int),bitssec=True)
                    
                    mi_left_run = nap.compute_1d_mutual_info(tc_left_run,  nap_pos.restrict(left_runs_int),bitssec=False)
                    mi_right_run = nap.compute_1d_mutual_info(tc_right_run,  nap_pos.restrict(right_runs_int),bitssec=False)
                    
                    mi_left_postrun = nap.compute_1d_mutual_info(tc_left_postrun,  nap_pos.restrict(left_psr_int),bitssec=True)
                    mi_right_postrun = nap.compute_1d_mutual_info(tc_right_postrun,  nap_pos.restrict(right_psr_int),bitssec=True)
                    #bz_find 1d place fields is a better approach for here
                    
                    # calculating the population coupling
                    
                    # calculate ripple participation (pre and post)
                    bool_rip_pre = temp_prerip.count()>0
                    bool_rip_post = temp_postrip.count()>0
                    
                    pre_rip_participation = np.sum(bool_rip_pre.values,axis=0)/bool_rip_pre.shape[0]
                    post_rip_participation = np.sum(bool_rip_post.values,axis=0)/bool_rip_post.shape[0]
                    
                    participation_pre_swr.append(pre_rip_participation[cells2keep])
                    participation_post_swr.append(post_rip_participation[cells2keep])
                    
                    # calculating the ripple rate pre and post sleep
                    # rip_fr_change = (temp_postrip.rates.values - temp_prerip.rates.values) / temp_prerip.rates.values
                    # firing rate during SWS
                    SWS_fr_pre.append(temp_presws.rates.values[cells2keep])
                    SWS_fr_post.append(temp_postsws.rates.values[cells2keep])
                    
                    #spatial information bits per sec
                    aux_SI = np.nanmean(np.concatenate((mi_left_prerun.values,mi_right_prerun.values),axis=1),axis=1)
                    SI_pre.append(aux_SI[cells2keep])
                    aux_SI = np.nanmean(np.concatenate((mi_left_run.values,mi_right_run.values),axis=1),axis=1)
                    SI_run.append(aux_SI[cells2keep])
                    aux_SI = np.nanmean(np.concatenate((mi_left_postrun.values,mi_right_postrun.values),axis=1),axis=1)
                    SI_post.append(aux_SI[cells2keep])
                    
                    # grid = plt.GridSpec(2,1,hspace=0.2)
                    # plt.figure(figsize=(7,15))
                    # plt.subplot(grid[0,0])
                    # plt.stem(V[0,:].T)
                    # plt.xlabel('Neurons')
                    # plt.ylabel('Loadings')
                    # plt.title('PC1')
                    # plt.subplot(grid[1,0])
                    # plt.stem(gcpca_mdl.loadings_[:,0])
                    # plt.xlabel('Neurons')
                    # plt.ylabel('Loadings')
                    # plt.title('gcPC1')
                    #%% temp plots
                    # aux_diff = mi_right_run.values-mi_right_prerun.values # change of spatial information
                    
                    # difference in post-pre rip firing rates
                    # aux_diff = (temp_postrip.rates.values - temp_prerip.rates.values) / temp_prerip.rates.values
                    
                    # difference in ripple participation
                    # aux_diff = (post_rip_participation - pre_rip_participation)
                    
                    # difference in post-pre sws firing rates
                    # aux_diff = (temp_postsws.rates.values / temp_presws.rates.values)
                    
                    # calculate peak difference
                    # idx_post = np.argmax(tc_left_postrun.values,axis=0)
                    # idx_pre = np.argmax(tc_left_prerun.values,axis=0)
                    # aux_diff = idx_post - idx_pre
                    # val = np.max(tc_left_postrun.values,axis=0)
                    
                    
                    # aux_diff = aux_diff[cells2keep]
                    
                    # limity = np.nanmax(np.abs(aux_diff))+0.01
                    # gcpcs_n = 0
                    # limitx = np.nanmax(np.abs(gcpca_mdl.loadings_[:,gcpcs_n]))+0.01
                    # plt.scatter(np.abs(gcpca_mdl.loadings_[:,gcpcs_n]),aux_diff)
                    # # plt.xlim([-limitx, limitx])
                    # # plt.yscale('symlog')
                    # plt.ylim([-limity, limity])
                    #%% running gcPCA
                    gcpca_mdl = gcPCA(method='v4')
                    gcpca_mdl.fit(postsws_data,presws_data)
                    
                    #%% tresholding the gcPC1 and 2
                    threshold = gcpca_mdl.loadings_[:,0].mean() + 1*gcpca_mdl.loadings_[:,0].std()
                    cellidx_gcpc1.append(np.abs(gcpca_mdl.loadings_[:,0])>threshold)
                    
                    threshold = gcpca_mdl.loadings_[:,1].mean() + 1*gcpca_mdl.loadings_[:,1].std()
                    cellidx_gcpc2.append(np.abs(gcpca_mdl.loadings_[:,1])>threshold)
                    
                    cell_loadings_gcpc1.append(gcpca_mdl.loadings_[:,0])
                    cell_loadings_gcpc2.append(gcpca_mdl.loadings_[:,1])
                    
                    #%% running PCA
                    _,_,V = np.linalg.svd(postsws_data,full_matrices=False)
                    
                    #%%
                    #identifying which run is safe or dangerous
                    n_r = len(nap_air_puff.restrict(right_runs_int))
                    n_l = len(nap_air_puff.restrict(left_runs_int))
                    if n_l>n_r:
                        runs_type={'left':'danger','right':'safe'}
                    else:
                        runs_type={'left':'safe','right':'danger'}
                    
                    #projection variance higher in safe or dangerous trial
                    prerun_gcpca = nap.TsdFrame(prerun_time,d=prerun_data.dot(gcpca_mdl.loadings_[:,:2]))
                    run_gcpca = nap.TsdFrame(run_time,d=run_data.dot(gcpca_mdl.loadings_[:,:2]))
                    
                    run_pca = nap.TsdFrame(run_time,d=run_data.dot(V[:2,:].T))
                    
                    run_left_gcpca = run_gcpca.restrict(left_runs_int).as_dataframe().var().values
                    run_right_gcpca = run_gcpca.restrict(right_runs_int).as_dataframe().var().values
                    
                    """compare with prerun variance on the same vector"""
                    loadings.append(gcpca_mdl.loadings_[:,:2])
                    if runs_type['left']=='safe':
                        safe_run.append(run_left_gcpca)
                        # safe_prun.append(prun_left_gcpca)
                        danger_run.append(run_right_gcpca)
                        # danger_prun.append(prun_right_gcpca)
                        color_l = 'k'
                        color_r = 'k'
                        color_ap_edge_r = ['firebrick','firebrick']
                        color_ap_edge_l =  ['white','firebrick']
                        color_r_dots = ['whitesmoke','dimgray']
                        color_l_dots =['whitesmoke','dimgray']
                        
                    else:
                        safe_run.append(run_right_gcpca)
                        # safe_prun.append(prun_right_gcpca)
                        danger_run.append(run_left_gcpca)
                        # danger_prun.append(prun_left_gcpca)
                        color_l = 'k'
                        color_r = 'k'
                        color_ap_edge_l = ['firebrick','firebrick']
                        color_ap_edge_r = ['white','firebrick']
                        color_l_dots =  ['whitesmoke','dimgray']
                        color_r_dots = ['whitesmoke','dimgray']
                    
                    right_ap.append(n_r)
                    left_ap.append(n_l)
                    ap_total.append(len(nap_air_puff))
                    subject.append(data_dict['hpc_bla_gg_dataset']['session_folder_name'][ses])
                    n_cell.append(sum(cells2keep))
                    #%% find the location of airpuff and get the vector variance 
                    #after the animal crosses that
                    temp_ap = nap.IntervalSet(start=nap_air_puff.index-0.05,end = nap_air_puff.index+0.05)
                    air_puff_loc = nap_pos.restrict(temp_ap).as_series().median()
                
                    #%% save projections and information from session
                    
                    #%% make plot
                    """PICK TRIALS WITH SPECIFIC RUNNING SPEED AND THAT THE ANIMAL DOESNT STOP IN THE MIDDLE"""
                    # fig, ax = plt.subplots()
                    mrkr_size=30
                    dim1 = np.interp(run_pos.index,run_gcpca.index,run_gcpca.values[:,0])
                    dim2 = np.interp(run_pos.index,run_gcpca.index,run_gcpca.values[:,1])
                    newd = np.concatenate((dim1[:,np.newaxis],dim2[:,np.newaxis]),axis=1)
                    new_r_gcpca = nap.TsdFrame(np.array(run_pos.index), d = newd)
                    
                    roll_wind = 20
                    run_gcpca = new_r_gcpca.as_dataframe().rolling(roll_wind).mean()
                    
                    ### plotting projection on maze
                    fig5, (tst) = plt.subplots(1, 2)
                    tst[0].plot(run_pos.values,run_gcpca.values[:,0],lw=0.25,zorder=0,c='k')
                    tst[0].scatter(air_puff_loc,0,c=['seagreen'],zorder=1)
                    plt.title('gcPCA')
                    tst[1].plot(run_pos.values,run_gcpca.values[:,1],lw=0.25,zorder=0,c='k')
                    tst[1].scatter(air_puff_loc,0,c=['seagreen'],zorder=1)
                    ###
                    
                    
                    #preparing gcpca prerun
                    dim1 = np.interp(pre_run_pos.index,prerun_gcpca.index,prerun_gcpca.values[:,0])
                    dim2 = np.interp(pre_run_pos.index,prerun_gcpca.index,prerun_gcpca.values[:,1])
                    newd = np.concatenate((dim1[:,np.newaxis],dim2[:,np.newaxis]),axis=1)
                    new_pr_gcpca = nap.TsdFrame(np.array(pre_run_pos.index), d = newd)
                    prerun_gcpca = new_pr_gcpca.as_dataframe().rolling(roll_wind).mean()
                    
                    #preparing for pca
                    dim1 = np.interp(run_pos.index,run_pca.index,run_pca.values[:,0])
                    dim2 = np.interp(run_pos.index,run_pca.index,run_pca.values[:,1])
                    newd = np.concatenate((dim1[:,np.newaxis],dim2[:,np.newaxis]),axis=1)
                    new_r_pca = nap.TsdFrame(np.array(run_pos.index), d = newd)
                    run_pca = new_r_pca.as_dataframe().rolling(roll_wind).mean()
                    
                    
                    ### plotting projection on maze
                    fig6, (tsx) = plt.subplots(1, 2)
                    tsx[0].plot(run_pos.values,run_pca.values[:,0],lw=0.25,zorder=0,c='m')
                    tsx[0].scatter(air_puff_loc,0,c=['seagreen'],zorder=1)
                    plt.title('PCA')
                    tsx[1].plot(run_pos.values,run_pca.values[:,1],lw=0.25,zorder=0,c='m')
                    tsx[1].scatter(air_puff_loc,0,c=['seagreen'],zorder=1)
                    ###
                    
                    #plotting speed
                    plt.figure()
                    all_int = left_runs_int.append(right_runs_int)
                    spd_dat = nap_spd.restrict(all_int).values
                    spd_t = nap_spd.restrict(all_int).index
                    plt.scatter(spd_t,spd_dat,s=10,c='k')
                    fig2, (acs) = plt.subplots(1, 2)
                    fig, (axs) = plt.subplots(2, 2,sharex=True,sharey=True)
                    for a in left_runs_int.values[1:20,:]:
                        c+=1
                        temp_is = nap.IntervalSet(a[0],end=a[1])
                        # tmpr = run_gcpca.restrict(temp_is)
                        # tempdf = tmpr.as_dataframe()
                        tempdf = new_r_gcpca.restrict(temp_is).as_dataframe().rolling(roll_wind).mean()

                        x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,0])),0]
                        y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,1])),1]

                        axs[0,0].plot(x,y,color_l,linewidth=0.5,zorder=0)
                        
                        axs[0,0].scatter(x[0],y[0],s=mrkr_size,c=color_l_dots[0],zorder=5,alpha=0.9,edgecolors='k')
                        axs[0,0].scatter(x[-1],y[-1],s=mrkr_size,c=color_l_dots[1],zorder=10,alpha=0.9,edgecolors='k')
                        #finding location index to plot
                        temp_pos = nap_pos.restrict(temp_is)
                        I = np.argmin(np.abs(temp_pos-air_puff_loc))
                        if I<len(x):  # location of air puff was getting where gcPCA is nan
                            axs[0,0].scatter(x[I],y[I],s=mrkr_size,c= color_ap_edge_l[0],edgecolors= color_ap_edge_l[1],zorder=15,alpha=0.9)
                        
                        #plot the trial trajectory to know if you are picking the correct one
                        acs[0].plot(temp_pos.to_numpy())
                        
                        """PLOT SPEED HERE SO WE KNOW IF THE ANIMAL IS SLOWING DOWN
                        
                        ALSO JUST DO PRE VS POST SLEEP AND SEE IF THE BIGGEST CHANGE IS IN THE MAZE
                        
                        NEW STRUCTURE WHERE THE PRE/POST SLEEP IS INCLUDED"""
                        
                    
                    for a in right_runs_int.values[1:20,:]:
                        c+=1
                        temp_is = nap.IntervalSet(a[0],end=a[1])
                        # tmpr = run_gcpca.restrict(temp_is)
                        # tempdf = tmpr.as_dataframe()
                        tempdf = new_r_gcpca.restrict(temp_is).as_dataframe().rolling(roll_wind).mean()
                        x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,0])),0]
                        y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,1])),1]
                        axs[1,0].plot(x,y,color_r,linewidth=0.5,zorder=0)
                        axs[1,0].scatter(x[0],y[0],s=mrkr_size,c=color_r_dots[0],zorder=5,alpha=0.9,edgecolors='k')
                        axs[1,0].scatter(x[-1],y[-1],s=mrkr_size,c=color_r_dots[1],zorder=10,alpha=0.9,edgecolors='k')
                        #finding location index to plot
                        temp_pos = nap_pos.restrict(temp_is)
                        I = np.argmin(np.abs(temp_pos-air_puff_loc))
                        if I<len(x):  # location of air puff was getting where gcPCA is nan
                            axs[1,0].scatter(x[I],y[I],s=mrkr_size,c= color_ap_edge_r[0],edgecolors= color_ap_edge_r[1],zorder=15,alpha=0.9)
                        
                        acs[1].plot(temp_pos.to_numpy())
                        
                        
                    axs[0,0].set_title('run gcPCA space')
                    axs[0,0].set_ylabel('gcPC2')
                    axs[0,0].set_xlabel('gcPC1')
                    axs[1,0].set_ylabel('gcPC2')
                    axs[1,0].set_xlabel('gcPC1')
                    # axs[1,0].set_title('gcPCA space')
                    fig.suptitle("location of air puff:"+air_puff_loc.astype(str))
                    
                    acs[0].set_title('x pos trial')
                    
                    #% making plot pca
                    
                    # newd = np.concatenate((dim1[:,np.newaxis],dim2[:,np.newaxis]),axis=1)
                    plt.figure()
                    for a in left_runs_int.values[1:20,:]:
                        c+=1
                        temp_is = nap.IntervalSet(a[0],end=a[1])
                        # tmpr = run_pca.restrict(temp_is)
                        # tempdf = tmpr.as_dataframe()
                        tempdf = new_r_pca.restrict(temp_is).as_dataframe().rolling(roll_wind).mean()
                        x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,0])),0]
                        y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,1])),1]
                        axs[0,1].plot(x,y,color_l,linewidth=0.5,zorder=0)
                        
                        axs[0,1].scatter(x[0],y[0],s=mrkr_size,c=color_l_dots[0],zorder=5,alpha=0.9,edgecolors='k')
                        axs[0,1].scatter(x[-1],y[-1],s=mrkr_size,c=color_l_dots[1],zorder=10,alpha=0.9,edgecolors='k')
                        
                        #finding location index to plot
                        temp_pos = nap_pos.restrict(temp_is)
                        I = np.argmin(np.abs(temp_pos-air_puff_loc))
                        if I<len(x):  # location of air puff was getting where gcPCA is nan
                            axs[0,1].scatter(x[I],y[I],s=mrkr_size,c= color_ap_edge_l[0],edgecolors= color_ap_edge_l[1],zorder=15,alpha=0.9)
                    
                    for a in right_runs_int.values[1:20,:]:
                        c+=1
                        temp_is = nap.IntervalSet(a[0],end=a[1])
                        # tmpr = run_pca.restrict(temp_is)
                        # tempdf = tmpr.as_dataframe()
                        tempdf = new_r_pca.restrict(temp_is).as_dataframe().rolling(roll_wind).mean()
                        x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,0])),0]
                        y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,1])),1]
                        axs[1,1].plot(x,y,color_r,linewidth=0.5,zorder=0)
                        axs[1,1].scatter(x[0],y[0],s=mrkr_size,c=color_r_dots[0],zorder=5,alpha=0.9,edgecolors='k')
                        axs[1,1].scatter(x[-1],y[-1],s=mrkr_size,c=color_r_dots[1],zorder=10,alpha=0.9,edgecolors='k')
                        #finding location index to plot
                        temp_pos = nap_pos.restrict(temp_is)
                        I = np.argmin(np.abs(temp_pos-air_puff_loc))
                        if I<len(x):  # location of air puff was getting where gcPCA is nan
                            axs[1,1].scatter(x[I],y[I],s=mrkr_size,c= color_ap_edge_r[0],edgecolors= color_ap_edge_r[1],zorder=15,alpha=0.9)
                    # axs[0,1].set_title('PCA space')
                    axs[0,1].set_title('PCA space')
                    axs[0,1].set_ylabel('PC2')
                    axs[0,1].set_xlabel('PC1')
                    axs[1,1].set_ylabel('PC2')
                    axs[1,1].set_xlabel('PC1')
                    
                    fig.savefig(save_fig_path+ses.astype(str)+"gcPCA_space_PCA_space.pdf", transparent=True)
                    


#%% preparing boxplots of different metrics

gcpc1_cells = np.hstack(cellidx_gcpc1)
ripple_pre_prob = np.hstack(participation_pre_swr)
ripple_post_prob = np.hstack(participation_post_swr)

spatial_info_pre = np.hstack(SI_pre)
spatial_info_run = np.hstack(SI_run)
spatial_info_post = np.hstack(SI_post)

SWS_FR_pre = np.hstack(SWS_fr_pre)
SWS_FR_post = np.hstack(SWS_fr_post)

cell_load_gcpc1 = np.hstack(cell_loadings_gcpc1)
cell_load_gcpc2 = np.hstack(cell_loadings_gcpc2)

df_gcpc1_cells = pd.DataFrame({
    "ripple_pre_participation" : ripple_pre_prob,
    "ripple_post_participation" : ripple_post_prob,
    "delta_ripple_participation" : ripple_post_prob-ripple_pre_prob,
    "spatial_info_pre" : spatial_info_pre,
    "spatial_info_run" : spatial_info_run,
    "spatial_info_gain" : spatial_info_run/spatial_info_pre,
    "spatial_info_post" : spatial_info_post,
    "SWS_firing_rate_pre" : np.log(SWS_FR_pre),
    "SWS_firing_rate_post" : np.log(SWS_FR_post),
    "cell_loadings_gcpc1" : np.abs(cell_load_gcpc1),
    "cell_loadings_gcpc2" : np.abs(cell_load_gcpc2),
    "label" : gcpc1_cells
})

# sns.swarmplot(data=df_gcpc1_cells, x = 'label', y='ripple_pre_participation')
# sns.swarmplot(data=df_gcpc1_cells, x = 'label', y='ripple_post_participation')
# sns.swarmplot(data=df_gcpc1_cells, x = 'label', y='spatial_info_pre')
# sns.swarmplot(data=df_gcpc1_cells, x = 'label', y='spatial_info_run')
# sns.swarmplot(data=df_gcpc1_cells, x = 'label', y='spatial_info_post')
# sns.swarmplot(data=df_gcpc1_cells, x = 'label', y='SWS_firing_rate_pre')
# sns.swarmplot(data=df_gcpc1_cells, x = 'label', y='SWS_firing_rate_post')

sns.regplot(data=df_gcpc1_cells, x = 'cell_loadings_gcpc1', y='ripple_pre_participation',
            marker='x',color='.3',line_kws=dict(color='r'))
sns.despine()

ax = sns.regplot(data=df_gcpc1_cells, x = 'cell_loadings_gcpc1', y='ripple_post_participation',
            marker='x',color='.3',line_kws=dict(color='r'))
ax.set(xlabel = 'Cell Loadings on gcPC1',ylabel = 'Probability of participation in ripple',title='post-run ripples')
sns.despine()


ax = sns.regplot(data=df_gcpc1_cells, x = 'cell_loadings_gcpc1', y='delta_ripple_participation',
            marker='x',color='.3',line_kws=dict(color='r'))
ax.set(xlabel = 'Cell Loadings on gcPC1',ylabel = 'Delta ripple participation',title='post-pre')
sns.despine()


ax = sns.regplot(data=df_gcpc1_cells, x = 'cell_loadings_gcpc1', y='spatial_info_run',
            marker='x',color='.3',line_kws=dict(color='r'))
sns.despine()
ax.set(xlabel = 'Cell Loadings on gcPC1',ylabel = 'Spatial Information (bits/spikes)')

ax = sns.regplot(data=df_gcpc1_cells, x = 'cell_loadings_gcpc1', y='spatial_info_gain',
            marker='x',color='.3',line_kws=dict(color='r'))
sns.despine()
ax.set(xlabel = 'Cell Loadings on gcPC1',ylabel = 'Spatial Information gain', title='run / pre')


# sns.regplot(data=df_gcpc1_cells, x = 'cell_loadings_gcpc1', y='spatial_info_post',
#             marker='x',color='.3',line_kws=dict(color='r'))
# sns.despine()

sns.scatterplot(data=df_gcpc1_cells, x = 'cell_loadings_gcpc1', y='spatial_info_run')
sns.scatterplot(data=df_gcpc1_cells, x = 'cell_loadings_gcpc1', y='spatial_info_post')
# sns.scatterplot(data=df_gcpc1_cells, x = 'cell_loadings_gcpc1', y='SWS_firing_rate_pre')

ax = sns.regplot(data=df_gcpc1_cells, x = 'cell_loadings_gcpc1', y='SWS_firing_rate_post',
            marker='x',color='.3',line_kws=dict(color='r'))
ax.set(xlabel = 'Cell Loadings on gcPC1',ylabel = 'SWS firing rate (log scale)',title='post-run SWS')
sns.despine()


 
 