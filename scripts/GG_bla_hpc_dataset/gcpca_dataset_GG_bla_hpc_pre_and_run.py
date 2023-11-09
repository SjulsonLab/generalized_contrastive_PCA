#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 12:36:53 2023

@author: eliezyer

script to do gcPCA analysis on HPC-BLA dataset
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
from scipy.signal import savgol_filter
import mat73 #to load matlab v7.3 files
import pickle
from numpy import linalg as LA

#%% parameters
min_n_cell = 15 #min number of cells in the brain area to be used
min_fr = 0.01 #minimum firing rate to be included in the analysis
bin_size = 0.01
bin_size_task = 0.05;
plt.rcParams['figure.dpi'] = 300
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
std_conv = 2
wind_cov = 5
#%% import custom modules
#repo_dir = "/gs/gsfs0/users/edeolive/github/normalized_contrastive_PCA/" #repository dir
repo_dir = "/home/eliezyer/Documents/github/normalized_contrastive_PCA/"
sys.path.append(repo_dir)
from contrastive_methods import gcPCA

#%% defining functions
def extract_trials(temp,temp_spd):
    """Function to extract trials from GG dataset where animals
    run in a linear track with airpuff
    
    todo: [ ] really improve the trial detection 
        [ ] add end of trial to reach the end of the maze 
    instead of time constrain
    [ ] picking trials with certain speed
    """
    
    max_len = 5
    min_len = 1
    logical_left  = np.diff(temp.clip(lower=150,upper=450).values)<0
    logical_right = np.diff(temp.clip(lower=150,upper=450).values)>0
    # logical_left  = np.diff(temp.values)<-0.3
    # logical_right = np.diff(temp.values)>0.3
    logical_left  = np.append(logical_left,False)
    logical_right = np.append(logical_right,False)
    logical_left  = np.insert(logical_left,0,False)
    logical_right = np.insert(logical_right,0,False)
    ts = temp.clip(lower=150,upper=450).index
    # ts = temp.index
    
    #finding start and stop of left runs (here we lose an index, correct later)
    temp_st = np.argwhere(np.diff(logical_left.astype(int))==1)+1
    temp_sp = np.argwhere(np.diff(logical_left.astype(int))==-1)
    
    #picking only the intervals that lasted > 1 s and <5s
    start1 = ts[temp_st]
    stop1  = ts[temp_sp]
    int2keep = ((stop1 - start1)>min_len) * ((stop1 - start1)<max_len)
    start2 = start1[int2keep].copy()
    stop2  = stop1[int2keep].copy()
    trials2keep = []
    for a in np.arange(len(start2)):
        interval = nap.IntervalSet(start=start2[a],end=stop2[a])
        if (temp.restrict(interval).max() - temp.restrict(interval).min())>200:
            trials2keep.append(True)
        else:
            trials2keep.append(False)
        # if temp_spd.restrict(interval).min()>20:
        #     trials2keep.append(True)
        # else:
        #     trials2keep.append(False)
    
    # """make detection of a trial from location 150 to the location 450
    # check the speed within the trial"""
    
    left_runs_interval = nap.IntervalSet(start = start2[trials2keep],end = stop2[trials2keep])
    # left_runs_interval = nap.IntervalSet(start = start2,end = stop2)
    
    
    # for a in left_runs_interval.values:
        # interval = nap.IntervalSet(start=a[0],end=a[1])
        # np.isin(temp_spd.index,a)
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
        if (temp.restrict(interval).max() - temp.restrict(interval).min())>200:
            trials2keep.append(True)
        else:
            trials2keep.append(False)
    # trials2keep = []
    # for a in np.arange(len(start2)):
    #     interval = nap.IntervalSet(start=start2[a],end=stop2[a])
    #     if temp_spd.restrict(interval).min()>20:
    #         trials2keep.append(True)
    #     else:
    #         trials2keep.append(False)
    right_runs_interval = nap.IntervalSet(start = start2[trials2keep],end = stop2[trials2keep])
    # right_runs_interval = nap.IntervalSet(start = start2,end = stop2)
    
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

def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))
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

for ses in np.arange(len(data_dict['hpc_bla_gg_dataset']['linspd'])):
    air_puff_times = data_dict['hpc_bla_gg_dataset']['air_puff_times'][ses]
    if len(air_puff_times)>10:
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
                temp_pr = spikes_times.restrict(pre_run_intervals)
                temp_r  = spikes_times.restrict(run_intervals)
                temp_psr = spikes_times.restrict(post_run_intervals)
                temp_presws = spikes_times.restrict(pre_sws_intervals)
                temp_postsws = spikes_times.restrict(post_sws_intervals)
                # temp_presws = spikes_times.restrict(pre_rip)
                # temp_postsws = spikes_times.restrict(post_rip)
                
                cells2keep = (temp_pr.rates.values>min_fr) * (temp_r.rates.values>min_fr);
                if sum(cells2keep)>min_n_cell:
                    #normalizing and smoothing data - pre run periods
                    temp_data = zscore(temp_pr.count(bin_size_task).rolling(window=wind_cov,
                            win_type='gaussian',center=True,min_periods=1, 
                            axis = 0).mean(std=std_conv).values)
                    prerun_time = np.array(temp_pr.count(bin_size_task).index)
                    prerun_data = temp_data[:,cells2keep].copy()
                    
                    #normalizing and smoothing data - run periods
                    temp_data = zscore(temp_r.count(bin_size_task).rolling(window=wind_cov,
                            win_type='gaussian',center=True,min_periods=1, 
                            axis = 0).mean(std=std_conv).values)
                    run_time = np.array(temp_r.count(bin_size_task).index)
                    run_data = temp_data[:,cells2keep].copy()
                    
                    #normalizing and smoothing data - post run periods
                    temp_data = zscore(temp_psr.count(bin_size_task).rolling(window=wind_cov,
                            win_type='gaussian',center=True,min_periods=1, 
                            axis = 0).mean(std=std_conv).values)
                    postrun_time = np.array(temp_psr.count(bin_size_task).index)
                    postrun_data = temp_data[:,cells2keep].copy()
                    #normalizing and smoothing data - sleep periods
                    temp_data = zscore(temp_presws.count(bin_size).rolling(window=wind_cov,
                            win_type='gaussian',center=True,min_periods=1, 
                            axis = 0).mean(std=std_conv).values)
                    presws_time = np.array(temp_presws.count(bin_size).index)
                    presws_data = temp_data[:,cells2keep].copy()
                    
                    #normalizing and smoothing data - sleep periods
                    temp_data = zscore(temp_postsws.count(bin_size).rolling(window=wind_cov,
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
                    
                    
                    #identifying trials of running left or right 
                    #pre run
                    temp = nap.Tsd(np.array(pre_run_pos.index),savgol_filter(np.array(pre_run_pos.values),300,3))
                    # temp_spd = pre_run_spd
                    left_pr_int2,right_pr_int2 = extract_trials(temp,nap_spd)
                    left_pr_int = left_pr_int2.merge_close_intervals(threshold=1)
                    right_pr_int = right_pr_int2.merge_close_intervals(threshold=1)
                    
                    #post run
                    temp = nap.Tsd(np.array(post_run_pos.index),savgol_filter(np.array(post_run_pos.values),300,3))
                    # temp_spd = pre_run_spd
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
                    
                    # tc_left_run = nap.compute_1d_tuning_curves(spikes_times, nap_pos.restrict(left_runs_int), nb_bins=80)
                    # tc_left_prerun = nap.compute_1d_tuning_curves(spikes_times, nap_pos.restrict(left_pr_int), nb_bins=80)
                    # tc_right_run = nap.compute_1d_tuning_curves(spikes_times, nap_pos.restrict(right_runs_int), nb_bins=80)
                    # tc_right_prerun = nap.compute_1d_tuning_curves(spikes_times, nap_pos.restrict(right_pr_int), nb_bins=80)
                    # #the input to mutual info is the output of 1d tuning curves
                    # mi_left_run = nap.compute_1d_mutual_info(tc_left_run,  nap_pos.restrict(left_runs_int))
                    
                    #bz_find 1d place fields is a better approach for here
                    
                    #%% running gcPCA
                    all_int = left_runs_int.append(right_runs_int)
                    preall_int = left_pr_int.append(right_pr_int)
                    idxtouse = np.isin(run_time,temp_r.count(bin_size).restrict(all_int).index)
                    
                    preidxtouse = np.isin(prerun_time,temp_pr.count(bin_size).restrict(preall_int).index)
                    gcpca_mdl = gcPCA(method='v4.1')
                    # gcpca_mdl.fit(run_data[idxtouse,:],prerun_data[preidxtouse,:])
                    # gcpca_mdl.fit(run_data[:prerun_data.shape[0],:],prerun_data)
                    # gcpca_mdl.fit(run_data,prerun_data)
                    gcpca_mdl.fit(postsws_data,presws_data)
                    
                    #%% running PCA
                    _,_,V = np.linalg.svd(postsws_data,full_matrices=False)
                    
                    
                    #%%
                    
                    #identifying which run is safe or dangerous
                    n_r = nap_air_puff.restrict(right_runs_int).shape[0]
                    n_l = nap_air_puff.restrict(left_runs_int).shape[0]
                    if n_l>n_r:
                        runs_type={'left':'danger','right':'safe'}
                    else:
                        runs_type={'left':'safe','right':'danger'}
                    
                    #projection variance higher in safe or dangerous trial
                    prerun_gcpca = nap.TsdFrame(prerun_time,d=prerun_data.dot(gcpca_mdl.loadings_[:,:2]))
                    run_gcpca = nap.TsdFrame(run_time,d=run_data.dot(gcpca_mdl.loadings_[:,:2]))
                    
                    run_pca = nap.TsdFrame(run_time,d=run_data.dot(V[:2,:].T))
                    
                    run_left_gcpca = run_gcpca.restrict(left_runs_int).var().values
                    run_right_gcpca = run_gcpca.restrict(right_runs_int).var().values
                    
                    prun_left_gcpca = prerun_gcpca.restrict(left_pr_int).var().values
                    prun_right_gcpca = prerun_gcpca.restrict(right_pr_int).var().values
                    
                    """compare with prerun variance on the same vector"""
                    loadings.append(gcpca_mdl.loadings_[:,:2])
                    if runs_type['left']=='safe':
                        safe_run.append(run_left_gcpca)
                        safe_prun.append(prun_left_gcpca)
                        danger_run.append(run_right_gcpca)
                        danger_prun.append(prun_right_gcpca)
                        # color_l = 'b'
                        # color_r = 'r'
                        color_l = 'k'
                        color_r = 'k'
                        color_ap_edge_r = ['firebrick','firebrick']
                        color_ap_edge_l =  ['white','firebrick']
                        color_r_dots = ['whitesmoke','dimgray']
                        color_l_dots =['whitesmoke','dimgray']
                        
                    else:
                        safe_run.append(run_right_gcpca)
                        safe_prun.append(prun_right_gcpca)
                        danger_run.append(run_left_gcpca)
                        danger_prun.append(prun_left_gcpca)
                        # color_l = 'r'
                        # color_r = 'b'
                        color_l = 'k'
                        color_r = 'k'
                        color_ap_edge_l = ['firebrick','firebrick']
                        color_ap_edge_r = ['white','firebrick']
                        color_l_dots =  ['whitesmoke','dimgray']
                        color_r_dots = ['whitesmoke','dimgray']
                    
                    right_ap.append(n_r)
                    left_ap.append(n_l)
                    ap_total.append(nap_air_puff.values.shape[0])
                    subject.append(data_dict['hpc_bla_gg_dataset']['session_folder_name'][ses])
                    n_cell.append(sum(cells2keep))
                    #%% find the location of airpuff and get the vector variance 
                    #after the animal crosses that
                    temp_ap = nap.IntervalSet(start=nap_air_puff.index-0.05,end = nap_air_puff.index+0.05)
                    air_puff_loc = nap_pos.restrict(temp_ap).median()
                
                
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
                    run_gcpca = new_r_gcpca.rolling(roll_wind).mean()
                    
                    ### plotting projection on maze
                    fig5, (tst) = plt.subplots(1, 2)
                    tst[0].plot(run_pos.values,run_gcpca.values[:,0],lw=0.25,zorder=0,c='k')
                    tst[0].scatter(air_puff_loc,0,c=['seagreen'],zorder=1)
                    tst[1].plot(run_pos.values,run_gcpca.values[:,1],lw=0.25,zorder=0,c='k')
                    tst[1].scatter(air_puff_loc,0,c=['seagreen'],zorder=1)
                    ###
                    
                    
                    #preparing gcpca prerun
                    dim1 = np.interp(pre_run_pos.index,prerun_gcpca.index,prerun_gcpca.values[:,0])
                    dim2 = np.interp(pre_run_pos.index,prerun_gcpca.index,prerun_gcpca.values[:,1])
                    newd = np.concatenate((dim1[:,np.newaxis],dim2[:,np.newaxis]),axis=1)
                    new_pr_gcpca = nap.TsdFrame(np.array(pre_run_pos.index), d = newd)
                    prerun_gcpca = new_pr_gcpca.rolling(roll_wind).mean()
                    
                    #preparing for pca
                    dim1 = np.interp(run_pos.index,run_pca.index,run_pca.values[:,0])
                    dim2 = np.interp(run_pos.index,run_pca.index,run_pca.values[:,1])
                    newd = np.concatenate((dim1[:,np.newaxis],dim2[:,np.newaxis]),axis=1)
                    new_r_pca = nap.TsdFrame(np.array(run_pos.index), d = newd)
                    run_pca = new_r_pca.rolling(roll_wind).mean()
                    
                    
                    ### plotting projection on maze
                    fig6, (tsx) = plt.subplots(1, 2)
                    tsx[0].plot(run_pos.values,run_pca.values[:,0],lw=0.25,zorder=0,c='m')
                    tsx[0].scatter(air_puff_loc,0,c=['seagreen'],zorder=1)
                    tsx[1].plot(run_pos.values,run_pca.values[:,1],lw=0.25,zorder=0,c='m')
                    tsx[1].scatter(air_puff_loc,0,c=['seagreen'],zorder=1)
                    ###
                    
                    #plotting
                    plt.figure()
                    spd_dat = nap_spd.restrict(all_int).values
                    spd_t = nap_spd.restrict(all_int).index
                    plt.scatter(spd_t,spd_dat,s=10,c='k')
                    fig2, (acs) = plt.subplots(1, 2)
                    fig, (axs) = plt.subplots(2, 2,sharex=True,sharey=True)
                    for a in left_runs_int.values[1:20,:]:
                        c+=1
                        temp_is = nap.IntervalSet(a[0],end=a[1])
                        tmpr = run_gcpca.restrict(temp_is)
                        tempdf = tmpr.as_dataframe()
                        # tempdf = tempdf.rolling(30).mean()
                        x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,0])),0]
                        # x = smooth(np.array(x),30)
                        y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,1])),1]
                        # x = smooth(np.array(y),30)
                        axs[0,0].plot(x,y,color_l,linewidth=0.5,zorder=0)
                        
                        axs[0,0].scatter(x[0],y[0],s=mrkr_size,c=color_l_dots[0],zorder=5,alpha=0.9,edgecolors='k')
                        axs[0,0].scatter(x[-1],y[-1],s=mrkr_size,c=color_l_dots[1],zorder=10,alpha=0.9,edgecolors='k')
                        #finding location index to plot
                        temp_pos = nap_pos.restrict(temp_is)
                        I = np.argmin(np.abs(temp_pos-air_puff_loc))
                        axs[0,0].scatter(x[I],y[I],s=mrkr_size,c= color_ap_edge_l[0],edgecolors= color_ap_edge_l[1],zorder=15,alpha=0.9)
                        
                        #plot the trial trajectory to know if you are picking the correct one
                        acs[0].plot(temp_pos.to_numpy())
                        
                        """PLOT SPEED HERE SO WE KNOW IF THE ANIMAL IS SLOWING DOWN
                        
                        ALSO JUST DO PRE VS POST SLEEP AND SEE IF THE BIGGEST CHANGE IS IN THE MAZE
                        
                        NEW STRUCTURE WHERE THE PRE/POST SLEEP IS INCLUDED"""
                        
                    
                    for a in right_runs_int.values[1:20,:]:
                        c+=1
                        temp_is = nap.IntervalSet(a[0],end=a[1])
                        tmpr = run_gcpca.restrict(temp_is)
                        tempdf = tmpr.as_dataframe()
                        # tempdf = tempdf.rolling(30).mean()
                        x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,0])),0]
                        # x = smooth(np.array(x),30)
                        y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,1])),1]
                        # x = smooth(np.array(y),30)
                        axs[1,0].plot(x,y,color_r,linewidth=0.5,zorder=0)
                        axs[1,0].scatter(x[0],y[0],s=mrkr_size,c=color_r_dots[0],zorder=5,alpha=0.9,edgecolors='k')
                        axs[1,0].scatter(x[-1],y[-1],s=mrkr_size,c=color_r_dots[1],zorder=10,alpha=0.9,edgecolors='k')
                        #finding location index to plot
                        temp_pos = nap_pos.restrict(temp_is)
                        I = np.argmin(np.abs(temp_pos-air_puff_loc))
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
                        tmpr = run_pca.restrict(temp_is)
                        tempdf = tmpr.as_dataframe()
                        # tempdf = tempdf.rolling(30).mean()
                        x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,0])),0]
                        # x = smooth(np.array(x),30)
                        y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,1])),1]
                        # x = smooth(np.array(y),30)
                        axs[0,1].plot(x,y,color_l,linewidth=0.5,zorder=0)
                        
                        axs[0,1].scatter(x[0],y[0],s=mrkr_size,c=color_l_dots[0],zorder=5,alpha=0.9,edgecolors='k')
                        axs[0,1].scatter(x[-1],y[-1],s=mrkr_size,c=color_l_dots[1],zorder=10,alpha=0.9,edgecolors='k')
                        
                        #finding location index to plot
                        temp_pos = nap_pos.restrict(temp_is)
                        I = np.argmin(np.abs(temp_pos-air_puff_loc))
                        axs[0,1].scatter(x[I],y[I],s=mrkr_size,c= color_ap_edge_l[0],edgecolors= color_ap_edge_l[1],zorder=15,alpha=0.9)
                    
                    for a in right_runs_int.values[1:20,:]:
                        c+=1
                        temp_is = nap.IntervalSet(a[0],end=a[1])
                        tmpr = run_pca.restrict(temp_is)
                        tempdf = tmpr.as_dataframe()
                        # tempdf = tempdf.rolling(roll_wind).mean()
                        x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,0])),0]
                        # x = smooth(np.array(x),30)
                        y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,1])),1]
                        # x = smooth(np.array(y),30)
                        axs[1,1].plot(x,y,color_r,linewidth=0.5,zorder=0)
                        axs[1,1].scatter(x[0],y[0],s=mrkr_size,c=color_r_dots[0],zorder=5,alpha=0.9,edgecolors='k')
                        axs[1,1].scatter(x[-1],y[-1],s=mrkr_size,c=color_r_dots[1],zorder=10,alpha=0.9,edgecolors='k')
                        #finding location index to plot
                        temp_pos = nap_pos.restrict(temp_is)
                        I = np.argmin(np.abs(temp_pos-air_puff_loc))
                        axs[1,1].scatter(x[I],y[I],s=mrkr_size,c= color_ap_edge_r[0],edgecolors= color_ap_edge_r[1],zorder=15,alpha=0.9)
                    # axs[0,1].set_title('PCA space')
                    axs[1,1].set_title('PCA space')
                    axs[0,1].set_ylabel('PC2')
                    axs[0,1].set_xlabel('PC1')
                    axs[1,1].set_ylabel('PC2')
                    axs[1,1].set_xlabel('PC1')
                    
                    fig.savefig(save_fig_path+ses.astype(str)+"gcPCA_space_PCA_space.pdf", transparent=True)
                    # pre run plot
                    # for a in left_pr_int.values[1:20,:]:
                    #     c+=1
                    #     temp_is = nap.IntervalSet(a[0],end=a[1])
                    #     tmpr = prerun_gcpca.restrict(temp_is)
                    #     tempdf = tmpr.as_dataframe()
                    #     # tempdf = tempdf.rolling(30).mean()
                    #     x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,0])),0]
                    #     # x = smooth(np.array(x),30)
                    #     y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,1])),1]
                    #     # x = smooth(np.array(y),30)
                    #     axs[0,1].plot(x,y,color_l,linewidth=0.5)
                        
                    #     axs[0,1].scatter(x[0],y[0],s=50,c=color_l_dots[0])
                    #     axs[0,1].scatter(x[-1],y[-1],s=50,c=color_l_dots[1])
                        
                    #     #finding location index to plot
                    #     temp_pos = nap_pos.restrict(temp_is)
                    #     I = np.argmin(np.abs(temp_pos-air_puff_loc))
                    #     axs[0,1].scatter(x[I],y[I],s=50,c=['seagreen'])
                    
                    # for a in right_pr_int.values[1:20,:]:
                    #     c+=1
                    #     temp_is = nap.IntervalSet(a[0],end=a[1])
                    #     tmpr = prerun_gcpca.restrict(temp_is)
                    #     tempdf = tmpr.as_dataframe()
                    #     # tempdf = tempdf.rolling(roll_wind).mean()
                    #     x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,0])),0]
                    #     # x = smooth(np.array(x),30)
                    #     y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:,1])),1]
                    #     # x = smooth(np.array(y),30)
                    #     axs[1,1].plot(x,y,color_r,linewidth=0.5)
                    #     axs[1,1].scatter(x[0],y[0],s=50,c=color_r_dots[0])
                    #     axs[1,1].scatter(x[-1],y[-1],s=50,c=color_r_dots[1])
                    #     #finding location index to plot
                    #     temp_pos = nap_pos.restrict(temp_is)
                    #     I = np.argmin(np.abs(temp_pos-air_puff_loc))
                    #     axs[1,1].scatter(x[I],y[I],s=50,c=['seagreen'])
                    # axs[0,1].set_title(' pre run - gcPCA space')
                    # axs[1,1].set_title('gcPCA space')
                #%% make plots with projection
                # prerun_gcpca = nap.TsdFrame(prerun_time,d=prerun_data.dot(gcpca_mdl.loadings_[:,:2]))
                # run_gcpca = nap.TsdFrame(run_time,d=run_data.dot(gcpca_mdl.loadings_[:,:2]))
                
                
                # left_run_gcpca = trials_projection(run_gcpca,left_runs_int)
                # right_run_gcpca = trials_projection(run_gcpca,right_runs_int)
                # # #gcPCA projection on every left trial run
                
                # c = 0
                # plt.figure()
                # for a in left_runs_int.values:
                #     c+=1
                #     temp_is = nap.IntervalSet(a[0],end=a[1])
                #     tmpr = prerun_gcpca.restrict(temp_is)
                #     tempdf = tmpr.as_dataframe()
                #     tempdf = tempdf.rolling(3).mean()
                #     plt.plot(tempdf.values[:,0],tempdf.values[:,1],'b',linewidth=0.5)
                    
                # c = 0
                # plt.figure()
                # for a in right_runs_int.values:
                #     c+=1
                #     temp_is = nap.IntervalSet(a[0],end=a[1])
                #     tmpr = prerun_gcpca.restrict(temp_is)
                #     tempdf = tmpr.as_dataframe()
                #     tempdf = tempdf.rolling(3).mean()
                #     plt.plot(tempdf.values[:,0],tempdf.values[:,1],'r',linewidth=0.5)
                    
                # c = 0
                # for a in left_runs_int.values:
                #     c+=1
                #     temp_is = nap.IntervalSet(a[0],end=a[1])
                #     tmpr = run_gcpca.restrict(temp_is)
                #     tempdf = tmpr.as_dataframe()
                #     # tempdf = tempdf.rolling(20).mean()
                #     tempdf.columns = {'gcpc1','gcpc2'}
                    
                #     #append a new column to dataframe with trial information
                #     tempdf.insert(loc=2,
                #                   column='trial',
                #                   value = c*np.ones((tmpr.shape[0],1)) )
                #     if c==1:
                #         left_run_gcpca = tempdf
                #     else:
                #         left_run_gcpca = left_run_gcpca.append(tempdf)
                    
                    # sns.lineplot(
                    #     data=left_run_gcpca,
                    #     x="dim1", y="dim2", hue=None, units="trial",
                    #     estimator=None, lw=1,
                    # )
                # #gcPCA projection on every trial
                # c = 0
                # for a in left_runs_int.values:
                #     c+=1
                #     temp_is = nap.IntervalSet(a[0],end=a[1])
                #     tmpr = run_gcpca.restrict(temp_is)
                #     tempdf = tmpr.as_dataframe()
                #     # tempdf = tempdf.rolling(20).mean()
                #     tempdf.columns = {'gcpc1','gcpc2'}
                    
                #     #append a new column to dataframe with trial information
                #     tempdf.insert(loc=2,
                #                   column='trial',
                #                   value = c*np.ones((tmpr.shape[0],1)) )
                #     if c==1:
                #         left_run_gcpca = tempdf
                #     else:
                #         left_run_gcpca = left_run_gcpca.append(tempdf)

#%% finding periods of pre/post running

# pre_run_idx = []
# run_idx     = []
# c = 0
# for a in task['labels']:
#     if np.char.find(a[0][0],'-prerun') != -1:
#         pre_run_idx.append(c)
#     if np.char.find(a[0][0],'-run') != -1:
#         run_idx.append(c)
#     c+=1

# pre_run_intervals = task['ints'][np.round(pre_run_idx[0]/2).astype(int),:]
# run_intervals = task['ints'][np.round(run_idx[0]/2).astype(int),:]

#%% getting position in pre and pos and idenitifying trials



