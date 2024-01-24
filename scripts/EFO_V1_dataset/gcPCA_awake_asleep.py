#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:29:58 2024

@author: eliezyer

script to run gcPCA on awake versus sleep

"""

#%% importing
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
from scipy.io import loadmat
import mat73 #to load matlab v7.3 files

repo_dir = "/home/eliezyer/Documents/github/generalized_contrastive_PCA/"
sys.path.append(repo_dir)
from contrastive_methods import gcPCA


#%% parameters

min_n_cell = 30 #min number of cells in the brain area to be used
min_fr = 0.01 #minimum firing rate to be included in the analysis
bin_size = 0.1 # 0.01
bin_size_task = 0.05; # 0.05
plt.rcParams['figure.dpi'] = 300
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 12
#%% sessions we are using

sessions = ['/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_F2763/210711',
    '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_F2763/210712',
    '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_F2763/210713',
    '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_F2763/210714',
    '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_F2762/210723',
    '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_F2762/210724',
    '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_M2041/230518',
    '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_M2041/230519',
    '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_M2041/230521',
    '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_M2074/230606',
    '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_M2074/230607',
    '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_M2074/230608',
    '/mnt/SSD4TB/Preprocessing/sleep_preserved_dimensions/VC_EFO_M2074/230609']

for session in sessions:
    #%% main analysis
    _,basename = os.path.split(session)
    # load spikes
    tmp = loadmat(os.path.join(session,basename+'.spikes.cellinfo.mat'))
    tmp_var = tmp['spikes']['times'][0][0][0]
    
    # passing to TS
    temp_ts = {}
    for a,cll in enumerate(tmp_var):
        temp_ts[a] =nap.Ts(cll)
    spikes_times = nap.TsGroup(temp_ts)
    
    # load mua
    tmp = loadmat(os.path.join(session,basename+'.mua.cellinfo.mat'))
    tmp_var = tmp['mua']['times'][0][0][0]
    
    # passing to TS
    temp_ts = {}
    for a,cll in enumerate(tmp_var):
        temp_ts[a] =nap.Ts(cll)
    mua_times = nap.TsGroup(temp_ts)
    
    # load states
    tmp = loadmat(os.path.join(session,basename+'.SleepState.states.mat'))
    tmp_var = tmp['SleepState']['ints'][0][0][0]
    
    SWS_interval = nap.IntervalSet(start=tmp_var['NREMstate'][0][:,0],end=tmp_var['NREMstate'][0][:,1])
    AWK_interval = nap.IntervalSet(start=tmp_var['WAKEstate'][0][:,0],end=tmp_var['WAKEstate'][0][:,1])
    
    # load motion
    tmp = mat73.loadmat(os.path.join(session,basename+'.rebinned_variables.mat'))
    face_motion_1s = tmp['rebinned']['face_motion_1s'];
    bins_face_motion_1s = tmp['rebinned']['bins1s']
    
    # load visual stimulation
    tmp = loadmat(os.path.join(session,basename+'.visual_stimulation.mat'))
    tmp_var = tmp['visual_stimulation']['natural_scenes'][0][0][0]['intervals']
    NS_intervals = nap.IntervalSet(start=tmp_var[0][:,0],end=tmp_var[0][:,1])
    tmp_var = tmp['visual_stimulation']['natural_scenes'][0][0][0]['identity']
    NS_identity  = tmp_var[0][0]
    
    #%% gcPCA awake vs sleep
    tmp = spikes_times.restrict(AWK_interval).count(bin_size)
    awake_zsc_data = zscore(tmp.values)
    
    tmp = spikes_times.restrict(SWS_interval).count(bin_size)
    asleep_zsc_data = zscore(tmp.values)
    
    gcpca_mdl = gcPCA(method='v4',normalize_flag=True)
    gcpca_mdl.fit(awake_zsc_data,asleep_zsc_data)
    
    #%% prediction of face motion
    
    # separate motion prediction to first 40 min
    face_motion_1s
    # cross validate

    # separate
