# %% importing essentials
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
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import *
import seaborn as sns

from scipy.stats import zscore
import pickle
from numpy import linalg as LA

# %% parameters
min_n_cell = 50  # min number of cells in the brain area to be used

# %% import custom modules
# repo_dir = "/gs/gsfs0/users/edeolive/github/normalized_contrastive_PCA/" #repository dir
repo_dir = "/home/eliezyer/Documents/github/normalized_contrastive_PCA/"
sys.path.append(repo_dir)
from ncPCA import ncPCA
import ncPCA_project_utils as utils  # this is going to be our package of reusable functions

# %% preparing data to be loaded

data_directory = '/mnt/probox/allen_institute_data/ecephys/'  # must be a valid directory in your filesystem
# data_directory = '/gs/gsfs0/users/edeolive/allen_institute_data/ecephys/'
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# getting brain observatory dataset
sessions = cache.get_session_table()
selected_sessions = sessions[(sessions.session_type == 'brain_observatory_1.1')]

# %% get units size
# getting neurons for each session
array_of_ba = ['LGd', 'VISp', 'VISrl', 'VISpm']
dict_units = {}

# %% performing analysis, session loop starts here
cosine_sim = []
brain_area_name = []
session_name = []

variance_ncpca = []
variance_pca = []
session_list = []
ba_list = []
component_list = []

import math
def cosine_similarity(vec1, vec2):
  # calculate the dot product between the two vectors
  dot_product = sum(x * y for x, y in zip(vec1, vec2))

  # calculate the magnitudes of the two vectors
  vec1_magnitude = math.sqrt(sum(x ** 2 for x in vec1))
  vec2_magnitude = math.sqrt(sum(y ** 2 for y in vec2))

  # calculate and return the cosine similarity
  return dot_product / (vec1_magnitude * vec2_magnitude)


# %% getting cosine similarity and variance decay session wise
for session_id in selected_sessions.index.values:

    loaded_session = cache.get_session_data(session_id)

    # getting natural scenes and static gratings info
    stimuli_info = loaded_session.get_stimulus_table(["natural_scenes", "static_gratings"])

    # getting spikes times and information
    temp_spk_ts = loaded_session.spike_times
    temp_spk_info = loaded_session.units

    # here I'm picking all the brain areas that are related to visual system, i.e.,
    # starting with V or LG
    spikes_info = temp_spk_info[temp_spk_info["ecephys_structure_acronym"].str.contains("V|LG")]
    units2use = spikes_info.index.values

    temp_spk_ts_2 = {}
    for aa in np.arange(0, len(units2use)):
        temp_spk_ts_2[aa] = temp_spk_ts[units2use[aa]]

    # passing to pynapple ts group
    spikes_times = nap.TsGroup(temp_spk_ts_2)

    # adding structure info into the spike times TSgroup
    structs = pd.DataFrame(index=np.arange(len(units2use)), data=spikes_info.ecephys_structure_acronym.values,
                           columns=['struct'])
    spikes_times.set_info(structs)

    # passing natural scenes to intervalsets
    df_stim_ns = stimuli_info.query("stimulus_name=='natural_scenes'")
    ns_intervals = nap.IntervalSet(start=df_stim_ns.start_time.values, end=df_stim_ns.stop_time.values)

    df_stim_sg = stimuli_info.query("stimulus_name=='static_gratings'")
    sg_intervals = nap.IntervalSet(start=df_stim_sg.start_time.values, end=df_stim_sg.stop_time.values)

    # binning through using a loop in the cells until I can find a better way
    # start timer
    t_start = time.time()
    spikes_binned_ns = np.empty((len(ns_intervals.values), len(spikes_times.data)))
    bins_ns = ns_intervals.values.flatten()
    for aa in np.arange(len(spikes_times.data)):
        tmp = np.array(np.histogram(spikes_times.data[aa].index.values, bins_ns))
        spikes_binned_ns[:, aa] = tmp[0][np.arange(0, tmp[0].shape[0], 2)]
    t_stop = time.time()

    # same method of binning through a loop in the cells, but for static gratings
    spikes_binned_sg = np.empty((len(sg_intervals.values), len(spikes_times.data)))
    bins_sg = sg_intervals.values.flatten()
    for aa in np.arange(len(spikes_times.data)):
        tmp = np.array(np.histogram(spikes_times.data[aa].index.values, bins_sg))
        spikes_binned_sg[:, aa] = tmp[0][np.arange(0, tmp[0].shape[0], 2)]

    array_of_ba = spikes_info["ecephys_structure_acronym"].unique();

    spikes_zsc_ns = zscore(spikes_binned_ns)  # remove this later and update the variables name accordingly
    spikes_zsc_sg = zscore(spikes_binned_sg)

    # %% getting ncPCA and cPCA loadings

    brain_area_dict = {};

    for ba_name in array_of_ba:
        units_idx = spikes_info["ecephys_structure_acronym"] == ba_name
        if sum(units_idx.values) >= min_n_cell:

            X_train_ns, X_test_ns = train_test_split(spikes_zsc_ns, train_size=0.5)
            X_train_sg, _ = train_test_split(spikes_zsc_sg, train_size=0.5)

            # zeroing any cell that was nan (i.e. no activity)
            X_train_ns[np.isnan(X_train_ns)] = 0
            X_test_ns[np.isnan(X_test_ns)] = 0
            X_train_sg[np.isnan(X_train_sg)] = 0

            # ncPCA loadings
            ncPCA_mdl = ncPCA(basis_type='intersect', Nshuffle=10000)
            ncPCA_mdl.fit(X_train_sg, X_train_ns)
            X_fw_ns = ncPCA_mdl.loadings_
            data_ncPCA_fw_ns = []
            data_ncPCA_fw_ns = np.dot(X_test_ns, X_fw_ns)

            # PCA loadings
            _, _, Vns = np.linalg.svd(X_train_ns, full_matrices=False)
            data_PCA_ns = []
            data_PCA_ns = np.dot(X_test_ns, Vns.T)

            # cosine similarrity
            temp_cos = []
            for i in range(np.shape(data_ncPCA_fw_ns)[1]):
                temp_cos.append(cosine_similarity(data_ncPCA_fw_ns[:, i],
                                                  data_PCA_ns[:, i]))

            cosine_sim.append(np.mean(temp_cos))
            brain_area_name.append(ba_name)
            session_name.append(session_id)

            # variance decay
            variance_ncpca.extend(np.var(data_ncPCA_fw_ns, axis=0).tolist())
            variance_pca.extend(np.var(data_PCA_ns, axis=0).tolist())
            session_list.extend(np.tile(session_id, len(variance_ncpca)).tolist())
            ba_list.extend(np.tile(ba_name, len(variance_ncpca)).tolist())
            component_list.extend([i + i for i in range(len(variance_ncpca))])

# %% parameters for plotting
rcParams['figure.dpi'] = 500
rcParams['lines.linewidth'] = 2.5
rcParams['axes.linewidth'] = 1.5
rcParams['font.size'] = 12

# %% potting cosine similarrity
cosine_sim_df = pd.DataFrame(
    {'cosine_sim': cosine_sim,
     'brain_area_name': brain_area_name,
     'session_name': session_name
     })

design = 'black_bg'
if design == 'black_bg':
    style.use('dark_background')

sns.boxplot(data=cosine_sim_df, x='brain_area_name', y='cosine_sim')


#%% plotting variances

design = 'black_bg'
if design == 'black_bg':
    style.use('dark_background')

variance =pd.DataFrame(
    {'variance_ncpca': variance_ncpca,
     'variance_pca': variance_pca,
     'session':session_list,
     'ba_name': ba_list,
     'components':component_list
    })

plt.figure()

fig,ax= plt.subplots(2,2)

sns.lineplot(ax=ax[0,0], data = variance[variance['ba_name'] =='VISpm'], x = 'variance_ncpca', y = 'components')
sns.lineplot(ax=ax[0,0], data = variance[variance['ba_name'] =='VISpm'], x = 'variance_pca', y = 'components')
ax[0,0].legend(["ncPCA", "PCA"])
ax[0,0].set_title("VISpm")

sns.lineplot(ax=ax[0,0], data = variance[variance['ba_name'] =='VISp'], x = 'variance_ncpca', y = 'components')
sns.lineplot(ax=ax[0,0], data = variance[variance['ba_name'] =='VISp'], x = 'variance_pca', y = 'components')
ax[0,1].legend(["ncPCA", "PCA"])
ax[0,1].set_title("VISp")

sns.lineplot(ax=ax[0,0], data = variance[variance['ba_name'] =='LGd'], x = 'variance_ncpca', y = 'components')
sns.lineplot(ax=ax[0,0], data = variance[variance['ba_name'] =='LGd'], x = 'variance_pca', y = 'components')
ax[1,0].legend(["ncPCA", "PCA"])
ax[1,0].set_title("LGd")

sns.lineplot(ax=ax[0,0], data = variance[variance['ba_name'] =='VISrl'], x = 'variance_ncpca', y = 'components')
sns.lineplot(ax=ax[0,0], data = variance[variance['ba_name'] =='VISrl'], x = 'variance_pca', y = 'components')
ax[1,1].legend(["ncPCA", "PCA"])
ax[1,1].set_title("VISrl")

fig.set_figwidth(8)
fig.set_figheight(8)

plt.show()
