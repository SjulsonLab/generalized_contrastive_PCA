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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
repo_dir = "/home/eliezyer/Documents/github/normalized_contrastive_PCA/" #repository dir
sys.path.append(repo_dir)
from ncPCA import ncPCA
import math
from numpy import linalg as LA


data_directory = r"D:\Desktop\nCPA_\data" # must be a valid directory in your filesystem
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
loaded_session = cache.get_session_data(715093703)

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

array_of_ba = spikes_info["ecephys_structure_acronym"].unique();

spikes_zsc_ns = zscore(spikes_binned_ns)
spikes_zsc_sg = zscore(spikes_binned_sg)

min_n_cell = 50
brain_area_dict = {}

#%% getting ncPCA and cPCA loadings
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

        brain_area_dict['data_ncPCA_fw_ns_' + ba_name] = data_ncPCA_fw_ns
        brain_area_dict['data_PCA_ns_' + ba_name] = data_PCA_ns

#cosine similarity
def cosine_similarity(vec1, vec2):
  # calculate the dot product between the two vectors
  dot_product = sum(x * y for x, y in zip(vec1, vec2))

  # calculate the magnitudes of the two vectors
  vec1_magnitude = math.sqrt(sum(x ** 2 for x in vec1))
  vec2_magnitude = math.sqrt(sum(y ** 2 for y in vec2))

  # calculate and return the cosine similarity
  return dot_product / (vec1_magnitude * vec2_magnitude)


cos1 =[]
for i in range(np.shape(brain_area_dict["data_ncPCA_fw_ns_VISpm"])[1]):
    cos1.append(cosine_similarity(brain_area_dict["data_ncPCA_fw_ns_VISpm"][:,i],
                                  brain_area_dict["data_PCA_ns_VISpm"][:,i]))

cos2 =[]
for i in range(np.shape(brain_area_dict["data_ncPCA_fw_ns_VISp"])[1]):
    cos2.append(cosine_similarity(brain_area_dict["data_ncPCA_fw_ns_VISp"][:,i],
                                  brain_area_dict["data_PCA_ns_VISp"][:,i]))

cos3 =[]
for i in range(np.shape(brain_area_dict["data_ncPCA_fw_ns_VISp"])[1]):
    cos3.append(cosine_similarity(brain_area_dict["data_ncPCA_fw_ns_LGd"][:,i],
                                  brain_area_dict["data_PCA_ns_LGd"][:,i]))

cos4 =[]
for i in range(np.shape(brain_area_dict["data_ncPCA_fw_ns_VISp"])[1]):
    cos4.append(cosine_similarity(brain_area_dict["data_ncPCA_fw_ns_VISrl"][:,i],
                                  brain_area_dict["data_PCA_ns_VISrl"][:,i]))

# plotting
design = 'black_bg'
if design=='black_bg':
    style.use('dark_background')

g = sns.boxplot(data= [cos1,cos2,cos3,cos4])
g.set_xticklabels( ['VISpm', 'VISp', 'LGd', 'VISrl'])



plt.figure()
fig,ax= plt.subplots(2,2)

var1 = np.var(brain_area_dict["data_ncPCA_fw_ns_VISpm"],axis =0)
var1_= np.var(brain_area_dict["data_PCA_ns_VISpm"], axis =0)

ax[0,0].plot(var1, 'r')
ax[0,0].plot(var1_, 'k')
ax[0,0].legend(["ncPCA", "PCA"])
ax[0,0].set_title("VISpm")

var2 = np.var(brain_area_dict["data_ncPCA_fw_ns_VISp"],axis =0)
var2_= np.var(brain_area_dict["data_PCA_ns_VISp"], axis =0)
ax[0,1].plot(var2, 'r')
ax[0,1].plot(var2_, 'k')
ax[0,1].legend(["ncPCA", "PCA"])
ax[0,1].set_title("VISp")

var3 = np.var(brain_area_dict["data_ncPCA_fw_ns_LGd"],axis =0)
var3_= np.var(brain_area_dict["data_PCA_ns_LGd"], axis =0)
ax[1,0].plot(var3, 'r')
ax[1,0].plot(var3_, 'k')
ax[1,0].legend(["ncPCA", "PCA"])
ax[1,0].set_title("LGd")

var4 = np.var(brain_area_dict["data_ncPCA_fw_ns_VISrl"],axis =0)
var4_= np.var(brain_area_dict["data_PCA_ns_VISrl"], axis =0)
ax[1,1].plot(var4, 'r')
ax[1,1].plot(var4_, 'k')
ax[1,1].legend(["ncPCA", "PCA"])
ax[1,1].set_title("VISrl")

fig.set_figwidth(8)
fig.set_figheight(8)

plt.show()
