# TODO: improve toy dataset to merge clusters during cPCA

import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import ListedColormap
warnings.filterwarnings('ignore')

# for plotting
cmap2 = ListedColormap(['r', 'k'])
cmap4 = ListedColormap([])

from scipy.stats import ortho_group
np.random.seed(0) # for reproducibility

# from CPCA toy dataset
# In A there are four clusters.
N = 400; D = 30; gap=1.5
rotation = ortho_group.rvs(dim=D)

target_ = np.zeros((N, D))
target_[:,0:10] = np.random.normal(0,10,(N,10))
# group 1
target_[0:100, 10:20] = np.random.normal(-gap,1,(100,10))
target_[0:100, 20:30] = np.random.normal(-gap,1,(100,10))
# group 2
target_[100:200, 10:20] = np.random.normal(-gap,1,(100,10))
target_[100:200, 20:30] = np.random.normal(gap,1,(100,10))
# group 3
target_[200:300, 10:20] = np.random.normal(2*gap,1,(100,10))
target_[200:300, 20:30] = np.random.normal(-gap,1,(100,10))
# group 4
target_[300:400, 10:20] = np.random.normal(2*gap,1,(100,10))
target_[300:400, 20:30] = np.random.normal(gap,1,(100,10))
target_ = target_.dot(rotation)
sub_group_labels_ = [0]*100+[1]*100+[2]*100+[3]*100

background_ = np.zeros((N, D))
background_[:,0:10] = np.random.normal(0,10,(N,10))
background_[:,10:20] = np.random.normal(0,3,(N,10))
background_[:,20:30] = np.random.normal(0,1,(N,10))
background_ = background_.dot(rotation)

data_ = np.concatenate((background_, target_))
labels_ = len(background_)*[0] + len(target_)*[1]

# norm values
amp = np.linalg.norm(background_)
amp1 = np.linalg.norm(target_[0:100,:])
amp2 = np.linalg.norm(target_[100:200,:])
amp3 = np.linalg.norm(target_[200:300,:])
amp4 = np.linalg.norm(target_[300:400,:])

target = np.zeros(np.shape(target_))
target[0:100,:] = (target_ [0:100,:] * .5* amp)/amp1
target[100:200,:] = (target_ [100:200,:] *  1.7* amp)/amp2
target[200:300,:] = (target_ [200:300,:] * 1.4 * amp)/amp3
target[300:400,:] = (target_ [300:400,:] * .3* amp)/amp4
background= background_

# getting and plotting cPCA loadings
import sys
sys.path.append(r"D:\Desktop\nCPA_\contrastive-master")
from contrastive import CPCA
mdl = CPCA()
projected_data_ = mdl.fit_transform(target, background, plot=True, active_labels=sub_group_labels_)

# getting ncPCA loadings
sys.path.append(r"D:\Desktop\nCPA_\ncPCA")
from ncPCA import ncPCA
fg = target
bg = background
ncPCA_mdl = ncPCA(basis_type="all")
ncPCA_mdl.fit(bg, fg)
target_projected = ncPCA_mdl.N2_scores_

# plotting ncPCA loadings
colors = ['k', 'r', 'g', 'b']
plt.figure()
plt.scatter(target_projected[0:100,0], target_projected[0:100,1], color = colors[0], alpha = 0.5)
plt.scatter(target_projected[100:200,0], target_projected[100:200,1], color = colors[1], alpha = 0.5)
plt.scatter(target_projected[200:300,0], target_projected[200:300,1], color = colors[2], alpha = 0.5)
plt.scatter(target_projected[300:400,0], target_projected[300:400,1], color = colors[3], alpha = 0.5)
plt.title("ncPCA")