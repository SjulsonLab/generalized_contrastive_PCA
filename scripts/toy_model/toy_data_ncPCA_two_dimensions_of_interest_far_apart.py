# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 09:12:38 2023

@author: eliezyer

script to make the toy model for ncPCA with one big PC that is irrelavant,
but two PCs

Loadings of each is exactly the same, i.e., U and V are identical
"""


# %% importing libraries and defining main variables

#libraries
import numpy as np
from matplotlib.pyplot import *
from scipy.linalg import orth
from scipy.stats import zscore
import sys
import seaborn as sns

repo_dir = "C:\\Users\\fermi\\Documents\\GitHub\\normalized_contrastive_PCA" #repository dir
sys.path.append(repo_dir)
from ncPCA import ncPCA
from ncPCA import cPCA

sns.set_style("whitegrid")
sns.set_context("talk")
#variables
N_times    = 1000 #number of observations
N_features = 100 #number of features (and PCs, data is full rank)
pc_num1 = 40
pc_num2 = 95 #pc that is going to be altered more than the rest
pc_num3 = 75
#%% generating toy data

#background data
temp_S = np.linspace(1,stop=N_features,num=N_features) #variance of background activity, decays in 1/f
# S_bg   = 1/temp_S
S_bg   = -1*np.arange(len(temp_S))+100.1
# S_bg   = np.ones(len(temp_S))+10

#foreground data, where we want to compare the change to background
#delta_var = np.random.randn(N_features)/100 #how much variance to vary by default, we are doing a normali distribution of 1% change in the SD
#S_fg      = S_bg*(1+delta_var)
S_fg = S_bg.copy()
S_fg[0] = S_bg[0]*1.03;
S_fg[2] = S_bg[2]*1.04;
S_fg[1] = S_bg[1]*1.02;
S_fg[10] = S_bg[10]*1.07;

S_fg[pc_num1] = S_fg[pc_num1]*(1.05)
S_fg[pc_num2] = S_fg[pc_num2]*(1.10)
S_fg[pc_num3] = S_fg[pc_num3]*(1.06)
#S_bg[pc_num] = 0

# generating random orthogonal loadings
U = orth(np.random.randn(N_times,N_features))
V = orth(np.random.randn(N_features,N_features))

# #trying to model on U
# Unew = U.copy()
# Unew[:,pc_num1] = np.concatenate((np.random.normal(1,0.1,250),np.random.normal(0,0.1,500),np.random.normal(-1,0.1,250)))
# Unew[:,pc_num2] = np.concatenate((np.random.normal(0,0.1,250),np.random.normal(1,0.1,250),np.random.normal(-1,0.1,250),np.random.normal(0,0.1,250)))

# Unew[:,pc_num1] = Unew[:,pc_num1]/np.linalg.norm(Unew[:,pc_num1]) 
# Unew[:,pc_num2] = Unew[:,pc_num2]/np.linalg.norm(Unew[:,pc_num2])


figure()
loglog(np.arange(1,101),S_bg,label='background dataset')
loglog(np.arange(1,101),S_fg,label='foreground dataset')
legend()
xlabel('PCs')
ylabel('Eigenvalues')
tight_layout()
#figure;plot((S_bg-S_fg)/(S_bg+S_fg))

#making another plot to show the difference
fig,axs = subplots(2)
fig.suptitle('Difference of eigenvalues')
axs[0].plot(np.arange(1,101),S_fg-S_bg)
axs[0].set(ylabel='diff in mag')


axs[1].plot(np.arange(1,101),(S_fg-S_bg)/(S_bg))
axs[1].set(xlabel='PCs')
axs[1].set(ylabel='diff in %')
fig.tight_layout()
"""
#%% testing generating the difference in the data and then rotating the data

from scipy.stats import ortho_group
np.random.seed(0) # for reproducibility

# from CPCA toy dataset
# In A there are four clusters.
N = 400; D = 30; gap=1.5
scaling = 1;
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
target_[200:300, 20:30] = np.random.normal(-gap,1,(100,10))*scaling
# group 4
target_[300:400, 10:20] = np.random.normal(2*gap,1,(100,10))
target_[300:400, 20:30] = np.random.normal(gap,1,(100,10))*scaling
target_ = target_.dot(rotation)

sub_group_labels_ = [0]*100+[1]*100+[2]*100+[3]*100

background_ = np.zeros((N, D))
background_[:,0:10] = np.random.normal(0,10,(N,10))
background_[0:200,10:20] = np.random.normal(0,3,(int(N/2),10))
background_[0:200,20:30] = np.random.normal(0,1,(int(N/2),10))*scaling*2
background_[200:400,10:20] = np.random.normal(0,3,(int(N/2),10))
background_[200:400,20:30] = np.random.normal(0,1,(int(N/2),10))*scaling*2
background_ = background_.dot(rotation)

#% testing the model above

alpha2use = 7.4

#fitting cPCA
cPCs_all,w,eigidx = cPCA(background_,target_,alpha=alpha2use,n_components=target_.shape[1])

#fitting ncPCA
mdl = ncPCA(basis_type='intersect',normalize_flag=False)
mdl.fit(background_,target_)
ncPCs_all = mdl.loadings_


#projecting
cpcs_proj = target_.dot(cPCs_all[:,:2])
ncpcs_proj = target_.dot(ncPCs_all[:,:2])

#plotting
figure()
subplot(1,2,1)
scatter(cpcs_proj[0:100,0],cpcs_proj[0:100, 1],c='b')
scatter(cpcs_proj[100:200,0],cpcs_proj[100:200, 1],c='r')
scatter(cpcs_proj[200:300,0],cpcs_proj[100:200, 1],c='g')
scatter(cpcs_proj[300:400,0],cpcs_proj[300:400, 1],c='k')

subplot(1,2,2)
scatter(ncpcs_proj[0:100,0],ncpcs_proj[0:100, 1],c='b')
scatter(ncpcs_proj[100:200,0],ncpcs_proj[100:200, 1],c='r')
scatter(ncpcs_proj[200:300,0],ncpcs_proj[100:200, 1],c='g')
scatter(ncpcs_proj[300:400,0],ncpcs_proj[300:400, 1],c='k')
"""
#%% reconstruct data

data_bg = np.linalg.multi_dot((U,np.diag(S_bg),V.T));
data_fg = np.linalg.multi_dot((U,np.diag(S_fg),V.T));

# data_bg = np.linalg.multi_dot((Unew,np.diag(S_bg),V.T));
# data_fg = np.linalg.multi_dot((Unew,np.diag(S_fg),V.T));


#%% now run cPCA and ncPCA
alpha2use = 1.1306
#cPCs,w,eigidx = cPCA(data_bg,data_fg,alpha=alpha2use)[:,0]

cPCs_all,w,eigidx = cPCA(data_bg,data_fg,alpha=alpha2use,n_components=len(V))
#"""
mdl = ncPCA(basis_type='all',normalize_flag=False)
mdl.fit(data_bg, data_fg)
#ncPCs = mdl.loadings_[:,0]
ncPCs_all = mdl.loadings_
#"""
#plot(np.corrcoef(cPCs_all.T,V[:,pc_num])[-1,:-1])
#plot(np.corrcoef(ncPCs_all.T,V[:,pc_num])[-1,:-1])
#% get the correlation of cPCs 1 to the modeled

cPCs_corr = np.corrcoef(V.T,cPCs_all[:,0])[-1,:len(cPCs_all)]
ncPCs_corr = np.corrcoef(V.T,ncPCs_all[:,0])[-1,:len(ncPCs_all)]

figure()
plot(np.abs(cPCs_corr),label='cPCs')
plot(np.abs(ncPCs_corr),'--',label='ncPCs')
legend()
xlabel('PCs')
ylabel('|Corr| with modeled vectors')
title('top PC of each method')
tight_layout()


cPCs_corr = np.corrcoef(V.T,cPCs_all[:,1])[-1,:len(cPCs_all)]
ncPCs_corr = np.corrcoef(V.T,ncPCs_all[:,1])[-1,:len(ncPCs_all)]

figure()
plot(np.abs(cPCs_corr),label='cPCs')
plot(np.abs(ncPCs_corr),'--',label='ncPCs')
legend()
xlabel('PCs')
ylabel('|Corr| with modeled vectors')
title('second PC of each method')
tight_layout()

#%% run multiple alphas

alphas_vec = np.linspace(0.2,1.5,num=50)
cPC1st = []
cPC2nd = []
cPC3rd = []
cPC4th = []
for alpha in alphas_vec:
    cPCs_all,w,eigidx = cPCA(data_bg,data_fg,alpha=alpha,n_components=len(V))
    cPCs_corr = np.abs(np.corrcoef(V.T,cPCs_all[:,0])[-1,:len(cPCs_all)])
    cPC1st.append(np.argmax(cPCs_corr))
    cPCs_corr = np.abs(np.corrcoef(V.T,cPCs_all[:,1])[-1,:len(cPCs_all)])
    cPC2nd.append(np.argmax(cPCs_corr))
    cPCs_corr = np.abs(np.corrcoef(V.T,cPCs_all[:,2])[-1,:len(cPCs_all)])
    cPC3rd.append(np.argmax(cPCs_corr))
    cPCs_corr = np.abs(np.corrcoef(V.T,cPCs_all[:,3])[-1,:len(cPCs_all)])
    cPC4th.append(np.argmax(cPCs_corr))
    
figure()
scatter(alphas_vec,cPC1st,alpha=0.5,label='1st cPC')
scatter(alphas_vec,cPC2nd,alpha=0.5,label='2nd cPC')
scatter(alphas_vec,cPC3rd,alpha=0.5,label='3rd cPC')
scatter(alphas_vec,cPC4th,alpha=0.5,label='4th cPC')
plot(alphas_vec,np.ones(len(alphas_vec))*10,'r--',alpha=0.4,label='dimensions of interest')
plot(alphas_vec,np.ones(len(alphas_vec))*pc_num2,'r--',alpha=0.4)
plot(alphas_vec,np.ones(len(alphas_vec))*75,'r--',alpha=0.4,label='dimensions of interest')
plot(alphas_vec,np.ones(len(alphas_vec))*41,'r--',alpha=0.4,label='dimensions of interest')
xlabel('alpha values')
ylabel('Recovered dim. from model')
legend()
tight_layout()

#%% plot subtracting the eigenspectrum
sns.set_style("ticks")
SFG = np.tile(S_fg,len(alphas_vec))
SFG.resize((50,100))

SBG = np.tile(S_bg,len(alphas_vec))
SBG.resize((50,100))

newalpha = np.tile(alphas_vec,100)
newalpha.resize((100,50))
figure()
cpc_eq = SFG-np.multiply(newalpha.T,SBG)
imshow(cpc_eq,extent=[1, 100, alphas_vec[-1],alphas_vec[0]],aspect="auto")
clim((-0.1,0.050))

figure()
cpc_eq = SFG-np.multiply(newalpha.T,SBG)
imshow(zscore(cpc_eq.T).T,extent=[1, 100, alphas_vec[-1],alphas_vec[0]],aspect="auto")
clim((-0.5,0.5))