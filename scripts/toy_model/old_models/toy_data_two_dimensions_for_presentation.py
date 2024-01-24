#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:08:31 2024

@author: eliezyer

script to make toy model and plots for introduction presentation, 
this shows the covariance matrices generated, increased, and also a perfect
scenario where A-B would work, then cPCA solve it with lambda and we solve with
normalization.
"""

# %% importing libraries and defining main variables

#libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orth
from scipy.stats import zscore
import sys
import seaborn as sns
import random

# repo_dir = "C:\\Users\\fermi\\Documents\\GitHub\\generalized_contrastive_PCA\\" #repository dir on laptop
repo_dir = "/home/eliezyer/Documents/github/generalized_contrastive_PCA/" #repository dir on linux machine
sys.path.append(repo_dir)
from contrastive_methods import gcPCA
from ncPCA_project_utils import cosine_similarity_multiple_vectors

sns.set_style("whitegrid")
sns.set_context("talk")
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 150
plt.rc('axes',edgecolor='k')
# folder_save_plot = "C:\\Users\\fermi\\Dropbox\\figures_ncPCA\\toy_data\\" # on laptop
folder_save_plot = "/mnt/extraSSD4TB/CloudStorage/Dropbox/figures_gcPCA/toy_data/" #on linux machine
#variables
N_samples  = 100 #number of observations

N_features = 30 #number of features (and PCs, data is full rank)
pc_change  = [10,28] #pcs that are going to be changed in variance
ratio_change = [1.05,2]
np.random.seed(1)
#%% generating toy data with linear decay

# B dataset or background data
S_B = np.linspace(0,stop=4,num=N_features)[::-1]+10**-4

# A dataset, or foreground data, what we want to identify the highly exposed
S_A_perfect = S_B.copy()
S_A = 1.1*S_B.copy()  # LATER ON TRY TO DO RANDOM VARIABILITY ON EACH COMP.

#injecting variance in specific components
S_A_perfect[pc_change[0]] = S_A_perfect[pc_change[0]]*ratio_change[0];
S_A_perfect[pc_change[1]] = S_A_perfect[pc_change[1]]*ratio_change[1];

S_A[pc_change[0]] = S_A[pc_change[0]]*ratio_change[0];
S_A[pc_change[1]] = S_A[pc_change[1]]*ratio_change[1];

#S_bg[pc_num] = 0

# generating random orthogonal loadings
U = orth(np.random.randn(N_samples,N_features))
V = orth(np.random.randn(N_features,N_features))

plt.figure()
plt.plot(np.arange(1,N_features+1),S_B,':',label='Condition B')
plt.plot(np.arange(1,N_features+1),S_A,'--',label='Condition A')
plt.legend()
plt.xlabel('Components')
plt.ylabel('Eigenvalues')
plt.tight_layout()



#plot the cPCA and show that is not symmetrical
plt.figure()
plt.plot(np.arange(1,N_features+1),(S_A)/(S_B),'--',color='k')
plt.plot(np.arange(1,N_features+1),(S_B)/(S_A),color='r')
sns.despine()
plt.grid()
plt.xticks(np.arange(5,N_features+1,step=5))
plt.xlabel('Dimensions')
plt.ylabel('A/B')


#plot to check the gcPCA
plt.figure()
plt.plot(np.arange(1,N_features+1),(S_A-S_B)/(S_B),'--',color='k')
plt.plot(np.arange(1,N_features+1),(S_B-S_A)/(S_A),color='r')
sns.despine()
plt.grid()
plt.xticks(np.arange(5,N_features+1,step=5))
plt.xlabel('Dimensions')
plt.ylabel('(A-B) / (B)')


#plot to check the gcPCA
plt.figure()
plt.plot(np.arange(1,N_features+1),(S_A-S_B)/(S_A+S_B),'--',color='k')
plt.plot(np.arange(1,N_features+1),(S_B-S_A)/(S_B+S_A),color='r')
sns.despine()
plt.grid()
plt.xticks(np.arange(5,N_features+1,step=5))
plt.xlabel('Dimensions')
plt.ylabel('(A-B) / (A+B)')
#%% build the data from components and eigenvalues

data_B = np.linalg.multi_dot((U,np.diag(S_B),V.T));
data_A = np.linalg.multi_dot((U,np.diag(S_A),V.T));
data_A_perfect = np.linalg.multi_dot((U,np.diag(S_A_perfect),V.T));

cov_B = data_B.T.dot(data_B)
cov_A = data_A.T.dot(data_A)
cov_A_perfect = data_A_perfect.T.dot(data_A_perfect)

#%% plot of components and vectors

sns.set_style("ticks")
plt.figure()
ax=plt.subplot(1,1,1)
plt.scatter(np.arange(start=1,stop=len(S_B)+1),S_B,c='seagreen',marker='s',s=30)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.xlabel('Components')
plt.ylabel('Eigenvalues')

#%% plot of eigenvalues and covariance matrices of perfect example and A-B
from matplotlib.ticker import StrMethodFormatter
sns.set_style("ticks")
grid1 = plt.GridSpec(1, 2, left=0.2 , wspace = 0.4, hspace=0.05)
plt.figure(figsize=(15,5))
ax=plt.subplot(grid1[0])
plt.plot(np.arange(1,N_features+1),S_A_perfect,'-',label='Condition A',c='seagreen')
plt.plot(np.arange(1,N_features+1),S_B,'--',label='Condition B',c='goldenrod')
plt.title('Conditions Eigenspectra')
plt.ylabel('Eigenvalues')
plt.xlabel('Components')
plt.legend()
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

ax=plt.subplot(grid1[1])
plt.plot(np.arange(1,N_features+1),S_A_perfect-S_B,'-',label='Condition A',c='k')
plt.title('Contrasting A - B')
plt.ylabel('A-B')
plt.xlabel('Components')
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

# repeat past plot but for covariance
sns.set_style("ticks")
grid1 = plt.GridSpec(1, 3, left=0.2 , wspace = 0.4, hspace=0.05)
plt.figure(figsize=(15,5))
ax=plt.subplot(grid1[0])
plt.imshow(cov_A_perfect-np.diag(np.diag(cov_A_perfect)),cmap='bwr',clim=(-3,3))
plt.title('Condition A')
plt.ylabel('Features')
plt.xlabel('Features')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')


ax=plt.subplot(grid1[1])
plt.imshow(cov_B-np.diag(np.diag(cov_B)),cmap='bwr',clim=(-3,3))
plt.title('Condition B')
plt.ylabel('Features')
plt.xlabel('Features')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

ax=plt.subplot(grid1[2])
auxplot = cov_A_perfect - cov_B;
auxplot = auxplot - np.diag(np.diag(auxplot))
plt.imshow(auxplot,cmap='bwr',clim=(-0.1,0.1))
plt.title('Contrasting A - B')
plt.xlabel('Features')
plt.ylabel('Features')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

#%% plot of eigenvalues and datasets with more realistic dataset A and B

from matplotlib.ticker import StrMethodFormatter
grid1 = plt.GridSpec(1, 2, left=0.2 , wspace = 0.4, hspace=0.05)
plt.figure(figsize=(15,5))
ax=plt.subplot(grid1[0])
plt.plot(np.arange(1,N_features+1),S_A,'-',label='Condition A',c='seagreen')
plt.plot(np.arange(1,N_features+1),S_B,'-',label='Condition B',c='goldenrod')
plt.title('Conditions Eigenspectra')
plt.ylabel('Eigenvalues')
plt.xlabel('Components')
plt.legend()
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

ax=plt.subplot(grid1[1])
plt.plot(np.arange(1,N_features+1),S_A-S_B,'-',label='Condition A',c='k')
plt.title('Contrasting A - B')
plt.ylabel('A-B')
plt.xlabel('Components')
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

# repeat past plot but for covariance
sns.set_style("ticks")
grid1 = plt.GridSpec(1, 3, left=0.2 , wspace = 0.4, hspace=0.05)
plt.figure(figsize=(15,5))
ax=plt.subplot(grid1[0])
plt.imshow(cov_A-np.diag(np.diag(cov_A)),cmap='bwr',clim=(-3,3))
plt.title('Condition A')
plt.ylabel('Features')
plt.xlabel('Features')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')


ax=plt.subplot(grid1[1])
plt.imshow(cov_B-np.diag(np.diag(cov_B)),cmap='bwr',clim=(-3,3))
plt.title('Condition B')
plt.ylabel('Features')
plt.xlabel('Features')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

ax=plt.subplot(grid1[2])
auxplot = cov_A - cov_B;
auxplot = auxplot - np.diag(np.diag(auxplot))
plt.imshow(auxplot,cmap='bwr',clim=(-0.1,0.1))
plt.title('Contrasting A - B')
plt.xlabel('Features')
plt.ylabel('Features')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
#%%
#plot of covariance matrices
plt.figure();
plt.imshow(cov_B-np.diag(np.diag(cov_B)),cmap='bwr',clim=(-3,3))
plt.colorbar()
plt.title('B covariance')
plt.xlabel('Features')
plt.ylabel('Features')
plt.grid(False)
plt.savefig(folder_save_plot+"B_cov.pdf", transparent=True)
plt.savefig(folder_save_plot+"B_cov.png", transparent=True,bbox_inches='tight')

plt.figure();
plt.imshow(cov_A-np.diag(np.diag(cov_A)),cmap='bwr',clim=(-3,3))
plt.colorbar()
plt.title('A covariance')
plt.xlabel('Features')
plt.ylabel('Features')
plt.grid(False)
plt.savefig(folder_save_plot+"A_cov.pdf", transparent=True)
plt.savefig(folder_save_plot+"A_cov.png", transparent=True,bbox_inches='tight')

plt.figure();
auxplot = cov_A - cov_B;
auxplot = auxplot - np.diag(np.diag(auxplot))
plt.imshow(auxplot,cmap='bwr',clim=(-1,1))
plt.colorbar()
plt.title('A - B')
plt.xlabel('Features')
plt.ylabel('Features')
plt.grid(False)
plt.savefig(folder_save_plot+"A-B_cov.pdf", transparent=True)
plt.savefig(folder_save_plot+"A-B_cov.png", transparent=True,bbox_inches='tight')

#plot of
data_A_increased = U[:,pc_change]*S_A[pc_change]@V.T[pc_change,:];
# cpca_cov = cov_fg - 1.25*cov_bg
# plt.figure();
# plt.imshow(cpca_cov,cmap='bwr')
# plt.title('cPCA cov')
# plt.xlabel('features')
# plt.ylabel('features')

A_inc_cov= data_A_increased.T.dot(data_A_increased);
plt.figure();
plt.imshow(A_inc_cov - np.diag(np.diag(A_inc_cov)),cmap='bwr',clim=(-1,1))
plt.colorbar()
plt.title('increased dimensions cov')
plt.xlabel('Features')
plt.ylabel('Features')
plt.grid(False)
plt.savefig(folder_save_plot+"increased_in_A.pdf", transparent=True)
plt.savefig(folder_save_plot+"increased_in_A.png", transparent=True,bbox_inches='tight')

# gcPCA_mdl = gcPCA(method='v1',normalize_flag=False,alpha=1.25)
gcPCA_mdl = gcPCA(method='v4',normalize_flag=False)
gcPCA_mdl.fit(data_A,data_B)

R_loadings = gcPCA_mdl.loadings_[:,:2].dot(gcPCA_mdl.loadings_[:,:2].T)
gcPCs_cov = cov_A.dot(R_loadings)
plt.figure();
plt.imshow(gcPCs_cov - np.diag(np.diag(gcPCs_cov)),cmap='bwr',clim=(-1,1))
plt.colorbar()
plt.title('top 2 gcPCs')
plt.xlabel('Features')
plt.ylabel('Features')
plt.grid(False)
plt.savefig(folder_save_plot+"gcPC_recovered_cov.pdf", transparent=True)
plt.savefig(folder_save_plot+"gcPC_recovered_cov.png", transparent=True,bbox_inches='tight')
#%% now run cPCA and ncPCA

# cPCs = cPCA(data_bg,data_fg,alpha=1)[:,0]
gcPCA_mdl = gcPCA(method='v1',normalize_flag=False,alpha=1.25)
gcPCA_mdl.fit(data_A,data_B)

cPCs_all = gcPCA_mdl.loadings_


cPC1_cossim = np.abs(cosine_similarity_multiple_vectors(V, cPCs_all[:,0]))
cPC2_cossim = np.abs(cosine_similarity_multiple_vectors(V, cPCs_all[:,1]))

plt.figure()
plt.plot(cPC1_cossim,color='red',label='1st cPC')
plt.plot(cPC2_cossim,'--',color='green',label='2nd cPC')
plt.legend()
plt.xlabel('PCs')
plt.ylabel('Cos. similarity')
plt.title('Modeled PCs captured by cPCA')
plt.tight_layout()


#%% make plot of multiple alpha and the distortion


alphas_vec = np.linspace(0.8,1.6,num=25)
cPC1st = []
cPC2nd = []
cPC1st_allV = []
for a in alphas_vec:
    cPCA_mdl = gcPCA(method='v1',normalize_flag=False,alpha=a)
    cPCA_mdl.fit(data_A,data_B)
    cPCs_all = cPCA_mdl.loadings_
    
    cPCs_cossim = np.abs(cosine_similarity_multiple_vectors(V, cPCs_all[:,0]))
    cPC1st_allV.append(cPCs_cossim)
    cPC1st.append(np.argmax(cPCs_cossim))
    
    cPCs_cossim = np.abs(cosine_similarity_multiple_vectors(V, cPCs_all[:,1]))
    cPC2nd.append(np.argmax(cPCs_cossim))
    
plt.figure()
plt.scatter(alphas_vec,cPC1st,color='red',alpha=0.5,label='1st cPC')
plt.scatter(alphas_vec,cPC2nd,color='green',alpha=0.5,label='2nd cPC')
plt.plot(alphas_vec,np.ones(len(alphas_vec))*pc_change[0],'k--',alpha=0.4,label='dimensions of interest')
plt.plot(alphas_vec,np.ones(len(alphas_vec))*pc_change[1],'k--',alpha=0.4)
plt.xlabel('alpha values')
plt.ylabel('Recovered dim. from model')
plt.legend()
plt.tight_layout()

#%% plot subtracting the eigenspectrum

alphas2use = alphas_vec[np.arange(0,25,5)]
sns.set_style("ticks")
SA = np.tile(S_A,len(alphas_vec))
SA.resize((len(alphas_vec),N_features))

SB = np.tile(S_B,len(alphas_vec))
SB.resize((len(alphas_vec),N_features))

newalpha = np.tile(alphas_vec,N_features)
newalpha.resize((N_features,len(alphas_vec)))


grid2 = plt.GridSpec(1, 2, left=0.2 , wspace = 0.4, hspace=0.05)
plt.figure(figsize=(12,5))
ax = plt.subplot(grid2[0])
plt.plot(np.arange(1,N_features+1),S_A,'-',label='Condition A',c='seagreen')
plt.plot(np.arange(1,N_features+1),S_B,'--',label='Condition B',c='goldenrod')
plt.title('Conditions Eigenspectra')
plt.ylabel('Eigenvalues')
plt.xlabel('Components')
plt.legend()
[ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')]

ax = plt.subplot(grid2[1])
cpc_eq = SA-np.multiply(newalpha.T,SB)
plt.plot(cpc_eq[np.arange(0,25,5),:].T)
plt.grid()
plt.xlabel('Components')
plt.ylabel('A - '+ r'$\alpha$' + '*B')
plt.text(7,0.75,r'$\alpha$' + ' = '+ str(alphas2use[0])[:4],color='tab:blue',rotation=335)
plt.text(7,0.35,r'$\alpha$' + ' = '+ str(alphas2use[1])[:4],color='tab:orange',rotation=-16)
plt.text(7,0.05,r'$\alpha$' + ' = '+ str(alphas2use[2])[:4],color='tab:green',rotation=1)
plt.text(7,-0.45,r'$\alpha$' + ' = '+ str(alphas2use[3])[:4],color='tab:red',rotation=16)
plt.text(7,-0.91,r'$\alpha$' + ' = '+ str(alphas2use[4])[:4],color='tab:purple',rotation=22)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

#%% plot of gcPCA solving the problems


from matplotlib.ticker import StrMethodFormatter
grid1 = plt.GridSpec(1, 2, left=0.2 , wspace = 0.4, hspace=0.05)
plt.figure(figsize=(15,5))
ax=plt.subplot(grid1[0])
plt.plot(np.arange(1,N_features+1),S_A,'-',label='Condition A',c='seagreen')
plt.plot(np.arange(1,N_features+1),S_B,'-',label='Condition B',c='goldenrod')
plt.title('Conditions Eigenspectra')
plt.ylabel('Eigenvalues')
plt.xlabel('Components')
plt.legend()
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

ax=plt.subplot(grid1[1])
plt.plot(np.arange(1,N_features+1),(S_A-S_B)/(S_A+S_B),'-',label='Condition A',c='k')
plt.title('Contrasting (A-B) / (A-B)')
plt.ylabel('(A-B) / (A-B)')
plt.xlabel('Components')
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

# repeat past plot but for covariance
sns.set_style("ticks")
grid1 = plt.GridSpec(1, 3, left=0.2 , wspace = 0.4, hspace=0.05)
plt.figure(figsize=(15,5))
ax=plt.subplot(grid1[0])
plt.imshow(cov_A-np.diag(np.diag(cov_A)),cmap='bwr',clim=(-3,3))
plt.title('Condition A')
plt.ylabel('Features')
plt.xlabel('Features')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')


ax=plt.subplot(grid1[1])
plt.imshow(cov_B-np.diag(np.diag(cov_B)),cmap='bwr',clim=(-3,3))
plt.title('Condition B')
plt.ylabel('Features')
plt.xlabel('Features')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

ax=plt.subplot(grid1[2])
auxplot = cov_A - cov_B;
auxplot = auxplot - np.diag(np.diag(auxplot))
plt.imshow(auxplot,cmap='bwr',clim=(-0.1,0.1))
plt.title('Contrasting A - B')
plt.xlabel('Features')
plt.ylabel('Features')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')