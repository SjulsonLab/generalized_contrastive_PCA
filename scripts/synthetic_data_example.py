# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 17:45:45 2023

@author: fermi

script to generate synthetic data with a small
"""

import sys
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt

repo_dir = "C:\\Users\\fermi\\Documents\\GitHub\\generalized_contrastive_PCA\\" #repository dir
sys.path.append(repo_dir)
from contrastive_methods import gcPCA
plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':150, 'font.size':14})
#%% defining function to return data
def generate_data():
    N_features = 100
    N_samples  = 1000
    factor_noise_div = 100

    #% generating toy data
    #generating background
    t = np.linspace(0,0.5,N_samples)

    latent_factor1 = np.sin(2*np.pi*1*t+np.pi/4) #+ np.random.randn(N_samples)/factor_noise_div;
    latent_factor2 = np.cos(2*np.pi*1*t+np.pi/4) #+ np.random.randn(N_samples)/factor_noise_div;
    latent_factor1 = latent_factor1 / np.linalg.norm(latent_factor1)
    latent_factor2 = latent_factor2 / np.linalg.norm(latent_factor2)

    #variance for the model
    Sm = np.linspace(0,stop=4,num=N_features)[::-1]+10**-4
    Sm[0] = 1.3*Sm[0] #boosting up the first PC for visualization
    Sm[1] = 1.3*Sm[1] #boosting up the first PC for visualization
    #getting orthogonal weights
    W,_,_ = np.linalg.svd(np.random.rand(N_features,N_features),full_matrices=False)

    #getting samples from the different factors in the data, one is scenario 1, the other scenario 2 (left or right for example)
    samples1 = np.outer(latent_factor1,Sm[0]*W[0,:]) + np.outer(latent_factor2,Sm[0]*W[1,:]) #+ np.random.randn(N_samples,N_features)/factor_noise_div #samples for scenario 1
    samples2 = np.outer(latent_factor1,Sm[1]*W[0,:]) + np.outer(latent_factor2,Sm[1]*W[1,:]) #+ np.random.randn(N_samples,N_features)/factor_noise_div #samples for scenario 2


    #all the other pcs with the respective eigenvalue
    rest_factors1 = np.random.randn(N_samples,N_features-2) #times 2 because of the samples previously
    rest_factors2 = np.random.randn(N_samples,N_features-2) #times 2 because of the samples previously
    rest_factors1 = np.divide(rest_factors1,np.linalg.norm(rest_factors1,axis=0)) #normalizing
    rest_factors2 = np.divide(rest_factors2,np.linalg.norm(rest_factors2,axis=0))
    #assigning the variance mag to the weights
    auxSm = np.repeat(Sm[2:,np.newaxis],N_features,axis=1)
    newW=np.multiply(auxSm,W[2:,:])
    rest_samples1 = np.dot(rest_factors1,newW)
    rest_samples2 = np.dot(rest_factors2,newW)

    #concatenating the data
    data_bg = np.concatenate((samples1,samples2),axis=0)+np.concatenate((rest_samples1,rest_samples2),axis=0)
    
    #generating foreground
    latent_factor1 = np.sin(2*np.pi*1*t+np.pi/4) #+ np.random.randn(N_samples)/factor_noise_div;
    latent_factor2 = np.cos(2*np.pi*1*t+np.pi/4) #+ np.random.randn(N_samples)/factor_noise_div;
    latent_factor1 = latent_factor1 / np.linalg.norm(latent_factor1)
    latent_factor2 = latent_factor2 / np.linalg.norm(latent_factor2)
    
    latent_factor3 = t #+ np.random.randn(N_samples)/factor_noise_div;
    # latent_factor4 = np.cos(2*np.pi*1*t+np.pi/4) + np.random.randn(N_samples)/factor_noise_div;
    latent_factor4 = np.exp(t/0.05)
    latent_factor4 = latent_factor4-latent_factor4.mean()
    latent_factor3 = latent_factor3 / np.linalg.norm(latent_factor3)
    latent_factor4 = latent_factor4 / np.linalg.norm(latent_factor4)
    
    #variance for the model
    Sm = np.linspace(0,stop=4,num=N_features)[::-1]+10**-4
    Sm[0] = 1.3*Sm[0] #boosting up the first PC for visualization
    Sm[1] = 1.3*Sm[1] #boosting up the first PC for visualization
    Sm[80] = 2.5*Sm[80] #boosting a pc that has important dynamics
    Sm[81] = 2.4*Sm[81] #boosting a pc that has important dynamics
    #getting orthogonal weights
    # W,_,_ = np.linalg.svd(np.random.rand(N_features,N_features),full_matrices=False)
    
    #getting samples from the different factors in the data, one is scenario 1, the other scenario 2 (left or right for example)
    samples1 = np.outer(latent_factor1,Sm[0]*W[0,:]) + np.outer(latent_factor2,Sm[0]*W[1,:]) #+ np.random.randn(N_samples,N_features)/factor_noise_div #samples for scenario 1
    samples2 = np.outer(latent_factor1,Sm[1]*W[0,:]) + np.outer(latent_factor2,Sm[1]*W[1,:]) #+ np.random.randn(N_samples,N_features)/factor_noise_div #samples for scenario 2
    
    samples3 = np.outer(latent_factor3,Sm[80]*W[80,:]) + np.outer(latent_factor4,Sm[81]*W[81,:]) #+ np.random.randn(N_samples,N_features)/factor_noise_div #samples for scenario 1
    samples4 = np.outer(-1*latent_factor3,Sm[80]*W[80,:]) + np.outer(latent_factor4,Sm[81]*W[81,:]) #+ np.random.randn(N_samples,N_features)/factor_noise_div #samples for scenario 2
    
    
        
    #all the other pcs with the respective eigenvalue
    rest_factors1 = np.random.randn(N_samples,N_features-22)
    rest_factors2 = np.random.randn(N_samples,N_features-22)
    rest_factors1 = np.divide(rest_factors1,np.linalg.norm(rest_factors1,axis=0)) #normalizing
    rest_factors2 = np.divide(rest_factors2,np.linalg.norm(rest_factors2,axis=0))
    rest_factors3 = np.random.randn(N_samples,N_features-82) 
    rest_factors4 = np.random.randn(N_samples,N_features-82) 
    rest_factors3 = np.divide(rest_factors3,np.linalg.norm(rest_factors3,axis=0)) #normalizing
    rest_factors4 = np.divide(rest_factors4,np.linalg.norm(rest_factors4,axis=0))
    #assigning the variance mag to the weights
    auxSm = np.repeat(Sm[2:80,np.newaxis],N_features,axis=1)
    newW=np.multiply(auxSm,W[2:80,:])
    rest_samples1 = np.dot(rest_factors1,newW)
    rest_samples2 = np.dot(rest_factors2,newW)
    
    auxSm = np.repeat(Sm[82:,np.newaxis],N_features,axis=1)
    newW=np.multiply(auxSm,W[82:,:])
    rest_samples3 = np.dot(rest_factors3,newW)
    rest_samples4 = np.dot(rest_factors4,newW)
    
    
    #concatenating the data and zscoring
    data_fg = np.concatenate((samples1+samples3,samples2+samples4),axis=0)+np.concatenate((rest_samples1+rest_samples3,rest_samples2+rest_samples4),axis=0)
    
    return data_bg,data_fg,W
#%%parameters for generating data with 2 tracks

data_bg,data_fg,W = generate_data()
#getting the covariance of the data
cov_data_bg = data_bg.T.dot(data_bg)

# U,S,V = np.linalg.svd(cov_data_bg,full_matrices=False)

#start plotting
fig = plt.figure(num=1)
fig.set_figwidth(17)
grid1 = plt.GridSpec(2, 5,left=0.05,right=0.98,wspace=0.05, hspace=0.4)
#plot of manifold
plt.subplot(grid1[0,0])
plt.text(0.5, 0.5, 'Condition A', 
         horizontalalignment='center',
         verticalalignment='center',
         color='green',
         fontweight='bold',
         fontsize=26)
plt.xticks([])
plt.yticks([])
plt.subplot(grid1[1,0])
plt.text(0.5, 0.5, 'Condition B', 
         horizontalalignment='center',
         verticalalignment='center',
         color='black',
         fontweight='bold',
         fontsize=26)
plt.xticks([])
plt.yticks([])
plt.figtext(0.03, 0.93, 'A', fontsize=22,fontweight='bold')
# plt.axis('off')

#%% plot of synthetic data latent features 
# grid2 = plt.GridSpec(2, 5, wspace=0.05, hspace=0.3)
grid2 = plt.GridSpec(2, 5, left=0.13, right=0.83, wspace=0.2, hspace=0.4)
plot1_xlim = (-0.4, 0.4)
plot1_ylim = (-0.4, 0.4)
#plot of synthetic data latent features on dataset A

ax=plt.subplot(grid2[0,1])
plt.plot(data_fg[:1000,:]@W[0,:].T,data_fg[:1000,:]@W[1,:].T,
         color='blue',
         linewidth=4)
plt.plot(data_fg[1000:,:]@W[0,:].T,data_fg[1000:,:]@W[1,:].T,
         color='red',
         linewidth=4)
plt.xticks([])
plt.yticks([])
plt.title('High variance\nmanifold')
plt.xlabel('dim 1')
plt.ylabel('dim 2')
ax.set_xlim(plot1_xlim)
ax.set_ylim(plot1_ylim)
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
ax.set_aspect(1.0)

#plot of synthetic data latent features on dataset A
ax=plt.subplot(grid2[0,2])
plt.plot(data_fg[:1000,:]@W[80,:].T,data_fg[:1000,:]@W[81,:].T,
         color='blue',
         linewidth=4)
plt.plot(data_fg[1000:,:]@W[80,:].T,data_fg[1000:,:]@W[81,:].T,
         color='red',
         linewidth=4)
plt.xticks([])
plt.yticks([])
plt.title('Low variance\nmanifold')
plt.xlabel('dim 81')
plt.ylabel('dim 82')
ax.set_xlim(plot1_xlim)
ax.set_ylim(plot1_ylim)
ax.set_aspect(1.0)
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.legend() #see about changing labels later


ax=plt.subplot(grid2[1,1])
plt.plot(data_bg[:1000,:]@W[0,:].T,data_bg[:1000,:]@W[1,:].T,
         color='blue',
         linewidth=4)
plt.plot(data_bg[1000:,:]@W[0,:].T,data_bg[1000:,:]@W[1,:].T,
         color='red',
         linewidth=4)
plt.xticks([])
plt.yticks([])
# plt.title('High variance\nmanifold')
plt.xlabel('dim 1')
plt.ylabel('dim 2')
ax.set_xlim(plot1_xlim)
ax.set_ylim(plot1_ylim)
ax.set_aspect(1.0)
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

ax=plt.subplot(grid2[1,2])
plt.plot(data_bg[:1000,:]@W[80,:].T,data_bg[:1000,:]@W[81,:].T,
         color='blue',
         linewidth=4)
plt.plot(data_bg[1000:,:]@W[80,:].T,data_bg[1000:,:]@W[81,:].T,
         color='red',
         linewidth=4)
plt.xticks([])
plt.yticks([])
# plt.title('Low variance\nmanifold')
plt.xlabel('dim 81')
plt.ylabel('dim 82')
ax.set_xlim(plot1_xlim)
ax.set_ylim(plot1_ylim)
ax.set_aspect(1.0)
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.figtext(0.26, 0.93, 'B', fontsize=22, fontweight='bold')
plt.figtext(0.395, 0.69, '+', fontsize=30, fontweight='bold')
plt.figtext(0.395, 0.25, '+', fontsize=30, fontweight='bold')

#%% plot of eigenspectrum of each condition

ax=plt.subplot(grid1[0:2,3])
test=data_bg@W.T
bg_eigspec = np.sqrt(np.diag(test.T@test))
test=data_fg@W.T
fg_eigspec = np.sqrt(np.diag(test.T@test))
x=np.arange(1,101,step=1)
plt.plot(x,fg_eigspec,
         color='green',
         linestyle='dashed',
         label='condition A',
         linewidth=4,
         zorder=2)
plt.plot(x,bg_eigspec,
         color='black',
         label='condition B',
         linewidth=4,
         zorder=1)
plt.xlabel('Dimensions')
plt.ylabel('Magnitude')
plt.legend()
# ax.set_aspect(30)
plt.figtext(0.58, 0.93, 'C', fontsize=22,fontweight='bold')
plt.xticks((1, 21, 41, 61, 81))
plt.grid(color='grey',
         linestyle='-', 
         linewidth=2,
         alpha=0.4,
         zorder=0)

#%%parameters for generating same data as before but also with a clear separation hidden

# ax=plt.subplot(grid[0,1])
# plt.plot(data_fg[:1000,:]@W[80,:].T,data_fg[:1000,:]@W[81,:].T,color='blue')
# plt.plot(data_fg[1000:,:]@W[80,:].T,data_fg[1000:,:]@W[81,:].T,color='red')
# plt.xticks([])
# plt.yticks([])
# plt.title('Low variance\nmanifold')
# plt.xlabel('dim80')
# plt.ylabel('dim81')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
#plt.plot(latent_factor1,latent_factor2)
# plt.plot(data_fg[:1000,:]@V[0,:].T,data_fg[:1000,:]@V[1,:].T);plt.plot(data_fg[1000:,:]@V[0,:].T,data_fg[1000:,:]@V[1,:].T)
# plt.plot(data_2[:1000,:]@V[80,:].T,data_2[:1000,:]@V[81,:].T);plt.plot(data_2[1000:,:]@V[80,:].T,data_2[1000:,:]@V[81,:].T)
# plt.plot(data_fg[:1000,:]@W[80,:].T,data_fg[:1000,:]@W[81,:].T);plt.plot(data_fg[1000:,:]@W[80,:].T,data_fg[1000:,:]@W[81,:].T)
# plt.plot(samples1@V[80,:].T,samples1@V[81,:].T);plt.plot(samples2@V[80,:].T,samples2@V[81,:].T)
#%% plot of covariances of each manifold

# plt.subplot(grid[1,0])
# cov_high_var_manifold = np.linalg.multi_dot((data_fg.T.dot(data_fg),W[:2,:].T,W[:2,:]))
# plt.imshow(cov_high_var_manifold,clim=(-2,2))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('features'),plt.ylabel('features')
# cbr = plt.colorbar(pad=0.05,shrink=0.7)
# cbr.ax.tick_params(labelsize=8)
# plt.title('High variance\nmanifold')

# plt.subplot(grid[1,1])
# cov_low_var_manifold = np.linalg.multi_dot((data_fg.T.dot(data_fg),W[80:81,:].T,W[80:81,:]))
# plt.imshow(cov_low_var_manifold,clim=(-0.15,0.15))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('features'),plt.ylabel('features')
# cbr=plt.colorbar(pad=0.05,shrink=0.7)
# cbr.ax.tick_params(labelsize=8)
# plt.title('Low variance\nmanifold')


#%% fit LDA?
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDA = LinearDiscriminantAnalysis()
labels = np.concatenate((np.zeros(1000),np.ones(1000)),axis=0)
LDA.fit(data_fg,labels)

# ax=plt.subplot(grid[2:4,1:2])
# ax=fig.add_subplot(3,3,8)
# plt.plot(data_fg[:1000,:]@LDA.coef_.T,color='blue')
# plt.plot(data_fg[1000:,:]@LDA.coef_.T,color='red')
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('Samples'),plt.ylabel('LDA dim1')
# plt.title('LDA')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
#%% plot different methods capturing the manifolds
#getting the covariance of the data for PCA
cov_data_fg = data_fg.T.dot(data_fg)
U,S_2,V = np.linalg.svd(cov_data_fg,full_matrices=False)
ax=plt.subplot(grid1[0,4])
plt.plot(data_fg[:1000,:]@V[0,:].T,data_fg[:1000,:]@V[1,:].T,color='blue')
plt.plot(data_fg[1000:,:]@V[0,:].T,data_fg[1000:,:]@V[1,:].T,color='red')
plt.xticks([])
plt.yticks([])
plt.xlabel('dim 1'),plt.ylabel('dim 2')
plt.title('PCA on A')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

#%% test gcPCA
gcPCA_mdl = gcPCA(method='v4',normalize_flag=False)
gcPCA_mdl.fit(data_fg,data_bg)

Vg = gcPCA_mdl.loadings_
ax=plt.subplot(grid1[1,4])
# ax=fig.add_subplot(3,3,9)
plt.plot(data_fg[:1000,:]@Vg[:,0],data_fg[:1000,:]@Vg[:,1],color='blue')
plt.plot(data_fg[1000:,:]@Vg[:,0],data_fg[1000:,:]@Vg[:,1],color='red')
plt.xticks([])
plt.yticks([])
plt.xlabel('dim 1'),plt.ylabel('dim 2')
plt.title('gcPCA on A vs. B')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.show()
plt.figtext(0.81, 0.93, 'D', fontsize=22,fontweight='bold')
fig.savefig('toy_data_high_variance_low_variance_manifold')