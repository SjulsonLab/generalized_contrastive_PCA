#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 22:14:08 2022

Script to create toy data to show use of ncPCA
@author: eliezyer

"""
#%% importing essentials
import os
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import time
from scipy.stats import zscore

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from sklearn import svm
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from matplotlib.pyplot import *
import seaborn as sns

from scipy.stats import zscore
import pickle
from numpy import linalg as LA

#%% parameters
#%% import custom modules
repo_dir = "/home/eliezyer/Documents/github/normalized_contrastive_PCA/" #repository dir
sys.path.append(repo_dir)
from ncPCA import ncPCA
from ncPCA import cPCA
#import ncPCA_project_utils as utils #this is going to be 

#%% create data with high variance data

N = 100; #number of neurons
t = 1000; #number of bins (time)
Nfactors = 2; #having two factors, one is high variance and specific for each scenario]
#weights for the factors low and high variance

#generating random vectors orthogonal to be used

_,_,W = LA.svd(np.random.randn(5,N),full_matrices=False)
W_hv_a = W[0,:] #high variance and only in A
W_hv_b = W[1,:] #high variance and only in B
W_lv_a = W[2,:] #low variance and in both scenarios, higher in A
W_lv_b = W[3,:] #low variance and in both scecnarios, higher in b
W_sv_sh = W[4,:] #same variance, shared

W_hv_a = W_hv_a.reshape(W_hv_a.shape[0],1)
W_hv_b = W_hv_b.reshape(W_hv_b.shape[0],1)
W_lv_a = W_lv_a.reshape(W_lv_a.shape[0],1)
W_lv_b = W_lv_b.reshape(W_lv_b.shape[0],1)
W_sv_sh = W_sv_sh.reshape(W_sv_sh.shape[0],1)

"""
W_hv_a = W_hv_a/np.linalg.norm(W_hv_a)
W_hv_b = W_hv_b/np.linalg.norm(W_hv_b)
W_lv_a = W_lv_a/np.linalg.norm(W_lv_a)
W_lv_b = W_lv_b/np.linalg.norm(W_lv_b)"""



#simulating scenario A
HV_factor = np.random.randn(t,1) #high variance factor
SV_factor = np.random.randn(t,1) #shared variance factor
LV_factor_a = np.random.randn(t,1) #low variance factor
LV_factor_b = np.random.randn(t,1)
tempDa = np.outer(5*HV_factor,W_hv_a) + \
    np.outer(4*SV_factor,W_sv_sh) + \
    np.outer(2*LV_factor_a,W_lv_a) + \
    np.outer(0.1*LV_factor_b,W_lv_b) + \
    2*np.random.randn(t,N)
    
"""REGRESS OUT W_HV_B FROM Da, repeat the same for Db"""
outer_B = np.outer(W_hv_b,W_hv_b.T)
Da = zscore(tempDa - np.dot(tempDa,outer_B))
 
#simulating scenario B
HV_factor = np.random.randn(t,1) #high variance factor
LV_factor_a = np.random.randn(t,1)
LV_factor_b = np.random.randn(t,1)
tempDb = np.outer(5*HV_factor,W_hv_b) + \
    np.outer(4*SV_factor,W_sv_sh) + \
    np.outer(0.1*LV_factor_a,W_lv_a) + \
    np.outer(2*LV_factor_b,W_lv_b) + \
    2*np.random.randn(t,N)

"""this is to make sure there's no variance in that vector inthe other data"""
outer_A = np.outer(W_hv_a,W_hv_a.T)
Db = zscore(tempDb - np.dot(tempDb,outer_A))
#Da = zscore(tempDa)
#Db = zscore(tempDb)
#%% analyzing the simulated data


#turning the plots black

sns.set_style("ticks")
sns.set_context("talk")
style.use('dark_background')

lims_y = (-0.1,1)
ncPCA_mdl = ncPCA(basis_type='intersect',Nshuffle=0)
ncPCA_mdl.fit(Da,Db)
x= ncPCA_mdl.loadings_

figure()
fig, axes = subplots(1, 2, figsize=(10, 5), sharey=True)
subplot(1,2,1)
cos_sim = np.abs(np.dot(x.T,W_lv_b)) #the norm of every vector is 1
plot(cos_sim,label='low_var',color='r')
cos_sim = np.abs(np.dot(x.T,W_hv_b)) #the norm of every vector is 1
plot(cos_sim,label='high_var',color='grey')
cos_sim = np.abs(np.dot(x.T,W_sv_sh)) #the norm of every vector is 1
plot(cos_sim,label='sh_var',color='w');ylim(lims_y)
legend()
ylabel('cos. sim.')
xlabel('ncPCs')
title('state B')
sns.despine()

ax= subplot(1,2,2)
cos_sim = np.abs(np.dot(x.T,W_lv_a)) #the norm of every vector is 1
plot(cos_sim,label='low_var',color='green')
cos_sim = np.abs(np.dot(x.T,W_hv_a)) #the norm of every vector is 1
plot(cos_sim,color='grey')
cos_sim = np.abs(np.dot(x.T,W_sv_sh)) #the norm of every vector is 1
plot(cos_sim,color='w');ylim(lims_y)
legend()
title('state A')
xlabel('ncPCs')
sns.despine() 
suptitle('ncPCA')

#making plot with loadings from PCs in each state
_,_,Va = LA.svd(Da,full_matrices = False)
_,_,Vb = LA.svd(Db,full_matrices = False)

figure()
fig, axes = subplots(1, 2, figsize=(10, 5), sharey=True)
subplot(1,2,1)
x = np.linspace(1,Vb.shape[1],num=Vb.shape[1])
cos_sim = np.abs(np.dot(Vb,W_lv_b)) #the norm of every vector is 1
plot(x,cos_sim,label='low var',color='r')
cos_sim = np.abs(np.dot(Vb,W_hv_b)) #the norm of every vector is 1
plot(x,cos_sim,label='high var',color='grey')
cos_sim = np.abs(np.dot(Vb,W_sv_sh)) #the norm of every vector is 1
plot(x,cos_sim,label='sh var',color='w');ylim(lims_y)
#legend()
ylabel('cos. sim.')
xlabel('PCs')
title('state B')
xscale('log')
sns.despine()


subplot(1,2,2)
x = np.linspace(1,Va.shape[1],num=Va.shape[1])
cos_sim = np.abs(np.dot(Va,W_lv_a)) #the norm of every vector is 1
plot(x,cos_sim,label='low_var',color='green')
cos_sim = np.abs(np.dot(Va,W_hv_a)) #the norm of every vector is 1
plot(x,cos_sim,color='grey')
cos_sim = np.abs(np.dot(Va,W_sv_sh)) #the norm of every vector is 1
plot(x,cos_sim,color='w');ylim(lims_y)
title('state A')
xlabel('PCs')
xscale('log')
suptitle('PCA')
sns.despine()