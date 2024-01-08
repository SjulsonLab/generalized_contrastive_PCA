#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 13:22:51 2022

Script to find the best alpha to maximize the separation of classes in the DS 
mouse model and protein expression dataset.

@author: eliezyer
"""

#%% importing packages

import os
import numpy as np
from scipy import optimize
from scipy.stats import zscore
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from matplotlib.colors import ListedColormap
import seaborn as sns

#%%
repo_dir = "/home/eliezyer/Documents/github/normalized_contrastive_PCA/" #repository dir
data_dir = "/home/eliezyer/Documents/github/normalized_contrastive_PCA/datasets/from_cPCA_paper/"
sys.path.append(repo_dir)
from gcPCA import gcPCA
from gcPCA import cPCA
#%%
#define an accuracy function to be minimized
cvfold = StratifiedKFold(shuffle=True)
def cPCA_accuracy(Xbg,Xtg,labels,alpha):
    #X is the full data, preprocessed for cPCA
    #Xtg is the X target and Xbg is the background
    
    #parameters
    
    # start the cv
    accuracy = []
    for train_ind,test_ind in cvfold.split(Xtg,labels):
        #model to use
        clf = LinearSVC()
        #get train and test sets
        #train_ind_tg = train_ind.reshape(train_ind.shape[0],1)
        #test_ind_tg  = test_ind.reshape(test_ind.shape[0],1)
        
        
        #separate train and test set of both datasets
        Xtg_train,Xtg_test = zscore(Xtg[train_ind,:]),zscore(Xtg[test_ind,:])#
        labels_train,labels_test = labels[train_ind],labels[test_ind]
        #Xbg_train,Xbg_test = zscore(Xbg[train_ind_bg,:]),zscore(Xbg[test_ind_bg,:])#
        #get cPCA transformed Xtarget
        
        cPCs = cPCA(Xbg,Xtg_train,alpha=alpha)
        Xtg_train_proj = np.dot(Xtg_train,cPCs)#projcted X target data
        Xtg_test_proj = np.dot(Xtg_test,cPCs)#projcted X target data
        
        #fitting and predicting
        clf.fit(Xtg_train_proj,labels_train)
        accuracy.append(clf.score(Xtg_test_proj, labels_test))
    
    accuracy = -1*(np.hstack(accuracy))
    return accuracy

def ncPCA_accuracy(Xbg,Xtg,labels):
    #X is the full data, preprocessed for cPCA
    #Xtg is the X target and Xbg is the background
    
    #parameters
    n_components=2
    # start the cv
    accuracy = []
    for train_ind,test_ind in cvfold.split(Xtg,labels):
        #model to use
        clf = LinearSVC()
        #get train and test sets
        train_ind_tg = train_ind.reshape(train_ind.shape[0],1)
        test_ind_tg  = test_ind.reshape(test_ind.shape[0],1)
        
        #separate train and test set of both datasets
        Xtg_train,Xtg_test = zscore(Xtg[train_ind,:]),zscore(Xtg[test_ind,:])#
        #Xbg_train,Xbg_test = zscore(Xbg[train_ind_bg,:]),zscore(Xbg[test_ind_bg,:])#
        #get cPCA transformed Xtarget
        
        
        #get ncPCA transformed Xtarget
        ncPCA_mdl = ncPCA(basis_type='intersect',Nshuffle=0)
        ncPCA_mdl.fit(Xbg,Xtg_train)
        ncPCs = ncPCA_mdl.loadings_[:,:n_components]
        Xtg_train_proj = np.dot(Xtg_train,ncPCs)#projcted X target data
        Xtg_test_proj = np.dot(Xtg_test,ncPCs)#projcted X target data
        
        #fitting and predicting
        clf.fit(Xtg_train_proj,labels[train_ind])
        accuracy.append(clf.score(Xtg_test_proj, labels[test_ind]))
        
    accuracy = np.hstack(accuracy)
    return accuracy
#%% plot to test the cPCA
#figure()
#subplot(1,2,1)
#scatter(Xtg_train_proj[np.where(labels_train==0),0],Xtg_train_proj[np.where(labels_train==0),1],color='w')
#scatter(Xtg_train_proj[np.where(labels_train==1),0],Xtg_train_proj[np.where(labels_train==1),1],color='r')
#subplot(1,2,2)
#scatter(Xtg_test_proj[np.where(labels_test==0),0],Xtg_test_proj[np.where(labels_test==0),1],color='w')
#scatter(Xtg_test_proj[np.where(labels_test==1),0],Xtg_test_proj[np.where(labels_test==1),1],color='r')
#%% loading data

#going to the repo directory
os.chdir(repo_dir)


data = np.genfromtxt(os.path.join(data_dir,'Data_Cortex_Nuclear.csv'),
                     delimiter=',', skip_header=1,usecols=range(1,78),filling_values=0)
classes = np.genfromtxt(os.path.join(data_dir,'Data_Cortex_Nuclear.csv'),delimiter=',',
                        skip_header =1,usecols=range(78,81),dtype=None,invalid_raise=False)

#%% preparing data for analysis

# identifying which mice to use
target_idx_A = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))[0]
target_idx_B = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Ts65Dn'))[0]


sub_group_labels = len(target_idx_A)*[0] + len(target_idx_B)*[1] # for identification on plots

target_idx = np.concatenate((target_idx_A,target_idx_B))

target = data[target_idx]
target = (target-np.mean(target,axis=0)) / np.std(target,axis=0) # standardize the dataset

background_idx = np.where((classes[:,-1]==b'C/S') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))
background = data[background_idx]
background = (background-np.mean(background,axis=0)) / np.std(background,axis=0) # standardize the dataset

labels = np.array(sub_group_labels)
#%%
#use scipy optimize fmin to minimize it
def cPCA_accuracy_loaded(alpha):
    neg_acc = cPCA_accuracy(background,target,labels,alpha)
    return neg_acc.mean()

#test = optimize.fmin(cPCA_accuracy_loaded,x0=0,xtol=1e-9,ftol=1e-6)
test = optimize.minimize(cPCA_accuracy_loaded,x0=7,method='Nelder-Mead')
#test = optimize.basinhopping(cPCA_accuracy_loaded,x0=7)

