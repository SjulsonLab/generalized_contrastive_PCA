#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 10:08:07 2022
This file will keep all the functions we use in the 

@author: eliezyer
"""
import numpy as np
from sklearn import svm

def cumul_accuracy_projected(Xtrain,Ytrain,Xtest,Ytest,loadings, analysis='fw'):
    #    """ X is the predictor, Y is what to be predicted (labels, etc) and loadings is
    #    the matrix with loadings to project X into, where rows are the loadings and
    #    columns are the dimensions we want to project on.
    #
    #    """


   stepSize = 5  # number of PCs to jump in the loop, too big may overlap the results and too low may make the code run very slow

   if analysis== "both":
       loadings_flip = np.flip(loadings, axis=1)

   fw = []
   bw = []

   for dim in np.arange(loadings.shape[1],step=stepSize):
        fw_train = np.dot(Xtrain,loadings[:,:dim+1])
        fw_test = np.dot(Xtest,loadings[:,:dim+1])
        clf_fw = svm.SVC().fit(fw_train, Ytrain) # fitting
        clf_fw_score = clf_fw.score(fw_test,Ytest) # predicton
        fw.append(clf_fw_score)

        if loadings_flip:
            bw_train= np.dot(Xtrain,loadings_flip[:,:dim+1])
            bw_test= np.dot(Xtest,loadings_flip[:,:dim+1])
            clf_fw = svm.SVC().fit(bw_train, Ytrain) # fitting
            clf_fw_score = clf_fw.score(bw_test, Ytest) # prediction
            bw.append(clf_fw_score)

   fw = np.hstack(fw)
   if loadings_flip:
       bw = np.hstack(bw)
       return fw,bw
   else:
       return fw

