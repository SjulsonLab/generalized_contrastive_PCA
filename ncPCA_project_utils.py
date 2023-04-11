#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 10:08:07 2022
This file will keep all the functions we use in the 

@author: eliezyer
"""
import numpy as np
from sklearn import svm,linear_model

def cumul_accuracy_projected(Xtrain,Ytrain,Xtest,Ytest,loadings, analysis='fw',step_size = 5):
   #    """ X is the predictor, Y is what to be predicted (labels, etc) and loadings is
   #    the matrix with loadings to project X into, where rows are the loadings and
   #    columns are the dimensions we want to project on.
   #
   #    """
   """
   TODO: [] add description of function, []add description of inputs, [] add description of outputs
   
   """

   if analysis=="both":
       loadings_flip = np.flip(loadings, axis=1)

   fw = []
   bw = []

   for dim in np.arange(loadings.shape[1],step=step_size):
        fw_train = np.dot(Xtrain,loadings[:,:dim+1])
        fw_test = np.dot(Xtest,loadings[:,:dim+1])
        clf_fw = svm.SVC().fit(fw_train, Ytrain) # fitting
        clf_fw_score = clf_fw.score(fw_test,Ytest) # predicton
        fw.append(clf_fw_score)

        if analysis=="both":
            bw_train= np.dot(Xtrain,loadings_flip[:,:dim+1])
            bw_test= np.dot(Xtest,loadings_flip[:,:dim+1])
            clf_bw = svm.LinearSVC().fit(bw_train, Ytrain) # fitting
            clf_bw_score = clf_bw.score(bw_test, Ytest) # prediction
            bw.append(clf_bw_score)

   x = np.arange(loadings.shape[1],step=step_size)
   fw = np.hstack(fw)
   if analysis=="both":
       bw = np.hstack(bw)
       return x,fw,bw
   else:
       return x,fw
   
def cumul_error_projected(Xtrain,Ytrain,Xtest,Ytest,loadings, analysis='fw',step_size = 5):
   #    """ X is the predictor, Y is what to be predicted (labels, etc) and loadings is
   #    the matrix with loadings to project X into, where rows are the loadings and
   #    columns are the dimensions we want to project on.
   #
   #    """
   """
   TODO: [] add description of function, []add description of inputs, [] add description of outputs
   
   """

   if analysis=="both":
       loadings_flip = np.flip(loadings, axis=1)

   fw = []
   bw = []

   for dim in np.arange(loadings.shape[1],step=step_size):
        fw_train = np.dot(Xtrain,loadings[:,:dim+1])
        fw_test = np.dot(Xtest,loadings[:,:dim+1])
        clf_fw = linear_model.LinearRegression().fit(fw_train, Ytrain) # fitting
        Yhat = clf_fw.predict(fw_test) # predicton
        fw_score = np.median(Ytest-Yhat)
        fw.append(fw_score)

        if analysis=="both":
            bw_train= np.dot(Xtrain,loadings_flip[:,:dim+1])
            bw_test= np.dot(Xtest,loadings_flip[:,:dim+1])
            clf_bw = linear_model.LinearRegression().fit(bw_train, Ytrain) # fitting
            Yhat = clf_bw.predict(bw_test) # prediction
            bw_score =  np.median(Ytest-Yhat)
            bw.append(bw_score)

   x = np.arange(loadings.shape[1],step=step_size)
   fw = np.hstack(fw)
   if analysis=="both":
       bw = np.hstack(bw)
       return x,fw,bw
   else:
       return x,fw
   
def cosine_similarity_multiple_vectors(A,B):
    """A and B are vectors or matrices where we calculate 
    the cosine similarity, the loadings are in the first index 
    and dimensions in the second index"""
    from scipy import spatial
    
    #preparing input in case is one dimensional
    if A.ndim == 1:
        A = A.reshape(A.shape[0],1)
    if B.ndim == 1:
        B = B.reshape(B.shape[0],1)
    
    cos_sim = np.zeros((A.shape[1],B.shape[1]))
    
    #checking how many vectors are there to test
    if A.ndim>1:
        num_a = A.shape[1]
    else:
        num_a = 1    
    
    if B.ndim>1:
        num_b = B.shape[1]
    else:
        num_b = 1
    
    for a in range(num_a):
        x = A[:,a]
        for b in range(num_b):
            y = B[:,b]
            cos_sim[a,b] = 1 - spatial.distance.cosine(x,y)
            
    return cos_sim

