#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 10:08:07 2022
This file will keep all the functions we use in the 

@author: eliezyer
"""


def cumul_accuracy_projected(Xtrain,Ytrain,Xtest,Ytest,loadings):
    """ X is the predictor, Y is what to be predicted (labels, etc) and loadings is 
    the matrix with loadings to project X into, where rows are the loadings and 
    columns are the dimensions we want to project on.
    
    """
    
    #TO DO: add backward and forward like I did, so you save time in the for loop,
    #the less for loops the better/faster!!!
    
    for dim in np.arange(loadings.shape[1]):
        # project the data
        
        #train SVM in the projected train data
        
        
        #get the score
        
    return scores_forward,scores_backward