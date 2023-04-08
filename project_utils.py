# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 21:41:39 2023

@author: fermi

sets of functions to use in this project
"""
import numpy as np

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