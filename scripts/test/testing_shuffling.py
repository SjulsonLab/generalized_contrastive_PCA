# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:50:40 2023

@author: fermi
"""

#testing the gcPCA
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import sys

A = np.random.randn(1000,40)
B = np.random.randn(1000,40)

for a in np.arange(1000,step=10):
    A[a,:21] = 1

# Azsc = zscore(A)
# Bzsc = zscore(B)

repo_dir = "C:\\Users\\fermi\\Documents\\GitHub\\generalized_contrastive_PCA\\" #repository dir
sys.path.append(repo_dir)
from contrastive_methods import gcPCA

gcPCA_mdl = gcPCA(method='v4',Nshuffle=1000)
gcPCA_mdl.fit(A,B)