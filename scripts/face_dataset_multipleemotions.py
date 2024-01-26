# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:46:06 2024

@author: fermi

script to run the gcPCA analysis in the face with happy (C_a) vs stressed 
emotions (C_b) vs PCA on either. I want to expose both top and bottom gcPCs
"""
# libraries
import numpy as np
from scipy.linalg import norm
from scipy.stats import zscore
from scipy.io import loadmat
import copy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sys

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
import seaborn as sns

# repository dir
repo_dir = r'C:\Users\fermi\Documents\GitHub\generalized_contrastive_PCA' #on laptop
sys.path.append(repo_dir)
from contrastive_methods import gcPCA

#%%
data_path = r'C:\Users\fermi\Dropbox\preprocessing_data\gcPCA_files\face\CFD_V3\Images\CFD'
tempmat = loadmat(data_path+r'\all_face_emotions.mat')

data_A = tempmat['all_face_emotions']['data_A'][0][0]  # this is the dataset with happy smiling and closed mouth
data_B = tempmat['all_face_emotions']['data_B'][0][0]  # this is the dataset with fear/angry 
data_N = tempmat['all_face_emotions']['data_N'][0][0]  # this is the neutral faces
labels_A = tempmat['all_face_emotions']['labels_A'][0][0] # 0 is happy smiling and 1 is happy closed mouth
labels_B = tempmat['all_face_emotions']['labels_B'][0][0] # 0 is fear and 1 is angry faces
Mask = tempmat['all_face_emotions']['EllipseMask'][0][0]

A = np.reshape(data_A,(data_A.shape[0]*data_A.shape[1],data_A.shape[2]))
A_zsc = zscore(A)
A_norm = A_zsc/norm(A_zsc,axis=0)

B = np.reshape(data_B,(data_B.shape[0]*data_B.shape[1],data_B.shape[2]))
B_zsc = zscore(B)
B_norm = B_zsc/norm(B_zsc,axis=0)

#%% doing PCA and getting the two first PCs to plot
Ua,_,Va = np.linalg.svd(A_zsc.T,full_matrices=False)
Ub,_,Vb = np.linalg.svd(B_zsc.T,full_matrices=False)

pcs = Va.T
temp1 = pcs[:,0]
image_pc1a = np.reshape(temp1,(data_A.shape[0],data_A.shape[1])).copy()
temp1 = pcs[:,1]
image_pc2a = np.reshape(temp1,(data_A.shape[0],data_A.shape[1])).copy()

pcs = Vb.T
temp1 = pcs[:,0]
image_pc1b = np.reshape(temp1,(data_A.shape[0],data_A.shape[1])).copy()
temp1 = pcs[:,1]
image_pc2b = np.reshape(temp1,(data_A.shape[0],data_A.shape[1])).copy()


#%% doing gcPCA and getting the two first gcPCs
gcPCA_mdl = gcPCA(method='v4',normalize_flag=False)
gcPCA_mdl.fit(A_norm.T,B_norm.T)
U_gcpca_A = gcPCA_mdl.Ra_scores_
U_gcpca_B = gcPCA_mdl.Rb_scores_

gcpcs = gcPCA_mdl.loadings_
temp1 = gcpcs[:,0]
image_gcpc1 = np.reshape(temp1,(data_A.shape[0],data_A.shape[1])).copy()
temp1 = gcpcs[:,1]
image_gcpc2 = np.reshape(temp1,(data_A.shape[0],data_A.shape[1])).copy()

temp1 = gcpcs[:,-1]
image_gcpc_minus1 = np.reshape(temp1,(data_A.shape[0],data_A.shape[1])).copy()
temp1 = gcpcs[:,-2]
image_gcpc_minus2 = np.reshape(temp1,(data_A.shape[0],data_A.shape[1])).copy()

#%%  perform LDA!!!!
#even though I think it's not necessary
# LDA_mdl = LinearDiscriminantAnalysis()
# LDA_mdl.fit(A_norm.T,labels)

# lda_x = LDA_mdl.fit_transform(A_norm.T,labels)

# temp1 = LDA_mdl.coef_
# image_LDA = np.reshape(temp1,(data_A.shape[0],data_A.shape[1])).copy()

# plt.figure()
# m_lim = np.max(np.abs(image_LDA*Mask))
# plt.imshow(image_LDA*Mask)
# plt.title("LDA coeff")
# plt.clim(-1*m_lim,m_lim)
# plt.axis('off')
# plt.scatter(lda_X[lbl_hc,3],U_gcpca[lbl_hc,2],c='seagreen',alpha=0.5)
# plt.scatter(U_gcpca[lbl_hc,0].mean(),U_gcpca[lbl_hc,1].mean(),c='b',alpha=0.7)
# plt.scatter(U_gcpca[lbl_a,3],U_gcpca[lbl_a,2],c='blueviolet',alpha=0.5)
#%% making figures

# plot parameters
m_size = 100  # markersize for scatter
happy_close_example = 5
smile_example = 28+happy_close_example
angry_example = 5
fear_example = 28+angry_example


from matplotlib import colors as clrs
# sns.set_style("whitegrid")
sns.set_style("ticks")
sns.set_context("talk")
# making custom colormap and adding as the default
cmap = clrs.LinearSegmentedColormap.from_list("", ["seagreen","white","blueviolet"])
plt.rcParams['image.cmap'] = cmap
plt.rcParams.update({'figure.dpi':150, 'font.size':24})

# starting plot
fig = plt.figure(num=1, figsize=(26, 12))
grid1 = plt.GridSpec(4, 8,left=-0.1,right=0.95,wspace=0.05, hspace=0.8) # used for scores
grid2 = plt.GridSpec(4, 8,left=0.05,right=0.5,wspace=0.05, hspace=0.8) # use for faces
grid3 = plt.GridSpec(4, 8,left=0.21,right=0.72,wspace=0.001, hspace=0.8) # used for loadings

# % plotting examples from dataset
# angry
plt.subplot(grid2[2:4,0])
temp = data_B[:,:,angry_example].astype(float)*Mask;
temp2  = temp.copy()
angry_face  = temp.copy()
angry_face[(temp2==0.0)]=np.nan
plt.imshow(angry_face,cmap='gray',aspect='auto')
plt.xlim((35,145))
plt.ylim((205,45))
plt.axis('off')
plt.title('Angry')
plt.figtext(0.02, 0.17, 'condition B', fontsize=40, rotation=90, fontweight='bold')


plt.subplot(grid2[2:4,1])
temp = data_B[:,:,fear_example].astype(float)*Mask;
temp2  = temp.copy()
fear_face  = temp.copy()
fear_face[(temp2==0.0)]=np.nan
plt.imshow(fear_face,cmap='gray',aspect='auto')
plt.xlim((35,145))
plt.ylim((205,45))
plt.axis('off')
plt.title('Fear')


# happy closed mouth example
plt.subplot(grid2[0:2,0])
temp = data_A[:,:,happy_close_example].astype(float)*Mask;
temp2  = temp.copy()
happy_face  = temp.copy()
happy_face[(temp2==0.0)]=np.nan
plt.imshow(happy_face,cmap='gray',aspect='auto')
plt.xlim((35,145))
plt.ylim((205,45))
plt.axis('off')
plt.title('Happy')

plt.subplot(grid2[0:2,1])
temp = data_A[:,:,smile_example].astype(float)*Mask;
temp2  = temp.copy()
smile_face  = temp.copy()
smile_face[(temp2==0.0)]=np.nan
plt.imshow(smile_face,cmap='gray',aspect='auto')
plt.xlim((35,145))
plt.ylim((205,45))
plt.axis('off')
plt.title('Smile')
plt.figtext(0.03, 0.93, 'A', fontsize=40, fontweight='bold')
plt.figtext(0.02, 0.60, 'condition A', fontsize=40, rotation=90, fontweight='bold')

# % plotting pc loadings
lbl_hc = np.argwhere(labels_A==0)[:,0]
lbl_smile = np.argwhere(labels_A==1)[:,0]
lbl_angry = np.argwhere(labels_B==0)[:,0]
lbl_fear = np.argwhere(labels_B==1)[:,0]

m_lim = np.max(np.abs(image_pc1a*Mask))/0.7

plt.subplot(grid3[0:2,0])
plt.imshow(image_pc1a*Mask,aspect='auto')
plt.title("PC1")
plt.xlim((35,145))
plt.ylim((205,45))
plt.clim(-1*m_lim,m_lim)
plt.axis('off')

ax=plt.subplot(grid3[0:2,1])
aux_imshow=plt.imshow(image_pc2a*Mask,aspect='auto')
plt.title("PC2")
plt.xlim((35,145))
plt.ylim((205,45))
plt.clim(-1*m_lim,m_lim)
plt.axis('off')
cax = ax.inset_axes([1.04, 0.2, 0.05, 0.6])
cbar = fig.colorbar(aux_imshow, cax=cax)
cbar.set_label('Magnitude', rotation=270, labelpad=18, fontsize=22)

plt.figtext(0.20, 0.93, 'B', fontsize=40, fontweight='bold')
plt.figtext(0.24, 0.915, 'Loadings', fontsize=30)

# plotting the examples for PC on condition B
plt.subplot(grid3[2:4,0])
plt.imshow(image_pc1b*Mask,aspect='auto')
plt.title("PC1")
plt.xlim((35,145))
plt.ylim((205,45))
plt.clim(-1*m_lim,m_lim)
plt.axis('off')

ax=plt.subplot(grid3[2:4,1])
aux_imshow=plt.imshow(image_pc2b*Mask,aspect='auto')
plt.title("PC2")
plt.xlim((35,145))
plt.ylim((205,45))
plt.clim(-1*m_lim,m_lim)
plt.axis('off')
cax = ax.inset_axes([1.04, 0.2, 0.05, 0.6])
cbar = fig.colorbar(aux_imshow, cax=cax)
cbar.set_label('Magnitude', rotation=270, labelpad=18, fontsize=22)
plt.figtext(0.24, 0.48, 'Loadings', fontsize=30)

# plotting the scores of PCs data set A, i.e., each dot is a subject
ax = plt.subplot(grid1[0:2,4])
ax.scatter(Ua[lbl_hc,0],Ua[lbl_hc,1],c='blue',alpha=0.5,label='Happy',s=m_size)
ax.scatter(Ua[lbl_smile,0],Ua[lbl_smile,1],c='red',alpha=0.5,label='Smile',s=m_size)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.legend()
# plt.figtext(0.60, 0.93, 'C', fontsize=40, fontweight='bold')

# plotting the scores of PCs data set B, i.e., each dot is a subject
ax = plt.subplot(grid1[2:4,4])
ax.scatter(Ub[lbl_angry,0],Ub[lbl_angry,1],c='blue',alpha=0.5,label='Angry',s=m_size)
ax.scatter(Ub[lbl_fear,0],Ub[lbl_fear,1],c='red',alpha=0.5,label='Fear',s=m_size)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.legend()

# % plotting gcPC loadings faces dataset A
m_limc = np.max(np.abs(image_gcpc1*Mask))/1.8
plt.subplot(grid3[0:2,6])
plt.imshow(image_gcpc1,aspect='auto')
plt.xlim((35,145))
plt.ylim((205,45))
plt.title("gcPC1")
plt.clim(-1*m_limc, m_limc)
plt.axis('off')

ax = plt.subplot(grid3[0:2,7])
aux_imshow=plt.imshow(image_gcpc2,aspect='auto')
plt.xlim((35,145))
plt.ylim((205,45))
plt.title("gcPC2")
plt.clim(-1*m_limc, m_limc)
plt.axis('off')
cax = ax.inset_axes([1.04, 0.2, 0.05, 0.6])
cbar = fig.colorbar(aux_imshow, cax=cax)
cbar.set_label('Magnitude', rotation=270, labelpad=18, fontsize=22)


# % plotting gcPC loadings faces dataset b
m_limc = np.max(np.abs(image_gcpc_minus1*Mask))/3
plt.subplot(grid3[2:4,6])
plt.imshow(image_gcpc_minus1,aspect='auto')
plt.xlim((35,145))
plt.ylim((205,45))
plt.title("gcPC_last")
plt.clim(-1*m_limc, m_limc)
plt.axis('off')

ax = plt.subplot(grid3[2:4,7])
aux_imshow=plt.imshow(image_gcpc_minus2,aspect='auto')
plt.xlim((35,145))
plt.ylim((205,45))
plt.title("gcPC_last-1")
plt.clim(-1*m_limc, m_limc)
plt.axis('off')
cax = ax.inset_axes([1.04, 0.2, 0.05, 0.6])
cbar = fig.colorbar(aux_imshow, cax=cax)
cbar.set_label('Magnitude', rotation=270, labelpad=18, fontsize=22)

# plotting the scores on gcPCA
ax = plt.subplot(grid1[0:2,7])
ax.scatter(U_gcpca_A[lbl_hc,0],U_gcpca_A[lbl_hc,1],c='blue',alpha=0.5,label='Happy',s=m_size)
ax.scatter(U_gcpca_A[lbl_smile,0],U_gcpca_A[lbl_smile,1],c='red',alpha=0.5,label='Smile',s=m_size)
ax.set_xlabel('gcPC1')
ax.set_ylabel('gcPC2')
ax.legend()
plt.xlim((-0.35,0.35))
plt.ylim((-0.35,0.35))

# dataset B
ax = plt.subplot(grid1[2:4,7])
ax.scatter(U_gcpca_B[lbl_angry,-1],U_gcpca_B[lbl_angry,-2],c='blue',alpha=0.5,label='Angry',s=m_size)
ax.scatter(U_gcpca_B[lbl_fear,-1],U_gcpca_B[lbl_fear,-2],c='red',alpha=0.5,label='Fear',s=m_size)
ax.set_xlabel('gcPC1')
ax.set_ylabel('gcPC2')
ax.legend()
plt.xlim((-0.35,0.35))
plt.ylim((-0.35,0.35))