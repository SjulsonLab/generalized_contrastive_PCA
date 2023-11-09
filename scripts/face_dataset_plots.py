# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 19:33:51 2023

@author: fermi

script to run the gcPCA analysis in the face with emotions vs neutral vs PCA on
face with emotions
"""



#libraries
import numpy as np
from scipy.linalg import norm
from scipy.stats import zscore
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

repo_dir = "C:\\Users\\fermi\\Documents\\GitHub\\normalized_contrastive_PCA\\" #repository dir
sys.path.append(repo_dir)
from contrastive_methods import gcPCA
from ncPCA_project_utils import cosine_similarity_multiple_vectors

#%% importing keras to preprocess the images

# from tensorflow import convert_to_tensor
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.image import grayscale_to_rgb
# model = VGG16(weights='imagenet', include_top=False)
#%% setting plot parameters

# sns.set_style("whitegrid")
sns.set_style("ticks")
sns.set_context("talk")
plt.rcParams['image.cmap'] = 'bwr'


#%%
data_path = "C:\\Users\\fermi\\Dropbox\\preprocessing_data\\gcPCA_files\\face\\CFD_V3\\Images\\CFD\\"
tempmat = loadmat(data_path+"face_emotions.mat")

data_A = tempmat['face_emotions']['data_A'][0][0]
data_B = tempmat['face_emotions']['data_B'][0][0]
labels = tempmat['face_emotions']['labels'][0][0]
Mask = tempmat['face_emotions']['EllipseMask'][0][0]

A = np.reshape(data_A,(data_A.shape[0]*data_A.shape[1],data_A.shape[2]))
A_zsc = zscore(A)
A_norm = A_zsc/norm(A_zsc,axis=0)

B = np.reshape(data_B,(data_B.shape[0]*data_B.shape[1],data_B.shape[2]))
B_zsc = zscore(B)
B_norm = B_zsc/norm(B_zsc,axis=0)

#%% extracting features with vgg16
# A_feature_list = [];
# for idx in np.arange(data_A.shape[2]):
        
        
#         img_data = image.img_to_array(data_A[:,:,idx])
#         img_data = image.smart_resize(img_data,(224,224))
#         img_data = grayscale_to_rgb(convert_to_tensor(img_data))
#         img_data = np.expand_dims(img_data, axis=0)

#         vgg16_feature = model.predict(img_data)
#         vgg16_feature_np = np.array(vgg16_feature)
#         A_feature_list.append(vgg16_feature_np.flatten())
        
# A_feature_list_np = np.array(A_feature_list)

# B_feature_list = [];
# for idx in np.arange(data_B.shape[2]):
        
        
#         img_data = image.img_to_array(data_B[:,:,idx])
#         img_data = image.smart_resize(img_data,(224,224))
#         img_data = grayscale_to_rgb(convert_to_tensor(img_data))
#         img_data = np.expand_dims(img_data, axis=0)

#         vgg16_feature = model.predict(img_data)
#         vgg16_feature_np = np.array(vgg16_feature)
#         B_feature_list.append(vgg16_feature_np.flatten())
        
# B_feature_list_np = np.array(B_feature_list)
#%% doing PCA and getting the two first PCs to plot
U,S,V = np.linalg.svd(A_zsc.T,full_matrices=False)

pcs = V.T
temp1 = pcs[:,0]
image_pc1 = np.reshape(temp1,(data_A.shape[0],data_A.shape[1])).copy()
temp1 = pcs[:,1]
image_pc2 = np.reshape(temp1,(data_A.shape[0],data_A.shape[1])).copy()
temp1 = pcs[:,2]
image_pc3 = np.reshape(temp1,(data_A.shape[0],data_A.shape[1])).copy()


#%% doing gcPCA and getting the two first gcPCs
gcPCA_mdl = gcPCA(method='v4',normalize_flag=False)
gcPCA_mdl.fit(A_norm.T,B_norm.T)
U_gcpca = gcPCA_mdl.Ra_scores_

cpcs = gcPCA_mdl.loadings_
temp1 = cpcs[:,0]
image_cpc1 = np.reshape(temp1,(data_A.shape[0],data_A.shape[1])).copy()
temp1 = cpcs[:,1]
image_cpc2 = np.reshape(temp1,(data_A.shape[0],data_A.shape[1])).copy()
temp1 = cpcs[:,2]
image_cpc3 = np.reshape(temp1,(data_A.shape[0],data_A.shape[1])).copy()

#%%  perform LDA!!!!
#even though I think it's not necessary
LDA_mdl = LinearDiscriminantAnalysis()
LDA_mdl.fit(A_norm.T,labels)

lda_x = LDA_mdl.fit_transform(A_norm.T,labels)

temp1 = LDA_mdl.coef_
image_LDA = np.reshape(temp1,(data_A.shape[0],data_A.shape[1])).copy()

plt.figure()
m_lim = np.max(np.abs(image_LDA*Mask))
plt.imshow(image_LDA*Mask)
plt.title("LDA coeff")
plt.clim(-1*m_lim,m_lim)
plt.axis('off')
# plt.scatter(lda_X[lbl_hc,3],U_gcpca[lbl_hc,2],c='seagreen',alpha=0.5)
# plt.scatter(U_gcpca[lbl_hc,0].mean(),U_gcpca[lbl_hc,1].mean(),c='b',alpha=0.7)
# plt.scatter(U_gcpca[lbl_a,3],U_gcpca[lbl_a,2],c='blueviolet',alpha=0.5)
#%% making figures

"""
#todo:
    [ ] PLOT FACE OF PEOPLE ANGRY AND SMILING IN GREY SCALE
    [ ] PLOT THE PROJECTION OF THE DATA IN THESE PCS, FOR SHOWING THEY ARE MORE SEPARABLE"""

lbl_hc = np.argwhere(labels==0)
lbl_a = np.argwhere(labels==1)
m_lim = np.max(np.abs(image_pc1*Mask))

plt.figure()
plt.subplot(131)
plt.imshow(image_pc1*Mask)
plt.title("PC1")
plt.clim(-1*m_lim,m_lim)
# patch = patches.Circle((10, 5), radius=7)
# im.set_clip_path(Mask)
plt.axis('off')
plt.subplot(132)
plt.imshow(image_pc2*Mask)
plt.title("PC2")
plt.clim(-1*m_lim,m_lim)
plt.axis('off')
plt.subplot(133)
plt.imshow(image_pc3*Mask)
plt.title("PC3")
plt.clim(-1*m_lim,m_lim)
plt.axis('off')
# plt.scatter(U[lbl_hc,0],U[lbl_hc,1],c='b',alpha=0.2)
# plt.scatter(U[lbl_a,0],U[lbl_a,1],c='r',alpha=0.2)
# plt.title("Top PCs")

m_limc = np.max(np.abs(image_cpc1*Mask))/2
plt.figure()
plt.subplot(131)
plt.imshow(image_cpc1)
plt.title("gcPC1")
plt.clim(-1*m_limc,m_limc)
plt.axis('off')
plt.subplot(132)
plt.imshow(image_cpc2)
plt.title("gcPC2")
plt.clim(-1*m_limc,m_limc)
plt.axis('off')
plt.subplot(133)
plt.imshow(image_cpc3)
plt.title("gcPC3")
plt.clim(-1*m_limc,m_limc)
plt.axis('off')

#%% PLOTS OF THE PROCTIONS

plt.figure()
plt.scatter(U[lbl_hc,0],U[lbl_hc,1],c='seagreen',alpha=0.5,label='Happy')
plt.scatter(U[lbl_a,0],U[lbl_a,1],c='blueviolet',alpha=0.5,label='Angry')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.legend()

plt.figure()
plt.scatter(U[lbl_hc,1],U[lbl_hc,2],c='seagreen',alpha=0.5,label='Happy')
plt.scatter(U[lbl_a,1],U[lbl_a,2],c='blueviolet',alpha=0.5,label='Angry')
plt.xlabel('PC2')
plt.ylabel('PC3')
plt.tight_layout()
plt.legend()


plt.figure()
plt.scatter(U_gcpca[lbl_hc,0],U_gcpca[lbl_hc,1],c='seagreen',alpha=0.5,label='Happy')
# plt.scatter(U_gcpca[lbl_hc,0].mean(),U_gcpca[lbl_hc,1].mean(),c='b',alpha=0.7)
plt.scatter(U_gcpca[lbl_a,0],U_gcpca[lbl_a,1],c='blueviolet',alpha=0.5,label='Angry')
# plt.scatter(U_gcpca[lbl_a,0].mean(),U_gcpca[lbl_a,1].mean(),c='r',alpha=0.7)
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
plt.tight_layout()
plt.legend()

plt.figure()
plt.scatter(U_gcpca[lbl_hc,1],U_gcpca[lbl_hc,2],c='seagreen',alpha=0.5,label='Happy')
# plt.scatter(U_gcpca[lbl_hc,0].mean(),U_gcpca[lbl_hc,1].mean(),c='b',alpha=0.7)
plt.scatter(U_gcpca[lbl_a,1],U_gcpca[lbl_a,2],c='blueviolet',alpha=0.5,label='Angry')
# plt.scatter(U_gcpca[lbl_a,0].mean(),U_gcpca[lbl_a,1].mean(),c='r',alpha=0.7)
plt.xlabel('gcPC2')
plt.ylabel('gcPC3')
plt.tight_layout()
plt.legend()

#%% example of faces
plt.figure()
plt.subplot(121)
temp = data_A[:,:,5]*Mask;
temp2  = temp.copy()
happy_face  = temp.copy()
happy_face[(temp2==0.0)]=255
plt.imshow(happy_face,cmap='gray')
plt.axis('off')
plt.title('Happy')

plt.subplot(122)
temp = data_A[:,:,73]*Mask;
temp2  = temp.copy()
angry_face  = temp.copy()
angry_face[(temp2==0.0)]=255
plt.imshow(angry_face,cmap='gray')
plt.axis('off')
plt.title('Angry')

plt.figure()
plt.subplot(121)
temp = data_B[:,:,5]*Mask;
temp2  = temp.copy()
happy_face  = temp.copy()
happy_face[(temp2==0.0)]=255
plt.imshow(happy_face,cmap='gray')
plt.axis('off')
plt.title('Neutral')

#%% gcPCA on vgg16 feature extraction
gcPCA_mdl = gcPCA(method='v4',normalize_flag=True)
gcPCA_mdl.fit(A_feature_list_np,B_feature_list_np)


lbl_hc = np.argwhere(labels==0)
lbl_a = np.argwhere(labels==1)
U_gcpca = gcPCA_mdl.Ra_scores_;
plt.figure()
plt.scatter(U_gcpca[lbl_hc,0],U_gcpca[lbl_hc,1],c='seagreen',alpha=0.5,label='Happy')
# plt.scatter(U_gcpca[lbl_hc,0].mean(),U_gcpca[lbl_hc,1].mean(),c='b',alpha=0.7)
plt.scatter(U_gcpca[lbl_a,0],U_gcpca[lbl_a,1],c='blueviolet',alpha=0.5,label='Angry')
# plt.scatter(U_gcpca[lbl_a,0].mean(),U_gcpca[lbl_a,1].mean(),c='r',alpha=0.7)
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
plt.tight_layout()
plt.legend()

#%% pca only
U,S,V = np.linalg.svd(A_feature_list_np,full_matrices=False)

plt.figure()
plt.scatter(U[lbl_hc,0],U[lbl_hc,1],c='seagreen',alpha=0.5,label='Happy')
# plt.scatter(U_gcpca[lbl_hc,0].mean(),U_gcpca[lbl_hc,1].mean(),c='b',alpha=0.7)
plt.scatter(U[lbl_a,0],U[lbl_a,1],c='blueviolet',alpha=0.5,label='Angry')
# plt.scatter(U_gcpca[lbl_a,0].mean(),U_gcpca[lbl_a,1].mean(),c='r',alpha=0.7)
plt.xlabel('gcPC1')
plt.ylabel('gcPC2')
plt.tight_layout()
plt.legend()


#%% TEST FOR ONLY HAPPY AND THE DIVERSITY OF EXPRESSIONS
#the goal here is to test if gcPCA on only the happy faces will expose how different people smile
#or if it will separate in women/men

lbl_hc = np.argwhere(labels==0)

data_A_reduced = data_A[:,:,lbl_hc[:,0]]
A = np.reshape(data_A_reduced,(data_A_reduced.shape[0]*data_A_reduced.shape[1],data_A_reduced.shape[2]))
A_zsc = zscore(A)
A_norm = A_zsc/norm(A_zsc,axis=0)

B = np.reshape(data_B,(data_B.shape[0]*data_B.shape[1],data_B.shape[2]))
B_zsc = zscore(B)
B_norm = B_zsc/norm(B_zsc,axis=0)


gcPCA_mdl = gcPCA(method='v4',normalize_flag=False)
gcPCA_mdl.fit(A_norm.T,B_norm.T)
U_gcpca = gcPCA_mdl.Ra_scores_

#picture of first gcpc
cpcs = gcPCA_mdl.loadings_
temp1 = cpcs[:,0]
image_cpc1 = np.reshape(temp1,(data_A.shape[0],data_A.shape[1])).copy()

gpc_sort = np.argsort(U_gcpca[:,0])

data_A_norm = np.reshape(A_norm,(data_A.shape[0],data_A.shape[1],data_A_reduced.shape[2])).copy()
sorted_faces = data_A_norm[:,:,gpc_sort]