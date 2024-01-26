# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 19:33:51 2023

@author: fermi

script to run the gcPCA analysis in the face with emotions vs neutral vs PCA on
face with emotions
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
repo_dir = r'C:\Users\fermi\Documents\GitHub\generalized_contrastive_PCA'
sys.path.append(repo_dir)
from contrastive_methods import gcPCA

#%%
data_path = r'C:\Users\fermi\Dropbox\preprocessing_data\gcPCA_files\face\CFD_V3\Images\CFD'
tempmat = loadmat(data_path+r'\face_emotions.mat')

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
neutral_example = 5
hc_example = 5
angry_example = 65+hc_example

from matplotlib import colors as clrs
# sns.set_style("whitegrid")
sns.set_style("ticks")
sns.set_context("talk")
# making custom colormap and adding as the default
cmap = clrs.LinearSegmentedColormap.from_list("", ["seagreen","white","blueviolet"])
plt.rcParams['image.cmap'] = cmap
plt.rcParams.update({'figure.dpi':150, 'font.size':24})

# starting plot
fig = plt.figure(num=1)
fig.set_figwidth(22)
fig.set_figheight(12)
grid1 = plt.GridSpec(4, 4,left=-0.3,right=0.95,wspace=0.05, hspace=0.8)
grid2 = plt.GridSpec(4, 4,left=0.05,right=0.5,wspace=0.05, hspace=0.8)
grid3 = plt.GridSpec(4, 4,left=0.22,right=0.60,wspace=0.001, hspace=0.8)

# % plotting examples from dataset
plt.subplot(grid2[2:4,1])
temp = data_B[:,:,neutral_example].astype(float)*Mask;
temp2  = temp.copy()
happy_face  = temp.copy()
happy_face[(temp2==0.0)]=np.nan
plt.imshow(happy_face,cmap='gray',aspect='auto')
plt.xlim((35,145))
plt.ylim((205,45))
plt.axis('off')
plt.title('Neutral')
plt.figtext(0.02, 0.17, 'condition B', fontsize=40, rotation=90, fontweight='bold')

plt.subplot(grid2[0:2,0])
temp = data_A[:,:,hc_example].astype(float)*Mask;
temp2  = temp.copy()
happy_face  = temp.copy()
happy_face[(temp2==0.0)]=np.nan
plt.imshow(happy_face,cmap='gray',aspect='auto')
plt.xlim((35,145))
plt.ylim((205,45))
plt.axis('off')
plt.title('Happy')

plt.subplot(grid2[0:2,1])
temp = data_A[:,:,angry_example].astype(float)*Mask;
temp2  = temp.copy()
angry_face  = temp.copy()
angry_face[(temp2==0.0)]=np.nan
plt.imshow(angry_face,cmap='gray',aspect='auto')
plt.xlim((35,145))
plt.ylim((205,45))
plt.axis('off')
plt.title('Angry')
plt.figtext(0.03, 0.93, 'A', fontsize=40, fontweight='bold')
plt.figtext(0.02, 0.60, 'condition A', fontsize=40, rotation=90, fontweight='bold')

# % plotting pc projection
lbl_hc = np.argwhere(labels==0)[:,0]
lbl_a = np.argwhere(labels==1)[:,0]
m_lim = np.max(np.abs(image_pc1*Mask))/0.7

plt.subplot(grid3[0:2,1])
plt.imshow(image_pc1*Mask,aspect='auto')
plt.title("PC1")
plt.xlim((35,145))
plt.ylim((205,45))
plt.clim(-1*m_lim,m_lim)
plt.axis('off')

# ax1 = fig.add_subplot()
# auxp = ax1.imshow(image_pc2*Mask,aspect='auto')
# ax1.set_title("PC2")
# ax1.set_xlim((35,145))
# ax1.set_ylim((205,45))
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# ax1 = fig.add_subplot(grid3[0:2,2])
# divider = make_axes_locatable(ax1)
# cax = divider.append_axes('right', size="0.1%", pad=-0.5)
# auxp = ax1.imshow(image_pc2*Mask,aspect='auto')
# cbar = fig.colorbar(auxp, cax=cax);
# ax1.set_title("PC2")
# ax1.set_xlim((35,145))
# ax1.set_ylim((205,45))

ax = plt.subplot(grid3[0:2,2])
auxp=plt.imshow(image_pc2*Mask,aspect='auto')
plt.title("PC2")
plt.xlim((35,145))
plt.ylim((205,45))
plt.clim(-1*m_lim,m_lim)
cax = ax.inset_axes([1.04, 0.2, 0.05, 0.6])
#legend colorbar
cbar = fig.colorbar(auxp, cax=cax)
cbar.set_label('Magnitude', rotation=270, labelpad=18, fontsize=22)
plt.axis('off')


plt.figtext(0.30, 0.93, 'B', fontsize=40, fontweight='bold')
plt.figtext(0.37, 0.915, 'Loadings', fontsize=30)

# plotting the examples

# % plotting gcPC projection
m_limc = np.max(np.abs(image_gcpc1*Mask))/1.8
plt.subplot(grid3[2:4,1])
plt.imshow(image_gcpc1,aspect='auto')
plt.xlim((35,145))
plt.ylim((205,45))
plt.title("gcPC1")
plt.clim(-1*m_limc, m_limc)
plt.axis('off')

ax2 = plt.subplot(grid3[2:4,2])
auxp=plt.imshow(image_gcpc2,aspect='auto')
plt.xlim((35,145))
plt.ylim((205,45))
plt.title("gcPC2")
plt.clim(-1*m_limc, m_limc)
cax = ax2.inset_axes([1.04, 0.2, 0.05, 0.6])
# #legend colorbar
cbar = fig.colorbar(auxp, cax=cax)
cbar.set_label('Magnitude', rotation=270, labelpad=18, fontsize=22)
plt.axis('off')
plt.figtext(0.30, 0.46, 'D', fontsize=40, fontweight='bold')
plt.figtext(0.37, 0.475, 'Loadings', fontsize=30)
# plt.subplot(grid3[2:4,3])
# plt.imshow(image_gcpc3)
# plt.title("gcPC3")
# plt.clim(-1*m_limc,m_limc)
# plt.axis('off')

# % plots of the projections

# ax = fig.add_subplot(grid1[0,3],projection='3d')
# ax.scatter3(xs=U[lbl_hc,0],ys=U[lbl_hc,1],zs=U[lbl_hc,2],c='seagreen',alpha=0.5,label='Happy',s=40)
# ax.scatter3D(xs=U[lbl_a,0],ys=U[lbl_a,1],zs=U[lbl_a,2],s=40,c='blueviolet',alpha=0.5,label='Angry')
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# ax.legend()
# plt.tight_layout()
# plt.legend()

ax = plt.subplot(grid1[0:2,3])

ax.scatter(U[lbl_hc,0],U[lbl_hc,1],c='blue',alpha=0.5,label='Happy',s=m_size)
ax.scatter(U[lbl_a,0],U[lbl_a,1],c='red',alpha=0.5,label='Angry',s=m_size)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.figtext(0.60, 0.93, 'C', fontsize=40, fontweight='bold')

# adding faces in PC space
my_cmap = copy.copy(plt.cm.get_cmap('gray')) # get a copy of the gray color map
my_cmap.set_bad(alpha=0) # set how the colormap handles 'bad' values
face_temp = np.squeeze(data_A[:,:,hc_example]).astype(float)*Mask
face_plot = face_temp.copy()
face_plot[face_temp==0.0]=np.nan
im = OffsetImage(face_plot[45:205,35:145], zoom=0.6, cmap=my_cmap)
x,y = U[hc_example,:2]
ab = AnnotationBbox(im,
                    [x, y],
                    xybox=(-0.086,0.1),
                    xycoords='data',
                    frameon=False,
                    arrowprops=dict(arrowstyle="->",color='black'))
ax.add_artist(ab)

face_temp = np.squeeze(data_A[:,:,angry_example]).astype(float)*Mask
face_plot = face_temp.copy()
face_plot[face_temp==0.0]=np.nan
im = OffsetImage(face_plot[45:205,35:145], zoom=0.6, cmap=my_cmap)
x,y = U[angry_example,:2]
ab = AnnotationBbox(im,
                    [x, y],
                    xybox=(-0.085,-0.01),
                    xycoords='data',
                    frameon=False,
                    arrowprops=dict(arrowstyle="->",color='black'))
ax.add_artist(ab)

ax = plt.subplot(grid1[2:4,3])
ax.scatter(U_gcpca[lbl_hc,0],U_gcpca[lbl_hc,1],c='blue',alpha=0.5,label='Happy',s=m_size)
ax.scatter(U_gcpca[lbl_a,0],U_gcpca[lbl_a,1],c='red',alpha=0.5,label='Angry',s=m_size)
ax.set_xlabel('gcPC1')
ax.set_ylabel('gcPC2')
ax.legend()
plt.xlim((-0.35,0.35))
plt.ylim((-0.35,0.35))

# adding faces in PC space
my_cmap = copy.copy(plt.cm.get_cmap('gray')) # get a copy of the gray color map
my_cmap.set_bad(alpha=0) # set how the colormap handles 'bad' values
face_temp = np.squeeze(data_A[:,:,hc_example]).astype(float)*Mask
face_plot = face_temp.copy()
face_plot[face_temp==0.0]=np.nan
im = OffsetImage(face_plot[45:205,35:145], zoom=0.6, cmap=my_cmap)
x,y = U_gcpca[hc_example,:2]
ab = AnnotationBbox(im,
                    [x, y],
                    xybox=(-0.3,0.2),
                    xycoords='data',
                    frameon=False,
                    arrowprops=dict(arrowstyle="->",color='black'))
ax.add_artist(ab)

face_temp = np.squeeze(data_A[:,:,angry_example]).astype(float)*Mask
face_plot = face_temp.copy()
face_plot[face_temp==0.0]=np.nan
im = OffsetImage(face_plot[45:205,35:145], zoom=0.6, cmap=my_cmap)
x,y = U_gcpca[angry_example,:2]
ab = AnnotationBbox(im,
                    [x, y],
                    xybox=(0.3,-0.1),
                    xycoords='data',
                    frameon=False,
                    arrowprops=dict(arrowstyle="->",color='black'))
ax.add_artist(ab)
plt.figtext(0.60, 0.46, 'E', fontsize=40, fontweight='bold')
plt.savefig("face_expression_figure2.pdf", format="pdf")
plt.savefig("face_expression_figure2.png", format="png")

# ax.autoscale()
# #picking a face
# face_temp = np.squeeze(data_A[:,:,face_selection_1])*Mask
# face_plot = face_temp.copy()
# face_plot[face_temp==0.0]=np.nan
# im = OffsetImage(face_plot, zoom=1, cmap=my_cmap)
# ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False,alpha=0.5)
# ax.add_artist(ab)

# plt.figure()
# plt.subplot(121)
# plt.imshow(face_plot,cmap='gray')

# temp_selection = np.argwhere( np.logical_and((U_gcpca[:,0] < -0.005), (U_gcpca[:,0] > -0.015) ))
# face_selection_1 = np.random.permutation(temp_selection)[0]
# face_temp = np.squeeze(data_A[:,:,face_selection_1])*Mask
# face_plot = face_temp.copy()
# face_plot[(face_temp==0.0)]=255

# plt.subplot(122)
# plt.imshow(face_plot,cmap='gray')


# temp = data_B[:,:,5]*Mask;
# temp2  = temp.copy()
# happy_face  = temp.copy()
# happy_face[(temp2==0.0)]=255
# plt.imshow(happy_face,cmap='gray')



#%% make a whole plot of faces
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
# import copy
# my_cmap = copy.copy(plt.cm.get_cmap('gray')) # get a copy of the gray color map
# my_cmap.set_bad(alpha=0) # set how the colormap handles 'bad' values

# # plt.rcParams.update({'font.size':28})
# fig, ax = plt.subplots(figsize=(30, 21))
# c = 0
# for x,y in U_gcpca[:,0:2]:
#     face_temp = np.squeeze(data_A[:,:,c]).astype(float)*Mask
#     face_plot = face_temp.copy()
#     face_plot[face_temp==0.0]=np.nan
#     im = OffsetImage(face_plot, zoom=1, cmap=my_cmap)
#     ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
#     ax.add_artist(ab)
#     c+=1
# ax.update_datalim(U_gcpca[:,0:2])
# ax.autoscale()
# ax.set_xlabel('gcPC1',fontsize=40)
# ax.set_ylabel('gcPC2',fontsize=40)
# ax.tick_params(axis='both', which='major', labelsize=30)


# %% gcPCA on vgg16 feature extraction
# gcPCA_mdl = gcPCA(method='v4',normalize_flag=True)
# gcPCA_mdl.fit(A_feature_list_np,B_feature_list_np)


# lbl_hc = np.argwhere(labels==0)
# lbl_a = np.argwhere(labels==1)
# U_gcpca = gcPCA_mdl.Ra_scores_;
# plt.figure()
# plt.scatter(U_gcpca[lbl_hc,0],U_gcpca[lbl_hc,1],c='seagreen',alpha=0.5,label='Happy')
# # plt.scatter(U_gcpca[lbl_hc,0].mean(),U_gcpca[lbl_hc,1].mean(),c='b',alpha=0.7)
# plt.scatter(U_gcpca[lbl_a,0],U_gcpca[lbl_a,1],c='blueviolet',alpha=0.5,label='Angry')
# # plt.scatter(U_gcpca[lbl_a,0].mean(),U_gcpca[lbl_a,1].mean(),c='r',alpha=0.7)
# plt.xlabel('gcPC1')
# plt.ylabel('gcPC2')
# plt.tight_layout()
# plt.legend()