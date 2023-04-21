#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 10:08:07 2022
This file will keep all the functions we use in the 

@author: eliezyer
"""
import numpy as np
from sklearn import svm,linear_model
import tensorflow as tf
import os
from PIL import Image

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

# adapted from abid et al., 2019
def resize_and_crop(img, size=(100, 100), crop_type='middle'):
    # ancillary function to get noisy mnist dataset

    # If height is higher we resize vertically, if not we resize horizontally
    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    # The image is scaled/cropped vertically or horizontally
    # depending on the ratio
    if ratio > img_ratio:
        img = img.resize((
            size[0],
            int(round(size[0] * img.size[1] / img.size[0]))),
            Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, img.size[0], size[1])
        elif crop_type == 'middle':
            box = (
                0,
                int(round((img.size[1] - size[1]) / 2)),
                img.size[0],
                int(round((img.size[1] + size[1]) / 2)))
        elif crop_type == 'bottom':
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else:
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize((
            int(round(size[1] * img.size[0] / img.size[1])),
            size[1]),
            Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, size[0], img.size[1])
        elif crop_type == 'middle':
            box = (
                int(round((img.size[0] - size[0]) / 2)),
                0,
                int(round((img.size[0] + size[0]) / 2)),
                img.size[1])
        elif crop_type == 'bottom':
            box = (
                img.size[0] - size[0],
                0,
                img.size[0],
                img.size[1])
        else:
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    else:
        img = img.resize((
            size[0],
            size[1]),
            Image.ANTIALIAS)
    # If the scale is the same, we do not need to crop
    return img


def get_noisy_mnist(natural_image_path):
    # loading mnist dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    target_idx = np.where(y_train < 2)[0]
    foreground = x_train[target_idx, :][:5000]
    target_labels = y_train[target_idx][:5000]

    nsamples, nx, ny = foreground.shape
    foreground = foreground.reshape((nsamples, nx * ny))

    # loading natural images
    IMAGE_PATH = natural_image_path

    natural_images = list()  # dictionary of pictures indexed by the pic # and each value is 100x100 image
    for filename in os.listdir(IMAGE_PATH):
        if filename.endswith(".JPEG") or filename.endswith(".JPG") or filename.endswith(".jpg"):
            try:
                im = Image.open(os.path.join(IMAGE_PATH, filename))
                im = im.convert(mode="L")  # convert to grayscale
                im = resize_and_crop(im)  # resize and crop each picture to be 100px by 100px
                natural_images.append(np.reshape(im, [10000]))
            except Exception as e:
                pass  # print(e)

    natural_images = np.asarray(natural_images, dtype=float)
    natural_images /= 255  # rescale to be 0-1

    # superimpose mnist and natural images
    np.random.seed(0)  # for reproducibility

    rand_indices = np.random.permutation(natural_images.shape[0])  # just shuffles the indices
    split = int(len(rand_indices) / 2)
    target_indices = rand_indices[0:split]  # choose the first half of images to be superimposed on target
    background_indices = rand_indices[split:]  # choose the second half of images to be background dataset

    target = np.zeros(foreground.shape)

    background = np.zeros(foreground.shape)

    for i in range(target.shape[0]):
        idx = np.random.choice(target_indices)  # randomly pick a image
        loc = np.random.randint(70, size=(2))  # randomly pick a region in the image
        superimposed_patch = np.reshape(
            np.reshape(natural_images[idx, :], [100, 100])[loc[0]:loc[0] + 28, :][:, loc[1]:loc[1] + 28], [1, 784])
        target[i] = (0.002 * foreground[i]) + superimposed_patch  # chnaged from .25 to 0.002

        idx = np.random.choice(background_indices)  # randomly pick a image
        loc = np.random.randint(70, size=(2))  # randomly pick a region in the image
        background_patch = np.reshape(
            np.reshape(natural_images[idx, :], [100, 100])[loc[0]:loc[0] + 28, :][:, loc[1]:loc[1] + 28], [1, 784])
        background[i] = background_patch

    return target, background, target_labels
