#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:35:36 2022

@author: msakong
"""

import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import nibabel as nib
import random

flair = np.array(nib.load('/projectnb/bil/Minseok/test/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii').get_fdata())
seg = np.array(nib.load('/projectnb/bil/Minseok/test/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii').get_fdata())
t1 = np.array(nib.load('/projectnb/bil/Minseok/test/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1.nii').get_fdata())
t1ce = np.array(nib.load('/projectnb/bil/Minseok/test/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii').get_fdata())
t2 = np.array(nib.load('/projectnb/bil/Minseok/test/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t2.nii').get_fdata())

slice = random.randint(0,flair.shape[2])
plt.figure(figsize=(12,8))

plt.subplot(231)
plt.imshow(flair[:,:,slice],cmap='gray')
plt.title('Flair')
plt.subplot(232)
plt.imshow(t1[:,:,slice],cmap='gray')
plt.title('t1')
plt.subplot(233)
plt.imshow(t1ce[:,:,slice],cmap='gray')
plt.title('t1ce')
plt.subplot(234)
plt.imshow(t2[:,:,slice],cmap='gray')
plt.title('t2')
plt.subplot(235)
plt.imshow(seg[:,:,slice])
plt.title('mask')





