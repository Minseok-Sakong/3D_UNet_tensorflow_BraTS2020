#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:54:58 2022

@author: msakong
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def prepare_dataset(data_path,output_path):
    '''
    Prepare training and validation dataset
    -> combine FLAIR, T1CE, and T2 to one 3 channels numpy array
    -> each channel is normalized using MinMaxScaler
    '''
    
    
    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:
            if 't1.nii' in filename:
                subj_name = filename[-10:-7]
                flair_path = dirname+'/'+'BraTS20_Training_'+subj_name+'_flair.nii'
                flair = np.array(nib.load(flair_path).get_fdata())
                # Make an empty numpy array for combiniing 3 channels together
                combined = np.zeros((flair.shape[0],flair.shape[1],flair.shape[2],3))
                # Normalize it (flatten->normalize->reconstruct to original shape)
                scaled_flair = scaler.fit_transform(flair.reshape(-1,flair.shape[-1])).reshape(flair.shape)
                # Append flair into the array
                combined[:,:,:,0] = scaled_flair
                t1ce_path = dirname+'/'+'BraTS20_Training_'+subj_name+'_t1ce.nii'
                t1ce = np.array(nib.load(t1ce_path).get_fdata())
                scaled_t1ce = scaler.fit_transform(t1ce.reshape(-1,t1ce.shape[-1])).reshape(t1ce.shape)
                combined[:,:,:,1] = scaled_t1ce
                t2_path = dirname+'/'+'BraTS20_Training_'+subj_name+'_t2.nii'
                t2 = np.array(nib.load(t2_path).get_fdata())
                scaled_t2 = scaler.fit_transform(t2.reshape(-1,t2.shape[-1])).reshape(t2.shape)
                combined[:,:,:,2] = scaled_t2
                #Save combined numpy array as npy file
                np.save(output_path+'/'+subj_name+'_combined.npy',combined)
                print('Processed '+subj_name)

if __name__=='__main__':
    data_path = '/projectnb/bil/Minseok/test/dataset/BraTS2020_TrainingData/'
    output_path = '/projectnb/bil/Minseok/test/dataset/combined'
    prepare_dataset(data_path,output_path)
