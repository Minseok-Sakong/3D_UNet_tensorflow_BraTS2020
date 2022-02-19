#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 14:10:12 2021

@author: msakong
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os

def prepare_dataset(data_path):
    '''
    Prepares Dataset for data loading

    Parameters
    ----------
    data_path : String
    : Base data path

    Returns
    -------
    data_df : Pandas dataframe
        Pandas DataFrame containing file_paths for training dataset

    '''
    FLAIR = []
    T1CE = []
    T2 = []
    MASK = []
    
    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:
            if 't1.nii' in filename:
                subj_name = filename[-10:-7]
                flair_path = dirname+'/'+'BraTS20_Training_'+subj_name+'_flair.nii'
                FLAIR.append(flair_path)
                t1ce_path = dirname+'/'+'BraTS20_Training_'+subj_name+'_t1ce.nii'
                T1CE.append(t1ce_path)
                t2_path = dirname+'/'+'BraTS20_Training_'+subj_name+'_t2.nii'
                T2.append(t2_path)
                mask_path = dirname+'/'+'BraTS20_Training_'+subj_name+'_seg.nii.nii'
                MASK.append(mask_path)
    
    data_df = pd.DataFrame({'FLAIR':FLAIR, 'T1CE':T1CE, 'T2':T2, 'MASK':MASK})
    
    return data_df


if __name__=='__main__':
    print("Test")
    df = prepare_dataset("")
    