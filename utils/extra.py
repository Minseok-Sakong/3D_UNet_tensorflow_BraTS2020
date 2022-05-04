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

def prepare_dataframe(data_path):
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
    IMAGE = []
    MASK = []
    
    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:
            if 'combined.npy' in filename:
                subj_name = filename[:-13]
                image_path = dirname+subj_name+'_combined.npy'
                IMAGE.append(image_path)
                # dirname for mask files
                mask_dirname = dirname[:-9]+'combined_mask/'
                mask_path = mask_dirname+subj_name+'_mask.npy'
                MASK.append(mask_path)
    data_df = pd.DataFrame({'image':IMAGE, 'mask':MASK})
    
    return data_df


if __name__=='__main__':
    print("Test")
    df = prepare_dataset("/projectnb/bil/Minseok/test/dataset/combined/")
    