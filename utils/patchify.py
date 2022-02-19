#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 18:09:01 2022

@author: msakong
"""

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import random
import os

def patchify(input: np.ndarray, patch_size: int):
    '''
    Intentionally create overlapping patches
    to enhance predictions on edges
    '''
    x = input.shape[0]
    y = input.shape[1]
    z = input.shape[2]
    x_num = x//patch_size
    y_num = y//patch_size
    z_num = z//patch_size
    random_patches = np.zeros((128,patch_size,patch_size,patch_size))
    location_logs = dict()
    cnt=0
    for i in range(x_num):
        for j in range(y_num):
            for k in range(z_num):
                random_patches[cnt,:,:,:]=input[i*patch_size:i*patch_size+patch_size,
                                                  j*patch_size:j*patch_size+patch_size,
                                                  k*patch_size:k*patch_size+patch_size]
                location_logs[cnt]=(i*patch_size,
                                    j*patch_size,
                                    k*patch_size)
                cnt+=1
    
    for i in range(cnt,128):
        rand_x = random.randint(0,x-patch_size-1)
        rand_y = random.randint(0,y-patch_size-1)
        rand_z = random.randint(0,z-patch_size-1)
        patch = input[rand_x:rand_x+patch_size, rand_y:rand_y+patch_size, rand_z:rand_z+patch_size]
        random_patches[i,:,:,:] = patch
        location_logs[i]=(rand_x,rand_y,rand_z)
    return random_patches, location_logs

def unpatchify(input:np.ndarray, location_list: dict, orig_size:int,patch_size:int,num_classes:int):
    orig_array = np.zeros((orig_size,orig_size,orig_size,num_classes))
    for i in range(128):
        x = location_list[i][0]
        y = location_list[i][1]
        z = location_list[i][2]
        orig_array[x:x+patch_size,y:y+patch_size,z:z+patch_size,:] += input[i,:,:,:,:]
    return orig_array

if __name__=='__main__':
    #Testing Purpose
    
    #test = np.ones((192,192,192))
    #one, two = random_patchify(test,64)
    #orig = unpatchify_to_original(one,two,192,64)








    