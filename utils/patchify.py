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

def calc_steps(input_arr,step):
    '''
    Calculate how many patches will fit in the input_arr
    '''
    x = input_arr.shape[0]
    y = input_arr.shape[1]
    z = input_arr.shape[2]
    if x%step==0:
        x_num = x//step
    else:
        x_num = x//step+1
    if y%step==0:
        y_num = y//step
    else:
        y_num = y//step+1
    if z%step==0:
        z_num = y//step
    else:
        z_num = z//step+1
    return x_num, y_num, z_num

def patchify(input_arr: np.ndarray, patch_size: int, step: int):
    '''
    Intentionally create overlapping patches
    to enhance predictions along edges
    '''
    x = input_arr.shape[0]
    y = input_arr.shape[1]
    z = input_arr.shape[2]
    x_num, y_num, z_num = calc_steps(input_arr, step)
    patches = []
    location_logs = dict()
    cnt=0
    for i in range(x_num):
        for j in range(y_num):
            for k in range(z_num):
                loc_x = i*step
                loc_y = j*step
                loc_z = k*step
                patch = input_arr[loc_x:loc_x+patch_size,
                              loc_y:loc_y+patch_size,
                              loc_z:loc_z+patch_size]
                if patch.shape == (patch_size,patch_size,patch_size):
                    patches.append(patch)
                else:
                    # if patch is cutoff, because it is located near borders of the input
                    temp = np.array(patch.shape) - np.array((patch_size, patch_size,patch_size))
                    
                    loc_x = loc_x+temp[0]
                    loc_y = loc_y+temp[1]
                    loc_z = loc_z+temp[2]
                    patch = input_arr[loc_x:loc_x+patch_size,
                                  loc_y:loc_y+patch_size,
                                  loc_z:loc_z+patch_size]
                    patches.append(patch)    
                # Save location information of each patch
                location_logs[cnt]=(loc_x,
                                    loc_y,
                                    loc_z)
                cnt+=1

    return patches, location_logs

def unpatchify(input_arr:np.ndarray, location_list: dict, orig_size:int,patch_size:int,num_classes:int=0):
    if num_classes==0:
        orig_array = np.zeros((orig_size,orig_size,orig_size))
        for i in range(len(location_list)):
            x = location_list[i][0]
            y = location_list[i][1]
            z = location_list[i][2]
            orig_array[x:x+patch_size,y:y+patch_size,z:z+patch_size] += input_arr[i,:,:,:]
    else:
        orig_array = np.zeros((orig_size,orig_size,orig_size,num_classes))
        for i in range(location_list):
            x = location_list[i][0]
            y = location_list[i][1]
            z = location_list[i][2]
            orig_array[x:x+patch_size,y:y+patch_size,z:z+patch_size,:] += input_arr[i,:,:,:,:]
    return orig_array

if __name__=='__main__':
    #Testing Purpose
    
    test = np.array(range(0,1000)).reshape(10,10,10)
    one, two = patchify(test,5,3)
    oone = np.array(one)
    orig = unpatchify(oone,two,10,5,0)








    