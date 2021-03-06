#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 21:45:10 2022

@author: msakong
"""

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv3D, Input, MaxPooling3D, Dropout, concatenate, UpSampling3D
import tensorflow as tf
import config

def Unet(patch_size, num_channels=3, num_classes=1):
    x= Input(shape=(patch_size, patch_size, patch_size, num_channels))
    conv1 = Conv3D(32, 3, activation = 'relu', padding = 'same',data_format="channels_last")(x)
    conv1 = Conv3D(32, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = Conv3D(64, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv3D(64, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = Conv3D(128, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv3D(128, 3, activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = Conv3D(256, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv3D(256, 3, activation = 'relu', padding = 'same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(512, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv3D(512, 3, activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv3D(256, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(drop5))
    merge6 = concatenate([drop4,up6],axis=-1)
    conv6 = Conv3D(256, 3, activation = 'relu', padding = 'same')(merge6)
    conv6 = Conv3D(256, 3, activation = 'relu', padding = 'same')(conv6)

    up7 = Conv3D(128, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(conv6))
    merge7 = concatenate([conv3,up7],axis=-1)
    conv7 = Conv3D(128, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv3D(128, 3, activation = 'relu', padding = 'same')(conv7)

    up8 = Conv3D(64, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(conv7))
    merge8 = concatenate([conv2,up8],axis=-1)
    conv8 = Conv3D(64, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv3D(64, 3, activation = 'relu', padding = 'same')(conv8)

    up9 = Conv3D(32, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(conv8))
    merge9 = concatenate([conv1,up9],axis=-1)
    conv9 = Conv3D(32, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv3D(32, 3, activation = 'relu', padding = 'same')(conv9)
    conv10 = Conv3D(num_classes, 1, activation = 'softmax')(conv9)
    model = Model(inputs=x, outputs = conv10)
    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

if __name__=='__main__':
    model = Unet(patch_size=64,num_channels=3,num_classes=4)
    print(model.summary())