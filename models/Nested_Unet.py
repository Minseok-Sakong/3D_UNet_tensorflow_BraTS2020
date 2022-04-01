#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 11:29:23 2022

@author: msakong

Netsted_Unet
:https://arxiv.org/pdf/1807.10165.pdf
"""


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv3D, ZeroPadding3D, UpSampling3D, Conv3DTranspose
from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling3D, MaxPooling3D
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from tensorflow.keras.layers import ELU, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GaussianDropout
from tensorflow.keras.utils import plot_model
import config

import numpy as np

smooth = 1.
dropout_rate = 0.5

########################################
# 3D Standard Unit
########################################

def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):

    act = 'elu'
    x = Conv3D(nb_filter, (kernel_size, kernel_size,kernel_size), activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    x = Conv3D(nb_filter, (kernel_size, kernel_size,kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)
    
    return x

########################################

def Nested_Unet(patch_size, num_channels=3, num_classes=1, deep_supervision=False):

    nb_filter = [32,64,128,256,512]
    act = 'elu'

    # Handle Dimension Ordering for different backends
    global bn_axis
    
    bn_axis = -1
    img_input = Input(shape=(patch_size, patch_size, patch_size, num_channels), name='main_input')
    
    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool2')(conv2_1)

    up1_2 = Conv3DTranspose(nb_filter[0], (2, 2, 2), strides=(2, 2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool3')(conv3_1)

    up2_2 = Conv3DTranspose(nb_filter[1], (2, 2, 2), strides=(2, 2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv3DTranspose(nb_filter[0], (2, 2, 2), strides=(2, 2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool4')(conv4_1)

    up3_2 = Conv3DTranspose(nb_filter[2], (2, 2, 2), strides=(2, 2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv3DTranspose(nb_filter[1], (2, 2, 2), strides=(2, 2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv3DTranspose(nb_filter[0], (2, 2, 2), strides=(2, 2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv3DTranspose(nb_filter[3], (2, 2, 2), strides=(2, 2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv3DTranspose(nb_filter[2], (2, 2, 2), strides=(2, 2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv3DTranspose(nb_filter[1], (2, 2, 2), strides=(2, 2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv3DTranspose(nb_filter[0], (2, 2, 2), strides=(2, 2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv3D(num_classes, (1, 1, 1), activation='softmax', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv3D(num_classes, (1, 1, 1), activation='softmax', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv3D(num_classes, (1, 1, 1), activation='softmax', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv3D(num_classes, (1, 1, 1), activation='softmax', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    if deep_supervision:
        model = Model(inputs=img_input, outputs=[nestnet_output_1,
                                               nestnet_output_2,
                                               nestnet_output_3,
                                               nestnet_output_4])
    else:
        model = Model(inputs=img_input, outputs=[nestnet_output_4])
        #model.plot_model(model, to_file='tmp.png', show_shapes=True)
    return model

if __name__=='__main__':
    model = Nested_Unet(patch_size=64,num_channels=3,num_classes=4)
    print(model.summary())