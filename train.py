#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 22:50:14 2022

@author: msakong
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import utils.extra as extra
from dataloader import DataSet_random
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from models.MultiRes_Unet import MultiResUnet3D, dice_coefficient_loss
import os
import config


data_df = extra.prepare_dataframe("/projectnb/bil/Minseok/test/dataset/combined/")

brain_filenames = data_df['image'].values
mask_filenames = data_df['mask'].values


patch_size = config.PATCH_SIZE
n_classes = config.NUM_CLASSES
channels=config.NUM_CHANNELS
LR=0.0001
optim = Adam(learning_rate=LR)
brain_tr, brain_val, mask_tr, mask_val = train_test_split(brain_filenames, mask_filenames, test_size=0.3)
iou_score = tf.keras.metrics.MeanIoU(num_classes=n_classes)
metrics = [iou_score]

tr_ds = DataSet_random(brain_tr, mask_tr, 4,True, None)
val_ds = DataSet_random(brain_val, mask_val, 4,True, None)
rlr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, mode='min',verbose=1)
ely_cb = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)
#ds_all = DataSet(brain_filenames, mask_filenames, 1, None, True, preprocess_input)
model = MultiResUnet3D(patch_size=patch_size,num_channels=channels,num_classes=n_classes)
model.compile(optimizer = optim, loss=dice_coefficient_loss, metrics=metrics)

history=model.fit(tr_ds,
            epochs=100,
            verbose=1,
            validation_data=val_ds,
            callbacks=([rlr_cb,ely_cb]))
model.save("/projectnb/bil/Minseok/model_weights/unetplus_100epochs_adni.h5")

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(val_loss) + 1)
plt.plot(epochs, loss, label= ' Training loss')
plt.plot(epochs, val_loss, label= ' Validation loss')

plt.title('Training & Validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend(loc=4, prop={'size': 6})
#plt.savefig('val_iou.pdf')
plt.show()