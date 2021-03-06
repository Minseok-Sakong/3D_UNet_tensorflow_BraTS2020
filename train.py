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
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from models.MultiRes_Unet import MultiResUnet3D, dice_coefficient_loss
from models.Nested_Unet import Nested_Unet
import os
import config
'''
configproto = tf.compat.v1.ConfigProto() 
configproto.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=configproto) 
tf.compat.v1.keras.backend.set_session(sess)
'''
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

tr_ds = DataSet_random(brain_tr, mask_tr, 1,True, None)
val_ds = DataSet_random(brain_val, mask_val, 1, False, None)
rlr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, mode='min',verbose=1)
ely_cb = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)
#ds_all = DataSet(brain_filenames, mask_filenames, 1, None, True, preprocess_input)
model = Nested_Unet(patch_size=patch_size,num_channels=channels,num_classes=n_classes)
model.compile(optimizer = optim, loss=dice_coefficient_loss, metrics=metrics)
model_checkpoint_callback = ModelCheckpoint(
    filepath="/projectnb/bil/Minseok/test/model_weights/{epoch:02d}-{val_loss:.2f}.hdf5",
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
history=model.fit(tr_ds,
            epochs=1000,
            verbose=1,
            validation_data=val_ds,
            callbacks=([rlr_cb,ely_cb,model_checkpoint_callback]),
            )
model.save("/projectnb/bil/Minseok/test/model_weights/final_model.h5")

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