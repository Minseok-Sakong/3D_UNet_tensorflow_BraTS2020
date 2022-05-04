#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 12:43:07 2022

@author: msakong

Custom Dataloader inherited from keras.Sequence
: Pushes certain numbers(batch size) of patches on each training step
: Also, handles data augmentation
"""

import numpy as np
import nibabel as nib
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
import sklearn
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt
import config
from utils.extra import prepare_dataframe

PATCH_SIZE = config.PATCH_SIZE
num_classes = config.NUM_CLASSES
pixels = PATCH_SIZE*PATCH_SIZE*PATCH_SIZE
channel_nums = config.NUM_CHANNELS



def pre_func_example(input):
    '''
    Normalize input intensity/pixel values to 0~1
    '''
    scaler = MinMaxScaler()
    scaled_input = scaler.fit_transform(input.reshape(-1,1)).reshape(input.shape)
	
    return scaled_input

class DataSet_random(Sequence):
	"""
	brain_filenames = [list] a list containing absolute paths of image files
	mask_filenames = [list] a list contating absolute paths of mask files
	batch_size = [int] batch size
	shuffle = [boolean] if true, shuffle images and masks on each epoch
	pre_func = [function] a function to normalize/standardize input files
	"""

	def __init__(self, brain_filenames, mask_filenames, batch_size, shuffle=False, pre_func=None):
		self.brain_filenames = brain_filenames
		self.mask_filenames = mask_filenames
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.pre_func = pre_func
		if self.shuffle:
			self.on_epoch_end()
		assert len(self.brain_filenames)==len(self.mask_filenames), "Images and masks do not match."

	def __len__(self):
		return int(len(self.mask_filenames))

	def __getitem__(self, index):
		'''
		returns randomly cropped patches(= # of batch size) from a subject image file
		'''
		brain_name_batch = self.brain_filenames[index]
		if self.mask_filenames is not None:
			mask_name_batch = self.mask_filenames[index]
		# channel_nums depends on the input image. Ex) RGB image: channel_nums=3, Grayscale image: channel_nums=1
		brain_batch = np.zeros((self.batch_size, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, channel_nums), dtype='float32')
		mask_batch = np.zeros((self.batch_size, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, num_classes), dtype='float32')

		# load nifti files to numpy arrays
		'''
		image = np.array(nib.load(brain_name_batch).get_fdata())
		mask = np.array(nib.load(mask_name_batch).get_fdata())
		'''
		# load numpy array files
		image = np.load(brain_name_batch)
		mask = np.load(mask_name_batch)
		# standardize/normalize image intensity value
		if self.pre_func is not None:
			image = self.pre_func(image)
		else:
			image = image

		#Input image file is a 3D image
		x = image.shape[0]
		y = image.shape[1]
		z = image.shape[2]
		cnt = 0

		while cnt < self.batch_size:
			rand_x = random.randint(0, x - PATCH_SIZE - 1)
			rand_y = random.randint(0, y - PATCH_SIZE - 1)
			rand_z = random.randint(0, z - PATCH_SIZE - 1)

			new_brain = np.zeros(shape=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE,channel_nums))
			new_brain = image[rand_x:rand_x + PATCH_SIZE, rand_y:rand_y + PATCH_SIZE, rand_z:rand_z + PATCH_SIZE,:]
			new_mask = np.zeros(shape=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE))
			new_mask = mask[rand_x:rand_x + PATCH_SIZE, rand_y:rand_y + PATCH_SIZE, rand_z:rand_z + PATCH_SIZE]
			# Background pixel value = 0
			background_pixels = np.count_nonzero(new_mask == 0)

			background_percentage = background_pixels / pixels
			# Check background is over 90% in a patch. If true, discard it.
			if (background_percentage > 0.90):
				continue
			else:
				train_mask = np.expand_dims(new_mask, axis=3)
				train_mask_cat = to_categorical(train_mask, num_classes=num_classes)

				brain_batch[cnt] = new_brain
				mask_batch[cnt] = train_mask_cat
				cnt += 1

		return brain_batch, mask_batch

	def on_epoch_end(self):
		if (self.shuffle):
			self.brain_filenames, self.mask_filenames = sklearn.utils.shuffle(self.brain_filenames, self.mask_filenames)
		else:
			pass

if __name__ =='__main__':
    data_df = prepare_dataframe("/projectnb/bil/Minseok/test/dataset/combined/")
    brain_filenames = data_df['image'].values
    mask_filenames = data_df['mask'].values
    tr_ds = DataSet_random(brain_filenames, mask_filenames, 1, True, None)
    batch = next(iter(tr_ds))
    image = batch[0]
    mask = batch[1]
    plt.figure(figsize=(16, 12))
    plt.subplot(231)
    plt.title('Original Brain')
    plt.imshow(image[0,:,32,:,2], cmap='gray')
    '''
    data_df = make_dataframe_adni_random()

    brain_filenames = data_df['brain_paths'].values
    mask_filenames = data_df['mask_paths'].values
    tr_ds = DataSet_random_t(brain_filenames, mask_filenames, 1,False,True, None)
    
    batch = next(iter(tr_ds))
    image = batch[0]
    mask = batch[1]
    plt.figure(figsize=(16, 12))
    plt.subplot(231)
    plt.title('Original Brain')
    plt.imshow(image[0,:,32,:,1], cmap='gray')
    plt.subplot(232)
    plt.title('Original Mask')
    plt.imshow(mask[0,:,32,:])
    plt.show()
    '''


















