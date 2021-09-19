import tensorflow as tf
import os
import random
import numpy as np
from tqdm import tqdm 
import nibabel as nib
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras import layers

num_classes = 2
img_rows,img_cols = 200,200
batch_size = 128

train_data_dir = '/content/drive/MyDrive/train (2)'
validation_data_dir = '/content/drive/MyDrive/val (1)'
test_data_dir = '/content/drive/MyDrive/test (2)'

train_datagen = ImageDataGenerator(#rotation_range = 180,
                                         validation_split = 0.2, horizontal_flip=True,
                                        )
validation_datagen = ImageDataGenerator(validation_split = 0.2)

train_generator = train_datagen.flow_from_directory(directory = train_data_dir,
                                                    target_size = (img_rows,img_cols),
                                                    batch_size = 128,
                                                    class_mode = "binary",
                                                    subset = "training"
                                                   )
validation_generator = validation_datagen.flow_from_directory( directory = test_data_dir,
                                                              target_size = (img_rows,img_cols),
                                                              batch_size = 128,
                                                              class_mode = "binary",
                                                              subset = "validation"
                                                             )

