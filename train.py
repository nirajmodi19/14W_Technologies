#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 20:01:01 2019

@author: niraj
"""

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
from keras.applications.vgg16 import preprocess_input
#from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator,img_to_array

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from imutils import paths
from glob import glob
import random
import cv2
import os

EPOCHS = 5
BS = 32

data = []
labels = []
 
imagePaths = sorted(list(paths.list_images("data")))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (100, 100))
	image = img_to_array(image)
	data.append(image)
 
	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == "vadapav" else 0
	labels.append(label)
    
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
 
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.20, random_state=42)
 
# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

vgg = VGG16(input_shape = [100, 100] + [3], weights='imagenet',include_top = False)

for layer in vgg.layers:
    layer.trainable = False
    
x = Flatten()(vgg.output)
prediction = Dense(2, activation = 'softmax')(x)
model = Model(inputs = vgg.input, output = prediction)
model.summary()
model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy']) 

# construct the image generator for data augmentation

gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True)

H = model.fit_generator(gen.flow(trainX, trainY, batch_size = BS),
                         validation_data=(testX, testY), steps_per_epoch = len(trainX) // BS,
                         epochs = EPOCHS, verbose=1)

model.save('VadaPav.model')