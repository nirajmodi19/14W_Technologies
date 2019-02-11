#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:37:36 2019

@author: niraj
"""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()  
ap.add_argument("-i", "--image", required = True, help = "Path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

image = cv2.resize(image, (100, 100))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis = 0)

model = load_model('VadaPav.model')

(notVadaPav, VadaPav) = model.predict(image)[0]


label = "VadaPav" if VadaPav > notVadaPav else "Not VadaPav"
proba = VadaPav if VadaPav > notVadaPav else notVadaPav
proba *= 100
print("The given image is {} with {:.2f}%". format(label, proba))

