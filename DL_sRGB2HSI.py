# -*- coding: utf-8 -*-

import numpy as np # linear algebra
from scipy.io import savemat

from IPython.display import display, Image
from matplotlib.pyplot import imshow
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb, rgb2lab

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import glob
import cv2 as cv2
import os
import pdb
from tkinter import Tcl
import scipy.io
#images = [cv2.imread(file) for file in glob.glob("C:/Users/morte/OneDrive/Desktop/Colorssss/20200416_RCHIV E_MORTEZA/urban/*.jpeg")]

folder_path='C:/Users/Morteza/Desktop/YouTube/Ref_CNN/sRGB_Images/' 
files =  os.listdir('C:/Users/Morteza/Desktop/YouTube/Ref_CNN/sRGB_Images/')

images1 = []
files=Tcl().call('lsort', '-dict', files)

for img in  files:
    #print(folder_path+img)
    img=folder_path+img
    img = load_img(img, target_size=(512, 512)) 
    img = img_to_array(img)
    #lab_image = rgb2lab(img)
    #lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]
    # The input will be the black and white layer
    #X = lab_image_norm[:,:,0]
    X=img
    #X = np.expand_dims(X, axis=2) 
    images1.append(X)
#pdb.set_trace()

folder_path='C:/Users/Morteza/Desktop/YouTube/Ref_CNN/Hyper_Mat/' 
files =  os.listdir('C:/Users/Morteza/Desktop/YouTube/Ref_CNN/Hyper_Mat/')


files=Tcl().call('lsort', '-dict', files)
images2 = []
for img in files:
    #print(folder_path+img)
    img=folder_path+img
    img = scipy.io.loadmat(img)
    img = img['Ref']  
    img = np.array(img)
    #lab_image = rgb2lab(img)
    #lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]
    # The input will be the black and white layer
    # Y = lab_image_norm[:,:,1:]
    Y=img
    #Y = np.expand_dims(Y, axis=2)
    images2.append(Y)
#pdb.set_trace()

# The Conv2D layer we will use later expects the inputs and training outputs to be of the following format:
# (samples, rows, cols, channels), so we need to do some reshaping
# https://keras.io/layers/convolutional/
#X = X.reshape(34, X.shape[0], X.shape[1], 1)
#Y = Y.reshape(34, Y.shape[0], Y.shape[1], 2)
X = np.array(images1)
Y = np.array(images2)
#pdb.set_trace()

model = Sequential()
model.add(InputLayer(input_shape=(None, None, 3)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
#model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
#model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(31, (3,3), activation='sigmoid', padding='same'))

# Finish model
model.compile(optimizer='rmsprop', loss='mse')
model.fit(x=X, y=Y, batch_size=1, epochs=1000, verbose=0)

model.evaluate(X, Y, batch_size=1)
#pdb.set_trace()

folder_path='C:/Users/Morteza/Desktop/YouTube/Ref_CNN/sRGB_Images_test/' 
img='sRGBpompoms_ms_31.png'
img=folder_path+img
img = load_img(img, target_size=(512, 512)) 
img = img_to_array(img)
#kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#img_sharp = cv2.filter2D(img, -1, kernel)
#edges = cv2.Canny(img,100,200)
#img_edge=img_sharp-img
#lab_image = rgb2lab(img)
#lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]
    # The input will be the black and white layer
#X = lab_image_norm[:,:,0]
#
X = np.array(img)
X = np.expand_dims(X, axis=2)
X=np.reshape(X,(1,512,512,3))
#X = img
output = model.predict(X)
#cur = np.zeros((200, 200, 3))
#cur[:,:,0] = X[0][:,:,0]
#cur[:,:,1:] = output[0]
#cur = (cur * [100, 255, 255]) - [0, 128, 128]
#rgb_image = lab2rgb(cur)
output=np.reshape(output,(512,512,31))
#import cv2 as cv2
#kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#output = cv2.filter2D(output, -1, kernel)
savemat("Ref_rec.mat", {"Ref_rec":output})