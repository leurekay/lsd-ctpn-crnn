#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:14:43 2018

@author: ly
"""

import os
from PIL import Image


import numpy as np
import pandas as pd


from keras.models import Model
from keras.layers import Dense, Dropout, Flatten,Input,Activation,Reshape,Lambda
from keras.layers import Conv2D, MaxPooling2D,MaxPooling3D,Conv3D,Deconv2D,Deconv3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD

from keras.utils import np_utils  
from keras.utils import plot_model  

import resnet3D

import tensorflow as tf
import keras
import keras.backend as K

import config
config=config.config


def f(x):
    return x**2

g=f

print (g(6))





# =============================================================================
# def generate_num(n_samples,shuffle=True):
#     
#     ids=np.array(np.arange(n_samples))
# 
#     while True:
#         if shuffle:
#             np.random.shuffle(ids)
#         for i in ids:
#             yield np.array([1,1])*i
#         print ('epoch done!')
#         time.sleep(10)
#         
# 
# box=[]
# for i in generate_num(20):
#     box.append(i[0])
# =============================================================================





# =============================================================================
# def slice(x):  
#     """ Define a tensor slice function 
#     """  
#     return x[:,:,:,:,:2]  
# 
# 
# coord=Input(shape=(32,32,32,3))
# x=concatenate([coord,coord])
# x=Lambda(slice)(x)  
# model=Model(inputs=coord,outputs=x )
# model.summary()
# plot_model(model, to_file='images/ooxx.png',show_shapes=True)
# =============================================================================
    
    
