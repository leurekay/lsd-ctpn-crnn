#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:50:14 2018

@author: ly
"""
from __future__ import division

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Conv2D, Conv3D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K





ROW_AXIS=1
COL_AXIS=2
DEP_AXIS=3
CHANNEL_AXIS=-1


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization()(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1,1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        
        return _bn_relu(conv)

    return f

def _conv_bn(**conv_params):
    """Helper to build a conv -> BN
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1,1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return BatchNormalization()(conv)

    return f

def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    stride_depth = int(round(input_shape[DEP_AXIS] / residual_shape[DEP_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or stride_depth > 1 or not equal_channels:
        shortcut = Conv3D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1,1),
                          strides=(stride_width, stride_height,stride_depth),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])



def basic_block2(filters, init_strides=(1, 1,1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv3D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _conv_bn_relu(filters=filters, kernel_size=(3, 3,3),
                                  strides=init_strides)(input)

        residual = _conv_bn(filters=filters, kernel_size=(3, 3,3))(conv1)
        return _shortcut(input, residual)

    return f



class PostRes():
    def __init__(self, x, n_out, stride = 1):
        self.input=x
        input_shape = K.int_shape(x)
        n_in=input_shape[-1]
        self.conv1 = Conv3D(filters=n_out, kernel_size = 3, strides = stride, padding = 'same')
        self.bn1 = BatchNormalization()
        self.relu = Activation('relu')
        self.conv2 = Conv3D(filters=n_out, kernel_size = 3, strides = stride, padding = 'same')
        self.bn2 = BatchNormalization(gamma_initializer='zeros',moving_variance_initializer='zeros')
        
        def f(x):
            conv=Conv3D(n_out, kernel_size = 1, strides = stride)(x)
            return BatchNormalization()(conv)
            
        
        if stride != 1 or n_out != n_in:
            self.shortcut = f
        else:
            self.shortcut = None
    
    
    def forward(self):
        x=self.input
#        input_shape = K.int_shape(x)
#        print (input_shape)
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = add([out, residual])
        out = Activation('relu')(out)
        return out


def f():
    def g(x):
        return x**2
    return g

def test(ispool,**conv):
    return conv["a"]+ispool
