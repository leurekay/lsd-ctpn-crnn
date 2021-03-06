#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午3:25
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : config.py
# @IDE: PyCharm Community Edition
"""
Set some global configuration
"""
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# Train options
__C.TRAIN = edict()

# Set the shadownet training epochs
__C.TRAIN.EPOCHS = 50000
# Set the display step
__C.TRAIN.DISPLAY_STEP = 1
# Set the test display step during training process
__C.TRAIN.TEST_DISPLAY_STEP = 100
# Set the momentum parameter of the optimizer
__C.TRAIN.MOMENTUM = 0.9
# Set the initial learning rate
__C.TRAIN.LEARNING_RATE = 0.1
# Set the GPU resource used during training process
__C.TRAIN.GPU_MEMORY_FRACTION = 0.85
# Set the GPU allow growth parameter during tensorflow training process
__C.TRAIN.TF_ALLOW_GROWTH = True
# Set the shadownet training batch size
__C.TRAIN.BATCH_SIZE = 64
# Set the shadownet validation batch size
__C.TRAIN.VAL_BATCH_SIZE = 64
# Set the learning rate decay steps
__C.TRAIN.LR_DECAY_STEPS = 10000
# Set the learning rate decay rate
__C.TRAIN.LR_DECAY_RATE = 0.96
# Set the L2 regularization decay rate
__C.TRAIN.L2_DECAY_RATE = 0.00001
# Set the horizon flip data augmentation method
__C.TRAIN.USE_HORIZON_FLIP = True
# Set the vertical flip data augmentation method
__C.TRAIN.USE_VERTICAL_FLIP = False
# Set the random crop data augmentation method
__C.TRAIN.USE_RANDOM_CROP = False
__C.TRAIN.RANDOM_CROP_VALUE = [300, 300, 3]
# Set the random brightness data augmentation method
__C.TRAIN.USE_RANDOM_BRIGHTNESS = False
__C.TRAIN.RANDOM_BRIGHTNESS_VALUE = 100  # you can check the preprocess.py scripts to learn the way how it's implemented
# Set the random contrast data augmentation method
__C.TRAIN.USE_RANDOM_CONTRAST = False
__C.TRAIN.RANDOM_CONTRAST_LOWER_VALUE = 0.4
__C.TRAIN.RANDOM_CONTRAST_HIGHER_VALUE = 0.6
# Set the std normalization data augmentation method
__C.TRAIN.USE_STD_NORMALIZATION = False
# Set the min max normalization data augmentation method
__C.TRAIN.USE_MINMAX_NORMALIZATION = False
# Set the central normalization data augmentation method
__C.TRAIN.USE_CENTRAL_NORMALIZATION = False
__C.TRAIN.CENTRAL_NORMALIZATION_VALUE = [103.939, 116.779, 123.68]

# Test options
__C.TEST = edict()

# Set the GPU resource used during testing process
__C.TEST.GPU_MEMORY_FRACTION = 0.5
# Set the GPU allow growth parameter during tensorflow testing process
__C.TEST.TF_ALLOW_GROWTH = False
# Set the test batch size
__C.TEST.BATCH_SIZE = 32

# Set the invokernet output dims
__C.TRAIN.OUT_DIMS = 10
