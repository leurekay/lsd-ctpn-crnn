#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 17:50:47 2018

@author: dirac
"""

import lsd

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import solve


image_path='data/bank5.jpg'

img_raw = cv2.imread(image_path)

img_cali=lsd.get_calibration(img_raw)
cv2.imwrite(image_path.replace('data','result'),img_cali)