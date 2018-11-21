#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:30:15 2018

@author: dirac
"""

import cv2
import numpy as np
import os

base_dir='data'

file_list=os.listdir(base_dir)
img = cv2.imread(os.path.join(base_dir,file_list[4]))


#img=np.random.randint(0,255,size=[1400,1400,3])
#img=img.astype('uint8')

cv2.imshow('image',img)


dist=cv2.GaussianBlur(img,(7,7),1)
cv2.imshow('gauss_blur',dist)

cv2.waitKey(0)
cv2.destroyAllWindows()