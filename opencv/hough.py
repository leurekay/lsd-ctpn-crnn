#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:18:30 2018

@author: dirac
"""

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

base_dir='data'

file_list=os.listdir(base_dir)
file_path=os.path.join(base_dir,file_list[16])

img = cv2.imread('data/bank5.jpg')
img=cv2.GaussianBlur(img,(3,3),1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,60,150,apertureSize = 3)


fig=plt.figure(figsize=[20,10])
plt.subplot(221),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

#plt.show()

minLineLength = 50
maxLineGap = 30
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)

for line in lines:
    x1,y1,x2,y2=line[0]

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('houghlines3.jpg',img)
plt.subplot(223),plt.imshow(img)
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

