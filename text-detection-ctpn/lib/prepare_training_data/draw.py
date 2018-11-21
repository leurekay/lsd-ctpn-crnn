#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 12:00:08 2018

@author: dirac
"""

from __future__ import print_function

import cv2
import glob
import os
import shutil
import sys

import numpy as np
import tensorflow as tf


def draw_boxes(src,dst, boxes):
    img = cv2.imread(src)
    color = (255, 0, 0)
    color_list=[(255,0,0),(0,255,0),(0,0,255)]
    for i,box in enumerate(boxes):
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color_list[i%3], 1)
        cv2.line(img, (int(box[2]), int(box[3])), (int(box[4]), int(box[5])), color_list[i%3], 1)
        cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color_list[i%3], 1)
        cv2.line(img, (int(box[6]), int(box[7])), (int(box[0]), int(box[1])), color_list[i%3], 1)
    cv2.imwrite(dst, img)


def draw_boxes_byTxt(src,dst,txtPath):
    with open(txtPath,'r') as f:
        lines=f.readlines()
        
        if lines[0].startswith('text'):
            for i,line in enumerate(lines):
                lines[i]=line.strip('text').strip().split()
                lines[i]=[lines[i][0],lines[i][1],lines[i][2],lines[i][1],
                          lines[i][2],lines[i][3],lines[i][0],lines[i][3]]
        else:
            for i,line in enumerate(lines):
                lines[i]=line.strip().split(',')[:8]          
    boxes=np.array(lines,dtype='int16')
    
    draw_boxes(src,dst, boxes)
                
    return lines
    
if __name__=='__main__':
    draw_dir='draw'
    
    if not os.path.exists(draw_dir):
        os.mkdir(draw_dir)
        
    
    image_dir='/data/ctpn/image'
    label_dir='/data/ctpn/label'
    
    image2_dir='re_image'
    label2_dir='label_tmp'
    
    image_list=os.listdir(image_dir)
    image_list.sort()
    for filename in image_list:
        print (filename)
        name, ext = os.path.splitext(filename)
        src=os.path.join(image_dir,filename)
        dst=os.path.join(draw_dir,filename)
        txtPath=os.path.join(label_dir,'gt_'+name+'.txt')
        draw_boxes_byTxt(src,dst,txtPath)
        
        src2=os.path.join(image2_dir,filename)
        dst2=os.path.join(draw_dir,name+'_.jpg')
        txtPath2=os.path.join(label2_dir,name+'.txt')
        draw_boxes_byTxt(src2,dst2,txtPath2) 
       
    
#    txtPath='label_tmp/img_95.txt'
#    ll=draw_boxes_byTxt(src,dst,txtPath)

    
    
