#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 10:25:58 2018

in each  gt_img_3.txt ,the code of head 3 char was wrong
so ,i will delete these 3 characters

@author: dirac
"""

import os
import shutil

gt_path = '/data/ctpn/label'
gt_copy_path='/data/ctpn/label_copy'
if not os.path.exists(gt_copy_path):
    os.makedirs(gt_copy_path)

#shutil.copytree(gt_path,gt_copy_path,True)


files = os.listdir(gt_path)
files.sort()

for name in files:
    path=os.path.join(gt_path,name)
    path_copy=os.path.join(gt_path,name)
    with open(path, 'r') as f:
        txt = f.read()
        txt=txt[3:]
    with open(path_copy,'w') as ff:
        ff.write(txt)

    
    
    