#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 15:44:17 2018

@author: ly
"""

import os
import time
import re

import numpy as np
import matplotlib.pyplot as plt
#plt.switch_backend('agg')

save_dir='images'

def fuck(data_dir):
    patient_list=os.listdir(data_dir)
    
    ct_list=filter(lambda x:x.split('_')[-1]=='clean.npy',patient_list)
    label_list=filter(lambda x:x.split('_')[-1]=='label.npy',patient_list)
    id_list_by_ct=map(lambda x:x.split('_')[0],ct_list)
    id_list_by_label=map(lambda x:x.split('_')[0],label_list)
    id_list=set.intersection(set(id_list_by_ct),set(id_list_by_label))
    idcs=list(id_list)
    idcs.sort()
    
    
    labels = []
    for idx in idcs:
        l = np.load(os.path.join(data_dir, '%s_label.npy' %idx))
        if np.all(l==0):
            l=np.array([])
        labels.append(l)
    
    dic={}
    for item in labels:
        if len(item) in dic:
             dic[len(item)]+=1
        else:
             dic[len(item)]=1
    
    
    
    box=[]
    for item in labels:
        if len(item)>0:
            for nod in item:
                box.append(nod[3])
    return box,dic



box_train,dic_train=fuck(data_dir='/data/lungCT/luna/temp/luna_npy/train')
box_val,dic_val=fuck(data_dir='/data/lungCT/luna/temp/luna_npy/val')

fig, ax = plt.subplots(figsize=[18,8])
plt.subplot(2, 2, 1)
plt.hist(box_train,bins=50,label='train-loss')

plt.subplot(2, 2, 3)
plt.hist(box_val,bins=50)


bar_width = 0.4

opacity = 0.5
error_config = {'ecolor': '0.3'}


plt.subplot(1, 2, 2)
rects1 = plt.bar(np.array(dic_train.keys())-bar_width/2, dic_train.values(), bar_width,
                alpha=opacity, color='r',
                label='Men')
rects2 = plt.bar(np.array(dic_val.keys())+bar_width/2, dic_val.values(), bar_width,
                alpha=opacity, color='g',
                label='Men')
ax.legend()
plt.savefig(os.path.join(save_dir,'distribution.jpg'))