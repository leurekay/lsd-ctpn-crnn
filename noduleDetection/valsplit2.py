#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:49:21 2018

@author: ly
"""

import numpy as np
import pandas as pd


import shutil
import os
import time
import random



import data
from  config import config


data_dir=config['data_prep_dir']
shuffle=config['data_split_shuffle']

ratio_train=config['train_val_test_ratio'][0]
ratio_val=config['train_val_test_ratio'][1]





patient_list=os.listdir(data_dir)

ct_list=filter(lambda x:x.split('_')[-1]=='clean.npy',patient_list)
label_list=filter(lambda x:x.split('_')[-1]=='label.npy',patient_list)
id_list_by_ct=map(lambda x:x.split('_')[0],ct_list)
id_list_by_label=map(lambda x:x.split('_')[0],label_list)
id_list=set.intersection(set(id_list_by_ct),set(id_list_by_label))
idcs=list(id_list)

if shuffle:
    random.shuffle(idcs)
else:
    idcs.sort()



length=len(idcs)
cutpoint1=int(ratio_train*length)
cutpoint2=int((ratio_train+ratio_val)*length)
train=idcs[:cutpoint1]
val=idcs[cutpoint1:cutpoint2]
test=idcs[cutpoint2:]

dic={'train':train,'val':val,'test':test}
display=1
for phase in ['train','val','test']:
    uuids=dic[phase]
    labels=[]
    phase_dir=os.path.join(data_dir,phase)

    if  os.path.exists(phase_dir):
        shutil.rmtree(phase_dir)
    os.makedirs(phase_dir)
    for uuid in uuids:
        print (display)
        display+=1
        src1=os.path.join(data_dir,uuid+'_clean.npy')
        dis1=os.path.join(phase_dir,uuid+'_clean.npy')
        src2=os.path.join(data_dir,uuid+'_label.npy')
        dis2=os.path.join(phase_dir,uuid+'_label.npy')
        src3=os.path.join(data_dir,uuid+'_info.npy')
        dis3=os.path.join(phase_dir,uuid+'_info.npy')        
        shutil.copy(src1,dis1)
        shutil.copy(src2,dis2)
        shutil.copy(src3,dis3)
        
        l=np.load(src2)
        if np.all(l==0):
            l=np.array([])
        labels.append(l)
    print ('=====================================')



    df=pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','diameter_mm'])
    count=0
    for index,uid in enumerate(uuids):
        label=labels[index]
        if len(label)==0:
            continue
        for entry in label:
            df.loc[count]=[uid,entry[0],entry[1],entry[2],entry[3]]
            count+=1
    save_path_anno=os.path.join(phase_dir,phase+'_anno.csv')
    df.to_csv(save_path_anno,index=None)
    time.sleep(2)
