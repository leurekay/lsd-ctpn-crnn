#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:07:17 2018

@author: ly
"""

import os

import argparse

import numpy as np
import pandas as pd



import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model,load_model


import data
import config
import layers

config=config.config

stage1_submit_path='/data/lungCT/luna/temp/submit/free-model6-epoch46-val.csv'

model_path='/data/lungCT/luna/temp/savemodel_fpr/model4/epoch:002-trainloss:0.076-valloss:0.454.h5'

data_dir=config['data_prep_dir']



#command line parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='val', type=str, 
                    help='train,val,test')
args = parser.parse_args()
data_phase=args.phase





#load model
if os.path.exists(model_path):
    print ("****************************************************\n restore model from %s\n****************************************************"%model_path)
    model=load_model(model_path,compile=False)  
else:
    raise Exception("no model")
    
    

df=pd.read_csv(stage1_submit_path)

df['p2']=0



for i in range(df.shape[0]):
    df.iloc[i,5]=i
    point=list(df.iloc[i,1:4].values)
    uid=df.iloc[i,0]
    old_prob=df.iloc[i,4]
    
    cube_size=np.array([32,32,32],'int')
    
 
    
    path=os.path.join(data_dir,uid+'_clean.npy')
    img=np.load(path)
    img_shape=img.shape
  
    xyz=point
    
    xyz=np.array(xyz,'int')
    if (xyz[0]>img_shape[1]) or (xyz[1]>img_shape[2]) or (xyz[2]>img_shape[3]):
        df.iloc[i,5]=old_prob
        print (xyz,img_shape)
        continue
    
    start=xyz-cube_size/2
    start=start.astype('int')
    comp=np.vstack((start,np.array([0,0,0])))
    start=np.max(comp,axis=0)
    end=start+cube_size
    end=end.astype('int')
    comp1=np.vstack((end,np.array(img_shape[1:])))
    end=np.min(comp1,axis=0)
#        print (end)
    
    
    
    cube=img[:,start[0]:end[0],start[1]:end[1],start[2]:end[2]]
    delta=32-(end-start)
    
    cube=np.pad(cube, ((0,0),(0,delta[0]),(0,delta[1]),(0,delta[2])), 
                         'constant', constant_values=170)
    
    cube = (cube.astype(np.float32)-128)/128
    cube=np.expand_dims(cube,-1)
    pred=model.predict(cube)
    new_prob=pred[0][0]
    df.iloc[i,5]=new_prob
    if pred[0][0]>0.4:
        print ('old_propred:%.3f , new_prob:%.3f'%(old_prob,new_prob))
        
df['probability'],df['p2']=df['p2'],df['probability']

save_path=stage1_submit_path.replace('.csv','-fpr.csv')
df.to_csv(save_path,index=None)  
print ("csv file saved in %s"%save_path)