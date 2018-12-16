#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:52:58 2018

@author: ly
"""

from keras.backend.tensorflow_backend import set_session




import os
import time
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



#tfconfig = tf.ConfigProto(device_count={'cpu':0})
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
set_session(tf.Session(config=tfconfig))


#load config
config=config.config





#command line parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='val', type=str, 
                    help='train,val,test')
parser.add_argument('--model', default='model6',
                    type=str, help='model')
parser.add_argument('--epoch', default=89,
                    type=int, help='epoch')
parser.add_argument('--savepath', default=None,
                    type=str, help='savepath')


args = parser.parse_args()

data_phase=args.phase
model_n=args.model
epoch=args.epoch
save_path=args.savepath

nms_th=config['nms_th']
pos_th=config['pos_th']
data_dir=config['data_prep_dir']
ctinfo_path=config['ctinfo_path']
pred_save_dir=config['pred_save_dir']
if not os.path.exists(pred_save_dir):
    os.makedirs(pred_save_dir)
model_dir='/data/lungCT/luna/temp/savemodel/'

if not save_path:
    save_path=os.path.join(pred_save_dir,model_n+'-epoch'+str(epoch)+'-'+data_phase+'-'+str(int(time.time()))+'.csv')

    


model_dir=os.path.join(model_dir,model_n)
if not os.path.exists(model_dir):
    raise Exception('Directory %s does not exist!'%model_n)
filelist=os.listdir(model_dir)
filelist=[x for x in filelist if x.endswith('.h5')]
filelist.sort()
epoch_id_list=map(lambda x:int(x.split('-')[0].split(':')[-1]),filelist)
epoch_file_dict=dict(zip(epoch_id_list,filelist))
if epoch not in epoch_file_dict:
    raise Exception('epoch-%d does not exist in %s!'%(epoch,model_n))

filename=epoch_file_dict[epoch]
model_path=os.path.join(model_dir,filename)

ctinfo=pd.read_csv(ctinfo_path,index_col='seriesuid')
uid_origin_dict={}
uids=list(ctinfo.index)
for uid in uids:
    origin=list(ctinfo.loc[uid])[:3]
    uid_origin_dict[uid]=origin
   
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def localVoxel_To_globalWorld(label,origin,extend):
    """
    input [z,y,x]  same as the input of neual net computing 
    return [x,y,z] same as the annotation.csv
    
    
    !!!!!!!!!this function is wrong!!!!
    because i didn't consider the flip case
    
    """
    ret= label+origin+extend
    return ret[[2,1,0]]


#load model
if os.path.exists(model_path):
    print ("****************************************************\n restore model from %s\n****************************************************"%model_path)
    model=load_model(model_path,compile=False)  
else:
    raise Exception("no model")


dataset=data.DataBowl3Detector(data_dir,config,phase=data_phase)
get=layers.GetPBB(data.config)

uids=dataset.uids
len_uids=len(uids)
labels=dataset.sample_bboxes

pred_df=pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','probability'])
count=0

for index in range(len_uids):
    time_s=time.time()
    
    image,patch_box,bboxes,origin,extend = dataset.package_patches(index)
    uid=dataset.uids[index]
#    origin=uid_origin_dict[uid]
    
    _,xsize,ysize,zsize,_=image.shape
 
    
    box=[]
    for i,(patch,coord,start) in enumerate(patch_box):
#        print (i)
        sx,sy,sz=start
   
        pred=model.predict([patch,coord])
        pred=pred[0]
        pred[:,:,:,:,0]=sigmoid(pred[:,:,:,:,0])
         
        pos_pred=get.__call__(pred,pos_th)
    #    pos_pred=layers.nms(pos_pred,nms_th)
        pos_pred[:,1]+=sx
        pos_pred[:,2]+=sy
        pos_pred[:,3]+=sz
        
#        pos_pred[:,1]+=origin[0]
#        pos_pred[:,2]+=origin[1]
#        pos_pred[:,3]+=origin[2]
        box.append(pos_pred)
    box=np.concatenate(box)
    box_nms=layers.nms(box,nms_th)
#    box_nms_world=localVoxel_To_globalWorld(box_nms[1:],origin,extend)
#    box_nms_world=np.array([box_nms[0]]+box_nms_world.tolist())
    time_e=time.time()
    maxprob=-9.0 if len(box_nms)==0 else box_nms[0][0]
    maxdiameter=0 if bboxes.shape[0]==0 else max(bboxes[:,3])
    print ('%s-%03d, nodules:%2d, max_D:%.2f, maxprob:%.2f, pos:%3d, pos_nms:%3d, patches:%3d, shape-[%3d,%3d,%3d], time:%.1fs'%(data_phase,index,bboxes.shape[0],maxdiameter,maxprob,box.shape[0],box_nms.shape[0],len(patch_box),xsize,ysize,zsize,time_e-time_s))

    for entry in box_nms:
        pred_df.loc[count]=[uid,entry[1],entry[2],entry[3],entry[0]]
        count+=1

pred_df.to_csv(save_path,index=None)  
print ("output %d postive predictions"%pred_df.shape[0])
print ("csv file saved in %s"%save_path)
    


    









