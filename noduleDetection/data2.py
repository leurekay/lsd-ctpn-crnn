#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:31:00 2018

@author: ly
"""

import numpy as np
import pandas as pd
import os
import time
import collections
import random
from layers import iou
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate

import config

config=config.config


import sys
sys.path.append('preprocessing')
import prepare


def batch_transform(x,info,clean_img_shape):
    # input :z-y-x order
    #output: z-y-x order
    box=[]
    shape=info[2]
    for xx in x:
        xxx=prepare.simple_label_transform(xx,info)[:3]
        
        #set the points outside the image to nan.
        for i in range(len(shape)):
            if xxx[0,i]<=0 or xxx[0,i]>=clean_img_shape[i]:
                xxx[0,i]=np.nan
                
        box.append(xxx)
#        print clean_img_shape
    return np.concatenate(box)






class FPR():
    def __init__(self, data_dir,candidates_path,config, phase = 'train'):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase     
        self.stride = config['stride']       


        self.isScale = config['aug_scale']

        self.augtype = config['augtype']
        self.pad_value = config['pad_value']

        self.crop_size = config['crop_size']



        data_dir=os.path.join(data_dir,phase)
        self.data_dir=data_dir
        patient_list=os.listdir(data_dir)
        
        ct_list=filter(lambda x:x.split('_')[-1]=='clean.npy',patient_list)
        label_list=filter(lambda x:x.split('_')[-1]=='label.npy',patient_list)
        id_list_by_ct=map(lambda x:x.split('_')[0],ct_list)
        id_list_by_label=map(lambda x:x.split('_')[0],label_list)
        id_list=set.intersection(set(id_list_by_ct),set(id_list_by_label))
        idcs=list(id_list)
        idcs.sort()
        self.uids=idcs
        
        df_candidates=pd.read_csv(candidates_path)
        df_candidates=df_candidates[df_candidates['seriesuid'].isin(idcs)]
        
        uid_info_Dict={}
        
        for idx in idcs:
            clean_img=np.load(os.path.join(data_dir, '%s_clean.npy' %idx))
            clean_img_shape=clean_img.shape[1:]
            info=np.load(os.path.join(data_dir, '%s_info.npy' %idx))
            uid_info_Dict[idx]=info
            
            origin=info[0]
            extend=info[4]
            origin=np.flip(origin,0)
            extend=np.flip(extend,0)
            
            batches= df_candidates.loc[df_candidates['seriesuid']==idx,['coordX','coordY','coordZ']]
            
            batches=batches.values[:,::-1]
            
            batches=batch_transform(batches,info,clean_img_shape)[:,:-1]
            batches=batches[:,::-1]
            
            df_candidates.loc[df_candidates['seriesuid']==idx,['coordX','coordY','coordZ']]=batches
        
#        df_candidates.loc[((df_candidates['coordX']<0) | (df_candidates['coordY']<0) | (df_candidates['coordZ']<0)),['class']]=np.nan
        df_candidates.dropna(inplace=True)
        
        self.uid_info_Dict=uid_info_Dict
        
        
        self.df_candidates=df_candidates
        
        df_pos=df_candidates[df_candidates['class']==1]
        df_neg=df_candidates[df_candidates['class']==0]
        
        p_indexs=self.p_indexs=df_pos.index.values
        n_indexs=self.n_indexs=df_neg.index.values
        
        ratio=int(len(n_indexs) / len(p_indexs) )
        
        
        if phase=='train':
       
            indexs=[p_indexs for i in range(ratio)]
            indexs.append(n_indexs)
            indexs=np.concatenate(indexs)
            np.random.shuffle(indexs)
            self.indexs=indexs[:100000]
            
        else:
            indexs=df_candidates.index.values
            self.indexs=indexs
        
        self.len=len(self.indexs)
        
        
            
        
    def get_item(self,index):
        cube_size=np.array([32,32,32],'int')
        margin=10
        
        
        
            
        entry=self.df_candidates.loc[index]
        entry=list(entry.values)
        
        
        path=os.path.join(self.data_dir,entry[0]+'_clean.npy')
        img=np.load(path)
        img_shape=img.shape
        
        class_label=entry[4]
        xyz=entry[1:4]
        xyz=list(reversed(xyz))
        xyz=np.array(xyz,'int')

        if self.phase=='train'  and  index in self.p_indexs:
            start=xyz-np.random.randint(margin,32-margin,[3,])
        else:
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
        
        if cube.shape != (1,32,32,32):
            raise Exception(img_shape,start,end,entry)
        
        return cube,class_label
            
        


if __name__=='__main__':
     data_dir='/data/lungCT/luna/temp/luna_npy'
     label_path='/data/lungCT/luna/candidates.csv'   
     data=FPR(data_dir,label_path,config,phase='val')

     df=data.df_candidates
     a=df[((df['coordX']<0) | (df['coordY']<0) | (df['coordZ']<0))]    
#     ooxx=data.get_item(13)
     
     
#     for i in range(1000000):
#         p=data.get_item(np.random.choice([True,False]))
#         if i%100==0:
#             print (i)
    
# =============================================================================
#     data_dir='/data/lungCT/luna/temp/luna_npy'
#     label_path='/data/lungCT/luna/pull_aiserver/candidates.csv'
#     annotations_path='/data/lungCT/luna/annotations.csv'
#     df_a=pd.read_csv(annotations_path)
#     df_xyz01=pd.read_csv(label_path)
#     data=FPR(data_dir,label_path,config)
#     df=data.df_candidates
#     a=df[((df['coordX']<0) | (df['coordY']<0) | (df['coordZ']<0)) & (df['class']==1)]
#     aa=a.groupby('seriesuid').count()
#     uid='1.3.6.1.4.1.14519.5.2.1.6279.6001.801945620899034889998809817499'
#     uid='1.3.6.1.4.1.14519.5.2.1.6279.6001.534083630500464995109143618896'
#     info=np.load(os.path.join(data_dir,uid+'_info.npy'))
#     label=np.load(os.path.join(data_dir,uid+'_label.npy'))
#     world_xyz=df_a[df_a['seriesuid']==uid].iloc[:,1:4]
#     xyz01=df_xyz01[(df_xyz01['seriesuid']==uid) & (df_xyz01['class']==1)].iloc[:,1:4]
#     jj=info[0,:]+info[1,:]+label[0,:3]
# =============================================================================
