#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 11:19:18 2018

@author: ly
"""

import numpy as np
import pandas as pd

import os
import sys
sys.path.append('../../preprocessing')
import prepare

df_=pd.read_csv('annotations_excluded.csv')

df=df_.copy()

luna_npy_dir='/data/lungCT/luna/temp/luna_npy'

uids=list(df.seriesuid.unique())

uid=uids[0]
info_path=os.path.join(luna_npy_dir,uid+'_info.npy')
info=np.load(info_path)


length=df.shape[0]

for index in range(length):
    uid=df.iloc[index,0]
    entry=df.iloc[index,1:4]
    info_path=os.path.join(luna_npy_dir,uid+'_info.npy')
    info=np.load(info_path)

    xx=np.array(list(entry))
    
    xxx=prepare.simple_label_transform(xx,info)[:,:3]
    df.iloc[index,1:4]=xxx.flatten()

df.to_csv('annotations_excluded_.csv',index=None)