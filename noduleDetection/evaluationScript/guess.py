#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:15:20 2018

@author: ly
"""

import numpy as np
import pandas as pd
import scipy

annotations_filename='/data/lungCT/luna/evaluationScript/annotations/annotations.csv'
seriesuids_filename='/data/lungCT/luna/evaluationScript/annotations/seriesuids.csv'
results_filename              = '/data/lungCT/luna/evaluationScript/exampleFiles/submission/guess.csv'


df_id=pd.read_csv(seriesuids_filename,header=None)
df_an=pd.read_csv(annotations_filename)


ids=list(df_id[0])


def gauss_guess(bbox):
    a=np.copy(bbox)
    xyz=a[:3]
    r=a[-1]
    coord=np.random.normal(xyz,[r,r,r])
    dist_vec=coord-xyz
    dist=np.linalg.norm(dist_vec)
#    prob=np.exp(-0.4*dist/r)
    prob=np.random.uniform()
    coord=coord.tolist()
    return coord+[prob]

count=0
pred_df=pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','probability'])
for ooxx,uid in enumerate(ids):
    print (ooxx)
    sel=df_an[df_an['seriesuid']==uid]
    len_sel=sel.shape[0]
    if len_sel == 0:
        continue
        
    for i in range(len_sel):
        ii=list(sel.iloc[i])
        for _ in range(np.random.randint(50,200)):
            gen=[ii[0]]+gauss_guess(ii[1:])   
            
        
            pred_df.loc[count,:]=gen
            count+=1
        
pred_df.to_csv(results_filename,index=None)