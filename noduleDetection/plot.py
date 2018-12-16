#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 17:12:14 2018

@author: ly
"""

import os
import time
import re

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')


model_dir='/data/lungCT/luna/temp/savemodel/model3'
save_dir='/data/lungCT/luna/temp/image'


saved_models=os.listdir(model_dir)
saved_models=[x for x in saved_models if x.endswith('.h5')]
saved_models.sort()
epoch_ids=map(lambda x : int(x.split('-')[0].split(':')[-1]),saved_models)
epoch_file_dict=zip(epoch_ids,saved_models)
epoch_file_dict=dict(epoch_file_dict)

box=[]
for key in epoch_ids:
    filename=epoch_file_dict[key]
    loss=re.findall('[0-9]+[.][0-9]+',filename)
    box.append(np.array(loss).reshape([1,-1]).astype('float32'))
box=np.concatenate(box)





shapeList=['o-','x-','s-','|-','x-']


title=model_dir.split('/')[-1]+' training curve'
# Create plots with pre-defined labels.
fig, ax = plt.subplots(figsize=[12,9])
ax.plot(epoch_ids, box[:,0], 'g-', label='train-loss',linewidth=2.5)
ax.plot(epoch_ids, box[:,1], 'g--', label='train-cls-loss',linewidth=3)

ax.plot(epoch_ids, box[:,2], 'c-', label='val-loss',linewidth=2.5)
ax.plot(epoch_ids, box[:,3], 'c--', label='val-cls-loss',linewidth=3)

plt.ylim(0,1.6)
plt.xlabel('epoch', fontsize=25)
plt.ylabel('loss', fontsize=25)
plt.title(title,fontsize=30)


legend = ax.legend(loc='upper right', shadow=True, fontsize=22)

# Put a nicer background color on the legend.
#legend.get_frame().set_facecolor('#00DDAA')

plt.show()
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
fig.savefig(os.path.join(save_dir,title+'-'+str(int(time.time()))+'-'+'.png'))



