#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 18:20:38 2018

@author: ly
"""

from keras.backend.tensorflow_backend import set_session

import data
import data2

import layers


import time
import os
import shutil
from PIL import Image

import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model,load_model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint,Callback
from keras.utils import np_utils

from sklearn.metrics import roc_auc_score,classification_report,log_loss,roc_curve


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import config
import argparse



tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
set_session(tf.Session(config=tfconfig))


#load config
config=config.config



BATCH_SIZE=64

EPOCHS=100
InitialEpoch=0
data_dir=config['data_prep_dir'] #including  *_clean.npy and *_label.npy
model_dir=config['model_dir_fpr']
candidate_path=config['candidate_path']    


#command line parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--startepoch', default=InitialEpoch, type=int, 
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--modeldir', default=model_dir, type=str)
args = parser.parse_args()
InitialEpoch=args.startepoch
model_dir=args.modeldir



#deal saved model dir. Mapping epoch id to file
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    epoch_file_dict={}
else:
    saved_models=os.listdir(model_dir)
    saved_models=[x for x in saved_models if x.endswith('.h5')]
    epoch_ids=map(lambda x : int(x.split('-')[0].split(':')[-1]),saved_models)
    epoch_file_dict=zip(epoch_ids,saved_models)
    epoch_file_dict=dict(epoch_file_dict)
    
    
    
#judge how to load model which restore or Start from scratch
if InitialEpoch==0:
    model=layers.fpr_net()
else:
    if InitialEpoch in  epoch_file_dict:
        model_path=os.path.join(model_dir,epoch_file_dict[InitialEpoch])
        model=load_model(model_path,compile=False)
        print ("************\nrestore from %s\n************"%model_path)
    else:
        raise Exception("epoch-%d has not been trained"%InitialEpoch)




#compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


#checkpoint for callback
checkpoint=ModelCheckpoint(filepath=os.path.join(model_dir,'epoch:{epoch:03d}-trainloss:{loss:.3f}-valloss:{val_loss:.3f}.h5'), 
                                monitor='val_loss', 
                                verbose=0, 
                                save_best_only=False, 
                                save_weights_only=False, 
                                period=1)


#controled learning rate for callback  
def lr_decay(epoch):
    lr=0.001
    if epoch>2:
        lr=0.001
    if epoch>5:
        lr=0.0001
    if epoch>10:
        lr=0.00001
    return lr
lr_scheduler = LearningRateScheduler(lr_decay)




def ploter(x,y,savepath):
    fig, ax = plt.subplots(figsize=[12,9])
    ax.plot(x,y, 'g-', label='curve',linewidth=2.5)
    
    plt.ylim(0,1.2)
    plt.xlabel('FP', fontsize=25)
    plt.ylabel('TP', fontsize=25)
    plt.title('kkkkkkk',fontsize=30)
    
    
#    legend = ax.legend(loc='upper right', shadow=True, fontsize=22)
    
    
    fig.savefig(savepath)


#AUC callback
class RocAucEvaluation(Callback):
    def __init__(self,  interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.val=[]
    def gen(self,phase,batch_size,shuffle) :
        
        for (x_batch,y_batch) in generate_arrays(phase,batch_size,shuffle):
            self.val.append(y_batch)
            yield x_batch
            
            
        
    def on_epoch_end(self, epoch, log={}):
        
        
        
        if epoch % self.interval == 0:
            y_pred = self.model.predict_generator(self.gen('val',BATCH_SIZE,shuffle=False),steps=n_val/BATCH_SIZE,verbose=1)
            _=self.val.pop()
            y_val=np.concatenate(self.val) 
            
            score = roc_auc_score(y_val, y_pred)
            loss=log_loss(y_val, y_pred)
            curve=roc_curve(y_val,y_pred,pos_label=1)
            
            self.val=[]
            print('\n ROC_AUC - epoch:%d - score:%.6f - loss:%.6f \n' % (epoch+1, score,loss))
            x=curve[0]
            y=curve[1]
            savepath=os.path.join(model_dir,str(epoch)+'.png')
            ploter(x,y,savepath)
            
            
RocAuc = RocAucEvaluation( interval=1)



#callback list
callback_list = [checkpoint,RocAuc]
#callback_list = [RocAuc]







#read data and processing by CPU ,during training.
#Don't load all data into memory at onece!
def generate_arrays(phase,batch_size,shuffle=True):
    dataset=data2.FPR(data_dir,candidate_path,config,phase)
    indexs=dataset.indexs
        
    while True:
        np.random.shuffle(indexs)
        batches=[]
        for i in range(len(indexs)):
            s=i
            e=s+batch_size
            i=e
            if e<len(indexs):
                batches.append(indexs[s:e])
            elif s<(len(indexs)):
                batches.append(indexs[s:])
        for batch in batches:
            x_batch=[]
            y_batch=[]
            for index in batch:
                x,y=dataset.get_item(index)
                x=np.expand_dims(x,axis=-1)
                x_batch.append(x)
                y_batch.append(y)
                
            x_batch=np.concatenate(x_batch) 
            y_batch=np.array(y_batch)
            yield (x_batch,y_batch)







n_train=data2.FPR(data_dir,candidate_path,config,'train').len
n_val=data2.FPR(data_dir,candidate_path,config,'val').len


#training
model.fit_generator(generate_arrays('train',BATCH_SIZE),
                     steps_per_epoch=n_train/BATCH_SIZE,
                     epochs=100,
                     initial_epoch=InitialEpoch,
                     verbose=1,
                     callbacks=callback_list,
                     validation_data=generate_arrays('val',BATCH_SIZE),
                     validation_steps=n_val/BATCH_SIZE,
                     workers=1,)









##read data and processing by CPU ,during training.
##Don't load all data into memory at onece!
#def generate_arrays(phase,shuffle=True):
#    dataset=data2.FPR(data_dir,candidate_path,config,phase)
#    n_samples=dataset.n_pos
#
#    while True:
#        for i in range(n_samples):
#            box=[]
#            y=[]
#            for _ in range(2):
#                coin=np.random.choice([True,False])
#                x = dataset.get_item(isPos=coin)
#                x=np.expand_dims(x,axis=-1)
#                box.append(x)
#                y.append(int(coin))
#           
#            box=np.concatenate(box,axis=0)
#            y=np.array(y)
##            y = np_utils.to_categorical(y, num_classes=2)
#            yield (box,y)
