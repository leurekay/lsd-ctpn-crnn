#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:20:17 2018

@author: ly
"""
from keras.backend.tensorflow_backend import set_session

import data

import layers


import time
import os
import shutil
from PIL import Image

import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model,load_model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

import numpy as np
import pandas as pd
import config
import argparse



tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
set_session(tf.Session(config=tfconfig))


#load config
config=config.config





EPOCHS=100
InitialEpoch=0
data_dir=config['data_prep_dir'] #including  *_clean.npy and *_label.npy
model_dir=config['model_dir']



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


#copy config file to model dir
timstamp=int(time.time())
config_backup=os.path.join(model_dir,str(InitialEpoch)+'.config')
layers_backup=os.path.join(model_dir,str(InitialEpoch)+'.layers')

shutil.copy('config.py',config_backup)
shutil.copy('layers.py',layers_backup)






#judge how to load model which restore or Start from scratch
if InitialEpoch==0:
    model=layers.n_net()
else:
    if InitialEpoch in  epoch_file_dict:
        model_path=os.path.join(model_dir,epoch_file_dict[InitialEpoch])
        model=load_model(model_path,compile=False)
        print ("************\nrestore from %s\n************"%model_path)
    else:
        raise Exception("epoch-%d has not been trained"%InitialEpoch)







#custumn metric function
#def metric1(y_true, y_pred):
#    return tf.reduce_mean(tf.reduce_sum(y_true*y_pred,axis=1),axis=0)
loss_cls=layers.loss_cls
recall=layers.recall

nohard=layers.nohard
cls_nohard=layers.cls_nohard


#custumn loss function
myloss=layers.myloss

#compile
model.compile(optimizer=config['optimizer'],
              loss=myloss,
              metrics=[loss_cls,recall,nohard,cls_nohard])







#checkpoint for callback
checkpoint=ModelCheckpoint(filepath=os.path.join(model_dir,'epoch:{epoch:03d}-loss-cls-recall:{loss:.3f}-{loss_cls:.3f}-{recall:.3f}-{nohard:.3f}-{cls_nohard:.3f}-valloss_cls_recall:{val_loss:.3f}-{val_loss_cls:.3f}-{val_recall:.3f}-{val_nohard:.3f}-{val_cls_nohard:.3f}.h5'), 
                                monitor='val_loss', 
                                verbose=0, 
                                save_best_only=False, 
                                save_weights_only=False, 
                                period=1)

#controled learning rate for callback  
#def lr_decay(epoch):
#    lr=0.01
#    if epoch>2:
#        lr=0.01
#    if epoch>4:
#        lr=0.001
#    if epoch>10:
#        lr=0.0001
#    return lr
from config import lr_decay 
lr_scheduler = LearningRateScheduler(lr_decay)


#callback list
callback_list = [checkpoint,lr_scheduler]








# numbers of sample correspoding train and val
train_dataset=data.DataBowl3Detector(data_dir,config,phase='train')
train_samples=train_dataset.__len__()
val_dataset=data.DataBowl3Detector(data_dir,config,phase='val')
val_samples=val_dataset.__len__()




#read data and processing by CPU ,during training.
#Don't load all data into memory at onece!
def generate_arrays(phase,shuffle=True):
    dataset=data.DataBowl3Detector(data_dir,config,phase=phase)
    n_samples=dataset.__len__()
    ids=np.array(np.arange(n_samples))

    while True:
        if shuffle:
            np.random.shuffle(ids)
        for i in ids:
            x, y ,coord = dataset.__getitem__(i)
            x=np.expand_dims(x,axis=0)
            y=np.expand_dims(y,axis=0)
            coord=np.transpose(coord,[3,1,2,0])
            coord=np.expand_dims(coord,axis=0)
            yield ([x,coord],y)


#training
model.fit_generator(generate_arrays(phase='train'),
                    steps_per_epoch=train_samples,
                    epochs=EPOCHS,initial_epoch=InitialEpoch,
                    verbose=1,
                    callbacks=callback_list,
                    validation_data=generate_arrays('val'),
                    validation_steps=val_samples,
                    workers=4,)

















#
##loss function
#myloss=layers.myloss
#
#loss_cls=layers.loss_cls
#
#
#
#def lr_decay(epoch):
#	lr = 0.0001
#	if epoch >= 150:
#		lr = 0.0003
#	if epoch >= 220:
#		lr = 0.00003
#	return lr 
#lr_scheduler = LearningRateScheduler(lr_decay)
#
#
##load model
#if os.path.exists(SAVED_MODEL):
#    print ("*************************\n restore model\n*************************")
#    model=load_model(SAVED_MODEL)  
#else:
#    model=layers.n_net()
#    
#adam=keras.optimizers.Adam(lr=1000, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#sgd=keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(optimizer=adam,
#              loss=myloss,
#              metrics=[loss_cls])
#
#
#
## numbers of sample correspoding train and val
#train_dataset=data.DataBowl3Detector(data_dir,config,phase='train')
#train_samples=train_dataset.__len__()
#val_dataset=data.DataBowl3Detector(data_dir,config,phase='val')
#val_samples=val_dataset.__len__()
#
#
#
##call back.   save model named by (time,train_loss,val_loss)
#class EpochSave(keras.callbacks.Callback):
#    def on_epoch_begin(self,epoch, logs={}):
#        self.losses = []
#        
#
#    def on_epoch_end(self, epoch, logs={}):
#        time_now=int(time.time())
#        train_loss=logs.get('loss')
#        val_loss=logs.get('val_loss')
#        self.losses.append([train_loss,val_loss])
##        print ('epoch:',epoch,'    ',self.losses)
#        file_name=str(time_now)+'_'+time.strftime('%Y%m%d-%H:%M:%S')+'_train_%.3f_val_%.3f.h5'%(train_loss,val_loss)
#        model.save(os.path.join(model_dir,file_name),include_optimizer=False)
#epoch_save = EpochSave()
#
#
##read data and processing by CPU ,during training.
##Don't load all data into memory at onece!
#def generate_arrays(phase,shuffle=True):
#    dataset=data.DataBowl3Detector(data_dir,config,phase=phase)
#    n_samples=dataset.__len__()
#    ids=np.array(np.arange(n_samples))
#
#    while True:
#        if shuffle:
#            np.random.shuffle(ids)
#        for i in ids:
#            x, y ,_ = dataset.__getitem__(i)
#            x=np.expand_dims(x,axis=0)
#            y=np.expand_dims(y,axis=0)
#            yield (x, y)
#
#model.save(SAVED_MODEL,include_optimizer=False)
#model.fit_generator(generate_arrays(phase='train'),
#                    steps_per_epoch=train_samples,epochs=EPOCHS,
#                    verbose=1,callbacks=[epoch_save,lr_scheduler],
#                    validation_data=generate_arrays('val'),
#                    validation_steps=val_samples,
#                    workers=4,)
#
#
#
#model.save(SAVED_MODEL,include_optimizer=False)