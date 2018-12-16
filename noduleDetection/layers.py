#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:04:36 2018

@author: ly
"""

import os
from PIL import Image


import numpy as np
import pandas as pd


from keras.models import Model
from keras.layers import Dense, Dropout, Flatten,Input,Activation,Reshape,Lambda
from keras.layers import Conv2D, MaxPooling2D,MaxPooling3D,Conv3D,Deconv2D,Deconv3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD

from keras.utils import np_utils  
from keras.utils import plot_model  

import resnet3D

import tensorflow as tf
import keras
import keras.backend as K

import config
config=config.config


leaky_alpha=config['leaky_alpha']



def slice_last_dim(x,start,end):
   
    return x[:,:,:,:,start:end]
 
    

def res_block(x,conv_filters,pool_size,pool_strides):
    for i in range(3):  
        x=resnet3D.basic_block2(filters=conv_filters)(x)
        x=Activation(activation='relu')(x)
    x=MaxPooling3D(pool_size=pool_size,strides=pool_strides)(x)    
    return x


def n_net2():
    """
    2nd version ,add coord
    """
    
    input_img = Input(shape=(128,128,128,1))
    coord=Input(shape=(32,32,32,3))
    
    #first 2 conv layers
    x = Conv3D(24, (3, 3,3), padding='same', activation='relu')(input_img)
    x = Conv3D(24, (3, 3,3), strides=(1,1,1),padding='same', activation='relu')(x)
    
    
    #first residual block,including 3 residual units
    """
    resnet block get from github
    https://github.com/leurekay/keras-resnet/blob/master/resnet.py
    """
    x=res_block(x,conv_filters=32,pool_size=(2,2,2),pool_strides=(2,2,2))
    
    #next 3 residual block
    r2=res_block(x,conv_filters=64,pool_size=(2,2,2),pool_strides=(2,2,2))
    r3=res_block(r2,conv_filters=64,pool_size=(2,2,2),pool_strides=(2,2,2))
    r4=res_block(r3,conv_filters=64,pool_size=(2,2,2),pool_strides=(2,2,2))
    
    
    #feedback path
    x=Deconv3D(64,kernel_size=(2,2,2),strides=2)(r4)
    x=concatenate([r3,x])
    x=res_block(x,64,(1,1,1),(1,1,1))
    x=Deconv3D(64,kernel_size=(2,2,2),strides=2)(x)
    
    
    
    #either add coord to net or not
    x=concatenate([r2,x,coord])
    x=Lambda(slice_last_dim,arguments={'start':0,'end':128})(x)
    
    
#    x=concatenate([r2,x])
    x=res_block(x,128,(1,1,1),(1,1,1))
    x= Conv3D(64, (3, 3,3), strides=(1,1,1),padding='same', )(x)
    x=LeakyReLU(alpha=config['leaky_alpha'])(x)
    x=Dropout(0.5)(x)
    x= Conv3D(15, (3, 3,3), strides=(1,1,1),padding='same', )(x)
    x=LeakyReLU(alpha=config['leaky_alpha'])(x)
    x= Reshape((32,32,32,3,5))(x)
    
    #predictions = Dense(10, activation='softmax')(x)
    model=Model(inputs=[input_img,coord],outputs=x )
    return model




def res_block_rewrite(x,conv_filters):
    x=resnet3D.PostRes(x,conv_filters).forward()
    x=resnet3D.PostRes(x,conv_filters).forward()
    x=resnet3D.PostRes(x,conv_filters).forward()
    return x

def n_net3():
    """
    3rd version. rewrite the net in imitation of the source code 
    """
    input_img = Input(shape=(128,128,128,1))
    coord=Input(shape=(32,32,32,3))
    
    #first 2 conv layers
    x = Conv3D(24, (3, 3,3), strides=(1,1,1),padding='same')(input_img)
    x =BatchNormalization()(x)
    x= Activation('relu')(x)
#    x=LeakyReLU(alpha=config['leaky_alpha'])(x)
    x = Conv3D(24, (3, 3,3), strides=(1,1,1),padding='same')(x)
    x =BatchNormalization()(x)
    x= Activation('relu')(x)
#    x=LeakyReLU(alpha=config['leaky_alpha'])(x)
    
    
    x=MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(x)    
    x=res_block_rewrite(x,conv_filters=32)
    
    #next 3 residual block
    r2=MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(x)  
    r2=res_block_rewrite(r2,conv_filters=64)
    
    r3=MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(r2)  
    r3=res_block_rewrite(r3,conv_filters=64)
    
    r4=MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(r3)  
    r4=res_block_rewrite(r4,conv_filters=64)
    
    
#    feedback path
    x=Deconv3D(64,kernel_size=(2,2,2),strides=2)(r4)
    x =BatchNormalization()(x)
    x= Activation('relu')(x)
#    x=LeakyReLU(alpha=config['leaky_alpha'])(x)
    
    x=concatenate([r3,x])
    x=res_block_rewrite(x,64)
    
    x=Deconv3D(64,kernel_size=(2,2,2),strides=2)(x)
    x =BatchNormalization()(x)
    x= Activation('relu')(x)
#    x=LeakyReLU(alpha=config['leaky_alpha'])(x)
    
    
    #either add coord to net or not
    x=concatenate([r2,x,coord])
    x=Lambda(slice_last_dim,arguments={'start':0,'end':128})(x)

    
    
    
    x=res_block_rewrite(x,128)
    
    #2 convolution
    x=Dropout(0.5)(x)
    x = Conv3D(64, (1, 1,1), strides=(1,1,1),padding='same')(x)
    x= Activation('relu')(x)  
#    x=LeakyReLU(alpha=config['leaky_alpha'])(x)
    x = Conv3D(15, (1, 1,1), strides=(1,1,1),padding='same')(x)
#    x=LeakyReLU(alpha=config['leaky_alpha'])(x)
    
    x= Reshape((32,32,32,3,5))(x)
    
    
    model=Model(inputs=[input_img,coord],outputs=x )
    return model
    


def n_net_without_coord():#origin version, no coord, only one input
    """
    1st version. 
    """
    input_img = Input(shape=(128,128,128,1))
    
    #first 2 conv layers
    x = Conv3D(24, (3, 3,3), padding='same', activation='relu')(input_img)
    x = Conv3D(24, (3, 3,3), strides=(1,1,1),padding='same', activation='relu')(x)
    
    
    #first residual block,including 3 residual units
    """
    resnet block get from github
    https://github.com/leurekay/keras-resnet/blob/master/resnet.py
    """
    x=res_block(x,conv_filters=32,pool_size=(2,2,2),pool_strides=(2,2,2))
    
    #next 3 residual block
    r2=res_block(x,conv_filters=64,pool_size=(2,2,2),pool_strides=(2,2,2))
    r3=res_block(r2,conv_filters=64,pool_size=(2,2,2),pool_strides=(2,2,2))
    r4=res_block(r3,conv_filters=64,pool_size=(2,2,2),pool_strides=(2,2,2))
    
    
    #feedback path
    x=Deconv3D(64,kernel_size=(2,2,2),strides=2)(r4)
    x=concatenate([r3,x])
    x=res_block(x,64,(1,1,1),(1,1,1))
    x=Deconv3D(64,kernel_size=(2,2,2),strides=2)(x)
    x=concatenate([r2,x])
    x=res_block(x,128,(1,1,1),(1,1,1))
    x= Conv3D(64, (3, 3,3), strides=(1,1,1),padding='same', )(x)
    x=LeakyReLU(alpha=0.3)(x)
    x=Dropout(0.5)(x)
    x= Conv3D(15, (3, 3,3), strides=(1,1,1),padding='same', )(x)
    x=LeakyReLU(alpha=0.3)(x)
    x= Reshape((32,32,32,3,5))(x)
    
    #predictions = Dense(10, activation='softmax')(x)
    model=Model(inputs=input_img,outputs=x )
    return model

def n_net_test():
    input_img = Input(shape=(128,128,128,1,))
    #first 2 conv layers
    x = Conv3D(24, (3, 3,3), padding='same', activation='relu')(input_img)
    x = Conv3D(24, (3, 3,3), strides=(1,1,1),padding='same', activation='relu')(x)
    x = Conv3D(15 ,(4, 4,4), strides=(4,4,4),padding='same', activation='relu')(x)
    x= Reshape((32,32,32,3,5))(x)
    #predictions = Dense(10, activation='softmax')(x)
    model=Model(inputs=input_img,outputs=x )
    return model




n_net=n_net3










def fpr_net():
    input_img = Input(shape=(32,32,32,1))

    #first 2 conv layers
    x = Conv3D(32, (3, 3,3), padding='same', activation='relu')(input_img)
    x = Conv3D(32, (3, 3,3), strides=(1,1,1),padding='same', activation='relu')(x)
    x=MaxPooling3D(pool_size=(2, 2, 2), strides=(2,2,2))(x)

    
    
    
    x = Conv3D(64, (3, 3,3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3, 3,3), strides=(1,1,1),padding='same', activation='relu')(x)
    x=MaxPooling3D(pool_size=(2, 2, 2), strides=(2,2,2))(x)

    
#
    x = Conv3D(128, (3, 3,3), padding='same', activation='relu')(x)
    x = Conv3D(128, (3, 3,3), strides=(1,1,1),padding='same', activation='relu')(x)
    x=MaxPooling3D(pool_size=(2, 2, 2), strides=(2,2,2))(x)

    
    x=Flatten()(x)
    
    x=Dense(1024, )(x)
#    x=LeakyReLU(alpha=0.3)(x)
    x=Dropout(0.5)(x)
    
    x=Dense(512, )(x)
#    x=LeakyReLU(alpha=0.3)(x)
    x=Dropout(0.5)(x)
    
    x=Dense(1,activation='sigmoid')(x)
   
    model=Model(inputs=input_img,outputs=x )
    return model



def fpr_3d_dcnn():
    input_img = Input(shape=(32,32,32,1))
    
    
    x =BatchNormalization()(input_img)
    
    
 
    x = Conv3D(32, (3, 3,3), padding='same', activation='relu')(x)
    x =BatchNormalization()(x)
    x = Conv3D(32, (3, 3,3), padding='same', activation='relu')(x)
    x =BatchNormalization()(x)
    x=MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(x)

    x=Dropout(0.5)(x)
    
    x = Conv3D(64, (3, 3,3), padding='same', activation='relu')(x)
    x =BatchNormalization()(x)
    x = Conv3D(64, (3, 3,3), padding='same', activation='relu')(x)
    x =BatchNormalization()(x)
    x=MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(x)   
    
    x=Dropout(0.5)(x)
    
    x = Conv3D(128, (3, 3,3), padding='same', activation='relu')(x)
    x =BatchNormalization()(x)
    x = Conv3D(128, (3, 3,3), padding='same', activation='relu')(x)
    x =BatchNormalization()(x)
    x=MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(x)   
    
    x=Dropout(0.5)(x)
    
    


    
#    x=Flatten()(x)
#    
#    x=Dense(1024, )(x)
##    x=LeakyReLU(alpha=0.3)(x)
#    x=Dropout(0.5)(x)
#    
#    x=Dense(512, )(x)
##    x=LeakyReLU(alpha=0.3)(x)pos
#    x=Dropout(0.5)(x)
#    
#    x=Dense(1,activation='sigmoid')(x)
   
    model=Model(inputs=input_img,outputs=x )
    return model




class GetPBB(object):
    def __init__(self, config):
        self.stride = config['stride']
        self.anchors = np.asarray(config['anchors'])

    def __call__(self, output,thresh = -3, ismask=False):
        stride = self.stride
        anchors = self.anchors
        output = np.copy(output)
        offset = (float(stride) - 1) / 2
        output_size = output.shape
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)
        
        output[:, :, :, :, 1] = oz.reshape((-1, 1, 1, 1)) + output[:, :, :, :, 1] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 2] = oh.reshape((1, -1, 1, 1)) + output[:, :, :, :, 2] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 3] = ow.reshape((1, 1, -1, 1)) + output[:, :, :, :, 3] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 4] = np.exp(output[:, :, :, :, 4]) * anchors.reshape((1, 1, 1, -1))
        mask = output[..., 0] > thresh
        xx,yy,zz,aa = np.where(mask)
        
        output = output[xx,yy,zz,aa]
        if ismask:
            return output,[xx,yy,zz,aa]
        else:
            return output

        #output = output[output[:, 0] >= self.conf_th] 
        #bboxes = nms(output, self.nms_th)
def nms(output, nms_th):
    if len(output) == 0:
        return output

    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]
    
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
    
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes

def iou(box0, box1):
    
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union




def acc(pbb, lbb, conf_th, nms_th, detect_th):
    pbb = pbb[pbb[:, 0] >= conf_th] 
    pbb = nms(pbb, nms_th)

    tp = []
    fp = []
    fn = []
    l_flag = np.zeros((len(lbb),), np.int32)
    for p in pbb:
        flag = 0
        bestscore = 0
        for i, l in enumerate(lbb):
            score = iou(p[1:5], l)
            if score>bestscore:
                bestscore = score
                besti = i
        if bestscore > detect_th:
            flag = 1
            if l_flag[besti] == 0:
                l_flag[besti] = 1
                tp.append(np.concatenate([p,[bestscore]],0))
            else:
                fp.append(np.concatenate([p,[bestscore]],0))
        if flag == 0:
            fp.append(np.concatenate([p,[bestscore]],0))
    for i,l in enumerate(lbb):
        if l_flag[i]==0:
            score = []
            for p in pbb:
                score.append(iou(p[1:5],l))
            if len(score)!=0:
                bestscore = np.max(score)
            else:
                bestscore = 0
            if bestscore<detect_th:
                fn.append(np.concatenate([l,[bestscore]],0))

    return tp, fp, fn, len(lbb)    

def topkpbb(pbb,lbb,nms_th,detect_th,topk=30):
    conf_th = 0
    fp = []
    tp = []
    while len(tp)+len(fp)<topk:
        conf_th = conf_th-0.2
        tp, fp, fn, _ = acc(pbb, lbb, conf_th, nms_th, detect_th)
        if conf_th<-3:
            break
    tp = np.array(tp).reshape([len(tp),6])
    fp = np.array(fp).reshape([len(fp),6])
    fn = np.array(fn).reshape([len(fn),5])
    allp  = np.concatenate([tp,fp],0)
    sorting = np.argsort(allp[:,0])[::-1]
    n_tp = len(tp)
    topk = np.min















def hard_mining(neg_output, neg_labels, num_hard):
    prob=neg_output[:,0]
    val, idcs = tf.nn.top_k(prob,k=num_hard)
    box=[]
    for i in range(num_hard):
        out=neg_output[idcs[i]]
        out=tf.reshape(out,[1,5])
        box.append(out)
    out=tf.concat(box,axis=0)
    
    box2=[]
    for i in range(num_hard):
        out2=neg_labels[idcs[i]]
        out2=tf.reshape(out2,[1,5])
        box2.append(out2)
    label=tf.concat(box2,axis=0)
    return out,label






def myloss(y_true, y_pred):
    
    y_true=tf.reshape(y_true,[-1,5])
    y_pred=tf.reshape(y_pred,[-1,5])
    mask_pos=tf.greater(y_true[:,0],0.5)
    mask_neg=tf.less(y_true[:,0],-0.5)
    y_pos_true=tf.boolean_mask(y_true,mask_pos)
    y_neg_true=tf.boolean_mask(y_true,mask_neg)
    
    y_pos_pred=tf.boolean_mask(y_pred,mask_pos)
    y_neg_pred=tf.boolean_mask(y_pred,mask_neg)
    
    y_neg_pred,y_neg_true=hard_mining(y_neg_pred,y_neg_true,config['num_hard'])
    y_neg_true=y_neg_true+1.
    
    y_true=tf.concat([y_pos_true,y_neg_true],axis=0)
    y_pred=tf.concat([y_pos_pred,y_neg_pred],axis=0)
    
    
    
    #add weights to loss respectively to pos and neg
    N_pos=tf.reduce_sum(y_pos_true[:,0])
    N_neg=config['num_hard']
    loss_cls_pos=tf.losses.sigmoid_cross_entropy(y_pos_true[:,0],y_pos_pred[:,0])
    loss_cls_neg=tf.losses.sigmoid_cross_entropy(y_neg_true[:,0],y_neg_pred[:,0])
    loss_cls=(config['beta_pos']*loss_cls_pos*N_pos+config['beta_neg']*loss_cls_neg*N_neg)/(N_pos+N_neg)
    
    
#    y_pred_sigmoid=tf.sigmoid(y_pred[:,0])
##    loss_cls=tf.losses.log_loss(y_true[:,0],y_pred_sigmoid)
#    loss_cls=tf.losses.sigmoid_cross_entropy(y_true[:,0],y_pred[:,0])
    

    def smoothL1(x,y):
        """
        x,y :both are tensors with same shape
        """
        mask=tf.greater(tf.abs(x-y),1)
        l=tf.where(mask,tf.abs(x-y),tf.square(x-y))
        return tf.reduce_sum(l,axis=1)
        
    loss_reg=y_true[:,0]*smoothL1(y_true[:,1:5],y_pred[:,1:5]) 
    loss_reg=tf.reduce_mean(loss_reg)   
    
    #loss_cls=tf.cast(loss_cls,tf.float32)
    loss=tf.add(loss_cls,loss_reg)
    return loss



def loss_cls(y_true, y_pred):
    y_true=tf.reshape(y_true,[-1,5])
    y_pred=tf.reshape(y_pred,[-1,5])
    mask_pos=tf.greater(y_true[:,0],0.5)
    mask_neg=tf.less(y_true[:,0],-0.5)
    y_pos_true=tf.boolean_mask(y_true,mask_pos)
    y_neg_true=tf.boolean_mask(y_true,mask_neg)
    
    y_pos_pred=tf.boolean_mask(y_pred,mask_pos)
    y_neg_pred=tf.boolean_mask(y_pred,mask_neg)
    
    y_neg_pred,y_neg_true=hard_mining(y_neg_pred,y_neg_true,config['num_hard'])
    y_neg_true=y_neg_true+1.
    
    y_true=tf.concat([y_pos_true,y_neg_true],axis=0)
    y_pred=tf.concat([y_pos_pred,y_neg_pred],axis=0)
    
    
    
    #add weights to loss respectively to pos and neg
    N_pos=tf.reduce_sum(y_pos_true[:,0])
    N_neg=config['num_hard']
    loss_cls_pos=tf.losses.sigmoid_cross_entropy(y_pos_true[:,0],y_pos_pred[:,0])
    loss_cls_neg=tf.losses.sigmoid_cross_entropy(y_neg_true[:,0],y_neg_pred[:,0])
    loss_cls=(config['beta_pos']*loss_cls_pos*N_pos+config['beta_neg']*loss_cls_neg*N_neg)/(N_pos+N_neg)
    
    
    
#    y_pred_sigmoid=tf.sigmoid(y_pred[:,0])
##    loss_cls=tf.losses.log_loss(y_true[:,0],y_pred_sigmoid)
#    loss_cls=tf.losses.sigmoid_cross_entropy(y_true[:,0],y_pred[:,0])
    return loss_cls












#no hard_mining

def nohard(y_true, y_pred):
    
    y_true=tf.reshape(y_true,[-1,5])
    y_pred=tf.reshape(y_pred,[-1,5])
    mask_pos=tf.greater(y_true[:,0],0.5)
    mask_neg=tf.less(y_true[:,0],-0.5)
    y_pos_true=tf.boolean_mask(y_true,mask_pos)
    y_neg_true=tf.boolean_mask(y_true,mask_neg)
    
    y_pos_pred=tf.boolean_mask(y_pred,mask_pos)
    y_neg_pred=tf.boolean_mask(y_pred,mask_neg)
    
#    y_neg_pred,y_neg_true=hard_mining(y_neg_pred,y_neg_true,config['num_hard'])
    y_neg_true=y_neg_true+1.
    
    y_true=tf.concat([y_pos_true,y_neg_true],axis=0)
    y_pred=tf.concat([y_pos_pred,y_neg_pred],axis=0)
    
    
    
    #add weights to loss respectively to pos and neg
    N_pos=tf.reduce_sum(y_pos_true[:,0])
    N_neg=config['num_hard']
    loss_cls_pos=tf.losses.sigmoid_cross_entropy(y_pos_true[:,0],y_pos_pred[:,0])
    loss_cls_neg=tf.losses.sigmoid_cross_entropy(y_neg_true[:,0],y_neg_pred[:,0])
    loss_cls=(config['beta_pos']*loss_cls_pos*N_pos+config['beta_neg']*loss_cls_neg*N_neg)/(N_pos+N_neg)
    
    
#    y_pred_sigmoid=tf.sigmoid(y_pred[:,0])
##    loss_cls=tf.losses.log_loss(y_true[:,0],y_pred_sigmoid)
#    loss_cls=tf.losses.sigmoid_cross_entropy(y_true[:,0],y_pred[:,0])
    

    def smoothL1(x,y):
        """
        x,y :both are tensors with same shape
        """
        mask=tf.greater(tf.abs(x-y),1)
        l=tf.where(mask,tf.abs(x-y),tf.square(x-y))
        return tf.reduce_sum(l,axis=1)
        
    loss_reg=y_true[:,0]*smoothL1(y_true[:,1:5],y_pred[:,1:5]) 
    loss_reg=tf.reduce_mean(loss_reg)   
    
    #loss_cls=tf.cast(loss_cls,tf.float32)
    loss=tf.add(loss_cls,loss_reg)
    return loss



def cls_nohard(y_true, y_pred):
    y_true=tf.reshape(y_true,[-1,5])
    y_pred=tf.reshape(y_pred,[-1,5])
    mask_pos=tf.greater(y_true[:,0],0.5)
    mask_neg=tf.less(y_true[:,0],-0.5)
    y_pos_true=tf.boolean_mask(y_true,mask_pos)
    y_neg_true=tf.boolean_mask(y_true,mask_neg)
    
    y_pos_pred=tf.boolean_mask(y_pred,mask_pos)
    y_neg_pred=tf.boolean_mask(y_pred,mask_neg)
    
#    y_neg_pred,y_neg_true=hard_mining(y_neg_pred,y_neg_true,config['num_hard'])
    y_neg_true=y_neg_true+1.
    
    y_true=tf.concat([y_pos_true,y_neg_true],axis=0)
    y_pred=tf.concat([y_pos_pred,y_neg_pred],axis=0)
    
    
    
    #add weights to loss respectively to pos and neg
    N_pos=tf.reduce_sum(y_pos_true[:,0])
    N_neg=config['num_hard']
    loss_cls_pos=tf.losses.sigmoid_cross_entropy(y_pos_true[:,0],y_pos_pred[:,0])
    loss_cls_neg=tf.losses.sigmoid_cross_entropy(y_neg_true[:,0],y_neg_pred[:,0])
    loss_cls=(config['beta_pos']*loss_cls_pos*N_pos+config['beta_neg']*loss_cls_neg*N_neg)/(N_pos+N_neg)
    
    
    
#    y_pred_sigmoid=tf.sigmoid(y_pred[:,0])
##    loss_cls=tf.losses.log_loss(y_true[:,0],y_pred_sigmoid)
#    loss_cls=tf.losses.sigmoid_cross_entropy(y_true[:,0],y_pred[:,0])
    return loss_cls


















def loss_cls_pos(y_true, y_pred):
    y_true=tf.reshape(y_true,[-1,5])
    y_pred=tf.reshape(y_pred,[-1,5])
    mask_pos=tf.greater(y_true[:,0],0.5)
    mask_neg=tf.less(y_true[:,0],-0.5)
    y_pos_true=tf.boolean_mask(y_true,mask_pos)
    y_neg_true=tf.boolean_mask(y_true,mask_neg)
    
    y_pos_pred=tf.boolean_mask(y_pred,mask_pos)
    y_neg_pred=tf.boolean_mask(y_pred,mask_neg)
    
    y_neg_pred,y_neg_true=hard_mining(y_neg_pred,y_neg_true,config['num_hard'])
    y_neg_true=y_neg_true+1.
    
    y_true=tf.concat([y_pos_true,y_neg_true],axis=0)
    y_pred=tf.concat([y_pos_pred,y_neg_pred],axis=0)
    
    
    
    #add weights to loss respectively to pos and neg
    N_pos=tf.reduce_sum(y_pos_true[:,0])
    N_neg=config['num_hard']
    loss_cls_pos=tf.losses.sigmoid_cross_entropy(y_pos_true[:,0],y_pos_pred[:,0])
    loss_cls_neg=tf.losses.sigmoid_cross_entropy(y_neg_true[:,0],y_neg_pred[:,0])
    loss_cls=(config['beta_pos']*loss_cls_pos*N_pos+config['beta_neg']*loss_cls_neg*N_neg)/(N_pos+N_neg)
    
    
    
#    y_pred_sigmoid=tf.sigmoid(y_pred[:,0])
##    loss_cls=tf.losses.log_loss(y_true[:,0],y_pred_sigmoid)
#    loss_cls=tf.losses.sigmoid_cross_entropy(y_true[:,0],y_pred[:,0])
    return (config['beta_pos']*loss_cls_pos*N_pos)/(N_pos+N_neg)

def loss_cls_neg(y_true, y_pred):
    y_true=tf.reshape(y_true,[-1,5])
    y_pred=tf.reshape(y_pred,[-1,5])
    mask_pos=tf.greater(y_true[:,0],0.5)
    mask_neg=tf.less(y_true[:,0],-0.5)
    y_pos_true=tf.boolean_mask(y_true,mask_pos)
    y_neg_true=tf.boolean_mask(y_true,mask_neg)
    
    y_pos_pred=tf.boolean_mask(y_pred,mask_pos)
    y_neg_pred=tf.boolean_mask(y_pred,mask_neg)
    
    y_neg_pred,y_neg_true=hard_mining(y_neg_pred,y_neg_true,config['num_hard'])
    y_neg_true=y_neg_true+1.
    
    y_true=tf.concat([y_pos_true,y_neg_true],axis=0)
    y_pred=tf.concat([y_pos_pred,y_neg_pred],axis=0)
    
    
    
    #add weights to loss respectively to pos and neg
    N_pos=tf.reduce_sum(y_pos_true[:,0])
    N_neg=config['num_hard']
    loss_cls_pos=tf.losses.sigmoid_cross_entropy(y_pos_true[:,0],y_pos_pred[:,0])
    loss_cls_neg=tf.losses.sigmoid_cross_entropy(y_neg_true[:,0],y_neg_pred[:,0])
    loss_cls=(config['beta_pos']*loss_cls_pos*N_pos+config['beta_neg']*loss_cls_neg*N_neg)/(N_pos+N_neg)
    
    
    
#    y_pred_sigmoid=tf.sigmoid(y_pred[:,0])
##    loss_cls=tf.losses.log_loss(y_true[:,0],y_pred_sigmoid)
#    loss_cls=tf.losses.sigmoid_cross_entropy(y_true[:,0],y_pred[:,0])
    return (config['beta_neg']*loss_cls_neg*N_neg)/(N_pos+N_neg)

    

def recall(y_true, y_pred):
    #recall
    
    y_true=tf.reshape(y_true,[-1,5])
    y_pred=tf.reshape(y_pred,[-1,5])
    mask_pos=tf.greater(y_true[:,0],0.5)
    mask_neg=tf.less(y_true[:,0],-0.5)
    y_pos_true=tf.boolean_mask(y_true,mask_pos)
    y_neg_true=tf.boolean_mask(y_true,mask_neg)
    
    y_pos_pred=tf.boolean_mask(y_pred,mask_pos)
    y_neg_pred=tf.boolean_mask(y_pred,mask_neg)
    
    label_pos=tf.sigmoid(y_pos_pred[:,0])
    tp=tf.floor(label_pos+0.5)
    tp=tf.reduce_sum(tp)
    return tp


    
if __name__=='__main__':
    model=n_net()

    model.summary()
    
    plot_model(model, to_file='images/3dmodel.png',show_shapes=True)
    
    
#    6.20 test
#    import data 
#    
#    data_dir='/data/lungCT/luna/temp/luna_npy'
#    dataset=data.DataBowl3Detector(data_dir,data.config)
#    patch,label,coord=dataset.__getitem__(112)
#     
#    y_true=tf.constant(label)
#
#    a=myloss(y_true,y_true)
##     
##  
## #    hard=hard_mining(a,a,4)
##     
#    init=tf.global_variables_initializer()
#    sess=tf.Session()
#    sess.run(init)
#    aa=sess.run(a)
    
    
    
    
    
    
    
#     
# #    hh=sess.run(hard)
#     
#     
#     get=GetPBB(data.config)
#     table=get.__call__(label,0.5)
#     tabel_before_transform=label.reshape([-1,5])
#     
#     index=np.argsort(-table[:,0])
#     tabel_before_transform=tabel_b efore_transform[index]
#     table=table[index]
#     
# #    boxes=nms(table,0.5)
    
    