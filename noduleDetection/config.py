#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 16:02:40 2018

@author: ly
"""

config = {}
config['anchors'] = [ 6.0, 12.0, 24.]
config['chanel'] = 1
config['crop_size'] = [128, 128, 128]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 2
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 4. #mm
config['sizelim2'] = 15
config['sizelim3'] = 28
config['aug_scale'] = False
config['r_rand_crop'] = 0.3
config['pad_value'] = 170
config['augtype'] = {'flip':False,'swap':False,'scale':False,'rotate':False}
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38','990fbe3f0a1b53878669967b9afd1441','adc3bbc63d40f8761c59be10f1e504c3']
config['train_over_total']=0.8
config['optimizer']='adam'
config['leaky_alpha']=0.3
config['beta_pos']=0.6  #coefficient act on loss_cls_pos
config['beta_neg']=0.6


#when test
config['pos_th']=0.02
config['nms_th']=0.6



config['data_prep_dir']='/data/lungCT/luna/temp/luna_npy'
config['valsplit_dir']='splitdata'
config['train_val_test_ratio']=[0.8,0.18,0.02]
config['data_split_shuffle']=True
config['model_dir']='/data/lungCT/luna/temp/savemodel/model4/'
config['ctinfo_path']='preprocessing/ct_info.csv'
config['pred_save_dir']='/data/lungCT/luna/temp/submit'



#False positive reduction
config['model_dir_fpr']='/data/lungCT/luna/temp/savemodel_fpr/model4/'
config['candidate_path']='/data/lungCT/luna/candidates.csv'    




def lr_decay(epoch):
    lr=0.003
    if epoch>2:
        lr=0.001
    if epoch>5:
        lr=0.0003
    if epoch>10:
        lr=0.0001
    if epoch>20:
        lr=0.00003
    if epoch>40:
        lr=0.00001
    if epoch>60:
        lr=0.000003
    return lr



