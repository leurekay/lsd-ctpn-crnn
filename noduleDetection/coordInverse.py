#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:47:38 2018

@author: ly
"""

import sys
sys.path.append('preprocessing')

from  prepare import *


if __name__=='__main__':
    
        
    luna_segment_dir = '/data/lungCT/luna/seg-lungs-LUNA16'
    luna_data_dir = '/data/lungCT/luna/subset1'
    annotations_path='/data/lungCT/luna/annotations.csv'
    
    uid_sp1='1.3.6.1.4.1.14519.5.2.1.6279.6001.282512043257574309474415322775'
    
    file_list=os.listdir(luna_data_dir)
    patient_list=[x.split('.mhd')[0] for x in file_list if x.endswith('.mhd')]
#    for uid in patient_list:
#        path=os.path.join(luna_data_dir,uid+'.mhd')
#        ooxx=load_itk_image(path)
#        print (uid,ooxx[3])
#        
#        with open(path) as f:
#            contents = f.readlines() 
    


    
    path='/data/lungCT/luna/subset8/1.3.6.1.4.1.14519.5.2.1.6279.6001.225515255547637437801620523312.mhd'
    
    path_sp1=os.path.join(luna_data_dir,uid_sp1+'.mhd')
    
    ooxx=load_itk_image(path)

    with open(path) as f:
        contents = f.readlines()    
    
