import os
import shutil
import numpy as np
from config_training import config


from scipy.io import loadmat
import numpy as np
import h5py
import pandas
import scipy
from scipy.ndimage.interpolation import zoom
from skimage import measure
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
import pandas
from multiprocessing import Pool
from functools import partial
import sys
sys.path.append('../preprocessing')
from step1 import step1_python
import warnings
import pandas as pd

def resample(imgs, spacing, new_spacing,order=2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')
def worldToVoxelCoord(worldCoord, origin, spacing):
     
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any( transformM!=np.array([1,0,0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing,isflip

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>1.5*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask


def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg


def savenpy(id,annos,filelist,data_path,prep_folder):        
    resolution = np.array([1,1,1])
    name = filelist[id]
    label = annos[annos[:,0]==name]
    label = label[:,[3,1,2,4]].astype('float')
    
    im, m1, m2, spacing = step1_python(os.path.join(data_path,name))
    Mask = m1+m2
    
    newshape = np.round(np.array(Mask.shape)*spacing/resolution)
    xx,yy,zz= np.where(Mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
    extendbox = extendbox.astype('int')



    convex_mask = m1
    dm1 = process_mask(m1)
    dm2 = process_mask(m2)
    dilatedMask = dm1+dm2
    Mask = m1+m2
    extramask = dilatedMask - Mask
    bone_thresh = 210
    pad_value = 170
    im[np.isnan(im)]=-2000
    sliceim = lumTrans(im)
    sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
    bones = sliceim*extramask>bone_thresh
    sliceim[bones] = pad_value
    sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
    sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                extendbox[1,0]:extendbox[1,1],
                extendbox[2,0]:extendbox[2,1]]
    sliceim = sliceim2[np.newaxis,...]
    np.save(os.path.join(prep_folder,name+'_clean.npy'),sliceim)

    
    if len(label)==0:
        label2 = np.array([[0,0,0,0]])
    elif len(label[0])==0:
        label2 = np.array([[0,0,0,0]])
    elif label[0][0]==0:
        label2 = np.array([[0,0,0,0]])
    else:
        haslabel = 1
        label2 = np.copy(label).T
        label2[:3] = label2[:3][[0,2,1]]
        label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        label2[3] = label2[3]*spacing[1]/resolution[1]
        label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
        label2 = label2[:4].T
    np.save(os.path.join(prep_folder,name+'_label.npy'),label2)

    print(name)

def full_prep(step1=True,step2 = True):
    warnings.filterwarnings("ignore")

    #preprocess_result_path = './prep_result'
    prep_folder = config['preprocess_result_path']
    data_path = config['stage1_data_path']
    finished_flag = '.flag_prepkaggle'
    
    if not os.path.exists(finished_flag):
        alllabelfiles = config['stage1_annos_path']
        tmp = []
        for f in alllabelfiles:
            content = np.array(pandas.read_csv(f))
            content = content[content[:,0]!=np.nan]
            tmp.append(content[:,:5])
        alllabel = np.concatenate(tmp,0)
        filelist = os.listdir(config['stage1_data_path'])

        if not os.path.exists(prep_folder):
            os.mkdir(prep_folder)
        #eng.addpath('preprocessing/',nargout=0)

        print('starting preprocessing')
        pool = Pool()
        filelist = [f for f in os.listdir(data_path)]
        partial_savenpy = partial(savenpy,annos= alllabel,filelist=filelist,data_path=data_path,prep_folder=prep_folder )

        N = len(filelist)
            #savenpy(1)
        _=pool.map(partial_savenpy,range(N))
        pool.close()
        pool.join()
        print('end preprocessing')
    f= open(finished_flag,"w+")        

def savenpy_luna(id,annos,luna_segment,luna_data,savepath):
    islabel = True
    isClean = True
    isInfo=True
    resolution = np.array([1,1,1])
#     resolution = np.array([2,2,2])
#    name = filelist[id]
    name=id
    
    Mask,origin,spacing,isflip = load_itk_image(os.path.join(luna_segment,name+'.mhd'))
    if isflip:
        Mask = Mask[:,::-1,::-1]
    newshape = np.round(np.array(Mask.shape)*spacing/resolution).astype('int')
    m1 = Mask==3
    m2 = Mask==4
    Mask = m1+m2
    print
    xx,yy,zz= np.where(Mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T

    this_annos=annos[annos.seriesuid==name]
#    this_annos = np.copy(annos[annos[:,0]==name]) 
    this_annos.iloc[:,0]=666       
    this_annos=np.array(this_annos)
    
    if isClean:
        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2
        Mask = m1+m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        sliceim,origin,spacing,isflip = load_itk_image(os.path.join(luna_data,name+'.mhd'))
        if isflip:
            sliceim = sliceim[:,::-1,::-1]
            print('flip!')
        sliceim = lumTrans(sliceim)
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = (sliceim*extramask)>bone_thresh
        sliceim[bones] = pad_value
        
        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        sliceim = sliceim2[np.newaxis,...]
        np.save(os.path.join(savepath,name+'_clean.npy'),sliceim)


    if islabel:

#        this_annos = np.copy(annos[annos[:,0]==int(name)])
        label = []
        if len(this_annos)>0:
            
            for c in this_annos:
                pos = worldToVoxelCoord(c[1:4][::-1],origin=origin,spacing=spacing)
                if isflip:
                    pos[1:] = Mask.shape[1:3]-pos[1:]
                label.append(np.concatenate([pos,[c[4]/spacing[1]]]))
            
        label = np.array(label)
        if len(label)==0:
            label2 = np.array([[0,0,0,0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
            label2[3] = label2[3]*spacing[1]/resolution[1]
            label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
            label2 = label2[:4].T
        np.save(os.path.join(savepath,name+'_label.npy'),label2)
    
    if isInfo:
        if isflip:
            flipinfo=np.array([[1,1,1]])
        else:
            flipinfo=np.array([[0,0,0]])
        origin_entendbox=np.concatenate([origin.reshape([1,-1]),spacing.reshape([1,-1]),np.array([Mask.shape]),np.array([sliceim2.shape]),extendbox.T,flipinfo])
        np.save(os.path.join(savepath,name+'_info.npy'),origin_entendbox)
    print(name)
# ----------------------the content saved in *_info.npy------------------------
# =============================================================================
#                       origin
# =============================================================================
#                       spacing
# =============================================================================
#                       mask_shape
# =============================================================================
#                       clean_image_shape
# =============================================================================
#                       extend[0]
#                       extend[1]
# =============================================================================
#                       flip
# =============================================================================







def preprocess_luna():
    luna_segment = config['luna_segment']
    savepath = config['preprocess_result_path']
    luna_data = config['luna_data']
    luna_label = config['luna_label']
    finished_flag = '.flag_preprocessluna'
    print('starting preprocessing luna')
    if not os.path.exists(finished_flag):
        filelist = [f.split('.mhd')[0] for f in os.listdir(luna_data) if f.endswith('.mhd') ]
        annos = np.array(pandas.read_csv(luna_label))

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        
        pool = Pool()
        partial_savenpy_luna = partial(savenpy_luna,annos=annos,filelist=filelist,
                                       luna_segment=luna_segment,luna_data=luna_data,savepath=savepath)

        N = len(filelist)
        #savenpy(1)
        _=pool.map(partial_savenpy_luna,range(N))
        pool.close()
        pool.join()
    print('end preprocessing luna')
    f= open(finished_flag,"w+")
    
def prepare_luna():
    print('start changing luna name')
    luna_raw = config['luna_raw']
    luna_abbr = config['luna_abbr']
    luna_data = config['luna_data']
    luna_segment = config['luna_segment']
    finished_flag = '.flag_prepareluna'
    
    if not os.path.exists(finished_flag):

        subsetdirs = [os.path.join(luna_raw,f) for f in os.listdir(luna_raw) if f.startswith('subset') and os.path.isdir(os.path.join(luna_raw,f))]
        if not os.path.exists(luna_data):
            os.makedirs(luna_data)

#         allnames = []
#         for d in subsetdirs:
#             files = os.listdir(d)
#             names = [f[:-4] for f in files if f.endswith('mhd')]
#             allnames = allnames + names
#         allnames = np.array(allnames)
#         allnames = np.sort(allnames)

#         ids = np.arange(len(allnames)).astype('str')
#         ids = np.array(['0'*(3-len(n))+n for n in ids])
#         pds = pandas.DataFrame(np.array([ids,allnames]).T)
#         namelist = list(allnames)
        
        abbrevs = np.array(pandas.read_csv(config['luna_abbr'],header=None))
        namelist = list(abbrevs[:,1])
        ids = abbrevs[:,0]
        
        for d in subsetdirs:
            files = os.listdir(d)
            files.sort()
            for f in files:
                name = f[:-4]
                id = ids[namelist.index(name)]
                filename = '0'*(3-len(str(id)))+str(id)
                shutil.move(os.path.join(d,f),os.path.join(luna_data,filename+f[-4:]))
                print(os.path.join(luna_data,str(id)+f[-4:]))

        files = [f for f in os.listdir(luna_data) if f.endswith('mhd')]
        for file in files:
            with open(os.path.join(luna_data,file),'r') as f:
                content = f.readlines()
                id = file.split('.mhd')[0]
                filename = '0'*(3-len(str(id)))+str(id)
                content[-1]='ElementDataFile = '+filename+'.raw\n'
                print(content[-1])
            with open(os.path.join(luna_data,file),'w') as f:
                f.writelines(content)

                
        seglist = os.listdir(luna_segment)
        for f in seglist:
            if f.endswith('.mhd'):

                name = f[:-4]
                lastfix = f[-4:]
            else:
                name = f[:-5]
                lastfix = f[-5:]
            if name in namelist:
                id = ids[namelist.index(name)]
                filename = '0'*(3-len(str(id)))+str(id)

                shutil.move(os.path.join(luna_segment,f),os.path.join(luna_segment,filename+lastfix))
                print(os.path.join(luna_segment,filename+lastfix))


        files = [f for f in os.listdir(luna_segment) if f.endswith('mhd')]
        for file in files:
            with open(os.path.join(luna_segment,file),'r') as f:
                content = f.readlines()
                id =  file.split('.mhd')[0]
                filename = '0'*(3-len(str(id)))+str(id)
                content[-1]='ElementDataFile = '+filename+'.zraw\n'
                print(content[-1])
            with open(os.path.join(luna_segment,file),'w') as f:
                f.writelines(content)
    print('end changing luna name')
    f= open(finished_flag,"w+")


def batch_process_luna(luna_data_dir,savepath,luna_segment_dir,annotations_path):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    df_anno=pd.read_csv(annotations_path)
    
    patients_list=os.listdir(luna_data_dir)
    patients_list=filter(lambda x:x.split('.')[-1]=='mhd' ,patients_list)
    patients_list=[x.split('.mhd')[0] for x in patients_list]
    
    already_list=os.listdir(savepath)
    
    already_list=filter(lambda x:x.split('_')[-1]=='label.npy' ,already_list) 
    already_list=[x.split('_label.npy')[0] for x in already_list]
    
    wait_list=[x for x in patients_list if x not in already_list]
    print ('this dir has %d patient CT'%len(patients_list))
    print ('we have already processed %d CT'%len(already_list))


    pool = Pool()
    partial_savenpy_luna = partial(savenpy_luna,annos=df_anno,
                                   luna_segment=luna_segment_dir,luna_data=luna_data_dir,savepath=savepath)

    _=pool.map(partial_savenpy_luna,wait_list)
    pool.close()
    pool.join()
    print('complete %s'%savepath)        
    
    
    
   
def simple_label_transform(zyx,info):
    """
    zyx: world coordinate sorted by z,y,z order
    info: saved by *_info.npy
    
    return : pixel coorinate ,compare with *_clean.npy image's origin relatively
             z,y,z order
             if zyx is annotation,then return == *_label.npy
    """
    origin,spacing,mask_shape,_,_,_,flipinfo=info
    extendbox=info[4:6].T
    if flipinfo[0]:
        isflip=True
    else:
        isflip=False
    
    resolution=np.array([1,1,1])
    label = []
    pos = worldToVoxelCoord(zyx,origin=origin,spacing=spacing)
    if isflip:
        pos[1:] = mask_shape[1:3]-pos[1:]
    label.append(np.concatenate([pos,np.array([9999])/spacing[1]]))
        
    label = np.array(label)
    if len(label)==0:
        label2 = np.array([[0,0,0,0]])
    else:
        label2 = np.copy(label).T
        label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        label2[3] = label2[3]*spacing[1]/resolution[1]
        label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
        label2 = label2[:4].T
    return label2
    
    
    
if __name__=='__main__':
    
    subsets=[0,1,2,3,4,5,6,7,8,9]
#    subsets=[0]
    subsets=map(lambda x : 'subset'+str(x),subsets)
    
    for subset in subsets:
        
        luna_segment_dir = '/data/lungCT/luna/seg-lungs-LUNA16'
        savepath = '/data/lungCT/luna/temp/luna_npy/'+subset
        luna_data_dir = '/data/lungCT/luna/'+subset
        annotations_path='/data/lungCT/luna/annotations.csv'
        
        #generate *_clean.npy and *_label.npy
        batch_process_luna(luna_data_dir,savepath,luna_segment_dir,annotations_path)

    
    
#    path='/data/lungCT/luna/subset8/1.3.6.1.4.1.14519.5.2.1.6279.6001.225515255547637437801620523312.mhd'
#    
#    ooxx=load_itk_image(path)
#    
#    
#    with open(path) as f:
#        contents = f.readlines()   
    
    
    
    
    """    
    

    n=31
    datadir='/data/lungCT/luna/subset1/'
    listfiles=os.listdir(datadir)
    listfiles=[x for x in listfiles if x.endswith('.mhd')]
    listuids=[x.split('.mhd')[0] for x in listfiles]
    uid=listuids[n]
    annotations_path='/data/lungCT/luna/annotations.csv'
    path_label='/data/lungCT/luna/temp/luna_npy/'+uid+'_label.npy'
    mask_path=os.path.join('/data/lungCT/luna/seg-lungs-LUNA16',uid+'.mhd')
    
    path='/data/lungCT/luna/subset0/'+uid+'.mhd'
    npy_img_path=os.path.join('/data/lungCT/luna/temp/luna_npy/',uid+'_clean.npy')
    
    
#    img,lb_origin,lb_spacing,_=load_itk_image(path)
#    
##    image_npy=np.load(npy_img_path)
##    label_new=np.load(path_label)
    df=pd.read_csv(annotations_path)
#    label_initial=df[df.seriesuid==uid]
#    label_initial=np.array(list(label_initial.iloc[0,1:]))
#    mask=load_itk_image(mask_path)
#    itkimage = sitk.ReadImage(path)
    
    import matplotlib.pyplot as plt
#    plt.imshow(image_npy[0,:,:,160], cmap=plt.cm.gray)
#    plt.imshow(img[:,:,255], cmap=plt.cm.gray)
    
    
    uid='1.3.6.1.4.1.14519.5.2.1.6279.6001.282512043257574309474415322775'
    savenpy_luna(uid,df,'/data/lungCT/luna/seg-lungs-LUNA16',datadir,'.')
    path_label='/data/lungCT/luna/temp/luna_npy/val/'+uid+'_label.npy'
    bl=np.load(path_label)
    b_world_label=df[df['seriesuid']==uid]
    b_world_label=b_world_label.iloc[:,1:]
    b_world_label=list(b_world_label.values)[0]
    
    b_info=np.load(uid+'_info.npy')
    b_label=np.load(uid+'_label.npy')
    b_img=np.load(uid+'_clean.npy')
    
    b_origin,b_spacing=b_info[0],b_info[1]
    
    b_world_coord=b_world_label[:3][::-1]
    kk=simple_label_transform(b_world_coord,b_info)
    
    """