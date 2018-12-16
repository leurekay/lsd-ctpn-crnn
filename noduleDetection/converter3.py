#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 17:10:34 2018

@author: ly
"""

import vtk
from vtk.util import numpy_support
import cv2  
import os  
import  dicom 
import numpy  
import SimpleITK  as sitk




base_dir="/data/WanliyunLungCT/dcmdata/"
#base_dir='/data/lungCT/dsb2017/sample_images/'
patients_list=os.listdir(base_dir)
patients_list.sort(key=lambda x : x)


patient=patients_list[0]
PathDicom = os.path.join(base_dir,patient)



reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(PathDicom)
reader.SetFileNames(dicom_names)
image = reader.Execute()
image_array = sitk.GetArrayFromImage(image) # z, y, x
origin = image.GetOrigin() # x, y, z
spacing = image.GetSpacing() # x, y, z









#PathDicom = "./MyHead/"
reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName(PathDicom)
reader.Update()
    
    
    
# Load dimensions using `GetDataExtent`
_extent = reader.GetDataExtent()
ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

# Load spacing values
ConstPixelSpacing = reader.GetPixelSpacing()

# Get the 'vtkImageData' object from the reader
imageData = reader.GetOutput()
# Get the 'vtkPointData' object from the 'vtkImageData' object
pointData = imageData.GetPointData()
# Ensure that only one array exists within the 'vtkPointData' object
assert (pointData.GetNumberOfArrays()==1)
# Get the `vtkArray` (or whatever derived type) which is needed for the `numpy_support.vtk_to_numpy` function
arrayData = pointData.GetArray(0)

# Convert the `vtkArray` to a NumPy array
ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
# Reshape the NumPy array to 3D using 'ConstPixelDims' as a 'shape'
ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')