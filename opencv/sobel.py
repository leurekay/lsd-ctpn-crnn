#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:05:48 2018

@author: dirac
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import solve




img = cv2.imread('data/bank4.jpg')
img_=np.copy(img)

grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.GaussianBlur(grey,(5,5),4)


def hough_line(img,threh_hough):
    
    height,width=img.shape[0],img.shape[1]
    edges = cv2.Canny(img,90,150)
    lines = cv2.HoughLines(edges,1,np.pi/180,threh_hough)
    lines=lines.reshape([-1,2])
    n,_=lines.shape
    
    remove_list=[]
    
    box=[]
    for i in range(n):
        if i in remove_list:
            continue
        for j in range(i+1,n):
            if j in remove_list:
                continue
            rho1,theta1=lines[i]
            rho2,theta2=lines[j]
            if abs(rho1-rho2)<50 and abs(theta1-theta2)<0.1:
                remove_list.append(j)
                continue
            
            a1,b1 = np.cos(theta1),np.sin(theta1)
            a2,b2 = np.cos(theta2),np.sin(theta2)
    
            a = np.array([[a1,b1],[a2,b2]])
            b = np.array([rho1,rho2])
            
            try:
                x = solve(a, b)
            except :
                pass
#                print ('================\n',a,b,'\n==============')
            else:
                if (x[0]>0 and x[0]<width) and (x[1]>0 and x[1]<height):
                    box.append(x)
    box=np.array(box)
#    print (box)
    
    index_line=[x for x in range(n) if x not in remove_list]
    return lines[index_line],box


def clockwise_sort(X):
    """
    X: (N,2) numpyArray
    """
    X=np.array(X)
    center=np.mean(X,axis=0)
    center_X=X-center
    center_X_norm=center_X/np.linalg.norm(center_X,axis=1).reshape([-1,1])
#    xAxis=np.array([1,0])
    def get_complex(a):
        return complex(a[0],a[1])
    center_X_complex=np.apply_along_axis(get_complex,1,center_X_norm)
    angle=np.angle(center_X_complex)*180/np.pi
    order=np.argsort(angle)
    
    X=X[order]
    return center,X




edges = cv2.Canny(img,80,150)
fig=plt.figure(figsize=[20,10])
plt.subplot(321),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(322),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

#plt.show()

def iterator(img,initial_threh):
    threh=initial_threh
    while threh<220:
        
        lines,crosspoints = hough_line(img,threh)
        threh+=1
        if lines.shape[0]==4 and  crosspoints.shape[0]==4:
            break
    return lines,crosspoints
            
        

#lines,crosspoints = hough_line(img,140)
lines,crosspoints = iterator(img,90)
cc,crosspoints=clockwise_sort(crosspoints)

pts2 = np.float32([[0,0],[1600,0],[1600,900],[0,900]]) 

M = cv2.getPerspectiveTransform(crosspoints,pts2)
dst = cv2.warpPerspective(img_,M,(1600,900))

for line in lines:
    rho,theta=line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img_,(x1,y1),(x2,y2),(0,0,255),6)
for i,point in enumerate(crosspoints):
    cv2.circle(img_,tuple(point),25,(145,154,55),10*i+10)
cv2.circle(img_,tuple(cc),25,(14,124,155),-1)

cv2.imwrite('houghlines3.jpg',img_)

cv2.imwrite('ooxx.jpg',dst)


plt.subplot(323),plt.imshow(img_)
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.subplot(324),plt.imshow(dst)
plt.title('Calibration'), plt.xticks([]), plt.yticks([])

dst_grey=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
img=cv2.GaussianBlur(dst_grey,(5,5),3)
ret,th = cv2.threshold(dst_grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.subplot(325),plt.imshow(th,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])