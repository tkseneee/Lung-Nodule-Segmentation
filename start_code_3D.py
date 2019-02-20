# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:23:54 2019

@author: senthilku
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

img=cv2.imread('lung_img.jpg',0)
cv2.imshow("Lung Image",img)
cv2.waitKey()

th,ostu_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#ada_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv2.THRESH_BINARY,11,2)

cv2.imshow("Ostu Segmentation",ostu_img)
cv2.waitKey()
#cv2.imshow("Adaptive Segmentation",ada_img)

from skimage.segmentation import clear_border
img2=clear_border(ostu_img)
cv2.imshow("clear_border",img2)
cv2.waitKey()

#img22=clear_border(ada_img)
#cv2.imshow("clear_border_Adaptive",img22)

img3=255-ostu_img
cv2.imshow("Inverted_image",img3)
cv2.waitKey()

img4=clear_border(img3)
cv2.imshow("clear_border1",img4)
cv2.waitKey()

se_fill=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
img4_fill = cv2.morphologyEx(img4, cv2.MORPH_CLOSE, se_fill)
cv2.imshow("filled_image",img4_fill)
cv2.waitKey()

se_open=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
img4_open = cv2.morphologyEx(img4_fill, cv2.MORPH_OPEN, se_open)
cv2.imshow("Open_image",img4_open)
cv2.waitKey()

paren=img & img4_open
cv2.imshow("Parenchyma",paren)
cv2.waitKey()

thos,nod_th = cv2.threshold(paren,100,255,cv2.THRESH_BINARY)
cv2.imshow("Nodules",nod_th)
cv2.waitKey()

#from skimage import morphology
#nodules = morphology.remove_small_objects(nod_th, min_size=10, connectivity=2)
#cv2.imshow("Final_Nodules",nodules)

se_open=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
nodules = cv2.morphologyEx(nod_th, cv2.MORPH_OPEN, se_open)
cv2.imshow("Final_Nodules",nodules)
cv2.waitKey()

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(nodules)
#nodules_fin = morphology.remove_small_objects(labels, min_size=100, connectivity=2)
cv2.imshow("Final_Nodules",nodules)
cv2.waitKey()

sizes = stats[1:, -1];
min_size = 15
nodules1 = np.zeros((labels.shape),dtype='uint8')

#for every component in the image, you keep it only if it's above min_size
for i in range(0, nlabels-1):
    if sizes[i] >= min_size:
        nodules1[labels == i + 1] = 255 
cv2.imshow("Final_Nodules1",nodules1)
cv2.waitKey()
nlabels1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(nodules1)
#pos1=np.where(stats[3]>10)
#nod1=np.zeros([512,512],dtype='uint8')
all_nod=[]
feat=np.zeros([nlabels1-1,9],dtype='float')
for i in range(1,nlabels1):
    nod1=np.zeros([512,512],dtype='uint8')
    pos=np.where(labels1==i)
    nod1[pos]=nodules1[pos]
    cv2.imshow("nod1",nod1)
    cv2.waitKey()
    all_nod.append(nod1)
    
    x,y,w,h = cv2.boundingRect(nod1)
    feat[i-1,0] = float(w)/h #Aspect Ratio
    im2, cc, hierarchy = cv2.findContours(nod1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(cc[0])
    feat[i-1,1]=area
    rect_area = w*h
    feat[i-1,2]= float(area)/rect_area #Extent
    hull = cv2.convexHull(cc[0])
    hull_area = cv2.contourArea(hull)
    feat[i-1,3]=hull_area #hull area
    feat[i-1,4] = float(area)/hull_area #solidity
    feat[i-1,5] = np.sqrt(4*area/np.pi) #equi_diameter
    [(x,y),(MA,ma),angle] = cv2.fitEllipse(cc[0])
    feat[i-1,6] =angle
    
feat[:,7]=centroids1[1:,0]
feat[:,8]=centroids1[1:,1]

    
 



#params = cv2.SimpleBlobDetector_Params()
## Filter by Area.
#params.filterByArea = True
#params.minArea = 10
#
## Filter by Circularity
#params.filterByCircularity = True
#params.minCircularity = 0.1
#
#
## Filter by Convexity
#params.filterByConvexity = True
#params.minConvexity = 0.87
#
## Filter by Inertia
#params.filterByInertia = True
#params.minInertiaRatio = 0.01
#  
#detector = cv2.SimpleBlobDetector_create(params)

## Otsu's thresholding
#ret2,nod_ost = cv2.threshold(paren,0,255,cv2.THRESH_BINARY)
#cv2.imshow("Nodules1",nod_ost)

#ada_img = cv2.adaptiveThreshold(paren,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv2.THRESH_BINARY,7,4)
#cv2.imshow("Nodules",ada_img)
##Imfill
#h, w = img4.shape[:2]
#mask = np.zeros((h+2, w+2), np.uint8)
#cv2.floodFill(img4, mask, (0,0), 255);
#im_floodfill_inv = cv2.bitwise_not(img4)
#im_fill = img4 | im_floodfill_inv
#cv2.imshow("clear_border1",im_fill)


#thr = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,
#                            cv2.THRESH_BINARY,11,2)
#
#img_th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv2.THRESH_BINARY,11,2)
#
#cv2.imshow("Adaptive Image",img_th)