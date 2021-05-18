# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 17:01:55 2020

@author: DAR9KOR
"""
import cv2
import numpy as np

def add_mask(frame,res1):
    '''logic for blur box: don't use full bounding box
    In x-direction, it is half of box-width about centre -> box moves; take full
    In y-direction, it is 1/3 box height from top
    '''
    size = 99
    #print(res1.shape)
    for i in range(res1.shape[0]):
        x1 = int(res1[i,4]) #int((3*res1[i,4] + res1[i,5])/4)  #int(res1[i,4]) ; instead of xmin, use (3xmin + xmax)/4
        x2 = int(res1[i,5]) #int((res1[i,4] + 3*res1[i,5])/4)  #int(res1[i,5])
        y1 = int(res1[i,2])
        y2 = int((2*res1[i,2] + res1[i,3])/3)  #int(res1[i,3])
        #print(y1,x1,y2,x2)  # this is correct
        
        #frame[x1:x2, y1:y2] = cv2.GaussianBlur(frame[x1:x2, y1:y2],(size,size),cv2.BORDER_DEFAULT)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2],(size,size),cv2.BORDER_DEFAULT)
        # the above step takes y-coord first, then x; otherwise the blurred box is 90deg rotated
    return frame

