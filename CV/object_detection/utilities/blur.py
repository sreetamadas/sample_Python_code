# blur section of an image

import cv2
import numpy as np
from imutils.video import FPS

def add_mask(frame,res1):
    '''logic for blur box: don't use full bounding box
    In x-direction, it is half of box-width about centre -> box moves; take full
    In y-direction, it is 1/3 box height from top
    '''
    size = 99
    #print(res1.shape)
    for i in range(res1.shape[0]):
        x1 = int(res1[i,2]) #int((3*res1[i,4] + res1[i,5])/4)  #int(res1[i,4]) ; instead of xmin, use (3xmin + xmax)/4
        x2 = int(res1[i,3]) #int((res1[i,4] + 3*res1[i,5])/4)  #int(res1[i,5])
        y1 = int(res1[i,0])
        y2 = int((2*res1[i,0] + res1[i,1])/3)  #int(res1[i,3])
        #print(y1,x1,y2,x2)  # this is correct
        
        #frame[x1:x2, y1:y2] = cv2.GaussianBlur(frame[x1:x2, y1:y2],(size,size),cv2.BORDER_DEFAULT)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2],(size,size),cv2.BORDER_DEFAULT)
        # the above step takes y-coord first, then x; otherwise the blurred box is 90deg rotated
    return frame

  
  
cap = cv2.VideoCapture(input_video)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Unable to read camera feed")

print('FPS info:')
print(round(cap.get(cv2.CAP_PROP_FPS)))


while(cap.isOpened() and frames<50): #3000 ; 5400
    ret, frame = cap.read() 
    # coordinates to blur
    res = [ymin, xmin, ymax, xmax]  
    add_mask(frame,res)
