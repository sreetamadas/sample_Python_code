# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:41:53 2020

@author: NGG6KOR
"""
import numpy as np
import cv2
#from numba import jit


def get_pred(image, W, H, input_w, input_h, net, ln, input_conf, input_thres):

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (input_w, input_h), swapRB=False, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    # print('layer_op')
    # print(layerOutputs)
    

    boxes = []
    confidences = []
    classIDs = []
    results = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            # print ('classid: '+str(classID))
            confidence = scores[classID]
    
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > input_conf and classID==0: #args["confidence"]:
               # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
    
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
    
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    #idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
    #    args["threshold"])
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, input_conf,
        input_thres)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # if classIDs[i]==0: #"person":
                # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # print(classIDs[i], confidences[i],y,x,y+h,x+w)

            results.append({'label':classIDs[i], 'score':'%.3f'%confidences[i], 'ymin':y, 'xmin':x, 'ymax':y+h, 'xmax':x+w})
    
    return results
