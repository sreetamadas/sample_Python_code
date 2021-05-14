# roi
# https://medium.com/beyondlabsey/creating-a-simple-region-of-interest-roi-inside-a-video-streaming-using-opencv-in-python-30fd671350e0

# https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
# yolov3 has been trained on MSCOCO dataset
# The code below is using pre-trained YOLO v3 model to do predictions
# no further training has been done


# import the necessary packages
import numpy as np
#import argparse
import time
import cv2
import os



###########################################################################################################
def get_pred(photo_filename, input_w, input_h, net, ln, input_conf, input_thres, upper_left, bottom_right):
    
    #load and prepare image
    image = cv2.imread(photo_filename)
    (H, W) = image.shape[:2]
    print('H,W: ' + str(H) + ',' + str(W))

    # take ROI
    #upper_left = (50, 50)  ### (x1,y1)
    #bottom_right = (300, 300)  ### (x2,y2)
    upper_left = upper_left
    bottom_right = bottom_right
    r = cv2.rectangle(image, upper_left, bottom_right, (100, 50, 200), 5)  ###
    rect_img = image[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]  ### [y1:y2, x1:x2]
    H1 = bottom_right[1] - upper_left[1]  #250  ###
    W1 = bottom_right[0] - upper_left[0] #250  ###


    blob = cv2.dnn.blobFromImage(rect_img, 1 / 255.0, (input_w, input_h), swapRB=False, crop=False)  #####
    net.setInput(blob)
    #layerOutputs = net.forward(ln)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    #print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    model_time = end - start
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
                box = detection[0:4] * np.array([W1, H1, W1, H1])  #[W, H, W, H]
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
    end1 = time.time()
    return results, idxs, classIDs, confidences, boxes, image, end1, model_time, rect_img 



###########################################################################################################

### check on multiple images

def give_frames(image_paths, out_path, name):
    file_no = os.listdir(image_paths)
    
    input_conf = 0.4  # class_thres
    input_thres = 0.4  # nms_thres
    input_w, input_h = 416, 416
    
    # specify ROI
    ul_x = 100
    ul_y = 300 #100
    br_x = 450 #850
    br_y = 700 #450
    upper_left = (ul_x, ul_y) #(50, 50)  ###
    bottom_right = (br_x, br_y) #(300, 300)  ###
    

    df = []
    
    for j in range(0, len(file_no)): 
        photo_filename = image_paths + file_no[j]

        #print(photo_filename)
        start1 = time.time()
        results, idxs, classIDs, confidences, boxes, image, end1, model_time, rect_img = get_pred(photo_filename, input_w, input_h, net, ln, input_conf, input_thres, upper_left, bottom_right)
        code_time = end1 - start1
        #print("code took {:.6f} seconds".format(end1 - start1))
        df.append([file_no[j], model_time, code_time, len(idxs)])

        #draw_boxes(src_img, box, Score, Label, colors, classes)
        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        #cv2.imshow('image', src_image)

        # define the labels
        LABELS = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
                  "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                  "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                  "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                  "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                  "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
                  
        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                if classIDs[i]==0: #"person":
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    #print(classIDs[i], confidences[i],y,x,y+h,x+w)
                    
                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(image, (x + ul_x, y + ul_y), (x + w + ul_x, y + h + ul_y), color, 2)
                    cv2.rectangle(image, upper_left, bottom_right, (100, 50, 200), 5)
                    #text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    #    0.5, color, 2)
        
        # show the output image
        #cv2.imshow("Image", image)
        out_img = out_path + name + file_no[j]
        cv2.imwrite(out_img, image)
    return df


#################################################################################################################

# inputs
lpath = "/content/det_yolo/coco_yolo/"
weightsPath = lpath+"yolov3.weights"
configPath = lpath+"yolov3.cfg"

# loading model
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# I/O
img_path =  "/content/img/"
out_path = "/content/det_yolo/img_yolo/"



#####################################################################

# code RUN

name = 'yolo_'
df = give_frames(img_path, out_path, name)

import pandas as pd
df = pd.DataFrame(df, columns=['image','model_time', 'code_time', 'num_ppl'])
df.to_excel("/content/det_yolo/img_yolo/yolo_out.xls") 


###########################################################################

# Displaying a single image with detctions in ROI
from IPython import display
display.display(display.Image(os.path.join(out_path, 'out_yolo_536.jpg')))




