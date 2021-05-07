#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import json
import numpy as np
from matplotlib import pyplot as plt


# In[50]:


#data path
data_path="C://Users//Desktop//Data//"

with open("test.txt",'r') as file:
    lines=file.readlines()


# In[51]:


#to read the files
files=[]
for line in lines:
    files.append(data_path+line.rstrip("\n"))
    


# In[55]:


#to read the gt files
for f in files:
    image = cv2.imread(f)
    (H, W) = image.shape[:2]
    #print(H,W)
    #print(image)
    print("file",f)
    # read GT
    gt_file_name = f.split('.jpg')[0] + '.txt'
    print(gt_file_name)
    gt_file=[]
    with open (gt_file_name,'r') as file_object:
        for line in file_object:
            gt_file.append(line.rstrip("\n"))
    file_object.close()
    for eachline in gt_file:
        det = eachline.rstrip("\n").split(" ")
        #print(det)
        # scale the bounding box coordinates back relative to the
        # size of the image, keeping in mind that YOLO actually
        # returns the center (x, y)-coordinates of the bounding
        # box followed by the boxes' width and height
        #box = det[1:4] * np.array([W, H, W, H])
        #(centerX, centerY, width, height) = box.astype("int")
        centerX = float(det[1]) * W
        centerY = float(det[2]) * H
        width = float(det[3]) * W
        height = float(det[4]) * H
        classId = det[0]

        # use the center (x, y)-coordinates to derive the top and
        # and left corner of the bounding box
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))
        print("Ground Truth Image Details",x,y,(x+int(width)),(y+int(height)))

        # draw a bounding box rectangle and label on the image
        #color = [int(c) for c in COLORS[classId]]
        #cv2.rectangle(image, (x, y), (x + int(width), y + int(height)), color, 2)
        cv2.rectangle(image, (x, y), (x + int(width), y + int(height)), [225,225,0], 2)
        #text = "{}: {:.4f}".format(LABELS[classId])
        #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
        cv2.putText(image, classId, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, [0,225,0], 2)
        #plt.imshow(image)



    # show the output image
    #cv2.imshow("Image", image)  #cv2.imshow() is disabled in Colab, because it causes Jupyter sessions to crash;
    #plt.imshow(image)
    plt.figure(figsize=(4,4))  #12,16
    plt.imshow(image)
    plt.show()
    #out_img = out_path + name + file_no[j]
    #cv2.imwrite(out_img, image)
    


# In[ ]:


# together with Ground Truth and Predicted Results


# In[68]:


with open("yolo4_pred_bbox.json") as file:
    data=json.load(file)


# In[69]:


out_path="C://Users//Desktop//Localization//output//yolo_v4//"


# In[70]:


for i in range(0,len(data)):
    name = data[i]['filename']
    print(name)
    print(data[i]['objects'])
    
    file_name=name.split("/")[-1]
    filename=data_path+file_name

    print(filename)
    image = cv2.imread(filename)
    (H, W) = image.shape[:2]
    # read GT
    gt_file_name = filename.split('.jpg')[0] + '.txt'
    print(gt_file_name)
    gt_file=[]
    with open (gt_file_name,'r') as file_object:
        for line in file_object:
            gt_file.append(line.rstrip("\n"))
    file_object.close()
    print(gt_file)
    for eachline in gt_file:
        det = eachline.rstrip("\n").split(" ")
        #print(det)
        # scale the bounding box coordinates back relative to the
        # size of the image, keeping in mind that YOLO actually
        # returns the center (x, y)-coordinates of the bounding
        # box followed by the boxes' width and height
        #box = det[1:4] * np.array([W, H, W, H])
        #(centerX, centerY, width, height) = box.astype("int")
        centerX = float(det[1]) * W
        centerY = float(det[2]) * H
        width = float(det[3]) * W
        height = float(det[4]) * H
        classId = det[0]

        # use the center (x, y)-coordinates to derive the top and
        # and left corner of the bounding box
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))
        print("Ground Truth Image Details",x,y,(x+int(width)),(y+int(height)))

        # draw a bounding box rectangle and label on the image
        #color = [int(c) for c in COLORS[classId]]
        #cv2.rectangle(image, (x, y), (x + int(width), y + int(height)), color, 2)
        cv2.rectangle(image, (x, y), (x + int(width), y + int(height)), [225,255,255], 2)
        #text = "{}: {:.4f}".format(LABELS[classId])
        #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
        cv2.putText(image, classId, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, [0,225,0], 2)
        #plt.imshow(image)
    
    #for each filename & GT - correct here
    for j in range(0,len(data[i]['objects'])):
        classId = data[i]['objects'][j]['class_id']
        conf = data[i]['objects'][j]['confidence']
        centerX = data[i]['objects'][j]['relative_coordinates']['center_x'] * W
        centerY = data[i]['objects'][j]['relative_coordinates']['center_y'] * H
        width = data[i]['objects'][j]['relative_coordinates']['width'] * W
        height = data[i]['objects'][j]['relative_coordinates']['height'] * H
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))
        print("YOLO Predicted Co-ordinates:",x,y,(x+int(width)),(y+int(height)))

        # draw a bounding box rectangle and label on the image
        #color = [int(c) for c in COLORS[classId]]
        #cv2.rectangle(image, (x, y), (x + int(width), y + int(height)), color, 2)
        cv2.rectangle(image, (x, y), (x + int(width), y + int(height)), [225,225,0], 2)
        #text = "{}: {:.4f}".format(LABELS[classId])
        #text = "{}: {:.4f}".format(classId, conf)
        
        #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
        cv2.putText(image, str(classId), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, [255,0,0], 2)
        #plt.imshow(image)
    
    out_img = out_path + file_name
    cv2.imwrite(out_img, image)
    
    plt.figure(figsize=(4,4))  #12,16
    plt.imshow(image)
    plt.show()
    


# In[ ]:




