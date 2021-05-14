# get only inferences for people detected
# https://github.com/xuannianz/EfficientDet/blob/master/inference.py


import os
os.chdir("EfficientDet-master")

'''
# workaround for issue "no module named 'utils.compute_overlap' "  when running "from model import efficientdet"

!python setup.py build_ext --inplace
'''

import cv2
import json
import numpy as np
#import os
import time
import glob

from model import efficientdet
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes


##################################################################################################

def get_pred(photo_filename, model, score_threshold, img_path):
    image = cv2.imread(photo_filename)
    src_image = image.copy()
    # BGR -> RGB
    image = image[:, :, ::-1]
    h, w = image.shape[:2]

    image, scale = preprocess_image(image, image_size=image_size)

    # run network
    start = time.time()
    boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
    boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
    #print(time.time() - start)
    end = time.time()
    #print("model took {:.6f} seconds".format(end - start))
    model_time = end - start
    boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

    # select indices which have a score above the threshold
    indices = np.where(scores[:] > score_threshold)[0]

    # select those detections
    #boxes = boxes[indices]
    #labels = labels[indices]

    ## select only people indices
    indices1 = np.where(labels[indices] == 0)[0]

    # select those detections
    #boxes1 = boxes[indices1]
    #labels1 = labels[indices1]
    #print(indices1)
    #print("*")
    #print(list(indices1))
    print(len(indices1))
    print("*")
    #print(labels[indices1])
    #print(labels[1])
    #print(scores[1])
    #print(boxes[1][0])

    ### get the labels, scores & boxes for people
    # in Darknet YOLOv3, the order of box corners was: x, y, w, h; printed as y,x,y+h,x+w
    # in efficientDet, the order of box corners is: x, y, x+w, y+h ; print as y, x, y+h, x+w
    results = []
    
    if len(indices1) > 0:
        for i in indices1:
        #print(i) 
        #Label = labels[i]
        #Class_score = scores[i]
        #x = boxes[i][0]
        #y = boxes[i][1]
        #x2 = boxes[i][2]  # x+w
        #y2 = boxes[i][3] # y+h
        #(x, y) = (boxes[i][0], boxes[i][1])
        #(x2, y2) = (boxes[i][2], boxes[i][3])
        #print(Label)
        #print(Class_score)
        #print(x)
        #print(y)
        #print(x2)
        #print(y2)

        results.append({'label':labels[i], 'score':'%.3f'%scores[i], 'ymin':boxes[i][1], 'xmin':boxes[i][0], 'ymax':boxes[i][3], 'xmax':boxes[i][2]})
        print(labels[i], '%.3f'%scores[i], boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2])
        #results.append({'label':Label, 'score':'%.3f'%Class_score, 'ymin':y, 'xmin':x, 'ymax':y2, 'xmax':x2})
        #print(Label, '%.3f'%Class_score, y, x, y2, x2)

    
    #return results, boxes[indices1], scores[indices1], labels[indices1], src_image
    end1 = time.time()
    return results, boxes[indices1], scores[indices1], labels[indices1], src_image, end1, model_time, len(indices1)

#draw_boxes(src_image, boxes, scores, labels, colors, classes)

#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#cv2.imshow('image', src_image)
#out_img = img_path + 'out_img.png'
#cv2.imwrite(out_img, src_image)
#cv2.waitKey(0)


###################################################################################################

### check on multiple images

def give_frames(image_paths, out_path, name):
    file_no = os.listdir(image_paths)

    df = []
    
    for j in range(0, len(file_no)): 
        photo_filename = image_paths + file_no[j]
        print(photo_filename)

        start1 = time.time()
        results, Box, Score, Label, src_img, end1, model_time, ppl_detected = get_pred(photo_filename, model, score_threshold, img_path)
        code_time = end1 - start1
        #print("code took {:.6f} seconds".format(end1 - start1))
        df.append([file_no[j], model_time, code_time, ppl_detected])

        #draw_boxes(src_img, box, Score, Label, colors, classes)
        for b, l, s in zip(Box, Label, Score):
          class_id = int(l)
          class_name = classes[class_id]
          xmin, ymin, xmax, ymax = list(map(int, b))
          score = '{:.4f}'.format(s)
          color = colors[class_id]
          label = '-'.join([class_name, score])
          ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
          #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
          cv2.rectangle(src_img, (xmin, ymin), (xmax, ymax), (102, 220, 225), 2)
          #cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
          #cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)



        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        #cv2.imshow('image', src_image)
        out_img = out_path + name + file_no[j]
        cv2.imwrite(out_img, src_img)
    return df


########################################################################################################

## using function to load model

def get_model(phi, weighted_bifpn, num_classes, score_threshold):
    model_path = 'efficientdet-d' + str(phi) + '.h5'
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    
    _, model = efficientdet(phi=phi,
                        weighted_bifpn=weighted_bifpn,
                        num_classes=num_classes,
                        score_threshold=score_threshold)
    return model, model_path


#####################################################################################################

# coco classes
classes = {value['id'] - 1: value['name'] for value in json.load(open('coco_90.json', 'r')).values()}
num_classes = 90
score_threshold = 0.4  #0.3   ## this is class_thres
colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]


###############################################################################################

# model
phi = 1
weighted_bifpn = True
model, model_path = get_model(phi, weighted_bifpn, num_classes, score_threshold)
model.load_weights(model_path, by_name=True)


#start1 = time.time()
#results, box, Score, Label, src_img = get_pred(photo_filename, model, score_threshold, img_path)
#end1 = time.time()
#print("[INFO] code took {:.6f} seconds".format(end1 - start1))

img_path = "/content/data/"
out_path = "/content/out/"


name = 'ef1_'
df = []
df = give_frames(img_path, out_path, name)

import pandas as pd
df = pd.DataFrame(df, columns=['image','model_time', 'code_time','num_ppl'])
out_excel = out_path + 'eff1_out.csv'
df.to_csv(out_excel) 





