# get only inferences for people detected
# https://github.com/xuannianz/EfficientDet/blob/master/inference.py


# the bounding box coordinates from the video directly, won't match those from the extracted images, because 
# the size of the images was reduced during extraction
# ground truth will be with respect to the original video
# so the extracted images should not be used during validation


import os
os.chdir("C:/Users/DAR9KOR/Desktop/data/sample_datasets/distance_detection/code/efficientDet_xuan/EfficientDet-master/")


import cv2
import json
import numpy as np
import pandas as pd
import os
import time
#import glob
import argparse
import imutils

from model import efficientdet
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes



input_video = "/content/data/video/BIAL/bial.mp4"
out_path = "/content/output/"
out_vid = out_path + "out.avi"

##################################################################
# coco classes
classes = {value['id'] - 1: value['name'] for value in json.load(open('coco_90.json', 'r')).values()}
num_classes = 90

score_threshold = 0.5

np.random.seed(42)
colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
#colors = np.random.randint(0, 255, size=(num_classes, 3), dtype="uint8")


################################################################
# model
phi = 1
#phi = 2
#phi = 3
#phi = 4

model_path = 'efficientdet-d' + str(phi) + '.h5'  #'d1.h5'
weighted_bifpn = True
image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
image_size = image_sizes[phi]


_, model = efficientdet(phi=phi,
                        weighted_bifpn=weighted_bifpn,
                        num_classes=num_classes,
                        score_threshold=score_threshold)
model.load_weights(model_path, by_name=True)


########################################################################

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(input_video)
writer = None
(w, h) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1


####################################################################

start = time.time()
results = []
#out_json = "/content/det_xuan/" + 'detections_eff1.json'
#outj = open(out_json, 'a')
#outj.write("[")
count = 0

# loop over frames from the video file stream
while True:
	# read the next frame from the file
    (grabbed, image) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
    if not grabbed:
        break
    
    t1 = time.time()
    src_image = image.copy()
    # BGR -> RGB
    image = image[:, :, ::-1]

    # if the frame dimensions are empty, grab them
    if w is None or h is None:
        (h, w) = image.shape[:2]
    #h, w = image.shape[:2]

    image, scale = preprocess_image(image, image_size=image_size)
    # run network
    #start = time.time()
    boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
    boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
    #print(time.time() - start)
    boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

    # select indices which have a score above the threshold
    indices = np.where(scores[:] > score_threshold)[0]
    
    ## select only people indices
    indices1 = np.where(labels[indices] == 0)[0]
    
    # select those detections
    #boxes = boxes[indices1]
    #labels = labels[indices1]
    #scores = scores[indices1]
    
    
    # create output file
    #count = 0
    if count > 4500:  #2
        break
    #out_file_name = out_path + str(count) + '.txt'
    #print('')
    #print(count)
    #outf = open(out_file_name, 'a')

  
    #draw_boxes(src_image, boxes, scores, labels, colors, classes)
    # http://cocodataset.org/#format-results
    #[{
    #  "image_id": int,
    #  "category_id": int,
    #  "bbox": [x,y,width,height],
    #  "score": float,
    #}]

    # https://github.com/cocodataset/cocoapi/blob/master/results/instances_val2014_fakebbox100_results.json
    #[{"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236},
    #{"image_id":73,"category_id":11,"bbox":[61,22.75,504,609.67],"score":0.318},
    #{"image_id":1292,"category_id":47,"bbox":[66.74,228.43,32.05,32.89],"score":0.097}]

    if len(indices1) > 0:
      for i in indices1:
        x1 = boxes[i][0]
        y1 = boxes[i][1]
        x2 = boxes[i][2]  # x+w
        y2 = boxes[i][3] # y+h
        width = x2 - x1
        height = y2 - y1

        #results.append({"image_id":count, "category_id":labels[i],'bbox', 'score':'%.3f'%scores[i], 'ymin':boxes[i][1], 'xmin':boxes[i][0], 'ymax':boxes[i][3], 'xmax':boxes[i][2]})
        results.append([count, x1, y1, x2, y2, '%.3f'%scores[i]])  #x, y, x+w, y+h
        #s_js = '{"image_id":' + str(count) + ',"category_id":' + str(labels[i]) + ',"bbox":[' + str(int(x1)) + ',' + str(int(y1)) + ',' + str(int(width)) + ',' + str(int(height)) + '],"score":' + str('%.3f'%scores[i]) + '},'
        #outj.write(s_js)
        #ss = str(labels[i]) + ' ' + str('%.3f'%scores[i]) + ' ' + str(int(x1)) + ' ' + str(int(y1)) + ' ' + str(int(width)) + ' ' + str(int(height)) + "\n"
        #outf.write(ss)

    #outf.close()

    t2 = time.time()
    #time = end1 - start1
    print('FPS', 1/(t2-t1), '  time_per_frame', t2-t1, '  frame_no', count)
    #print(count)
    count = count + 1
    #return results, boxes[indices1], scores[indices1], labels[indices1], src_image, end1, model_time, len(indices1)

    '''
    ## THE FOLLOWING IS FOR DRAWING BOXES
    # select those detections
    boxes = boxes[indices1]
    labels = labels[indices1]
    scores = scores[indices1]
    
    if len(indices1) > 0:
        #print(len(indices1))
        for b, l, s in zip(boxes, labels, scores):
            class_id = int(l)
            class_name = classes[class_id]
            xmin, ymin, xmax, ymax = list(map(int, b))
            score = '{:.4f}'.format(s)

        #    fis = str(score) + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax)
        #    print(fis)
            
            color = colors[class_id]
            #color = [int(c) for c in colors[class_id]]
            label = '-'.join([class_name, score])
            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), (102, 220, 225), 2)
            ###cv2.rectangle(src_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
            ###cv2.putText(src_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

	  # check if the video writer is None
    if writer is None:
		# initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(out_vid, fourcc, 1, #30,
                                 (src_image.shape[1], src_image.shape[0]), True)

	  # write the output frame to disk
    writer.write(src_image)
    #out_img = out_path + str(count) + '_img.png'
    #cv2.imwrite(out_img, src_image)
'''

end = time.time()
#outj.write("]")
#outj.close()



#####################################################################

'''
# release the file pointers , if writing video out
print("[INFO] cleaning up...")
writer.release()
vs.release()
'''

results = pd.DataFrame(results, columns = ['frameNumber', 'xmin', 'ymin', 'xmax', 'ymax', 'score'])
out_file = out_path + "detections_eff" + str(phi) + "_conf05.csv"
results.to_csv(out_file)

print("code took {:.6f} seconds".format(end - start))
