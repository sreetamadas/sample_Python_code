# https://github.com/rafaelpadilla/Object-Detection-Metrics#create-the-ground-truth-files

# connect drive
from google.colab import drive
drive.mount('/content/gdrive')


# set path
import os
os.chdir("/content/")
os.mkdir("metric")
os.chdir("metric")
os.getcwd()


# get tools: metric calculation code module
!cp -r /content/gdrive/My\ Drive/detect/Object-Detection-Metrics-master/  /content/metric/


# get data
!unzip -q "/content/gdrive/My Drive/detect/validation_video/detections_yolo.zip" -d "/content/gdrive/My Drive/detect/validation_video/"
!unzip -q "/content/gdrive/My Drive/detect/validation_video/groundtruth_txt.zip" -d "/content/gdrive/My Drive/detect/validation_video/"

## verify sam eno. of files in both
!ls -l /content/gdrive/My\ Drive/detect/validation_video/detections_yolo/ | wc -l
!ls -l /content/gdrive/My\ Drive/detect/validation_video/groundtruth_txt/ | wc -l



# run evaluation
# https://github.com/rafaelpadilla/Object-Detection-Metrics#asterisk
# need to specify:
# 1. groundtruth folder
# 2. dtection folder
# 3. IOU threshold
# 4. GT format (xyrb; default is xywh)
# 5. detection format (,,)
# 6. output folder

# relative coordinates of bounding box can be used, but requires specifying image size
# this will work if all images are of same size

!mkdir /content/out_yolo/
!python /content/metric/Object-Detection-Metrics-master/pascalvoc.py --gtfolder /content/groundtruths/ --detfolder /content/detections_yolo/ 
    --threshold 0.40 --savepath /content/out_yolo/

#!python /content/metric/Object-Detection-Metrics-master/pascalvoc.py --gtfolder /content/ground_truth/ --detfolder /content/detections_yolo/ 
#    --threshold 0.50 -gtformat xyrb -detformat xyrb --savepath /content/out_yolo/


# save results
!cp -r /content/out_yolo/ /content/gdrive/My\ Drive/detect/validation_video/


