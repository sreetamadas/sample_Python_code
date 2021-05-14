# utilities to 
# check video properties - FPS, dimensions, no. of frames etc
# and alter video FPS, length, etc


import cv2
cv2.__version__  

'''
import numpy as np
import os
import time
#import glob
import argparse
import imutils
'''

input_video = "C:/Users/data/multi_camera_tracking/SALSA/salsa_ps_cam2.avi"


#################################################################
## check video fps
# https://www.learnopencv.com/how-to-find-frame-rate-or-frames-per-second-fps-in-opencv-python-cpp/

video = cv2.VideoCapture(input_video)
fps = video.get(cv2.CAP_PROP_FPS)
print(fps)

'''
if __name__ == '__main__' :

    video = cv2.VideoCapture(input_video);
    
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')  

    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print(fps) #"Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        print(fps) #"Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)
        
    video.release();
'''

################################################################
## no of frames / length of video

# video frames
vs = cv2.VideoCapture(input_video)
writer = None
(W, H) = (None, None)

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


#################################################################
### check video dimensions

vs = cv2.VideoCapture(input_video)
writer = None
(W, H) = (None, None)

while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	print(H)
	print(W)
	break


#################################################################
### convert video FPS

'''
# method 1: gave error "MoviePy couldn't find the codec associated with the filename. Provide the 'codec' parameter in write_videofile."
!pip install moviepy
from moviepy.editor import *

# https://stackoverflow.com/questions/44179498/change-videos-frame-rate-fps-in-python
clip = VideoFileClip(input_video)

video_output_path = "/content/outvid_fps15.avi"
clip.write_videofile(video_output_path, fps=15)

'''

# Method 2
# https://stackoverflow.com/questions/45462731/using-ffmpeg-to-change-framerate/45465730#45465730
!ffmpeg -y -r 15 -i TownCentreXVID.avi TownCentre_fps15.avi



#################################################################
## snip video

video = cv2.VideoCapture(input_video)
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )
video.release()

# generate snipped video
!ffmpeg -ss 00:03:30 -i /content/TownCentreXVID.avi -to 00:01:00 -c copy /content/sample_1min.avi












