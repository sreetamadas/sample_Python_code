# split a video to create 2 videos with overlap in frames

import os
import cv2
from imutils.video import FPS

'''input & output dir'''
os.chdir("C:/Users/Desktop/data/video/")


'''operate directly on video'''
input_video = "sample_1min.avi"
cap = cv2.VideoCapture(input_video)

frames=-1
fps = FPS().start()


width = int(cap.get(3))
height = int(cap.get(4))

factor = 2
out_h = height//2
out_w = width//2
out_w = 0.75 * out_w # for overlapping video
#out_w = 0.5 * out_w  # for non-overlapping video

#out_l = cv2.VideoWriter("left.avi",cv2.VideoWriter_fourcc('M','J','P','G'), round(cap.get(cv2.CAP_PROP_FPS)), 
#                      (int(out_w),int(out_h)),True)
#out_r = cv2.VideoWriter("right.avi",cv2.VideoWriter_fourcc('M','J','P','G'), round(cap.get(cv2.CAP_PROP_FPS)), 
#                      (int(out_w),int(out_h)),True)
out_dir_l = "left_images_overlap/"
out_dir_r = "right_images_overlap/"


while(cap.isOpened() and frames<255): #500
    ret, frame = cap.read() 
    frames+=1
    
    if ret == True:
        image = cv2.resize(frame, (1920 // factor, 1080 // factor))
        
        # left crop
        start_y = 0
        end_y = image.shape[0]  # height,width = image.shape()
        start_x = 0
        #end_x = 720   # int(out_w)
        end_x = int(out_w)
        l_crop = image[start_y:end_y, start_x:end_x]  # startY and endY coordinates, followed by the startX and endX coordinates
        #out_l.write(l_crop)

        
        # right crop
        #start_x = 240  #
        start_x = int(width//2 - out_w)
        end_x = image.shape[1]
        r_crop = image[start_y:end_y, start_x:end_x]
        #out_r.write(r_crop)
        
        fps.update()
        
        out_l = 'l_' + str(frames) + '.jpg' #file_no[j]
        cv2.imwrite(str(out_dir_l + out_l), l_crop)
        out_r = 'r_' + str(frames) + '.jpg' #file_no[j]
        cv2.imwrite(str(out_dir_r + out_r), r_crop)
        
    else:
        break

# When everything done, release the video capture and video write objects
cap.release()
out_l.release()
out_r.release()

    
                 


#'''
# operating on frames extracted before

in_dir = "images1/"
out_dir_l = "left_images_overlap/"
out_dir_r = "right_images_overlap/"

os.mkdir(out_dir_l)
os.mkdir(out_dir_r)


# create crops of the frames

# dimension: 960 x 540
file_no = os.listdir(in_dir)

for j in range(0,170): #len(file_no)):
    photo_filename = in_dir + str(j) + '.jpg' #file_no[j]
    image = cv2.imread(photo_filename)
    
    # left crop
    start_y = 0
    end_y = image.shape[0]  # height,width = image.shape()
    start_x = 0
    end_x = 720
    l_crop = image[start_y:end_y, start_x:end_x]  # startY and endY coordinates, followed by the startX and endX coordinates
    
    # right crop
    start_x = 240
    end_x = image.shape[1]
    r_crop = image[start_y:end_y, start_x:end_x]
    
    out_l = 'l_' + str(j) + '.jpg' #file_no[j]
    cv2.imwrite(str(out_dir_l + out_l), l_crop)
    out_r = 'r_' + str(j) + '.jpg' #file_no[j]
    cv2.imwrite(str(out_dir_r + out_r), r_crop)
#'''    



