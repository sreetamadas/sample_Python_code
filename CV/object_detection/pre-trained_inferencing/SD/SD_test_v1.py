
import os
import multiprocessing
import datetime
import cv2
from imutils.video import FPS
#import algo
import warnings
warnings.filterwarnings("ignore")

os.chdir('C:/Users/code/New_models/')
import algo

root = "C:/Users/data/video/" #"D:/Projects/people_distance/data"
files = ['bus_lr.mp4','TownCentreXVID.avi','023.mov','100.mov','168.mov', '256.mov', 'sample3.avi']
f_index = 1  
# input_video = "http://182.75.71.150:86/nphMotionJpeg?Resolution=640x480&Quality=Standard"
input_video = os.path.join(root,files[f_index])



cap = cv2.VideoCapture(input_video)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Unable to read camera feed")

print('FPS info:')
print(round(cap.get(cv2.CAP_PROP_FPS)))

frames=-1
fps = FPS().start()
logs_queue = multiprocessing.Queue(-1)
sd_obj = algo.BoschDL_SD(f_index,logs_queue)

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.

# op_video = str(f_index)+'.avi'
op_video = files[f_index].split(".")[0]+'_op_'+str(datetime.datetime.now()).split(" ")[0]+'.avi'
out = cv2.VideoWriter(op_video,cv2.VideoWriter_fourcc('M','J','P','G'), round(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(3)),int(cap.get(4))),True)

while(cap.isOpened() and frames<50): #3000 ; 5400
    ret, frame = cap.read() 
    frames+=1
    if ret == True: 
        
        if frames%1==0:
            print('frame count : '+ str(frames))
            img = frame.copy()
            result = sd_obj.gen(frame,str(datetime.datetime.now()).split(".")[0] )
        # Display the resulting frame 
        out.write(result['frame'])  # out.write(frame)
        cv2.imshow('frame',result['frame'])
        print(result['message'])
        
        fps.update()

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break  

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows() 
