
import cv2
import numpy as np


input_video = 'TownCentreXVID.avi'

# Create a VideoCapture object
cap = cv2.VideoCapture(input_video) 

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Unable to read camera feed")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


# Define the codec and create VideoWriter object.The output is stored in 'output.avi' file.
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

frames=0
while(cap.isOpened() and frames<50):
    ret, frame = cap.read() 
    frames+=1

    if ret == True: 
        # do steps ...
        # frame = cv2.putText(frame, str(frames), (200,200), cv2.FONT_HERSHEY_SIMPLEX , 2, (255,255,255), 2, cv2.LINE_AA) 
        out.write(frame)

        # Display the resulting frame    
        cv2.imshow('frame',frame)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break  

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows() 
