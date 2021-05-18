
import numpy as np
import pandas as pd
import cv2 
# import os
from scipy.optimize import least_squares
from people_detection_v2 import *

class Calibrate:
    
    bounding_box = None
    currentframe = None
    last_static_frame = None
    instance = None
    
    def __init__(self,instance):
        self.bounding_box = []
        self.currentframe = 0
        self.instance = instance
        
    def fun(self,x, ph, pt, h):
        # return x[0] + x[1] * np.exp(x[2] * t + p) - y
        # p = (pt-0.5)*x[1]
        # return abs(2*(x[0]*np.cos(p)**2*(ph-pt)*np.tan(x[1]/2))/(np.sin(x[2]+p)*np.cos(x[2]+p)) - h) #+ 0.01*abs(x[0]-20) + 10*abs(x[1]-np.pi/4) + 0.1*abs(x[2]-np.pi/3)   
        p = (pt-0.5)*x[1]
        return 0.5*h*np.sin(x[2]-p)*np.cos(x[2]-p)/(x[0]*np.tan(0.5*x[1])*np.cos(p)**2) - (ph-pt)

    def draw_shapes(self,frame,frame_width,frame_height,res):
    
        # frame = cv2.putText(frame, str(frames), (200,200), cv2.FONT_HERSHEY_SIMPLEX , 2, (255,255,255), 2, cv2.LINE_AA)
        for i in range(len(res)): 
            clr = [255,255,255]
            ymin = int(res[i]['ymin'])
            xmin = int(res[i]['xmin'])
            ymax = int(res[i]['ymax'])
            xmax = int(res[i]['xmax'])
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), clr, 2)
        return frame
    
    def calibrate(self,net,ln,input_conf,input_thres,dim,ds,ha,frame):
        print("calibrate.")
        
        img = np.float32(frame)
        frame_height, frame_width, _ = img.shape
        
        if len(self.bounding_box) > 100:
            # generate parameter file
            df = pd.DataFrame(self.bounding_box)
            x0 = np.array([15,np.pi/4,np.pi/4])
            # x0 = np.array([40,0.17,0.5])
            h = frame_height
            # df = pd.read_csv(bb_op_path)
            df['ymin']=df['ymin'].where(df['ymin'].between(0,h))
            df['ymax']=df['ymax'].where(df['ymax'].between(0,h))
            df = df.dropna().reset_index()
            ha = 5.6
            h_test = ha*np.ones(len(df))
            ph_test = 1-df['ymin'].values/h
            pt_test = 1-df['ymax'].values/h
            res_lsq = least_squares(self.fun, x0, loss='soft_l1', f_scale=0.1, bounds=([0,np.pi/8,np.pi/8], [50,np.pi/3,np.pi/3]), args=(ph_test, pt_test, h_test))
            res_lsq.x
            # save to parameter file 
            file1 = open("cam_parameters_"+str(self.instance)+".txt","w") 
            L = str((res_lsq.x[0],res_lsq.x[1],res_lsq.x[2]))
              
            file1.writelines(L) 
            file1.close() 
            #to change file access modes  
            return  {
                'frame' : self.last_static_frame,
                'isCalibrationRequired' : False
            }
            # return false : don't continue
        
        img = frame.copy()
        if (self.currentframe%10==0):
        # print(frames)
        # img = input_prep(img,dim)
        # res = get_pred(img,frame_width,frame_height,dim[1],dim[0])
            res = get_pred(img, frame_width, frame_height, dim[1],dim[0], net, ln, input_conf, input_thres)
            self.bounding_box += res
            # print('len(bounding_box)')
            # print(len(bounding_box))
        
            disp_info = str(min(len(self.bounding_box),100))+'% completed'
        # frame = cv2.putText(frame, 'Camera is being calibrated, may take around 15 mins', (20,50), cv2.FONT_HERSHEY_SIMPLEX , 2, (255,255,255), 2, cv2.LINE_AA)
            frame = self.draw_shapes(frame,frame_width,frame_height,res) 
            frame = cv2.putText(frame, 'Camera is being calibrated, may take ~15 mins', (20,40), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0,0,0), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, 'Please make sure people are properly visible in the video feed and', (20,80), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,0,0), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, 'bounding boxes are getting generated enclosing their whole body,', (20,110), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,0,0), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, 'Else camera setup needs to be changed to ensure', (20,140), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,0,0), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, 'proper lighting and better angle to detect people', (20,170), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,0,0), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, 'Recommended camera height : 12 ft.', (20,210), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,0,0), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, 'Recommended camera angle : 30 degrees', (20,240), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,0,0), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, disp_info, (20,280), cv2.FONT_HERSHEY_SIMPLEX , .6, (0,0,0), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, 'Camera is being calibrated, may take ~15 mins', (20,40), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (255,255,255), 1, cv2.LINE_AA)
            frame = cv2.putText(frame, 'Please make sure people are properly visible in the video feed and', (20,80), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255,255,255), 1, cv2.LINE_AA)
            frame = cv2.putText(frame, 'bounding boxes are getting generated enclosing their whole body,', (20,110), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255,255,255), 1, cv2.LINE_AA)
            frame = cv2.putText(frame, 'Else camera setup needs to be changed to ensure', (20,140), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255,255,255), 1, cv2.LINE_AA)
            frame = cv2.putText(frame, 'proper lighting and better angle to detect people', (20,170), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255,255,255), 1, cv2.LINE_AA)
            frame = cv2.putText(frame, 'Recommended camera height : 12 ft.', (20,210), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255,255,255), 1, cv2.LINE_AA)
            frame = cv2.putText(frame, 'Recommended camera angle : 30 degrees', (20,240), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255,255,255), 1, cv2.LINE_AA)
            frame = cv2.putText(frame, disp_info, (20,280), cv2.FONT_HERSHEY_SIMPLEX , .6, (50,250,50), 1, cv2.LINE_AA)
            self.last_static_frame = frame
            # if video is still left continue creating images 
            # frame,msg,currentframe,res,ellipses,red_bounds = process_frame(frame,net,ln,input_conf,input_thres,dim,x,ds,ha,currentframe,frame_height,frame_width)
            # out.write(frame)
    
            # Display the resulting frame    
            # end = time.time()
        self.currentframe+=1
        
        #return true: continue
        return {
            'frame' : self.last_static_frame,
            'isCalibrationRequired' : True
        }
