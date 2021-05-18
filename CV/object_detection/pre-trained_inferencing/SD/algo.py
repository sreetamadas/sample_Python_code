
import os
import ast ##
import cv2
import numpy as np
import logging
import logging.handlers
from imutils.video import FPS
import warnings
warnings.filterwarnings("ignore")
from people_distance_v3 import *
from people_detection_v2 import *
import calibrate_v1 as calib
import sys
from utils import *

class BoschDL_SD:

    logger = None

    
    # SD 
    netsd = None
    x = None
    dim = None
    ha = None
    ds = None
    ln = None
    input_conf = None
    input_thres = None
    
    frame_width = None
    frame_height = None
    
    CALIBRATE = None
    Calibrate_obj = None
    
    
    
    
    def __init__(self,instance,logs_queue):
        
        self.instance = instance
        h = logging.handlers.QueueHandler(logs_queue)
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(logging.DEBUG)
        self.logger = logging.getLogger('p.boschDL.'+str(instance))
        
        self.Calibrate_obj = calib.Calibrate(instance)
        
        # SD loading model
        weightsPath = 'yolov3.weights'
        configPath = 'yolov3.cfg'
        
        self.netsd = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        
        # set up to access CUDA
        self.netsd.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.netsd.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA) 
        
        self.dim = (416,416)
        self.ha = 5.5
        self.ds = 6
        
        # determine only the *output* layer names that we need from YOLO
        self.ln = self.netsd.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.netsd.getUnconnectedOutLayers()]
        
        self.input_conf = 0.4  # class_thres
        self.input_thres = 0.6  # nms_thres    
        
        #callibration check
        try:
            cam_par = open("cam_parameters_"+str(instance)+".txt","r+")  
            self.x = ast.literal_eval(cam_par.read()) # eval(cam_par.read())
        except:
            self.CALIBRATE = True
            
 
       
    
    #SRI call the algo here 
    def gen(self,frame,current_time): 
        global frame_counter,device,net,cfg,logger,model1,unmasked_count,masked_count,previous_time
        
        if(self.CALIBRATE):
            print("gen.calibrate called "+str(self.instance))
            self.logger.info("gen.calibrate camera "+str(self.instance))
            frame_calib = frame.copy()
            res = self.Calibrate_obj.calibrate(self.netsd,self.ln,self.input_conf,self.input_thres,self.dim,self.ds,self.ha,frame_calib)
            self.CALIBRATE = res['isCalibrationRequired']
            if(not self.CALIBRATE):
                self.logger.info("Calibration finished, reading param file."+str(self.instance))
                cam_par = open("cam_parameters_"+str(self.instance)+".txt","r+")  
                self.x = ast.literal_eval(cam_par.read())   # eval(cam_par.read())
            return {
            	"frame" : res['frame'],
            	"timestamp" : current_time,
            	"stats"  : False,
            	"module" : "SD",
            	"message" : None
            } 
        
        img = np.float32(frame)
        im_height, im_width, _ = img.shape
        self.frame_width = im_width
        self.frame_height = im_height

        # SD --         
        res = get_pred(img, self.frame_width, self.frame_height, self.dim[1],self.dim[0], self.netsd, self.ln, self.input_conf, self.input_thres)
        #print(res)

        if len(res)>0:
            res1 = bb2array(res)
            #print(res1)
            #print(res1[:,2])
            frame = add_mask(frame,res1)
            ellipses = get_ellipses(res1,self.x,self.frame_height)
            red_bounds,groups,violations = distancing(self.x,res1,ellipses,self.ds,self.frame_width,self.frame_height)

            frame = bounding_ellipse(frame,self.x,res,ellipses,self.frame_width,self.frame_height,red_bounds)
            #frame = add_mask(frame,res1)
            msg = {'crowd_count':len(res),'groups':groups,'unsafe':len(red_bounds),'violations':violations}
        else:
            msg = {'crowd_count':0,'groups':{},'unsafe':0,'violations':0}
            
        #if len(res)>0:
        #    res1 = bb2array(res)
        #    ellipses = get_ellipses(res1,self.x,self.frame_height)
        #    red_bounds,groups,violations = distancing(self.x,res1,ellipses,self.ds,self.frame_width,self.frame_height)
        #SD --

                
                #sd --
            #if len(res)>0:
            #    frame = bounding_ellipse(frame,self.x,res,ellipses,self.frame_width,self.frame_height,red_bounds)
            #    msg = {'crowd_count':len(res),'groups':groups,'unsafe':len(red_bounds),'violations':violations}
            #else:
            #    msg = {'crowd_count':0,'groups':{},'unsafe':0,'violations':0}
                #sd--

        
        #global previous_time
        print(current_time)
        ret = {
        	"frame" : frame,
        	"timestamp" : current_time,
        	"stats"  : True,
        	"module" : "SD",
        	"message" : msg
        }
        return ret

 
