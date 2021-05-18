


import os
import cv2
import numpy as np
import logging
import logging.handlers
from imutils.video import FPS
import warnings
import sys
import ast
from motpy import Box, Detection, MultiObjectTracker
warnings.filterwarnings("ignore")

import lib.calibrate_v1 as calib
from lib.people_detection_v2 import *
from lib.people_distance_v3 import *
from lib.tracker import *
from lib.utils import *

class BoschDL_SD:

    def __init__(self,instance,logs_queue,fps=5,parent_dir=None):
        self.CALIBRATE = None
        self.instance = instance
        h = logging.handlers.QueueHandler(logs_queue)
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(logging.DEBUG)
        self.logger = logging.getLogger('p.boschDL.'+str(instance))
        
        if parent_dir==None:
            # parent_dir = os.path.dirname(os.getcwd())
            parent_dir = os.path.abspath(__file__)
        else:
            parent_dir = parent_dir
        
        self.Calibrate_obj = calib.Calibrate(instance,parent_dir)
        
        # SD loading model
        print('parent_directory for algo.py')
        print(parent_dir)
        self.model_dir = os.path.join(parent_dir,'models')
        weightsPath = os.path.join(self.model_dir,'yolov3.weights')
        configPath = os.path.join(self.model_dir,'yolov3.cfg')
        
        self.netsd = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        
        # set up to access CUDA
        #self.netsd.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        #self.netsd.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA) 
        
        self.dim = (416,416)
        self.ha = 5.5
        self.ds = 6
        
        # determine only the *output* layer names that we need from YOLO
        self.ln = self.netsd.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.netsd.getUnconnectedOutLayers()]
        
        self.input_conf = 0.5  # class_thres
        self.input_thres = 0.5  # nms_thres    
        
        #callibration check
        try:
            cam_par_path = os.path.join(self.model_dir,"cam_parameters_{}.txt".format(str(self.instance)))
            cam_par = open(cam_par_path,"r+")  
            self.x = ast.literal_eval(cam_par.read()) #eval(cam_par.read()) 
        except:
            self.CALIBRATE = True
            
        #classifier
        # self.classifier = load_classifier()
        
        #tracker
        self.tracker = MultiObjectTracker(
        dt=1 / fps, tracker_kwargs={'max_staleness': 5},
        model_spec='constant_acceleration_and_static_box_size_2d',
        matching_fn_kwargs={'min_iou': 0.25})
        self.drop_detection_prob=0.0
        self.add_detection_noise=0.0
            
 
       
    
    #SRI call the algo here 
    def gen(self,frame,current_time): 
        global logger
        
        if(self.CALIBRATE):
            print("gen.calibrate called "+str(self.instance))
            self.logger.info("gen.calibrate camera "+str(self.instance))
            frame_calib = frame.copy()
            res = self.Calibrate_obj.calibrate(self.netsd,self.ln,self.input_conf,self.input_thres,self.dim,self.ds,self.ha,frame_calib)
            self.CALIBRATE = res['isCalibrationRequired']
            if(not self.CALIBRATE):
                self.logger.info("Calibration finished, reading param file."+str(self.instance))
                cam_par_path = os.path.join(self.model_dir,"cam_parameters_{}.txt".format(str(self.instance)))
                cam_par = open(cam_par_path,"r+") 
                # cam_par = open("cam_parameters_"+str(self.instance)+".txt","r+")  
                self.x = ast.literal_eval(cam_par.read())  #eval(cam_par.read())
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
        res = get_pred(frame, self.frame_width, self.frame_height, self.dim[1],self.dim[0], self.netsd, self.ln, self.input_conf, self.input_thres)
        # res = get_pred(img, self.frame_width, self.frame_height, self.dim[1],self.dim[0], self.netsd, self.ln, self.input_conf, self.input_thres)
        res = occlusion_correction(frame, self.frame_width, self.frame_height,res)
        # print(res)
        # res = [{'label': 0, 'score': '0.992', 'ymin': 406, 'xmin': 835, 'ymax': 636, 'xmax': 960}, {'label': 0, 'score': '0.978', 'ymin': 622, 'xmin': 1611, 'ymax': 889, 'xmax': 1726}, {'label': 0, 'score': '0.978', 'ymin': 830, 'xmin': 200, 'ymax': 1086, 'xmax': 327}, {'label': 0, 'score': '0.973', 'ymin': 24, 'xmin': 1458, 'ymax': 185, 'xmax': 1548}, {'label': 0, 'score': '0.969', 'ymin': 221, 'xmin': 736, 'ymax': 407, 'xmax': 811}, {'label': 0, 'score': '0.968', 'ymin': 234, 'xmin': 821, 'ymax': 412, 'xmax': 890}, {'label': 0, 'score': '0.966', 'ymin': 54, 'xmin': 890, 'ymax': 243, 'xmax': 946}, {'label': 0, 'score': '0.953', 'ymin': 101, 'xmin': 1647, 'ymax': 278, 'xmax': 1706}, {'label': 0, 'score': '0.951', 'ymin': -5, 'xmin': 989, 'ymax': 120, 'xmax': 1055}, {'label': 0, 'score': '0.908', 'ymin': 216, 'xmin': 306, 'ymax': 405, 'xmax': 394}, {'label': 0, 'score': '0.894', 'ymin': 292, 'xmin': 235, 'ymax': 491, 'xmax': 336}, {'label': 0, 'score': '0.888', 'ymin': -1, 'xmin': 1379, 'ymax': 120, 'xmax': 1438}, {'label': 0, 'score': '0.886', 'ymin': -3, 'xmin': 1323, 'ymax': 40, 'xmax': 1359}, {'label': 0, 'score': '0.874', 'ymin': -2, 'xmin': 1149, 'ymax': 72, 'xmax': 1192}, {'label': 0, 'score': '0.803', 'ymin': 975, 'xmin': 644, 'ymax': 1076, 'xmax': 806}]
        frame,res = track(frame,res,tracker=self.tracker,drop_detection_prob=self.drop_detection_prob, add_detection_noise=self.add_detection_noise)
        #print(res) ### CHANGED TO CHECK
        # res = occlusion_correction(frame, self.frame_width, self.frame_height,self.classifier,res)
        if len(res)>0:
            res1 = bb2array(res)
            frame = add_mask(frame,res1)
            ellipses = get_ellipses(res1,self.x,self.frame_height)
            red_bounds,groups,violations = distancing(self.x,res1,ellipses,self.ds,self.frame_width,self.frame_height)

            frame = bounding_ellipse(frame,self.x,res,ellipses,self.frame_width,self.frame_height,red_bounds)
            msg = {'crowd_count':len(res),'groups':groups,'unsafe':len(red_bounds),'violations':violations,'results':res}
        else:
            msg = {'crowd_count':0,'groups':{},'unsafe':0,'violations':0,'results':[]}

        
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

 
