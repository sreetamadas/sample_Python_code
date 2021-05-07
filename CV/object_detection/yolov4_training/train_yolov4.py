import time
import glob
import subprocess
import PIL
import PIL.Image as Image

def train():  
    model="yolo4_tiny"
    if model=="yolo4_tiny":
        cfg = "/mnt/yolo/tiny_yolo/cfg/yolo4-tiny-obj_v6.cfg"
        weights = "/mnt/yolo/tiny_yolo/weights/yolov4-tiny.conv.29"
        inference_weights = "/mnt/yolo/checkpoints/yolo4-tiny-obj_v6_best.weights" 
    else:
        cfg="/mnt/yolo/cfg/yolo-obj.cfg"
        weights="/mnt/yolo/weights/yolov4.conv.137"
        inference_weights = "/mnt/yolo/checkpoints/yolo-obj_best.weights"   
    out = subprocess.Popen([ './darknet', 'detector','train','/mnt/obj/obj.data', cfg , weights,'-dont_show','-mjpeg_port 8090','-map'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout,stderr = out.communicate()	
    val = subprocess.Popen([ './darknet', 'detector','map','/mnt/obj/obj.data', cfg , inference_weights],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout,stderr = val.communicate()	
    with open('/mnt/output/mAP_tiny_v6.txt', 'w') as f:
		f.write("%s\n" % stdout)        
if __name__ == '__main__':
	train()
