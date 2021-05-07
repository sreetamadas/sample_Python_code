
import time
import glob
import subprocess
import PIL
import PIL.Image as Image
import json
import random

def inference():
    model="yolo4_tiny"
    if model=="yolo4_tiny":
        cfg = "/mnt/cfg/yolo4-tiny-obj.cfg"
        inference_weights = "/mnt/tiny_yolo/checkpoints/yolo4-tiny-obj_best.weights"
    else:
        cfg="/mnt/sujit/neu_yolo4tiny/cfg/yolo-obj.cfg"
        inference_weights = "/mnt/weights/yolo-obj_best.weights"	
    test_images_filepath = "/mnt/obj/test.txt"
    image_List = []
    with open((test_images_filepath), 'r') as fobj:
        for line in fobj:
            image_List.append(line.rstrip("\n"))
    fobj.close()
    d=0
    Test_Results = {}
    tot_time=0
    process_time=0
    len_images=len(image_List)
    for image in image_List:
        tmp=0
        tmp1=0
        print(image)
        start_time=time.time()
        out = subprocess.Popen(['./darknet', 'detector', 'test', '/mnt/tiny_yolo/obj/obj.data',cfg,
                                inference_weights,
                                '-dont_show', '-ext_output', '-out', '/mnt/tiny_yolo/output/tmp_file.json', image], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        tmp=(time.time() - start_time)
        print(" Inference time: %s seconds " % (tmp)) 
        tot_time=tot_time+tmp
        stdout, stderr = out.communicate()
        predicted_image = Image.open("predictions.jpg")
        output = "/mnt/tiny_yolo/output/inference_op/predicted_image%d.jpg" % d
        predicted_image.save(output)
        tmp1=(time.time()-start_time)
        print(" Inferencing & Processing time: %s seconds " % (tmp1))
        process_time=process_time+tmp1  
        pred_file = "/mnt/tiny_yolo/output/tmp_file.json" 
        with open(pred_file, 'r+') as f:
            data = json.load(f) 
        Res_tmp=[]
        ClassID=[]
        Conf=[]
        for i in range(0, len(data)):
            classID = []
            conf = []
            for j in range(0, len(data[i]['objects'])):
                ClassID.append(data[i]['objects'][j]['name'])
                Conf.append(data[i]['objects'][j]['confidence'])   
        # assign metadata
        mc_list = ['Z1123420-A', 'Z1123421-A', 'Y1123420-B', 'Y1123421-B']
        mc_id = random.choice(mc_list)
        if mc_id == 'Z1123420-A':
            line_id = 'A-1'
        elif mc_id == 'Z1123421-A':
            line_id = 'A-1'
        elif mc_id == 'Y1123420-B':
            line_id = 'B-1'
        elif mc_id == 'Y1123421-B':
            line_id = 'B-1' 
        Res_tmp.append(ClassID)
        Res_tmp.append(Conf)
        Res_tmp.append(line_id)
        Res_tmp.append(mc_id)
        Res_tmp.append(output)
        Test_Results["Test_Image_"+str(d)] = Res_tmp  
        with open('//mnt/output/tinyyolo4_test_results.json', 'w') as f:
            json.dump(Test_Results, f)
        d += 1  
    print(" Avg. inference time: %s seconds " % (tot_time/len_images))  
    print(" Avg. processing time including inferencing : %s seconds " % (process_time/len_images))         

if __name__ == '__main__':
    inference()              
