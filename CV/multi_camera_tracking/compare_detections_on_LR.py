# In[]:
'''code to compare common detections (boxes) in 2 images
STEPS:
    1. split video into left & right - done
    
    2. Standardisation of common FOV: take 2 images, one from each, & find overlapping FOV
    Common FOV (using frames l_1 & r_1):
    Left image: [(720, 0), (245, 16), (370, 514), (720, 540)]
    Right image: [(0, 0), (333, 9), (308, 516), (0, 540)]

    
    3. run detection on the 2 videos & get boxes - saved as log file
    
    
    4. find detections in the FOV
    
    5. compare from overlapping FOVs
'''

# In[]:
# requires openCV 3.4.2 for SIFT
import cv2
import cv2.xfeatures2d as cv
import numpy
from matplotlib import pyplot as plt
#import json
import ast
#from skimage.measure import structural_similarity as ssim
from skimage.metrics import structural_similarity as ssim
import pandas as pd


# In[]:
''' common FOV'''
left_FOV = [(720, 0), (245, 16), (370, 514), (720, 540)] # order: top-right, top-left, bottom-left, bottom-right
right_FOV = [(0, 0), (333, 9), (308, 516), (0, 540)]  # order: top-left, top-right, bottom-right, bottom-left


# In[]:
'''extract detections for the corresponding frames'''
path = "C:/Users/id/Desktop/data/multi_camera_tracking/SIFT_keypoint_match/"

# left image detections
file = path + "l_163_log.json"
with open(file, 'r') as f:
    data = f.read() #json.load(f)
    data = "\"".join(data.split("'"))
    left_pred = ast.literal_eval(data)
    
# f = open(file)
# dat = f.read()
# dat = "\"".join(dat.split("'"))
# left_pred = json.loads(dat)[0] #json.loads(json.dumps(dat))[0] #json.loads(dat) #json.loads(json.dumps(dat))
# type(left_pred)
# f.close()


# right image detections
file = path + "r_163_log.json"
with open(file, 'r') as f2:
    data = f2.read() #json.load(f)
    data = "\"".join(data.split("'"))
    right_pred = ast.literal_eval(data)



# In[]:
'''get detections within FOV'''

'''left image'''
img2 = cv2.imread(r'C:/Users/id/desktop/data/left_images_overlap/l_163.jpg')
#output (this is the figure with FOV superposed from matching.py)

left_list = []
for i in range(0,len(left_pred["results"])):
    xmin = left_pred["results"][i]["xmin"]
    ymin = left_pred["results"][i]["ymin"]
    ymax = left_pred["results"][i]["ymax"]
    xmax = left_pred["results"][i]["xmax"]
    tid = left_pred["results"][i]["id"]
    if ymin >= left_FOV[1][1] and ymax <= left_FOV[2][1] and xmin >= left_FOV[1][0] and xmin >= left_FOV[2][0]:
        left_list.append(left_pred["results"][i])
        # check if the boxes saved here are correct, by printing them onto the image
        cv2.rectangle(img2, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255,255,0), 2)
        #text = "{}:".format(tid)
        #cv2.putText(output, tid, (int(xmin), (ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
#cv2.imshow("Image", img2)  
plt.imshow(img2)  #output

# detections on entire image
# img2 = cv2.imread(r'C:/Users/id/Desktop/data/left_images_overlap/l_163.jpg')
# for i in range(0,len(left_pred["results"])):
#     xmin = left_pred["results"][i]["xmin"]
#     ymin = left_pred["results"][i]["ymin"]
#     ymax = left_pred["results"][i]["ymax"]
#     xmax = left_pred["results"][i]["xmax"]
#     tid = left_pred["results"][i]["id"]
#     cv2.rectangle(img2, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255,255,0), 2)
#     text = "{}:".format(tid)
#     cv2.putText(img2, text, (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
# plt.imshow(img2)   #cv2.imwrite("C:/Users/DAR9KOR/Desktop/test.jpg", img2)
    


'''right image'''
img1 = cv2.imread(r'C:/Users/id/Desktop/data/right_images_overlap/r_163.jpg')

right_list = []
for i in range(0,len(right_pred["results"])):
    xmax = right_pred["results"][i]["xmax"]
    ymin = right_pred["results"][i]["ymin"]
    ymax = right_pred["results"][i]["ymax"]
    xmin = right_pred["results"][i]["xmin"]
    tid = right_pred["results"][i]["id"]
    ###print(tid, (ymin, right_FOV[1][1]), (ymax, right_FOV[2][1]), (xmax, right_FOV[1][0], right_FOV[2][0]))
    ###print(type(ymin), type(right_FOV[1][1]))
    if ymin >= float(right_FOV[1][1]) and ymax <= float(right_FOV[2][1]) and xmax <= float(right_FOV[1][0]) and xmax <= float(right_FOV[2][0]):
        right_list.append(right_pred["results"][i])
        # check if the boxes saved here are correct, by printing them onto the image
        cv2.rectangle(img1, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255,255,0), 2)
        #text = "{}:".format(tid)
        #cv2.putText(output, text, (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
plt.imshow(img1) #output


# detections on entire image
# img1 = cv2.imread(r'C:/Users/DAR9KOR/Desktop/data/sample_datasets/distance_detection/data/video/right_images_overlap/r_163.jpg')
# for i in range(0,len(right_pred["results"])):
#     xmin = right_pred["results"][i]["xmin"]
#     ymin = right_pred["results"][i]["ymin"]
#     ymax = right_pred["results"][i]["ymax"]
#     xmax = right_pred["results"][i]["xmax"]
#     tid = right_pred["results"][i]["id"]
#     cv2.rectangle(img1, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255,255,0), 2)
#     text = "{}:".format(tid)
#     cv2.putText(img1, text, (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
# plt.imshow(img1)  
# cv2.imwrite("C:/Users/DAR9KOR/Desktop/test.jpg", img1)

# In[]:
'''compare & find common detections'''
# https://stackoverflow.com/questions/11541154/how-can-i-assess-how-similar-two-images-are-with-opencv
# comparing histograms, template matching, feature matching
# https://gist.github.com/duhaime/211365edaddf7ff89c0a36d9f3f7956c

#left_com = []
#right_com = []

df = []



### convert the images to gray_scale, for SSIM  & sift_sim
img_left = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img_right = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)



### convert images to HSV, for histogram matching
img_left = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
img_right = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
### histogram parameters
h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]
# hue varies from 0 to 179, saturation from 0 to 255
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges # concat lists
# Use the 0-th and 1-st channels
channels = [0, 1]



### feature matching
def sift_sim(l_crop, r_crop):
  '''   Use SIFT features to measure image similarity  '''
  # get the detection regions: l_crop, r_crop
  
  # initialize the sift feature detector
  #orb = cv2.ORB_create()    # cv.SIFT_create()
  # find the keypoints and descriptors with SIFT (inputs should be grayscale)
  #kp_a, desc_a = orb.detectAndCompute(l_crop, None)  # sift.detectAndCompute(l_crop, None)
  #kp_b, desc_b = orb.detectAndCompute(r_crop, None)

  # initialize the bruteforce matcher
  #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  
  # match.distance is a float between {0:100} - lower means more similar
  #matches = bf.match(desc_a, desc_b)
  #similar_regions = [i for i in matches if i.distance < 70]
  #if len(matches) == 0:
  #  return 0
  #return len(similar_regions) / len(matches)  
  
  
  # initialize detector
  sift=cv.SIFT_create()
  # find the keypoints and descriptors with SIFT (inputs should be grayscale)
  kp1, desc_a = sift.detectAndCompute(l_crop, None)
  kp2, desc_b = sift.detectAndCompute(r_crop, None)
  

  # initialize the bruteforce matcher
  bf = cv2.BFMatcher()

  # match
  matches = bf.knnMatch(desc_a,desc_b, k=2)
  good1 = []
  for m,n in matches:
      if m.distance < 0.65*n.distance:  # 0.65
        good1.append([m])

  matches = bf.knnMatch(desc_b,desc_a, k=2)
  good2 = []
  for m,n in matches:
    if m.distance < 0.65*n.distance:  # 0.65
        good2.append([m])
  
  good=[]
  for i in good1:
    img1_id1=i[0].queryIdx
    img2_id1=i[0].trainIdx

    (x1,y1)=kp1[img1_id1].pt
    (x2,y2)=kp2[img2_id1].pt

    for j in good2:
        img1_id2=j[0].queryIdx
        img2_id2=j[0].trainIdx

        (a1,b1)=kp2[img1_id2].pt
        (a2,b2)=kp1[img2_id2].pt

        if (a1 == x2 and b1 == y2) and (a2 == x1 and b2 == y1):
            good.append(i)

  
  if len(matches) == 0:
    return 0
  return len(good) / len(matches)


###################################################
### crop the box & compare 
for i in range(0,len(left_list)):
    for j in range(0,len(right_list)):
        
        # left_crop
        start_x = round(left_list[i]["xmin"]) # use round instead of int
        end_x = round(left_list[i]["xmax"])
        start_y = round(left_list[i]["ymin"])
        end_y = round(left_list[i]["ymax"])
        l_id = left_list[i]["id"]
        #print(end_x - start_x, end_y - start_y)
        l_crop = img_left[start_y:end_y, start_x:end_x]  # startY and endY coordinates, followed by the startX and endX coordinates
        
        # right crop
        start_x = round(right_list[j]["xmin"])
        end_x = round(right_list[j]["xmax"])
        start_y = round(right_list[j]["ymin"])
        end_y = round(right_list[j]["ymax"])
        r_id = right_list[j]["id"]
        #print(end_x - start_x, end_y - start_y)
        r_crop = img_right[start_y:end_y, start_x:end_x]
        
        
        ### sift similarity
        s = sift_sim(l_crop,r_crop)
        df.append([i,l_id,j,r_id,s])
        
        
        '''
        ### compare with SSIM (structural similarity index) - requires input images in grayscale format
        # s = ssim(l_crop,r_crop)
        # this works only if the crops are of the same size
        # but resizing may be affecting quality of match
        s = ssim(cv2.resize(l_crop,(50,150)), cv2.resize(r_crop,(50,150)))
        #print(i,l_id,j,r_id,s)
        # 4 40055 3 29971 0.5484132000847112
        # 5 8774 5 55193 0.2181087538131895
        # see below, values reduced further with cubic interpolation during resize
        #s = ssim(cv2.resize(l_crop,(50,150),interpolation=cv2.INTER_CUBIC), cv2.resize(r_crop,(50,150),interpolation=cv2.INTER_CUBIC))
        #print(i,l_id,j,r_id,s)
        # 4 40055 3 29971 0.4985266258721171
        # 5 8774 5 55193 0.1612343079955875
        df.append([i,l_id,j,r_id,s])
        '''
        
        '''
        ### using histogram comparison
        # https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
        # https://docs.opencv.org/3.4/d6/dc7/group__imgproc__hist.html#ga994f53817d621e2e4228fc646342d386
        hist_l = cv2.calcHist([l_crop], channels, None, histSize, ranges, accumulate=False)
        cv2.normalize(hist_l, hist_l, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hist_r = cv2.calcHist([r_crop], channels, None, histSize, ranges, accumulate=False)
        cv2.normalize(hist_r, hist_r, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # compareHist(image1, image2, comparison method=int)
        s = cv2.compareHist(hist_l, hist_r, 0)  # 0 : correlation (high for good match)
        s1 = cv2.compareHist(hist_l, hist_r, 1) # 1 : chi-square (low .....)
        s2 = cv2.compareHist(hist_l, hist_r, 2) # 2 : intersection  (high ....)
        s3 = cv2.compareHist(hist_l, hist_r, 3) # 3 : Bhattacharyya distance  (low ...)
        
        #print(i,l_id,j,r_id, s, s1, s2, s3)
        # 4 40055 3 29971 0.9897954971549056
        # 5 8774 5 55193 0.9829198170374923
        df.append([i,l_id,j,r_id, s, s1, s2, s3])
        #'''

'''
# save SSIM results
df = pd.DataFrame(df, columns=['i','i_id','j','r_id','SSIM'])
df = df.sort_values('SSIM',ascending=False)
df.to_excel(path+"ssim_match.xlsx", index=False)
'''

'''
# save hist results
df = pd.DataFrame(df, columns = ['i','l_id','j','r_id','corr','chi','intersect','bhat_dist'])        
df = df.sort_values('corr',ascending=False)
df.to_excel(path+"hist_match.xlsx", index=False)
'''

# save sift_sim results
df = pd.DataFrame(df, columns=['i','i_id','j','r_id','sift_sim'])
df = df.sort_values('sift_sim',ascending=False)
df.to_excel(path+"sift_sim_match.xlsx", index=False)
