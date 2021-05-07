#!/usr/bin/env python
# coding: utf-8

# In[ ]:
'''
Hough Transform: Findings
1.	The circles are not generated correctly for all the test images – need to finetune the pre-processing step. 
        tried several variants
        OTSU thresholding performed worse than adaptive for some cases, with complete darkening of the image

2.	Also, I found the code was very slow when processing the full size images (2048 x 1536, ~9MB), about 66-240 s , compared to < 1 s for the resized images.
	# without resizing: results not generating on the 1st image in 2-3 min; interrupted the code

3.     I generated the intermediate images – it shows the well boundary – not sure why Hough transform is not picking it up.
		added morphological open-close to improve

4.     For the large-sized images from phase 1, the results change if I resize the images.

5.    I also tried the contour-based method. It works for some images, but not for the others. Results are poorer than the Hough circle method.

6.    floodfill & connectedComponents did not work


TO DO:
1. mask & segment the well
2. remove the boundary & pad by black / white pixels
'''


# generate masks for well content using image processing

# adaptive threshold + dilation-erosion + opening



# output:
# 3 panel image : original, original + circle, mask (0 & 1)
# excel sheet with serial no, image path+name

# check the outputs, & add the correct/ incorrect labels to the above excel

# the correctly labeled images will be used for training segmentation


# In[1]:


import os
import time
import numpy as np
import pandas as pd
#import argparse
#import imutils
#from imutils import perspective
#from imutils import contours
import cv2


# #### create input image filelist

# In[2]:


def find_files( files, dirs=[], extensions=[]):
    new_dirs = []
    for d in dirs:
        try:
            new_dirs += [ os.path.join(d, f) for f in os.listdir(d) ]
        except OSError:
            if os.path.splitext(d)[1] in extensions:
                files.append(d)

    if new_dirs:
        find_files(files, new_dirs, extensions )
    else:
        return


# In[ ]:


'''
filedir = os.listdir(file_location)
    
for Dir in filedir:
    #print(Dir)
    Dir = path + Dir
    files = []
    find_files( files, dirs=[Dir], extensions=['.tif', '.TIF'] )    
    print(files[0])
'''


# In[5]:


#filedir = os.listdir(file_location)
filedir = ['folder1', 'dir2', 'dir3']
path = "C:/Users/Desktop/data/DETECT_WELL/New_Sample_Images_Resized/labeled/"


# In[9]:


files = []
for idx, item in enumerate(filedir):
    filedir[idx] = path + item

filedir


# In[10]:


find_files( files, dirs=filedir, extensions=['.tif', '.TIF'] )
print(files[0])


# In[11]:


print(files[95])


# In[12]:


print(files[96])


# In[18]:


len(files)


# In[ ]:





# #### getting circles on well boundary

# In[17]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
def showimage(img):
    plt.imshow(img) #(img,cmap='gray')
    plt.show()

# create mask
h = 512
w = 512

image = cv2.imread('C:/Users/Desktop/data/DETECT_WELL/New_Sample_Images_Resized/labeled/img.tif')
showimage(image)
circle_image = np.zeros((h, w), dtype=image.dtype)
cv2.circle(circle_image, (int(w/2),int(h/2)), 100, 255, -1)
showimage(circle_image)


# In[ ]:





# In[19]:


# adaptive thres + dilate-erode + open/close

def Hough_transform(filename,outfile):
    start_time=time.time()
    image = cv2.imread(filename)  
    print(" read time: %s seconds " % (time.time() - start_time)) 
    
    h, w, c = image.shape
    
    # resize large images
    if h>512:
        scale_percent = 30 # percent of original size
        w = int(image.shape[1] * scale_percent / 100)
        h = int(image.shape[0] * scale_percent / 100)
        dim = (w, h)
        # resize image
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    mindist = int(min(h,w) * 0.8)
    
    
    # keep copy of input image for final view
    output = image.copy()
    orig = image.copy()
    
    
    # convert image to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    # apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
    gray_blur = cv2.medianBlur(gray,9)
    #gray = cv2.medianBlur(gray,7)
    #gray = cv2.GaussianBlur(gray,(3,3),0)  # the tuple is the gaussian Kernel : controls the amount of blurring
    
    # threshold
    gray_thres = cv2.adaptiveThreshold(gray_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,3.5)
    #ret3,gray_thres = cv2.threshold(gray_blur,50,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Otsu
    
    
    # erosion-dilation
    #gray_e = cv2.erode(gray_thres,None,iterations = 3)
    #gray_d = cv2.dilate(gray_e,None,iterations = 3)
    # dilation-erosion
    gray_d = cv2.dilate(gray_thres,None,iterations = 1)
    gray_e = cv2.erode(gray_d,None,iterations = 1)
    #kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    #gray_d = cv2.dilate(gray_thres,kernel1,iterations = 1)
    #gray_e = cv2.erode(gray_d,kernel1,iterations = 1)
    
    # hole filling
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)) #(11,11)  (15,15)  (9,9)
    #close = cv2.morphologyEx(gray_e,cv2.MORPH_CLOSE,kernel1)
    # the closing operation adds white pixels; since the well boundary is black, it is breaking up,
    # need to use open operation instead
    close = cv2.morphologyEx(gray_e,cv2.MORPH_OPEN,kernel1)
    #close = cv2.morphologyEx(gray_thres,cv2.MORPH_OPEN,kernel1)
    
    # flood-fill
    #mask = np.zeros((h+2, w+2), np.uint8)
    #im_floodfill = close.copy()
    #cv2.floodFill(im_floodfill, mask, (0,0), 255);
    #im_floodfill_inv = cv2.bitwise_not(im_floodfill)
                                       
    
    # modify the parameters for houghtransform & retry    
    #circles = cv2.HoughCircles(gray_e, cv2.HOUGH_GRADIENT, 2, mindist)#, param1=30, param2=65, minRadius=0, maxRadius=500)
    circles = cv2.HoughCircles(close, cv2.HOUGH_GRADIENT, 2, mindist)#, param1=30, param2=65, minRadius=0, maxRadius=500)
    #circles = cv2.HoughCircles(im_floodfill_inv, cv2.HOUGH_GRADIENT, 2, mindist)#, param1=30, param2=65, minRadius=0, maxRadius=500)

    
    # create mask file
    circle_image = np.zeros((h, w), dtype=image.dtype)
    #cv2.circle(circle_image, (int(w/2),int(h/2)), r, 255, -1)
    
    
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            cv2.circle(circle_image, (x,y), r, 255, -1)
        
        # show the output image
        #cv2.imshow("output", np.hstack([image, output]))
        #cv2.waitKey(0)
        #showimage(output)
        #cv2.imwrite(outfile, output)   
        
    #else:
    #    cv2.imwrite(outfile, output)
    
    
        
    # Make the grey scale image have three channels
    #gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    #gray_blur3 = cv2.cvtColor(gray_blur, cv2.COLOR_GRAY2BGR)
    #gray_thres3 = cv2.cvtColor(gray_thres, cv2.COLOR_GRAY2BGR)
    #gray_e3 = cv2.cvtColor(gray_e, cv2.COLOR_GRAY2BGR)
    #gray_d3 = cv2.cvtColor(gray_d, cv2.COLOR_GRAY2BGR)
    #close3 = cv2.cvtColor(close, cv2.COLOR_GRAY2BGR)
    #im_floodfill3 = cv2.cvtColor(im_floodfill, cv2.COLOR_GRAY2BGR)
    #im_floodfill_inv3 = cv2.cvtColor(im_floodfill_inv, cv2.COLOR_GRAY2BGR)
    #output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    circle_image3 = cv2.cvtColor(circle_image, cv2.COLOR_GRAY2BGR)

    
    
    # concatenate the images
    #numpy_horizontal_concat = np.concatenate((gray_thres3, gray_d3, gray_e3, output), axis=1)
    #numpy_horizontal_concat = np.concatenate((gray_thres3, close3, output), axis=1)
    numpy_horizontal_concat = np.concatenate((orig, circle_image3, output), axis=1)
    cv2.imwrite(outfile, numpy_horizontal_concat)
    
    print(" Inference time: %s seconds " % (time.time() - start_time)) 


# In[20]:


# images from 3 folders

d = 0

output_path = "C:/Users/Desktop/data/DETECT_WELL/mask/"
os.mkdir(output_path)

df = []

for image in files: #image_List:
    print(d,image)
    outfile = output_path + "out_image_" + str(d) + ".jpg"
    Hough_transform(image,outfile)
    df.append([d,image,outfile])
    d = d+1


# In[21]:


df = pd.DataFrame(df, columns=['sr','in_file','out_file'])
out_df = output_path + 'mask_gen_perf.csv'
df.to_csv(out_df)


# In[ ]:





# In[ ]:





# In[25]:


# generate masks with a subset of images

image_List=[]
with open(("sample_filelist_mask.txt"),'r') as fobj:
    for line in fobj:
        #print(line)
        image_List.append(line.rstrip("\n"))
fobj.close()


# In[26]:


d = 0

output_path = "C:/Users/Desktop/data/DETECT_WELL/sample_all/"
#os.mkdir(output_path)

df = []

for image in image_List:
    print(d,image)
    outfile = output_path + "out_image_" + str(d) + ".jpg"
    Hough_transform(image,outfile)
    df.append([d,image,outfile])
    d = d+1


# In[ ]:





# #### separate image & masks

# In[ ]:


# output images in /DETECT_WELL/sample_all ; crop & generate the img & mask tile in 2 folders


# In[27]:


output_path = "C:/Users/Desktop/data/DETECT_WELL/sample_all/"
filedir = os.listdir(output_path)


# In[28]:


filedir[0]


# In[29]:


out1 = "C:/Users/Desktop/data/DETECT_WELL/sample_segment/images/"
out2 = "C:/Users/Desktop/data/DETECT_WELL/sample_segment/masks/"


# In[31]:


os.mkdir("C:/Users/Desktop/data/DETECT_WELL/sample_segment/")
os.mkdir(out1)
os.mkdir(out2)


# In[32]:


for filename in filedir:
    image = cv2.imread(output_path + filename)
    f = 'mask_' + filename
    
    # image
    start_y = 0
    end_y = image.shape[0]  # height,width = image.shape()
    start_x = 0
    end_x = end_y
    im = image[start_y:end_y, start_x:end_x]  
    
    # mask
    start_x = end_y
    end_x = 2 * end_y
    mask = image[start_y:end_y, start_x:end_x]
    
    cv2.imwrite(str(out1 + filename), im)
    cv2.imwrite(str(out2 + f), mask)
    


# In[ ]:





# In[ ]:




