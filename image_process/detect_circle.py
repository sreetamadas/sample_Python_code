# detect well & remove extraneous well portions from images
### pad to square or rectangle (by neighbouring pixels(?) or B/W) : diameter = length of square
### or should it be cropped from inside the well? : diameter = diagonal of square
### if the image is already a crop of the well, it should be left intact?

### https://content.instructables.com/ORIG/FYS/C6X1/IKECQ3CY/FYSC6X1IKECQ3CY.py https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
### https://stackoverflow.com/questions/21612258/filled-circle-detection-using-cv2-in-python
### https://stackoverflow.com/questions/35519102/detect-circle-like-shapes-opencv https://stackoverflow.com/questions/15878325/what-are-the-possible-fast-ways-to-detect-circle-in-an-image

### https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html (median filter vs gaussian filter: gaussian filter blurs edges; median filter is good for salt-&-pepper noise)
### https://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/ImageProcessing-html/topic4.htm (erosion, dilation) https://northstar-www.dartmouth.edu/doc/idl/html_6.2/Eroding_and_Dilating_Image_Objects.html
### https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html (simple vs adaptive thresholding)

# summary : HoughTramnsform based method worked (combination of blurring + threshold + erosion-dilation + Hough transform) FindContours did not work; need to check on more images
# other pre-processing to use: image smoothing, contrast enhancement, morphological operations, resize


import os
import numpy as np
import argparse
#import imutils
#from imutils import perspective
#from imutils import contours
import cv2


from matplotlib import pyplot as plt
%matplotlib inline
def showimage(img):
    plt.imshow(img) #(img,cmap='gray')
    plt.show()


# methods tried

# 1. grayscale + hough transform
# 2. grayscale + modified parameters of Hough transform
def Hough_transform(path,filename):
    image = cv2.imread(path+filename)  #'SUHV1_A1_POS.TIF'
    h, w, c = image.shape
    mindist = int(min(h,w) * 0.8)
    
    # keep copy of input image for final view
    output = image.copy()
    
    # convert image to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # modify the parameters for houghtransform & retry    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, mindist)#, param1=30, param2=65, minRadius=0, maxRadius=500)
    
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
        
        # show the output image
        #cv2.imshow("output", np.hstack([image, output]))
        #cv2.waitKey(0)
        showimage(output)


# 3. gray + blur Hough transform
def Hough_transform(path,filename):
    image = cv2.imread(path+filename)  #'SUHV1_A1_POS.TIF'
    h, w, c = image.shape
    mindist = int(min(h,w) * 0.8)
    
    # keep copy of input image for final view
    output = image.copy()
    
    # convert image to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
    #gray = cv2.GaussianBlur(gray,(5,5),0); # the tuple is the gaussian Kernel : controls the amount of blurring
    #gray = cv2.medianBlur(gray,5)
    
    # or
    gray = cv2.medianBlur(gray,5)
    gray = cv2.medianBlur(gray,7)
    gray = cv2.medianBlur(gray,9)
    # or
    gray = cv2.medianBlur(gray,7)
    gray = cv2.GaussianBlur(gray,(3,3),0)
    
    # modify the parameters for houghtransform & retry    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, mindist)#, param1=30, param2=65, minRadius=0, maxRadius=500)
    
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
        
        # show the output image
        #cv2.imshow("output", np.hstack([image, output]))
        #cv2.waitKey(0)
        showimage(output)


# 4. gray + blur + erosion-dilation + Hough transform
def Hough_transform(path,filename):
    image = cv2.imread(path+filename)  #'SUHV1_A1_POS.TIF'
    h, w, c = image.shape
    mindist = int(min(h,w) * 0.8)
    
    # keep copy of input image for final view
    output = image.copy()
    
    # convert image to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    # apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
    gray = cv2.GaussianBlur(gray,(5,5),0); # the tuple is the gaussian Kernel : controls the amount of blurring
    gray = cv2.medianBlur(gray,5)
    
    # erosion-dilation
    gray = cv2.erode(gray,None,iterations = 1)
    gray = cv2.dilate(gray,None,iterations = 1)
    
    
    # modify the parameters for houghtransform & retry    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, mindist)#, param1=30, param2=65, minRadius=0, maxRadius=500)
    
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
        
        # show the output image
        #cv2.imshow("output", np.hstack([image, output]))
        #cv2.waitKey(0)
        showimage(output)



# 5. gray + blur + threshold + erosion-dilation + Hough Transform
def Hough_transform(filename,outfile):
    start_time=time.time()
    image = cv2.imread(filename)  #'SUHV1_A1_POS.TIF'
    print(" read time: %s seconds " % (time.time() - start_time)) 
    
    h, w, c = image.shape
    mindist = int(min(h,w) * 0.8)
    
    # keep copy of input image for final view
    output = image.copy()
    
    # convert image to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    # apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
    gray = cv2.medianBlur(gray,9)
    #gray = cv2.medianBlur(gray,7)
    #gray = cv2.GaussianBlur(gray,(3,3),0)  # the tuple is the gaussian Kernel : controls the amount of blurring
    
    # threshold
    gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,3.5)
    
    # erosion-dilation
    gray = cv2.erode(gray,None,iterations = 3)
    gray = cv2.dilate(gray,None,iterations = 3)
    
    
    # modify the parameters for houghtransform & retry    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, mindist)#, param1=30, param2=65, minRadius=0, maxRadius=500)
    
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
        
        # show the output image
        #cv2.imshow("output", np.hstack([image, output]))
        #cv2.waitKey(0)
        #showimage(output)
        cv2.imwrite(outfile, output)
    else:
        cv2.imwrite(outfile, output)
    print(" Inference time: %s seconds " % (time.time() - start_time)) 



image_List=[]
with open(("filelist.txt"),'r') as fobj:
    for line in fobj:
        print(line)
        image_List.append(line.rstrip("\n"))
fobj.close()

d = 0
output_path = "C:/Users/DAR9KOR/Desktop/data/HEALTHCARE/Merck/OUTPUT_mblur9/"
for image in image_List:
    print(image)
    outfile = output_path + "out_image_" + str(d) + ".jpg"
    Hough_transform(image,outfile)
    d = d+1
    


## dp: This parameter is the inverse ratio of the accumulator resolution to the image resolution (see Yuen et al. for more details). 
Essentially, the larger the dp gets, the smaller the accumulator array gets.
## minDist: Minimum distance between the center (x, y) coordinates of detected circles. If the minDist is too small, multiple circles in the same neighborhood as the
original may be (falsely) detected. If the minDist is too large, then some circles may not be detected at all.
## param1: Gradient value used to handle edge detection in the Yuen et al. method.
## param2: Accumulator threshold value for the cv2.HOUGH_GRADIENT method. The smaller the threshold is, the more circles will be detected (including false circles).
The larger the threshold is, the more circles will potentially be returned.
## minRadius: Minimum size of the radius (in pixels). maxRadius: Maximum size of the radius (in pixels).

​



