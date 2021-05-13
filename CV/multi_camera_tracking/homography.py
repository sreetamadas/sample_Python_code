# -*- coding: utf-8 -*-
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html#feature-homography
# https://python.hotexamples.com/examples/cv2/-/drawMatchesKnn/python-drawmatchesknn-function-examples.html	 ## CHECK CODE

import cv2
import cv2.xfeatures2d as cv
import numpy as np
from matplotlib import pyplot as plt

a=1
b=2
img1 = cv2.imread(r'C:\Users\id\data\raw\cam'+str(a)+r'\2.jpg')  
img2 = cv2.imread(r'C:\Users\id\data\raw\cam'+str(b)+r'\2.jpg') 

''' convert to grayscale'''
t1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
t2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

'''SIFT features'''
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
#kp1_img = cv2.drawKeypoints(img1, kp1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	
kp2, des2 = sift.detectAndCompute(img2,None)
#kp2_img=cv2.drawKeypoints(img2,kp2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

''' find common keypoints'''
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)


#src = []
#dst = []

'''store good matches'''
good = []
for m,n in matches:
	if m.distance < 0.7*n.distance:
		good.append([m])
len(good)

'''If enough matches are found, we extract the locations of matched keypoints in both the images. 
They are passed to find the perpective transformation. Once we get this 3x3 transformation matrix, 
we use it to transform the corners of queryImage to corresponding points in trainImage. Then we draw it.
'''
MIN_MATCH_COUNT = 10
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = t1.shape #img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None


# the below lines are not working correctly
# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                   singlePointColor = None,
#                   matchesMask = matchesMask, # draw only inliers
#                   flags = 2)
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params)  # drawMatches requires all matches

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, img2, flags=2)   # drawMatchesKnn requires good matches from ratio test
#plt.imshow(img3, 'gray'),plt.show()
#cv2.imshow("img",img3)
#cv2.waitKey(0) 
#cv2.destroyAllWindows() 
cv2.imwrite(r'C:\Users\id\data\raw\\'+str(a)+'-'+str(b)+'.jpg', img3)
