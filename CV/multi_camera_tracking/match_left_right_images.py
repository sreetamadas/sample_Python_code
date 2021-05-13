#!/usr/bin/env python
# coding: utf-8

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
# https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/
# In[2]:

# requires openCV 3.4.2 ; #opencv>3.4.2.16 - SIFT not available ; can replace with something else instead of SIFT, for higher version of openCV
import cv2
import cv2.xfeatures2d as cv
import numpy
from matplotlib import pyplot as plt


# In[3]:

a=1
b=2


'''right image'''
right = 'C:/Users/id/data/right_images_overlap/r_1.jpg'
img1 = cv2.imread(r'C:/Users/id/data/right_images_overlap/r_1.jpg')  

'''left image'''
left = 'C:/Users/id/data/left_images_overlap/l_1.jpg'
img2 = cv2.imread(r'C:/Users/id/data/left_images_overlap/l_1.jpg')



# In[4]:

''' convert to grayscale'''
t1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
t2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# In[5]:

''' get SIFT features/ descriptors - can replace with something else instead of SIFT, for higher version of openCV'''
sift=cv.SIFT_create()
#opencv>3.4.2.16 - SIFT not available

kp1, des1 = sift.detectAndCompute(t1, None)
kp2, des2 = sift.detectAndCompute(t2, None)


# In[9]:

''' length of descriptor : ususally 128'''
#len(des1[0])


# In[9]:

'''show keypoints in each image - for understanding only '''

#f=cv2.drawKeypoints(t1,kp1,None,[0,0,255],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#plt.imshow(f)
#nf=cv2.drawKeypoints(t2,kp2,None,[255,0,0],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#plt.imshow(nf)

# In[10]:
'''find common keypoints in both images, using ratio test'''

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1,des2, k=2) 
# knnMatch to get k best matches; else bf.match to get best match only
good1 = []
for m,n in matches:
    if m.distance < 0.65*n.distance:  # 0.65
        good1.append([m])

matches = bf.knnMatch(des2,des1, k=2)
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


# In[11]:

'''show the matches'''
len(good)
thres = int(0.75*len(good))
good_matches = good[:thres]  #10  # set this no. to 75% of matches(= len(good)) in prev. 2 step

result=cv2.drawMatchesKnn(t1,kp1,t2,kp2,good_matches,None,[0,0,255],flags=2)
plt.imshow(result)

#cv2.imwrite(r'D:\PARKING\park\matches\\'+str(a)+'-'+str(b)+'.jpg', result)
#cv2.imshow("res",result)
#cv2.waitKey(0) 
#cv2.destroyAllWindows() 

# In[12]:

#good_matches

# In[12]:

''' getting the keypoint coordinates (x,y) sorted by x or y '''
list_kp1 = []
list_kp2 = []
for mat in good_matches:
    img1_idx = mat[0].queryIdx
    img2_idx = mat[0].trainIdx

    # Get the coordinates
    (x1, y1) = kp1[img1_idx].pt
    (x2, y2) = kp2[img2_idx].pt

    # Append to each list
    list_kp1.append((int(x1), int(y1)))
    list_kp2.append((int(x2), int(y2)))
    
list_kp1 = sorted(list_kp1, key=lambda k : k[0])  # sort by value of x-coord in (x,y) 
list_kp2 = sorted(list_kp2, key=lambda k : k[0])

list_kp1 = sorted(list_kp1, key=lambda k : k[1])  # sort by value of y-coord in (x,y) 
list_kp2 = sorted(list_kp2, key=lambda k : k[1])

# In[13]:

#print(list_kp2)
#list_kp1

# In[136]:
''' connecting keypoints with line on img1 '''
#im = img1
#for i in range(len(list_kp1)-1):
#    im = cv2.line(img1, list_kp1[i], list_kp1[i+1], (0,255,0), 3)
#plt.imshow(im)

# In[137]:
''' connecting keypoints with line on img2 '''
#im = img2
#for i in range(len(list_kp2)-1):
#    im = cv2.line(img2, list_kp2[i], list_kp2[i+1], (0,255,0), 3)
#plt.imshow(im)

# In[140]:
#list_kp1
#list_kp2

# In[145]:

'''for the right hand side image,'''
# take x- coordinates of keypoints which are  having higher value
# right image uploaded as img1

x_max= list_kp1[0][0]   # list_kp1[0][0]; changed indices from 1 to 2
listabv_1=[]
for i in range(len(list_kp1)):
    if list_kp1[i][0]>=x_max:
        listabv_1.append(list_kp1[i])
        x_max = list_kp1[i][0]
listbel_1 = []
x_max = list_kp1[len(list_kp1)-1][0]
for i in range(len(list_kp1)):
    i=len(list_kp1)-i-1
    if list_kp1[i][0]>=x_max:
        listbel_1.append(list_kp1[i])
        x_max = list_kp1[i][0]



list_1=[]

#list_1.append((img1.shape[1],0))  # image width
list_1.append((0,0))
list_1.append((listabv_1[0][0],0))
[list_1.append(listabv_1[i]) for i in range(len(listabv_1))]
[list_1.append(listbel_1[len(listbel_1)-1-i]) for i in range(1,len(listbel_1))]
list_1.append((listbel_1[0][0],img1.shape[0]))
#list_1.append((img1.shape[1],img1.shape[0]))  # image width, height
list_1.append((0,img1.shape[0]))


#im = cv2.line(img1, (listabv_1[0][0],0), listabv_1[0], (0,255,0), 3)
'''#for i in range(len(listabv_1)-1):
#    im = cv2.line(img2, listabv_1[i], listabv_1[i+1], (0,255,0), 3)'''
#im = cv2.line(img1, listabv_1[len(listabv_1)-1], (listabv_1[len(listabv_1)-1][0],img1.shape[0]),
#              (0,255,0), 3)
#plt.imshow(im)


# In[12]:

''' for the left hand side image, '''
#take x- coordinates of keypoints which are  having lower value
# left image uploaded as img2

x_min=list_kp2[0][0]
listabv_2=[]
for i in range(len(list_kp2)):
    if list_kp2[i][0]<=x_min:
        listabv_2.append(list_kp2[i])
        x_min = list_kp2[i][0]
listbel_2 = []
x_min = list_kp2[len(list_kp2)-1][0]
for i in range(len(list_kp2)):
    i=len(list_kp2)-i-1
    if list_kp2[i][0]<=x_min:
        listbel_2.append(list_kp2[i])
        x_min = list_kp2[i][0]


list_2=[]
list_2.append((img2.shape[1],0))  # image width
list_2.append((listabv_2[0][0],0))
[list_2.append(listabv_2[i]) for i in range(len(listabv_2))]
[list_2.append(listbel_2[len(listbel_2)-1-i]) for i in range(1,len(listbel_2))]
list_2.append((listbel_2[0][0],img2.shape[0]))
list_2.append((img2.shape[1],img2.shape[0]))  # image width, height


#im = cv2.line(img2, (listabv_2[0][0],0), listabv_2[0], (0,255,0), 3)
'''#for i in range(len(listabv_1)-1):
#    im = cv2.line(img2, listabv_1[i], listabv_1[i+1], (0,255,0), 3)'''
#im = cv2.line(img2, listabv_2[len(listabv_2)-1], (listabv_2[len(listabv_2)-1][0],img2.shape[0]),
#              (0,255,0), 3)
#plt.imshow(im)


# In[17]:

''' overlay for left image'''
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

#image = Image.open(r'C:\Users\id\data\raw\0_left.jpg')
image = Image.open(left)

overlay = image.copy()
output = image.copy()

draw = ImageDraw.Draw(overlay)  # ImageDraw.Draw(Image.fromarray(overlay))

# points = ((1,1), (2,1), (2,2), (1,2), (0.5,1.5))
# points = ((100, 100), (200, 100), (200, 200), (100, 200), (50, 150))
draw.polygon(list_2, fill=200)
#draw.polygon(list_2_n, fill=200)  '''see below'''


overlay = numpy.array(overlay)
output = numpy.array(output)

cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
plt.imshow(output)

# 

''' overlay for right image'''
#import PIL.ImageDraw as ImageDraw
#import PIL.Image as Image

#image = Image.open(r'C:\Users\id\data\raw\0_right.jpg')
image = Image.open(right)

overlay = image.copy()
output = image.copy()

draw = ImageDraw.Draw(overlay)  # ImageDraw.Draw(Image.fromarray(overlay))

# points = ((1,1), (2,1), (2,2), (1,2), (0.5,1.5))
# points = ((100, 100), (200, 100), (200, 200), (100, 200), (50, 150))
draw.polygon(list_1, fill=200)
#draw.polygon(list_1_n, fill=200)  '''see below'''

overlay = numpy.array(overlay)
output = numpy.array(output)

cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
plt.imshow(output)

# output = output*5+overlay*5


# In[ ]:

'''selecting 4 points to denote minimal ROI'''
######################################
''' for left image: list_2 '''
'''top right'''
list_2_n=[]
list_2_n.append((img2.shape[1],0))  # image width

'''top left'''
y1 = 0 #list_2[0][0]
x1 = 0
#listabv_2=[]
for i in range(len(list_2)):
    if list_2[i][1] <= int(0.1 * img2.shape[0]):
        y1 = list_2[i][1]
        x1 = list_2[i][0]

'''bottom left:scroll the list in reverse'''
y2 = 0
x2 = 0
for i in range(len(list_2)):
    i=len(list_2)-i-1
    if list_2[i][1] >= int(0.9 * img2.shape[0]):
        y2 = list_2[i][1]
        x2 = list_2[i][0]

list_2_n.append((x1,y1))
list_2_n.append((x2,y2))
'''bottom right'''
list_2_n.append((img2.shape[1],img2.shape[0]))


########################################
''' for right image: list_1 '''
'''top left'''

list_1_n=[]
list_1_n.append((0,0))  # image width

'''top right'''
y1 = 0 #list_2[0][0]
x1 = 0
#listabv_2=[]
for i in range(len(list_1)):
    if list_1[i][1] <= int(0.1 * img1.shape[0]):
        y1 = list_1[i][1]
        x1 = list_1[i][0]

'''bottom rightt:scroll the list in reverse'''
y2 = 0
x2 = 0
for i in range(len(list_1)):
    i=len(list_1)-i-1
    if list_1[i][1] >= int(0.9 * img1.shape[0]):
        y2 = list_1[i][1]
        x2 = list_1[i][0]

list_1_n.append((x1,y1))
list_1_n.append((x2,y2))
'''bottom left'''
list_1_n.append((0,img1.shape[0]))



