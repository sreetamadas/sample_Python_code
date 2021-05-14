# create video from images
# input images are of different sizes

# https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
# https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md#assemble-video-from-sequence-of-frames

# step 1: list the images in proper order
# step 2: make the images square: using the dimensions of the largest images
# step 3: create the video


import os
import pandas as pd
import cv2
import numpy as np
from PIL import Image


## STEP 1
path = "/content/gdrive/My Drive/detect/img/"
os.chdir(path)
flist = os.listdir(path)
file_list = [f for f in flist if f.endswith('.jpg')]   # .txt  # .csv
flist = sorted(file_list, key = lambda x: (isinstance(x, str), x))




## STEP 2
## get image dimensions
all_mode = []
for filename in flist:
  img = cv2.imread(filename)
  height, width, layers = img.shape
  all_mode.append([filename,height,width])
all_mode = pd.DataFrame(all_mode, columns=['filename','height','width'])

## show max & min dim
print(all_mode['height'].max())
print(all_mode['height'].min())
print(all_mode['width'].max())
print(all_mode['width'].min())

## functions
#from PIL import Image
def make_square(im, min_size=512, fill_color=(255)):   #fill_color=(255, 255, 255, 0)  # fill_color=(255, 255, 255)
    # fill_color=(0, 0, 0, 0) <- black
    # fill_color=(255, 255, 255, 0) <- white
    x, y = im.size
    
    # make square image,
    if x != y:
        size = max(x, y)
        new_im = Image.new('RGB', (size, size), fill_color)   # Image.new('RGB',  #'RGBA' is not supported in jpg image
        new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
        #imResize = new_im.resize((min_size,min_size), Image.ANTIALIAS)
    else:
        new_im = im
        # resize images
        #imResize = im.resize((min_size,min_size), Image.ANTIALIAS)           
    return new_im


def im_resize(im, min_size=512):
    # https://www.daniweb.com/programming/software-development/code/216637/resize-an-image-python
    x, y = im.size    
    if x > min_size:  # downsample
        imResize = im.resize((min_size,min_size), Image.ANTIALIAS)   # use of other options? cubic spline?
    elif x < min_size: # upsample
        imResize = im.resize((min_size, min_size), Image.BILINEAR)
    else:
        imResize = im    
    return imResize

  
## resize images & append
#img_array = []
os.mkdir("reshaped")
count = 1

#for i in range(len(all_mode)):
for filename in flist:
  im = Image.open(filename) #cv2.imread(filename)
  #print(all_mode.loc[i, 'filename'])
  #im = Image.open(all_mode.loc[i, 'filename'])
  new_im = make_square(im)
  #imResize = im_resize(new_im)
  dst = "./reshaped/img-" + "{:0>3d}".format(count) + '.jpg'
  count = count + 1
  new_im.save(dst)
  #imResize.save(dst)
  #img = cv2.imread(filename)
  #imResize = resize_image(img)
  #imResize = resizeAndPad(img, (512,512), 127)
  #img_array.append(imResize)  
  
  
## STEP 3
os.chdir("reshaped")
!ffmpeg -framerate 2 -i img-%003d.jpg video.avi

'''
# STEP 3: method 2 (did not work)
#out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)  #MJPG
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'MJPG'), 5, (512,512))  #MJPG
 
for i in range(len(img_array)):
    out.write(img_array[i])
#for i in range(len(all_mode)):
  ##img = cv2.imread(filename)
  #imResize = resizeAndPad(img, (512,512), 127)
  #out.write(imResize)
cv2.destroyAllWindows()
out.release()
'''


#####################################################
# step 2: alternate methods : did not work

  
'''def resize_image(img, size=(512,512)):
    h, w = img.shape[:2]

    if h == w: 
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)
'''

'''
def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img
'''

















