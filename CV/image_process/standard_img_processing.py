from __future__ import print_function
from __future__ import division
from scipy import ndimage
import cv2
from random import randint
import numpy as np
import os




def brightness_augmentation_f(image_input):
    beta_int = 0
    alpha_random_int = randint(50, 150)
    alpha_int = alpha_random_int / 100
    image_output = cv2.convertScaleAbs(image_input, alpha=alpha_int, beta=beta_int)
    return image_output


def contrast_augmentation_f(image_input):
    alpha_int = 1.0
    beta_random_int = randint(50, 120)
    beta_int = beta_random_int - 100
    image_output = cv2.convertScaleAbs(image_input, alpha=alpha_int, beta=beta_int)
    return image_output


def gamma_augmentation_f(image_input):
    gamma_random_int = randint(80, 120)
    gamma_int = gamma_random_int / 100
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma_int) * 255.0, 0, 255)
    image_output = cv2.LUT(image_input, lookUpTable)
    return image_output

  
def saturation_image_f(image_input):
    try:
        saturation = randint(20, 80)
        image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2HSV)
        v = image_input[:, :, 2]
        v = np.where(v <= 255 - saturation, v + saturation, 255)
        image_input[:, :, 2] = v
        image_output = cv2.cvtColor(image_input, cv2.COLOR_HSV2BGR)
    except cv2.error:
        image_output = image_input

    return image_output  
  
  
  
def gausian_blur_f(image_input):
    blur_int = randint(0, 3)
    image_output = cv2.GaussianBlur(image_input,(5,5),blur_int)
    return image_output
  
  
  
