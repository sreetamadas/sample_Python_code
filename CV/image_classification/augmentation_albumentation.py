'''
# apart from albumentation package, refer to other packages in the code for augmentation for object detection
other ref:
https://analyticsindiamag.com/hands-on-guide-to-albumentation/
https://albumentations.ai/docs/getting_started/image_augmentation/ - Classification

'''

#########################################################
####  Passing augmentation transformation as part of fastai  ######

import pandas as pd
from fastai.vision import *
torch.cuda.empty_cache()

import albumentations as A

def tensor2np(x):
    np_image = x.cpu().permute(1, 2, 0).numpy()
    np_image = (np_image * 255).astype(np.uint8)    
    return np_image


def alb_tfm2fastai(alb_tfm):
    def _alb_transformer(x):
        # tensor to numpy
        np_image = tensor2np(x)

        # apply albumentations
        transformed = alb_tfm(image=np_image)['image']

        # back to tensor
        tensor_image = pil2tensor(transformed, np.float32)
        tensor_image.div_(255)
        return tensor_image

    transformer = TfmPixel(_alb_transformer) 
    return transformer()


src = (ImageList.from_csv(path,'training_manual.csv',folder='training', suffix='.jpg')  
        .split_from_df(col='Validation')
        .label_from_df())
# training_manual.csv has 3 columns: image_filename , defect_class, Validation (TRUE / FALSE)

tfms = alb_tfm2fastai(A.Compose(
    [                   
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    ]
))


data = (src.transform(get_transforms(xtra_tfms=tfms),size=256)
       .databunch(num_workers=0)
       .normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(12,9))



########################################################
####   checking albumentation  #####
import albumentations as A
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)

#Create an instance of subclass compose and choose augmentation technique that works for the dataset
#And provide a probability value for each Aug element
#Set seed if you want many transformed images
transform = A.Compose([
    #OneOf([
    #    CLAHE(p=0.5),  
     #   GaussianBlur(3, p=0.3),
      #  IAASharpen(alpha=(0.2, 0.3), p=0.3),
    #], p=1),
    #A.RandomCrop(width=300, height=256),
    A.HorizontalFlip(p=0.2),
    A.Rotate((-5, +5)), # Rotaes the images to 90degree by default and doesn't suit our dataset
    A.VerticalFlip(p=0.5),
    A.Blur(blur_limit =(3, 7), p=0.2),
    A.RGBShift(p=0.5),
    A.HueSaturationValue(p=0.4),
    #A.CLAHE(p=0.2),
    #A.GaussianBlur(p = 0.4),
    #A.GaussNoise(p=0.),
    #A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
])    


pillow_image = Image.open(r"D:\project\sample\img_02.jpg")
image = np.array(pillow_image)
visualize(image)

transformed = transform(image=image)["image"]
visualize(transformed)


### augmenting many images in a folder
import os
import random #Set seed if you want many transformed images

file_dir=r'D:\project\sample_images'
output_path = r'D:\project\sample_Aug'

import os
import random
for root, _, files in os.walk(file_dir):
    print("Root",root)
    for file in files:
        name_int = file[:len(file)-4]
        Name=root+"\\"+file
        Original_image = cv2.imread(Name)
        image = cv2.cvtColor(Original_image, cv2.COLOR_BGR2RGB)
        transform = A.Compose([
        #A.HorizontalFlip(p=0.5),
        #A.Rotate((-30, +30)), # Rotaes the images to 90degree by default and doesn't suit our dataset
        #A.Normalize(p=0.8),
        #A.CLAHE(p=0.3),
        #A.VerticalFlip(p=0.5),
        #A.Blur(blur_limit =(3, 7), p=0.3),
        A.RGBShift(p=0.3),
        #A.HueSaturationValue(p=0.4),
        #A.RandomCrop(150,150, p=0.4)
        #A.RandomScale(p = 0.2)
        #A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.3)#, min_holes=None, min_height=None, min_width=None, fill_value=0, mask_fill_value=None, always_apply=False, p=0.5)
        #A.Equalize(mode='cv', p=0.2),
        #A.RandomBrightness(p=0.5)
        #A.RandomFog(p=0.3)
        #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.80, rotate_limit=45, p=.75),
        #A.ShiftScaleRotate(scale_limit=0.5), #zoom
        #A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.8, p= 0.2),
        #A.Gaussian(p=0.3)
        ])
        random.seed(7) 
        augmented_image = transform(image=image)['image']
        cv2.imwrite(output_path+"\\"+'%s' %str(name_int)+'_hue.jpg', augmented_image)

        
        
        
        
        
