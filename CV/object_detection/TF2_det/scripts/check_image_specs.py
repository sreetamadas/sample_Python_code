# check image format

import os
from PIL import Image
import pathlib


# GC10 dataset

im_path = pathlib.Path('C:/Users/Desktop/data/img_02_425501900_00017.jpg')  #.glob('*/images/*.png')
display(Image.open(im_path))

img = Image.open('C:/Users/Desktop/data/img_02_425501900_00017.jpg')
print(img.size) 


img.mode





