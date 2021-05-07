# check image size & type (channels)

import pathlib
from PIL import Image
import pandas as pd

def get_image_spec(im_paths):
    path_sorted = sorted([x for x in im_paths])
    all_mode = []
    
    for idx in range(len(path_sorted)):
        im_path = path_sorted[idx]
        img = Image.open(im_path)
        width, height = img.size
        m = img.mode
        #all_size[str(width) + '_' + str(height)] = ''  # saving height width as keys of dict
        all_mode.append([im_path,m, width, height])
        
    all_mode = pd.DataFrame(all_mode, columns=['path','mode', 'width','height'])
    return all_mode


im_paths = pathlib.Path('/mnt/tensorflow/workspace2/training_demo/images/').glob('*/*.jpg')  #pathlib.Path('./ml/').glob('*/*/*')
image_data = get_image_spec(im_paths)
print(image_data.shape)
print("")


# create a df with unique width & height,
df = image_data.drop_duplicates(['width','height'])
print("unique width, height")
print(df)


# create df with unique modes
print("")
print(image_data.drop_duplicates(['mode']))
