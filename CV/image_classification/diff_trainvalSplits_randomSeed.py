# auto-updating
%reload_ext autoreload
%autoreload 2
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import glob
import random
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os
import torch
import fastai
import time
from fastai.vision import *
np.random.seed(0) 
from torchvision.models import resnet50
from torchvision.models import mobilenet_v2
from torchvision.models import vgg16
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from fastai import *
# from fastai.dataset import ModelData,ArraysIndexDataset
# from fastai.dataloader import DataLoader
# from fastai.learner import Learner
import torch
import torch.nn as nn
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# Set seed
def random_seed(seed_value, use_cuda):  
    np.random.seed(seed_value) 
    torch.manual_seed(seed_value) 
    random.seed(seed_value) 
    torch.backends.cudnn.deterministic = True
    if use_cuda: torch.cuda.manual_seed_all(seed_value) 

#random_seed(0,False)
random_seed(0,True)


directory = 'path'
root = [img for img in glob.glob(os.path.join(directory)+"\*.bmp")]
filenames = [i for i in os.listdir(directory) if i.endswith(".bmp")]

root.sort() # ADD THIS LINE
images = []

for img in root:
    n= cv2.imread(img, 0)
    #n_299 = cv2.resize(n, (299, 299))   ###Resizing to 299X299 which is input size for InceptionResNet v2 model 
    n_rgb = cv2.cvtColor(n,cv2.COLOR_GRAY2RGB)   ### Converting Grayscale to RGB which the model expects
    images.append(n_rgb)

df = pd.DataFrame({'Filename': filenames, 'Images': images})
df['Class'] = 'NA'
df.loc[(df['Filename'].str.contains('Cr')), 'Class'] = 'Cr'
df.loc[(df['Filename'].str.contains('In')), 'Class'] = 'In'
df.loc[(df['Filename'].str.contains('Pa')), 'Class'] = 'Pa'
df.loc[(df['Filename'].str.contains('PS')), 'Class'] = 'PS'
df.loc[(df['Filename'].str.contains('RS')), 'Class'] = 'RS'
df.loc[(df['Filename'].str.contains('Sc')), 'Class'] = 'Sc'





def preprocess(df, seed, split):
    
    from sklearn.utils import shuffle
    df_shuffle = shuffle(df, random_state = 0)
    
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df_shuffle, test_size=0.2, stratify = df_shuffle['Class'],  random_state = seed)
    train_df, val_df = train_test_split(train_df, test_size=split, stratify = train_df['Class'], random_state = seed)
        
    """ Extracting the image pixel data from each dataframe   """
    x_train, x_test, x_val = list(train_df['Images'].values), list(test_df['Images'].values), list(val_df['Images'].values)
    x_train, x_test, x_val = np.asarray(x_train), np.asarray(x_test), np.asarray(x_val) 
    
    """   Reshaping the data array into correct dimensions for input to model   """
    x_train, x_test, x_val = x_train.reshape(-1, 200,200, 3), x_test.reshape(-1, 200,200, 3), x_val.reshape(-1, 200,200, 3)
    x_train.shape, x_test.shape, x_val.shape
    
    """     Normalizing values between 0 to 1   """
    x_train, x_test, x_val = x_train.astype('float32'), x_test.astype('float32'), x_val.astype('float32')
    x_train, x_test, x_val = x_train / 255. , x_test / 255. , x_val / 255.
    
    y_train, y_test, y_test = train_df['Class'].values, test_df['Class'].values, val_df['Class'].values
    
    """     Label Encoding   """
    le = LabelEncoder()
    y_train, y_test, y_val = le.fit_transform(y_train), le.fit_transform(y_test), le.fit_transform(y_test)
    
    """    Convert Numpy Arrays to Torch Tensors    """
    y_train_torch, y_test_torch, y_val_torch = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long), torch.tensor(y_val, dtype=torch.long)
    x_train_torch, x_test_torch, x_val_torch = torch.from_numpy(x_train), torch.from_numpy(x_test), torch.from_numpy(x_val)
    
    """  Changing the shape/format of the Torch Tensors from NHWC to NCHW    """
    x_train_torch, x_test_torch, x_val_torch = x_train_torch.permute(0, 3, 1, 2), x_test_torch.permute(0, 3, 1, 2), x_val_torch.permute(0, 3, 1, 2)
    
    """    Create Tensor Dataset from Torch Tensors    """
    import torch.utils.data as data_utils

    train_torch = data_utils.TensorDataset(x_train_torch, y_train_torch)
    val_torch = data_utils.TensorDataset(x_val_torch, y_val_torch)
    test_torch = data_utils.TensorDataset(x_train_torch, y_train_torch)
    
    """    Create DataBunch from Tensor Datasets    """
    batch_size = 16
    data = ImageDataBunch.create(train_ds = train_torch, valid_ds = val_torch, test_ds = test_torch, bs=batch_size, num_workers=2)
    data.c = 6
    
    return data
    
    
    
######################## MODEL BUILDING ###################
precision = Precision()
recall = Recall()
metrics = [accuracy,precision,recall]



results1 = pd.DataFrame(columns = ['Val_Acc', 'Train_Acc', 'Train_Val_Split', 'Random_Seed'])



# Generate a list of n random samples (for n random iterations)
n = 5
sample = random.sample(range(0, 50), n)

# List of different train_validation set sizes
split = [0.95, 0.9583, 0.9667, 0.975]

for s in split:
    for i in sample:
        # Call preprocess function and store resulting ImageDataBunch in 'data'
        data = preprocess(df, i, s)

        # Initialize cnn_learner object with the right model and its pretrained weights
        learn = cnn_learner(data, mobilenet_v2, pretrained=True, metrics=metrics, cut=-1)
        # Below is needed since you get negative loss otherwise Ref: https://forums.fast.ai/t/negative-loss-with-letters-mnist/30889/15
        learn.loss_func = torch.nn.functional.cross_entropy

        # Fit one cycle method for a number of epochs
        epcohs = 10
        learn.fit_one_cycle(epcohs)

        # Get the predictions for validation set and store as numpy array
        preds = learn.get_preds()[0].argmax(dim=-1).numpy()
        val_actual = learn.get_preds()[1].numpy()

        # Get the predictions for training set and store as numpy array
        train_pred = learn.get_preds(ds_type=DatasetType.Fix)[0].argmax(dim=-1).numpy()
        train_actual = learn.get_preds(ds_type=DatasetType.Fix)[1].numpy()

        # Calculate accuracy_score and append to Dataframe
        val_acc = accuracy_score(val_actual, preds)
        train_acc = accuracy_score(train_actual, train_pred)
        
        # Temporary Dataframe to store this iteration's values
        temp = pd.DataFrame([[val_acc, train_acc, s, i]], columns = results.columns)
        results1 = results1.append(temp)
    

mean_res1 = results1.groupby(['Train_Val_Split']).mean()
mean_res1


# Since Train_Val_Split of 0.95 is repeated in both results and results1, removing that before appending
final_res_all = results.loc[(results['Train_Val_Split'] != 0.95)].append(results1)
final_res_all

mean_res = results.groupby(['Train_Val_Split']).mean()
mean_res

final_res = mean_res.append(mean_res1)
final_res

final_res = final_res.groupby(['Train_Val_Split']).mean()
final_res


final_res['Val_Acc_StdDev'] = final_res_all.groupby(['Train_Val_Split']).std()['Val_Acc']
final_res = final_res.reset_index()
final_res_all = final_res_all.reset_index()
final_res['Training_Set_size'] = final_res['Train_Val_Split'].apply(lambda x: int(round((1-x)*1440)))
final_res_all['Training_Set_size'] = final_res_all['Train_Val_Split'].apply(lambda x: int(round((1-x)*1440)))



# plots
sns.set(rc={'figure.figsize':(12,10)})
ax = sns.boxplot(x="Training_Set_size", y="Val_Acc", data=final_res_all)
ax = sns.swarmplot(x="Training_Set_size", y="Val_Acc", data=final_res_all, color=".25",  size= 7)
plt.xlabel('Training_Set_Size')
plt.ylabel('Validation_Accuracy')
plt.title('Validation Accuracy Vs Training Set Size for 10 Epochs for 6 Classes')


plt.errorbar(final_res['Training_Set_size'], final_res['Val_Acc'], yerr = final_res['Val_Acc_StdDev'], color='green', fmt='o', 
             linestyle='dashed', linewidth=2)
plt.xlabel('Training_Set_Size')
plt.ylabel('Validation_Accuracy')
plt.title('Validation Accuracy Vs Training Set Size for 10 Epochs for 6 Classes')


    
    



