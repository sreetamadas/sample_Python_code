# -*- coding: utf-8 -*-
"""

### CHANGES COMMON TO MULTI-CLASS CLASSIFICATION & MULTI-LABEL CLASSIFICATION:
    
    NOTE:
    The following code with both multi-class & multi-label did not run, but is useful to understand the differences.

    
    # training
1. input separate train & val sets (as in localization with yolo), instead of getting fastai to do the split.
    This will enable selective augmentation of the train set but not val set, in case of data imbalance
    - we need to decouple data creation & make it common for both classification techniques, & localization

2. script to segregate images into different folders by class. 
   Alternately, provision to read image classes from a file, instead of moving to folders. 
   Fastai uses something called datablock (instead of imagedatabunch) for this 
   https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-planet.ipynb ; 
   more detailed explanations here https://github.com/hiromis/notes/blob/master/Lesson3.md ; 
   Reading from excel will also be useful for multi-label classification
   
   input the filenames & label mappings through a csv file (this has the benefit of passing multiple labels 
   in case of multi-label classification)
   
3. separate code snippet to create csv with train, val, test splits; classnames & filenames

4. explicitly defined the transforms being passed in get_transforms -
   (default: left-right flips, rotation, brightness & contrast changes, crop-pad;
    we have not tried cutouts/ random erasing),  https://docs.fast.ai/vision.augment
   
   
5. link to multiple architectures; provide path to pre-trained weights? (default loc: C:\Users\name\.cache\torch\checkpoints\) 

6. usage of class weights (is easier for multi-class classification; 
                           pose problems for multi-label classification)

7. Selection of best weights instead of final epoch weights (through callback)

8. Precision, recall, accuracy – set to 2 decimals

9. Extract training progress info – epoch no, validation metrics, loss - currently, the
   callback_fns=[partial(CSVLogger)] saves the training progress as "history.csv" in data_path.
   we can extract the metrics from that file
   
10. (To Do) decouple data_analysis module from dataloader.py & make it general to both the classification techniques,
    & localization. currently, it works on train/val split created by fastai
    
11. random number seed assignment is being done both through a function, & the codes placed outside a function
    - need to remove one
    
12. the metrics & other terms (patience etc) being passed in the callbacks - need to check if some of these should be 
   passed as variable names instead of being hardcoded
   
13. Use of learning rates https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb .
    - can selection of optimal learning rate be automated?

14. There are different ways to save models from callback – either the best one, or every epoch. 
  Links on callback: https://medium.com/@lessw/fastais-callbacks-for-better-cnn-training-meet-savemodelcallback-e55f254f1af5 ,
  https://towardsdatascience.com/customize-your-training-loop-with-callbacks-9d93b415a602 ,
  https://forums.fast.ai/t/how-to-save-the-best-model-during-training-in-fastv1/28187,
  documentation link for fastai v1 https://fastai1.fast.ai/
  
15. Model hyperparameters for classification: batch size, validation percentage.
    Need to see what other hyperparameters we can tune  https://fastai1.fast.ai/train.html
    
16. check what are the pros & cons of saving the model weights in .pkl vs .h5 format
   
    

  # testing / inferencing - Following are to be done
1. Class names are hard-coded - generalize
2. Resize input images during inferencing 
3. Add model explainability during model inference



### CHANGES SPECIFIC TO MULTI-LABEL CLASSIFICATION:

    # training
1. (in train.py) metrics for multilabel classification will be different from multiclass classification
2. Sending multiple input labels for each image : input the filenames & label mappings through a csv file (this has the
   benefit of passing multiple labels in case of multi-label classification)  

"""

##########################################

#%%  # data/dataloader.py

from fastai.vision import ImageDataBunch,cnn_learner,Image,image
#from fastai.imports import *
from fastai.vision import *
from torchvision.models import mobilenet_v2,resnet50
#from common import prop as cfg
import numpy as np
import os
import pandas as pd
    

def data_creation(data_path):
    # List all files in directory
    files = os.listdir(data_path)

    train_size,test_size=0.8,0.2
    train_files=[]

    while(len(train_files)!=round(len(files)*train_size)):
        index = random.randrange(0, len(files))
        if files[index] not in train_files:
            train_files.append(files[index])
        
    #Get the remaining files
    test_files = [x for x in files if x not in train_files]

    train_path=os.path.join(data_path,"train")
    test_path=os.path.join(data_path,"test")

    if not os.path.exists(train_path):
        print("hey")
       	os.makedirs(train_path)
        for file in train_files:
            data_path = data_path+"/"
            shutil.move(data_path+file,train_path)

        #'''
        # the following segment creates different folders for different classes
        # this section should be replaced with logic for generating csv with filename,  in appropiate path
        # cols in csv file: classname & val (True/false) [val indicates whether part of val set or not]
        for file in os.listdir(train_path):
            directory_name=file[0:2]
            #directory_path=train_path+"/"+directory_name
            directory_path=os.path.join(train_path,directory_name)
            #to check if directory exists
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            if os.path.exists(directory_path):
                #file_path=train_path+"/"+file
                file_path=os.path.join(train_path,file)
                shutil.move(file_path,directory_path)
        #'''
        
        '''
        # generating csv file for splitting into train + val
        
        df = []
        for file in os.listdir(train_path):
            #directory_name=file[0:2]
            #print(file)
            #class_name = file.split('_')[0] # for multiclass
            class_name = ",".join(file.split('_')[0].split("-")) # for multilabel
            df.append([file,class_name])
        df = pd.DataFrame(df,columns=['filename','class'])
        #df['Validation'] = np.random.choice(['True' * 2 +'False' * 8], size=len(df))
        #for i in range(0,len(df)):
        #    df.iloc[i,'Validation'] = label_random()
        x = np.zeros(len(df))
        ind=[i for i in range(len(df)) if i<=int(0.2*len(df))] # 20% val set
        x[ind]=1
        #print(x)
        random.shuffle(x)
        df['Validation'] = x
        df['Validation'] = [True if x == 1 else False for x in df['Validation']]
        #print(df)
        outfile = data_path + 'training2.csv'
        print(outfile)
        df.to_csv(outfile,index=False)
        '''

    if not os.path.exists(test_path):
        os.makedirs(test_path)
        for file in test_files:
            shutil.move(data_path+file,test_path)
    return train_path

########################################
'''
# for general % splits of train & test

def data_creation1(data_path,train,test):
    # List all files in directory
    files = os.listdir(data_path)

    train_size,test_size=train,test
    train_files=[]

    while(len(train_files)!=round(len(files)*train_size)):
        index = random.randrange(0, len(files))
        if files[index] not in train_files:
            train_files.append(files[index])
        
    #Get the remaining files
    test_files = [x for x in files if x not in train_files]

    train_path=os.path.join(data_path,"train")
    test_path=os.path.join(data_path,"test")

    if not os.path.exists(train_path):
        print("hey")
       	os.makedirs(train_path)
        for file in train_files:
            data_path = data_path+"/"
            shutil.move(data_path+file,train_path)

        for file in os.listdir(train_path):
            directory_name=file[0:2]
            #directory_path=train_path+"/"+directory_name
            directory_path=os.path.join(train_path,directory_name)
            #to check if directory exists
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            if os.path.exists(directory_path):
                #file_path=train_path+"/"+file
                file_path=os.path.join(train_path,file)
                shutil.move(file_path,directory_path)

    if not os.path.exists(test_path):
        os.makedirs(test_path)
        for file in test_files:
            shutil.move(data_path+file,test_path)
    return train_path,test_path
'''
#############################################

def random_seed(seed_value, use_cuda):  
    np.random.seed(seed_value) 
    torch.manual_seed(seed_value) 
    random.seed(seed_value) 
    torch.backends.cudnn.deterministic = True
    #if use_cuda: torch.cuda.manual_seed_all(seed_value) 
    if use_cuda: 
        torch.cuda.manual_seed_all(seed_value) 
        torch.cuda.manual_seed(seed_value)
  
    
def load_model(data_path,hyp):
    model_type=hyp['model_type']
    valid_percentage=hyp['valid_percentage']
    batch_size=hyp['batch_size']
    image_size=NEU['image_size']  # cfg.NEU['image_size']
    number_workers=NEU['number_workers']  #cfg.NEU['number_workers']
    seed=NEU['seed'] #cfg.NEU['seed'] ***
    print("batch_size",batch_size)
    image_path = Path(data_path)
    print("Image Path",image_path)
    # data_read = ImageDataBunch.from_folder(image_path, 
    #                           valid_pct=valid_percentage,
    #                           ds_tfms=get_transforms(), 
    #                           size=image_size, 
    #                           bs=batch_size, 
    #                           num_workers=number_workers,
    #                           seed=seed).normalize(imagenet_stats)
    
    ## explicitly define the data augmentations to be applied
    ## get transforms returns a tuple of two lists of transforms:
    ## one for the training set and one for the validation set
    tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, 
                          max_zoom=1.05, max_lighting=0.1, max_warp=0.)
    ## for multi-label classification, each image can have mutiple labels
    ## put file -> class mapping in a csv
    ## also, create separate train & val sets at the start, instead of via fastai
    ## so that only train data can be selectively augmented to increase dataset size

    if model_type == 'multiclass':
        src = (ImageList.from_csv(data_path, 'training.csv', folder='multilabel_img_train', suffix='.jpg')
               # image names are listed in a csv file (names found by default in first column)
               # path to the data is `path` , the folder containing the data is `folder`
               # if the file ext is missing from the names in the csv, add `suffix` to them.
               #.split_by_rand_pct(0.25) # #.splitting train data randomly into train+val
               .split_from_df(col='Validation')
               .label_from_df())
    elif model_type == 'multilabel':
        src = (ImageList.from_csv(data_path, 'training.csv', folder='multilabel_img_train', suffix='.jpg')
               # image names are listed in a csv file (names found by default in first column)
               # path to the data is `path` , the folder containing the data is `folder`
               # if the file ext is missing from the names in the csv, add `suffix` to them.
               #.split_by_rand_pct(0.25) # #.splitting train data randomly into train+val
               .split_from_df(col='Validation')
               .label_from_df(label_delim=',')) 
    
    data_read = (src.transform(tfms,size=image_size)
                 .databunch(num_workers=number_workers, bs=batch_size)
                 .normalize(imagenet_stats))
    print("Data Read",data_read)
    return data_read
     
                          
def data_analysis(data):
    print("Classes Names in the data:",data.classes)
    print("Length of Classes in the data:",data.c) 
    print("Training Samples size:",len(data.train_ds))
    print("Validation Samples size:",len(data.valid_ds))
    #value count in each class
    value_count_train = pd.value_counts(data.train_ds.y.items, sort =False)
    value_count_train.index = data.classes
    value_count_valid = pd.value_counts(data.valid_ds.y.items, sort =False)
    value_count_valid.index = data.classes
    print("No of Samples in each Category Train and Valid:",value_count_train,value_count_valid)
  
def show_batch_image(data):
    data.show_batch(rows=3, figsize=(10,10))
    
# Need to do few changes in class weight calculation in the line: length_files= len(os.listdir(os.path.join(dataPath,f)))
# def cal_classWeight(dataPath):   
#     # dataPath= r"C:\Users\Desktop\Data\Augmented_Images"
#     files= os.listdir(dataPath)
#     Class_weights= []
#     for f in files:
#         length_files= len(os.listdir(os.path.join(dataPath,f)))   # Need to correct the code here
#         print(length_files)
#         weights= 1/length_files
#         Class_weights.append(weights)
#         class_weights= torch.FloatTensor(Class_weights).cuda()    
#     return class_weights
     
############################################

#%%  # model/train.py

from fastai.vision import cnn_learner
from fastai.vision import *
from fastai.callbacks import *
import torch
from torchvision.models import mobilenet_v2,resnet18, resnet34 #resnet50,vgg16
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
#from common import prop as cfg


        
def train(data,model,hyp):
    ''' Training the Model'''
    # Initialize cnn_learner object with the right model and its pretrained weights
    
    model_type=hyp['model_type']
    ## the following are metrics for multi-class classification; we can implement as if condition
    # *** inputs to cnn_learner will change for multi-class vs multi-label classification
    if model_type == 'multiclass':
        precision = Precision()
        recall = Recall()
        metrics = [accuracy,precision,recall]
    elif model_type == 'multilabel':
        acc_02 = partial(accuracy_thresh, thresh=0.2)
        f_score = partial(fbeta, thresh=0.2)
        metrics=[acc_02, f_score]
        
    if model == 'mobilenet_v2':
        model_run = mobilenet_v2
        learn = cnn_learner(data, model_run, pretrained=True, metrics=metrics, cut=-1,callback_fns=[partial(CSVLogger)])   
        learn.loss_func = torch.nn.functional.cross_entropy
    elif model == 'resnet18' :
        model_run = models.resnet18
        learn = cnn_learner(data, model_run, pretrained=True, metrics=metrics,callback_fns=[partial(CSVLogger)])        
    elif model == 'resnet34' :
        model_run = models.resnet34
        learn = cnn_learner(data, model_run, pretrained=True, metrics=metrics,callback_fns=[partial(CSVLogger)])        
    else:
        pass
    

    #learn.loss_func = torch.nn.functional.cross_entropy
    #learn.loss_func = nn.CrossEntropyLoss(weight= class_weights)
    if model_type == 'multiclass':
        callbacks = [EarlyStoppingCallback(learn, monitor='accuracy', min_delta=0.01, patience=3),
                     SaveModelCallback(learn, every='improvement', monitor='valid_loss', name= str(model))]
    elif model_type == 'multilabel':
        callbacks = [EarlyStoppingCallback(learn, monitor='accuracy_thresh', min_delta=0.01, patience=3),
                     SaveModelCallback(learn, every='improvement', monitor='valid_loss', name=str(model))]

    learn.callbacks = callbacks
    learn.fit_one_cycle(hyp['epoch'])
    #learn.recorder.plot_losses()
    
    name_w=str(model)+".pkl"
    learn.export(NEU['export_path']+"/"+str(model)+".pkl") # learn.export(cfg.NEU['export_path']+"/"+str(model)+".pkl")
    
    ## the following are metrics for multi-class classification; we can implement as if condition
    if model_type == 'multiclass':
        # Get the predictions for validation set and store as numpy array
        val_pred = learn.get_preds()[0].argmax(dim=-1).numpy()
        val_actual = learn.get_preds()[1].numpy()     
        
        # Get the predictions for training set and store as numpy array
        train_pred = learn.get_preds(ds_type=DatasetType.Train)[0].argmax(dim=-1).numpy()
        train_actual = learn.get_preds(ds_type=DatasetType.Train)[1].numpy()
        
        # Calculate accuracy_score and append to Dataframe
        val_acc = accuracy_score(val_actual, val_pred)
        train_acc = accuracy_score(train_actual, train_pred) 
        #Confusion Metric
        Confusion_Metric_Training=confusion_matrix(train_actual,train_pred)
        Confusion_Metric_Validation=confusion_matrix(val_actual,val_pred)
        return learn,val_acc,train_acc,Confusion_Metric_Training,Confusion_Metric_Validation,name_w

    # the following are the metrics for multilabel classification
    elif model_type == 'multilabel':
        # preds on val set
        val_pred = learn.get_preds()[0].numpy()
        val_actual = learn.get_preds()[1].numpy()
        #print('val_pred');    print(val_pred);    print('val_actual');    print(val_actual)
        thresh = 0.5  # thres has been set to 0.2 earlier - CHECK
        # to calculate accuracy_thresh & fbeta, convert the prediction probabilities to classes using thresh
        # then convert the actuals & preds to tensors
        ##labelled_preds = [','.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in val_pred]
        labelled_preds = []
        for lst in val_pred:
            lst = [1 if x > thresh else 0 for x in lst]
            labelled_preds.append(lst)
        #print('labeled preds');        print(labelled_preds)
        actual = tensor(val_actual)
        pred = tensor(labelled_preds)
        #print('acc_thres')
        #acc_thres = accuracy_thresh(val_pred, val_actual, thresh=thresh, sigmoid=False)
        acc_thres = accuracy_thresh(pred, actual, thresh=thresh, sigmoid=False)
        #print(str(acc_thres)) # this prints as tensor(value); printing as string does not resolve
        fbeta_sklearn = fbeta_score(actual, pred, beta=2, average = 'samples')
        #print('fbeta')
        #print(fbeta_sklearn) 
        return learn,acc_thres,fbeta_sklearn,name_w
    

#%%  model/visualization.py

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

def plot_loss(model):
    return model.recorder.plot_losses()

def confusion_metric():
    Confusion_Matrix=confusion_matrix(Actual,Prediction)
    return Confusion_Matrix
    
def classification_report(Actual,Prediction,class_names):
     classification_report(Actual, Prediction, target_names=class_names)
     return classification_report 
    

def plot_confusion_matrix(cm,class_names,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')


#%%
# resources/library.py

#from common import prop as cfg
#from data.dataloader import random_seed,load_model,data_analysis,show_batch_image,data_creation
#from model.train import *
#from model.test import test
#from model.visualisation import plot_loss
    
def modelbuilding(hyp):
    """Builds model, loads data, trains and evaluates"""
    data_path = NEU['data_path'] #cfg.NEU['data_path'] ***
    data_path = data_path+"/"
    print(data_path)
    #train_path=data_creation(data_path) #*** this can be skipped since we are giving only train path, through csv
    
    SEED = hyp['seed'] #1
    Cuda = torch.cuda.is_available()
    print("Is CUDA Available??", Cuda)
    random_seed(SEED,Cuda) #*** why the next 7 lines, since we are already setting through the function random_seed?
    #random_seed(0,True) #*** why the next 7 lines, since we are already setting through the function random_seed?
    '''
    # For reproducibility
    #torch.manual_seed(SEED)
    #if Cuda:
    #    torch.cuda.manual_seed(SEED)
    CUDA_LAUNCH_BLOCKING=1  
    '''
    train_path= data_path 
    data=load_model(train_path,hyp) #*** give data_path here
    #Analysis = data_analysis(data) #*** this works only for train+val splits done through fastai
    #show_batch_image(data)
    model=hyp["classifer_model"]
    print("model",model)
    
    model_type=hyp['model_type']
    # the following is for multi-class classification
    if model_type == 'multiclass':
        learn,Validation_Accuracy,Training_Accuracy,Confusion_Metric_Train,Confusion_Metric_Validation,name=train(data,model,hyp)
        print("Validation Accuracy:",Validation_Accuracy)
        print("Training Accuracy",Training_Accuracy)
        print("Confusion Metric Train",Confusion_Metric_Train)
        print("Confusion Metric Validation",Confusion_Metric_Validation)
        return learn,Validation_Accuracy,Confusion_Metric_Validation,name #Training_Accuracy,Confusion_Metric_Train,
    
    # the following is for multilabel classification
    elif model_type == 'multilabel':
        learn,acc_thres,fbeta_sklearn,name=train(data,model,hyp)
        acc_thres = str(acc_thres).split(')')[0].split('(')[1]
        print("Validation Accuracy:",acc_thres)
        print("fbeta:",fbeta_sklearn)
        return learn,acc_thres,fbeta_sklearn,name
    #plot_loss(learn)
    
def testing():
    #Testing the model
    # the following code is for multi-class classification
    class_names = ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc']
    #class_path=cfg.NEU['class_names']
    #class_names=[]
    #for file in os.listdir(class_path):
    #    class_names.append(file)
    print(class_names)
    Test_Results=test(class_names)
    print("Testing Accuracy:",Accuracy)
    print("Classification_Report",Classification_Report)
    print("Confusion Matrix:",Confusion_Matrix)
    return Test_Results

wdir = 'C:/Users/Desktop/data/classification/training/app'
#'C:/Users/DAR9KOR/Desktop/data/sample_datasets/defect_detection/2_codes/mvp2.0_codes/MODULAR/Classification_Pipeline/training/app/'
#wdir = os.getcwd()
data_path=os.path.join(wdir, 'data','images')  # ***
test_path=os.path.join(wdir, 'data','images','test')  # ***
export_path=os.path.join(wdir, 'weights')  # ***

hyp ={    # ***
      "data_path":data_path,  # ***
      "test_path":test_path,  # ***
      "export_path":export_path,  # ***
       "batch_size":4,  # ***
       "valid_percentage":0.35,   # ***
       "image_size":512,   # ***
       "number_workers":1,   # ***
       "seed":9,   # ***
       "epoch":1,   # ***
        "classifer_model":'mobilenet_v2', # 'resnet50' 'vgg16'      # ***
        "classifier_name":'mobilenet_v2',  #classifier name  # ***
        "model_type":'multilabel' # 'multiclass'
    }

NEU={    # ***
      "data_path":data_path,  # ***
      "test_path":test_path,  # ***
      "export_path":export_path,  # ***
       "batch_size":4,  # ***
       "valid_percentage":0.35,   # ***
       "image_size":512,   # ***
       "number_workers":1,   # ***
       "seed":9,   # ***
       "epoch":1,   # ***
        "classifer_model":'mobilenet_v2', # 'resnet18' 'resnet34'   'mobilenet_v2'   # ***
        "classifier_name":'mobilenet_v2',  #classifier name  # ***
        "model_type":'multilabel'
    }

if __name__ == '__main__':
    #train_run()
    #test_run()
    modelbuilding(hyp)


