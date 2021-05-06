#!/usr/bin/env python
# coding: utf-8

# In[1]:


## sleep stage classification - 2 class - using SAX features

## feature set variables: window size (0.5, 0.2, 0.1, 0.05), alphabet size (3, 4, 5), word size (dimer, trimer)

## do 10-fold CV
## inside cross-validation, split into train, validation & test indices
## do calc on train & val to get optimal params
#     try :
#     balance with smote / balanced class weight in SVM
#     grid search + CV  -> show plots on train & val error
#     optimise prob thres

## with the optimal params, re-train using train & val, then predict on test set
## merge test set predictions from all folds into a single set - obtain esnsitivity, specificity & conf mat on this set

import pandas as pd
import numpy as np
import os
import pickle
from scipy import stats
#import itertools
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.preprocessing import StandardScaler #, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

import itertools
from sklearn.svm import SVC
from sklearn.metrics import auc,accuracy_score
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
from sklearn.metrics import classification_report


# In[2]:


###################function to plot confusion matrix#####################################
def plot_confusion_matrix(cm, classes,
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
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
###########################################################################################


# In[3]:


## create train & test sets by ID
def CreateTrainTestID(dfull):
    
    ## select IDs
    ID = dfull['pID'].unique() # create ID list
    #np.random.seed(456)   ## select the IDs for the 2 sets
    
    '''
    msk = np.random.rand(len(ID)) < 0.8
    #print('msk')
    #print(msk)
    trainIDs = pd.DataFrame()
    testIDs = pd.DataFrame()
    trainIDs['pID'] = ID[msk]
    testIDs['pID'] = ID[~msk]
    #print(ID[msk])
    #print(ID[~msk])
    print(trainIDs['pID'].tolist())
    print(testIDs['pID'].tolist())
    Train = dfull.loc[dfull['pID'].isin(ID[msk])] # create the train-test sets
    Test = dfull.loc[dfull['pID'].isin(ID[~msk])] #'''
    
    index_list = list(range(len(ID)))
    #print('index list')
    #print(index_list)
    train_index = np.random.choice(np.array(index_list), size = round(0.8*(len(index_list))), replace=False)
    test_index = [i for i in index_list if i not in train_index]
    Train = dfull.loc[dfull['pID'].isin(ID[train_index])]
    Test = dfull.loc[dfull['pID'].isin(ID[test_index])]
    #print('train')
    print(train_index)
    #print('test')
    print(test_index)
        
    return Train, Test

def createXYsets(Train,Test):    
    ## create the sets
    X_train1 = Train.drop(['pID','day','epoch','class_label','class2'], 1).reset_index() #inplace=True, drop=True)  ## create X & y splits
    X_train1 = X_train1.drop(['index'], 1)
    X_test = Test.drop(['pID','day','epoch','class_label','class2'], 1).reset_index() #inplace=True, drop=True)
    X_test = X_test.drop(['index'],1)
    y_train1 = Train['class2'].reset_index() #inplace=True, drop=True)
    y_train1 = y_train1.drop(['index'],1)
    y_test = Test['class2'].reset_index() #inplace=True, drop=True)
    y_test = y_test.drop(['index'],1)
    print(X_train1.head(1))
    return X_train1, y_train1, X_test, y_test


# In[28]:


#for i in range(10):
#    print('')    
#    print("Iteration: " + str(i))
#    
#    Train, Test = CreateTrainTestID(dfull)
    #X_train1, y_train1, X_test, y_test = createXYsets(Train,Test)
#    print('')


# In[ ]:





# In[4]:


### load data ###

file_location='C:/Users/DAR9KOR/Desktop/data/HEALTHCARE/EEG/data/eeg_epochs_dec12_2018/raw_data' #C:\\Users\\DAR9KOR\\Desktop\\data\\HEALTHCARE\\EEG\\data\\eeg_epochs_dec12_2018'
os.chdir('C:/Users/DAR9KOR/Desktop/data/HEALTHCARE/EEG/data/eeg_epochs_dec12_2018/raw_data') #'/home/intern_eyecare/Desktop/EEG/raw_data') #C:\\Users\\DAR9KOR\\Desktop\\data\\HEALTHCARE\\EEG\\data\\eeg_epochs_dec12_2018')

# remove 2nd night of subject 13, as there was data loss - pkl file created accordingly
dfull = pickle.load(open('Frac_sax_dimer_0.1s_EEG_all_patients.pkl', 'rb'))
dfull.head(5)


# In[ ]:


### check the data ###


# In[6]:


# 1. look for null values
dfull.info()


# In[7]:


dfull.isnull().sum()


# In[8]:


# 2. check data distribution
dfull.groupby('class_label').size()


# In[9]:


dfull.groupby(['pID','class_label']).size()


# In[5]:


### convert from 3-class to 2-class ###
dfull['class2'] = dfull['class_label']
dfull['class2'] = [0 if x == 'wake' else 1 for x in dfull['class2']]

## convert selected columns to string/categories 
## DO NOT CONVERT 'class2' from numeric to category - this probably gives error in grid search
dfull[['class_label']] = dfull[['class_label']].astype('str')        # ,'class2'


# In[ ]:





# In[14]:


'''## do multiple iterations
sensitivity = []
specificity = []
accuracy = []
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

for i in range(10):
    print('')    
    print("Iteration: " + str(i))
    
    Train, Test = CreateTrainTestID(dfull)
    X_train1, y_train1, X_test, y_test = createXYsets(Train,Test)
    print('')
    
    ### data scaling ###
    scaler = StandardScaler().fit(X_train1)  # StandardScaler();  MinMaxScaler()
    X_train_scaled = scaler.transform(X_train1)    # fit_transform(X_train1)
    X_test_scaled = scaler.transform(X_test)
    
    ## model fitting
    model_svc = SVC(class_weight='balanced')  # probability=True
    model_svc.fit(X_train_scaled, y_train1)
    
    
    ## predict on test data - check metrics
    y_pred = model_svc.predict(X_test_scaled)
    
    
    #####    Compute confusion matrix     ####
    class_names = ['0','1'] #['wake','sleep_stage_1','sleep_stage_2']  # wake, SS1, SS2  ; # '0','1','2'
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    #print(cnf_matrix)
    # Plot normalized confusion matrix : normalisation shows nan for class'0' no signal has class=0 as true label
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
    plt.show()
    
    ## print metrics
    print(classification_report(y_test, y_pred, target_names=class_names))
    #print('accuracy: ' + str(accuracy_score(y_test,y_pred)))
    p = precision_recall_fscore_support(y_test,y_pred)
    #sen = p[1][1]
    #spec = p[1][0]
    #print('sen: ' + str(sen) +' , spec: ' + str(spec))
    sensitivity.append(p[1][1])
    specificity.append(p[1][0])
    accuracy.append(accuracy_score(y_test,y_pred))
    
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('')

sensitivity = pd.DataFrame(sensitivity)
specificity = pd.DataFrame(specificity)
metrics = pd.concat([sensitivity, specificity], axis=1)
metrics.columns = ['sensitivity', 'specificity']
print(metrics)
print('avg sensitivity: ' + str(metrics.sensitivity.mean()))
print('avg specificity: ' + str(metrics.specificity.mean()))
'''    


# In[ ]:





# In[29]:


## do multiple iterations
sensitivity = []
specificity = []
accuracy = []
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

for i in range(10):
    print('')    
    print("Iteration: " + str(i))
    
    Train, Test = CreateTrainTestID(dfull)
    X_train1, y_train1, X_test, y_test = createXYsets(Train,Test)
    print('')
    
    ### data scaling ###
    scaler = StandardScaler().fit(X_train1)  # StandardScaler();  MinMaxScaler()
    X_train_scaled = scaler.transform(X_train1)    # fit_transform(X_train1)
    X_test_scaled = scaler.transform(X_test)
    
    ## model fitting
    model_svc = SVC(class_weight='balanced')  # probability=True
    model_svc.fit(X_train_scaled, y_train1)
    
    
    ## predict on test data - check metrics
    y_pred = model_svc.predict(X_test_scaled)
    
    
    #####    Compute confusion matrix     ####
    class_names = ['0','1'] #['wake','sleep_stage_1','sleep_stage_2']  # wake, SS1, SS2  ; # '0','1','2'
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    #print(cnf_matrix)
    # Plot normalized confusion matrix : normalisation shows nan for class'0' no signal has class=0 as true label
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
    plt.show()
    
    ## print metrics
    print(classification_report(y_test, y_pred, target_names=class_names))
    #print('accuracy: ' + str(accuracy_score(y_test,y_pred)))
    p = precision_recall_fscore_support(y_test,y_pred)
    #sen = p[1][1]
    #spec = p[1][0]
    #print('sen: ' + str(sen) +' , spec: ' + str(spec))
    sensitivity.append(p[1][1])
    specificity.append(p[1][0])
    accuracy.append(accuracy_score(y_test,y_pred))
    
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('')

sensitivity = pd.DataFrame(sensitivity)
specificity = pd.DataFrame(specificity)
accuracy = pd.DataFrame(accuracy)
metrics = pd.concat([sensitivity, specificity, accuracy], axis=1)
metrics.columns = ['sensitivity', 'specificity', 'accuracy']
print(metrics)
print('avg sensitivity: ' + str(metrics.sensitivity.mean()))
print('avg specificity: ' + str(metrics.specificity.mean()))
print('avg accuracy: ' + str(metrics.accuracy.mean()))    


# In[ ]:





# In[7]:


from sklearn.model_selection import KFold, cross_validate, cross_val_score
pid_tr = dfull['pID'].unique() 
K = 10
kf = KFold(n_splits=K,shuffle=True,random_state=42)
dn = dfull
sensitivity = []
specificity = []
accuracy = []
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


for train_index, test_index in kf.split(pid_tr):
    print("TRAIN:", pid_tr[train_index], "TEST:", pid_tr[test_index])
    
    df_test = pd.DataFrame()
    df_train = pd.DataFrame()
    for pid in train_index:
        df_tr = dn[(dn['pID']==pid_tr[pid])]
        df_train = pd.concat([df_train,df_tr])
    for pid in test_index:
        df_ts = dn[(dn['pID']==pid_tr[pid])]
        df_test = pd.concat([df_test,df_ts], ignore_index = True)
    
    X_train = df_train.drop([   'pID',  'epoch','class_label', 'class2'],1).reset_index()
    X_train = X_train.drop(['index'],1)
        
    y_train = df_train['class2'].reset_index()
    y_train = y_train.drop(['index'],1)
    
    X_test = df_test.drop([   'pID',  'epoch','class_label', 'class2'],1).reset_index()
    X_test = X_test.drop(['index'],1)
    
    y_test = df_test[['class2']].reset_index()
    y_test = y_test.drop(['index'],1)
    
    ### data scaling ###
    scaler = StandardScaler().fit(X_train)  # StandardScaler();  MinMaxScaler()
    X_train_scaled = scaler.transform(X_train)    # fit_transform(X_train1)
    X_test_scaled = scaler.transform(X_test)
    
    ## model fitting
    model_svc = SVC(class_weight='balanced')  # probability=True
    model_svc.fit(X_train_scaled, y_train)
    
    
    ## predict on test data - check metrics
    y_pred = model_svc.predict(X_test_scaled)
    
    
    #####    Compute confusion matrix     ####
    class_names = ['0','1'] #['wake','sleep_stage_1','sleep_stage_2']  # wake, SS1, SS2  ; # '0','1','2'
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    #print(cnf_matrix)
    # Plot normalized confusion matrix : normalisation shows nan for class'0' no signal has class=0 as true label
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
    plt.show()
    
    ## print metrics
    print(classification_report(y_test, y_pred, target_names=class_names))
    #print('accuracy: ' + str(accuracy_score(y_test,y_pred)))
    p = precision_recall_fscore_support(y_test,y_pred)
    #sen = p[1][1]
    #spec = p[1][0]
    #print('sen: ' + str(sen) +' , spec: ' + str(spec))
    sensitivity.append(p[1][1])
    specificity.append(p[1][0])
    accuracy.append(accuracy_score(y_test,y_pred))
    
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('')

sensitivity = pd.DataFrame(sensitivity)
specificity = pd.DataFrame(specificity)
accuracy = pd.DataFrame(accuracy)
metrics = pd.concat([sensitivity, specificity, accuracy], axis=1)
metrics.columns = ['sensitivity', 'specificity', 'accuracy']
print(metrics)
print('avg sensitivity: ' + str(metrics.sensitivity.mean()))
print('avg specificity: ' + str(metrics.specificity.mean()))
print('avg accuracy: ' + str(metrics.accuracy.mean()))    


# In[ ]:





# In[ ]:


############# TRY PARAMATER OPTIMISATIONS ####################

## do 5-fold CV
## inside cross-validation, split into train, validation & test indices
## do calc on train & val to get optimal params
#     try :
#     balance with smote / balanced class weight in SVM
#     grid search + CV  -> show plots on train & val error
#     optimise prob thres


# In[8]:


#'''
# https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65

from sklearn.metrics import make_scorer,recall_score,accuracy_score,precision_score   #roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold   # train_test_split,

clf = SVC(class_weight='balanced') #n_jobs=-1)  # this runs the computation in parallel

param_grid = [
  #{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50], 'gamma': [0.01, 0.05, 0.1, 0.5], 'kernel': ['rbf']},
 ]


scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
}



def grid_search_wrapper(refit_score='recall_score'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score,
                           cv=skf, return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train.values)  # X_train.values

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)
    
    results = pd.DataFrame(grid_search_clf.cv_results_)
    results = results[['param_C','param_gamma','mean_train_recall_score','mean_test_recall_score','mean_train_accuracy_score','mean_test_accuracy_score','mean_train_precision_score','mean_test_precision_score']] 
   
    
    # make the predictions
    y_pred = grid_search.predict(X_test.values)

    # confusion matrix on the test data.
    print('\nConfusion matrix of SVM optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    return grid_search, results
#'''


# In[9]:


## check a single iteration of train-test; the train is further split into train_val 10 times

Train, Test = CreateTrainTestID(dfull)
X_train, y_train, X_test, y_test = createXYsets(Train,Test)
print('')
    
### data scaling ###
scaler = StandardScaler().fit(X_train)  # StandardScaler();  MinMaxScaler()
X_train_scaled = scaler.transform(X_train)    # fit_transform(X_train1)
X_test_scaled = scaler.transform(X_test)

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print('')

### run grid search ###
grid_search_clf, results = grid_search_wrapper(refit_score='recall_score')

print('')
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# In[ ]:


## sort by different metrics ##
results = results.sort_values(by='mean_test_precision_score', ascending=False)
results


# In[ ]:


## sort by different metrics ##
results = results.sort_values(by='mean_test_recall_score', ascending=False)
results


# In[ ]:


## sort by different metrics ##
results = results.sort_values(by='mean_train_accuracy_score', ascending=False)

# dataframe & plots
fig, ax2 = plt.subplots(1, 1)
ax2.hold(True)
ax2.plot(results['mean_train_accuracy_score'],label='Traning set')
ax2.plot(results['mean_test_accuracy_score'],label='Validation set')
#ax2.set_title("Training and Validation precision")
ax2.set_ylabel("accuracy")
ax2.set_xlabel("Iterations")
ax2.set_xlim([0,20])
ax2.legend(fancybox=True)


# In[ ]:




