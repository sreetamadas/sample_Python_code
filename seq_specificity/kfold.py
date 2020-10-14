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

from sklearn.model_selection import KFold, cross_validate, cross_val_score


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





################################################################################################
############# TRY PARAMETER OPTIMISATIONS ####################

## do 5-fold CV
## inside cross-validation, split into train, validation & test indices
## do calc on train & val to get optimal params
#     try :
#     balance with smote / balanced class weight in SVM
#     grid search + CV  -> show plots on train & val error
#     optimise prob thres



#####################################################################################
### run grid search ###
# https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65

from sklearn.metrics import make_scorer,recall_score,accuracy_score,precision_score   #roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold   # train_test_split,

clf = SVC(class_weight='balanced') #n_jobs=-1)  # this runs the computation in parallel

param_grid = [
  #{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50], 'gamma': [0.01, 0.05, 0.1, 0.5], 'kernel': ['rbf']},
 ]

# C: penalty parameter, which represents misclassification or error term. 
# The misclassification or error term tells the SVM optimization how much error is bearable.
# This is how you can control the trade-off between decision boundary and misclassification term.
# A smaller value of C creates a simpler decision function at the cost of training accuracy
## creates a small-margin hyperplane and a larger value of C creates a larger-margin hyperplane

# A lower value of Gamma will loosely fit the training dataset (cannot capture the complexity or “shape” of the data.),
# whereas a higher value of gamma will exactly fit the training dataset, which causes over-fitting
# If gamma is too large, the radius of the area of influence of the support vectors only includes the support vector
# itself and no amount of regularization with C will be able to prevent overfitting'''

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
###

grid_search_clf, results = grid_search_wrapper(refit_score='recall_score')
#grid_search_clf

results = pd.DataFrame(grid_search_clf.cv_results_)
results = results.sort_values(by='mean_test_recall_score', ascending=False)
results.columns
#results
#results[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score']].round(3)#.head() 
# 'param_max_features', 'param_min_samples_split', 'param_n_estimators'
#results.round(3)
results[['mean_test_accuracy_score', 'mean_test_precision_score', 'mean_test_recall_score',
         'param_C', 'param_gamma', 'param_kernel','rank_test_recall_score',
       'std_test_accuracy_score', 'std_test_precision_score', 'std_test_recall_score']].round(3)

#######################################################################################

#### random search 
# https://www.mikulskibartosz.name/xgboost-hyperparameter-tuning-in-python-using-grid-search/ 

clf = xgb(objective = 'binary:logistic', scale_pos_weight=1)  # this runs the computation in parallel

param_grid = {"n_estimators": [1, 4, 8, 16, 64, 100, 500],
              "learning rate": [0.1, 0.05, 0.01],
              "max_depth": [2, 8, 16, 64],
               "subsample": [0.8, 0.9, 1], 
              "colsample_bytree": [0.8, 1],
              "gamma": [0,1,5]
              }  

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
}

def random_search_wrapper(refit_score='recall_score'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    skf = StratifiedKFold(n_splits = 4)
    random_search_clf = RandomizedSearchCV(clf, param_grid, scoring=scorers, refit=refit_score,
                           cv=skf, return_train_score=True, n_jobs=-1)
    random_search_clf.fit(x_tr, y_tr.values)  # X_train.values
    print('grid done!')

    print('Best params for {}'.format(refit_score))
    print(random_search_clf.best_params_)
    
    results = pd.DataFrame(random_search_clf.cv_results_)
    #print(results)
    results = results[['mean_train_recall_score','mean_test_recall_score','mean_train_accuracy_score','mean_test_accuracy_score','mean_train_precision_score','mean_test_precision_score']] 
    #print(results.T)
    
    # make the predictions
    y_prd = random_search_clf.predict(x_ts)

    # confusion matrix on the test data.
    print('\nConfusion matrix of XGB optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_ts, y_prd),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    print("classification report", classification_report(y_ts, y_prd))
    return random_search_clf, results
  
  
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size =0.3, stratify = y, random_state = 42)
random_search_clf, results = random_search_wrapper(refit_score='recall_score')

## sort by different metrics & plot ##
results = results.sort_values(by='mean_train_accuracy_score', ascending=False)
# dataframe & plots
fig, ax2 = plt.subplots(1, 1)
#ax2.hold(True)
ax2.plot(results['mean_train_accuracy_score'],label='Traning set')
ax2.plot(results['mean_test_accuracy_score'],label='Validation set')
#ax2.set_title("Training and Validation precision")
ax2.set_ylabel("accuracy")
ax2.set_xlabel("Iterations")
ax2.set_xlim([0,20])
ax2.legend(fancybox=True)

#model_xgb = xgb(objective = 'binary:logistic', subsample = 0.9, n_estimators= 100, max_depth= 64, learning_rate =0.05, gamma = 1, colsample_bytree = 1, random_state = 42)
#model_xgb.fit(x_train, y_train)
#p =model_xgb.predict(x_test)
#print('f1_score:', f1_score(y_test,p, average = 'weighted'))
#print('accuracy:', accuracy_score(y_test, p))
#print(classification_report(y_test, p))


  





