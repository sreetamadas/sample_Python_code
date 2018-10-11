#### steps ####
##  https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65  ##########


import pandas as pd
import numpy as np
import os

# set path, get data


#### data pre-processing  ####
# 0. check data - missing or not
df.describe

# 1. clean data - take care of missing values

# 2. check data types
df.dtypes

# check distribution of data in different classes
dfull.groupby('class').agg({'X1': 'count'})

# 3. create new features

# 4. subset only required colunms


# 5. convert factor colunms
df[['class','class2']] = df[['class','class2']].astype('str') 
df[['class','class2']] = df[['class','class2']].astype('category')



# 6. create train-test data : 2 class
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X = df[['X1','X2','X3','X4','X6']]  
y = df[['class2']]   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
###### use "stratify" option for consistent class distribution between training and test sets  ####
##  https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65  ##########


# 7. normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#X_train_scaled.shape


# 8. use PCA, instead of actual features, if too many features (or, do a feature importance map)
pca = decomposition.PCA(n_components=2)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print((pca.explained_variance_ratio_)*100)


######################################################################################
###### model building - with normalized or PCA-transformed data ######
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import auc,accuracy_score
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix

## basic model building ##
# 1. fit model - vary ML algo & hyperparameters
model_svc = SVC()  # kernel='linear'
model_svc.fit(X_train_scaled, y_train)

model_rf = RandomForestClassifier(n_estimators = 501) ## max_depth=5, random_state=0,verbose =0)  
# max_features=2, min_samples_split=4, n_estimators=50, min_samples_leaf=2
model_rf.fit(X_train_scaled, y_train)

# 2. predict
y_pred_svc = model_svc.predict(X_test_scaled)
y_pred_rf = model_rf.predict(X_test_scaled)

# 3. check model performance
from sklearn.metrics import classification_report
#from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
print(classification_report(y_test, y_pred, target_names=class_names))
# also, see the separate code for confusion matrix



#### cross-validation & hyper-parameter tuning  ####
# https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65


####   combine different models  ####
# voting classifier - weighted, unweighted
# adaboost, xgboost, stacking regressor



####  bagging data sets  ####



#################################################################################
### compare results from multiple models ###

# check if RMSE of predictions from 2 models are similar; rmse calculated over multiple iterations
import scipy.stats as stats
def FindRMSE(pred,act):
    rmse = math.sqrt(np.mean((pred - act) ** 2))
    return rmse

stats.ttest_ind(df_pred_model1.rmse, df_pred_model2.rmse, equal_var=True)  ## assuming equal variance"
stats.ttest_ind(df_pred_model1.rmse, df_pred_model2.rmse, equal_var=False)  ## not assuming equal variance"


## kappa statistic confusion matrix  ; Mcnemar's Test P-Value confusion matrix
#http://standardwisdom.com/softwarejournal/2012/01/comparing-two-confusion-matrices
#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
import sklearn
sklearn.metrics.cohen_kappa_score(df_model1.Pred, df_model2.Pred, labels=None, weights=None, sample_weight=None)"


#########################################################################
### pickling models ###
# https://stackoverflow.com/questions/35067957/how-to-read-pickle-file

# save the built model as pickled file
pickle.dump(regr_model, open("C:\\Users\\Desktop\\data\\model_5param.sav",'wb'))
# to save pickle file built with python 3 as python 2 compatible
pickle.dump(regr_model, open("C:\\Users\\Desktop\\data\\model_5param.sav",'wb'), protocol=2)


# load a pre-saved model for predictions on new data
loaded_model = pickle.load(open("C:\\Users\\Desktop\\data\\model_5param.sav", 'rb'))
# or,
filename="C:\\Users\\Desktop\\data\\model_5param.sav"
with open(filename, 'rb') as f:
    x = pickle.load(f)
# or,
x = pd.read_pickle("C:\\Users\\Desktop\\data\\model_5param.sav")




