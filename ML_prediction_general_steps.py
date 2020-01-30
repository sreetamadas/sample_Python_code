####################################################################################################################
####    key steps in using ML for prediction  (skip to end-of-file for additional reading material)    ####

import pandas as pd
import numpy as np
import os

# set path - either through terminal, or by following command
os.chdir('C:\\User\\Desktop\\data\\Input_output')    # "C:/user/Desktop/data/input"    <- both valid


# get data -> save in a dataframe named "df"
df = pd.read_csv(main_file_path)  # from csv
df = pd.read_excel(main_file_path)  # from excel  "/home/user/loc/file.xls"
df.head(n=5)  # show top 5 rows


#####################################################################################
#### data exploration & pre-processing  ####
# 1. check data - missing or not  
df.describe  # summary statistics
df['colname'].value_counts()   # shows if any column has less rows
df.shape   # (no. of rows, no. of cols)

# 1. check data types (in different columns) - 
df.info()
df.dtypes
df.columns  # list of columns in the df 

##### => check other custom commands from pandas (this section is optional)
# https://towardsdatascience.com/pandas-tips-that-will-save-you-hours-of-head-scratching-31d8572218c9
# https://www.youtube.com/watch?v=RlIiVeig3hc   => pandas-profiling
# https://github.com/8080labs/pyforest   (auto imports libraries) 
# https://towardsdatascience.com/python-for-data-science-8-concepts-you-may-have-forgotten-i-did-825966908393
#       above page has links to (arange, map, filter, lambda function, etc.)  
######



# 2. clean data 
# take care of missing values (missing data imputation or removal), outliers
# https://analyticsindiamag.com/5-ways-handle-missing-values-machine-learning-datasets/
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4

# generally, drop columns with many missing values,


# impute columns with less missing values


# drop rows with many missing values


# remove duplicates


# remove col. with 'constant' feature value


# cleaning an object column with mixed data types  (https://lnkd.in/eFNQqVM)
# look at encoding categorical data



# remove multi-collinearity


# explore the data - see the distribution of values in each column (using histogram or boxplot) - are there outliers?
# # https://towardsdatascience.com/5-useful-statistics-data-scientists-need-to-know-5b4ac29a7da9
# outlier data imputation or removal
# https://heartbeat.fritz.ai/how-to-make-your-machine-learning-models-robust-to-outliers-44d404067d07


# convert factor colunms  (use one hot encoding, if reqd, for Y in classification problems)
# convert if column data type is not as expected
df[['class','class2']] = df[['class','class2']].astype('str') 
df[['class','class2']] = df[['class','class2']].astype('category')
# LabelEncoder() ; OneHotEncoder() ; LabelBinarizer() ; to_categorical (keras.utils) - check which to use when
# https://stackoverflow.com/questions/50473381/scikit-learns-labelbinarizer-vs-onehotencoder
# GOOGLE: label binarizer vs to categorical
## some built-in functions ###
# int(obj)	convert object to integer
# str(obj)	convert object to string
# type(obj)	return type of object
# len(obj)	return length of object



# 3. check distribution of data in different classes (for classification problems)/ subgroups
dfull.groupby('class').agg({'X1': 'count'})
dfull.groupby('class').size()


# 4. create new features (optional, if required; may be based on domain knowledge)
# https://medium.com/vickdata/four-feature-types-and-how-to-transform-them-for-machine-learning-8693e1c24e80
# https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219
#### feature aggregation (group by some feature & compute mean, sum, etc.)
## can convert numeric column to categorical by binning


# 5. subset only required columns (optional, if required; may be based on domain knowledge)



# 6. create train-test data : 2 class
# should have separate train, validation & test sets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X = df[['X1','X2','X3','X4','X6']]  
y = df[['class2']]   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
###### use "stratify" option for consistent class distribution between training and test sets  ####
##  https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65  ##########

## handling imbalanced data sets - upsample, downsample, SMOTE & its variants (ADASYN?)
# https://medium.com/coinmonks/smote-and-adasyn-handling-imbalanced-data-set-34f5223e167
# https://towardsdatascience.com/having-an-imbalanced-dataset-here-is-how-you-can-solve-it-1640568947eb
# https://towardsdatascience.com/dealing-with-imbalanced-classes-in-machine-learning-d43d6fa19d2
#  https://machinelearningmastery.com/data-sampling-methods-for-imbalanced-classification/
#  How to Deal with Imbalanced Data using SMOTE   https://medium.com/analytics-vidhya/balance-your-data-using-smote-98e4d79fcddb
# cross-validation   https://medium.com/lumiata/cross-validation-for-imbalanced-datasets-9d203ba47e8

"""
def train_test_split(data,frac=0.7,shuffle=True):   # not the inbuilt train_test_split function
    if shuffle:
        train_data = data.sample(frac=frac, replace=False, random_state=1)
        test_index = list(set(range(len(data)))-set(train_data.index))
    else:
        train_data = data[:int(0.7*len(data))]
        test_data = data[int(0.7*len(data)):]
    train_data = train_data.reset_index(drop=True)
    test_data = data.iloc[test_index,:].reset_index(drop=True)
    print('Shape of train data is : ',train_data.shape)
    print('Shape of test data is : ',test_data.shape)
    return train_data,test_data
"""



# 7. normalize the data; can also use other scalers like MinMaxScaler()
# https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#X_train_scaled.shape


# 8. feature reduction (using PCA) or selection
# use PCA, instead of actual features, if there are too many features (or, do a feature importance map)
# check: t-SNE (compute intensive), SVD    
# https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8
# https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b?gi=cf02ae64f802
# https://towardsdatascience.com/feature-selection-techniques-for-classification-and-python-tips-for-their-application-10c0ddd7918b
# https://towardsdatascience.com/the-5-feature-selection-algorithms-every-data-scientist-need-to-know-3a6b566efd2
from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print((pca.explained_variance_ratio_)*100)
# feature importance can be obtained from
#     randomForest (using MI?),
#     forward or backward selection (see if these are available in python)
#     additionally using ridge or LASSO for regression problems (see L1 vs L2 norm) 



######################################################################################
###### model building - with normalized or PCA-transformed data ######
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import auc,accuracy_score
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
# https://www.kdnuggets.com/2020/01/guide-precision-recall-confusion-matrix.html
# https://medium.com/swlh/recall-precision-f1-roc-auc-and-everything-542aedf322b9
# https://www.analyticsvidhya.com/blog/2019/08/detailed-guide-7-loss-functions-machine-learning-python-code/
# https://www.pyimagesearch.com/2016/09/05/multi-class-svm-loss/      (Svm -> hinge loss)
# https://medium.com/activating-robotic-minds/demystifying-kl-divergence-7ebe4317ee68
# https://medium.com/usf-msds/choosing-the-right-metric-for-evaluating-machine-learning-models-part-2-86d5649a5428
# https://www.dataquest.io/blog/understanding-regression-error-metrics/

## for unsupervised learning (clustering, etc.) , the following section should be replaced by appropriate algos.



#########    basic model building : CLASSIFICATION   #######
# 1. fit model - vary ML algo & hyperparameters
# types of ML algo: 
#       logistic, SVM, tree-methods (DT), Bayesian models
#       generalised linear model/GLM (for regression?)
#       neural net/MLP, CNN & variants (deep learning based methods, transfer learning)
#       ensemble methods: bagging, random forest, boosting, stacking
#       https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205?gi=f30deb598cb4
#       https://towardsdatascience.com/stacking-classifiers-for-higher-predictive-performance-566f963e4840
#       https://www.kdnuggets.com/2017/02/stacking-models-imropved-predictions.html
#       https://www.analyticsvidhya.com/blog/2015/08/introduction-ensemble-learning/
# for a comprehensive list, see websites: machinelearningmastery, scikitlearn
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



#######    basic model building : REGRESSION   #######
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error


# 1. fit model
regr = linear_model.LinearRegression()
# regr = linear_model.Ridge(alpha=0.1)
# regr = linear_model.Lasso()
# regr = RandomForestRegressor(n_estimators =10,max_depth=5, random_state=0,verbose =0) 
#                              max_features=2, min_samples_split=4, n_estimators=50, min_samples_leaf=2
#gb = GradientBoostingRegressor(loss='quantile', learning_rate=0.0001, n_estimators=50, max_features='log2', min_samples_split=2, max_depth=1)
#ada_tree_backing = DecisionTreeRegressor(max_features='sqrt', splitter='random', min_samples_split=4, max_depth=3)
#ab = AdaBoostRegressor(ada_tree_backing, learning_rate=0.1, loss='square', n_estimators=1000)
regr.fit(X_train_pca, y_train)

# 2. predict
y_test_predict = regr.predict(X_test_pca)

# 3. model validation
score = cross_val_score(regr, X_train_pca,y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))

print(mean_absolute_error(y_test_predict, y_test))

print('Reg score train ' + str(regr.score(X_train_pca, y_train, sample_weight=None)))
print('Reg score test ' + str(regr.score(X_test_pca, y_test, sample_weight=None)))
y_test_reshaped = (y_test.reshape(y_test.shape[0]))
print('Reg r2score ' + str(r2_score(y_test_reshaped , dataY_pred)))

y_train_reshaped = (y_train.reshape(y_train.shape[0]))
accuracies = cross_val_score(estimator = regr, X = X_train_pca, y = y_train_reshaped, cv = 10,scoring='neg_mean_squared_error')
print('cross_val_score_mean ' + str(accuracies.mean()))
print('cross_val_score_std ' + str(accuracies.std()))

# Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / y_test)
# print(mape)



########   use of pipelines   ###########
# https://www.youtube.com/watch?v=SawQZdAcazY



########     cross-validation & hyper-parameter tuning     ########
# https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
# https://towardsdatascience.com/hyperparameter-tuning-explained-d0ebb2ba1d35
# https://towardsdatascience.com/an-example-of-hyperparameter-optimization-on-xgboost-lightgbm-and-catboost-using-hyperopt-12bc41a271e
# https://towardsdatascience.com/hyperparameter-tuning-c5619e7e6624
# create: training, validation & test sets
# use of k-fold on the training data

## check for overfitting by comparing performance on train & test sets


#######    combine different models   #######
# voting classifier - weighted, unweighted
# adaboost, xgboost, stacking regressor



#######   bagging data sets    #######
# creating multiple train sets by taking subsets from original data, and building a model on each of these data subsets



#################################################################################
### compare results from multiple models ###

# using T-test,  check if RMSE of predictions from 2 models are similar; rmse calculated over multiple iterations
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



############################################################################################################################
############################################################################################################################

###### interesting article  ####
# https://gab41.lab41.org/the-10-algorithms-machine-learning-engineers-need-to-know-f4bb63f5b2fa

# https://towardsdatascience.com/how-do-you-know-you-have-enough-training-data-ad9b1fd679ee
# https://towardsdatascience.com/introducing-model-bias-and-variance-187c5c447793

# https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65 
# https://notast.netlify.com/post/explaining-predictions-random-forest-post-hoc-analysis-permutation-impurity-variable-importance/
# https://www.analyticsvidhya.com/blog/2015/12/improve-machine-learning-results/

# https://analyticsindiamag.com/4-python-libraries-for-getting-better-model-interpretability/
# https://www.analyticsvidhya.com/blog/2019/08/decoding-black-box-step-by-step-guide-interpretable-machine-learning-models-python/

# https://heartbeat.fritz.ai/top-7-libraries-and-packages-of-the-year-for-data-science-and-ai-python-r-6b7cca2bf000
# https://towardsdatascience.com/automl-and-autodl-simplified-b6786e5560ff
# https://analyticsindiamag.com/10-popular-automl-tools-developers-can-use/


####  interesting websites   ######
# https://www.analyticsvidhya.com/blog/category/machine-learning/
# machinelearningmastery
# kaggle Learn, Datacamp
# Andrew NG course (youtube) & book chapters  (machine learning yearning?)
# Andriy Burkov: machine learning engineering   http://themlbook.com/wiki/doku.php
# https://yle.fi/uutiset/osasto/news/finland_offers_free_online_artificial_intelligence_course_to_anyone_anywhere/10206283
# https://www.engadget.com/2018/04/02/microsoft-public-courses-building-ai-skills/?_lrsc=d4ac881b-dc72-457c-9ee3-5b597cbdcc0e
# https://qz.com/1206229/this-is-the-best-book-for-learning-modern-statistics-its-free/



###########################################################################
####  not covered here  ####
# git usage, SQL

# data visualization
# https://towardsdatascience.com/become-a-pandas-power-user-with-these-display-customizations-6d3a5a5885c1
# https://www.marsja.se/python-data-visualization-techniques-you-should-learn-seaborn/
# https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000202&fbclid=IwAR0lZAThKTkasEquIeDRtYTJIJnMGYkjBHVgbkeRAvOU4uAi4hNpqPy3rXQ

# data set generation (through website scraping: BeautifulSoup, Scrapy, and rvest)

# graph algorithms, reinforcement learning, HMM, Monte Carlo, genetic algo, GAN
# https://medium.com/free-code-camp/a-brief-introduction-to-reinforcement-learning-7799af5840db
# https://www.analyticsindiamag.com/meet-autogan-the-neural-architecture-search-for-generative-adversarial-networks/
# https://medium.com/sigmoid/https-medium-com-rishabh-anand-on-the-origin-of-genetic-algorithms-fc927d2e11e0
# https://medium.com/@MohammedAmer/evolutionary-computation-a-primer-e3ca6fb0db5c

# optimisation techniques


