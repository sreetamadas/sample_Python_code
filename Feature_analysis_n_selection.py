#################################################################################################################
### feature selection : detect factors influencing Y ###

## to do ##
## add plot of Y-vs-time to show variation in Y for similar days

# should correlated variables be removed ?

# scaling for test data - same or different from training data?
# changes in R2 & CC for RandomForest with scaling in data
# how to incorporate factor variables, since text data is not accepted by RF in python? 
# Should the numeric representation of these labels be scaled?

# estimates of errors - add MSE?

# how is feature importance computed here? what is std ?
# should correlated variables be removed b4 calculating feature importance? use RIDGE/ LASSO ?
# feature importance on training data

# automate parameter tuning
# include k-fold cross validation ?is it required for regression/ random forest
# how to take care of class imbalance?
# find useful range of features
###########################################################################################################

import numpy as np
import pandas #as pd
import matplotlib.pyplot as plt
import seaborn as sns

#from sklearn import datasets

#from check_stats_timeSeriesData import dataCleaning
# https://stackoverflow.com/questions/20309456/call-a-function-from-another-file-in-python

#######################################################################################################################

## in the following example, we will merge data from 2 files by DateTime column, & then perform feature selection.
## file1 has the regression variable, whereas file2 has other independent variables


## fix time stamp in file1
def addTimeToFile1(df):
    "adding time to date column"
    # read as date col, if not read in correctly (check with prdcn.dtypes )
    df['Date'] = pandas.to_datetime(df['Date'], format="%d.%m.%Y")
    # change format
    df['Date'] = df['Date'].dt.strftime("%Y-%m-%d")
    
    # add time col
    df['Time'] = "04:00:00"
    df.loc[df['Shift'] == 2, 'Time'] = "12:00:00"
    df.loc[df['Shift'] == 3, 'Time'] = "20:00:00"
    
    # concatenate date & time
    df['DateTime'] = df['Date'].map(str) + ' ' + df['Time'].map(str)
    df['DateTime'] = pandas.to_datetime(df['DateTime'], format="%Y-%m-%d %H:%M:%S")
    #df.dtypes
    
    # delete unnecessary cols (alternate: use df.drop('column_name', axis=1, inplace=True))
    del df['Shift']
    del df['Time']
    
    return df;


def addTimeToFile2(sca):
    "preparing sca datetime for merging with df"
    # select date
    sca['DATETIME'] = pandas.to_datetime(sca['DATETIME'], format="%m/%d/%Y %H:%M")
    sca['Date'] = [d.date() for d in sca['DATETIME']]  # d.date() for date & d.time for time part
    sca['Date'] = pandas.to_datetime(sca['Date'], format="%Y-%m-%d")   # format date
    #sca['Date'] = sca['Date'].dt.strftime("%Y-%m-%d")

        
    # select hr from time, and add shift
    sca['hr'] = pandas.DatetimeIndex(sca['DATETIME']).hour
    sca['Time'] = "04:00:00"
    sca.loc[ sca['hr'] >= 8 , 'Time'] = "12:00:00"
    sca.loc[ sca['hr'] >= 16 , 'Time'] = "20:00:00"
    
    # rename original datetime col
    sca = sca.rename(columns={'DATETIME': 'DT'})
    
    # merge date & mid-time-of-shift cols
    sca['DateTime'] = sca['Date'].map(str) + ' ' + sca['Time'].map(str)
    sca['DateTime'] = pandas.to_datetime(sca['DateTime']) #, format="%Y-%m-%d %H:%M:%S")  # merge
    
    # get day of week (monday = 0; sunday = 6)
    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DatetimeIndex.dayofweek.html
    #sca['weekday'] = sca['DT'].apply( lambda x: x.dayofweek)
    #sca['weekday'] = pandas.DatetimeIndex(sca['DT']).dayofweek
    sca['weekday'] = sca['DT'].dt.dayofweek

    
    # delete unnecessary cols
    del sca['Time']
    #del sca['hr']
    del sca['Date']
    #sca.drop(['Time','Date'], axis=1, inplace=True)
    
    return sca;


### combine 2 dfs
# this step can change according to data format or type of data available
def combineDat(sca,df):
    "combine data"
    # step 1: rename Index col to standard format in 2 files
    sca = sca.rename(columns={'ID no': 'Index'})
    #sca = sca.rename(columns={'TT': 'TotalCount'})
    # following step has additional formatting, if the fields have extra characters in the 2 files
    #sca['Index'] = 'MC ' + sca['Index'].map(str).str[1:] # sca['Index'].map(str)[1:] 
    
    # step 2: merge by DateTime, and Index
    sca['Index'] = sca['Index'].astype(str)
    # https://pandas.pydata.org/pandas-docs/stable/merging.html
    # if merge returns empty df, check that the columns being merged on have same data-type in both dataframes
    new_df = pandas.merge(sca, df, how='inner', on=['DateTime','Index'])
    
    # step 3: add extra features: time-of-day, day-of-week (keep these 'factors' as numeric instead of text)
    # comes from sca file
    
    # step 4: remove unnecessary cols 
    new_df.drop(['Extracol1','Extra_col2'], axis=1, inplace=True)
    #del new_df['Extracol1']
    #del new_df['Extra_col2']
    
    return new_df;

#############################################################################################################

## prepare data from file1: read, clean, add DateTime
df = pandas.read_csv("C:\\Users\\username\\Desktop\\data\\file1.csv")
# clean df to take care of missing values, etc
df = addTimeToFile1(df)


## prepare data from file2: read, clean, add DateTime
sca = pandas.read_csv("C:\\Users\\username\\Desktop\\data\\file2.csv")
# clean sca to take care of missing values, etc
sca = addTimeToFile2(sca)


## merge dfs
test = combineDat(sca,df)
# remove all cols not required for feature selection


##############################################################################################################
#### STEP 1: create training and test data  ####
# define X & Y cols
features = test.drop(['Y'], axis=1)
target = test['Y']

# define training & test data
# http://blog.josephmisiti.com/help-commands-for-doing-machine-learning-in-python (to preserve % samples in each class)
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.8, random_state=0)


# apply scaling, if required (not required for RF; may be needed for other algos)
# same scaling should be applied on training & test set
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cols = list(set(features.columns.tolist()) - set(["Index","ModelNo","hr","weekday","Shift"]))  ## mention cols which should not be scaled
X_train.loc[:,cols] = scaler.fit_transform(X_train.loc[:,cols])
X_test.loc[:,cols] = scaler.fit_transform(X_test.loc[:,cols])

#std_scale = preprocessing.StandardScaler().fit(X_train)
#X_train_std = std_scale.transform(X_train)
#X_test_std = std_scale.transform(X_test)


######################################################################################################################
#### check correlation of features  ####
# method 1
#corr = X_train.corr()
#fig, ax = plt.subplots(figsize=(10, 10))  # figsize=(size, size)
#ax.matshow(corr)
#plt.xticks(range(len(corr.columns)), corr.columns);
#plt.yticks(range(len(corr.columns)), corr.columns);

#plt.matshow(X_train.corr())
#plt.show()


# method 2
f, ax = plt.subplots(figsize=(6, 4))
corr = test.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt.show()


# method 3
# pandas.scatter_matrix(X_train, alpha = 0.3, figsize = (14,8), diagonal = 'kde') ## errors
# sns.pairplot(X_train)  ## run did not complete


######################################################################################################################
###  check PCA  ###
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# https://7264-843222-gh.circle-artifacts.com/0/home/ubuntu/scikit-learn/doc/_build/html/stable/auto_examples/preprocessing/plot_scaling_importance.html

from sklearn.decomposition import PCA

pca = PCA().fit(X_train)  # PCA(n_components=2)
X_train_pc = pca.transform(X_train)  # how is X_train_pc different from X_train?
X_test_pc = pca.transform(X_test)
#pca.components_[0,]  # is this the contribution of diff features to PC1 ?

# how to remove correlated features?
#https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
# use ridge regression?


pca_var = pca.explained_variance_ratio_ 
#std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
indices = np.argsort(pca_var)[::-1]

# Print the feature ranking
#print("PC variance:")
#for f in range(X_train.shape[1]):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], pca_var[indices[f]]))

# Plot the PC variances
plt.figure()
plt.title("PCA variances")
plt.bar(range(X_train_pc.shape[1]), pca_var[indices],
       color="r", align="center")
plt.xticks(range(X_train_pc.shape[1]), indices)
plt.xlim([-1, X_train_pc.shape[1]])
plt.show()


###########################################################################################################################
### STEP 2: model building using random forest (TRY RIDGE, LASSO REGRESSION)  ####
# http://www.blopig.com/blog/2017/07/using-random-forests-in-python-with-scikit-learn/
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# http://scikit-learn.org/stable/modules/ensemble.html#forest
# https://www.dataquest.io/blog/python-vs-r/

def runRFmodel(X_train,y_train,X_test,y_test):
    "model building using random forest regressor"
    
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0, n_jobs=2, max_features="sqrt")
    # max_features=int(X_train.shape[1]/3)  , sqrt,  auto (this uses all)
    # X_train.shape[1] : no. of columns in X_train
    # hyper-parameters to vary: n_estimators (no. of trees; ntry in R)
    #                           max_features (no. of features to check at each split; mtry in R)
    
    rf.fit(X_train, y_train)
    
    # model prediction scores
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr, pearsonr
    
    predicted_train = rf.predict(X_train)
    predicted_test = rf.predict(X_test)
    test_score = r2_score(y_test, predicted_test)
    spearman = spearmanr(y_test, predicted_test)
    pearson = pearsonr(y_test, predicted_test)
    
    print('Out-of-bag R-2 score estimate: ', rf.oob_score_)
    # print('Out-of-bag score estimate:%8.10f' %rf.oob_score_) # https://www.python-course.eu/python3_formatted_output.php
    print('Test data R-2 score: ', test_score)
    print('Test data Spearman correlation: ', spearman[0])
    print('Test data Pearson correlation: ', pearson[0])
    
    return rf;

#rf = runRFmodel(X_train_pc,y_train,X_test_pc,y_test)
rf = runRFmodel(X_train,y_train,X_test,y_test)


### feature importance  ###
# http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
# what is this std? how is feature importance calculated in python?
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()


#################################################################################################################

# parameter tuning

# k-fold cross validation - is it required for regression/random forest ?

# select useful range of features


