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

#############################################################################################################

## prepare data from file1: read, clean, add DateTime








