
# coding: utf-8

# In[ ]:

### temperature monitoring of freezers ###
# keep 1 day's data in memory
# as each new data point comes in, push it to the end of the df, & push out elements from the top
# also, add this new data point to a new df

## LEVEL 1
# against each such entry, compute the median, & sd on deviation from median using the window  ******
# flag as anomaly if the deviation is >threshold (thres1)


## LEVEL 2
# compute hr of day
# check if there is a switch in state; assign a new no. for each such change
# calculate time
# calculate cum time
# check if cum time is above a threshold (thres2: 30min)


## parameters used: rolling window size, deviation threshold, cumulative time threshold

## there are some small differences in the no. of anomalies flagged by R-code (177) vs python (185) : this is due to
## small differences in the SD values (in decimals) computed by the tools


# In[1]:

## set directory
import os
os.chdir('c:\Users\Desktop\data\code_python')


# In[2]:

## read in data
import pandas as pd

#filepath = raw_input("enter fullPath & filename: ")  
filepath = "c:\\Users\\Desktop\\data\\data1.csv"
T = pd.read_csv(filepath)  

T.head(n=2)

#T.describe()

# count no. of rows with missing values
#print(T.isnull().sum())

#T.dtypes


# In[3]:

## non-standard format of datetime can be ascertained by checking the O/P for datetime col in T.dtypes
T["dateUTC"] = pd.to_datetime(T["dateUTC"]) #, format="%Y.%m.%d")  ## specify the format from input data here

## fix dateTime to IST from UTC : add 5 hrs 30 min
from datetime import datetime, timedelta
T['dateTime'] = T['dateUTC'] + timedelta(minutes=330)

### retain required cols only (drop unnecessary cols )
T = T[['dateTime','temp']]


# In[4]:

# insert missing time stamps (for data missed after operating hrs)
# fill in missing values with previous value
#T = T.set_index('dateTime')
#T = T.reindex(pd.date_range(start=T.index[0], end=T.index[-1], freq='300s'))

#T = T.fillna(method='ffill')  # backfill: fill using next value  ; ffill= using previous value
# can use a distribution to fill in instead of using flat values - see how to do this ?

## NOT USING DATA IMPUTATION as it may be problematic in real time

# set dateTime as a separate column instead of index
#T['dateTime'] = T.index
#T = T.reset_index()
#del T['index']
#T.head(n=2)


# In[5]:

## set thresholds:

## save one day's data (1 hr has 12 points; 1 day has 12*24 = 288 points with data imputation, 180 data points without)
# this will be used for calculating rolling terms
rownum = 180  # 288

## level 1 threshold for sd
thres = 3

## level 2 threshold for cumulative time
thres2 = 30


# In[6]:

## save one day's data
backup = T[:rownum]  # first rownum rows of T, from 0 - (rownum-1)
#backup.head(n=2)

# assign a new column to label anomalies for incoming data in the rolling window;
# the corresponding data points will not be used for calculating SD
backup = backup.assign(row_anom = 0)
#backup.tail(n=2)


# In[7]:

## assign initial values to the columns (push last 1 data point into a new df)
import numpy as np

current = pd.DataFrame()
row_index = 0

#current = T.iloc[[(rownum - 1)]]  # since index starts from 0 in python (vs 1 in R)
current.loc[row_index, 'dateTime'] = T.loc[(rownum - 1), 'dateTime']
current.loc[row_index, 'temp'] = T.loc[(rownum - 1), 'temp']


#hr = current.index.hour
current['dateTime'] = pd.to_datetime(current['dateTime'])
current.loc[row_index, 'hr'] = current.loc[row_index, 'dateTime'].hour
#current = pd.concat([current, pd.DataFrame(hr, index=current.index)], axis = 1)
#current = current.rename(columns={0: 'hr'})

current.loc[row_index, 'timedel'] = 0
current.loc[row_index, 'avg'] = backup['temp'].median()
current.loc[row_index, 'sd'] = np.std(backup['temp'] - backup['temp'].median()) #, dtypes=np.float64)

# 1st layer anomaly
current.loc[row_index, 'anom'] = np.where(current.temp - current.avg > thres*current.sd, 1, 0)
# https://stackoverflow.com/questions/19913659/pandas-conditional-creation-of-a-series-dataframe-column
# see this link for series of if-else conditions

#current.loc[row_index, 'state_change'] = 0  # this can be skipped; not being used except as an indicator
current.loc[row_index, 'csum_time'] = 0

# 2nd layer anomaly
current.loc[row_index, 'f_anom'] = 0

## how many data points were used to calculate sd
current.loc[row_index, 'sd_row'] = rownum


current


# In[8]:

# create a loop to simulate reading in new data and checking for anomalies

for i in range(rownum, len(T.index) ):  # this includes values from rownum to len(T.index)-1; see if last row in T is read correctly;  ## rownum+1  ; len(T.index)
    # add new data to end of backup, & remove row from top
    backup = backup.drop(backup.head(1).index) #, inplace=True)
    backup = backup.append(T.iloc[[i]]) #, axis=0)
    #backup.loc[(rownum-1), 'dateTime'] = T[i, 'dateTime']
    #backup.loc[(rownum-1), 'temp'] = T[i, 'temp']
    #backup.iloc[(rownum-1), 'row_anom'] = 0
    #backup.loc[len(backup.index), 'row_anom'] = 0
    backup.loc[:, "row_anom"].iloc[-1] = 0
    #https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/

    
    # go to new row of df 'current' & update
    row_index = row_index + 1
    current.loc[row_index, 'dateTime'] = T.loc[i, 'dateTime']
    current.loc[row_index, 'temp'] = T.loc[i, 'temp']
    current.loc[row_index, 'hr'] = current.loc[row_index, 'dateTime'].hour
    #current.loc[row_index, 'timedel'] =  (current.loc[row_index, 'dateTime'] - current.loc[(row_index-1), 'dateTime']) #.astype('timedelta64[m]')
    current.loc[row_index, 'timedel'] =  pd.Timedelta(pd.Timestamp(current.loc[row_index, 'dateTime']) - pd.Timestamp(current.loc[(row_index-1), 'dateTime'])).total_seconds()/60 #.astype('timedelta64[m]')
    current.loc[row_index, 'avg'] = backup['temp'].median()
    #current.loc[row_index, 'sd'] = np.std(backup['temp']) #, dtypes=np.float64)
    no_anom = backup[backup.row_anom == 0]
    current.loc[row_index, 'sd'] = np.std(no_anom['temp'] - backup['temp'].median())
    
    
    # 1st layer anomaly
    current.loc[row_index, 'anom'] = np.where(current.loc[row_index, 'temp'] - current.loc[row_index, 'avg'] > thres*current.loc[row_index, 'sd'], 1, 0)
    
    
    
    # get state change   # this can be skipped; not being used except as an indicator
    #current.loc[row_index, 'state_change'] = abs(current.loc[row_index, 'anom'] - current.loc[(row_index-1), 'anom'])
    
    # calculate cumulative time for each change of state
    if current.loc[row_index,'anom']!= current.loc[(row_index - 1),'anom']:
        current.loc[row_index, 'csum_time'] = current.loc[row_index, 'timedel']
    else:
        current.loc[row_index, 'csum_time'] = current.loc[(row_index -1), 'csum_time'] + current.loc[row_index, 'timedel']
    
    
    
    # 2nd layer anomaly
    current.loc[row_index, 'f_anom'] = 0
    # long dur of high temp outside loading hr
    #if (current.loc[row_index, 'hr'] < 11 or current.loc[row_index, 'hr'] > 17) and current.loc[row_index, 'anom'] == 1  and current.loc[row_index, 'csum_time'] > 10:
    #    current.loc[row_index, 'f_anom'] = 1
    # within loading hr, see high temp > 1hr & more than once
    #if (current.loc[row_index, 'hr'] >= 11 and current.loc[row_index, 'hr'] <= 17) and current.loc[row_index, 'anom'] ==1  and current.loc[row_index, 'csum_time'] > 60:
    #    current.loc[row_index, 'f_anom'] = 1
    ### flag all anomalies longer than thres2 = 30min
    if (current.loc[row_index, 'anom'] == 1  and current.loc[row_index, 'csum_time'] > thres2):
        current.loc[row_index, 'f_anom'] = 1

    ## how many data points were used to calculate sd
    current.loc[row_index, 'sd_row'] = len(no_anom.index)
    
    ## reassign the row_anom in backup data based on anomaly status in level 1
    backup.loc[:, "row_anom"].iloc[-1] = current.loc[row_index, 'anom']
 
    # print anomalies
    #if current.loc[row_index, 'f_anom'] == 1:
    #    print(current.loc[[row_index]])



# In[9]:

current.to_csv("python_anom_rollingMedian_SD.csv", sep=',')


# In[10]:

# plot raw data
import matplotlib.pyplot as plt

ax = current.plot(x='dateTime', y='temp',color="blue", linewidth=1, fontsize=6, figsize=(10,3))
ax.set_xlabel('date')
ax.set_ylabel('temperature')
plt.show()


# In[11]:

# plot moving average temps
ax1 = current.plot(x='dateTime', y='avg',color="black", linewidth=1, fontsize=6, figsize=(10,3))
ax1.set_xlabel('date')
ax1.set_ylabel('moving median temperature')
plt.show()


# In[12]:

# plot layer 1 anomalies
import numpy as np
current['col'] = np.where(current['anom'] == 1.0 , 'red', 'yellow')
# del current['col']
# current.head(2)
#current = current.set_index('dateTime')

df = current[['dateTime','temp', 'col']]
df['dateTime'] = pd.to_datetime(df['dateTime'])
df['t1'] = df.dateTime.astype(np.int64)

#fig, ax = plt.subplots(figsize=(10,3))
#plt.plot(x=df['t1'], y=df['temp'],c=df['col']) #, figsize=(10,3)) ; # blank plot
#df.plot(df.dateTime, df.temp, c=df['col'])
#df.plot('dateTime', 'temp', c=df['col'])

#https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter
# https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size
fig, ax = plt.subplots(figsize=(12,3))
ax.scatter(df['t1'],df['temp'],c=df['col'], marker = 'o', s=10, edgecolors='face', alpha=0.4) #, cmap = cm.jet ; marker = 'o'
#ax2.set_xlabel('date')
#ax2.set_ylabel('temperature')
plt.show()


# In[13]:

# plot layer 2 anomalies

current['col2'] = np.where(current['f_anom'] == 1.0 , 'red', 'yellow')
# del current['col']
# current.head(2)
#current = current.set_index('dateTime')

df1 = current[['dateTime','temp', 'col2']]
df1['dateTime'] = pd.to_datetime(df1['dateTime'])
df1['t1'] = df1.dateTime.astype(np.int64)

fig, ax = plt.subplots(figsize=(12,3))
ax.scatter(df1['t1'],df1['temp'],c=df1['col2'], marker = 'o', s=10, edgecolors='face', alpha=0.4) #, cmap = cm.jet ) #marker='.', markersize=10
#ax2.set_xlabel('date')
#ax2.set_ylabel('temperature')
plt.show()


# In[ ]:



