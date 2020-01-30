
# coding: utf-8

# In[1]:

### temperature monitoring of freezers ###
# keep 1 day's data in memory
# as each new data point comes in, push it to the end of the df, & push out elements from the top
# also, add this new data point to a new df

## LEVEL 1
# against each such entry, compute the mean & sd (currently being used; (or median & MAD)) using the window  ******
# flag as anomaly if the deviation is >threshold


## LEVEL 2
# compute hr of day
# check if there is a switch in state; assign a new no. for each such change
# calculate time
# calculate cum time
# check if cum time is above a threshold


# In[1]:

## set directory
import os
os.chdir('c:\Users\Desktop\data\code_python')


# In[2]:

## read in data
import pandas as pd

#filepath = raw_input("enter fullPath & filename: ")  
filepath = "c:\\Users\\Desktop\\data\\data.csv"
T = pd.read_csv(filepath)  

T.head(n=2)


# In[3]:

#T.describe()

# count no. of rows with missing values
#print(T.isnull().sum())

#T.dtypes


# In[4]:

## non-standard format of datetime can be ascertained by checking the O/P for datetime col in T.dtypes
T["dateUTC"] = pd.to_datetime(T["dateUTC"]) #, format="%Y.%m.%d")  ## specify the format from input data here

## fix dateTime to IST from UTC : add 5 hrs 30 min
from datetime import datetime, timedelta
T['dateTime'] = T['dateUTC'] + timedelta(minutes=330)


# In[5]:

### drop unnecessary cols - retain required cols only
T = T[['dateTime','temp']]


# In[6]:

# insert missing time stamps (for data missed after operating hrs)
# fill in missing values with previous value
T = T.set_index('dateTime')
T = T.reindex(pd.date_range(start=T.index[0], end=T.index[-1], freq='300s'))

T = T.fillna(method='ffill')  # backfill: fill using next value  ; ffill= using previous value

# can use a distribution to fill in instead of using flat values - see how to do this ?


# In[7]:

# set dateTime as a separate column instead of index
T['dateTime'] = T.index
T = T.reset_index()
del T['index']
T.head(n=2)


# In[8]:

## save one day's data (1 hr has 12 points; 1 day has 12*24 = 288 points)
# this will be used for calculating rolling terms
rownum = 288
backup = T[:rownum]  # first 288 rows of T, from 0 - 287


# In[9]:

## push last 1 data point into a new df
#current = T.iloc[[(rownum - 1)]]  # since index starts from 0 in python (vs 1 in R)
current = pd.DataFrame()
current.loc[0, 'dateTime'] = T.loc[(rownum - 1), 'dateTime']
current.loc[0, 'temp'] = T.loc[(rownum - 1), 'temp']
#current.index.name = 'dateTime'
#current


# In[10]:

## assign initial values to the columns
import numpy as np

row_index = 0
#hr = current.index.hour
current['dateTime'] = pd.to_datetime(current['dateTime'])
current.loc[row_index, 'hr'] = current.loc[row_index, 'dateTime'].hour
#current = pd.concat([current, pd.DataFrame(hr, index=current.index)], axis = 1)
#current = current.rename(columns={0: 'hr'})

current.loc[row_index, 'timedel'] = 0
current.loc[row_index, 'avg'] = backup['temp'].mean()
current.loc[row_index, 'sd'] = np.std(backup['temp']) #, dtypes=np.float64)

# 1st layer anomaly
current.loc[row_index, 'anom'] = np.where(current.temp - current.avg > 2.5*current.sd, 1, 0)
# https://stackoverflow.com/questions/19913659/pandas-conditional-creation-of-a-series-dataframe-column
# see this link for series of if-else conditions

#current.loc[row_index, 'state_change'] = 0  # this can be skipped; not being used except as an indicator
current.loc[row_index, 'csum_time'] = 0

# 2nd layer anomaly
current.loc[row_index, 'f_anom'] = 0

#current


# In[11]:

# create a loop to simulate reading in new data and checking for anomalies

for i in range(rownum, len(T.index) ):  #  rownum+1  ; len(T.index)
    # add new data to end of backup, & remove row from top
    backup = backup.drop(backup.head(1).index) #, inplace=True)
    backup = backup.append(T.iloc[[i]]) #, axis=0)
    
    # go to new row of df 'current' & update
    row_index = row_index + 1
    current.loc[row_index, 'dateTime'] = T.loc[i, 'dateTime']
    current.loc[row_index, 'temp'] = T.loc[i, 'temp']
    current.loc[row_index, 'hr'] = current.loc[row_index, 'dateTime'].hour
    #current.loc[row_index, 'timedel'] =  (current.loc[row_index, 'dateTime'] - current.loc[(row_index-1), 'dateTime']) #.astype('timedelta64[m]')
    current.loc[row_index, 'timedel'] =  pd.Timedelta(pd.Timestamp(current.loc[row_index, 'dateTime']) - pd.Timestamp(current.loc[(row_index-1), 'dateTime'])).total_seconds()/60 #.astype('timedelta64[m]')
    current.loc[row_index, 'avg'] = backup['temp'].mean()
    current.loc[row_index, 'sd'] = np.std(backup['temp']) #, dtypes=np.float64)
    
    
    # 1st layer anomaly
    current.loc[row_index, 'anom'] = np.where(current.loc[row_index, 'temp'] - current.loc[row_index, 'avg'] > 2.5*current.loc[row_index, 'sd'], 1, 0)
    
    
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
    if (current.loc[row_index, 'hr'] < 11 or current.loc[row_index, 'hr'] > 17) and current.loc[row_index, 'anom'] == 1  and current.loc[row_index, 'csum_time'] > 10:
        current.loc[row_index, 'f_anom'] = 1
    # within loading hr, see high temp > 1hr & more than once
    if (current.loc[row_index, 'hr'] >= 11 and current.loc[row_index, 'hr'] <= 17) and current.loc[row_index, 'anom'] ==1  and current.loc[row_index, 'csum_time'] > 60:
        current.loc[row_index, 'f_anom'] = 1
    
    # print anomalies
    #if current.loc[row_index, 'f_anom'] == 1:
    #    print(current.loc[[row_index]])



# In[13]:

current.to_csv("python_anom.csv", sep=',')


# In[14]:

# plot raw data
import matplotlib.pyplot as plt

ax = current.plot(x='dateTime', y='temp',color="blue", linewidth=2, fontsize=6, figsize=(10,3))
ax.set_xlabel('date')
ax.set_ylabel('temperature')
plt.show()


# In[15]:

# plot moving average temps
ax1 = current.plot(x='dateTime', y='avg',color="blue", linewidth=2, fontsize=6, figsize=(10,3))
ax1.set_xlabel('date')
ax1.set_ylabel('MA temperature')
plt.show()


# In[47]:

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

fig, ax = plt.subplots(figsize=(12,3))
ax.scatter(df['t1'],df['temp'],c=df['col'], marker = 'o') #, cmap = cm.jet )
#ax2.set_xlabel('date')
#ax2.set_ylabel('temperature')
plt.show()


# In[1]:

# plot layer 1 anomalies

current['col2'] = np.where(current['f_anom'] == 1.0 , 'red', 'white')
# del current['col']
# current.head(2)
#current = current.set_index('dateTime')

df1 = current[['dateTime','temp', 'col2']]
df1['dateTime'] = pd.to_datetime(df1['dateTime'])
df1['t1'] = df1.dateTime.astype(np.int64)

fig, ax = plt.subplots(figsize=(12,3))
ax.scatter(df1['t1'],df1['temp'],c=df1['col2'], marker = '.', markersize= 10) #, cmap = cm.jet ) #marker='.', markersize=10
#ax2.set_xlabel('date')
#ax2.set_ylabel('temperature')
plt.show()


# In[ ]:



