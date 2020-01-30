from datetime import datetime

## print current time
datetime.now().strftime('%Y-%m-%d %H:%M:%S')


## time & date format
## format time stamp, in case of non-standard format; 
## non-standard format of datetime can be ascertained by checking the O/P for datetime col in mc.dtypes
df["Date"] = pandas.to_datetime(df["Date"], format="%Y.%m.%d")  ## specify the format from input data here
# change format
df['Date'] = df['Date'].dt.strftime("%Y-%m-%d")


# extract date & time separately
mc['Day'] = [d.date() for d in mc['Date']]  # d.date() for date & d.time for time part


## extract portion from dateTime
df.loc[row_index, 'hr'] = df.loc[row_index, 'dateTime'].hour



#energy['Day'] = pandas.to_datetime(energy['Day']) 
energy['day_of_week2'] = energy['Day'].dt.weekday_name
energy['day_of_week'] = energy['Day'].dt.dayofweek
# https://stackoverflow.com/questions/30222533/create-a-day-of-week-column-in-a-pandas-dataframe-using-python
# https://stackoverflow.com/questions/28009370/get-weekday-day-of-week-for-datetime-column-of-dataframe


# adding an interval to a time stamp
df["dateUTC"] = pd.to_datetime(df["dateUTC"]) #, format="%Y.%m.%d")  ## specify the format from input data here
from datetime import datetime, timedelta
df['dateTime'] = df['dateUTC'] + timedelta(minutes=330)  ## fix dateTime to IST from UTC : add 5 hrs 30 min


# calculate difference/ interval b/w time stamps
#df.loc[row_index, 'timedel'] =  (df.loc[row_index, 'dateTime'] - df.loc[(row_index-1), 'dateTime']) #.astype('timedelta64[m]')
df.loc[row_index, 'timedel'] =  pd.Timedelta(pd.Timestamp(df.loc[row_index, 'dateTime']) - pd.Timestamp(df.loc[(row_index-1), 'dateTime'])).total_seconds()/60 #.astype('timedelta64[m]')



# insert missing time stamps (for data missed after operating hrs)
# fill in missing values with previous value
# 1. Set the datestamp columns as the index of your DataFrame
df = df.set_index('dateTime')
df = df.reindex(pd.date_range(start=df.index[0], end=df.index[-1], freq='300s'))
df = df.fillna(method='ffill')


# set dateTime as a separate column instead of index
df['dateTime'] = df.index
df = df.reset_index()
del df['index']
#df.head(n=2)

