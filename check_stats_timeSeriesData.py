######################################################################################################
#### get stats of time series data (from sensor)

## set directory / path

## import libraries
import pandas
from matplotlib import pyplot
from datetime import datetime

#########################################################################################################
###  FUNCTION DEFINITION  ###

#### data cleaning  #####
def dataCleaning(mc):
    "CHECK MISSING VALUES: individual cols & time stamps"
    
    ## check data types
    #mc.dtypes 
    
    ## format time stamp, in case of non-standard format
    # non-standard format of datetime can be ascertained by checking the O/P for datetime col in mc.dtypes
    # mc["Date"] = pandas.to_datetime(mc["Date"], format="%Y.%m.%d %H:%M:%S")  ## specify the format from input data here
    
    ## sort timestamps
    mc = mc.sort_values(by='Date')
    
    ## check for repeats in time stamps (?)
    
    ## check for missing values 
    # step 1: missing data in some columns; drop a column if too many missing values
    mc.describe()
    # code for adding missing values (median or NA or fill using previous value (up to 3 places)?) 
    # https://pandas.pydata.org/pandas-docs/stable/missing_data.html
    mc = mc.fillna(method='pad', limit=3) 
    
    # step 2: check for missing time stamps : leave out dates with too many (>4%) missing values
    # check the no. of entries for each date - leave out dates with less no. of entries
    # this does not add timestamps for days with a few missing points
    #  A. format DateTime colunm to only date
    from datetime import datetime
    mc['Day'] = [d.date() for d in mc['Date']]  # d.date() for date & d.time for time part
    #  B. count no. of rows
    entries = mc.groupby('Day').agg({'Date': 'count'}).reset_index()
    entries = entries.rename(columns={'Date': 'instances'})
    # write as csv
    # entries.to_csv('num_of_entries.csv') 
    #  C. select threshold to remove days with too few data (higher threshold allows days with more missing timestamps)
    perc = int(raw_input("enter threshold % value for tolerating missing timestamps (put 4(?) as default): )
    #  D. print dates with less no. of entries
    entries[entries.Date < (max(entries.Date) - int(max(entries.Date) * perc/100))]
    #  E. remove the above dates from dataframe mc 
    less = entries[entries.instances >= (max(entries.instances) - int(max(entries.instances) * perc/100))]
    mc = pandas.merge(mc, less, how='inner', on='Day')  # pd.merge(dfA, dfB, how='inner', on=['S', 'T'])
    # https://pandas.pydata.org/pandas-docs/stable/merging.html
    
    #(the following don't work if the exact frequency is not maintained throughout dataframe)
    #mc = mc.set_index('Date')  #from datetime import datetime  #mc['Date'] = pandas.to_datetime(mc['Date'])  #mc['Date'] = pandas.to_datetime(mc.Date)   
    #mc = mc.loc[pandas.date_range(mc.index.min(), mc.index.max(), freq='6min')]  #mc.reindex(pandas.date_range(start=mc.index[0], end=mc.index[-1], freq='6min')) 
    # https://stackoverflow.com/questions/19324453/add-missing-dates-to-pandas-dataframe
    # https://stackoverflow.com/questions/40419060/search-missing-timestamp-and-display-in-python
    # https://stackoverflow.com/questions/42673064/adding-missing-time-in-pandas-dataframe
    ### check difference b/w successive time stamps to locate gaps????
    
    return mc;

                         

## handle NAs & negative values in cols that should be non-negative
def DataCleaning2(df):
    "data cleaning for NAs & negative values"
    
    ## step 1: dealing with NA values
    # A. set NA to zero for selected cols  (google: pandas fillna multiple columns)
    df.update(df[['HeadCount','OKcount']].fillna(0))
    
    # B. remove rows where the important cols have 'NA' 
    df = df[df['ID'].notnull() & df['TotalCount'].notnull() & df['regression_var'].notnull() ]
    # https://stackoverflow.com/questions/42125131/delete-row-based-on-nulls-in-certain-columns-pandas
    # google: how to remove rows with NA values in selected columns in python dataframe
    #df.dropna(axis=0, how='any') # axis=0 indicates rows change, other wise rows with NA in other cols will also be removed
    #df.dropna(axis=0, subset=[['ID', 'TotalCount', 'regression_var']], how='any')
    #df = df[pandas.notnull(df['ID', 'TotalCount', 'regression_var'])]
    
    ## step 2: remove data for dates which have multiple entries (depends on the use case)
    ## for this, combine the date, index & ID column, & either remove all duplicates/ or retain unique rows
    #df = df.drop_duplicates(subset=['Date', 'index', 'ID'], keep=False)
    
    ## step 3: remove rows with negative value in OK count
    df = df[df.OKcount >= 0]
       
    ## remove any unnecessary columns
    #https://stackoverflow.com/questions/28035839/how-to-delete-a-column-from-a-data-frame-with-pandas
    df.drop('Col_not_required', axis=1, inplace=True)
    
    return df;
                         
                         
                         
#### aggregate data from individual time stamps to daily values  ####
def aggregateData(mc):
    "aggregate data by day"
    # Step 1: format DateTime colunm to only date
    from datetime import datetime
    mc['Day'] = [d.date() for d in mc['Date']]  # d.date() for date & d.time for time part
                         
    # step 2: aggregate values by day , & return as df                         
    agg_df = mc.groupby('Day').agg({'selected_col_value': 'sum'}).reset_index()  # df = mc.groupby('Day').agg({'selected_col_value': 'sum'})
  
    return agg_df;
        


# sample data
# dateTime   Index  TotalCount                         
# 17-01-01      A      20
# 17-01-01      B      10   
# 17-01-02      A      17
# 17-01-03      A      13
# 17-01-03      B      16
# .....   
# for each index, I can count by: 1. sum of values in the TotalCount col. (e.g. A -> 50)
#                                 2. sum of no. of instances (e.g. A -> 3)

                         
#### find aggregates by indices  ####
def index_aggregate(df):
    ### method 1: by column total  ###
    index_TotalCount = df.groupby('Index').agg({'TotalCount': 'sum'}).reset_index()
    index_TotalCount['perc_count'] = 100 * index_TotalCount['TotalCount']/index_TotalCount['TotalCount'].sum()
    index_TotalCount = index_TotalCount.sort_values('TotalCount', ascending=False)   ## sorted by total  
                         
    ### method 2: by instances of index
    entries = pandas.DataFrame(pandas.value_counts(df['Index']) * 100/df.shape[0]).reset_index()
    entries = entries.rename(columns={'Index': 'perc_instances', 'index': 'Index'})
    # following is an alternate way to count instances/ no. of rows of each index
    #entries = mc.groupby('Index').agg({'Date': 'count'}).reset_index()
    #entries = entries.rename(columns={'Date': 'instances'})
                         
    ## join both columns
    index_TotalCount = pandas.merge(index_TotalCount, entries, how='inner', on='Index')
                         
    ## selecting 1st & last row of df
    # index_TotalCount.iloc[[0, -1]]
    
    return index_TotalCount;
                         
                         

#### calculate cumulative values & sort ####
def cal_cumulative(df):
    ## total for each index
    index_TotalCount = df.groupby('Index').agg({'TotalCount': 'sum'}).reset_index()
    #index_TotalCount = pandas.DataFrame(df.Index.unique()).reset_index()
        
    ## calculate cumulative of percentage volume
    index_TotalCount['cum_sum'] = index_TotalCount['TotalCount'].cumsum()
    index_TotalCount['cum_perc'] = 100 * index_TotalCount['cum_sum']/index_TotalCount['TotalCount'].sum()
                         
    ## sort by total
    index_TotalCount = index_TotalCount.sort_values('TotalCount', ascending=False)
                         
    ## save indices above a threshold as list
    threshold = int(raw_input("enter % of cumulative total to select top indices (put 50 as default): "))
    topMost = index_TotalCount[ index_TotalCount.cum_perc <= threshold ]
    #top_produced
    # save the topmost indices as a list
    top_list = topMost['Index'].tolist()

    return index_TotalCount;   
                         
                         

### calculate median by group  ####
def getMedian(df):
    new_df = df.groupby('Index').agg({'regression_var': 'median'}).reset_index()    
    return new_df;
                         
                         
                         
####################################################################################################                         
####################################################################################################
### read in data
filepath = raw_input("enter fullPath & filename: ")
mc = pandas.read_excel(filepath)    
                         
### data cleaning
mc = dataCleaning(mc)                        
                     
                         
####################################################################################################                         
##### binning data #####
## step 1: express data as fraction of highest value/ or some pre-determined value
pre_val = int(raw_input("enter pre-determined value, as a fraction of which other values sould be expressed : ")) 
#check class of the variable 'rated' : ensure numeric input
# type(pre_val)  
mc['Frac'] = mc['raw_col_data'] * 100/pre_val

## step 2: factor data into bins, & add a separate column with bin label
mc['fac'] = pandas.cut(mc['Frac'], [-100, 0, 30, 80, 100, 500], labels=['0', 'L','M','H', 'vH']) 
#https://chrisalbon.com/python/pandas_binning_data.html  
#categories = pandas.cut(df['column_tobe_factored'], bins, labels=group_names) 
                         
## step 3: count instances in each bin: this is calculated as (#occurrences in each bin/ total #data points)
# The count is affected by missing data points.
counts = pandas.DataFrame(pandas.value_counts(mc['fac']) * 100/mc.shape[0]) 
# mc.shape[0] counts num_rows in mc; mc.shape[1] counts num_cols in mc
counts = counts.reindex(["0","L","M","H","vH"])                         
print "data in bin H (>80 to <=100%) ",counts.loc["H"]['fac'],"% of time"   
                         
                         
#######################################################################################################
##### check if data are above a threshold, subset & calculate % of such data points
threshold = int(raw_input("enter threshold % value (put 2 as default): "))
b_H = mc[(mc.fac == 'H') & (mc.selected_col_value > threshold) ]
print "col value >",threshold,"% in bin H for ", len(b_H)*100/len(mc),"% of time "

                 
#####################################################################################################                         
#######  PLOTS  ########                         
## plot 1: all data vs time
mc['Date'] = pandas.to_datetime(mc.Date)
pyplot.figure(figsize=(14,2), dpi=100) # (width, height)
pyplot.plot(mc['Date'], mc['selected_col_value'], linewidth=0.1, color='black')
pyplot.axhline(y=100, linewidth=1, color='r')
pyplot.ylabel('colname (units)')
pyplot.show()                         
                         

## plot 2: aggregated data vs time                         
agg_df = aggregateData(mc)
pyplot.figure(figsize=(14,2), dpi=100) # (width, height)
agg_df.plot(linewidth=0.5, color='black')  #pyplot.plot(mc['Date'], mc['Frac'], linewidth=0.1, color='black')
#new_df.plot(kind='bar', color='black', alpha=0.5)  #width=0.5, align='center',
pyplot.show()                        
                         
                         
# plot 3: aggregated data vs day of week (as boxplot)
from datetime import datetime
agg_df['Day'] = pandas.to_datetime(agg_df['Day'])  # 'day' has the date value from dateTime
agg_df['day_of_week'] = agg_df['Day'].dt.weekday_name
# https://stackoverflow.com/questions/30222533/create-a-day-of-week-column-in-a-pandas-dataframe-using-python
# https://stackoverflow.com/questions/28009370/get-weekday-day-of-week-for-datetime-column-of-dataframe
# CODE FOR BOXPLOT
                         
                         
# plot 5: plot binned data as bars
counts = pandas.DataFrame(pandas.value_counts(mc['fac']) * 100/mc.shape[0]) 
# mc.shape[0] counts num_rows in mc; mc.shape[1] counts num_cols in mc
counts = counts.reindex(["0","L","M","H","vH"])
ax = counts.plot(kind='bar', width=0.5, color='black', align='center', alpha=0.5)
ax.set_ylabel('% of total time')
ax.set_xlabel('bin level')
ax.legend_.remove()
pyplot.show()                         
                         
                         
                                                    
                         
######################################################################################################
