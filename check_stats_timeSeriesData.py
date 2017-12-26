#### get stats of time series data (from sensor)

## set directory

## read in data
import pandas

filepath = raw_input("enter fullPath & filename: ")
mc = pandas.read_excel(filepath)

#########################################################################################################
## data cleaning
def dataCleaning(mc):
    "CHECK MISSING VALUES: individual cols & time stamps"
    
    ## check data types
    #mc.dtypes 
        
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

####################################################################################################
##### binning data #####
## step 1: express data as fraction of highest value/ or some pre-determined value
pre_val = int(raw_input("enter pre-determined value, as a fraction of which other values sould be expressed : ")) 
#check class of the variable 'rated' : ensure numeric input
# type(pre_val)  
mc['Frac'] = mc['raw_col_data'] * 100/pre_val

## step 2: factor data into bins
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
                           
                         
######################################################################################################
##### aggregate data from individual time stamps to daily values
# Step 1: format DateTime colunm to only date
from datetime import datetime
mc['Day'] = [d.date() for d in mc['Date']]  # d.date() for date & d.time for time part
                         
# step 2: aggregate values by day , & return as df                         
new_df = mc.groupby('Day').agg({'selected_col_value': 'sum'}).reset_index()  # df = mc.groupby('Day').agg({'selected_col_value': 'sum'})
  
                         
                         
                         
                         
                         
