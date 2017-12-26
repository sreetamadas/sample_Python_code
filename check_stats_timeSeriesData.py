# get stats on time series data (from sensor)

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
## express data as fraction of highest value/ or dome pre-determined value
pre_val = int(raw_input("enter pre-determined value, as a fraction of which other values sould be expressed : ")) 
#check class of the variable 'rated' : ensure numeric input
# type(pre_val)  
mc['Frac'] = mc['raw_col_data'] * 100/pre_val
