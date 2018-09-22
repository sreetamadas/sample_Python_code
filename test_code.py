#!/usr/bin/python
" this is a test module"
import warnings

# check python version
import sys
sys.version


## set directory
import os
os.chdir('c:\Users\username\Desktop\data\code_python')   ### path is Windows
print os.getcwd()
##########################################################################################################

print "Hello World"   # 'print' adds a default NEWLINE. place a ',' to suppress this
#string="Hello world"  # place string inside ""
#print string

#num = raw_input("enter num: ")
#print "you entered ", num
#doubled = int(num) * 2  # if int() is not mentioned, the program prints the string '7' in num twice, giving 77
#print "double ", doubled

######################################################################################
### step 1: file opening & handling  ###
#  this reads the entire file first, closes the file & then iterates through the lines in the file
#  this permits file access to complete quickly, no back & forth b/w reading & printing a line
#  this won't work for very large data files
filename = 'fileABC'  
# filename = raw_input("enter filename: ")
FileIN = open(filename, 'r')  # 'r' is access mode (read-only); use w for write, a for append;  'FileIN' is the filehandle
for Line in FileIN:
	print Line,
FileIN.close()


## read in data
import pandas
filepath = raw_input("enter fullPath & filename: ")  # C:\Users\username\Desktop\data\sampleData.xls
dat= pandas.read_excel(filepath, 'Sheet1')  
# alternate way of path input, hardcoded
dat = pandas.read_csv("C:\\Users\\username\\Desktop\\data\\sampleData.csv")   

## see the data
print(dat.head(n=5))

# write to a file
df1.to_csv("modified_data.csv", sep=',')

########################################################################################################
## step 2: running checks on the input data as dataframe

## checking types of data in different cols 
df.dtypes
df.info()

## get summary statistics
df.describe()
df['colname'].value_counts()

## df.shape[0] counts num_rows in df; df.shape[1] counts num_cols in df
df.shape


## time & date format
## format time stamp, in case of non-standard format; 
## non-standard format of datetime can be ascertained by checking the O/P for datetime col in mc.dtypes
df["Date"] = pandas.to_datetime(df["Date"], format="%Y.%m.%d")  ## specify the format from input data here
# change format
df['Date'] = df['Date'].dt.strftime("%Y-%m-%d")

# adding an interval to a time stamp
df["dateUTC"] = pd.to_datetime(df["dateUTC"]) #, format="%Y.%m.%d")  ## specify the format from input data here
from datetime import datetime, timedelta
df['dateTime'] = df['dateUTC'] + timedelta(minutes=330)  ## fix dateTime to IST from UTC : add 5 hrs 30 min

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


# missing data
print(df.isnull())
print(df.notnull())

# count no. of rows with missing values
print(df.isnull().sum())

# code for adding missing values (median/mean or fill using previous value or next value) 
# https://pandas.pydata.org/pandas-docs/stable/missing_data.html
df = df.fillna(method='pad', limit=3)    # using previous value (up to 3 places)
df = df.fillna(method='bfill')  # backfill: fill using next value  ; ffill= using previous value



#### rename columns
oldname = raw_input("enter 1st columnname: ")  # AL1
df = df.rename(columns={oldname: 'newname'})


### drop unnecessary cols
# method 1
df.drop('Col_not_required', axis=1, inplace=True)
# method 2
del df['col_not_reqd']


####  remove all rows with duplicate values in selected cols
df_single = df.drop_duplicates(subset=['Date', 'Shift', 'Index'], keep=False)

####  get all rows with duplicate values in selected cols
df_multi = pandas.concat(g for _, g in df.groupby(['Date', 'Shift', 'Index']) if len(g) > 1)


########################################################################################################
## assigning data from one df to another
new_df.loc[0, 'dateTime'] = df.loc[(rownum), 'dateTime']


## extract portion from dateTime
df.loc[row_index, 'hr'] = df.loc[row_index, 'dateTime'].hour

# calculate difference/ interval b/w time stamps
#df.loc[row_index, 'timedel'] =  (df.loc[row_index, 'dateTime'] - df.loc[(row_index-1), 'dateTime']) #.astype('timedelta64[m]')
df.loc[row_index, 'timedel'] =  pd.Timedelta(pd.Timestamp(df.loc[row_index, 'dateTime']) - pd.Timestamp(df.loc[(row_index-1), 'dateTime'])).total_seconds()/60 #.astype('timedelta64[m]')
 

#######################################################################################################
###  arrays in python: list => [] , tuple => ()  ####
# list can be updated, tuple is read-only list
list1 = [3, 5, 7, 9]
print list1, "'print full array / list'"
#print list1[0], "'prints 1st element, numbering starts from 0'"
#print list1[2:], "'print all elements, 3rd position (0,1,2) and beyond'"
#print list1[:2], "'print all elements before second position'"


### hash in python is called dictionary or dict ###
# keys can be numbers or strings, values can be any python object
# for the example below, it prints the key a, then 1, then b - WHY? what is the sorting order?
dict1 = {}
dict1['a'] = 80  # this in an integer assignment
dict1['b'] = '80'  # this is a string asignment
dict1['1'] = 'four'
for key in dict1:
	print key, dict1[key]
	num = dict1[key] * 2
	#num = int(dict1[key]) * 2 # this gives error with the asignment 'four'
	#print "num: ", num

## alternate way to print keys of a dictionary
print(dict1.keys())	
	
## appending to a list ; also, see append vs extend in python	
t = []
list = [1,2,3,4,5,6,7,8,9,10]
for i in list:
    t.append(i)
t = pandas.DataFrame(t) #, columns='a')
len(t.index)


backup = df[:rownum]    # first rownum rows are saved in backup
for i in range(rownum, len(df.index) ):   # running for loop on certain rows of df
	#rest of the code ...
	backup = backup.drop(backup.head(1).index) #, inplace=True)  # remove row from top
        backup = backup.append(df.iloc[[i]]) #, axis=0)  ## add row at end



##############################################################################################################
##### if loop  ####
#if 1>0.9:  # the condition may be put inside ()
#	print "nonsense"
#elif expression2:
#	list-of-codes
#else:
#	print "1 < 2"


df.loc[row_index, 'X2'] = np.where(df.y1 - df.y2 > 5, 1, 0)  # if the diff>5 , x2=1; else x2 = 0-
# https://stackoverflow.com/questions/19913659/pandas-conditional-creation-of-a-series-dataframe-column
# see this link for series of if-else conditions


#######  while loop  #####
#while expression:
#	execute these commands


########  for loop  ######
list1 = [3, 5, 7, 9]
for item in list1:  # this is equivalent to foreach in perl rather than for($i=0; $i<=$#array; $i++)
  print item,
#print

## using range to make this behave as a conventional for loop
list1 = [3, 5, 7, 9]
for i in range(4):
	print list1[i],
#print
# 'print' adds a default NEWLINE. place a ',' to suppress this

hash1 = {'a':80, 'b':'80', '1':'four'}
for item in hash1:
  print item,
#print


######################################################################################
###  managing exceptions to a block of code  #####
#  the section following 'try' involves the code to be managed
#  the section following 'except' tells what to do in case of an exception
#try:
#	filename = 'readfile1'
#	FileIN = open(filename, 'r') 
#	for Line in FileIN:
#		print Line,
#	FileIN.close()
#except IOError, e:
#	print 'file open error:', e


#######################################################################################
##### declaring functions #####
 def function_name (comma-separated list of arguments):
	"optional documentation string"
	block of code

#def addMe2Me (x):
#	"apply + operation to argument"
#	return (x + x)
#x = [1, 'a']  # 'abc'
#y = addMe2Me(x)
#print y


#####################################################################################
### using string format ###
#who = 'birds'
#what = 'chi'
#print 'we are the %s who say %s' % (who, ((what + ' ') * 3) )


## some built-in functions ###
# int(obj)	convert object to integer
# str(obj)	convert object to string
# type(obj)	return type of object
# len(obj)	return length of object
# range([start,]stop[,step])	
# gives list of integers, begin at 'start' up to (but not including) 'stop' in increments of 'step'; default => start=0, step = 1


###############################################################################################################
import pandas as pd
#import numpy as np
import os

# create a list with full paths of all files from different locations
main_list = []

def finalList(L, main_list):
    files =  os.listdir(L)
    f_list = [f for f in files if f.endswith('matlab_format.txt')]
    
    for f in f_list:
        main_list.append(L+'/'+f)
    
    return main_list
    
l1 = 'C:\\Users\\Desktop\\data\\folder1'
l2 = 'C:\\Users\\Desktop\\data\\folder2'
l3 = 'C:\\Users\\Desktop\\data\\folder3'
l4 = 'C:\\Users\\Desktop\\data\\folder4'

main_list = finalList(l1, main_list)
main_list = finalList(l2, main_list)
main_list = finalList(l3, main_list)
main_list = finalList(l4, main_list)


## copy the files from different locations to a common location:
import shutil
#CK = df[:2]
for i in range(0, df.shape[0]):  # df has names of files to be copied
    name = str(df.iloc[i, 0])
    print(name)
    
    # extracting pattern from list
    # https://stackoverflow.com/questions/16304146/in-python-how-do-i-extract-a-sublist-from-a-list-of-strings-by-matching-a-strin
    orig = [x for x in main_list if name in x]
    #f = str(name)+'.txt'
    shutil.copy2(orig[0], 'C:\\Users\\Desktop\\data\\db\\')
    # https://stackoverflow.com/questions/123198/how-do-i-copy-a-file-in-python




#############################################################################################################
## NOTE: python codes are delimited by indentation. indentation should be same for all codes in a block (else, error)
## uses 4 spaces instead of tab at each indentation level

