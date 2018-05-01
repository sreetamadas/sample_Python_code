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


print "Hello World"   # 'print' adds a default NEWLINE. place a ',' to suppress this
#string="Hello world"  # place string inside ""
#print string

#num = raw_input("enter num: ")
#print "you entered ", num
#doubled = int(num) * 2  # if int() is not mentioned, the program prints the string '7' in num twice, giving 77
#print "double ", doubled

######################################################################################
### file opening & handling  ###
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
dat= pandas.read_excel(filepath)  
# alternate way of path input, hardcoded
dat = pandas.read_csv("C:\\Users\\username\\Desktop\\data\\sampleData.csv")   


########################################################################################################
## running checks on the input data as dataframe

# missing data
print(df.isnull())
print(df.notnull())

# count no. of rows with missing values
print(df.isnull().sum())

# code for adding missing values (median/mean or fill using previous value or next value) 
# https://pandas.pydata.org/pandas-docs/stable/missing_data.html
df = df.fillna(method='pad', limit=3)    # using previous value (up to 3 places)
df = df.fillna(method='bfill')  # backfill: fill using next value  ; ffill= using previous value

# df.shape[0] counts num_rows in df; df.shape[1] counts num_cols in df
df.shape

## checking types of data in different cols 
df.dtypes
df.info()
df.describe()
df.value_counts()


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


## time & date format
## format time stamp, in case of non-standard format; 
## non-standard format of datetime can be ascertained by checking the O/P for datetime col in mc.dtypes
df["Date"] = pandas.to_datetime(df["Date"], format="%Y.%m.%d")  ## specify the format from input data here
# change format
df['Date'] = df['Date'].dt.strftime("%Y-%m-%d")

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


t = []
list = [1,2,3,4,5,6,7,8,9,10]
for i in list:
    t.append(i)
t = pandas.DataFrame(t) #, columns='a')
len(t.index)

##############################################################################################################
##### if loop  ####
#if 1>0.9:  # the condition may be put inside ()
#	print "nonsense"
#elif expression2:
#	list-of-codes
#else:
#	print "1 < 2"


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


#############################################################################################################
## NOTE: python codes are delimited by indentation. indentation should be same for all codes in a block (else, error)
## uses 4 spaces instead of tab at each indentation level

