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

df.to_excel("C:\\Users\\Desktop\\data\\outfile.xlsx")

############################################################################################################################

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

# push the keys into a list
list(dict1.keys())

	
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
####  if loop  ####
#if 1>0.9:  # the condition may be put inside ()
#	print "nonsense"
#elif expression2:
#	list-of-codes
#else:
#	print "1 < 2"

# re-assign values in a column based on certain conditions
df_cleaned['status'] = [2 if 'Under' in x else 1 if 'Over' in x else 0 for x in df_cleaned['status']]


df.loc[row_index, 'X2'] = np.where(df.y1 - df.y2 > 5, 1, 0)  # if the diff>5 , x2=1; else x2 = 0
df['truth_healthy_or_not'] = np.where( (df['status_groundTruth'] == '1'), '1', '0' )
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


## exiting or breaking out of a loop
# https://www.digitalocean.com/community/tutorials/how-to-use-break-continue-and-pass-statements-when-working-with-loops-in-python-3
# https://stackoverflow.com/questions/19747371/python-exit-commands-why-so-many-and-when-should-each-be-used
try:
   loaded_model, coef_version = pickle.load(open(filename, 'rb'))
except:
   OverflowError
   value = 0
   value2 = 0
   f='status=0, value=' + str(int(value)) + ',Index = ' + str(round(np.median(value2),2))
   filename='Output.txt'
   filevals= open(file_location+'/'+filename,'w')
   filevals.write(f)
   filevals.close()  
   continue  # continue outside this block         
   #raise SystemExit
   #break



#############################################################################################################
## NOTE: python codes are delimited by indentation. indentation should be same for all codes in a block (else, error)
## uses 4 spaces instead of tab at each indentation level

