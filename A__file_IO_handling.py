## set directory
import os
os.chdir('c:\Users\username\Desktop\data\code_python')   ### path is Windows
print os.getcwd()


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



files=os.listdir(file_location)
file_list = [f for f in files if f.startswith('PPG_input')]   


# write to a file
df1.to_csv("modified_data.csv", sep=',')
df.to_excel("C:\\Users\\Desktop\\data\\outfile.xlsx")

