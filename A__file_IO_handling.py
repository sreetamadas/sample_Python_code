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


######################################
import os
## get list of all directories in jan_2019_cyient
filedir = os.listdir(file_location)

def find_files( files, dirs=[], extensions=[]):
    new_dirs = []
    for d in dirs:
        try:
            new_dirs += [ os.path.join(d, f) for f in os.listdir(d) ]
        except OSError:
            if os.path.splitext(d)[1] in extensions:
                files.append(d)

    if new_dirs:
        find_files(files, new_dirs, extensions )
    else:
        return


for Dir in filedir:
    #print(Dir)
    files = []
    find_files( files, dirs=[Dir], extensions=['.txt'] )
    
    print(files[0])
    f=open(file_location + files[0])

	
################3
#reading a file
for file_val in file_list:
    loc=file_location+'/'+file_val
    f=open(loc)
    data=f.read()
    data_new=data.splitlines()
    dat=[]
    for k in range(len(data_new)):


######################
# file copy
import shutil
#CK = df[:2]
for i in range(0, df.shape[0]):  # CK.shape[0]
    name = str(df.iloc[i, 0])
    print(name)
    
    # extracting pattern from list
    # https://stackoverflow.com/questions/16304146/in-python-how-do-i-extract-a-sublist-from-a-list-of-strings-by-matching-a-strin
    orig = [x for x in main_list if name in x]
    #f = str(name)+'.txt'
    shutil.copy2(orig[0], 'full_path_of 2nd folder')
    # https://stackoverflow.com/questions/123198/how-do-i-copy-a-file-in-python




