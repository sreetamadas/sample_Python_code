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

