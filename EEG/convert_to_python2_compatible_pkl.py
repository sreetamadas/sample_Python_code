## convert pickle files from higher version to python 2 compatible version

import pandas as pd
import numpy as np
import os
import pickle

file_location='/home/Desktop/EEG/raw_data' 
os.chdir('/home/Desktop/EEG/raw_data') 

files=os.listdir(file_location)
file_list = [f for f in files if f.endswith('.pickle')]   # .txt  # .csv

for file_val in file_list:
    new = str(file_val[0:8]) + str('_raw_python2.pkl')
    
    dfull = pickle.load(open(file_val, 'rb'))
    pickle.dump(dfull, open(new,'wb'), protocol=2)
