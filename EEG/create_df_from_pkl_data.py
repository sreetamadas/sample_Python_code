#### read in pickle files of raw data & create df - for train-test with CNN ####

import pandas as pd
import numpy as np
import os
import pickle

###############################################################
file_location='/home/Desktop/EEG/raw_data' 
os.chdir('/home/Desktop/EEG/raw_data') 

files=os.listdir(file_location)
file_list = [f for f in files if f.endswith('.pickle')]   # .txt  # .csv

############################################################################
def get_PSD_from_pickle(pid,day,x):
    df = []
    eeg_class = []
    epochnum = []

    
    for k in list(x['wake'].keys()):
        label = 'wake'  #seg = k.split('e')[0]
        epoch = k
        df.append(x['wake'][k])
        eeg_class.append(label)
        epochnum.append(k)

    for k in list(x['sleep_stage_1'].keys()):
        label = 'sleep_stage_1'   #seg = k.split('e')[0]
        epoch = k
        df.append(x['sleep_stage_1'][k])
        eeg_class.append(label)
        epochnum.append(k)
            
    for k in list(x['sleep_stage_2'].keys()):
        label = 'sleep_stage_2'   #seg = k.split('e')[0]
        epoch = k
        df.append(x['sleep_stage_2'][k])
        eeg_class.append(label)
        epochnum.append(k)

        
    df = pd.DataFrame(df)  ## also, add subject ID
    df['pID'] = pid
    df['day'] = day
    eeg_class = pd.DataFrame(eeg_class, columns=['class_label'])
    epochnum = pd.DataFrame(epochnum, columns=['epoch'])
    
    df = pd.concat([df,epochnum,eeg_class],axis=1)
    
    return df
    
################################################################
dfull = pd.DataFrame()

for file_val in file_list:
    #print(file_val)
    patient = file_val[2:5]
    day = file_val[5:6]
    
    with open(file_val, 'rb') as f:
        print(file_val)
        x = pickle.load(f)      
        dat = get_PSD_from_pickle(patient,day,x)
    f.close()
        
    dfull = pd.concat([dfull, dat],ignore_index=True)
    
####################################################################
## save data ##
pickle.dump(dfull, open('raw_EEG_all_patients_p2.pkl','wb'), protocol=2)

pickle.dump(dfull, open('raw_EEG_all_patients.pkl','wb'))
