from numpy.random import choice
import random
from datetime import datetime, timedelta, time
import pandas as pd


def defect_random():
    defects = ['Good'] + ['Crazing'] + ['Inclusion'] + ['Pitted Surface'] + ['Patches'] + ['Rolled-in'] + ['Scratches']
    new_d = choice(defects, 1, p=[0.95, 0.01, 0.005, 0.005, 0.003, 0.007, 0.02])
    return new_d[0]

def random_machine():
    mach_id = ['Z1123420-A','Y11123420-B','Z1123421-A','Y1123421-B']
    return random.choice(mach_id)

    


init_time = datetime(2020,9,7, 00,00,00)
temp = init_time.strftime("%H:%M:%S")

data = []

for i in range(1,17281): # length of table
    # defect and machine
    defect = defect_random()

    machine = random_machine()
    machineStr = str(machine)

    #defect num
    numOfDefects = 0
    numOfDefects = 1 if defect != 'Good' else 0   ## number of defects for a timestamp

    #time function
    time_str = init_time.strftime("%H:%M:%S")
    date_str = init_time.strftime("%Y/%m/%d")
    init_time += timedelta(seconds=5)

    #shift id
    shift = 3 if init_time.time() <= time(8,0,0) else 2 if init_time.time() <= time(16,0,0) else 1 ## assigning shift based on time for all 4 machines

    #line_num
    line = 'A-1' if machineStr[-1] == 'A' else 'B-1' ##assigning linenumber based on last character of machine_id
    
    #insert statement
    data.append([date_str,time_str,line,machine,'N/A',defect,shift,numOfDefects])


data = pd.DataFrame(data)
