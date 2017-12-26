####################################################################################################
##### OPTIMISE cost of production (of items on different machines), given time & demand  #######

import pandas #as pd
import numpy as np

# from statistics import median      ## no module named statistics in the installed version
import pulp
# alternate to pulp
# https://github.com/SCIP-Interfaces/PySCIPOpt

import itertools  # to use only those indices corresponding to model numbers present in demand
####################################################################################################

# calculate threshold production X (where Y: cost per piece)
from sklearn import linear_model
import math

# see function definition in get_threshold_kneeCurve.py ->    def thresholdX(temp):

#####################################################################################################

# read & clean production data
dat = pandas.read_csv("C:\\Users\\Desktop\\data\\production_data.csv")   
dat["Date"] = pandas.to_datetime(dat["Date"], format="%d.%m.%Y")
dat = DataCleaning(dat)

# find lists of all mcs & models
machineList = dat['Machine'].unique()
modelList = dat['ModelNo'].unique()  

####################################################################################################

# separate input data into training & testing sets
# test: 
dat['Date'] = dat['Date'].dt.strftime("%Y-%m-%d")
test = dat[(dat['Date'] < '2016-10-08')]    # df[(df['date'] > '2013-01-01') & (df['date'] < '2013-02-01')]

# train: remainder
prdcn = dat      # dat[(dat['Date'] >= '2016-10-08')]

######################################################################################################

# estimate demand
# method 1: build forecasting model, & predict demand
# https://petolau.github.io/Ensemble-of-trees-for-forecasting-time-series/?lipi=urn%3Ali%3Apage%3Ad_flagship3_feed%3BofqPdmLgQkiKXIvHBbktqQ%3D%3D

# method 2:
# using 1 week's production data as demand
production = test.groupby('ModelNo').agg({'TotalProductPcs': 'sum'}).reset_index()

#######################################################################################################

### calculate i) Y  &  ii) production time, per machine-model pair

# get shifts with single model production

# declare dictionary for storing threshold production & median Y (cost/pc)
Q_thres = [] 

# the modelList should contain all models in training data, as well as in demand data, if not the same in both cases
for w in modelList:
    for mc in machineList:
        # subset data
        temp = prdcn_single.loc[(prdcn_single.Machine == mc) & (prdcn_single.ModelNo == w)]
        
        # set NA to zero for relevant cols  (google: pandas fillna multiple columns)
        temp.update(temp[['TotalStopTime']].fillna(0))
        
        # find threshold production
        # final threshold can be the full matrix (mc-model), or minimum value in the matrix
        q = thresholdQ(temp)
        #Q_thres.append([mc, w, q])
                
        # calculate median cost over production values > threshold production  ## & < (threshold + 20)
        #    if there are too few data points for a mc-model pair, should Y be set to zero? - NOT USED
        temp = temp[temp.TotalProductPcs > q]
        
        # calculate time/pc for wheel production - 1 value for a wheel across all mc, or for each mc/model pair?
        #temp['timePerPc'] = 8 * 60/temp.TotalProductPcs  # in minute
        # include stop -time
        temp['timePerPc'] = (8 * 60 - temp.TotalStop_minute)/temp.TotalProductPcs  # in minute
        
        #if len(temp.index) > 0:  ### if len(temp.index) > 10:  # if #shifts(production-above-threshold) > 10
        Q_thres.append([mc, w, q, np.median(temp.Y), np.median(temp.timePerPc)])
        
Q_thres = pandas.DataFrame(Q_thres, columns=['machineID','variantID', 'Q_threshold', 'median_cost', 'timePerPc'])


# for models which don't have sufficient shift data to calculate the above metrics 
# take the highest value on each mc, & assign to the mc-new model pair
for mc in machineList:
    max_Y = Q_thres.ix[Q_thres.machineID == mc]['median_cost'].max()
    Q_thres.update(Q_thres[Q_thres.machineID == mc].median_cost.fillna(max_Y))

## calculate a median threshold across all mcs
med_Qthres = np.median(Q_thres.Q_threshold[~np.isnan(Q_thres.Q_threshold)])


### calculate median time/pc for a model from data for all machines  ###
medTimePerPc = []
for variantID in modelList: ## or on original modelList in Q_thres
    medTimePerPc.append([variantID, np.median(Q_thres.ix[(Q_thres.variantID == variantID)].timePerPc[~np.isnan(Q_thres.ix[(Q_thres.variantID == variantID)].timePerPc)])])
medTimePerPc = pandas.DataFrame(medTimePerPc, columns={'medTimePerPc', 'variantID'}) #.fillna(6)

# for models which do not have data to calculate median time, fill with the maximum value of median_timePerPc
max_TimePerPc = medTimePerPc['medTimePerPc'].max()
medTimePerPc.update(medTimePerPc['medTimePerPc'].fillna(max_TimePerPc))

# some shifts show full shifttime as show stopTime, hence coors time/pc = 0
# these are re-assigned through the following module
for i in range(len(medTimePerPc)):
    #medTimePerPc.ix[i,'medTimePerPc'] = 5.0 
    if medTimePerPc.ix[i,'medTimePerPc'] == 0:
        medTimePerPc.ix[i,'medTimePerPc'] = max_TimePerPc



###################################################################################################################

## get stop_time & stop_cost
# get shifts with multi-model production
prdcn_multi = getMultiModelShiftData(prdcn)

# set 'NA' values to '0' in the columns: TotalStop_cost , TotalStop_minute
prdcn_multi.update(prdcn_multi[['TotalStop_cost','TotalStop_minute']].fillna(0))

# calculate median stop time
stop_time = np.median(prdcn_multi.TotalStop_minute)

# calculate median cost during stop time
prdcn_multi['TotalStop_cost'] = prdcn_multi['TotalStop_cost'].str.replace(',', '')
# https://stackoverflow.com/questions/38516481/trying-to-remove-commas-and-dollars-signs-with-pandas-in-python
prdcn_multi['TotalStop_cost'] = prdcn_multi['TotalStop_cost'].astype(float)
prdcn_multi.update(prdcn_multi[['TotalStop_cost']].fillna(0)) 
stop_cost = np.median(prdcn_multi.TotalStop_cost)

#################################################################################################################

scheduleFor = 7*24*60  # in mins
variants = production.ModelNo  # list of variants in weekly demand
demand = production.TotalProductPcs
machines = machineList

##################################################################################################################
####################################################################################################################

# declare the optimisation problem
opmodel = pulp.LpProblem("cost optimization", pulp.LpMinimize)


# declare variables : numbers Nij
num_pieces = pulp.LpVariable.dicts("Number_pieces", [tuple(pair) for pair in Q_thres[['machineID','variantID']].as_matrix()],
                                   lowBound = 0, cat = pulp.LpInteger)
                                   

# declare Xij
produceOrNot = pulp.LpVariable.dicts("isProduced", [tuple(pair) for pair in Q_thres[['machineID','variantID']].as_matrix()],
                                   cat = pulp.LpBinary)



# Objective Function : two parts = production cost + cost of stopTime
'''
# method 1: this has all models in input list
opmodel += (
    pulp.lpSum([
        Q_thres.ix[i]['median_cost'] * num_pieces[(Q_thres.ix[i]['machineID'],Q_thres.ix[i]['variantID'])] for i in range(len(Q_thres))]
               + [stop_cost*produceOrNot[(Q_thres.ix[i]['machineID'],Q_thres.ix[i]['variantID'])]
                                      for i in range(len(Q_thres))]
               ))
'''

# method 2: using models only in demand list, to reduce computation
opmodel += (
    pulp.lpSum([
        Q_thres.ix[(Q_thres.machineID == machineID) & (Q_thres.variantID == variantID)]['median_cost'] * num_pieces[(machineID, variantID)] 
            for (machineID, variantID) in list(itertools.product(machines, variants))]
               + [stop_cost*produceOrNot[(machineID, variantID)]
                                      for (machineID, variantID) in list(itertools.product(machines, variants))]
               ))



# Constraint on numbers: production should meet demand of each variant
for i, variantID in enumerate(variants):
    opmodel += pulp.lpSum([num_pieces[machineID,variantID] for machineID in Q_thres[Q_thres['variantID'] == variantID]['machineID']]) >= demand[i]



# constraint for total production time on each machine: two parts = production time + stopTime for model change
for i, machineID in enumerate(machines):
    opmodel += pulp.lpSum([medTimePerPc.ix[medTimePerPc['variantID'] == variantID]['medTimePerPc']*num_pieces[machineID,variantID] for variantID in variants]  + 
                         [stop_time*produceOrNot[machineID,variantID] for variantID in variants] +
                         [-stop_time]
                         ) <= scheduleFor



# Indicator function
# constraint on Xij to consider stop times (3 constraints when threshold number is used)
for (machineID, variantID) in list(itertools.product(machines, variants)):
    opmodel += num_pieces[(machineID, variantID)] - med_Qthres <= 10000*produceOrNot[(machineID, variantID)]
    opmodel += num_pieces[(machineID, variantID)] <= 10000*produceOrNot[(machineID, variantID)]
    opmodel += pulp.lpSum([-1*num_pieces[(machineID, variantID)],med_Qthres,10000*produceOrNot[(machineID, variantID)], 1]) <= 10000
    # opmodel += num_pieces[(machineID, variantID)] > 10000*(produceOrNot[(machineID, variantID)] - 1)



from datetime import datetime
datetime.now().strftime('%Y-%m-%d %H:%M:%S')

opmodel.solve()

datetime.now().strftime('%Y-%m-%d %H:%M:%S')

#########################################################################################################################

# total cost from objective function (production + stop time)
pulp.value(opmodel.objective)

# production cost
C = 0
for (machineID, variantID) in list(itertools.product(machines, variants)):
    C += float(Q_thres.ix[(Q_thres.machineID == machineID) & (Q_thres.variantID == variantID)]['median_cost']) * num_pieces[(machineID, variantID)].varValue
C

# cost in original schedule
total_C = test['TotalCost'].sum()
total_C

############################################################################################################################

## production distribution
result = []
variants = list(variants)
for var in num_pieces:
    if var[1] in variants:    # only considering variants in production
        var_value = num_pieces[var].varValue
        x = produceOrNot[var].varValue  # produceOrNot[(var[0], var[1])].varValue
        result.append([var[0], var[1], var_value, x])
result = pandas.DataFrame(result) #, columns=['machine','ModelNo', 'pcs', 'producedOrNot'])
result.pivot(index=1, columns=0, values=2).to_csv('output_costA_thresholdB_variableTimePerPc.csv')



## prepare schedule

from datetime import datetime, timedelta

shiftStart = "2016-10-01 08:00:00"  ## CHANGE HERE
shiftStart = datetime.strptime(shiftStart, "%Y-%m-%d %H:%M:%S") #shiftStart.strftime("%Y-%m-%d %H:%M:%S")
#shiftStart
schedule = []

for mc in machineList:
    startTime = shiftStart 
    t = result[(result[0] == mc) & (result[3] == 1.0)]
    varList = list(t[1]) #.unique())
    for var in varList:
        pcs = float(result[(result[0] == mc) & (result[1] == var)][2])
        i = pcs * medTimePerPc.loc[medTimePerPc.variantID == var, 'medTimePerPc'].item()
        endTime = startTime +  timedelta(minutes=i)
        schedule.append([mc, var, pcs, format(startTime, "%Y-%m-%d %H:%M:%S"), format(endTime, "%Y-%m-%d %H:%M:%S")])
        startTime = endTime + timedelta(minutes=stop_time)

schedule = pandas.DataFrame(schedule, columns=['machine','model','pcs','starttime','endtime'])

schedule


##########################################################################################################################

# compare total produced pcs vs demand - how many extra?
new = []
for variant in variants:
    new.append([variant, production.loc[production.ModelNo == variant, 'TotalProductPcs'].item(), sum(result[result[1] == variant][2])])

new = pandas.DataFrame(new, columns=['ModelNo', 'demand', 'produced'])


