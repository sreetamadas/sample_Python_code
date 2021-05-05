####################################################################################################
'''
methods to solve optimisation prob:

gurobi solver : simplex method
CBC solver : branch & bound algorithm

'''
######################################################################################################
##### OPTIMISE cost of production (of items on different machines), given time & demand  #######

import pandas #as pd
import numpy as np

# from statistics import median      ## no module named statistics in the installed version
import pulp
# alternate to pulp
# https://github.com/SCIP-Interfaces/PySCIPOpt
from pyscipopt import *

import itertools  # to use only those indices corresponding to model numbers present in demand

####################################################################################################

## data preparation

##################################################################################################################
####################################################################################################################

## OPTIMISATION using PuLP

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
##############################################################################################################################

### OPTIMISATION using PySCIPopt

# declare the optimisation problem
#opmodel = pulp.LpProblem("Energy optimization", pulp.LpMinimize)
scipmodel = Model("Energy optimization")


# declare variables : numbers Nij
# num_pieces = pulp.LpVariable.dicts("Number_pieces", [tuple(pair) for pair in Q_thres[['machineID','variantID']].as_matrix()],
#                                   lowBound = 0, cat = pulp.LpInteger)
no_pieces = {}
for [machineID, variantID] in Q_thres[['machineID','variantID']].as_matrix():
    no_pieces[machineID, variantID] = scipmodel.addVar(vtype='I',name="Number_pieces_%s_%s"%(machineID, variantID))



# declare Xij
# produceOrNot = pulp.LpVariable.dicts("isProduced", [tuple(pair) for pair in Q_thres[['machineID','variantID']].as_matrix()],
#                                   cat = pulp.LpBinary)
produceornot = {}
for [machineID, variantID] in Q_thres[['machineID','variantID']].as_matrix():
    produceornot[machineID, variantID] = scipmodel.addVar(vtype='B',name="Number_pieces_%s_%s"%(machineID, variantID))

    

# Objective Function : two parts = wheel production + cost of stopTime
# using wheels only in demand list
'''
opmodel += (
    pulp.lpSum([
        Q_thres.ix[(Q_thres.machineID == machineID) & (Q_thres.variantID == variantID)]['median_cost'] * num_pieces[(machineID, variantID)] 
            for (machineID, variantID) in list(itertools.product(machines, variants))]
               + [stop_cost*produceOrNot[(machineID, variantID)]
                                      for (machineID, variantID) in list(itertools.product(machines, variants))]
               ))
'''
# Please change PATH variable with this “C:\Program Files\SCIPOptSuite 4.0.1\bin” 
# before it was “C:\Program Files\SCIPOptSuite 4.0.1” 
# … DLL file is actually present in **\bin folder so it couldn’t able find that early
scipmodel.setObjective(quicksum(float(Q_thres.ix[(Q_thres.machineID == machineID) & (Q_thres.variantID == variantID)]['median_cost'])*no_pieces[machineID, variantID]
                                for (machineID, variantID) in list(itertools.product(machines, variants))) + 
                       quicksum(stop_cost*produceornot[machineID, variantID] for (machineID, variantID) in list(itertools.product(machines, variants))))
    
    
    
# Constraint on numbers: production should meet demand of each variant
#for i, variantID in enumerate(variants):
#    opmodel += pulp.lpSum([num_pieces[machineID,variantID] for machineID in Q_thres[Q_thres['variantID'] == variantID]['machineID']]) >= demand[i]
for i, variantID in enumerate(variants):
    scipmodel.addCons(quicksum(no_pieces[machineID,variantID] for machineID in Q_thres[Q_thres['variantID'] == variantID]['machineID']) >= demand[i])
 


# constraint for total production time on each machine: two parts = production time + stopTime for model change
#for i, machineID in enumerate(machines):
#    opmodel += pulp.lpSum([medTimePerPc.ix[medTimePerPc['variantID'] == variantID]['medTimePerPc']*num_pieces[machineID,variantID] for variantID in variants]  + 
#                         [stop_time*produceOrNot[machineID,variantID] for variantID in variants] +
#                         [-stop_time]
#                         ) <= scheduleFor
for i, machineID in enumerate(machines):
    scipmodel.addCons(quicksum(float(medTimePerPc.ix[medTimePerPc['variantID'] == variantID]['medTimePerPc'])*no_pieces[machineID,variantID] for variantID in variants)  + 
                         quicksum(stop_time*produceornot[machineID,variantID] for variantID in variants) - stop_time
                          <= scheduleFor)



# Indicator function
# constraint on Xij to consider stop times (3 constraints when threshold number is used)
'''
for (machineID, variantID) in list(itertools.product(machines, variants)):
    ## change threshold in next line ***
    opmodel += num_pieces[(machineID, variantID)] - med_Qthres <= 10000*produceOrNot[(machineID, variantID)]
    opmodel += num_pieces[(machineID, variantID)] <= 10000*produceOrNot[(machineID, variantID)]
    ## change threshold in next line ***
    opmodel += pulp.lpSum([-1*num_pieces[(machineID, variantID)],med_Qthres,10000*produceOrNot[(machineID, variantID)], 1]) <= 10000
    # opmodel += num_pieces[(machineID, variantID)] > 10000*(produceOrNot[(machineID, variantID)] - 1)
'''
for (machineID, variantID) in list(itertools.product(machines, variants)):
    scipmodel.addCons(no_pieces[(machineID, variantID)] - med_Qthres <= 10000*produceornot[(machineID, variantID)])
    scipmodel.addCons(no_pieces[(machineID, variantID)] <= 10000*produceornot[(machineID, variantID)])
    scipmodel.addCons(-1*no_pieces[(machineID, variantID)] + med_Qthres + 10000*produceornot[(machineID, variantID)] + 1 <= 10000)


from datetime import datetime
datetime.now().strftime('%Y-%m-%d %H:%M:%S')
scipmodel.optimize()
datetime.now().strftime('%Y-%m-%d %H:%M:%S')
scipmodel.getStatus()


# cost from optimisation function
scipmodel.getObjVal()

# cost of production
C = 0
for (machineID, variantID) in list(itertools.product(machines, variants)):
    C += float(Q_thres.ix[(Q_thres.machineID == machineID) & (Q_thres.variantID == variantID)]['median_kpi']) * scipmodel.getVal(no_pieces[(machineID, variantID)])
C


###########################################################################################################################
############################################################################################################################

# scheduling code using optimisation

from pyscipopt import *
sModel = Model("Scheduling")
sModel.setMinimize()


# declaring new variables
allCombi = []
allstartTimes = {}
for machineID in machines:
    temp = []
    for variantID in variants:
        if ((machineID, variantID) in no_pieces):
            if (scipmodel.getVal(no_pieces[(machineID, variantID)])) > 0:
                temp.append((machineID, variantID))
                allstartTimes[machineID, variantID] = sModel.addVar(name = "startTime(%s, %s)"%(machineID, variantID), lb=-1, ub=scheduleFor)
    if len(temp) > 0:
        allCombi.extend(list(itertools.combinations(list(temp),2)))


# objective function
sModel.setObjective(quicksum(allstartTimes[keys] for keys in allstartTimes), "minimize")
        
        
        
## set up the constraints - see old notes
timeforone = 5  # 0.084 when schedulefor is in hrs

M=10000
Y = {}
for pair in allCombi:
    i = pair[0][0]
    j = pair[0][1]
    l = pair[1][0]
    m = pair[1][1]
    Y[i,j,l,m] = sModel.addVar(vtype = 'B',name = "varY((%s, %s), (%s, %s))"% (i, j, l, m))
    # print Y[i,j,l,m]
    sModel.addCons(allstartTimes[i,j] + medTimePerPc.ix[medTimePerPc['variantID'] == j]['medTimePerPc']*scipmodel.getVal(no_pieces[i, j]) - allstartTimes[l, m] <= M*Y[i,j,l,m])
    sModel.addCons(allstartTimes[l,m] + medTimePerPc.ix[medTimePerPc['variantID'] == m]['medTimePerPc']*scipmodel.getVal(no_pieces[l, m]) - allstartTimes[i, j] <= M*(1-Y[i,j,l,m]))

    #sModel.addCons(allstartTimes[i,j] + timeforone*scipmodel.getVal(no_pieces[i, j]) - allstartTimes[l, m] <= M*Y[i,j,l,m])
    #sModel.addCons(allstartTimes[l,m] + timeforone*scipmodel.getVal(no_pieces[l, m]) - allstartTimes[i, j] <= M*(1-Y[i,j,l,m]))
    
    # scheduleModel += pulp.lpSum([startTimes[i, j], timeforone*num_pieces[i, j].varValue, -1*startTimes[l, m]]) <= M*varY[pair]
    # scheduleModel += pulp.lpSum([startTimes[l, m], timeforone*num_pieces[l, m].varValue, -1*startTimes[i, j]]) <= M*(1 - varY[pair])
        
        
## run the optimisation
sModel.optimize()


## check schedule on selected machine
for key in allstartTimes:
    if key[0] == 'MC1':
        print key, sModel.getVal(allstartTimes[key])





