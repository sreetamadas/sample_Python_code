# method 1 : distance of raw data from straight line : is inappropriate for noisy data
from sklearn import linear_model
import math

def thresholdQ(temp):
    "calculate threshold production"
    # https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
    # this is susceptible to noise in the original data from the general trend
    # can try i. median of all thresholds for a model on different machines - DONE
    #         ii. curve-fitting and distance estimation
    #         iii. other method ?
    
    # find points at the 2 ends of the KPI-pcs curve
    #print temp
    max_Q = temp['TotalProductPcs'].max()
    KPI_maxQ = np.median(temp[temp.TotalProductPcs == max_Q].kWh_per_piece) #float(temp[temp.TotalProductPcs == max_Q].kWh_per_piece) #temp[temp.TotalProductPcs == max_Q].kWh_per_piece #
    max_KPI = temp['kWh_per_piece'].max()
    Q_maxKPI = np.median(temp[temp.kWh_per_piece == max_KPI].TotalProductPcs) # float(temp[temp.kWh_per_piece == max_KPI].TotalProductPcs) #temp[temp.kWh_per_piece == max_KPI].TotalProductPcs #
    
    # straight line b/w max values : y = ax + b
    # (y2 - y1)/(x2 - x1) = (y - y1)/(x - x1
    # coef: a = (y2 - y1)/(x2 - x1)  ; b = (x2.y1 - x1.y2)/(x2 - x1)
    a = (KPI_maxQ - max_KPI)/(max_Q - Q_maxKPI)
    b = (max_Q * max_KPI - Q_maxKPI * KPI_maxQ)/(max_Q - Q_maxKPI)
    
    # calculate distance of each pt in the data to the straight line
    # distance from a pt. (X,Y) in the data (with knee) to the straight line = (aX + b - Y)/sqrt(a^2 + 1)
    temp['dist'] = ( a * temp.TotalProductPcs + b - temp.kWh_per_piece)/math.sqrt(a*a + 1)
    
    # find point with max distance
    maxD = temp['dist'].max()
    Q_maxD = np.median(temp[temp.dist == maxD].TotalProductPcs) #float(temp[temp.dist == maxD].TotalProductPcs) # 
    
    return Q_maxD;
    
    
############################################################################################################    
# method 2: using curve fitting on the data
# GOOGLE: how to fit a curve to points in python
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
# https://stackoverflow.com/questions/19165259/python-numpy-scipy-curve-fitting

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func(x, a, b):
    return (a/x) + b;

def knee1(temp):
    "curve fitting (inverse)"
    
    # find points at the 2 ends of the KPI-pcs curve
    max_Q = temp['TotalProductPcs'].max()
    KPI_maxQ = np.median(temp[temp.TotalProductPcs == max_Q].kWh_per_piece) #float(temp[temp.TotalProductPcs == max_Q].kWh_per_piece) #temp[temp.TotalProductPcs == max_Q].kWh_per_piece #
    max_KPI = temp['kWh_per_piece'].max()
    Q_maxKPI = np.median(temp[temp.kWh_per_piece == max_KPI].TotalProductPcs) # float(temp[temp.kWh_per_piece == max_KPI].TotalProductPcs) #temp[temp.kWh_per_piece == max_KPI].TotalProductPcs #
    
    # sort the values in X-col
    #temp = temp.sort_values(by = 'TotalProductPcs', ascending = True) 
    x = np.array(pandas.to_numeric(temp.TotalProductPcs)) 
    y = np.array(temp.kWh_per_piece)  
    
    # straight line b/w max values : y = ax + b
    # (y2 - y1)/(x2 - x1) = (y - y1)/(x - x1
    # coef: a = (y2 - y1)/(x2 - x1)  ; b = (x2.y1 - x1.y2)/(x2 - x1)
    a = (KPI_maxQ - max_KPI)/(max_Q - Q_maxKPI)
    b = (max_Q * max_KPI - Q_maxKPI * KPI_maxQ)/(max_Q - Q_maxKPI)
    
    # curve fitting
    params, pcov = curve_fit(func, x, y)
    
    # calculate distance of each pt in the data to the straight line
    # distance from a pt. (X,Y) in the data (with knee) to the straight line = (aX + b - Y)/sqrt(a^2 + 1)
    temp['dist'] = ( a * x + b - func(x, *params))/math.sqrt(a*a + 1)

    # find point with max distance
    maxD = temp['dist'].max()
    Q_maxD = np.median(temp[temp.dist == maxD].TotalProductPcs) #float(temp[temp.dist == maxD].TotalProductPcs) 
    #print 'max dist: ',maxD,' ; Q at max dist: ',Q_maxD
    return Q_maxD;
    
    
    
#########################################################################################################
def knee2(temp):
    "curve fitting (polynomial)"
    
    # find points at the 2 ends of the KPI-pcs curve
    max_Q = temp['TotalProductPcs'].max()
    KPI_maxQ = np.median(temp[temp.TotalProductPcs == max_Q].kWh_per_piece) #float(temp[temp.TotalProductPcs == max_Q].kWh_per_piece) #temp[temp.TotalProductPcs == max_Q].kWh_per_piece #
    max_KPI = temp['kWh_per_piece'].max()
    Q_maxKPI = np.median(temp[temp.kWh_per_piece == max_KPI].TotalProductPcs) # float(temp[temp.kWh_per_piece == max_KPI].TotalProductPcs) #temp[temp.kWh_per_piece == max_KPI].TotalProductPcs #
    
    # sort the values in X-col
    #temp = temp.sort_values(by = 'TotalProductPcs', ascending = True) 
    x = np.array(pandas.to_numeric(temp.TotalProductPcs)) 
    y = np.array(temp.kWh_per_piece)  
    
    # straight line b/w max values : y = ax + b
    # (y2 - y1)/(x2 - x1) = (y - y1)/(x - x1
    # coef: a = (y2 - y1)/(x2 - x1)  ; b = (x2.y1 - x1.y2)/(x2 - x1)
    a = (KPI_maxQ - max_KPI)/(max_Q - Q_maxKPI)
    b = (max_Q * max_KPI - Q_maxKPI * KPI_maxQ)/(max_Q - Q_maxKPI)
    
    # curve fitting
    # calculate polynomial
    z = np.polyfit(x, y, 2)
    f = np.poly1d(z)

    # calculate new y's
    y_new = f(x)
    
    # calculate distance of each pt in the data to the straight line
    # distance from a pt. (X,Y) in the data (with knee) to the straight line = (aX + b - Y)/sqrt(a^2 + 1)
    temp['dist'] = ( a * x + b - y_new)/math.sqrt(a*a + 1)

    # find point with max distance
    maxD = temp['dist'].max()
    Q_maxD = np.median(temp[temp.dist == maxD].TotalProductPcs) #float(temp[temp.dist == maxD].TotalProductPcs) 
    #print 'max dist: ',maxD,' ; Q at max dist: ',Q_maxD
    return Q_maxD;
    
    
    
##################################################################################################
from numpy import sqrt, exp

def func2(x, a, b):
    return a * exp(-(b*x)); #+ c;


def knee3(temp):
    "curve fitting (exponential)"
    
    # find points at the 2 ends of the KPI-pcs curve
    max_Q = temp['TotalProductPcs'].max()
    KPI_maxQ = np.median(temp[temp.TotalProductPcs == max_Q].kWh_per_piece) #float(temp[temp.TotalProductPcs == max_Q].kWh_per_piece) #temp[temp.TotalProductPcs == max_Q].kWh_per_piece #
    max_KPI = temp['kWh_per_piece'].max()
    Q_maxKPI = np.median(temp[temp.kWh_per_piece == max_KPI].TotalProductPcs) # float(temp[temp.kWh_per_piece == max_KPI].TotalProductPcs) #temp[temp.kWh_per_piece == max_KPI].TotalProductPcs #
    
    
    x = np.array(pandas.to_numeric(temp.TotalProductPcs)) 
    y = np.array(temp.kWh_per_piece)  
    
    # straight line b/w max values : y = ax + b
    # (y2 - y1)/(x2 - x1) = (y - y1)/(x - x1
    # coef: a = (y2 - y1)/(x2 - x1)  ; b = (x2.y1 - x1.y2)/(x2 - x1)
    a = (KPI_maxQ - max_KPI)/(max_Q - Q_maxKPI)
    b = (max_Q * max_KPI - Q_maxKPI * KPI_maxQ)/(max_Q - Q_maxKPI)
    
    # curve fitting
    params2, pcov2 = curve_fit(func2, x, y)
    
    # calculate distance of each pt in the data to the straight line
    # distance from a pt. (X,Y) in the data (with knee) to the straight line = (aX + b - Y)/sqrt(a^2 + 1)
    temp['dist'] = ( a * x + b - func2(x, *params2))/math.sqrt(a*a + 1)

    # find point with max distance
    maxD = temp['dist'].max()
    Q_maxD = np.median(temp[temp.dist == maxD].TotalProductPcs) #float(temp[temp.dist == maxD].TotalProductPcs) 
    #print 'max dist: ',maxD,' ; Q at max dist: ',Q_maxD
    return Q_maxD;
    
    
    
    
    
    
