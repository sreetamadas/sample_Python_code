### calculate threshold X from knee curve  ###
## GOOGLE: how to find knee of a curve in noisy data
# method 1: analytical (distance calculation with original data - may be affected by noise in data)
# method 2: distance calculation with Y from curve fitted to original data 
# method 3: https://www1.icsi.berkeley.edu/~barath/papers/kneedle-simplex11.pdf

import pandas #as pd
import numpy as np
from numpy import sqrt, exp
from sklearn import linear_model
import math
from scipy.optimize import curve_fit



def thresholdX(temp):
    "method 1 : calculate threshold X"
    # https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
    
    # find points at the 2 ends of the X-Y curve
    #print temp
    max_X = temp['X'].max()
    Y_maxX = np.median(temp[temp.X == max_X].Y)   # float(temp[temp.X == max_X].Y) # temp[temp.X == max_X].Y
    max_Y = temp['Y'].max()
    X_maxY = np.median(temp[temp.Y == max_Y].X)   # float(temp[temp.Y == max_Y].X) #temp[temp.Y == max_Y].X
    
    # straight line b/w max values : y = ax + b
    # (y2 - y1)/(x2 - x1) = (y - y1)/(x - x1
    # coef: a = (y2 - y1)/(x2 - x1)  ; b = (x2.y1 - x1.y2)/(x2 - x1)
    a = (Y_maxX - max_Y)/(max_X - X_maxY)
    b = (max_X * max_Y - X_maxY * Y_maxX)/(max_X - X_maxY)
    
    # calculate distance of each pt in the data to the straight line
    # distance from a pt. (X,Y) in the data (with knee) to the straight line = (aX + b - Y)/sqrt(a^2 + 1)
    temp['dist'] = ( a * temp.X + b - temp.Y)/math.sqrt(a*a + 1)
    
    # find point with max distance
    maxD = temp['dist'].max()
    X_maxD = np.median(temp[temp.dist == maxD].X)   # float(temp[temp.dist == maxD].X) 
    return X_maxD;
 
    
    
# method 2: using curve fitting on the data
# GOOGLE: how to fit a curve to points in python
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
# https://stackoverflow.com/questions/19165259/python-numpy-scipy-curve-fitting

def func(x, a, b):
    "linear fit"
    return (a/x) + b;

def func2(x, a, b):
    "exponential decay"
    return a * exp(-(b*x));


def knee1(temp):
    "curve fitting (inverse)"
    
    # find points at the 2 ends of the X-Y curve
    #print temp
    max_X = temp['X'].max()
    Y_maxX = np.median(temp[temp.X == max_X].Y)   # float(temp[temp.X == max_X].Y) # temp[temp.X == max_X].Y
    max_Y = temp['Y'].max()
    X_maxY = np.median(temp[temp.Y == max_Y].X)   # float(temp[temp.Y == max_Y].X) #temp[temp.Y == max_Y].X
    
    x = np.array(pandas.to_numeric(temp.X)) 
    y = np.array(temp.Y)  
    
    # straight line b/w max values : y = ax + b
    # (y2 - y1)/(x2 - x1) = (y - y1)/(x - x1
    # coef: a = (y2 - y1)/(x2 - x1)  ; b = (x2.y1 - x1.y2)/(x2 - x1)
    a = (Y_maxX - max_Y)/(max_X - X_maxY)
    b = (max_X * max_Y - X_maxY * Y_maxX)/(max_X - X_maxY)
    
    # curve fitting
    params, pcov = curve_fit(func, x, y)    # or, with func2 for exp decay
    
    # calculate distance of each pt in the data to the straight line
    # distance from a pt. (X,Y) in the data (with knee) to the straight line = (aX + b - Y)/sqrt(a^2 + 1)
    temp['dist'] = ( a * x + b - func(x, *params))/math.sqrt(a*a + 1)

    # find point with max distance
    maxD = temp['dist'].max()
    Q_maxD = np.median(temp[temp.dist == maxD].X) 
    return Q_maxD;


def knee2(temp):
    "curve fitting (polynomial)"
    
    # find points at the 2 ends of the X-Y curve
    max_X = temp['X'].max()
    Y_maxX = np.median(temp[temp.X == max_X].Y)   # float(temp[temp.X == max_X].Y) # temp[temp.X == max_X].Y
    max_Y = temp['Y'].max()
    X_maxY = np.median(temp[temp.Y == max_Y].X)   # float(temp[temp.Y == max_Y].X) #temp[temp.Y == max_Y].X
    
    x = np.array(pandas.to_numeric(temp.X)) 
    y = np.array(temp.Y)    
    
    # straight line b/w max values : y = ax + b
    # (y2 - y1)/(x2 - x1) = (y - y1)/(x - x1
    # coef: a = (y2 - y1)/(x2 - x1)  ; b = (x2.y1 - x1.y2)/(x2 - x1)
    a = (Y_maxX - max_Y)/(max_X - X_maxY)
    b = (max_X * max_Y - X_maxY * Y_maxX)/(max_X - X_maxY)
    
    # curve fitting
    # calculate polynomial
    z = np.polyfit(x, y, 2)  # polynomial of degree 2
    f = np.poly1d(z)

    # calculate new y's
    y_new = f(x)
    
    # calculate distance of each pt in the data to the straight line
    # distance from a pt. (X,Y) in the data (with knee) to the straight line = (aX + b - Y)/sqrt(a^2 + 1)
    temp['dist'] = ( a * x + b - y_new)/math.sqrt(a*a + 1)

    # find point with max distance
    maxD = temp['dist'].max()
    Q_maxD = np.median(temp[temp.dist == maxD].X) # 
    #print 'max dist: ',maxD,' ; Q at max dist: ',Q_maxD
    return Q_maxD;

######################################################################################################    
# sort data by X-column
temp = temp.sort_values(by = 'X', ascending = True) 
x = thresholdX(temp)
x1 = knee1(temp)
x2 = knee2(temp)
    
