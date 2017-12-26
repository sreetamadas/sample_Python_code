### calculate threshold X from knee curve  ###
from sklearn import linear_model
import math

def thresholdX(temp):
    "calculate threshold X"
    # https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
    # 
    # can try i. method 1, see below (this is susceptible to noise in the original data from the general trend)
    #         ii. curve-fitting and distance estimation
    #         iii. other method ?
    
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
    
    
    x = thresholdX(temp)
    
    
