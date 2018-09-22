# given a series and alpha, return series of smoothed points

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def exponential_smoothing(x, alpha):
    """Computes exponential smoothing for the given data set which exhibits no trend and sesonality.
    Args:
    -----
        x (pandas.Series): dependent variable
        alpha            : Smoothing Parameter
    
    Returns:
    --------
        a series of smoothed values
    """
        
    result = [x[0]]
    for n in range(1, len(x)+1):
        result.append(alpha * x[n-1] + (1 - alpha) * result[n-1])
    return result
    
    
x=train.Y #grp_data.iloc[:,3]
exp_smt=exponential_smoothing(x,0.5)
exp_h = exponential_smoothing(x,0.9)
exp_l = exponential_smoothing(x,0.2)


plt.plot(train['Q'],'.--',c='green')  # train['week']
#plt.plot(train['Q'],'.--')
plt.plot(exp_smt[:-1],'.--', c='blue')
plt.plot(exp_h[:-1],'.--', c='red')
plt.plot(exp_l[:-1],'.--', c='brown')
plt.show()


