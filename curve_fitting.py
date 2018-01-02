# curve fitting

import pandas #as pd
import numpy as np
from numpy import sqrt, exp
from scipy.optimize import curve_fit


# sort df by values in X-col
temp = temp.sort_values(by = 'X', ascending = True)  
# df.sort_values(by=['name', 'score'], ascending=[False, True])

x = np.array(pandas.to_numeric(temp.TotalProductPcs)) #temp.as_matrix(columns=temp.columns['TotalProductPcs'])   temp.iloc['TotalProductPcs'].values
y = np.array(temp.kWh_per_piece)  #temp.as_matrix(columns=temp.columns['kWh_per_piece'])  #temp.iloc['kWh_per_piece'].values


#######################################################################
### method 1: fitting with defined function  ###
def func(x, a, b):
    return (a/x) + b;

params, pcov = curve_fit(func, x, y)

plt.scatter(x, y)  #plt.plot(x, y, 'b')
plt.plot(x, func(x, *params), 'g--')
plt.show()


### method 2: using polynomial fit instead of inverse ###
# calculate polynomial
z = np.polyfit(x, y, 2)
f = np.poly1d(z)

# calculate new x's and y's
#x_new = np.linspace(x[0], x[-1], 50)
#y_new = f(x_new)
y_new = f(x)

#plt.plot(x,y,'o', x_new, y_new)
plt.plot(x,y,'o', x, y_new)
plt.xlim([x[0]-1, x[-1] + 1 ])
plt.show()

