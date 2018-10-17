# use of math library for calculations
import math


# taking power
# y = = 0.983^(eTOTAL × 0.9)
y = math.pow(0.983,(math.exp(0.9 * total)))


# calculate sqrt
rmse = math.sqrt(np.mean((y_pred - y_actual) ** 2))
