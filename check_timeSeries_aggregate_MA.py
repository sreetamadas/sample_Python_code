## compute moving averages/ rolling means to understand 
##  1. long-term trends/ cycles
##  2. remove short term fluctuations
##  3. remove outliers

# df
# dateTime    Y
# 2018-01-01  398
# ...

df_mean = df.rolling(window=7).mean()   # set window according to seasonality

## compute aggregates to understand weekly/ monthly/ yearly means
# eg find mean values during each month of the year
df = df.set_index('dateTime')
index_month = df.index.month
Y_by_month = df.groupby(index_month).mean()
Y_by_month.plot()
plt.show()



## compute bouunds from MA 
# MA
ma = df.rolling(window=52).mean()
# Compute the 52 weeks rolling standard deviation of the DataFrame
mstd = df.rolling(window=52).std()
# Add the upper bound column to the ma DataFrame
ma['upper'] = ma['co2'] + (2 * mstd['Y'])
# Add the lower bound column to the ma DataFrame
ma['lower'] = ma['co2'] - (2 * mstd['Y'])



## check ACF
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
fig = tsaplots.plot_acf(df['y'], lags=40)
plt.show()
