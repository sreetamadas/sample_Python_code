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


########################################################################################
## check ACF
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
fig = tsaplots.plot_acf(df['y'], lags=40)
plt.show()


## check PACF  (PACF measures correlation after removing the effect of previous time points)
# PACF of order 3 returns the correlation between our time series (t1, t2, t3, ...) and lagged values of itself by 3 time points (t4, t5, t6, ...),
# after removing all effects attributable to lags 1 and 2
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
fig = tsaplots.plot_pacf(df['y'], lags=40)
plt.show()

#####################################################################################
## time series decomposition
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 11, 9
decomposition = sm.tsa.seasonal_decompose(df['y'])
fig = decomposition.plot()
plt.show()

# picking & plotting the components separately
#print(decomposition.seasonal)
#print(decomposition.trend)
#print(decomposition.resid)
decomp_seasonal = decomposition.seasonal
ax = decomp_seasonal.plot(figsize=(14, 2))
ax.set_xlabel('Date')
ax.set_ylabel('Seasonality of time series')
ax.set_title('Seasonal values of the time series')
plt.show()



## decompose multiple time series
# Import the statsmodel library
import statsmodels.api as sm
# Initialize a dictionary
my_dict = {}
# Extract the names of the time series
ts_names = df.columns
#print(ts_names)  #['ts1', 'ts2', 'ts3']
# Run time series decomposition
for ts in ts_names:
            ts_decomposition = sm.tsa.seasonal_decompose(jobs[ts])
            my_dict[ts] = ts_decomposition

            
## select the trend components of all the time series from the dict & make a new df
# Initialize a new dictionnary
my_dict_trend = {}
# Extract the trend component
for ts in ts_names:
            my_dict_trend[ts] = my_dict[ts].trend
# Convert to a DataFrame            
trend_df = pd.DataFrame.from_dict(my_dict_trend)

    
