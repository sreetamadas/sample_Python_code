## compute moving averages/ rolling means to understand 
##  1. long-term trends/ cycles
##  2. remove short term fluctuations
##  3. remove outliers

df_mean = df.rolling(window=7).mean()   # set window according to seasonality


## compute aggregates to understand weekly/ monthly/ yearly means
# eg find mean values during each month of the year
df = df.set_index('dateTime')
index_month = df.index.month
Y_by_month = df.groupby(index_month).mean()
Y_by_month.plot()
plt.show()
