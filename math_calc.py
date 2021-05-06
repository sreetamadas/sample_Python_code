# use of math library for calculations
import math


# taking power
# y = = 0.983^(eTOTAL × 0.9)
y = math.pow(0.983,(math.exp(0.9 * total)))


# calculate sqrt
rmse = math.sqrt(np.mean((y_pred - y_actual) ** 2))


# rolling median
df['median'] = df['y'].rolling(window=30, center=True).median().fillna(method='bfill').fillna(method='ffill')


# calculate median
df.loc[0,'i1'] = pd.DataFrame.median(data_file.iloc[:,2] - data_file.iloc[:,3])#.median()

# 
x.loc[i,'d'] = math.log(I1)-math.log(abs(x.loc[i,'i1']))
